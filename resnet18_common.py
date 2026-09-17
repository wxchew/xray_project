"""Shared ResNet18 pieces, so analysis notebooks do not restate them.

These definitions mirror `04b_chest_Xray_ResNet18.ipynb`, which is where the model was
trained. Names are kept identical to that notebook's, so 04b can drop its own copies
later as a pure deletion.

The one addition is `features_and_logits`, which analysis needs and training does not:
it returns the 512-d `avgpool` vector alongside the logit, from a single forward pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torchvision
from torch import nn
from torch.torch_version import TorchVersion
from torchvision.io import ImageReadMode, read_image
from torchvision.models import resnet18
from torchvision.transforms import v2

CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}

# The images on disk are already 224x224, so we skip the ImageNet weights' default
# Resize(256) + CenterCrop(224), which would cut the edges off a chest film.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

rn_transform = v2.Compose(
    [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


def read_image_rgb(path):
    return read_image(str(path), mode=ImageReadMode.RGB)


class FrozenResNet18(nn.Module):
    """ResNet18 with the classifier head replaced by Linear(512, 1) + Sigmoid.

    `weights=None` by default: when loading a trained checkpoint there is no reason to
    download the ImageNet weights first, since every one of them is about to be overwritten.
    """

    def __init__(self, weights=None):
        super().__init__()
        self.net = resnet18(weights=weights)
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.fc = nn.Sequential(nn.Linear(self.net.fc.in_features, 1), nn.Sigmoid())

    def train(self, mode=True):
        # requires_grad=False does not freeze BatchNorm: in train mode it would still
        # normalise with batch statistics and update its running averages.
        super().train(mode)
        for m in self.net.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
        return self

    def forward(self, x):
        return self.net(x)


def features_and_logits(model: FrozenResNet18, x: torch.Tensor):
    """Return the 512-d avgpool vector and the logit, from one forward pass.

    The vector is what the final linear layer actually sees, so it is the model's own
    summary of the image. `fc[0]` is that Linear, taken before the Sigmoid.
    """
    n = model.net
    x = n.maxpool(n.relu(n.bn1(n.conv1(x))))
    x = n.layer4(n.layer3(n.layer2(n.layer1(x))))
    features = torch.flatten(n.avgpool(x), 1)
    return features, n.fc[0](features)


def load_trained_model(checkpoint_path, device="cpu") -> FrozenResNet18:
    """Load a saved checkpoint, accepting either save format 04b has produced.

    04b records the torch version next to the weights, so the file holds a TorchVersion
    object. Since PyTorch 2.6 `weights_only=True` is the default and rejects it. We keep
    weights_only=True, which still refuses arbitrary code, and allowlist just that one
    class rather than turning the protection off.
    """
    with torch.serialization.safe_globals([TorchVersion]):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    if "conv1.weight" in state:  # saved without the wrapper prefix
        state = {f"net.{k}": v for k, v in state.items()}
    if "net.conv1.weight" not in state:
        raise ValueError(f"{checkpoint_path} does not hold a recognised ResNet18 state dict")

    model = FrozenResNet18()
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def make_dataset(data_dir) -> torchvision.datasets.ImageFolder:
    """ImageFolder with the preprocessing the model was trained with."""
    dataset = torchvision.datasets.ImageFolder(
        str(data_dir), loader=read_image_rgb, transform=rn_transform
    )
    if dataset.class_to_idx != CLASS_TO_IDX:
        raise ValueError(f"Unexpected class mapping {dataset.class_to_idx}")
    return dataset


def load_metadata(metadata_path) -> dict:
    """Read the model card and check it describes the model these helpers build."""
    meta = json.loads(Path(metadata_path).read_text())
    if meta["model"]["architecture"] != "resnet18":
        raise ValueError("Metadata does not describe a resnet18")
    if meta["input"]["shape"] != [3, 224, 224]:
        raise ValueError(f"Unexpected input shape {meta['input']['shape']}")
    if meta["data"]["class_to_idx"] != CLASS_TO_IDX:
        raise ValueError("Class order does not match")
    if meta["input"]["normalize_mean"] != IMAGENET_MEAN:
        raise ValueError("Metadata normalisation does not match rn_transform")
    return meta
