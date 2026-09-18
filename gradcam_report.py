#!/usr/bin/env python3
"""Generate a self-contained HiResCAM review for the trained pneumonia CNN."""

from __future__ import annotations

import argparse
import base64
import html
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import v2


CLASS_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}
CATEGORY_NAMES = {
    "TP": "True positive",
    "TN": "True negative",
    "FP": "False positive",
    "FN": "False negative",
}


class CNN(nn.Module):
    """Architecture used by 04_chest_Xray_CNN.ipynb."""

    def __init__(self, flatten_output_dim: int = 484) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 2, kernel_size=16, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(2, 4, kernel_size=5, stride=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(flatten_output_dim, 16),
            nn.Linear(16, 8),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        out = self.flatten(self.conv(x))
        out = self.classifier[0](out)
        out = self.classifier[1](out)
        return self.classifier[2](out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(x))


@dataclass(frozen=True)
class Prediction:
    path: Path
    true_class: int
    probability: float

    @property
    def predicted_class(self) -> int:
        return int(self.probability >= 0.5)

    @property
    def confidence(self) -> float:
        return self.probability if self.predicted_class else 1.0 - self.probability

    @property
    def category(self) -> str:
        return {
            (1, 1): "TP",
            (0, 0): "TN",
            (0, 1): "FP",
            (1, 0): "FN",
        }[(self.true_class, self.predicted_class)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--checkpoint", default="cnn_pneumonia_model.pt")
    parser.add_argument("--data-dir", default="data/chest_xray_224/test")
    parser.add_argument("--output", default="gradcam_report.html")
    parser.add_argument("--roi-threshold", type=float, default=0.6)
    parser.add_argument("--examples-per-category", type=int, default=3)
    return parser.parse_args()


def load_model(checkpoint_path: Path) -> tuple[CNN, dict]:
    # This checkpoint stores its two scalar metrics as NumPy float64 values. Allow
    # only the NumPy globals needed for those values while keeping restricted
    # weights-only deserialization enabled.
    safe_numpy_globals = [np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType]
    with torch.serialization.safe_globals(safe_numpy_globals):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {"model_state_dict", "train_accuracy", "valid_accuracy"}
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {sorted(missing)}")
    state_dict = checkpoint["model_state_dict"]
    flatten_output_dim = int(state_dict["classifier.0.weight"].shape[1])
    model = CNN(flatten_output_dim=flatten_output_dim)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, checkpoint


def make_dataset(data_dir: Path) -> datasets.ImageFolder:
    dataset = datasets.ImageFolder(
        data_dir,
        loader=read_image,
        transform=v2.ToDtype(torch.float32),
    )
    if dataset.class_to_idx != {"NORMAL": 0, "PNEUMONIA": 1}:
        raise ValueError(f"Unexpected class mapping: {dataset.class_to_idx}")
    return dataset


def evaluate(model: CNN, dataset: datasets.ImageFolder) -> list[Prediction]:
    probabilities: list[float] = []
    targets: list[int] = []
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    with torch.no_grad():
        for images, labels in loader:
            probabilities.extend(model(images).squeeze(1).tolist())
            targets.extend(labels.tolist())
    return [
        Prediction(Path(path), int(target), float(probability))
        for (path, _), target, probability in zip(dataset.samples, targets, probabilities)
    ]


def select_examples(records: list[Prediction], count: int) -> list[tuple[str, Prediction]]:
    if count < 1:
        raise ValueError("Example count must be at least 1")
    selected: list[tuple[str, Prediction]] = []
    for category in ("TP", "TN", "FP", "FN"):
        group = [record for record in records if record.category == category]
        if len(group) < count:
            raise ValueError(f"Need {count} {category} cases, found {len(group)}")
        median_confidence = float(np.median([record.confidence for record in group]))
        ranked_sets = [
            ("Highest confidence", sorted(group, key=lambda r: (-r.confidence, r.path.name))),
            (
                "Median confidence",
                sorted(group, key=lambda r: (abs(r.confidence - median_confidence), r.path.name)),
            ),
            (
                "Closest to threshold",
                sorted(group, key=lambda r: (abs(r.probability - 0.5), r.path.name)),
            ),
        ]
        used: set[Path] = set()
        for selection_name, ranking in ranked_sets[:count]:
            record = next(candidate for candidate in ranking if candidate.path not in used)
            used.add(record.path)
            selected.append((selection_name, record))

        # Preserve the three interpretable anchor choices above, then fill any
        # larger request with cases spread deterministically across confidence.
        extra_count = count - min(count, len(ranked_sets))
        if extra_count > 0:
            remaining = sorted(
                (record for record in group if record.path not in used),
                key=lambda r: (-r.confidence, r.path.name),
            )
            indices = np.linspace(0, len(remaining) - 1, extra_count, dtype=int)
            for number, index in enumerate(indices, start=1):
                record = remaining[int(index)]
                used.add(record.path)
                selected.append((f"Confidence spread {number}/{extra_count}", record))
    return selected


def hirescam(model: CNN, image: torch.Tensor, target_class: int) -> np.ndarray:
    captured: dict[str, torch.Tensor] = {}

    def capture_activation(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        captured["activation"] = output
        output.retain_grad()

    handle = model.conv[3].register_forward_hook(capture_activation)
    try:
        model.zero_grad(set_to_none=True)
        logit = model.forward_logits(image.unsqueeze(0)).squeeze()
        target_score = logit if target_class == 1 else -logit
        target_score.backward()
    finally:
        handle.remove()

    activation = captured["activation"]
    if activation.grad is None:
        raise RuntimeError("No gradient was captured for the final convolutional layer")
    feature_maps = activation.detach()[0]
    gradients = activation.grad.detach()[0]
    # HiResCAM keeps the position-specific gradient instead of replacing every
    # position in a channel with one spatially averaged Grad-CAM weight.
    cam = torch.relu((gradients * feature_maps).sum(dim=0))
    if not torch.isfinite(cam).all():
        raise RuntimeError("HiResCAM is non-finite")
    maximum = cam.max()
    if float(maximum) > 0:
        cam = cam / maximum
    else:
        cam = torch.zeros_like(cam)
    cam = F.interpolate(
        cam[None, None],
        size=image.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0, 0]
    result = cam.detach().cpu().numpy()
    if not np.isfinite(result).all() or result.min() < -1e-6 or result.max() > 1.000001:
        raise RuntimeError("Resized HiResCAM failed normalization checks")
    return np.clip(result, 0.0, 1.0)


def jet_colormap(values: np.ndarray) -> np.ndarray:
    red = np.clip(1.5 - np.abs(4.0 * values - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * values - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * values - 1.0), 0.0, 1.0)
    return np.uint8(np.stack([red, green, blue], axis=-1) * 255)


def png_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_images(path: Path, cam: np.ndarray, threshold: float) -> tuple[str, str, str, float]:
    original = Image.open(path).convert("L").resize((cam.shape[1], cam.shape[0]))
    gray = np.asarray(original, dtype=np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=-1)
    colored = jet_colormap(cam)
    continuous = np.uint8(0.45 * rgb + 0.55 * colored)

    mask = cam > threshold
    overlay = rgb.copy().astype(np.float32)
    overlay[mask] = 0.55 * overlay[mask] + 0.45 * np.array([255.0, 0.0, 0.0])
    interior = np.zeros_like(mask)
    interior[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
    )
    boundary = mask & ~interior
    overlay[boundary] = np.array([255.0, 0.0, 0.0])

    return (
        png_data_uri(Image.fromarray(rgb)),
        png_data_uri(Image.fromarray(continuous)),
        png_data_uri(Image.fromarray(np.uint8(np.clip(overlay, 0, 255)))),
        float(mask.mean()),
    )


def build_report(
    records: list[Prediction],
    selected: list[tuple[str, Prediction]],
    model: CNN,
    checkpoint: dict,
    roi_threshold: float,
    checkpoint_name: str,
    data_dir: Path,
) -> str:
    targets = [record.true_class for record in records]
    predictions = [record.predicted_class for record in records]
    probabilities = [record.probability for record in records]
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / len(records)
    saved_accuracy = float(checkpoint["valid_accuracy"])
    if not np.isclose(accuracy, saved_accuracy, atol=1e-12):
        raise RuntimeError(f"Accuracy {accuracy:.12f} does not match checkpoint {saved_accuracy:.12f}")

    sample = v2.ToDtype(torch.float32)(read_image(str(records[0].path)))
    with torch.no_grad():
        last_conv = model.conv[3](model.conv[:3](sample.unsqueeze(0)))
    input_channels, input_height, input_width = sample.shape
    map_channels, map_height, map_width = last_conv.shape[1:]
    checkpoint_label = html.escape(checkpoint_name)
    data_label = html.escape(str(data_dir))

    cards: list[str] = []
    for selection_name, record in selected:
        tensor = v2.ToDtype(torch.float32)(read_image(str(record.path)))
        cam = hirescam(model, tensor, record.predicted_class)
        original_uri, heatmap_uri, overlay_uri, roi_fraction = render_images(
            record.path, cam, roi_threshold
        )
        cards.append(
            f"""
            <article class="case {record.category.lower()}">
              <header>
                <span class="badge">{record.category} · {CATEGORY_NAMES[record.category]}</span>
                <span class="selection">{selection_name}</span>
                <h3>{html.escape(record.path.name)}</h3>
                <p>True: <b>{CLASS_NAMES[record.true_class]}</b> · Predicted:
                   <b>{CLASS_NAMES[record.predicted_class]}</b> · Pneumonia probability:
                   <b>{record.probability:.3f}</b></p>
                <p>Explained score: predicted {CLASS_NAMES[record.predicted_class]} class ·
                   ROI coverage at {roi_threshold:.2f}: {100 * roi_fraction:.1f}%</p>
              </header>
              <div class="images">
                <figure><img src="{original_uri}" alt="Original X-ray"><figcaption>Original</figcaption></figure>
                <figure><img src="{heatmap_uri}" alt="Continuous HiResCAM"><figcaption>Continuous HiResCAM</figcaption></figure>
                <figure><img src="{overlay_uri}" alt="Thresholded ROI overlay"><figcaption>ROI &gt; {roi_threshold:.2f}</figcaption></figure>
              </div>
            </article>
            """
        )

    auc = roc_auc_score(targets, probabilities)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{checkpoint_label} — HiResCAM review</title>
<style>
  :root {{ color-scheme: light; --ink:#172033; --muted:#64748b; --line:#dbe3ee; --paper:#f7f9fc; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font:15px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--ink); background:var(--paper); }}
  main {{ max-width:1320px; margin:auto; padding:32px 24px 64px; }}
  h1 {{ margin:0 0 8px; font-size:32px; }} h2 {{ margin-top:34px; }}
  .lead {{ color:var(--muted); max-width:900px; font-size:17px; }}
  .metrics {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; margin:24px 0; }}
  .metric {{ background:white; border:1px solid var(--line); border-radius:10px; padding:14px; }}
  .metric b {{ display:block; font-size:23px; }} .metric span {{ color:var(--muted); }}
  .matrix {{ border-collapse:collapse; background:white; }} .matrix th,.matrix td {{ border:1px solid var(--line); padding:9px 14px; text-align:center; }}
  .method {{ background:#eef4ff; border-left:4px solid #3b82f6; padding:14px 18px; border-radius:6px; max-width:1000px; }}
  .schematic {{ display:block; width:100%; height:auto; margin:16px 0 18px; background:white; border:1px solid var(--line); border-radius:12px; }}
  .schematic .box {{ fill:#f8fafc; stroke:#94a3b8; stroke-width:2; }}
  .schematic .roi-box {{ fill:#fff1f2; stroke:#ef4444; stroke-width:2; }}
  .schematic .step {{ font-size:17px; font-weight:700; fill:#172033; text-anchor:middle; }}
  .schematic .detail {{ font-size:13px; fill:#64748b; text-anchor:middle; }}
  .schematic .arrow {{ stroke:#3b82f6; stroke-width:3; fill:none; marker-end:url(#arrowhead); }}
  .case {{ background:white; border:1px solid var(--line); border-radius:12px; padding:18px; margin:18px 0; box-shadow:0 2px 8px #1720330d; }}
  .case header p {{ margin:4px 0; color:#475569; }} .case h3 {{ margin:9px 0 4px; font-size:17px; }}
  .badge,.selection {{ display:inline-block; border-radius:999px; padding:4px 9px; margin-right:6px; font-size:12px; font-weight:700; }}
  .badge {{ background:#e2e8f0; }} .selection {{ background:#f1f5f9; color:#475569; }}
  .tp .badge {{ background:#dcfce7; color:#166534; }} .tn .badge {{ background:#dbeafe; color:#1e40af; }}
  .fp .badge,.fn .badge {{ background:#fee2e2; color:#991b1b; }}
  .images {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; margin-top:15px; }}
  figure {{ margin:0; }} img {{ width:100%; aspect-ratio:1; object-fit:contain; background:#050505; border-radius:7px; }}
  figcaption {{ text-align:center; color:var(--muted); margin-top:5px; }}
  .warning {{ background:#fff7ed; border:1px solid #fed7aa; padding:14px 16px; border-radius:8px; }}
  code {{ background:#eef2f7; padding:1px 5px; border-radius:4px; }}
  @media (max-width:760px) {{ .images {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body><main>
<h1>Pneumonia CNN: HiResCAM prediction review</h1>
<p class="lead"><b>Checkpoint:</b> <code>{checkpoint_label}</code> · <b>Data:</b> <code>{data_label}</code>. {len(selected)} deterministic examples from this {len(records)}-image test/validation set. Each map explains the model's predicted class using position-specific gradients from the final convolutional layer.</p>

<section class="metrics">
  <div class="metric"><b>{accuracy:.1%}</b><span>Accuracy</span></div>
  <div class="metric"><b>{auc:.3f}</b><span>ROC-AUC</span></div>
  <div class="metric"><b>{sensitivity:.1%}</b><span>Pneumonia sensitivity</span></div>
  <div class="metric"><b>{specificity:.1%}</b><span>Normal specificity</span></div>
  <div class="metric"><b>{precision:.1%}</b><span>Pneumonia precision</span></div>
</section>

<h2>Confusion matrix</h2>
<table class="matrix"><tr><th>True / predicted</th><th>Normal</th><th>Pneumonia</th></tr>
<tr><th>Normal</th><td>{tn}</td><td>{fp}</td></tr><tr><th>Pneumonia</th><td>{fn}</td><td>{tp}</td></tr></table>

<h2>How the ROI was produced</h2>
<svg class="schematic" viewBox="0 0 1220 260" role="img" aria-labelledby="gradcam-title gradcam-desc">
  <title id="gradcam-title">HiResCAM to ROI workflow</title>
  <desc id="gradcam-desc">The X-ray passes through the CNN. Every activation from the last convolution is multiplied by its position-specific gradient for the predicted-class score, then the contributions are summed, rectified, normalized, resized, and thresholded to create the red ROI.</desc>
  <defs>
    <marker id="arrowhead" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#3b82f6"/>
    </marker>
  </defs>

  <rect class="box" x="20" y="68" width="140" height="110" rx="10"/>
  <text class="step" x="90" y="107">Input X-ray</text>
  <text class="detail" x="90" y="135">{input_channels} × {input_height} × {input_width}</text>

  <path class="arrow" d="M160 123 H190"/>
  <rect class="box" x="195" y="68" width="140" height="110" rx="10"/>
  <text class="step" x="265" y="107">CNN forward</text>
  <text class="detail" x="265" y="135">prediction + score</text>

  <path class="arrow" d="M335 123 H365"/>
  <rect class="box" x="370" y="68" width="150" height="110" rx="10"/>
  <text class="step" x="445" y="102">Last conv maps</text>
  <text class="detail" x="445" y="130">{map_channels} maps, {map_height} × {map_width}</text>
  <text class="detail" x="445" y="151">layer conv[3]</text>

  <path class="arrow" d="M520 123 H550"/>
  <rect class="box" x="555" y="68" width="150" height="110" rx="10"/>
  <text class="step" x="630" y="102">Backpropagate</text>
  <text class="detail" x="630" y="130">predicted-class score</text>
  <text class="detail" x="630" y="151">keep gradients by position</text>

  <path class="arrow" d="M705 123 H735"/>
  <rect class="box" x="740" y="68" width="150" height="110" rx="10"/>
  <text class="step" x="815" y="102">Combine maps</text>
  <text class="detail" x="815" y="130">Σ gradient × activation</text>
  <text class="detail" x="815" y="151">then ReLU</text>

  <path class="arrow" d="M890 123 H920"/>
  <rect class="box" x="925" y="68" width="125" height="110" rx="10"/>
  <text class="step" x="987" y="102">Heatmap</text>
  <text class="detail" x="987" y="130">normalize 0–1</text>
  <text class="detail" x="987" y="151">resize to image</text>

  <path class="arrow" d="M1050 123 H1080"/>
  <rect class="roi-box" x="1085" y="68" width="115" height="110" rx="10"/>
  <text class="step" x="1142" y="102">Red ROI</text>
  <text class="detail" x="1142" y="130">heatmap &gt;</text>
  <text class="detail" x="1142" y="151">{roi_threshold:.2f}</text>

  <text class="detail" x="610" y="223">Position-wise activation × gradient gives each location's signed contribution to the predicted-class score.</text>
</svg>
<p class="method">The target is the predicted class: pre-sigmoid logit <code>z</code> for pneumonia and <code>−z</code> for normal. HiResCAM multiplies each value in <code>conv[3]</code>'s {map_channels} feature maps of size {map_height}×{map_width} by the gradient at that same channel and position, sums across channels, keeps positive contributions with ReLU, divides by the maximum, and resizes to {input_height}×{input_width}. Red ROI pixels have HiResCAM intensity above {roi_threshold:.2f}. Unlike channel-averaged Grad-CAM, this preserves the spatially varying weights used by this CNN's flatten-and-linear classifier.</p>

<h2>Representative predictions</h2>
{''.join(cards)}

<p class="warning"><b>Interpretation limit:</b> HiResCAM shows positive contributions to the model score at the final convolutional layer. Its native spatial resolution here is only {map_height}×{map_width}, so the resized overlay cannot precisely localize pathology. The red ROI is not a lesion segmentation, does not establish clinical correctness, and has not been validated by a radiologist. It may faithfully reveal that the model relies on ribs, borders, markers, or other shortcuts rather than lung pathology. This dataset's test folder was used as validation during notebook training, so these results are not an untouched clinical evaluation.</p>
</main></body></html>"""


def main() -> None:
    args = parse_args()
    if not 0.0 < args.roi_threshold < 1.0:
        raise ValueError("--roi-threshold must lie between 0 and 1")
    root = args.root.resolve()
    checkpoint_path = root / args.checkpoint
    data_dir = root / args.data_dir
    model, checkpoint = load_model(checkpoint_path)
    dataset = make_dataset(data_dir)
    records = evaluate(model, dataset)
    selected = select_examples(records, args.examples_per_category)
    report = build_report(
        records,
        selected,
        model,
        checkpoint,
        args.roi_threshold,
        checkpoint_path.name,
        data_dir,
    )
    output = root / args.output
    output.write_text(report, encoding="utf-8")
    counts = {category: sum(r.category == category for r in records) for category in CATEGORY_NAMES}
    print(f"Wrote {output}")
    print(f"Evaluated {len(records)} images; outcomes: {counts}")
    print(f"Embedded {len(selected)} HiResCAM examples")


if __name__ == "__main__":
    main()
