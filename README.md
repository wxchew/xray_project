# X-ray CNN Project

Chest X-ray pneumonia classifier notebook, adapted from the
[SIB-SWISS PyTorch practical training course](https://github.com/sib-swiss/pytorch-practical-training)
(notebook `04_chest_Xray_CNN.ipynb`).

Dataset: Kermany et al. 2018 chest X-ray images, downsampled to 64x64 and 224x224
(`data/chest_xray_64`, `data/chest_xray_224`), as provided by the course repo.

## Contents

- `04_chest_Xray_CNN.ipynb` — main notebook
- `pytorchtools.py` — early stopping helper ([Bjarten/early-stopping-pytorch](https://github.com/Bjarten/early-stopping-pytorch))
- `solutions/` — reference solution snippets used by the notebook's `%load` cells
- `data/` — chest X-ray image subsets (64px and 224px)
- `images/` — sample X-ray thumbnails used in the notebook
