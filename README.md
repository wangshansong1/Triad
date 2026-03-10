# <p align=center>`Vision foundation model for 3D magnetic resonance imaging segmentation, classification, and registration`</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) ![Paper Status](https://img.shields.io/badge/Paper%20Status-Accepted%20by%20Medical%20Image%20Analysis-brightgreen)

This paper has been accepted by Medical Image Analysis.

- Our paper: https://www.sciencedirect.com/science/article/pii/S1361841526000617

## Quick Start

This repository provides a minimal example in [QuickStart.py](/data/Code2025/3DSimMIMdemo/QuickStart.py) for loading released encoder/backbone weights.

Available weights in this directory:

- `Triad-PlainConvUNet-MAE.pth`
  https://drive.google.com/file/d/1Mc5owEFhWkroe5Hjnk7Ex8v6Z-aBKF3U/view?usp=drive_link

- `Triad-PlainConvUNet-SimMIM.pth`
  https://drive.google.com/file/d/1EkrYbuNI64yi1_Yl4JZUM5yZ5K0mHzR3/view?usp=drive_link

- `Triad-SwinB-MAE.pth`
  https://drive.google.com/file/d/1F_6TNrCxPyqk-bPzXj0HHLxFy9mbNzjl/view?usp=drive_link

- `Triad-SwinB-SimMIM.pth`
  https://drive.google.com/file/d/1icLjmSpTdEAA9kEW3BWHnAYv-hsXMYxS/view?usp=drive_link

### PlainConvUNet

The default runnable example in `QuickStart.py` loads:

```python
ckpt = torch.load("Triad-PlainConvUNet-MAE.pth", weights_only=False)
```

You can switch it to:

- `Triad-PlainConvUNet-MAE.pth`
- `Triad-PlainConvUNet-SimMIM.pth`

Then run:

```bash
python QuickStart.py
```

### Swin-B

At the bottom of `QuickStart.py`, a Swin example is provided as commented code.

Uncomment the Swin block, then choose one checkpoint:

- `Triad-SwinB-MAE.pth`
- `Triad-SwinB-SimMIM.pth`

Set:

```python
ckpt = torch.load("Triad-SwinB-SimMIM.pth", weights_only=False)
```

Then run:

```bash
python QuickStart.py
```

## Citation

```bibtex
@article{WANG2026103992,
title = {Vision foundation model for 3D magnetic resonance imaging segmentation, classification, and registration},
journal = {Medical Image Analysis},
volume = {110},
pages = {103992},
year = {2026},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2026.103992},
url = {https://www.sciencedirect.com/science/article/pii/S1361841526000617},
author = {Shansong Wang and Mojtaba Safari and Qiang Li and Chih-Wei Chang and Richard {LJ Qiu} and Justin Roper and David S. Yu and Xiaofeng Yang},
}
```

## Acknowledgments

- This project is based on VoCo v2: https://github.com/Luffy03/Large-Scale-Medical
