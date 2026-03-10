# <p align=center>`Triad: Vision Foundation Model for 3D Magnetic Resonance Imaging`</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) ![Paper Status](https://img.shields.io/badge/Paper%20Status-Accepted%20by%20Medical%20Image%20Analysis-brightgreen)

This paper has been accepted by Medical Image Analysis.

- Our paper: https://www.sciencedirect.com/science/article/pii/S1361841526000617

## Quick Start

This repository provides a minimal example in [QuickStart.py](/data/Code2025/3DSimMIMdemo/QuickStart.py) for loading released encoder/backbone weights.

Available weights in this directory:

- `Triad-PlainConvUNet-MAE.pth`
- `Triad-PlainConvUNet-SimMIM.pth`
- `Triad-SwinB-MAE.pth`
- `Triad-SwinB-SimMIM.pth`

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
@article{wang2025triad,
  title={Triad: Vision Foundation Model for 3D Magnetic Resonance Imaging},
  author={Wang, Shansong and Safari, Mojtaba and Li, Qiang and Chang, Chih-Wei and Qiu, Richard LJ and Roper, Justin and Yu, David S and Yang, Xiaofeng},
  journal={arXiv preprint arXiv:2502.14064},
  year={2025}
}
```

## Acknowledgments

- This project is based on VoCo v2: https://github.com/Luffy03/Large-Scale-Medical
