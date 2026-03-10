# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------

from pathlib import Path

from scipy.ndimage import zoom
from torch.utils.data import Dataset
import numpy as np
import glob
import os

class NpyVolumeDataset(Dataset):
    def __init__(self, data_dir, transform=None, normalize=True, text_dir=None, target_size=(96, 96, 96)):

        self.paths = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if len(self.paths) == 0:
            raise RuntimeError(f"No .npy files found in {data_dir}")
        self.transform = transform
        self.normalize = normalize
        self.data_dir = Path(data_dir)
        self.text_dir = Path(text_dir) if text_dir is not None else self._infer_text_dir(self.data_dir)
        self.target_size = _normalize_target_size(target_size)

    def __len__(self):
        return len(self.paths)

    def _infer_text_dir(self, data_dir):
        parts = list(data_dir.parts)
        for i, part in enumerate(parts):
            if part == "TRIADNPY":
                parts[i] = "TRIADTEXTNPY"
                return Path(*parts)
        return data_dir.parent / "TRIADTEXTNPY"

    def _resolve_text_path(self, volume_path):
        text_name = (
            volume_path.name
            .replace("oldsize:", "oldsize_")
            .replace("newsize:", "newsize_")
        )
        text_path = self.text_dir / text_name
        if not text_path.exists():
            raise FileNotFoundError(
                f"Text embedding file not found for {volume_path.name}: expected {text_path}"
            )
        return text_path

    def _robust_minmax_foreground(self, vol, qmin=0.5, qmax=99.5, eps=1e-6):
        finite_mask = np.isfinite(vol)
        if finite_mask.any():
            fg = vol[finite_mask]
            lo = np.percentile(fg, qmin)
            hi = np.percentile(fg, qmax)
            vol = np.nan_to_num(vol, nan=lo, posinf=hi, neginf=lo)
            vol = np.clip(vol, lo, hi)
            vol = (vol - lo) / (hi - lo + eps)
            vol[~np.isfinite(vol)] = 0.0
        else:
            vol = np.zeros_like(vol, dtype=np.float32)
        return vol.astype(np.float32)

    def __getitem__(self, idx):
        p = Path(self.paths[idx])
        vol = np.load(p)
        p_text = self._resolve_text_path(p)
        text = np.load(p_text).astype(np.float32)
        vol = vol.astype(np.float32)

        if self.normalize:
            vol = self._robust_minmax_foreground(vol)

        vol = resize_volume(vol, target_size=self.target_size)

        mask = None
        if self.transform is not None:
            transformed = self.transform(vol)
            if isinstance(transformed, tuple):
                vol, mask = transformed
            else:
                vol = transformed

        return vol, text, mask
    
def _normalize_target_size(target_size):
    if isinstance(target_size, int):
        return (target_size, target_size, target_size)
    if len(target_size) != 3:
        raise ValueError(f"target_size must be an int or length-3 sequence, got {target_size}")
    return tuple(int(v) for v in target_size)


def resize_volume(vol, target_size=(96,96,96)):
    target_size = _normalize_target_size(target_size)
    factors = [t/s for t, s in zip(target_size, vol.shape)]
    vol_resized = zoom(vol, factors, order=1)  # order=1 线性插值
    return vol_resized.astype(np.float32)
