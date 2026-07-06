"""NIfTI I/O helpers. Everything keeps the original affine/header so masks stay
voxel-aligned with the input volume."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage

log = logging.getLogger(__name__)


def smooth_surface(mask: np.ndarray, sigma_mm: float,
                   spacing: Tuple[float, float, float]) -> np.ndarray:
    """Light surface smoothing of a binary mask (Gaussian blur + 0.5 threshold).

    Rounds hard edges -- e.g. the flat root cut -- so the rendered surface reads
    smoothly instead of as a sharp step. sigma_mm is the blur in millimetres;
    it is converted to per-axis voxels using the volume spacing. 0 = no-op."""
    if not sigma_mm or sigma_mm <= 0:
        return mask
    sig = [float(sigma_mm) / float(s) for s in spacing]
    blurred = ndimage.gaussian_filter(mask.astype(np.float32), sigma=sig)
    return blurred >= 0.5


@dataclass
class Volume:
    """A loaded volume plus the metadata needed to write aligned masks back out."""

    data: np.ndarray          # float32 image (or int labelmap)
    affine: np.ndarray        # 4x4 world matrix
    header: nib.Nifti1Header  # original header (preserves spacing, etc.)

    @property
    def spacing(self) -> Tuple[float, float, float]:
        """Voxel size in mm along (i, j, k)."""
        zooms = self.header.get_zooms()[:3]
        return tuple(float(z) for z in zooms)


def load_volume(path: str | Path, *, as_int: bool = False) -> Volume:
    """Load a .nii/.nii.gz file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Volume not found: {path}")
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    arr = arr.astype(np.int16 if as_int else np.float32)
    log.info("Loaded %s  shape=%s  spacing=%s", path.name, arr.shape,
             tuple(round(z, 3) for z in img.header.get_zooms()[:3]))
    return Volume(data=arr, affine=img.affine, header=img.header)


def save_mask(mask: np.ndarray, ref: Volume, path: str | Path) -> Path:
    """Write a binary/integer mask using the reference volume's geometry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = nib.Nifti1Image(mask.astype(np.uint8), ref.affine, ref.header)
    out.header.set_data_dtype(np.uint8)
    nib.save(out, str(path))
    log.info("Wrote %s  (%d voxels set)", path.name, int(mask.astype(bool).sum()))
    return path
