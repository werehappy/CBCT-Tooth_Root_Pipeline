"""ToothFairy2 label scheme and helpers to pull out teeth / jawbone masks.

The official ToothFairy2 dataset (Dataset112) has 42 foreground classes. The
canonical IDs are listed below for reference. IMPORTANT: label IDs must match
the checkpoint you actually run -- always cross-check against the dataset.json
shipped with your downloaded weights. If your checkpoint differs, override
`teeth_labels` / `bone_labels` in config.yaml instead of editing this file.
"""

from __future__ import annotations

import logging
from typing import Iterable, List

import numpy as np

log = logging.getLogger(__name__)

# --- Canonical ToothFairy2 label map (reference) -----------------------------
TOOTHFAIRY2_LABELS = {
    0: "background",
    1: "lower_jawbone",
    2: "upper_jawbone",
    3: "left_inferior_alveolar_canal",
    4: "right_inferior_alveolar_canal",
    5: "left_maxillary_sinus",
    6: "right_maxillary_sinus",
    7: "pharynx",
    8: "bridge",
    9: "crown",
    10: "implant",
    # Permanent teeth follow the FDI two-digit numbering (quadrant 1-4, tooth 1-8)
    **{fdi: f"tooth_{fdi}"
       for q in (10, 20, 30, 40)
       for fdi in range(q + 1, q + 9)},
}

# --- DentalSegmentator label map (Dataset111_453CT, nnU-Net v2.2) ------------
# 5 classes. Teeth are solid upper/lower masks, NOT per-tooth FDI numbers.
DENTALSEGMENTATOR_LABELS = {
    0: "background",
    1: "upper_skull",       # maxilla / upper facial skeleton
    2: "mandible",
    3: "upper_teeth",
    4: "lower_teeth",
    5: "mandibular_canal",
}
DENTALSEG_BONE_LABELS = [1, 2]    # upper skull + mandible
DENTALSEG_TEETH_LABELS = [3, 4]   # upper + lower teeth

# Default groupings (override in config if your checkpoint differs)
JAWBONE_LABELS: List[int] = [1, 2]
PERMANENT_TEETH_LABELS: List[int] = [fdi for q in (10, 20, 30, 40)
                                     for fdi in range(q + 1, q + 9)]
DENTAL_WORK_LABELS: List[int] = [8, 9, 10]  # bridge / crown / implant


def mask_from_labels(labelmap: np.ndarray, labels: Iterable[int]) -> np.ndarray:
    """Boolean mask = union of the requested label IDs."""
    labels = list(labels)
    if not labels:
        return np.zeros_like(labelmap, dtype=bool)
    return np.isin(labelmap, labels)


def extract_teeth(labelmap: np.ndarray, teeth_labels: Iterable[int],
                  include_crowns: bool = False) -> np.ndarray:
    labels = list(teeth_labels)
    if include_crowns:
        labels = labels + DENTAL_WORK_LABELS
    mask = mask_from_labels(labelmap, labels)
    log.info("Teeth mask: %d labels, %d voxels", len(labels), int(mask.sum()))
    return mask


def extract_bone(labelmap: np.ndarray, bone_labels: Iterable[int]) -> np.ndarray:
    mask = mask_from_labels(labelmap, bone_labels)
    log.info("Bone mask: %d voxels", int(mask.sum()))
    return mask
