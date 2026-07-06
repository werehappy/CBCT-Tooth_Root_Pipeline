"""CBCT trabecular segmentation pipeline.

Input : a CBCT volume as .nii / .nii.gz
Stages: nnU-Net (ToothFairy2) multi-structure segmentation
        -> split into teeth / jawbone
        -> separate cortical vs trabecular bone
Output: <patient>_<bone>.nii.gz, <patient>_teeth.nii.gz, <patient>_trabecular.nii.gz
"""

__version__ = "0.1.0"
