#!/usr/bin/env python3
"""CLI for the CBCT trabecular pipeline.

Examples
--------
Single volume (runs nnU-Net):
    python run.py -i data/123.nii.gz -o out/

Batch a folder of volumes:
    python run.py -i data/ -o out/

Skip nnU-Net and reuse an existing labelmap (no GPU/weights needed):
    python run.py -i data/123.nii.gz -o out/ --labelmap labels/123_seg.nii.gz

Override the patient id (otherwise derived from the filename):
    python run.py -i scan.nii.gz -o out/ --patient-id 123
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from cbct_pipeline.pipeline import run_pipeline

INPUT_SUFFIXES = (".nii", ".nii.gz")


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def find_inputs(path: Path) -> list[Path]:
    if path.is_dir():
        files = [p for p in sorted(path.iterdir())
                 if p.name.endswith(INPUT_SUFFIXES)]
        if not files:
            raise SystemExit(f"No .nii/.nii.gz files in {path}")
        return files
    if not path.name.endswith(INPUT_SUFFIXES):
        raise SystemExit(f"Input is not a .nii/.nii.gz file: {path}")
    return [path]


def main() -> int:
    ap = argparse.ArgumentParser(description="CBCT teeth/bone/trabecular pipeline")
    ap.add_argument("-i", "--input", required=True, type=Path,
                    help="Input .nii/.nii.gz file or a directory of them")
    ap.add_argument("-o", "--output", required=True, type=Path,
                    help="Output directory")
    ap.add_argument("-c", "--config", type=Path, default=Path("config.yaml"))
    ap.add_argument("--patient-id", default=None,
                    help="Override patient id (single-file mode only)")
    ap.add_argument("--labelmap", type=Path, default=None,
                    help="Reuse an existing nnU-Net labelmap; skips inference")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = load_config(args.config)
    inputs = find_inputs(args.input)

    if args.labelmap and len(inputs) > 1:
        raise SystemExit("--labelmap can only be used with a single input file")
    if args.patient_id and len(inputs) > 1:
        raise SystemExit("--patient-id can only be used with a single input file")

    failures = 0
    for vol in inputs:
        try:
            run_pipeline(vol, args.output, cfg,
                         patient_id=args.patient_id,
                         labelmap_path=args.labelmap)
        except Exception as exc:  # keep batch going on per-case failure
            failures += 1
            logging.error("FAILED on %s: %s", vol.name, exc, exc_info=args.verbose)

    if failures:
        logging.error("%d/%d case(s) failed", failures, len(inputs))
        return 1
    logging.info("All %d case(s) completed", len(inputs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
