# run_preprocess.py

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.loaders import make_dataloaders, LoaderConfig
from datasets.physiomio_four_channel import (
    FourChannelPhysioMioConfig,
    build_physiomio_four_channel_views,
)


def _write_synthetic_physiomio_fixture(raw_dir: Path, *, seed: int, patients: int = 10) -> None:
    rng = np.random.default_rng(seed)
    labels = ("Rest", "MassFlexion", "HookGrasp")
    samples_per_label = 1024
    channel_names = [f"channel_{i:02d}" for i in range(1, 65)]

    for patient_idx in range(1, patients + 1):
        patient_dir = raw_dir / f"patient{patient_idx:02d}" / "impaired_arm"
        patient_dir.mkdir(parents=True, exist_ok=True)
        rows = samples_per_label * len(labels)
        data = {
            ch: rng.normal(loc=0.0, scale=0.05, size=rows).astype(np.float32)
            for ch in channel_names
        }
        movement_type = []
        for label in labels:
            movement_type.extend([label] * samples_per_label)
        data["movement_type"] = movement_type
        pd.DataFrame(data).to_parquet(patient_dir / "synthetic_smoke.parquet")


# Removes existing processed tensors so data can be rebuilt
def _delete_existing(processed_dir: Path, train_file: str, val_file: str, test_file: str) -> None:
    for name in (train_file, val_file, test_file):
        p = processed_dir / name
        if p.exists():
            p.unlink()


def _build_one_split(
    *,
    arm_split: str,
    raw_dir: Path,
    processed_dir: Path,
    args: argparse.Namespace,
) -> None:
    os.makedirs(processed_dir, exist_ok=True)

    if args.force:
        _delete_existing(processed_dir, args.train_file, args.val_file, args.test_file)

    cfg = LoaderConfig(
        batch_size=64,
        num_workers=0,
        seed=args.seed,
        out_dim=args.out_dim,
        use_physiomio_if_missing=True,
        physiomio_raw_dir=str(raw_dir),
        physiomio_fs=float(args.fs if args.fs is not None else 2048.0),
        arm_split=arm_split,
        impaired_only=(arm_split == "impaired"),
        min_segment_samples=int(args.min_seg_samples),
        skip_rest=bool(args.skip_rest),
        max_patients=args.max_patients,
    )

    make_dataloaders(
        processed_dir=str(processed_dir),
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        cfg=cfg,
    )

    print(f"Saved {arm_split} processed data under: {processed_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()

    # Raw PhysioMio location
    parser.add_argument("--raw-dir", type=str, default="datasets/raw/physiomio")

    # Where processed tensors will be stored
    parser.add_argument("--processed-dir", type=str, default="datasets/processed/physiomio")

    parser.add_argument("--train-file", type=str, default="train.pt")
    parser.add_argument("--val-file", type=str, default="val.pt")
    parser.add_argument("--test-file", type=str, default="test.pt")

    parser.add_argument("--fs", type=float, default=None)
    parser.add_argument("--four-channel", action="store_true", default=False)
    parser.add_argument("--synthetic-smoke", action="store_true", default=False)
    parser.add_argument("--arm-split", choices=("impaired", "healthy", "both"), default="impaired")
    parser.add_argument("--separate-arm-dirs", action="store_true", default=False)
    parser.add_argument("--impaired-only", action="store_true", default=None)
    parser.add_argument("--include-healthy", action="store_true", default=False)

    parser.add_argument("--min-seg-samples", type=int, default=200)
    parser.add_argument("--skip-rest", action="store_true", default=False)
    parser.add_argument("--max-patients", type=int, default=None)

    parser.add_argument("--out-dim", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--force", action="store_true", default=False)

    args = parser.parse_args()

    print("Starting PhysioMio preprocessing...")

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    smoke_tmp = None
    if args.synthetic_smoke:
        if not args.four_channel:
            raise ValueError("--synthetic-smoke is currently supported for --four-channel preprocessing.")
        smoke_tmp = tempfile.mkdtemp(prefix="physiomio_4ch_smoke_")
        raw_dir = Path(smoke_tmp) / "raw" / "physiomio"
        _write_synthetic_physiomio_fixture(raw_dir, seed=args.seed)
        args.max_patients = args.max_patients or 10
        args.min_seg_samples = min(int(args.min_seg_samples), 200)

    arm_split = args.arm_split
    if args.include_healthy:
        arm_split = "both"
    elif args.impaired_only is True:
        arm_split = "impaired"

    if args.four_channel:
        cfg = FourChannelPhysioMioConfig(
            raw_root=str(raw_dir),
            processed_root=str(processed_dir),
            seed=args.seed,
            fs=float(args.fs if args.fs is not None else 2048.0),
            arm_split=arm_split,
            impaired_only=(arm_split == "impaired"),
            min_segment_samples=int(args.min_seg_samples),
            skip_rest=bool(args.skip_rest),
            max_patients=args.max_patients,
        )
        build_physiomio_four_channel_views(cfg)
        print("Four-channel PhysioMio processed views ready.")
        if smoke_tmp:
            shutil.rmtree(smoke_tmp, ignore_errors=True)
        return

    if arm_split == "both" and args.separate_arm_dirs:
        for split_name in ("healthy", "impaired"):
            _build_one_split(
                arm_split=split_name,
                raw_dir=raw_dir,
                processed_dir=processed_dir / split_name,
                args=args,
            )

        print("Processed data ready.")
        return

    _build_one_split(
        arm_split=arm_split,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        args=args,
    )

    print("Processed data ready.")


if __name__ == "__main__":
    main()
