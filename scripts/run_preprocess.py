# run_preprocess.py

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets.loaders import make_dataloaders, LoaderConfig


# Removes existing processed tensors so data can be rebuilt
def _delete_existing(processed_dir: Path, train_file: str, val_file: str, test_file: str) -> None:
    for name in (train_file, val_file, test_file):
        p = processed_dir / name
        if p.exists():
            p.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()

    # Raw PhysioMio location
    parser.add_argument("--raw-dir", type=str, default="datasets/raw/physiomio")

    # Where processed tensors will be stored
    parser.add_argument("--processed-dir", type=str, default="datasets/processed/physiomio")

    parser.add_argument("--train-file", type=str, default="train.pt")
    parser.add_argument("--val-file", type=str, default="val.pt")
    parser.add_argument("--test-file", type=str, default="test.pt")

    parser.add_argument("--fs", type=float, default=2000.0)
    parser.add_argument("--impaired-only", action="store_true", default=True)
    parser.add_argument("--include-healthy", action="store_true", default=False)

    parser.add_argument("--min-seg-samples", type=int, default=200)
    parser.add_argument("--skip-rest", action="store_true", default=False)

    parser.add_argument("--out-dim", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--force", action="store_true", default=False)

    args = parser.parse_args()

    print("Starting PhysioMio preprocessing...")

    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    os.makedirs(processed_dir, exist_ok=True)

    impaired_only = args.impaired_only and (not args.include_healthy)

    if args.force:
        _delete_existing(processed_dir, args.train_file, args.val_file, args.test_file)

    cfg = LoaderConfig(
        batch_size=64,
        num_workers=0,
        seed=args.seed,
        out_dim=args.out_dim,
        use_physiomio_if_missing=True,
        physiomio_raw_dir=str(raw_dir),
        physiomio_fs=float(args.fs),
        impaired_only=bool(impaired_only),
        min_segment_samples=int(args.min_seg_samples),
        skip_rest=bool(args.skip_rest),
    )

    # This call builds processed splits if they do not exist
    make_dataloaders(
        processed_dir=str(processed_dir),
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        cfg=cfg,
    )

    print("Processed data ready.")
    print(f"Saved under: {processed_dir}")


if __name__ == "__main__":
    main()