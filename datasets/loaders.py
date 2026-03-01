# datasets/loaders.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.dataset import RawSample, ProcessedTensorDataset, load_raw_sample

# uses your existing preprocessing.py
from data_processing.processing.preprocess import preprocess_emg
from data_processing.preprocess_config import PreprocessConfig


@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1
    seed: int = 42


@dataclass(frozen=True)
class DataConfig:
    raw_dir: str = "datasets/raw"
    processed_dir: str = "datasets/processed"
    out_dim: int = 5
    batch_size: int = 64
    num_workers: int = 0
    pin_memory: bool = True
    drop_last: bool = False


def _split_indices(n: int, split: SplitConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n <= 0:
        raise ValueError("No samples found to split.")
    total = split.train + split.val + split.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1. Got {total}.")

    rng = np.random.default_rng(split.seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    n_train = int(round(n * split.train))
    n_val = int(round(n * split.val))
    n_test = n - n_train - n_val

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    if len(test_idx) != n_test:
        test_idx = idx[n_train + n_val : n_train + n_val + n_test]

    return train_idx, val_idx, test_idx


def _find_raw_samples(raw_dir: Path) -> List[RawSample]:
    """
    Priority order:
      1) datasets/raw/manifest.json (recommended)
      2) auto-discover .npz and .pt files directly under raw_dir (or subfolders)

    manifest.json schema:
    {
      "samples": [
        {"path": "subject1/trial_01.npz", "fs": 2000},
        {"path": "subject1/trial_02.pt",  "fs": 2000}
      ]
    }
    """
    manifest = raw_dir / "manifest.json"
    samples: List[RawSample] = []

    if manifest.exists():
        obj = json.loads(manifest.read_text())
        entries = obj.get("samples", [])
        for e in entries:
            p = raw_dir / e["path"]
            fs = e.get("fs", None)
            samples.append(RawSample(path=p, fs=float(fs) if fs is not None else None))
        return samples

    # auto-discover fallback
    for p in raw_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".npz", ".pt"):
            samples.append(RawSample(path=p, fs=None))

    # stable ordering
    samples.sort(key=lambda s: str(s.path))
    return samples


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _processed_split_paths(processed_dir: Path) -> Dict[str, Path]:
    return {
        "train": processed_dir / "train.pt",
        "val": processed_dir / "val.pt",
        "test": processed_dir / "test.pt",
    }


def processed_exists(processed_dir: str | Path) -> bool:
    processed_dir = Path(processed_dir)
    paths = _processed_split_paths(processed_dir)
    return all(p.exists() for p in paths.values())


def build_processed_splits_if_needed(
    *,
    data_cfg: DataConfig,
    split_cfg: SplitConfig,
    preprocess_cfg: Optional[PreprocessConfig] = None,
) -> None:
    """
    If datasets/processed/{train,val,test}.pt don't exist, build them from datasets/raw/.
    """
    raw_dir = Path(data_cfg.raw_dir)
    processed_dir = Path(data_cfg.processed_dir)
    _ensure_dir(processed_dir)

    paths = _processed_split_paths(processed_dir)
    if all(p.exists() for p in paths.values()):
        return

    if preprocess_cfg is None:
        preprocess_cfg = PreprocessConfig()

    samples = _find_raw_samples(raw_dir)
    if len(samples) == 0:
        raise FileNotFoundError(
            f"No raw samples found in {raw_dir}. Add .npz/.pt files or create {raw_dir/'manifest.json'}."
        )

    # load + preprocess all samples into memory (fine for now; later you can stream)
    X_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []

    for s in samples:
        if not s.path.exists():
            raise FileNotFoundError(f"Raw sample not found: {s.path}")

        emg, y, fs_file, _meta = load_raw_sample(s.path)
        fs = s.fs if s.fs is not None else fs_file
        if fs is None:
            raise ValueError(f"Missing sampling rate fs for {s.path}. Put it in the file or manifest.json.")

        # emg: (n_samples, n_channels)
        # preprocess_emg returns X as (C, W, F) based on your code
        X = preprocess_emg(emg, fs=fs, config=preprocess_cfg)  # (C,W,F)
        if X.ndim != 3:
            raise ValueError(f"Expected preprocess_emg -> (C,W,F). Got shape {X.shape} for {s.path}")

        y = np.asarray(y).astype(np.float32).reshape(-1)
        if y.shape[0] != data_cfg.out_dim:
            raise ValueError(f"Expected y to have shape ({data_cfg.out_dim},). Got {y.shape} for {s.path}")

        X_list.append(torch.from_numpy(np.asarray(X, dtype=np.float32)))
        y_list.append(torch.from_numpy(y))

    X_all = torch.stack(X_list, dim=0)  # (N,C,W,F)
    y_all = torch.stack(y_list, dim=0)  # (N,out_dim)

    train_idx, val_idx, test_idx = _split_indices(X_all.shape[0], split_cfg)

    def save_split(name: str, idx: np.ndarray) -> None:
        payload = {
            "X": X_all[idx],
            "y": y_all[idx],
            "meta": {
                "raw_dir": str(raw_dir),
                "n_total": int(X_all.shape[0]),
                "split": {"train": split_cfg.train, "val": split_cfg.val, "test": split_cfg.test},
                "seed": split_cfg.seed,
            },
        }
        torch.save(payload, paths[name])

    save_split("train", train_idx)
    save_split("val", val_idx)
    save_split("test", test_idx)


def make_dataloaders(
    *,
    data_cfg: DataConfig,
    split_cfg: SplitConfig,
    preprocess_cfg: Optional[PreprocessConfig] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Main entry:
      - ensures processed splits exist (builds from raw if missing)
      - returns PyTorch DataLoaders over (X, y) where:
          X: (C,W,F)
          y: (out_dim,)
    """
    build_processed_splits_if_needed(
        data_cfg=data_cfg,
        split_cfg=split_cfg,
        preprocess_cfg=preprocess_cfg,
    )

    processed_dir = Path(data_cfg.processed_dir)
    paths = _processed_split_paths(processed_dir)

    train_ds = ProcessedTensorDataset(paths["train"])
    val_ds = ProcessedTensorDataset(paths["val"])
    test_ds = ProcessedTensorDataset(paths["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg.batch_size,
        shuffle=True,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        drop_last=data_cfg.drop_last,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=data_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        pin_memory=data_cfg.pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader