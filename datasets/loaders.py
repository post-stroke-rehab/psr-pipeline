# datasets/loaders.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.dataset import ProcessedTensorDataset
from data_processing.preprocess import preprocess_emg
from data_processing.preprocess_config import PreprocessConfig

import pandas as pd
from data_processing.mapping import gesture_to_5bit


# Loader configuration used by training_pipeline.py
@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int = 64
    num_workers: int = 0
    seed: int = 42
    out_dim: int = 5
    pin_memory: bool = True
    drop_last: bool = False

    # PhysioMio settings
    use_physiomio_if_missing: bool = True
    physiomio_raw_dir: str = "datasets/raw/physiomio"
    physiomio_fs: float = 2000.0
    impaired_only: bool = True
    min_segment_samples: int = 200
    skip_rest: bool = False
    max_patients: Optional[int] = 22


# Makes sure a folder exists
def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# Where we look for train/val/test processed tensors
def _processed_paths(processed_dir: str | Path, train_file: str, val_file: str, test_file: str) -> Dict[str, Path]:
    p = Path(processed_dir)
    return {
        "train": p / train_file,
        "val": p / val_file,
        "test": p / test_file,
    }


# Quick check to see if processed data already exists
def _processed_exist(paths: Dict[str, Path]) -> bool:
    return all(paths[k].exists() for k in ("train", "val", "test"))


# Finds all PhysioMio parquet files
# We usually train only on impaired_arm recordings
def _find_physiomio_parquets(raw_root: Path, impaired_only: bool) -> List[Path]:
    if not raw_root.exists():
        return []

    if impaired_only:
        return sorted(raw_root.rglob("impaired_arm/*.parquet"))

    return sorted(raw_root.rglob("healthy_arm/*.parquet")) + \
           sorted(raw_root.rglob("impaired_arm/*.parquet"))


# Extracts patient ID from the folder name (e.g., patient12)
def _patient_id_from_path(p: Path) -> str:
    for part in p.parts:
        if part.lower().startswith("patient"):
            return part
    return "unknown_patient"


# Collects EMG channel columns (channel_01 ... channel_64)
def _get_channel_cols(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if str(c).startswith("channel_")]
    if not cols:
        raise ValueError("No channel_* columns found in parquet.")
    return sorted(cols)


# Breaks a recording into contiguous chunks of the same movement_type
# Each chunk becomes one training sample
def _iter_contiguous_segments(mv: np.ndarray) -> List[Tuple[int, int, str]]:
    if mv.size == 0:
        return []

    starts = [0]
    for i in range(1, len(mv)):
        if mv[i] != mv[i - 1]:
            starts.append(i)

    ends = starts[1:] + [len(mv)]
    return [(s, e, str(mv[s])) for s, e in zip(starts, ends)]


# Splits by patient instead of randomly by segment
# This prevents leakage between train and test
def _patient_split_indices(patient_keys: List[str], cfg: LoaderConfig):
    rng = np.random.default_rng(cfg.seed)
    patients = sorted(set(patient_keys))
    rng.shuffle(patients)

    train_frac = 0.7
    val_frac = 0.15
    test_frac = 0.15

    nP = len(patients)
    n_train = int(round(nP * train_frac))
    n_val = int(round(nP * val_frac))

    trainP = set(patients[:n_train])
    valP = set(patients[n_train : n_train + n_val])
    testP = set(patients[n_train + n_val :])

    idx = np.arange(len(patient_keys))

    train_idx = idx[[p in trainP for p in patient_keys]]
    val_idx = idx[[p in valP for p in patient_keys]]
    test_idx = idx[[p in testP for p in patient_keys]]

    return train_idx, val_idx, test_idx, patients


# Builds processed splits from raw PhysioMio parquet files
def _build_processed_from_physiomio(
    *,
    raw_root: Path,
    paths: Dict[str, Path],
    cfg: LoaderConfig,
    preprocess_cfg: Optional[PreprocessConfig] = None,
) -> None:

    if preprocess_cfg is None:
        preprocess_cfg = PreprocessConfig()

    parquets = _find_physiomio_parquets(raw_root, cfg.impaired_only)

    rng = np.random.default_rng(cfg.seed)

    by_patient: Dict[str, List[Path]] = {}
    for pq in parquets:
        pid = _patient_id_from_path(pq)
        by_patient.setdefault(pid, []).append(pq)

    patients = sorted(by_patient.keys())
    rng.shuffle(patients)

    if cfg.max_patients is not None:
        patients = patients[:cfg.max_patients]

    parquets = []
    for pid in patients:
        parquets.extend(by_patient[pid])

    parquets = sorted(parquets)

    if len(parquets) == 0:
        raise FileNotFoundError("No PhysioMio parquet files found.")

    print(f"Starting PhysioMio preprocessing ({len(parquets)} parquet files)...")

    X_list: List[torch.Tensor] = []
    y_list: List[torch.Tensor] = []
    patient_keys: List[str] = []
    unknown_labels: Dict[str, int] = {}

    for i_pq, pq in enumerate(parquets, start=1):
        print(f"[{i_pq}/{len(parquets)}] Reading {pq}")

        df = pd.read_parquet(pq)

        if "movement_type" not in df.columns:
            raise ValueError(f"{pq} missing movement_type column.")

        ch_cols = _get_channel_cols(df)
        mv = df["movement_type"].astype(str).to_numpy()

        for s, e, label in _iter_contiguous_segments(mv):

            if (e - s) < cfg.min_segment_samples:
                continue

            if cfg.skip_rest and label.lower() == "rest":
                continue

            try:
                y5 = np.asarray(gesture_to_5bit(label), dtype=np.float32).reshape(-1)
            except Exception:
                unknown_labels[label] = unknown_labels.get(label, 0) + 1
                continue

            if y5.shape[0] != cfg.out_dim:
                raise ValueError(f"Label dim mismatch for {label}: got {y5.shape[0]}, expected {cfg.out_dim}")

            emg_seg = df.iloc[s:e][ch_cols].to_numpy(dtype=np.float32)

            # Full preprocessing pipeline
            X = preprocess_emg(emg_seg, fs=float(cfg.physiomio_fs), config=preprocess_cfg)

            X_list.append(torch.from_numpy(np.asarray(X, dtype=np.float32)))
            y_list.append(torch.from_numpy(y5))
            patient_keys.append(_patient_id_from_path(pq))

            if len(X_list) % 500 == 0:
                print(f"Built {len(X_list)} samples so far...")

    if unknown_labels:
        top = sorted(unknown_labels.items(), key=lambda kv: kv[1], reverse=True)[:10]
        raise ValueError(
            "movement_type labels not covered by mapping.py: "
            + ", ".join([f"{k}({v})" for k, v in top])
        )

    if len(X_list) == 0:
        raise ValueError("No usable segments were produced.")

    # Make all samples the same (C, W, F) so torch.stack works.
    W_max = max(int(x.shape[1]) for x in X_list)

    X_fixed: List[torch.Tensor] = []
    for x in X_list:
        C, W, F = x.shape

        if W < W_max:
            pad = torch.zeros((C, W_max - W, F), dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        elif W > W_max:
            x = x[:, :W_max, :]

        X_fixed.append(x)

    X_all = torch.stack(X_fixed, dim=0)
    y_all = torch.stack(y_list, dim=0)

    train_idx, val_idx, test_idx, patients = _patient_split_indices(patient_keys, cfg)

    def save_split(name: str, split_idx: np.ndarray) -> None:
        payload = {
            "X": X_all[split_idx],
            "y": y_all[split_idx],
            "meta": {
                "source": "physiomio",
                "patients": patients,
                "seed": cfg.seed,
                "fs": float(cfg.physiomio_fs),
                "impaired_only": bool(cfg.impaired_only),
            },
        }
        torch.save(payload, paths[name])

    save_split("train", train_idx)
    save_split("val", val_idx)
    save_split("test", test_idx)

    print(f"Preprocessing complete. Saved train/val/test to {paths['train'].parent}")


# Creates PyTorch DataLoaders from the processed splits
def make_dataloaders(
    *,
    processed_dir: str = "datasets/processed",
    train_file: str = "train.pt",
    val_file: str = "val.pt",
    test_file: str = "test.pt",
    cfg: LoaderConfig = LoaderConfig(),
    preprocess_cfg: Optional[PreprocessConfig] = None,
):

    paths = _processed_paths(processed_dir, train_file, val_file, test_file)
    _ensure_dir(Path(processed_dir))

    # If processed files already exist, nothing to do
    if not _processed_exist(paths) and cfg.use_physiomio_if_missing:
        raw_root = Path(cfg.physiomio_raw_dir)
        _build_processed_from_physiomio(
            raw_root=raw_root,
            paths=paths,
            cfg=cfg,
            preprocess_cfg=preprocess_cfg,
        )

    train_ds = ProcessedTensorDataset(paths["train"])
    val_ds = ProcessedTensorDataset(paths["val"])
    test_ds = ProcessedTensorDataset(paths["test"])

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=cfg.drop_last,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader