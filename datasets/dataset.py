# datasets/dataset.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class RawSample:
    """
    One raw sample entry.

    Supported raw formats:
      1) .npz containing keys:
           - "emg": (n_samples, n_channels) float-like
           - "y":   (5,) or (out_dim,) 0/1 or probabilities
           - "fs":  scalar sample rate (optional but recommended)
           - "meta": optional dict-like (ignored)

      2) .pt containing a dict with keys:
           - "emg", "y", "fs" (same meaning as above)

    If your raw data is NOT in one of these formats, you can still use this pipeline by
    creating a datasets/raw/manifest.json (see loaders.py for expected schema).
    """
    path: Path
    fs: Optional[float] = None


def _load_npz_sample(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[float], Dict[str, Any]]:
    data = np.load(path, allow_pickle=True)
    if "emg" not in data or "y" not in data:
        raise ValueError(f"{path} must contain keys 'emg' and 'y'.")

    emg = np.asarray(data["emg"])
    y = np.asarray(data["y"])
    fs = None
    if "fs" in data:
        fs = float(np.asarray(data["fs"]).item())

    meta: Dict[str, Any] = {}
    if "meta" in data:
        try:
            meta_obj = data["meta"].item() if hasattr(data["meta"], "item") else data["meta"]
            if isinstance(meta_obj, dict):
                meta = meta_obj
        except Exception:
            meta = {}

    return emg, y, fs, meta


def _load_pt_sample(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[float], Dict[str, Any]]:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"{path} must be a dict when using .pt raw samples.")
    if "emg" not in obj or "y" not in obj:
        raise ValueError(f"{path} must contain keys 'emg' and 'y'.")

    emg = obj["emg"]
    y = obj["y"]
    fs = obj.get("fs", None)
    meta = obj.get("meta", {})

    emg = emg.cpu().numpy() if isinstance(emg, torch.Tensor) else np.asarray(emg)
    y = y.cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)
    fs = float(fs) if fs is not None else None
    meta = meta if isinstance(meta, dict) else {}

    return emg, y, fs, meta


def load_raw_sample(path: Path) -> Tuple[np.ndarray, np.ndarray, Optional[float], Dict[str, Any]]:
    """
    Returns:
      emg: (n_samples, n_channels)
      y:   (out_dim,)
      fs:  float or None
      meta: dict
    """
    if path.suffix.lower() == ".npz":
        return _load_npz_sample(path)
    if path.suffix.lower() == ".pt":
        return _load_pt_sample(path)
    raise ValueError(f"Unsupported raw sample format: {path.suffix} (expected .npz or .pt)")


class ProcessedTensorDataset(Dataset):
    """
    Loads processed split files saved as:
      torch.save({"X": X, "y": y, "meta": meta}, split_path)

    Where:
      X: (N, C, W, F) float32
      y: (N, out_dim) float32
    """
    def __init__(self, split_path: str | Path):
        split_path = Path(split_path)
        payload = torch.load(split_path, map_location="cpu")
        if not isinstance(payload, dict) or "X" not in payload or "y" not in payload:
            raise ValueError(f"Processed split file invalid: {split_path}")

        self.X = payload["X"].float()
        self.y = payload["y"].float()
        self.meta = payload.get("meta", {})

        if self.X.ndim != 4:
            raise ValueError(f"Expected X with shape (N,C,W,F), got {tuple(self.X.shape)}")
        if self.y.ndim != 2:
            raise ValueError(f"Expected y with shape (N,out_dim), got {tuple(self.y.shape)}")
        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError("X and y must have same N dimension.")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]