"""Dataset-backed raw sEMG replay source for inference development."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd


SampleCallback = Callable[[np.ndarray], None]


def load_physiomio_channels(path: str | Path, channels: Sequence[str]) -> np.ndarray:
    if not channels:
        raise ValueError("At least one channel is required")
    if len(set(channels)) != len(channels):
        raise ValueError("Duplicate channels are not allowed")

    df = pd.read_parquet(path)
    missing = [ch for ch in channels if ch not in df.columns]
    if missing:
        raise ValueError(f"Missing channels in recording: {missing}")

    data = df.loc[:, list(channels)].to_numpy(dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != len(channels):
        raise RuntimeError(f"Unexpected replay data shape {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("Replay source contains NaN/Inf")
    return data


def replay_array(
    samples: np.ndarray,
    *,
    sample_rate: float,
    callback: SampleCallback,
    chunk_ms: float = 10.0,
    realtime: bool = True,
    speed: float = 1.0,
) -> dict:
    """Replay ``(N,C)`` samples, preserving source order exactly."""
    x = np.asarray(samples, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"samples must be (N,C), got {x.shape}")
    if sample_rate <= 0 or chunk_ms <= 0 or speed <= 0:
        raise ValueError("sample_rate, chunk_ms and speed must be positive")

    chunk_samples = max(1, int(round(sample_rate * chunk_ms / 1000.0)))
    start = time.monotonic()
    emitted = 0
    chunks = 0

    while emitted < len(x):
        end = min(len(x), emitted + chunk_samples)
        callback(x[emitted:end])
        emitted = end
        chunks += 1

        if realtime and emitted < len(x):
            # Schedule against an absolute target to avoid accumulating
            # callback/processing overhead as replay drift.
            target = start + (emitted / sample_rate) / speed
            delay = target - time.monotonic()
            if delay > 0:
                time.sleep(delay)

    elapsed = time.monotonic() - start
    return {
        "samples": emitted,
        "channels": int(x.shape[1]),
        "chunks": chunks,
        "source_sample_rate_hz": float(sample_rate),
        "source_duration_s": float(len(x) / sample_rate),
        "elapsed_s": float(elapsed),
        "effective_sample_rate_hz": float(emitted / elapsed) if elapsed > 0 else float("inf"),
        "realtime": bool(realtime),
        "speed": float(speed),
    }
