"""Inference-friendly rolling preprocessing for Raspberry Pi deployment.

This module deliberately reuses the repository's existing ``preprocess_emg``
implementation.  Incoming raw samples are buffered and, every inference
stride, the currently available rolling context is preprocessed exactly by the
same band-pass -> wavelet -> window -> feature code used offline.

The output layout is the distilled CNN-Micro training layout:
    (batch=1, windows, channels * 12 features)

This compatibility-first implementation is intentionally separate from a
future fully causal/stateful DSP implementation.  Benchmark it on the Pi; if
re-running the static pipeline over the rolling context is too expensive, a
causal filter/denoiser can be introduced with explicit model-parity testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from data_processing.preprocess import preprocess_emg
from data_processing.preprocess_config import PreprocessConfig


FEATURES_PER_CHANNEL = 12


@dataclass(frozen=True)
class RealtimePreprocessConfig:
    sample_rate: float = 2000.0
    channels: int = 64
    window_size: float = 0.2
    overlap: float = 0.5
    context_windows: int = 39

    @property
    def window_samples(self) -> int:
        return int(round(self.window_size * self.sample_rate))

    @property
    def stride_samples(self) -> int:
        return int(round(self.window_samples * (1.0 - self.overlap)))

    @property
    def context_samples(self) -> int:
        return self.window_samples + (self.context_windows - 1) * self.stride_samples

    @property
    def model_features(self) -> int:
        return self.channels * FEATURES_PER_CHANNEL


class RealtimePreprocessor:
    """Convert arbitrarily chunked raw EMG into fixed-length CNN input tensors."""

    def __init__(self, cfg: RealtimePreprocessConfig) -> None:
        if cfg.channels <= 0:
            raise ValueError("channels must be positive")
        if cfg.context_windows <= 0:
            raise ValueError("context_windows must be positive")
        if cfg.window_samples <= 0 or cfg.stride_samples <= 0:
            raise ValueError("invalid window/overlap configuration")

        self.cfg = cfg
        self._raw = np.empty((0, cfg.channels), dtype=np.float32)
        self._buffer_start = 0
        self._total_samples = 0
        self._next_emit = cfg.window_samples

        self._offline_cfg = PreprocessConfig(
            window_size=cfg.window_size,
            overlap=cfg.overlap,
            padding=False,
        )

    def reset(self) -> None:
        self._raw = np.empty((0, self.cfg.channels), dtype=np.float32)
        self._buffer_start = 0
        self._total_samples = 0
        self._next_emit = self.cfg.window_samples

    def push(self, samples: np.ndarray) -> List[np.ndarray]:
        """Push ``(N,C)`` raw samples and return zero or more ``(1,W,C*12)`` inputs."""
        x = np.asarray(samples, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.cfg.channels:
            raise ValueError(
                f"Expected samples (N,{self.cfg.channels}); got {tuple(x.shape)}"
            )
        if x.shape[0] == 0:
            return []
        if not np.isfinite(x).all():
            raise ValueError("samples contain NaN/Inf")

        self._raw = np.concatenate([self._raw, x], axis=0)
        self._total_samples += int(x.shape[0])

        outputs: List[np.ndarray] = []
        while self._total_samples >= self._next_emit:
            outputs.append(self._build_input(self._next_emit))
            self._next_emit += self.cfg.stride_samples

        # Keep only enough history for the next rolling context, while retaining
        # samples needed by a not-yet-emitted boundary.
        keep_from_global = max(0, self._total_samples - self.cfg.context_samples)
        drop = keep_from_global - self._buffer_start
        if drop > 0:
            self._raw = self._raw[drop:]
            self._buffer_start = keep_from_global

        return outputs

    def _build_input(self, end_global: int) -> np.ndarray:
        start_global = max(0, end_global - self.cfg.context_samples)
        local_start = start_global - self._buffer_start
        local_end = end_global - self._buffer_start
        context = self._raw[local_start:local_end]

        features = preprocess_emg(
            context,
            fs=self.cfg.sample_rate,
            config=self._offline_cfg,
        )
        # Existing preprocessing returns (C, W, 12).
        if features.ndim != 3 or features.shape[0] != self.cfg.channels:
            raise RuntimeError(f"Unexpected preprocessing output {features.shape}")
        if features.shape[2] != FEATURES_PER_CHANNEL:
            raise RuntimeError(
                f"Expected {FEATURES_PER_CHANNEL} features/channel, got {features.shape[2]}"
            )

        # Training pads each sample to a common W.  For startup, preserve that
        # convention by right-padding feature history with zeros. Once full,
        # retain only the most recent context_windows.
        w = features.shape[1]
        if w < self.cfg.context_windows:
            pad = np.zeros(
                (self.cfg.channels, self.cfg.context_windows - w, FEATURES_PER_CHANNEL),
                dtype=np.float32,
            )
            features = np.concatenate([features.astype(np.float32), pad], axis=1)
        elif w > self.cfg.context_windows:
            features = features[:, -self.cfg.context_windows :, :].astype(np.float32)
        else:
            features = features.astype(np.float32)

        # (C,W,F) -> (1,W,C*F), matching feature_tensor_to_sequences().
        model_input = features.transpose(1, 0, 2).reshape(
            1, self.cfg.context_windows, self.cfg.model_features
        )
        return np.ascontiguousarray(model_input, dtype=np.float32)
