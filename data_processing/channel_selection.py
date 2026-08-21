from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


ACTIVE_MUSCLE_ORDER: Tuple[str, str, str, str] = ("ECRB", "ECRL", "FDS", "FDP")
FEATURE_ORDER: Tuple[str, ...] = (
    "RMS",
    "MAV",
    "IEMG",
    "WL",
    "VAR",
    "ZC",
    "SSC",
    "WAMP",
    "MNF",
    "MDF",
    "SEN",
    "TP",
)

LEFT_CHANNELS_ONE_BASED: Tuple[int, int, int, int] = (1, 3, 9, 14)
RIGHT_CHANNELS_ONE_BASED: Tuple[int, int, int, int] = (15, 16, 9, 1)
LEFT_CHANNELS_ZERO_BASED: Tuple[int, int, int, int] = tuple(i - 1 for i in LEFT_CHANNELS_ONE_BASED)
RIGHT_CHANNELS_ZERO_BASED: Tuple[int, int, int, int] = tuple(i - 1 for i in RIGHT_CHANNELS_ONE_BASED)

PHYSIOMIO_FS_HZ = 2048.0
PHYSIOMIO_WINDOW_SECONDS = 0.2
PHYSIOMIO_OVERLAP = 0.5


@dataclass(frozen=True)
class ChannelPolicy:
    name: str
    zero_based: Tuple[int, int, int, int]
    one_based: Tuple[int, int, int, int]
    muscle_order: Tuple[str, str, str, str] = ACTIVE_MUSCLE_ORDER


CHANNEL_POLICIES: Dict[str, ChannelPolicy] = {
    "left": ChannelPolicy("left", LEFT_CHANNELS_ZERO_BASED, LEFT_CHANNELS_ONE_BASED),
    "right": ChannelPolicy("right", RIGHT_CHANNELS_ZERO_BASED, RIGHT_CHANNELS_ONE_BASED),
}


def physiomio_window_spec(
    fs: float = PHYSIOMIO_FS_HZ,
    window_size: float = PHYSIOMIO_WINDOW_SECONDS,
    overlap: float = PHYSIOMIO_OVERLAP,
) -> Tuple[int, int]:
    window_samples = int(round(float(fs) * float(window_size)))
    stride_samples = int(round(window_samples * (1.0 - float(overlap))))
    return window_samples, stride_samples


def get_channel_policy(name: str) -> ChannelPolicy:
    key = str(name).lower()
    if key not in CHANNEL_POLICIES:
        raise ValueError(f"Unknown channel policy {name!r}; expected one of {sorted(CHANNEL_POLICIES)}")
    return CHANNEL_POLICIES[key]


def select_active_channels(emg: np.ndarray, policy: str | ChannelPolicy) -> np.ndarray:
    policy_obj = get_channel_policy(policy) if isinstance(policy, str) else policy
    emg = np.asarray(emg)
    if emg.ndim != 2:
        raise ValueError(f"Expected raw EMG with shape (samples, channels), got {emg.shape}")

    max_idx = max(policy_obj.zero_based)
    if emg.shape[1] <= max_idx:
        raise ValueError(
            f"Policy {policy_obj.name!r} requires channel index {max_idx}, "
            f"but input has only {emg.shape[1]} channels"
        )
    return emg[:, list(policy_obj.zero_based)]


def flattened_feature_order(
    muscles: Sequence[str] = ACTIVE_MUSCLE_ORDER,
    features: Sequence[str] = FEATURE_ORDER,
) -> List[str]:
    return [f"{muscle}:{feature}" for muscle in muscles for feature in features]


def policy_metadata() -> Dict[str, object]:
    return {
        "active_muscle_order": list(ACTIVE_MUSCLE_ORDER),
        "feature_order_per_channel": list(FEATURE_ORDER),
        "flattened_feature_order": flattened_feature_order(),
        "ground_or_reference_included": False,
        "left": {
            "one_based": list(LEFT_CHANNELS_ONE_BASED),
            "zero_based": list(LEFT_CHANNELS_ZERO_BASED),
        },
        "right": {
            "one_based": list(RIGHT_CHANNELS_ONE_BASED),
            "zero_based": list(RIGHT_CHANNELS_ZERO_BASED),
        },
    }
