import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_processing.channel_selection import (
    ACTIVE_MUSCLE_ORDER,
    LEFT_CHANNELS_ZERO_BASED,
    PHYSIOMIO_FS_HZ,
    RIGHT_CHANNELS_ZERO_BASED,
    flattened_feature_order,
    physiomio_window_spec,
    policy_metadata,
    select_active_channels,
)


def test_channel_selection_preserves_canonical_muscle_order():
    emg = np.tile(np.arange(64, dtype=np.float32), (8, 1))

    left = select_active_channels(emg, "left")
    right = select_active_channels(emg, "right")

    assert tuple(ACTIVE_MUSCLE_ORDER) == ("ECRB", "ECRL", "FDS", "FDP")
    assert left.shape == (8, 4)
    assert right.shape == (8, 4)
    assert left[0].tolist() == list(LEFT_CHANNELS_ZERO_BASED)
    assert right[0].tolist() == list(RIGHT_CHANNELS_ZERO_BASED)


def test_ground_reference_is_not_part_of_channel_policy_or_features():
    metadata = policy_metadata()
    feature_order = flattened_feature_order()

    assert metadata["ground_or_reference_included"] is False
    assert "ground" not in " ".join(feature_order).lower()
    assert "reference" not in " ".join(feature_order).lower()
    assert len(feature_order) == 48


def test_physiomio_2048_hz_window_stride_math():
    window_samples, stride_samples = physiomio_window_spec(PHYSIOMIO_FS_HZ, 0.2, 0.5)

    assert window_samples == 410
    assert stride_samples == 205
