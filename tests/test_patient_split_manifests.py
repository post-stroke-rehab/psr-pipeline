import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.physiomio_four_channel import (
    assert_disjoint_patient_splits,
    make_patient_splits,
)


def test_patient_split_manifest_sets_are_disjoint():
    patient_ids = [f"patient{i:02d}" for i in range(12)]
    split_patients = make_patient_splits(patient_ids, seed=42)

    assert sorted(split_patients) == ["test", "train", "val"]
    assert_disjoint_patient_splits(split_patients)

    train = set(split_patients["train"])
    val = set(split_patients["val"])
    test = set(split_patients["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert sorted(train | val | test) == sorted(patient_ids)


def test_patient_split_manifest_rejects_leakage():
    with pytest.raises(ValueError, match="Patient split leakage"):
        assert_disjoint_patient_splits(
            {
                "train": ["patient01", "patient02"],
                "val": ["patient02"],
                "test": ["patient03"],
            }
        )
