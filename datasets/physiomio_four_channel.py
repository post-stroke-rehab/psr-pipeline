from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from data_processing.channel_selection import (
    PHYSIOMIO_FS_HZ,
    get_channel_policy,
    physiomio_window_spec,
    policy_metadata,
    select_active_channels,
)
from data_processing.mapping import gesture_to_5bit
from data_processing.preprocess import preprocess_emg
from data_processing.preprocess_config import PreprocessConfig


@dataclass(frozen=True)
class FourChannelPhysioMioConfig:
    raw_root: str = "datasets/raw/physiomio"
    processed_root: str = "datasets/processed/physiomio_4ch"
    seed: int = 42
    fs: float = PHYSIOMIO_FS_HZ
    arm_split: str = "impaired"
    impaired_only: bool = True
    min_segment_samples: int = 200
    skip_rest: bool = False
    max_patients: Optional[int] = None
    train_frac: float = 0.7
    val_frac: float = 0.1


@dataclass(frozen=True)
class FourChannelSample:
    left: Any
    right: Any
    y: Any
    patient_id: str
    source_file: str
    arm: str
    movement_type: str


def _canonical_json(obj: Mapping[str, object]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _torch():
    import torch

    return torch


def make_patient_splits(
    patient_ids: Sequence[str],
    *,
    seed: int = 42,
    train_frac: float = 0.7,
    val_frac: float = 0.1,
) -> Dict[str, List[str]]:
    patients = sorted(set(patient_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)

    n_patients = len(patients)
    n_train = int(round(n_patients * train_frac))
    n_val = int(round(n_patients * val_frac))

    return {
        "train": sorted(patients[:n_train]),
        "val": sorted(patients[n_train : n_train + n_val]),
        "test": sorted(patients[n_train + n_val :]),
    }


def assert_disjoint_patient_splits(split_patients: Mapping[str, Sequence[str]]) -> None:
    train = set(split_patients.get("train", []))
    val = set(split_patients.get("val", []))
    test = set(split_patients.get("test", []))
    overlaps = {
        "train_val": sorted(train & val),
        "train_test": sorted(train & test),
        "val_test": sorted(val & test),
    }
    leaking = {k: v for k, v in overlaps.items() if v}
    if leaking:
        raise ValueError(f"Patient split leakage detected: {leaking}")


def _split_indices(samples: Sequence[FourChannelSample], split_patients: Mapping[str, Sequence[str]]) -> Dict[str, List[int]]:
    split_sets = {name: set(patients) for name, patients in split_patients.items()}
    return {
        name: [i for i, sample in enumerate(samples) if sample.patient_id in patient_set]
        for name, patient_set in split_sets.items()
    }


def _pad_stack(tensors: Sequence[Any], target_windows: int) -> Any:
    torch = _torch()
    fixed: List[Any] = []
    for x in tensors:
        c, w, f = x.shape
        if w < target_windows:
            pad = torch.zeros((c, target_windows - w, f), dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)
        elif w > target_windows:
            x = x[:, :target_windows, :]
        fixed.append(x)
    return torch.stack(fixed, dim=0)


def _payload_for_view(
    samples: Sequence[FourChannelSample],
    indices: Sequence[int],
    *,
    view: str,
    target_windows: int,
    cfg: FourChannelPhysioMioConfig,
    preprocess_cfg: PreprocessConfig,
    split_name: str,
    split_patients: Mapping[str, Sequence[str]],
) -> Dict[str, object]:
    if view not in {"left", "right", "dual"}:
        raise ValueError(f"Unsupported view {view!r}")

    selected_samples = [samples[i] for i in indices]
    torch = _torch()
    if view == "dual":
        tensors = [s.left for s in selected_samples] + [s.right for s in selected_samples]
        labels = [s.y for s in selected_samples] + [s.y for s in selected_samples]
        channel_views = ["left"] * len(selected_samples) + ["right"] * len(selected_samples)
        meta_samples = selected_samples + selected_samples
    else:
        tensors = [getattr(s, view) for s in selected_samples]
        labels = [s.y for s in selected_samples]
        channel_views = [view] * len(selected_samples)
        meta_samples = selected_samples

    x = _pad_stack(tensors, target_windows) if tensors else torch.empty((0, 4, target_windows, 12))
    y = torch.stack(labels, dim=0) if labels else torch.empty((0, 5))

    meta = {
        "source": "physiomio",
        "split": split_name,
        "channel_policy": view,
        "channel_views": channel_views,
        "patient_ids": [s.patient_id for s in meta_samples],
        "patients": sorted(set(split_patients[split_name])),
        "source_files": [s.source_file for s in meta_samples],
        "movement_types": [s.movement_type for s in meta_samples],
        "arms": [s.arm for s in meta_samples],
        "fs": float(cfg.fs),
        "preprocess": asdict(preprocess_cfg),
        "channel_metadata": policy_metadata(),
        "output_label_order": ["thumb", "index", "middle", "ring", "little"],
    }
    return {"X": x.float(), "y": y.float(), "meta": meta}


def _manifest_for_view(
    *,
    view: str,
    view_dir: Path,
    cfg: FourChannelPhysioMioConfig,
    preprocess_cfg: PreprocessConfig,
    split_patients: Mapping[str, Sequence[str]],
    split_payloads: Mapping[str, Mapping[str, object]],
    split_file_hashes: Mapping[str, str],
) -> Dict[str, object]:
    assert_disjoint_patient_splits(split_patients)
    window_samples, stride_samples = physiomio_window_spec(
        cfg.fs, preprocess_cfg.window_size, preprocess_cfg.overlap
    )

    splits: Dict[str, object] = {}
    for split_name, payload in split_payloads.items():
        meta = payload["meta"]
        patients = sorted(set(meta["patient_ids"]))
        source_files = sorted(set(meta["source_files"]))
        splits[split_name] = {
            "patient_count": len(patients),
            "source_file_count": len(source_files),
            "sample_count": int(payload["X"].shape[0]),
            "tensor_shape": list(payload["X"].shape),
            "label_shape": list(payload["y"].shape),
            "split_file": str(view_dir / f"{split_name}.pt"),
            "split_file_sha256": split_file_hashes[split_name],
            "patient_set_sha256": _sha256_text(_canonical_json({"patients": patients})),
            "source_files_sha256": _sha256_text(_canonical_json({"source_files": source_files})),
        }

    manifest = {
        "version": 1,
        "source": "physiomio",
        "view": view,
        "fs": float(cfg.fs),
        "window_samples": window_samples,
        "stride_samples": stride_samples,
        "preprocessing": asdict(preprocess_cfg),
        "seed": int(cfg.seed),
        "arm_split": cfg.arm_split,
        "skip_rest": bool(cfg.skip_rest),
        "min_segment_samples": int(cfg.min_segment_samples),
        "max_patients": cfg.max_patients,
        "channel_policy": policy_metadata(),
        "output_label_order": ["thumb", "index", "middle", "ring", "little"],
        "split_patients": {k: list(v) for k, v in split_patients.items()},
        "split_patients_sha256": _sha256_text(_canonical_json({"split_patients": split_patients})),
        "splits": splits,
    }
    manifest["manifest_sha256"] = _sha256_text(_canonical_json(manifest))
    return manifest


def _selected_parquets(cfg: FourChannelPhysioMioConfig) -> List[Path]:
    from datasets.loaders import _find_physiomio_parquets, _patient_id_from_path, _resolve_arm_split

    class _LoaderLike:
        arm_split = cfg.arm_split
        impaired_only = cfg.impaired_only

    arm_split = _resolve_arm_split(_LoaderLike())
    parquets = _find_physiomio_parquets(Path(cfg.raw_root), arm_split)

    by_patient: Dict[str, List[Path]] = {}
    for pq in parquets:
        by_patient.setdefault(_patient_id_from_path(pq), []).append(pq)

    patients = sorted(by_patient)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(patients)
    if cfg.max_patients is not None:
        patients = patients[: cfg.max_patients]

    selected: List[Path] = []
    for pid in patients:
        selected.extend(by_patient[pid])
    return sorted(selected)


def _read_four_channel_samples(
    cfg: FourChannelPhysioMioConfig,
    preprocess_cfg: PreprocessConfig,
) -> List[FourChannelSample]:
    from datasets.loaders import (
        _arm_from_path,
        _get_channel_cols,
        _iter_contiguous_segments,
        _patient_id_from_path,
    )

    torch = _torch()
    parquets = _selected_parquets(cfg)
    if not parquets:
        raise FileNotFoundError(f"No PhysioMio parquet files found under {cfg.raw_root}")

    samples: List[FourChannelSample] = []
    unknown_labels: Dict[str, int] = {}
    for pq in parquets:
        df = pd.read_parquet(pq)
        if "movement_type" not in df.columns:
            raise ValueError(f"{pq} missing movement_type column")

        ch_cols = _get_channel_cols(df)
        mv = df["movement_type"].astype(str).to_numpy()
        for start, end, label in _iter_contiguous_segments(mv):
            if (end - start) < cfg.min_segment_samples:
                continue
            if cfg.skip_rest and label.lower() == "rest":
                continue

            try:
                y5 = np.asarray(gesture_to_5bit(label), dtype=np.float32).reshape(-1)
            except Exception:
                unknown_labels[label] = unknown_labels.get(label, 0) + 1
                continue
            if y5.shape[0] != 5:
                raise ValueError(f"Expected five output labels for {label}, got {y5.shape[0]}")

            emg = df.iloc[start:end][ch_cols].to_numpy(dtype=np.float32)
            left = preprocess_emg(select_active_channels(emg, "left"), fs=cfg.fs, config=preprocess_cfg)
            right = preprocess_emg(select_active_channels(emg, "right"), fs=cfg.fs, config=preprocess_cfg)

            samples.append(
                FourChannelSample(
                    left=torch.from_numpy(np.asarray(left, dtype=np.float32)),
                    right=torch.from_numpy(np.asarray(right, dtype=np.float32)),
                    y=torch.from_numpy(y5),
                    patient_id=_patient_id_from_path(pq),
                    source_file=str(pq),
                    arm=_arm_from_path(pq),
                    movement_type=label,
                )
            )

    if unknown_labels:
        top = sorted(unknown_labels.items(), key=lambda item: item[1], reverse=True)[:10]
        raise ValueError(
            "movement_type labels not covered by mapping.py: "
            + ", ".join(f"{label}({count})" for label, count in top)
        )
    if not samples:
        raise ValueError("No usable PhysioMio four-channel samples were produced")
    return samples


def build_physiomio_four_channel_views(
    cfg: FourChannelPhysioMioConfig,
    preprocess_cfg: Optional[PreprocessConfig] = None,
) -> Dict[str, Dict[str, object]]:
    torch = _torch()
    preprocess_cfg = preprocess_cfg or PreprocessConfig()
    if float(cfg.fs) != PHYSIOMIO_FS_HZ:
        raise ValueError(f"PhysioMio four-channel preprocessing expects {PHYSIOMIO_FS_HZ:g} Hz, got {cfg.fs}")

    for name in ("left", "right"):
        get_channel_policy(name)

    samples = _read_four_channel_samples(cfg, preprocess_cfg)
    split_patients = make_patient_splits(
        [s.patient_id for s in samples],
        seed=cfg.seed,
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
    )
    assert_disjoint_patient_splits(split_patients)
    split_indices = _split_indices(samples, split_patients)
    target_windows = max(max(int(s.left.shape[1]), int(s.right.shape[1])) for s in samples)

    manifests: Dict[str, Dict[str, object]] = {}
    root = Path(cfg.processed_root)
    for view in ("left", "right", "dual"):
        view_dir = root / view
        view_dir.mkdir(parents=True, exist_ok=True)

        split_payloads: Dict[str, Dict[str, object]] = {}
        split_hashes: Dict[str, str] = {}
        for split_name in ("train", "val", "test"):
            payload = _payload_for_view(
                samples,
                split_indices[split_name],
                view=view,
                target_windows=target_windows,
                cfg=cfg,
                preprocess_cfg=preprocess_cfg,
                split_name=split_name,
                split_patients=split_patients,
            )
            split_payloads[split_name] = payload
            split_path = view_dir / f"{split_name}.pt"
            torch.save(payload, split_path)
            split_hashes[split_name] = _sha256_file(split_path)

        manifest = _manifest_for_view(
            view=view,
            view_dir=view_dir,
            cfg=cfg,
            preprocess_cfg=preprocess_cfg,
            split_patients=split_patients,
            split_payloads=split_payloads,
            split_file_hashes=split_hashes,
        )
        manifest_path = view_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        manifests[view] = manifest

    return manifests
