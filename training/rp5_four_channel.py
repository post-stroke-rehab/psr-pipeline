from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from adapters.feature_to_sequence import feature_tensor_to_sequences
from evaluation.metrics import compute_multilabel_metrics
from models.CNN.students import CNN_Micro
from training.train import set_seed


MUSCLE_ORDER = ["ECRB", "ECRL", "FDS", "FDP"]
FULL_CHANNEL_ORDER = "PhysioMio 64-channel HD-sEMG grid"
FINGER_ORDER = ["thumb", "index", "middle", "ring", "little"]
FEATURES_PER_CHANNEL = 12
FULL_CHANNELS = 64
PHYSIOMIO_FS = 2048.0
WINDOW_SECONDS = 0.2
WINDOW_OVERLAP = 0.5
LEFT_CHANNELS_ZERO_BASED = [0, 2, 8, 13]
RIGHT_CHANNELS_ZERO_BASED = [14, 15, 8, 0]
CHANNEL_POLICIES = {
    "left": LEFT_CHANNELS_ZERO_BASED,
    "right": RIGHT_CHANNELS_ZERO_BASED,
}


def window_samples(fs: float = PHYSIOMIO_FS, window_seconds: float = WINDOW_SECONDS) -> int:
    return int(round(float(fs) * float(window_seconds)))


def stride_samples(
    fs: float = PHYSIOMIO_FS,
    window_seconds: float = WINDOW_SECONDS,
    overlap: float = WINDOW_OVERLAP,
) -> int:
    return int(round(window_samples(fs, window_seconds) * (1.0 - float(overlap))))


def feature_indices_for_channels(
    channel_indices: Sequence[int],
    *,
    features_per_channel: int = FEATURES_PER_CHANNEL,
) -> List[int]:
    indices: List[int] = []
    for channel in channel_indices:
        start = int(channel) * features_per_channel
        indices.extend(range(start, start + features_per_channel))
    return indices


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


class CNNMicroSequence(nn.Module):
    """CNN-Micro wrapper accepting adapted feature sequences as (batch, windows, features)."""

    def __init__(self, in_features: int = 48, out_dim: int = 5, dropout: float = 0.2):
        super().__init__()
        self.in_features = int(in_features)
        self.out_dim = int(out_dim)
        self.backbone = CNN_Micro(
            in_channels=self.in_features,
            num_classes=self.out_dim,
            dropout=float(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"CNNMicroSequence expects (N,W,F), got {tuple(x.shape)}")
        if x.shape[1] < 2:
            x = F.pad(x, (0, 0, 0, 2 - int(x.shape[1])))
        return self.backbone(x.permute(0, 2, 1).contiguous())


class FeatureContextDataset(Dataset):
    def __init__(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        context_windows: int,
        context_stride: int = 1,
    ):
        if x.ndim != 4:
            raise ValueError(f"Expected X as (N,C,W,F), got {tuple(x.shape)}")
        if y.ndim != 2:
            raise ValueError(f"Expected y as (N,5), got {tuple(y.shape)}")
        if x.shape[0] != y.shape[0]:
            raise ValueError("X and y sample counts must match.")
        self.x = x.float()
        self.y = y.float()
        self.context_windows = max(1, int(context_windows))
        self.context_stride = max(1, int(context_stride))
        self.index: List[Tuple[int, int]] = []
        windows = int(x.shape[2])
        if windows <= self.context_windows:
            self.index = [(i, 0) for i in range(int(x.shape[0]))]
        else:
            for sample_idx in range(int(x.shape[0])):
                starts = range(0, windows - self.context_windows + 1, self.context_stride)
                self.index.extend((sample_idx, int(start)) for start in starts)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_idx, start = self.index[idx]
        end = start + self.context_windows
        x = self.x[sample_idx, :, start:end, :]
        if x.shape[1] < self.context_windows:
            pad = torch.zeros(
                (x.shape[0], self.context_windows - x.shape[1], x.shape[2]),
                dtype=x.dtype,
            )
            x = torch.cat([x, pad], dim=1)
        return feature_tensor_to_sequences(x).squeeze(0), self.y[sample_idx]


class PairedFeatureContextDataset(Dataset):
    def __init__(
        self,
        x_student: torch.Tensor,
        x_teacher: torch.Tensor,
        y: torch.Tensor,
        *,
        context_windows: int,
        context_stride: int = 1,
    ):
        if x_student.shape[0] != x_teacher.shape[0] or x_student.shape[0] != y.shape[0]:
            raise ValueError("Student, teacher, and labels must have matching sample counts.")
        self.student = FeatureContextDataset(
            x_student,
            y,
            context_windows=context_windows,
            context_stride=context_stride,
        )
        self.teacher = FeatureContextDataset(
            x_teacher,
            y,
            context_windows=context_windows,
            context_stride=context_stride,
        )
        if len(self.student) != len(self.teacher):
            raise ValueError("Student and teacher context datasets must align.")

    def __len__(self) -> int:
        return len(self.student)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xs, y = self.student[idx]
        xt, yt = self.teacher[idx]
        if not torch.equal(y, yt):
            raise ValueError("Paired distillation labels diverged.")
        return xs, xt, y


@dataclass
class FourChannelRunConfig:
    run_id: str
    mode: str = "direct"
    processed_dir: str = "datasets/processed/physiomio_rp5_4ch"
    view: str = "left"
    full_processed_dir: str = "datasets/processed/physiomio"
    output_root: str = "experiments/rp5_4ch/runs"
    seed: int = 42
    context_windows: int = 1
    context_stride: int = 1
    batch_size: int = 128
    num_workers: int = 0
    epochs: int = 60
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.2
    patience: int = 12
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    threshold_metric: str = "f1_macro"
    out_dim: int = 5
    transfer_checkpoint: str = ""
    teacher_checkpoint: str = "results/distill_micro_from_cnn_a0.3_t2.0/checkpoint_best.pt"
    distill_alpha: float = 0.5
    distill_temperature: float = 2.0
    synthetic_smoke: bool = False
    max_train_batches: Optional[int] = None
    max_eval_batches: Optional[int] = None
    notes: Dict[str, Any] = field(default_factory=dict)


def _split_path(processed_dir: str | Path, view: str, split: str) -> Path:
    root = Path(processed_dir)
    candidates = [
        root / view / f"{split}.pt",
        root / f"{view}_{split}.pt",
        root / f"{split}_{view}.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {split} split for view={view!r}. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def load_split(processed_dir: str | Path, view: str, split: str) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    path = _split_path(processed_dir, view, split)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "X" not in payload or "y" not in payload:
        raise ValueError(f"Invalid processed split payload: {path}")
    return payload["X"].float(), payload["y"].float(), payload.get("meta", {})


def load_full_split(processed_dir: str | Path, split: str) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    path = Path(processed_dir) / f"{split}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "X" not in payload or "y" not in payload:
        raise ValueError(f"Invalid full-channel split payload: {path}")
    return payload["X"].float(), payload["y"].float(), payload.get("meta", {})


def synthetic_splits(seed: int = 42) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
    rng = torch.Generator().manual_seed(seed)
    splits = {"train": 24, "val": 8, "test": 8}
    out: Dict[str, Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]] = {}
    for split, n in splits.items():
        x = torch.randn((n, 4, 12, FEATURES_PER_CHANNEL), generator=rng)
        y = (torch.rand((n, 5), generator=rng) > 0.65).float()
        meta = {
            "source": "synthetic_smoke",
            "view": "synthetic",
            "patients_by_split": {split: [f"synthetic_{split}"]},
        }
        out[split] = (x, y, meta)
    return out


def make_loaders(
    splits: Dict[str, Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]],
    cfg: FourChannelRunConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    loaders = []
    for split in ("train", "val", "test"):
        x, y, _ = splits[split]
        ds = FeatureContextDataset(
            x,
            y,
            context_windows=cfg.context_windows,
            context_stride=cfg.context_stride,
        )
        loaders.append(
            DataLoader(
                ds,
                batch_size=cfg.batch_size,
                shuffle=(split == "train"),
                num_workers=cfg.num_workers,
                drop_last=False,
            )
        )
    return tuple(loaders)  # type: ignore[return-value]


def make_paired_loaders(
    reduced_splits: Dict[str, Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]],
    full_splits: Dict[str, Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]],
    cfg: FourChannelRunConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    loaders = []
    for split in ("train", "val", "test"):
        xs, y, _ = reduced_splits[split]
        xt, yt, _ = full_splits[split]
        if not torch.equal(y, yt):
            raise ValueError(f"Labels diverged between reduced and full {split} splits.")
        ds = PairedFeatureContextDataset(
            xs,
            xt,
            y,
            context_windows=cfg.context_windows,
            context_stride=cfg.context_stride,
        )
        loaders.append(
            DataLoader(
                ds,
                batch_size=cfg.batch_size,
                shuffle=(split == "train"),
                num_workers=cfg.num_workers,
                drop_last=False,
            )
        )
    return tuple(loaders)  # type: ignore[return-value]


def pos_weight_from_loader(loader: DataLoader, out_dim: int, device: torch.device) -> torch.Tensor:
    pos = torch.zeros(out_dim, device=device)
    total = 0
    for batch in loader:
        y = batch[-1].to(device).float()
        pos += y.sum(dim=0)
        total += int(y.shape[0])
    neg = total - pos
    return (neg / pos.clamp_min(1.0)).clamp(min=1.0, max=100.0)


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    paired: bool = False,
    max_batches: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    probs: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        if paired:
            x = batch[0].to(device)
            y = batch[2].to(device).float()
        else:
            x = batch[0].to(device)
            y = batch[1].to(device).float()
        probs.append(torch.sigmoid(model(x)).cpu())
        targets.append(y.cpu())
    return torch.cat(probs, dim=0), torch.cat(targets, dim=0)


def tune_thresholds(
    probs: torch.Tensor,
    targets: torch.Tensor,
    *,
    out_dim: int,
    metric: str = "f1_macro",
) -> List[float]:
    thresholds: List[float] = []
    grid = np.arange(0.05, 0.96, 0.05)
    for finger_idx in range(out_dim):
        best_t = 0.5
        best_score = -1.0
        for t in grid:
            single = compute_multilabel_metrics(
                probs[:, finger_idx : finger_idx + 1],
                targets[:, finger_idx : finger_idx + 1],
                threshold=float(t),
                num_classes=1,
            )
            score = float(single["f1_macro"] if metric == "f1_macro" else single["finger_accuracy"])
            if score > best_score:
                best_score = score
                best_t = float(t)
        thresholds.append(best_t)
    return thresholds


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    thresholds: Sequence[float] | float,
    out_dim: int,
    paired: bool = False,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    probs, targets = predict(model, loader, device, paired=paired, max_batches=max_batches)
    return compute_multilabel_metrics(probs, targets, threshold=thresholds, num_classes=out_dim)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    pos_weight: torch.Tensor,
    max_batches: Optional[int] = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x = x.to(device)
        y = y.to(device).float()
        optimizer.zero_grad()
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * int(y.shape[0])
        total_n += int(y.shape[0])
    return total_loss / max(1, total_n)


def distill_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    pos_weight: torch.Tensor,
    alpha: float,
    temperature: float,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    student.train()
    teacher.eval()
    total = {"loss": 0.0, "hard_loss": 0.0, "kd_loss": 0.0, "n": 0.0}
    for batch_idx, (x_student, x_teacher, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        x_student = x_student.to(device)
        x_teacher = x_teacher.to(device)
        y = y.to(device).float()
        optimizer.zero_grad()
        student_logits = student(x_student)
        with torch.no_grad():
            teacher_logits = teacher(x_teacher)
        hard = F.binary_cross_entropy_with_logits(student_logits, y, pos_weight=pos_weight)
        soft = F.binary_cross_entropy_with_logits(
            student_logits / temperature,
            torch.sigmoid(teacher_logits / temperature),
        ) * (temperature * temperature)
        loss = alpha * hard + (1.0 - alpha) * soft
        loss.backward()
        optimizer.step()
        n = float(y.shape[0])
        total["loss"] += float(loss.item()) * n
        total["hard_loss"] += float(hard.item()) * n
        total["kd_loss"] += float(soft.item()) * n
        total["n"] += n
    n_total = max(1.0, total.pop("n"))
    return {k: v / n_total for k, v in total.items()}


def normalize_state_dict(payload: Any) -> Dict[str, torch.Tensor]:
    state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a valid state dict.")
    return state


def load_cnn_micro_transfer(
    model: CNNMicroSequence,
    checkpoint_path: str | Path,
    *,
    source_channel_indices: Sequence[int],
) -> Dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    raw_state = normalize_state_dict(payload)
    state: Dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        next_key = key if key.startswith("backbone.") else f"backbone.{key}"
        state[next_key] = value

    proj_key = "backbone.proj.weight"
    if proj_key not in state:
        raise ValueError(
            "Transfer checkpoint is not a CNN_Micro-compatible checkpoint: "
            "expected a proj.weight first layer."
        )
    selected = feature_indices_for_channels(source_channel_indices)
    if state[proj_key].shape[1] < max(selected) + 1:
        raise ValueError(
            f"Checkpoint first layer has {state[proj_key].shape[1]} inputs; "
            f"need index {max(selected)} for selected channels."
        )
    if state[proj_key].shape[1] != model.in_features:
        state[proj_key] = state[proj_key][:, selected, :].contiguous()

    for norm_key in ("backbone.in_norm.weight", "backbone.in_norm.bias"):
        if norm_key in state and state[norm_key].ndim == 1 and state[norm_key].numel() != model.in_features:
            state[norm_key] = state[norm_key][selected].contiguous()

    missing, unexpected = model.load_state_dict(state, strict=False)
    return {
        "checkpoint": str(checkpoint_path),
        "source_channel_indices_zero_based": list(map(int, source_channel_indices)),
        "selected_feature_indices": selected,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def build_teacher_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    in_features: int,
    out_dim: int,
    dropout: float,
) -> nn.Module:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = normalize_state_dict(payload)
    keys = set(state.keys())
    if "conv1.weight" in keys:
        from training.train_distill import AdaptiveCNNStudent

        teacher = AdaptiveCNNStudent(
            in_features=in_features,
            out_dim=out_dim,
            width=int(state["conv1.weight"].shape[0]),
            fc_hidden=int(state["fc1.weight"].shape[0]),
            dropout=dropout,
        )
        teacher.load_state_dict(state)
        return teacher

    teacher = CNNMicroSequence(in_features=in_features, out_dim=out_dim, dropout=dropout)
    normalized = {
        (key if key.startswith("backbone.") else f"backbone.{key}"): value
        for key, value in state.items()
    }
    teacher.load_state_dict(normalized, strict=False)
    return teacher


def _jsonable_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value

    return convert(metrics)


def sanitize_split_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Keep run manifests useful without committing patient IDs or raw file paths."""

    sensitive = ("patient", "path", "file")

    def is_sensitive_key(key: str) -> bool:
        lower = key.lower()
        if lower.endswith("_count") or lower.endswith("_counts") or lower.endswith("_sha256"):
            return False
        return any(part in lower for part in sensitive)

    def compact_list(key: str, values: Sequence[Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {f"{key}_count": len(values)}
        safe_scalars = all(isinstance(v, (str, int, float, bool, type(None))) for v in values)
        if safe_scalars and not is_sensitive_key(key):
            unique = sorted({str(v) for v in values})
            out[f"{key}_unique"] = unique if len(unique) <= 20 else len(unique)
        elif safe_scalars:
            out[f"{key}_unique_count"] = len({str(v) for v in values})
        return out

    def scrub(obj: Any, key: str = "value") -> Any:
        if isinstance(obj, dict):
            cleaned: Dict[str, Any] = {}
            for child_key, child_value in obj.items():
                child_key_str = str(child_key)
                lower_key = child_key_str.lower()
                if lower_key.endswith("_present") and any(part in lower_key for part in sensitive):
                    continue
                if is_sensitive_key(child_key_str):
                    if isinstance(child_value, (list, tuple)):
                        cleaned.update(compact_list(child_key_str, child_value))
                    else:
                        cleaned[f"{child_key_str}_present"] = child_value is not None
                else:
                    cleaned[child_key_str] = scrub(child_value, child_key_str)
            return cleaned
        if isinstance(obj, (list, tuple)):
            if len(obj) <= 20:
                return [scrub(v, key) for v in obj]
            return compact_list(key, obj)
        return obj

    cleaned = scrub(meta)
    return cleaned if isinstance(cleaned, dict) else {"value": cleaned}


def sanitize_run_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Make saved run configs portable and safe for committed artifacts."""

    def clean_path(value: str) -> str:
        normalized = value.replace("\\", "/")
        for marker in (
            "datasets/processed/",
            "datasets/raw/",
            "experiments/",
            "results/",
        ):
            if marker in normalized:
                return normalized[normalized.index(marker) :]
        return normalized

    cleaned: Dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            cleaned[key] = sanitize_run_config(value)
        elif isinstance(value, str) and ("/" in value or "\\" in value):
            cleaned[key] = clean_path(value)
        else:
            cleaned[key] = value
    return cleaned


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def run_four_channel_experiment(cfg: FourChannelRunConfig) -> Dict[str, Any]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    run_dir = Path(cfg.output_root) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status = {"status": "running", "started_at": started}
    write_json(run_dir / "status.json", status)

    if cfg.mode == "source" and not cfg.synthetic_smoke:
        reduced_splits = {
            split: load_full_split(cfg.full_processed_dir, split)
            for split in ("train", "val", "test")
        }
    elif cfg.synthetic_smoke:
        reduced_splits = synthetic_splits(cfg.seed)
    else:
        reduced_splits = {
            split: load_split(cfg.processed_dir, cfg.view, split)
            for split in ("train", "val", "test")
        }

    train_loader, val_loader, test_loader = make_loaders(reduced_splits, cfg)
    x0, _ = next(iter(train_loader))
    input_features = int(x0.shape[-1])
    active_channel_count = input_features // FEATURES_PER_CHANNEL
    model = CNNMicroSequence(
        in_features=input_features,
        out_dim=cfg.out_dim,
        dropout=cfg.dropout,
    ).to(device)
    transfer_info = None
    if cfg.mode == "transfer":
        if not cfg.transfer_checkpoint:
            raise ValueError("--transfer-checkpoint is required for transfer mode.")
        if cfg.view == "dual":
            raise ValueError("Transfer mode requires a concrete left or right view for first-layer slicing.")
        transfer_info = load_cnn_micro_transfer(
            model,
            cfg.transfer_checkpoint,
            source_channel_indices=CHANNEL_POLICIES[cfg.view],
        )

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cfg.epochs))
    pos_weight = pos_weight_from_loader(train_loader, cfg.out_dim, device)
    best_score = -1.0
    best_epoch = 0
    best_path = run_dir / "checkpoint_best.pt"
    patience_ctr = 0
    history: List[Dict[str, Any]] = []

    distill_teacher = None
    paired_loaders = None
    if cfg.mode == "distill":
        if cfg.synthetic_smoke:
            full_splits = {
                split: (torch.randn(x.shape[0], FULL_CHANNELS, x.shape[2], x.shape[3]), y, meta)
                for split, (x, y, meta) in reduced_splits.items()
            }
        else:
            full_splits = {
                split: load_full_split(cfg.full_processed_dir, split)
                for split in ("train", "val", "test")
            }
            if cfg.view == "dual":
                full_splits = {
                    split: (
                        torch.cat([x, x], dim=0),
                        torch.cat([y, y], dim=0),
                        {**meta, "channel_policy": "dual_teacher_duplicate"},
                    )
                    for split, (x, y, meta) in full_splits.items()
                }
        paired_loaders = make_paired_loaders(reduced_splits, full_splits, cfg)
        xt0 = next(iter(paired_loaders[0]))[1]
        distill_teacher = build_teacher_from_checkpoint(
            cfg.teacher_checkpoint,
            in_features=int(xt0.shape[-1]),
            out_dim=cfg.out_dim,
            dropout=cfg.dropout,
        ).to(device)

    for epoch in range(1, cfg.epochs + 1):
        if cfg.mode == "distill":
            assert paired_loaders is not None and distill_teacher is not None
            train_stats = distill_one_epoch(
                model,
                distill_teacher,
                paired_loaders[0],
                optimizer,
                device,
                pos_weight=pos_weight,
                alpha=cfg.distill_alpha,
                temperature=cfg.distill_temperature,
                max_batches=cfg.max_train_batches,
            )
            val_metrics = evaluate_loader(
                model,
                paired_loaders[1],
                device,
                thresholds=0.5,
                out_dim=cfg.out_dim,
                paired=True,
                max_batches=cfg.max_eval_batches,
            )
            train_loss = train_stats["loss"]
        else:
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                pos_weight=pos_weight,
                max_batches=cfg.max_train_batches,
            )
            val_metrics = evaluate_loader(
                model,
                val_loader,
                device,
                thresholds=0.5,
                out_dim=cfg.out_dim,
                max_batches=cfg.max_eval_batches,
            )
        scheduler.step()
        val_score = float(val_metrics["f1_macro"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_metrics": _jsonable_metrics(val_metrics),
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            patience_ctr = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(cfg),
                    "model_config": {
                        "architecture": "CNN_Micro",
                        "input_shape": ["batch", "windows", input_features],
                        "output_shape": ["batch", 5],
                        "muscle_order": MUSCLE_ORDER if active_channel_count == 4 else FULL_CHANNEL_ORDER,
                        "finger_order": FINGER_ORDER,
                    },
                    "transfer_info": transfer_info,
                    "epoch": epoch,
                    "best_val_f1": best_score,
                    "timestamp": time.time(),
                },
                best_path,
            )
        else:
            patience_ctr += 1
        print(
            f"[{cfg.run_id}] epoch={epoch:03d} train={train_loss:.4f} "
            f"val_f1={val_score:.4f} best={best_score:.4f}"
        )
        if patience_ctr >= cfg.patience:
            break

    payload = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(payload["model_state"])
    val_probs, val_targets = predict(
        model,
        paired_loaders[1] if cfg.mode == "distill" and paired_loaders else val_loader,
        device,
        paired=cfg.mode == "distill",
        max_batches=cfg.max_eval_batches,
    )
    thresholds = tune_thresholds(
        val_probs,
        val_targets,
        out_dim=cfg.out_dim,
        metric=cfg.threshold_metric,
    )
    eval_test_loader = paired_loaders[2] if cfg.mode == "distill" and paired_loaders else test_loader
    test_metrics = evaluate_loader(
        model,
        eval_test_loader,
        device,
        thresholds=thresholds,
        out_dim=cfg.out_dim,
        paired=cfg.mode == "distill",
        max_batches=cfg.max_eval_batches,
    )
    val_metrics_tuned = compute_multilabel_metrics(
        val_probs,
        val_targets,
        threshold=thresholds,
        num_classes=cfg.out_dim,
    )

    checkpoint_hash = sha256_file(best_path)
    manifest = {
        "run_id": cfg.run_id,
        "git_commit": git_commit(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "duration_sec": time.time() - started,
        "fs_hz": PHYSIOMIO_FS,
        "window_samples": window_samples(),
        "stride_samples": stride_samples(),
        "muscle_order": MUSCLE_ORDER if active_channel_count == 4 else FULL_CHANNEL_ORDER,
        "finger_order": FINGER_ORDER,
        "active_channel_count": active_channel_count,
        "ground_included": False,
        "feature_shape_contract": {
            "raw_active_signals": ["samples", active_channel_count],
            "feature_tensor": ["batch", active_channel_count, "windows", FEATURES_PER_CHANNEL],
            "cnn_input": ["batch", "windows", input_features],
            "output_logits": ["batch", 5],
        },
        "sequence_padding": "CNN_Micro pads one-window contexts to two temporal steps before temporal pooling.",
        "checkpoint_sha256": checkpoint_hash,
        "split_meta": {
            split: sanitize_split_meta(meta)
            for split, (_, _, meta) in reduced_splits.items()
        },
    }

    write_json(run_dir / "config.json", sanitize_run_config(asdict(cfg)))
    write_json(run_dir / "history.json", history)
    write_json(run_dir / "thresholds.json", {"thresholds": thresholds, "source": "validation"})
    write_json(run_dir / "val_metrics.json", _jsonable_metrics(val_metrics_tuned))
    write_json(run_dir / "test_metrics.json", _jsonable_metrics(test_metrics))
    write_json(run_dir / "manifest.json", _jsonable_metrics(manifest))
    write_json(
        run_dir / "model_config.json",
        {
            "architecture": "CNN_Micro",
            "input_features": input_features,
            "output_logits": cfg.out_dim,
            "muscle_order": MUSCLE_ORDER if active_channel_count == 4 else FULL_CHANNEL_ORDER,
            "finger_order": FINGER_ORDER,
            "state_dict": "checkpoint_best.pt",
        },
    )
    write_json(
        run_dir / "status.json",
        {
            "status": "completed",
            "started_at": started,
            "ended_at": time.time(),
            "best_epoch": best_epoch,
            "best_val_f1": best_score,
            "checkpoint_sha256": checkpoint_hash,
        },
    )
    return {
        "run_id": cfg.run_id,
        "run_dir": str(run_dir),
        "best_val_f1": best_score,
        "test_metrics": _jsonable_metrics(test_metrics),
        "thresholds": thresholds,
        "checkpoint_sha256": checkpoint_hash,
    }


def summarize_runs(run_dirs: Iterable[str | Path], output_path: str | Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        metrics_path = run_dir / "test_metrics.json"
        config_path = run_dir / "config.json"
        if not metrics_path.exists() or not config_path.exists():
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        with open(config_path) as f:
            cfg = json.load(f)
        rows.append(
            {
                "run_id": cfg["run_id"],
                "mode": cfg["mode"],
                "view": cfg["view"],
                "seed": cfg["seed"],
                "context_windows": cfg["context_windows"],
                "accuracy": metrics["accuracy"],
                "finger_accuracy": metrics["finger_accuracy"],
                "f1_macro": metrics["f1_macro"],
                "auprc_macro": metrics["auprc_macro"],
                "auroc_macro": metrics["auroc_macro"],
            }
        )

    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = f"{row['mode']}|{row['view']}|ctx{row['context_windows']}"
        grouped.setdefault(key, {"runs": [], "summary": {}})["runs"].append(row)

    for key, group in grouped.items():
        for metric in ("accuracy", "finger_accuracy", "f1_macro", "auprc_macro", "auroc_macro"):
            values = np.asarray([r[metric] for r in group["runs"]], dtype=float)
            group["summary"][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "ci95": float(1.96 * values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0,
                "n": int(len(values)),
            }

    out = {"runs": rows, "groups": grouped}
    write_json(output_path, out)
    return out
