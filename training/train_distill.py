from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from adapters.feature_to_sequence import feature_tensor_to_sequences
from datasets.loaders import LoaderConfig, make_dataloaders
from evaluation.metrics import compute_multilabel_metrics
from training.teacher_ensemble import TeacherEnsemble, TeacherSpec
from training.train import TrainConfig, compute_pos_weight_from_loader, eval_loop, set_seed


class AdaptiveCNNStudent(nn.Module):
    """
    Small CNN student that accepts adapted input as (N, W, F),
    then permutes to Conv1d format (N, F, W).

    Uses adaptive pooling so it works with seq_len=39 instead of assuming 200.
    """

    def __init__(
        self,
        in_features: int,
        out_dim: int = 5,
        width: int = 20,
        fc_hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(width, width, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(width)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(width, fc_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected adapted input (N,W,F), got {tuple(x.shape)}")

        x = x.permute(0, 2, 1).contiguous()
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


@dataclass
class DistillConfig:
    student: str = "micro"
    teachers: str = "cnn,lstm"
    alpha: float = 0.5
    temperature: float = 2.0

    out_dim: int = 5
    threshold: float = 0.5
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 10
    patience: int = 5
    lr_patience: int = 3
    lr_factor: float = 0.5
    min_lr: float = 1e-6
    batch_size: int = 128
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    processed_dir: str = "datasets/processed/physiomio"
    train_file: str = "train.pt"
    val_file: str = "val.pt"
    test_file: str = "test.pt"

    results_dir: str = "results"


def kd_bce_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    soft_targets = torch.sigmoid(teacher_logits / temperature)
    return F.binary_cross_entropy_with_logits(
        student_logits / temperature,
        soft_targets,
    ) * (temperature * temperature)


def build_student(student_name: str, sample_x: torch.Tensor, out_dim: int) -> nn.Module:
    in_features = int(sample_x.size(-1))

    if student_name == "nano":
        return AdaptiveCNNStudent(
            in_features=in_features,
            out_dim=out_dim,
            width=16,
            fc_hidden=64,
            dropout=0.2,
        )

    if student_name == "micro":
        return AdaptiveCNNStudent(
            in_features=in_features,
            out_dim=out_dim,
            width=20,
            fc_hidden=128,
            dropout=0.2,
        )

    raise ValueError(f"Unknown student '{student_name}'. Use 'micro' or 'nano'.")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_distill(cfg: DistillConfig) -> Dict:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    print(f"[Device] {device}")
    print(f"[Distill] student={cfg.student}")
    print(f"[Distill] teachers={cfg.teachers}")
    print(f"[Distill] alpha={cfg.alpha}, temperature={cfg.temperature}")

    loader_cfg = LoaderConfig(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        out_dim=cfg.out_dim,
    )

    train_loader, val_loader, test_loader = make_dataloaders(
        processed_dir=cfg.processed_dir,
        train_file=cfg.train_file,
        val_file=cfg.val_file,
        test_file=cfg.test_file,
        cfg=loader_cfg,
    )

    x0_raw, _ = next(iter(train_loader))
    x0 = feature_tensor_to_sequences(x0_raw.to(device))

    student = build_student(cfg.student, x0, cfg.out_dim).to(device)
    print(f"[Student] trainable params: {count_params(student):,}")

    teacher_names = [t.strip() for t in cfg.teachers.split(",") if t.strip()]
    teacher_specs = [
        TeacherSpec(name=t, checkpoint_path=f"results/{t}/checkpoint_best.pt")
        for t in teacher_names
    ]

    teachers = TeacherEnsemble(
        specs=teacher_specs,
        sample_x=x0,
        device=device,
        out_dim=cfg.out_dim,
        batch_size=cfg.batch_size,
    ).to(device)

    pos_weight = compute_pos_weight_from_loader(train_loader, cfg.out_dim, device)

    optimizer = Adam(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.lr_factor,
        patience=cfg.lr_patience,
        min_lr=cfg.min_lr,
    )

    teacher_tag = "_".join(teacher_names)
    run_name = f"distill_{cfg.student}_from_{teacher_tag}_a{cfg.alpha}_t{cfg.temperature}"
    results_path = os.path.join(cfg.results_dir, run_name)
    os.makedirs(results_path, exist_ok=True)

    best_val_f1 = -1.0
    patience_ctr = 0
    best_ckpt_path = os.path.join(results_path, "checkpoint_best.pt")

    history = {
        "epoch": [],
        "train_loss": [],
        "train_hard_loss": [],
        "train_kd_loss": [],
        "val_loss": [],
        "val_f1": [],
    }

    eval_cfg = TrainConfig(
        model_name=run_name,
        out_dim=cfg.out_dim,
        threshold=cfg.threshold,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        device=cfg.device,
        seed=cfg.seed,
        processed_dir=cfg.processed_dir,
        train_file=cfg.train_file,
        val_file=cfg.val_file,
        test_file=cfg.test_file,
        results_dir=cfg.results_dir,
        use_feature_adapter=True,
        save_curves=True,
        save_training_curves=True,
    )

    for epoch in range(1, cfg.epochs + 1):
        student.train()
        total_loss = 0.0
        total_hard = 0.0
        total_kd = 0.0
        total_n = 0

        for x_raw, y in train_loader:
            x_raw = x_raw.to(device)
            y = y.to(device).float()
            x = feature_tensor_to_sequences(x_raw)

            with torch.no_grad():
                teacher_logits = teachers(x)

            student_logits = student(x)

            hard_loss = F.binary_cross_entropy_with_logits(
                student_logits,
                y,
                pos_weight=pos_weight,
            )
            soft_loss = kd_bce_loss(
                student_logits,
                teacher_logits,
                temperature=cfg.temperature,
            )
            loss = cfg.alpha * hard_loss + (1.0 - cfg.alpha) * soft_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            n = int(y.size(0))
            total_loss += float(loss.item()) * n
            total_hard += float(hard_loss.item()) * n
            total_kd += float(soft_loss.item()) * n
            total_n += n

        train_loss = total_loss / max(1, total_n)
        train_hard = total_hard / max(1, total_n)
        train_kd = total_kd / max(1, total_n)

        val_metrics = eval_loop(
            student,
            val_loader,
            device,
            eval_cfg,
            output_dir=None,
            pos_weight=pos_weight,
        )
        val_f1 = float(val_metrics["f1_macro"])
        scheduler.step(val_f1)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_hard_loss"].append(train_hard)
        history["train_kd_loss"].append(train_kd)
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_f1"].append(val_f1)

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            patience_ctr = 0
            torch.save(
                {
                    "epoch": epoch,
                    "best_val_f1": best_val_f1,
                    "model_state": student.state_dict(),
                    "config": asdict(cfg),
                    "student_params": count_params(student),
                    "teacher_names": teacher_names,
                    "val_metrics": val_metrics,
                    "timestamp": time.time(),
                },
                best_ckpt_path,
            )
        else:
            patience_ctr += 1

        print(
            f"[Epoch {epoch:03d}] "
            f"train={train_loss:.4f} "
            f"hard={train_hard:.4f} "
            f"kd={train_kd:.4f} "
            f"val_f1={val_f1:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f}"
        )

        if patience_ctr >= cfg.patience:
            print(f"[EarlyStop] No val improvement for {cfg.patience} epochs.")
            break

    print("[Test] Loading best student checkpoint...")
    payload = torch.load(best_ckpt_path, map_location=device)
    student.load_state_dict(payload["model_state"])

    test_metrics = eval_loop(
        student,
        test_loader,
        device,
        eval_cfg,
        output_dir=results_path,
        pos_weight=pos_weight,
    )

    with open(os.path.join(results_path, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    with open(os.path.join(results_path, "training_curves.json"), "w") as f:
        json.dump(history, f, indent=2)

    summary = {
        "run_name": run_name,
        "student": cfg.student,
        "student_params": count_params(student),
        "teachers": teacher_names,
        "alpha": cfg.alpha,
        "temperature": cfg.temperature,
        "best_val_f1": best_val_f1,
        "test_metrics": test_metrics,
    }

    with open(os.path.join(results_path, "distill_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[Test Results]")
    for k, v in test_metrics.items():
        if k != "per_finger":
            print(f"  {k}: {v:.4f}")

    print(f"\n[Done] Results saved to {results_path}/")
    return summary


def parse_args() -> DistillConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--student", choices=["micro", "nano"], default="micro")
    p.add_argument("--teachers", default="cnn,lstm")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    return DistillConfig(
        student=args.student,
        teachers=args.teachers,
        alpha=args.alpha,
        temperature=args.temperature,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    train_distill(parse_args())
