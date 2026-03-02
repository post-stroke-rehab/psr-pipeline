# training/training_pipeline.py
import os
import json
import time
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets.loaders import make_dataloaders, LoaderConfig
from adapters.feature_to_sequence import feature_tensor_to_sequences

from evaluation.metrics import compute_multilabel_metrics, compute_curves
from evaluation.plots import save_metric_curves


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig:
    model_name: str = "lstm"  # "lstm" | "gnn"

    out_dim: int = 5
    threshold: float = 0.5

    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 25

    lr_patience: int = 10
    lr_factor: float = 0.5
    min_lr: float = 1e-6

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    use_feature_adapter: bool = True

    processed_dir: str = "datasets/processed"
    train_file: str = "train.pt"
    val_file: str = "val.pt"
    test_file: str = "test.pt"

    batch_size: int = 64
    num_workers: int = 0

    ckpt_dir: str = "ckpts"
    run_dir: str = "runs"
    resume_path: Optional[str] = None

    save_curves: bool = False


def build_model(cfg: TrainConfig, sample_x: torch.Tensor) -> nn.Module:
    name = cfg.model_name.lower().strip()

    if name == "lstm":
        from models.lstm import LSTM_model, LSTMConfig  # you should place your lstm.py in models/

        if sample_x.dim() != 3:
            raise ValueError(f"LSTM expects (N,W,features). Got {tuple(sample_x.shape)}")

        in_features = int(sample_x.size(-1))
        mcfg = LSTMConfig(input_size=in_features, out_dim=cfg.out_dim)
        return LSTM_model(mcfg)

    if name == "gnn":
        from models.gnn import SEMGFingerPredictor, Config as GNNConfig  # place gnn.py in models/

        if sample_x.dim() != 3:
            raise ValueError(f"GNN expects (N,W,features). Got {tuple(sample_x.shape)}")

        in_features = int(sample_x.size(-1))
        seq_len = int(sample_x.size(1))

        gcfg = GNNConfig(
            seq_len=seq_len,
            in_features=in_features,
            out_dim=cfg.out_dim,
            batch_size=cfg.batch_size,
            device=cfg.device,
            use_feature_adapter=False,  # we already adapted before calling model
        )
        return SEMGFingerPredictor(gcfg)

    raise ValueError(f"Unknown model_name='{cfg.model_name}'. Use 'lstm' or 'gnn'.")


def bce_logits_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets)


@torch.no_grad()
def eval_loop(
    model: nn.Module,
    loader,
    device: torch.device,
    cfg: TrainConfig,
    *,
    output_dir: Optional[str] = None,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_n = 0
    all_probs = []
    all_targets = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()

        if cfg.use_feature_adapter:
            x = feature_tensor_to_sequences(x)  # (N,C,W,F)->(N,W,C*F)

        logits = model(x)
        loss = bce_logits_loss(logits, y)

        n = int(y.size(0))
        total_loss += float(loss.item()) * n
        total_n += n

        probs = torch.sigmoid(logits)
        all_probs.append(probs.detach().cpu())
        all_targets.append(y.detach().cpu())

    probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)

    metrics = compute_multilabel_metrics(
        probs, targets, threshold=cfg.threshold, num_classes=cfg.out_dim
    )

    out: Dict[str, float] = {
        "loss": total_loss / max(1, total_n),
        "accuracy": float(metrics["accuracy"]),
        "finger_accuracy": float(metrics["finger_accuracy"]),
        "precision_macro": float(metrics["precision_macro"]),
        "recall_macro": float(metrics["recall_macro"]),
        "f1_macro": float(metrics["f1_macro"]),
        "auprc_macro": float(metrics["auprc_macro"]),
        "auroc_macro": float(metrics["auroc_macro"]),
        "auprc_micro": float(metrics["auprc_micro"]),
        "auroc_micro": float(metrics["auroc_micro"]),
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(out, f, indent=2)

        if cfg.save_curves:
            curve_data = compute_curves(probs, targets, num_classes=cfg.out_dim)
            save_metric_curves(curve_data, output_dir, prefix="", save_per_finger=False)

    return out


def save_checkpoint(
    path: str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    cfg: TrainConfig,
    epoch: int,
    best_val_f1: float,
    last_val_metrics: Dict[str, float],
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "best_val_f1": best_val_f1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "config": asdict(cfg),
        "val_metrics": last_val_metrics,
        "timestamp": time.time(),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    *,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Tuple[int, float]:
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state"])

    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])

    start_epoch = int(payload.get("epoch", 0)) + 1
    best_val_f1 = float(payload.get("best_val_f1", 0.0))
    return start_epoch, best_val_f1


def train_loop(cfg: TrainConfig) -> Dict[str, float]:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

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

    x0, y0 = next(iter(train_loader))
    if cfg.use_feature_adapter:
        x0 = feature_tensor_to_sequences(x0)
    model = build_model(cfg, x0).to(device)

    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.lr_factor,
        patience=cfg.lr_patience,
        min_lr=cfg.min_lr,
        verbose=True,
    )

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    os.makedirs(cfg.run_dir, exist_ok=True)

    run_name = f"{cfg.model_name}_{int(time.time())}"
    run_path = os.path.join(cfg.run_dir, run_name)
    os.makedirs(run_path, exist_ok=True)

    start_epoch = 1
    best_val_f1 = 0.0
    if cfg.resume_path and os.path.isfile(cfg.resume_path):
        start_epoch, best_val_f1 = load_checkpoint(
            cfg.resume_path, model=model, optimizer=optimizer, scheduler=scheduler
        )

    best_ckpt = os.path.join(cfg.ckpt_dir, f"{cfg.model_name}_best.pt")
    last_ckpt = os.path.join(cfg.ckpt_dir, f"{cfg.model_name}_last.pt")

    patience_ctr = 0

    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_n = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).float()

            if cfg.use_feature_adapter:
                x = feature_tensor_to_sequences(x)

            optimizer.zero_grad()
            logits = model(x)
            loss = bce_logits_loss(logits, y)
            loss.backward()
            optimizer.step()

            n = int(y.size(0))
            total_loss += float(loss.item()) * n
            total_n += n

        train_loss = total_loss / max(1, total_n)

        val_metrics = eval_loop(model, val_loader, device, cfg)
        val_f1 = float(val_metrics["f1_macro"])
        scheduler.step(val_f1)

        save_checkpoint(
            last_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            epoch=epoch,
            best_val_f1=best_val_f1,
            last_val_metrics=val_metrics,
        )

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            patience_ctr = 0
            save_checkpoint(
                best_ckpt,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                epoch=epoch,
                best_val_f1=best_val_f1,
                last_val_metrics=val_metrics,
            )
        else:
            patience_ctr += 1

        if epoch == 1 or epoch % 5 == 0 or is_best:
            print(
                f"[Epoch {epoch:03d}] "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_f1={val_f1:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )

        if patience_ctr >= cfg.patience:
            print(f"[EarlyStop] No val improvement for {cfg.patience} epochs.")
            break

    print("[Test] Loading best checkpoint...")
    _ = load_checkpoint(best_ckpt, model=model)

    test_out_dir = os.path.join(run_path, "test")
    test_metrics = eval_loop(model, test_loader, device, cfg, output_dir=test_out_dir)

    print("[Test Results]")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    with open(os.path.join(run_path, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    return test_metrics


if __name__ == "__main__":
    cfg = TrainConfig()
    train_loop(cfg)