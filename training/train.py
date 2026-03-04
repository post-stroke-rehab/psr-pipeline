# training/train.py
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

    processed_dir: str = "datasets/processed/physiomio"
    train_file: str = "train.pt"
    val_file: str = "val.pt"
    test_file: str = "test.pt"

    batch_size: int = 64
    num_workers: int = 0

    ckpt_dir: str = "training/ckpts"
    run_dir: str = "training/runs"
    resume_path: Optional[str] = None

    save_curves: bool = True
    save_training_curves: bool = False


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
        )
        
        return SEMGFingerPredictor(gcfg)
    
    if name == "cnn":
        from models.CNN import cnn as cnn_impl

        if sample_x.dim() != 3:
            raise ValueError(f"CNN expects (N,W,F) before permute. Got {tuple(sample_x.shape)}")

        in_features = int(sample_x.size(-1))  # F
        seq_len = int(sample_x.size(1))       # W

        cnn_cfg = cnn_impl.Config(
            in_channels=in_features,   # channels = features
            seq_len=seq_len,           # length = time/window
            out_dim=cfg.out_dim,
            use_adaptive_pool=True,   
            conv4_kernel=5,  
        )

        base_cnn = cnn_impl.build_model(model_name="base", cfg=cnn_cfg)
        
        #current pipeline feeds model (N,W,features) but pytorch Conv1d expect (N,C,L) so W and features are swapped
        class PermuteToChannelsFirst(nn.Module):
            """Wrap a Conv1d CNN that expects (N, C, L) since our current pipeline provides (N, L, C)."""
            def __init__(self, backbone: nn.Module):
                super().__init__()
                self.backbone = backbone

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x: (N, W, F) -> (N, F, W)
                if x.dim() != 3:
                    raise ValueError(f"CNN wrapper expects 3D tensor (N,W,F). Got {tuple(x.shape)}")
                x = x.permute(0, 2, 1).contiguous()
                return self.backbone(x)

        # wrap so it accepts (N,W,F) 
        return PermuteToChannelsFirst(base_cnn)

    raise ValueError(f"Unknown model_name='{cfg.model_name}'. Use 'lstm' or 'gnn' or 'cnn'.")


# def bce_logits_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#     return F.binary_cross_entropy_with_logits(logits, targets)
def bce_logits_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    pos_weight: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

#tryihng to address class imbalance, this func just counts labels in the training dataset
def compute_pos_weight_from_loader(train_loader, out_dim: int, device: torch.device) -> torch.Tensor:
    pos = torch.zeros(out_dim, device=device)
    total = 0

    for _, y in train_loader:
        y = y.float().to(device)
        pos += y.sum(dim=0)      # positives per finger
        total += y.size(0)       # number of samples

    neg = total - pos
    pos_weight = neg / (pos + 1e-8)
    pos_weight = torch.clamp(pos_weight, min=1.0)

    print("[pos_weight]", pos_weight.detach().cpu().tolist())
    return pos_weight

@torch.no_grad()
def eval_loop(
    model: nn.Module,
    loader,
    device: torch.device,
    cfg: TrainConfig,
    *,
    output_dir: Optional[str] = None,
    pos_weight: Optional[torch.Tensor] = None
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
        #
        pred = (probs >= cfg.threshold).float()
        # print("avg_pred_positive_per_finger:", pred.mean(dim=0).tolist())
        # print("avg_prob_per_finger:", probs.mean(dim=0).tolist())


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
        "per_finger": metrics["per_finger"],
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
    payload = torch.load(path, map_location="cpu", weights_only=True)
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
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Device] Running on CUDA: {gpu_name}")
    else:
        print("[Device] Running on CPU")

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

    def print_label_balance(loader):
        total = 0
        pos = None

        for X, y in loader:
            y = y.float()
            if pos is None:
                pos = y.sum(dim=0)
            else:
                pos += y.sum(dim=0)
            total += y.shape[0]

        rates = pos / total
        print("\n[Train Label Positive Rates]")
        for i, r in enumerate(rates):
            print(f"  finger_{i}: {r.item():.3f}")

    print_label_balance(train_loader)

    pos_weight = compute_pos_weight_from_loader(train_loader, cfg.out_dim, device)

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
    
    history = None
    if cfg.save_training_curves:
        history = {"epoch": [], "train_loss": [], "val_loss": [], "val_f1": []}

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
            loss = bce_logits_loss(logits, y, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()

            n = int(y.size(0))
            total_loss += float(loss.item()) * n
            total_n += n

        train_loss = total_loss / max(1, total_n)

        val_metrics = eval_loop(model, val_loader, device, cfg, pos_weight=pos_weight)
        val_f1 = float(val_metrics["f1_macro"])
        scheduler.step(val_f1)
        if history is not None:
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(float(val_metrics["loss"]))
            history["val_f1"].append(val_f1)

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
    test_metrics = eval_loop(model, test_loader, device, cfg, output_dir=test_out_dir, pos_weight=pos_weight)

    print("[Test Results]")

    for k, v in test_metrics.items():
        if k != "per_finger":
            print(f"  {k}: {v:.4f}")

    print("\n[Per-Finger Metrics]")
    for finger, stats in test_metrics["per_finger"].items():
        print(
            f"  {finger}: "
            f"F1={stats['f1']:.4f}, "
            f"Precision={stats['precision']:.4f}, "
            f"Recall={stats['recall']:.4f}, "
            f"AUPRC={stats['auprc']:.4f}, "
            f"AUROC={stats['auroc']:.4f}"
        )

    if history is not None:
        with open(os.path.join(run_path, "training_curves.json"), "w") as f:
            json.dump(history, f, indent=2)
    with open(os.path.join(run_path, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    return test_metrics

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lstm",
                        choices=["lstm", "gnn", "cnn"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_training_curves", action="store_true")

    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.model_name = args.model
    cfg.epochs = args.epochs
    cfg.lr = args.lr
    cfg.batch_size = args.batch_size
    cfg.save_training_curves = args.save_training_curves

    train_loop(cfg)