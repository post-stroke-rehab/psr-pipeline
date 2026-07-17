# training/train.py
import os
import json
import time
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import numpy as np

from datasets.loaders import make_dataloaders, LoaderConfig
from adapters.feature_to_sequence import feature_tensor_to_sequences

from evaluation.metrics import compute_multilabel_metrics, compute_curves
from evaluation.plots import save_metric_curves, save_loss_curves, save_confusion_matrices

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


PROCESSED_ROOT = "datasets/processed/physiomio"
HEALTHY_PROCESSED_DIR = f"{PROCESSED_ROOT}/healthy"
IMPAIRED_PROCESSED_DIR = f"{PROCESSED_ROOT}/impaired"
TRAINING_STAGES = ("direct", "pretrain", "finetune")


@dataclass
class TrainConfig:
    model_name: str = "lstm"  # "lstm" | "gnn" | "cnn"
    training_stage: str = "direct"  # "direct" | "pretrain" | "finetune"

    out_dim: int = 5
    threshold: Union[float, Sequence[float]] = 0.5

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

    pretrained_ckpt_path: Optional[str] = None
    pretrained_init: str = "backbone"  # backbone | full
    freeze_backbone_epochs: int = 0
    backbone_lr: Optional[float] = None
    head_lr: Optional[float] = None

    lstm_hidden1: int = 128
    lstm_hidden2: int = 256
    lstm_fc_hidden: int = 128
    lstm_dropout: float = 0.5

    cnn_variant: str = "base"  # nano | micro | base | large | xlarge
    cnn_dropout: float = 0.2

    gnn_type: str = "gcn"  # gcn | sage | gat
    gnn_num_layers: int = 2
    gnn_hid_dim: int = 64
    gnn_aggregation: str = "mean"  # mean | max | add
    gnn_readout_layers: int = 1
    gnn_dropout: float = 0.5

    finger0_pos_weight_boost: float = 1.3
    verbose_data_stats: bool = True

    early_stop_metric: str = "f1_macro"  # f1_macro | finger_accuracy
    tune_threshold: bool = False

    batch_size: int = 64
    num_workers: int = 0

    ckpt_dir: str = "training/ckpts"
    run_dir: str = "training/runs"
    results_dir: str = "results"   # final benchmark outputs: results/{model_name}/
    resume_path: Optional[str] = None

    save_curves: bool = True
    save_training_curves: bool = True  # always save for benchmarking


def default_processed_dir(stage: str) -> str:
    if stage == "pretrain":
        return HEALTHY_PROCESSED_DIR
    if stage in ("direct", "finetune"):
        return IMPAIRED_PROCESSED_DIR
    raise ValueError(f"Unknown training_stage='{stage}'. Use one of {TRAINING_STAGES}.")


def resolve_processed_dir(cfg: TrainConfig) -> str:
    if cfg.processed_dir != PROCESSED_ROOT:
        return cfg.processed_dir
    if cfg.training_stage == "direct":
        return cfg.processed_dir
    return default_processed_dir(cfg.training_stage)


def stage_results_dir_name(stage: str) -> str:
    if stage == "pretrain":
        return "best_pretrain"
    if stage == "finetune":
        return "best_finetune"
    return stage


def stage_results_path(cfg: TrainConfig) -> str:
    return os.path.join(cfg.results_dir, stage_results_dir_name(cfg.training_stage), cfg.model_name)


def stage_run_path(cfg: TrainConfig) -> str:
    return os.path.join(cfg.run_dir, cfg.training_stage, f"{cfg.model_name}_{int(time.time())}")


def stage_ckpt_dir(cfg: TrainConfig) -> str:
    return os.path.join(cfg.ckpt_dir, cfg.training_stage, cfg.model_name)


def init_from_pretrained(model: nn.Module, ckpt_path: str, model_name: str, init_mode: str = "backbone") -> None:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = payload["model_state"]

    if model_name == "lstm" and init_mode == "backbone":
        from models.lstm import load_pretrained_backbone

        load_pretrained_backbone(model, state)
        return

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[Pretrained] Loaded with {len(missing)} missing keys (head/arch mismatch ok).")
    if unexpected:
        print(f"[Pretrained] Ignored {len(unexpected)} unexpected keys.")


def set_lstm_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    from models.lstm import set_backbone_trainable

    set_backbone_trainable(model, trainable)


def build_optimizer(model: nn.Module, cfg: TrainConfig) -> Adam:
    use_split_lr = (
        cfg.training_stage == "finetune"
        and cfg.pretrained_ckpt_path
        and cfg.model_name == "lstm"
    )

    if not use_split_lr:
        return Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    from models.lstm import backbone_parameters, head_parameters

    backbone_lr = cfg.backbone_lr if cfg.backbone_lr is not None else cfg.lr * 0.1
    head_lr = cfg.head_lr if cfg.head_lr is not None else cfg.lr
    param_groups: List[Dict[str, Any]] = [
        {"params": list(backbone_parameters(model)), "lr": backbone_lr},
        {"params": list(head_parameters(model)), "lr": head_lr},
    ]
    return Adam(param_groups, weight_decay=cfg.weight_decay)


def maybe_unfreeze_backbone(model: nn.Module, cfg: TrainConfig, epoch: int) -> None:
    if (
        cfg.training_stage != "finetune"
        or not cfg.pretrained_ckpt_path
        or cfg.model_name != "lstm"
        or cfg.freeze_backbone_epochs <= 0
    ):
        return

    if epoch == cfg.freeze_backbone_epochs + 1:
        set_lstm_backbone_trainable(model, True)
        print(f"[Finetune] Unfroze LSTM backbone at epoch {epoch}.")


def apply_finetune_defaults(cfg: TrainConfig) -> None:
    if cfg.model_name != "lstm":
        return
    if cfg.training_stage != "finetune" or not cfg.pretrained_ckpt_path:
        return
    if cfg.freeze_backbone_epochs == 0:
        cfg.freeze_backbone_epochs = 5
    if cfg.backbone_lr is None:
        cfg.backbone_lr = 5e-5
    if cfg.head_lr is None:
        cfg.head_lr = 5e-4


def validation_score(metrics: Dict[str, float], metric_name: str) -> float:
    if metric_name == "finger_accuracy":
        return float(metrics["finger_accuracy"])
    if metric_name == "f1_macro":
        return float(metrics["f1_macro"])
    raise ValueError(f"Unsupported early_stop_metric='{metric_name}'")


@torch.no_grad()
def collect_probs_and_targets(
    model: nn.Module,
    loader,
    device: torch.device,
    cfg: TrainConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    all_probs = []
    all_targets = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()
        if cfg.use_feature_adapter:
            x = feature_tensor_to_sequences(x)
        logits = model(x)
        all_probs.append(torch.sigmoid(logits).detach().cpu())
        all_targets.append(y.detach().cpu())

    return torch.cat(all_probs, dim=0), torch.cat(all_targets, dim=0)


def tune_threshold_on_val(
    probs: torch.Tensor,
    targets: torch.Tensor,
    *,
    out_dim: int,
) -> Tuple[Union[float, List[float]], float, str]:
    grid = np.arange(0.30, 0.71, 0.02)

    best_global = 0.5
    best_global_acc = -1.0
    for threshold in grid:
        metrics = compute_multilabel_metrics(
            probs,
            targets,
            threshold=float(threshold),
            num_classes=out_dim,
        )
        finger_acc = float(metrics["finger_accuracy"])
        if finger_acc > best_global_acc:
            best_global_acc = finger_acc
            best_global = float(threshold)

    per_finger: List[float] = []
    for j in range(out_dim):
        best_t = 0.5
        best_acc = -1.0
        col_probs = probs[:, j]
        col_targets = targets[:, j]
        grid = np.arange(0.20, 0.66, 0.02) if j == 0 else np.arange(0.30, 0.71, 0.02)
        for t in grid:
            acc = float(((col_probs >= t) == col_targets).float().mean().item())
            if acc > best_acc:
                best_acc = acc
                best_t = float(t)
        per_finger.append(best_t)

    per_finger_metrics = compute_multilabel_metrics(
        probs,
        targets,
        threshold=per_finger,
        num_classes=out_dim,
    )
    per_finger_acc = float(per_finger_metrics["finger_accuracy"])

    if per_finger_acc >= best_global_acc:
        return per_finger, per_finger_acc, "per_finger"
    return best_global, best_global_acc, "global"


def build_model(cfg: TrainConfig, sample_x: torch.Tensor) -> nn.Module:
    name = cfg.model_name.lower().strip()

    if name == "lstm":
        from models.lstm import LSTM_model, LSTMConfig  # you should place your lstm.py in models/

        if sample_x.dim() != 3:
            raise ValueError(f"LSTM expects (N,W,features). Got {tuple(sample_x.shape)}")

        in_features = int(sample_x.size(-1))
        mcfg = LSTMConfig(
            input_size=in_features,
            out_dim=cfg.out_dim,
            hidden1=cfg.lstm_hidden1,
            hidden2=cfg.lstm_hidden2,
            fc_hidden=cfg.lstm_fc_hidden,
            dropout=cfg.lstm_dropout,
        )
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
            model=cfg.gnn_type,
            num_gnn_layers=cfg.gnn_num_layers,
            hid_dim=cfg.gnn_hid_dim,
            aggregation=cfg.gnn_aggregation,
            readout_layers=cfg.gnn_readout_layers,
            dropout=cfg.gnn_dropout,
        )
        
        return SEMGFingerPredictor(gcfg)
    
    if name == "cnn":
        from models.CNN import cnn as cnn_impl

        if sample_x.dim() != 3:
            raise ValueError(f"CNN expects (N,W,F) before permute. Got {tuple(sample_x.shape)}")
        in_features = int(sample_x.size(-1))
        seq_len = int(sample_x.size(1))

        cnn_cfg = cnn_impl.Config(
            in_channels=in_features,
            seq_len=seq_len,
            out_dim=cfg.out_dim,
            use_adaptive_pool=True,
            conv4_kernel=5,
            dropout=cfg.cnn_dropout,
        )

        base_cnn = cnn_impl.build_model(model_name=cfg.cnn_variant, cfg=cnn_cfg)
        
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
def compute_pos_weight_from_loader(
    train_loader,
    out_dim: int,
    device: torch.device,
    cfg: Optional[TrainConfig] = None,
) -> torch.Tensor:
    pos = torch.zeros(out_dim, device=device)
    total = 0

    for _, y in train_loader:
        y = y.float().to(device)
        pos += y.sum(dim=0)
        total += y.size(0)

    neg = total - pos
    pos_weight = neg / (pos + 1e-8)
    pos_weight = torch.clamp(pos_weight, min=1.0)

    if cfg is not None and cfg.finger0_pos_weight_boost != 1.0:
        pos_weight[0] = pos_weight[0] * cfg.finger0_pos_weight_boost

    if cfg is None or cfg.verbose_data_stats:
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
            save_confusion_matrices(
                probs, targets, output_dir,
                threshold=(
                    float(np.mean(cfg.threshold))
                    if isinstance(cfg.threshold, (list, tuple))
                    else cfg.threshold
                ),
                num_classes=cfg.out_dim,
            )

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


def train_loop(
    cfg: TrainConfig,
    *,
    optuna_trial: Optional[Any] = None,
    tune_max_epochs: Optional[int] = None,
    tune_skip_test: bool = False,
    apply_defaults: bool = True,
) -> Dict[str, Any]:
    if cfg.training_stage not in TRAINING_STAGES:
        raise ValueError(f"Unknown training_stage='{cfg.training_stage}'. Use one of {TRAINING_STAGES}.")

    cfg.processed_dir = resolve_processed_dir(cfg)
    if apply_defaults:
        apply_finetune_defaults(cfg)
    set_seed(cfg.seed)
    max_epochs = tune_max_epochs if tune_max_epochs is not None else cfg.epochs
    tune_patience = max(10, cfg.patience // 2) if tune_skip_test else cfg.patience
    device = torch.device(cfg.device)
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Device] Running on CUDA: {gpu_name}")
    else:
        print("[Device] Running on CPU")

    print(f"[Stage] {cfg.training_stage} | model={cfg.model_name} | data={cfg.processed_dir}")

    loader_cfg = LoaderConfig(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        out_dim=cfg.out_dim,
        arm_split="healthy" if cfg.training_stage == "pretrain" else "impaired",
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
        if cfg.verbose_data_stats:
            print("\n[Train Label Positive Rates]")
            for i, r in enumerate(rates):
                print(f"  finger_{i}: {r.item():.3f}")

    print_label_balance(train_loader)

    pos_weight = compute_pos_weight_from_loader(train_loader, cfg.out_dim, device, cfg)

    x0, y0 = next(iter(train_loader))
    if cfg.use_feature_adapter:
        x0 = feature_tensor_to_sequences(x0)
    model = build_model(cfg, x0).to(device)

    if cfg.training_stage == "finetune" and cfg.pretrained_ckpt_path:
        if not os.path.isfile(cfg.pretrained_ckpt_path):
            raise FileNotFoundError(f"pretrained_ckpt_path not found: {cfg.pretrained_ckpt_path}")
        print(f"[Finetune] Loading pretrained weights from {cfg.pretrained_ckpt_path}")
        init_from_pretrained(
            model,
            cfg.pretrained_ckpt_path,
            cfg.model_name,
            init_mode=cfg.pretrained_init,
        )
        if cfg.model_name == "lstm" and cfg.freeze_backbone_epochs > 0:
            set_lstm_backbone_trainable(model, False)
            print(f"[Finetune] Froze LSTM backbone for first {cfg.freeze_backbone_epochs} epochs.")

    optimizer = build_optimizer(model, cfg)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.lr_factor,
        patience=cfg.lr_patience,
        min_lr=cfg.min_lr,
    )

    os.makedirs(stage_ckpt_dir(cfg), exist_ok=True)
    os.makedirs(cfg.run_dir, exist_ok=True)

    run_path = stage_run_path(cfg)
    os.makedirs(run_path, exist_ok=True)

    start_epoch = 1
    best_val_score = 0.0
    if cfg.resume_path and os.path.isfile(cfg.resume_path):
        start_epoch, best_val_score = load_checkpoint(
            cfg.resume_path, model=model, optimizer=optimizer, scheduler=scheduler
        )

    best_ckpt = os.path.join(stage_ckpt_dir(cfg), f"{cfg.model_name}_best.pt")
    last_ckpt = os.path.join(stage_ckpt_dir(cfg), f"{cfg.model_name}_last.pt")

    patience_ctr = 0
    
    history = None
    if cfg.save_training_curves:
        history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "val_finger_accuracy": [],
        }

    for epoch in range(start_epoch, max_epochs + 1):
        maybe_unfreeze_backbone(model, cfg, epoch)
        if (
            cfg.training_stage == "finetune"
            and cfg.model_name == "lstm"
            and cfg.pretrained_ckpt_path
            and epoch == cfg.freeze_backbone_epochs + 1
            and cfg.freeze_backbone_epochs > 0
        ):
            optimizer = build_optimizer(model, cfg)

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
        val_score = validation_score(val_metrics, cfg.early_stop_metric)
        val_f1 = float(val_metrics["f1_macro"])
        val_finger_acc = float(val_metrics["finger_accuracy"])
        scheduler.step(val_score)
        if history is not None:
            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(float(val_metrics["loss"]))
            history["val_f1"].append(val_f1)
            history["val_finger_accuracy"].append(val_finger_acc)

        save_checkpoint(
            last_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            cfg=cfg,
            epoch=epoch,
            best_val_f1=best_val_score,
            last_val_metrics=val_metrics,
        )

        is_best = val_score > best_val_score
        if is_best:
            best_val_score = val_score
            patience_ctr = 0
            save_checkpoint(
                best_ckpt,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                epoch=epoch,
                best_val_f1=best_val_score,
                last_val_metrics=val_metrics,
            )
        else:
            patience_ctr += 1

        if epoch == 1 or epoch % 5 == 0 or is_best:
            if optuna_trial is None:
                print(
                    f"[Epoch {epoch:03d}] "
                    f"train_loss={train_loss:.4f} "
                    f"val_loss={val_metrics['loss']:.4f} "
                    f"val_f1={val_f1:.4f} "
                    f"val_finger_acc={val_finger_acc:.4f} "
                    f"val_acc={val_metrics['accuracy']:.4f}"
                )

        if optuna_trial is not None:
            optuna_trial.report(val_score, epoch)
            if optuna_trial.should_prune():
                import optuna
                raise optuna.TrialPruned()

        if patience_ctr >= tune_patience:
            if optuna_trial is None:
                print(f"[EarlyStop] No val improvement for {tune_patience} epochs.")
            break

    if tune_skip_test:
        return {
            "val_score": best_val_score,
            "best_ckpt": best_ckpt,
            "results_path": None,
            "test_metrics": None,
        }

    print("[Test] Loading best checkpoint...")
    _ = load_checkpoint(best_ckpt, model=model)

    if cfg.tune_threshold:
        val_probs, val_targets = collect_probs_and_targets(model, val_loader, device, cfg)
        tuned_threshold, tuned_val_finger_acc, threshold_mode = tune_threshold_on_val(
            val_probs,
            val_targets,
            out_dim=cfg.out_dim,
        )
        if threshold_mode == "per_finger":
            print(
                f"[Threshold] Per-finger: "
                f"{[round(t, 2) for t in tuned_threshold]} "
                f"(val finger_accuracy={tuned_val_finger_acc:.4f})"
            )
        else:
            print(
                f"[Threshold] Global {tuned_threshold:.2f} "
                f"(val finger_accuracy={tuned_val_finger_acc:.4f})"
            )
        cfg.threshold = tuned_threshold

    # Final outputs go to results/{stage}/{model_name}/
    results_path = stage_results_path(cfg)
    os.makedirs(results_path, exist_ok=True)

    test_metrics = eval_loop(
        model, test_loader, device, cfg,
        output_dir=results_path,
        pos_weight=pos_weight,
    )

    print("[Test Results]")
    for k, v in test_metrics.items():
        if k != "per_finger":
            print(f"  {k}: {v:.4f}")

    print("\n[Per-Finger Metrics]")
    for finger, stats in test_metrics["per_finger"].items():
        print(
            f"  {finger}: "
            f"Acc={stats['accuracy']:.4f}, "
            f"F1={stats['f1']:.4f}, "
            f"Precision={stats['precision']:.4f}, "
            f"Recall={stats['recall']:.4f}, "
            f"AUPRC={stats['auprc']:.4f}, "
            f"AUROC={stats['auroc']:.4f}"
        )

    # Training curves → results/{model_name}/
    if history is not None:
        curves_path = os.path.join(results_path, "training_curves.json")
        with open(curves_path, "w") as f:
            json.dump(history, f, indent=2)
        save_loss_curves(history, results_path)

    # Config → results/{model_name}/config.json
    with open(os.path.join(results_path, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    # Copy best checkpoint → results/{model_name}/checkpoint_best.pt
    import shutil
    if os.path.isfile(best_ckpt):
        shutil.copy2(best_ckpt, os.path.join(results_path, "checkpoint_best.pt"))

    # Generate summary.md
    _write_summary_md(cfg, test_metrics, results_path)

    print(f"\n[Done] Results saved to {results_path}/")
    return {
        "test_metrics": test_metrics,
        "results_path": results_path,
        "best_ckpt": best_ckpt,
    }


def _write_summary_md(cfg: TrainConfig, test_metrics: Dict, results_path: str) -> None:
    """Write results/gnn/summary.md with split info, hyperparameters, and final metrics."""
    pf = test_metrics.get("per_finger", {})

    finger_rows = ""
    for finger, stats in pf.items():
        finger_rows += (
            f"| {finger} | {stats['accuracy']:.4f} | {stats['precision']:.4f} | {stats['recall']:.4f} "
            f"| {stats['f1']:.4f} | {stats['auroc']:.4f} | {stats['auprc']:.4f} |\n"
        )

    md = f"""# GNN Benchmark Results

## Split Method
- **Strategy:** Patient-level 70 / 10 / 20 split (train / val / test)
- **No subject overlap** between splits — patients assigned exclusively to one split
- **Seed:** {cfg.seed}
- **Dataset:** PhysioMio (`formove-ai/physiomio`), {cfg.training_stage} stage, data from `{cfg.processed_dir}`

## Hyperparameters
| Parameter | Value |
|-----------|-------|
| Model | {cfg.model_name.upper()} |
| Training stage | {cfg.training_stage} |
| Optimizer | Adam |
| Learning rate | {cfg.lr} |
| Weight decay | {cfg.weight_decay} |
| Batch size | {cfg.batch_size} |
| Max epochs | {cfg.epochs} |
| Early stopping patience | {cfg.patience} epochs |
| LR scheduler | ReduceLROnPlateau (factor={cfg.lr_factor}, patience={cfg.lr_patience}) |
| Min LR | {cfg.min_lr} |
| Threshold | {cfg.threshold} |
| Device | {cfg.device} |

## Final Test Metrics
| Metric | Value |
|--------|-------|
| Accuracy (exact match) | {test_metrics.get('accuracy', float('nan')):.4f} |
| Finger Accuracy | {test_metrics.get('finger_accuracy', float('nan')):.4f} |
| Precision (macro) | {test_metrics.get('precision_macro', float('nan')):.4f} |
| Recall (macro) | {test_metrics.get('recall_macro', float('nan')):.4f} |
| F1 (macro) | {test_metrics.get('f1_macro', float('nan')):.4f} |
| AUROC (macro) | {test_metrics.get('auroc_macro', float('nan')):.4f} |
| AUPRC (macro) | {test_metrics.get('auprc_macro', float('nan')):.4f} |
| AUROC (micro) | {test_metrics.get('auroc_micro', float('nan')):.4f} |
| AUPRC (micro) | {test_metrics.get('auprc_micro', float('nan')):.4f} |

## Per-Finger Breakdown
| Finger | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
|--------|----------|-----------|--------|----|-------|-------|
{finger_rows}
## Outputs
- `metrics.json` — full test metrics
- `training_curves.json` + `loss_curves.png` — training history
- `pr_curve.png` + `roc_curve.png` — PR and ROC curves
- `confusion_matrices.png` — per-finger confusion matrices
- `checkpoint_best.pt` — best model checkpoint
- `config.json` — full training config
"""
    with open(os.path.join(results_path, "summary.md"), "w") as f:
        f.write(md)
    print(f"[Summary] Written to {os.path.join(results_path, 'summary.md')}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lstm",
                        choices=["lstm", "gnn", "cnn"])
    parser.add_argument("--stage", type=str, default="direct",
                        choices=list(TRAINING_STAGES))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--processed-dir", type=str, default=None)
    parser.add_argument("--pretrained-ckpt", type=str, default=None)
    parser.add_argument("--pretrained-init", type=str, default="backbone",
                        choices=["backbone", "full"])
    parser.add_argument("--freeze-backbone-epochs", type=int, default=None)
    parser.add_argument("--backbone-lr", type=float, default=None)
    parser.add_argument("--head-lr", type=float, default=None)
    parser.add_argument("--early-stop-metric", type=str, default=None,
                        choices=["f1_macro", "finger_accuracy"])
    parser.add_argument("--no-tune-threshold", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None,
                        help="cuda, cpu, or cuda:0. Defaults to cuda when available.")

    args = parser.parse_args()

    cfg = TrainConfig()
    cfg.model_name = args.model
    cfg.training_stage = args.stage
    cfg.epochs = args.epochs
    cfg.results_dir = args.results_dir
    cfg.seed = args.seed
    cfg.save_training_curves = True
    if args.lr is not None:
        cfg.lr = args.lr
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.model == "lstm":
        if args.lr is None:
            cfg.lr = 5e-4
        if args.batch_size is None:
            cfg.batch_size = 64
        if args.early_stop_metric is None:
            cfg.early_stop_metric = "finger_accuracy"
        if not args.no_tune_threshold:
            cfg.tune_threshold = True
    if args.processed_dir is not None:
        cfg.processed_dir = args.processed_dir
    cfg.pretrained_ckpt_path = args.pretrained_ckpt
    cfg.pretrained_init = args.pretrained_init
    if args.freeze_backbone_epochs is not None:
        cfg.freeze_backbone_epochs = args.freeze_backbone_epochs
    cfg.backbone_lr = args.backbone_lr
    cfg.head_lr = args.head_lr
    if args.early_stop_metric is not None:
        cfg.early_stop_metric = args.early_stop_metric
    if args.no_tune_threshold:
        cfg.tune_threshold = False
    if args.device is not None:
        cfg.device = args.device

    train_loop(cfg)
