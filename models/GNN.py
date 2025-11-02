# gnn_pipeline.py
# sEMG Finger Prediction GNN Pipeline using PyTorch Geometric.
# Supports: sequential sEMG input (batch, seq_len, 3), 5-finger sigmoid output,
# hyperparameter tuning with Bayesian optimization, configurable GNN architectures.

import os
import math
import json
import time
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Core PyG pieces
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv,
    global_mean_pool, global_add_pool, global_max_pool
)
from torch_geometric.utils import to_undirected, dense_to_sparse

# Bayesian optimization
import optuna
from optuna.trial import TrialState


# -----------------------------
# Reproducibility utilities
# -----------------------------
def set_seed(seed: int = 42):
    """Set random seeds for Python, NumPy (if used), and PyTorch for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Note: Full determinism may reduce performance; enable if needed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class Config:
    # Task configuration
    seq_len: int = 100              # Length of sEMG sequence
    in_features: int = 3             # sEMG features per timestep (x, y, z)
    out_dim: int = 5                # Number of fingers to predict

    # GNN Architecture
    model: str = "gcn"              # "gcn" | "sage" | "gat"
    num_gnn_layers: int = 2         # Number of GNN layers
    hid_dim: int = 64               # Hidden dimension
    aggregation: str = "mean"       # "mean" | "max" | "add"
    dropout: float = 0.5

    # Readout layers (after GNN)
    readout_layers: int = 1         # Number of readout MLP layers
    readout_hid_dim: Optional[int] = None  # If None, uses same as hid_dim

    # Training
    lr: float = 1e-2
    weight_decay: float = 5e-4
    epochs: int = 300
    batch_size: int = 64
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Early stopping & Scheduler
    patience: int = 50              # epochs to wait for val improvement before stopping
    lr_patience: int = 20           # epochs with no improvement before LR is reduced
    lr_factor: float = 0.5          # LR reduction factor
    min_lr: float = 1e-5

    # Checkpointing
    checkpoint_dir: str = "ckpts"
    resume: Optional[str] = None    # path to checkpoint to resume from

    # Hyperparameter tuning
    tune_mode: bool = False         # Enable hyperparameter tuning
    n_trials: int = 50              # Number of optimization trials
    tune_timeout: int = 3600        # Timeout for tuning in seconds

    # Misc
    seed: int = 42


# -----------------------------
# Models
# -----------------------------
class GNNBackbone(nn.Module):
    """Generic GNN backbone that can be GCN, SAGE, or GAT."""
    def __init__(self, model_type: str, in_dim: int, hid_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.model_type = model_type.lower()
        self.num_layers = num_layers
        self.dropout = dropout
        self.hid_dim = hid_dim

        # Build GNN layers
        self.convs = nn.ModuleList()
        if self.model_type == "gcn":
            for i in range(num_layers):
                in_channels = in_dim if i == 0 else hid_dim
                self.convs.append(GCNConv(in_channels, hid_dim))
        elif self.model_type == "sage":
            for i in range(num_layers):
                in_channels = in_dim if i == 0 else hid_dim
                self.convs.append(SAGEConv(in_channels, hid_dim))
        elif self.model_type == "gat":
            # GAT with multi-head attention
            # First layer: input -> hid_dim with 8 heads
            heads = 8
            self.convs.append(GATConv(in_dim, hid_dim, heads=heads, dropout=dropout, concat=True))
            # Middle layers: (hid_dim * heads) -> hid_dim with fewer heads
            for i in range(1, num_layers - 1):
                self.convs.append(GATConv(hid_dim * heads, hid_dim, heads=heads, dropout=dropout, concat=True))
            # Last layer: (hid_dim * heads) -> hid_dim with single head (concat=False for mean)
            if num_layers > 1:
                self.convs.append(GATConv(hid_dim * heads, hid_dim, heads=1, dropout=dropout, concat=False))
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            # Apply activation and dropout for all but the last layer
            if i < self.num_layers - 1:
                x = F.elu(x) if self.model_type == "gat" else F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    
    def get_output_dim(self):
        """Return the output dimension of the GNN backbone."""
        return self.hid_dim


class SEMGFingerPredictor(nn.Module):
    """Complete sEMG finger prediction model with GNN backbone and readout."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        # GNN backbone
        self.gnn = GNNBackbone(
            model_type=cfg.model,
            in_dim=cfg.in_features,
            hid_dim=cfg.hid_dim,
            num_layers=cfg.num_gnn_layers,
            dropout=cfg.dropout
        )

        # Readout layers
        readout_dim = cfg.readout_hid_dim if cfg.readout_hid_dim is not None else cfg.hid_dim
        self.readout_layers = nn.ModuleList()

        # First readout layer
        self.readout_layers.append(nn.Linear(cfg.hid_dim, readout_dim))
        if cfg.readout_layers > 1:
            for _ in range(cfg.readout_layers - 1):
                self.readout_layers.append(nn.Linear(readout_dim, readout_dim))

        # Final output layer to 5 fingers with sigmoid
        self.final_layer = nn.Linear(readout_dim, cfg.out_dim)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, in_features) sEMG data
        Returns:
            out: (batch, 5) finger predictions with sigmoid activation
        """
        batch_size, seq_len, in_features = x.shape
        device = x.device

        # Create fully connected graph for each sequence
        # Each timestep is a node, fully connected within each graph
        num_nodes = seq_len
        
        # Generate complete graph edges (undirected)
        # For a complete graph with n nodes: connect all pairs (i,j) where i < j
        edge_index = torch.combinations(torch.arange(num_nodes, device=device), r=2).t()
        # Make undirected by adding reverse edges
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # Create batch of graphs: offset edge indices for each graph in the batch
        edge_index_list = []
        batch_indices_list = []
        for i in range(batch_size):
            # Offset node indices by i * seq_len for the i-th graph
            offset_edge_index = edge_index + i * seq_len
            edge_index_list.append(offset_edge_index)
            # Create batch indices for this graph
            batch_indices_list.append(torch.full((seq_len,), i, dtype=torch.long, device=device))
        
        # Concatenate all edges and batch indices
        edge_index_batched = torch.cat(edge_index_list, dim=1)
        batch_indices = torch.cat(batch_indices_list, dim=0)

        # Flatten batch and seq dimensions for GNN input
        x_flat = x.view(-1, in_features)  # (batch * seq_len, in_features)

        # GNN forward pass
        node_embeddings = self.gnn(x_flat, edge_index_batched)

        # Global pooling to get graph-level representation
        if self.cfg.aggregation == "mean":
            graph_emb = global_mean_pool(node_embeddings, batch_indices)
        elif self.cfg.aggregation == "max":
            graph_emb = global_max_pool(node_embeddings, batch_indices)
        elif self.cfg.aggregation == "add":
            graph_emb = global_add_pool(node_embeddings, batch_indices)
        else:
            raise ValueError(f"Unknown aggregation: {self.cfg.aggregation}")

        # Readout layers
        for layer in self.readout_layers:
            graph_emb = F.relu(layer(graph_emb))
            graph_emb = F.dropout(graph_emb, p=self.cfg.dropout, training=self.training)

        # Final prediction with sigmoid
        out = torch.sigmoid(self.final_layer(graph_emb))
        return out


def build_model(cfg: Config) -> nn.Module:
    """Factory that builds the sEMG finger prediction model."""
    return SEMGFingerPredictor(cfg)


# -----------------------------
# Dataset loading and processing
# -----------------------------
class SEMGDataset(Dataset):
    """Dataset for sEMG finger prediction."""
    def __init__(self, data_path: str, seq_len: int = 100):
        """
        Args:
            data_path: Path to data file (.pt or .npy)
            seq_len: Length of sequences to use
        """
        if data_path.endswith('.pt'):
            data = torch.load(data_path)
        elif data_path.endswith('.npy'):
            import numpy as np
            data = torch.from_numpy(np.load(data_path)).float()
        else:
            raise ValueError("Data file must be .pt or .npy")

        # Assume data format: (num_samples, seq_len, in_features + out_dim)
        # Last out_dim columns are finger labels (same across all timesteps)
        self.sequences = data[:, :seq_len, :-5]  # sEMG features: (num_samples, seq_len, in_features)
        self.labels = data[:, 0, -5:]  # Finger labels: (num_samples, 5) - take from first timestep

        print(f"Loaded dataset: {len(self.sequences)} samples, seq_len={seq_len}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def load_semg_dataset(cfg: Config, data_path: str) -> Tuple[TorchDataLoader, TorchDataLoader, TorchDataLoader]:
    """Load train/val/test datasets for sEMG finger prediction."""
    # This is a placeholder - you'll need to provide actual data paths
    # For now, we'll create dummy data for testing

    # Create dummy data
    num_samples = 1000
    # Generate random sEMG sequences
    train_sequences = torch.randn(num_samples, cfg.seq_len, cfg.in_features)
    val_sequences = torch.randn(200, cfg.seq_len, cfg.in_features)
    test_sequences = torch.randn(200, cfg.seq_len, cfg.in_features)
    
    # Generate random binary finger labels (0 or 1 for each finger)
    train_labels = torch.randint(0, 2, (num_samples, cfg.out_dim)).float()
    val_labels = torch.randint(0, 2, (200, cfg.out_dim)).float()
    test_labels = torch.randint(0, 2, (200, cfg.out_dim)).float()

    # Create datasets
    train_dataset = SEMGDataset.__new__(SEMGDataset)
    train_dataset.sequences = train_sequences
    train_dataset.labels = train_labels

    val_dataset = SEMGDataset.__new__(SEMGDataset)
    val_dataset.sequences = val_sequences
    val_dataset.labels = val_labels

    test_dataset = SEMGDataset.__new__(SEMGDataset)
    test_dataset.sequences = test_sequences
    test_dataset.labels = test_labels

    # Create data loaders
    train_loader = TorchDataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = TorchDataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


# -----------------------------
# Task-specific helpers
# -----------------------------
@torch.no_grad()
def finger_accuracy(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    """
    Compute finger prediction metrics.
    Args:
        predictions: (batch, 5) sigmoid outputs
        targets: (batch, 5) binary targets
        threshold: threshold for binary classification
    Returns:
        dict with accuracy, precision, recall, f1 metrics
    """
    pred_binary = (predictions > threshold).float()

    # Overall accuracy (all fingers correct)
    correct = (pred_binary == targets).all(dim=1).float()
    accuracy = correct.mean().item()

    # Per-finger accuracy
    finger_acc = (pred_binary == targets).float().mean(dim=0)

    # Precision, Recall, F1 (treating as multi-label)
    tp = (pred_binary * targets).sum(dim=0)
    fp = (pred_binary * (1 - targets)).sum(dim=0)
    fn = ((1 - pred_binary) * targets).sum(dim=0)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        "accuracy": accuracy,
        "finger_accuracy": finger_acc.mean().item(),
        "precision": precision.mean().item(),
        "recall": recall.mean().item(),
        "f1": f1.mean().item()
    }


def finger_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy loss for finger prediction."""
    return F.binary_cross_entropy(predictions, targets)


# -----------------------------
# Train / Eval loops
# -----------------------------
def train_one_epoch(model, train_loader, optimizer, device, cfg: Config) -> float:
    """One training epoch for sEMG finger prediction. Returns average loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for sequences, targets in train_loader:
        sequences, targets = sequences.to(device), targets.to(device)

        optimizer.zero_grad()
        predictions = model(sequences)  # (batch, 5)
        loss = finger_loss(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * sequences.size(0)
        total_samples += sequences.size(0)

    return total_loss / max(1, total_samples)


@torch.no_grad()
def evaluate(model, data_loader, device) -> Dict[str, float]:
    """Evaluate sEMG finger prediction model. Returns metrics dict."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    total_samples = 0

    for sequences, targets in data_loader:
        sequences, targets = sequences.to(device), targets.to(device)

        predictions = model(sequences)
        loss = finger_loss(predictions, targets)

        total_loss += float(loss.item()) * sequences.size(0)
        total_samples += sequences.size(0)

        all_predictions.append(predictions.cpu())
        all_targets.append(targets.cpu())

    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    metrics = finger_accuracy(all_predictions, all_targets)
    metrics["loss"] = total_loss / max(1, total_samples)

    return metrics


# -----------------------------
# Checkpointing
# -----------------------------
def save_checkpoint(path: str, model: nn.Module, optimizer, scheduler, cfg: Config,
                    epoch: int, best_val: float, extra_metrics: Dict[str, float]):
    """Save a training checkpoint with model/optimizer/scheduler states and metadata."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "best_val": best_val,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "config": asdict(cfg),
        "metrics": extra_metrics,
        "timestamp": time.time(),
    }
    torch.save(payload, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None, scheduler=None) -> Tuple[int, float, Dict]:
    """Load a checkpoint. If optimizer/scheduler are provided, restore their state."""
    payload = torch.load(path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None and payload.get("optimizer_state") is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    start_epoch = int(payload.get("epoch", 0)) + 1
    best_val = float(payload.get("best_val", float("-inf")))
    return start_epoch, best_val, payload.get("metrics", {})


# -----------------------------
# Hyperparameter tuning
# -----------------------------
def create_config_from_trial(trial: optuna.Trial, base_cfg: Config) -> Config:
    """Create config with hyperparameters suggested by Optuna trial."""
    cfg = Config(**asdict(base_cfg))

    # GNN architecture
    cfg.model = trial.suggest_categorical("model", ["gcn", "sage", "gat"])
    cfg.num_gnn_layers = trial.suggest_int("num_gnn_layers", 1, 4)
    cfg.hid_dim = trial.suggest_categorical("hid_dim", [32, 64, 128, 256])

    # Aggregation
    cfg.aggregation = trial.suggest_categorical("aggregation", ["mean", "max", "add"])

    # Readout
    cfg.readout_layers = trial.suggest_int("readout_layers", 1, 3)
    if cfg.readout_layers > 1:
        cfg.readout_hid_dim = trial.suggest_categorical("readout_hid_dim", [32, 64, 128])

    # Training
    cfg.dropout = trial.suggest_float("dropout", 0.1, 0.7, step=0.1)
    cfg.lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    cfg.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    cfg.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    return cfg


def objective(trial: optuna.Trial, base_cfg: Config) -> float:
    """Objective function for Optuna hyperparameter optimization."""
    # Create config from trial suggestions
    cfg = create_config_from_trial(trial, base_cfg)
    set_seed(cfg.seed)

    # Load data (using dummy data for now)
    train_loader, val_loader, _ = load_semg_dataset(cfg, "dummy_path")

    # Build model
    device = torch.device(cfg.device)
    model = build_model(cfg).to(device)

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=cfg.lr_factor,
                                  patience=cfg.lr_patience, min_lr=cfg.min_lr, verbose=False)

    # Early training with reduced epochs for tuning
    best_val_f1 = 0.0
    patience_counter = 0
    max_epochs = min(50, cfg.epochs)  # Limit epochs for tuning

    for epoch in range(1, max_epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg)

        # Validate
        val_metrics = evaluate(model, val_loader, device)
        val_f1 = val_metrics["f1"]
        scheduler.step(val_f1)

        # Early stopping check
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg.patience // 2:  # More aggressive early stopping for tuning
            break

        # Report intermediate results
        trial.report(val_f1, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_f1


def run_hyperparameter_tuning(cfg: Config):
    """Run Bayesian optimization for hyperparameter tuning."""
    def objective_wrapper(trial):
        return objective(trial, cfg)

    # Create study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=cfg.seed),
        pruner=optuna.pruners.MedianPruner()
    )

    print(f"[Tuning] Starting hyperparameter optimization with {cfg.n_trials} trials...")

    # Run optimization
    study.optimize(objective_wrapper, n_trials=cfg.n_trials, timeout=cfg.tune_timeout)

    # Print results
    print("[Tuning] Best trial:")
    trial = study.best_trial
    print(f"  Value (F1): {trial.value:.4f}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Create best config
    best_cfg = create_config_from_trial(trial, cfg)
    return best_cfg


# -----------------------------
# Main training entry
# -----------------------------
def main(cfg: Config):
    set_seed(cfg.seed)

    # -------------------------
    # Hyperparameter tuning (optional)
    # -------------------------
    if cfg.tune_mode:
        print("[Main] Running hyperparameter tuning...")
        cfg = run_hyperparameter_tuning(cfg)
        print("[Main] Using best hyperparameters for final training.")

    # -------------------------
    # Load sEMG dataset
    # -------------------------
    # TODO: Replace with actual data path
    data_path = "dummy_data.pt"  # You'll need to provide actual data
    try:
        train_loader, val_loader, test_loader = load_semg_dataset(cfg, data_path)
    except FileNotFoundError:
        print(f"[Warning] Data file {data_path} not found. Using synthetic data for demonstration.")
        train_loader, val_loader, test_loader = load_semg_dataset(cfg, "dummy_path")

    device = torch.device(cfg.device)
    model = build_model(cfg).to(device)

    # Optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=cfg.lr_factor,
                                  patience=cfg.lr_patience, min_lr=cfg.min_lr, verbose=True)

    # Optionally resume from checkpoint
    start_epoch = 1
    best_val_f1 = 0.0
    if cfg.resume and os.path.isfile(cfg.resume):
        start_epoch, best_val_f1, _ = load_checkpoint(cfg.resume, model, optimizer, scheduler)
        print(f"[Resume] Loaded checkpoint from {cfg.resume} @ epoch={start_epoch-1}, best_val_f1={best_val_f1:.4f}")

    # -------------------------
    # Training loop with early stopping
    # -------------------------
    patience_counter = 0
    best_ckpt_path = os.path.join(cfg.checkpoint_dir, "best.pt")
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    print(f"[Training] Starting training with config:")
    print(f"  Model: {cfg.model}, Layers: {cfg.num_gnn_layers}, Hidden: {cfg.hid_dim}")
    print(f"  Aggregation: {cfg.aggregation}, Dropout: {cfg.dropout}")
    print(f"  Readout layers: {cfg.readout_layers}, LR: {cfg.lr}, Batch size: {cfg.batch_size}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg)

        # Validate
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["f1"])

        # Check for best model
        is_best = val_metrics["f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics["f1"]
            patience_counter = 0
            save_checkpoint(best_ckpt_path, model, optimizer, scheduler, cfg,
                          epoch, best_val_f1, val_metrics)
        else:
            patience_counter += 1

        # Logging
        if epoch % 10 == 0 or is_best:
            print(f"[Epoch {epoch:03d}] loss={train_loss:.4f} "
                  f"val_loss={val_metrics['loss']:.4f} val_f1={val_metrics['f1']:.4f} "
                  f"val_acc={val_metrics['accuracy']:.4f}")

        # Early stopping
        if patience_counter >= cfg.patience:
            print(f"[EarlyStopping] No val improvement for {cfg.patience} epochs. Stop.")
            break

    # -------------------------
    # Final evaluation with best model
    # -------------------------
    print("[Final] Loading best model and evaluating on test set...")
    load_checkpoint(best_ckpt_path, model)
    test_metrics = evaluate(model, test_loader, device)

    print("[Final Results]")
    print(f"  Test Loss: {test_metrics['loss']:.4f}")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test Finger Accuracy: {test_metrics['finger_accuracy']:.4f}")
    print(f"  Test Precision: {test_metrics['precision']:.4f}")
    print(f"  Test Recall: {test_metrics['recall']:.4f}")
    print(f"  Test F1: {test_metrics['f1']:.4f}")

    print(f"[Complete] Best model saved to {best_ckpt_path}")
    return test_metrics


# -----------------------------
# CLI
# -----------------------------
def build_argparser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="sEMG Finger Prediction GNN Pipeline with hyperparameter tuning.")

    # Data configuration
    p.add_argument("--seq_len", type=int, default=100, help="Length of sEMG sequences.")
    p.add_argument("--in_features", type=int, default=3, help="Number of features per timestep (x,y,z).")

    # Model architecture
    p.add_argument("--model", type=str, default="gcn", choices=["gcn", "sage", "gat"],
                   help="GNN backbone type.")
    p.add_argument("--num_gnn_layers", type=int, default=2, help="Number of GNN layers.")
    p.add_argument("--hid_dim", type=int, default=64, help="Hidden dimension size.")
    p.add_argument("--aggregation", type=str, default="mean", choices=["mean", "max", "add"],
                   help="Global pooling method for graph aggregation.")
    p.add_argument("--readout_layers", type=int, default=1, help="Number of readout MLP layers.")
    p.add_argument("--readout_hid_dim", type=int, default=None, help="Readout hidden dimension (default: same as hid_dim).")
    p.add_argument("--dropout", type=float, default=0.5, help="Dropout rate.")

    # Training configuration
    p.add_argument("--lr", type=float, default=1e-2, help="Initial learning rate.")
    p.add_argument("--weight_decay", type=float, default=5e-4, help="L2 weight decay.")
    p.add_argument("--epochs", type=int, default=300, help="Maximum training epochs.")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    p.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu).")

    # Early stopping and scheduling
    p.add_argument("--patience", type=int, default=50, help="Early stopping patience.")
    p.add_argument("--lr_patience", type=int, default=20, help="LR scheduler patience.")
    p.add_argument("--lr_factor", type=float, default=0.5, help="LR scheduler reduction factor.")
    p.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate.")

    # Checkpointing
    p.add_argument("--checkpoint_dir", type=str, default="ckpts", help="Directory to save checkpoints.")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")

    # Hyperparameter tuning
    p.add_argument("--tune_mode", action="store_true", help="Enable hyperparameter tuning mode.")
    p.add_argument("--n_trials", type=int, default=50, help="Number of tuning trials.")
    p.add_argument("--tune_timeout", type=int, default=3600, help="Timeout for tuning in seconds.")

    # Misc
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


if __name__ == "__main__":
    args = build_argparser()
    cfg = Config(**vars(args))
    main(cfg)
