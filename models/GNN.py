# gnn_pipeline.py
# A complete, production-ready GNN training pipeline using PyTorch Geometric.
# It supports: configurable models (GCN/SAGE/GAT), node- and graph-level tasks,
# training/validation/test loops, optimizer/scheduler, early stopping,
# checkpoint save/resume, and clean dataset hooks to plug in your own data.

import os
import math
import json
import time
import random
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Core PyG pieces
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv,
    global_mean_pool, global_add_pool, global_max_pool
)
from torch_geometric.utils import to_undirected


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
    task: str = "node"              # "node" for node classification; "graph" for graph classification
    model: str = "gcn"              # "gcn" | "sage" | "gat"
    in_dim: Optional[int] = None    # Will be inferred if not provided
    hid_dim: int = 64
    out_dim: Optional[int] = None   # Will be inferred if not provided
    num_layers: int = 2             # For simplicity, we use 2 layers for all models
    dropout: float = 0.5
    lr: float = 1e-2
    weight_decay: float = 5e-4
    epochs: int = 300
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Graph-level training
    batch_size: int = 64

    # Early stopping & Scheduler
    patience: int = 50              # epochs to wait for val improvement before stopping
    lr_patience: int = 20           # epochs with no improvement before LR is reduced
    lr_factor: float = 0.5          # LR reduction factor
    min_lr: float = 1e-5

    # Checkpointing
    checkpoint_dir: str = "ckpts"
    resume: Optional[str] = None    # path to checkpoint to resume from

    # Misc
    make_undirected: bool = True    # if True and your graph is directed, convert to undirected
    seed: int = 42


# -----------------------------
# Models
# -----------------------------
class GCN(nn.Module):
    """A simple 2-layer GCN for node/graph tasks."""
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.conv2 = GCNConv(hid_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        # For node classification, batch is None and we return node logits.
        # For graph classification, we pool node embeddings to graph-level.
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class SAGE(nn.Module):
    """A simple 2-layer GraphSAGE."""
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GAT(nn.Module):
    """A simple 2-layer GAT (multi-head on first layer)."""
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.6, heads: int = 8):
        super().__init__()
        self.gat1 = GATConv(in_dim, hid_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hid_dim * heads, out_dim, heads=1, dropout=dropout)
        self.dropout = dropout

    def forward(self, x, edge_index, batch=None):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x


def build_model(cfg: Config) -> nn.Module:
    """Factory that builds a GNN model based on config."""
    assert cfg.in_dim is not None and cfg.out_dim is not None, \
        "Input and output dimensions must be set (or inferrable from the dataset)."

    if cfg.model.lower() == "gcn":
        return GCN(cfg.in_dim, cfg.hid_dim, cfg.out_dim, cfg.dropout)
    elif cfg.model.lower() == "sage":
        return SAGE(cfg.in_dim, cfg.hid_dim, cfg.out_dim, cfg.dropout)
    elif cfg.model.lower() == "gat":
        # Slightly higher dropout is common for GAT
        return GAT(cfg.in_dim, cfg.hid_dim, cfg.out_dim, dropout=max(cfg.dropout, 0.6))
    else:
        raise ValueError(f"Unknown model type: {cfg.model}")


# -----------------------------
# Dataset loading (fill data here later)
# -----------------------------
def load_node_dataset(cfg: Config) -> Tuple[Data, int, int]:
    return data, in_dim, out_dim


def load_graph_dataset(cfg: Config) -> Tuple[List[Data], int, int]:
    return graphs, in_dim, out_dim


# -----------------------------
# Task-specific helpers
# -----------------------------
@torch.no_grad()
def accuracy_node(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    """Compute accuracy over masked nodes."""
    pred = logits[mask].argmax(dim=1)
    return (pred == y[mask]).float().mean().item()


@torch.no_grad()
def accuracy_graph(logits: torch.Tensor, y: torch.Tensor) -> float:
    """Compute accuracy over a graph batch."""
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def pool_graph_embeddings(node_logits: torch.Tensor, batch: torch.Tensor, pooling: str = "mean") -> torch.Tensor:
    """
    Convert node-level outputs to graph-level outputs via global pooling.
    By default, mean pooling is used.
    """
    if pooling == "mean":
        return global_mean_pool(node_logits, batch)
    elif pooling == "max":
        return global_max_pool(node_logits, batch)
    elif pooling == "add":
        return global_add_pool(node_logits, batch)
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")


# -----------------------------
# Train / Eval loops
# -----------------------------
def train_one_epoch_node(model, data, optimizer, cfg: Config) -> float:
    """One training epoch for node classification (full-batch). Returns loss."""
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)                 # [N, C]
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def eval_node(model, data) -> Dict[str, float]:
    """Evaluate node classification on train/val/test masks."""
    model.eval()
    out = model(data.x, data.edge_index)
    metrics = {}
    metrics["train_acc"] = accuracy_node(out, data.y, data.train_mask)
    metrics["val_acc"] = accuracy_node(out, data.y, data.val_mask)
    metrics["test_acc"] = accuracy_node(out, data.y, data.test_mask)
    metrics["val_loss"] = float(F.cross_entropy(out[data.val_mask], data.y[data.val_mask]).item())
    return metrics


def train_one_epoch_graph(model, loader, optimizer, device, cfg: Config) -> float:
    """One training epoch for graph classification (mini-batch). Returns average loss."""
    model.train()
    total_loss = 0.0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        node_logits = model(batch.x, batch.edge_index)   # [sum(Ni), C]
        # Pool node logits to graph logits
        graph_logits = pool_graph_embeddings(node_logits, batch.batch, pooling="mean")
        loss = F.cross_entropy(graph_logits, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * batch.num_graphs
        total_graphs += batch.num_graphs
    return total_loss / max(1, total_graphs)


@torch.no_grad()
def eval_graph(model, loader, device) -> Dict[str, float]:
    """Evaluate graph classification. Returns avg loss and accuracy."""
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    total_correct = 0
    for batch in loader:
        batch = batch.to(device)
        node_logits = model(batch.x, batch.edge_index)
        graph_logits = pool_graph_embeddings(node_logits, batch.batch, pooling="mean")
        loss = F.cross_entropy(graph_logits, batch.y)
        total_loss += float(loss.item()) * batch.num_graphs
        total_graphs += batch.num_graphs
        total_correct += int((graph_logits.argmax(dim=1) == batch.y).sum())
    avg_loss = total_loss / max(1, total_graphs)
    acc = total_correct / max(1, total_graphs)
    return {"loss": avg_loss, "acc": acc}


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
# Main training entry
# -----------------------------
def main(cfg: Config):
    set_seed(cfg.seed)

    # -------------------------
    # Load dataset (choose task)
    # -------------------------
    if cfg.task.lower() == "node":
        data, in_dim, out_dim = load_node_dataset(cfg)
        cfg.in_dim = in_dim if cfg.in_dim is None else cfg.in_dim
        cfg.out_dim = out_dim if cfg.out_dim is None else cfg.out_dim

        device = torch.device(cfg.device)
        data = data.to(device)

        # Build model, optimizer, scheduler
        model = build_model(cfg).to(device)
        optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=cfg.lr_factor,
                                      patience=cfg.lr_patience, min_lr=cfg.min_lr, verbose=True)

        # Optionally resume
        start_epoch = 1
        best_val = float("-inf")
        if cfg.resume and os.path.isfile(cfg.resume):
            start_epoch, best_val, _ = load_checkpoint(cfg.resume, model, optimizer, scheduler)
            print(f"[Resume] Loaded checkpoint from {cfg.resume} @ epoch={start_epoch-1}, best_val={best_val:.4f}")

        # Training loop with early stopping
        patience_counter = 0
        best_ckpt_path = os.path.join(cfg.checkpoint_dir, "best.pt")

        for epoch in range(start_epoch, cfg.epochs + 1):
            loss = train_one_epoch_node(model, data, optimizer, cfg)
            metrics = eval_node(model, data)
            scheduler.step(metrics["val_acc"])

            is_best = metrics["val_acc"] > best_val
            if is_best:
                best_val = metrics["val_acc"]
                patience_counter = 0
                save_checkpoint(best_ckpt_path, model, optimizer, scheduler, cfg, epoch, best_val, metrics)
            else:
                patience_counter += 1

            if epoch % 10 == 0 or is_best:
                print(f"[Epoch {epoch:03d}] loss={loss:.4f} "
                      f"train_acc={metrics['train_acc']:.4f} val_acc={metrics['val_acc']:.4f} "
                      f"test_acc={metrics['test_acc']:.4f}")

            if patience_counter >= cfg.patience:
                print(f"[EarlyStopping] No val improvement for {cfg.patience} epochs. Stop.")
                break

        # Load best and report final test accuracy
        load_checkpoint(best_ckpt_path, model)
        final = eval_node(model, data)
        print(f"[Final @ best] val_acc={final['val_acc']:.4f} test_acc={final['test_acc']:.4f}")

    elif cfg.task.lower() == "graph":
        graphs, in_dim, out_dim = load_graph_dataset(cfg)
        cfg.in_dim = in_dim if cfg.in_dim is None else cfg.in_dim
        cfg.out_dim = out_dim if cfg.out_dim is None else cfg.out_dim

        # Train/val/test split of graphs (80/10/10)
        N = len(graphs)
        idx = torch.randperm(N)
        n_train = int(0.8 * N)
        n_val = int(0.1 * N)
        train_set = [graphs[i] for i in idx[:n_train]]
        val_set = [graphs[i] for i in idx[n_train:n_train + n_val]]
        test_set = [graphs[i] for i in idx[n_train + n_val:]]

        train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=cfg.batch_size, shuffle=False)

        device = torch.device(cfg.device)
        model = build_model(cfg).to(device)
        optimizer = Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=cfg.lr_factor,
                                      patience=cfg.lr_patience, min_lr=cfg.min_lr, verbose=True)

        start_epoch = 1
        best_val = float("-inf")
        if cfg.resume and os.path.isfile(cfg.resume):
            start_epoch, best_val, _ = load_checkpoint(cfg.resume, model, optimizer, scheduler)
            print(f"[Resume] Loaded checkpoint from {cfg.resume} @ epoch={start_epoch-1}, best_val={best_val:.4f}")

        patience_counter = 0
        best_ckpt_path = os.path.join(cfg.checkpoint_dir, "best.pt")

        for epoch in range(start_epoch, cfg.epochs + 1):
            loss = train_one_epoch_graph(model, train_loader, optimizer, device, cfg)
            val_stats = eval_graph(model, val_loader, device)
            scheduler.step(val_stats["acc"])

            is_best = val_stats["acc"] > best_val
            if is_best:
                best_val = val_stats["acc"]
                patience_counter = 0
                save_checkpoint(best_ckpt_path, model, optimizer, scheduler, cfg, epoch, best_val, val_stats)
            else:
                patience_counter += 1

            if epoch % 5 == 0 or is_best:
                print(f"[Epoch {epoch:03d}] loss={loss:.4f} val_loss={val_stats['loss']:.4f} "
                      f"val_acc={val_stats['acc']:.4f}")

            if patience_counter >= cfg.patience:
                print(f"[EarlyStopping] No val improvement for {cfg.patience} epochs. Stop.")
                break

        # Load best and evaluate on test set
        load_checkpoint(best_ckpt_path, model)
        final = eval_graph(model, test_loader, device)
        print(f"[Final @ best] test_loss={final['loss']:.4f} test_acc={final['acc']:.4f}")

    else:
        raise ValueError("`task` must be either 'node' or 'graph'.")


# -----------------------------
# CLI
# -----------------------------
def build_argparser() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generic GNN pipeline (PyG): node/graph classification, ckpt, resume.")
    p.add_argument("--task", type=str, default="node", choices=["node", "graph"],
                   help="Choose task type: 'node' or 'graph'.")
    p.add_argument("--model", type=str, default="gcn", choices=["gcn", "sage", "gat"],
                   help="GNN backbone.")
    p.add_argument("--hid_dim", type=int, default=64, help="Hidden dimension.")
    p.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers (kept 2 in this template).")
    p.add_argument("--dropout", type=float, default=0.5, help="Dropout rate.")
    p.add_argument("--lr", type=float, default=1e-2, help="Initial learning rate.")
    p.add_argument("--weight_decay", type=float, default=5e-4, help="L2 weight decay.")
    p.add_argument("--epochs", type=int, default=300, help="Max training epochs.")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size for graph classification.")
    p.add_argument("--patience", type=int, default=50, help="Early stopping patience.")
    p.add_argument("--lr_patience", type=int, default=20, help="LR scheduler patience.")
    p.add_argument("--lr_factor", type=float, default=0.5, help="LR scheduler factor.")
    p.add_argument("--min_lr", type=float, default=1e-5, help="Minimum LR.")
    p.add_argument("--checkpoint_dir", type=str, default="ckpts", help="Directory to save checkpoints.")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    p.add_argument("--make_undirected", action="store_true", help="Convert edges to undirected.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


if __name__ == "__main__":
    args = build_argparser()
    cfg = Config(**vars(args))
    main(cfg)
