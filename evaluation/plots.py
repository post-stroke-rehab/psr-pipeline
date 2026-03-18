"""
Save Precision-Recall and ROC curves from curve_data produced by compute_curves.
Also loss curves and per-finger confusion matrices.
"""

import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
except ImportError:
    sklearn_confusion_matrix = None


def _to_numpy(x: Any) -> np.ndarray:
    """Convert tensor or array to numpy."""
    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def save_metric_curves(
    curve_data: Dict[str, Any],
    output_dir: str,
    *,
    prefix: str = "",
    save_per_finger: bool = False,
) -> Dict[str, str]:
    """
    Save PR and ROC curves as PNG files.

    Args:
        curve_data: Output from compute_curves (keys "pr", "roc").
        output_dir: Directory to write plot files.
        prefix: Optional filename prefix (e.g. "test_" -> "test_pr_curve.png").
        save_per_finger: If True, also save one PR and one ROC figure with per-finger curves.

    Returns:
        dict with keys "pr_curve", "roc_curve" (and if save_per_finger: "pr_curve_per_finger",
        "roc_curve_per_finger") mapping to saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    pr = curve_data.get("pr", {})
    roc = curve_data.get("roc", {})

    # Macro PR
    prec = pr.get("precision_macro")
    rec = pr.get("recall_macro")
    if prec is not None and rec is not None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.plot(rec, prec)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall (macro)")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        fig.tight_layout()
        p = os.path.join(output_dir, f"{prefix}pr_curve.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths["pr_curve"] = p

    # Macro ROC
    fpr = roc.get("fpr_macro")
    tpr = roc.get("tpr_macro")
    if fpr is not None and tpr is not None:
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        ax.plot(fpr, tpr)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC (macro)")
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        fig.tight_layout()
        p = os.path.join(output_dir, f"{prefix}roc_curve.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths["roc_curve"] = p

    if save_per_finger:
        per_pr = pr.get("per_finger", [])
        per_roc = roc.get("per_finger", [])
        if per_pr and per_roc:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            for j, finger in enumerate(per_pr):
                ax.plot(
                    finger["recall"],
                    finger["precision"],
                    label=f"finger_{j}",
                    alpha=0.8,
                )
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall (per finger)")
            ax.legend(loc="best", fontsize=8)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            fig.tight_layout()
            p = os.path.join(output_dir, f"{prefix}pr_curve_per_finger.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths["pr_curve_per_finger"] = p

            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            for j, finger in enumerate(per_roc):
                ax.plot(
                    finger["fpr"],
                    finger["tpr"],
                    label=f"finger_{j}",
                    alpha=0.8,
                )
            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC (per finger)")
            ax.legend(loc="best", fontsize=8)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            fig.tight_layout()
            p = os.path.join(output_dir, f"{prefix}roc_curve_per_finger.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            paths["roc_curve_per_finger"] = p

    return paths


def save_loss_curves(history: Dict[str, List[float]], output_dir: str) -> str:
    """
    Plot training and validation loss vs epoch and save as loss_curves.png.

    Args:
        history: Dict with keys "epoch", "train_loss", "val_loss" (and optionally "val_f1").
        output_dir: Directory to write the plot file.

    Returns:
        Path to the saved PNG file.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = history.get("epoch", [])
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    if not epochs:
        return ""
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(epochs, train_loss, label="Train loss")
    ax.plot(epochs, val_loss, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation loss")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "loss_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_confusion_matrices(
    probs: Any,
    targets: Any,
    output_dir: str,
    *,
    threshold: float = 0.5,
    num_classes: int = 5,
) -> str:
    """
    Compute per-finger 2x2 confusion matrices and save as one figure.

    Args:
        probs: (N, num_classes) continuous predictions.
        targets: (N, num_classes) binary labels.
        output_dir: Directory to write the plot file.
        threshold: Binarization threshold.
        num_classes: Number of fingers/classes.

    Returns:
        Path to the saved PNG file.
    """
    if sklearn_confusion_matrix is None:
        return ""
    os.makedirs(output_dir, exist_ok=True)
    probs_np = _to_numpy(probs)
    targets_np = _to_numpy(targets)
    pred_binary = (probs_np >= threshold).astype(np.int32)

    ncols = min(5, num_classes)
    nrows = (num_classes + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))
    if num_classes == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for j in range(num_classes):
        ax = axes[j]
        y_true_j = targets_np[:, j].astype(np.int32)
        y_pred_j = pred_binary[:, j]
        cm = sklearn_confusion_matrix(y_true_j, y_pred_j, labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"])
        ax.set_yticklabels(["Neg", "Pos"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Finger {j}")
        for ri in range(2):
            for ci in range(2):
                ax.text(ci, ri, str(cm[ri, ci]), ha="center", va="center", color="black")

    for j in range(num_classes, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()
    path = os.path.join(output_dir, "confusion_matrices.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
