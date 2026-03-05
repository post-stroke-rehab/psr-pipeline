"""
Save Precision-Recall and ROC curves from curve_data produced by compute_curves.
"""

import os
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
