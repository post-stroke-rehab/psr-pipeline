"""
Multilabel (e.g. 5-finger) metrics: precision, recall, F1, AUPRC, AUROC, per-finger and curves.
"""

from typing import Dict, Any, Optional

import numpy as np
import torch

try:
    from sklearn.metrics import (
        precision_score,
        recall_score,
        f1_score,
        average_precision_score,
        roc_auc_score,
        precision_recall_curve,
        roc_curve,
    )
except ImportError:
    raise ImportError("evaluation.metrics requires scikit-learn. Install with: pip install scikit-learn")


def _to_numpy(y: torch.Tensor) -> np.ndarray:
    return y.cpu().numpy() if isinstance(y, torch.Tensor) else np.asarray(y)


def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Per-class AUPRC; returns NaN if class has no positive or no negative."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    try:
        return float(average_precision_score(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Per-class AUROC; returns NaN if class has no positive or no negative."""
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _nanmean(x: np.ndarray) -> float:
    """Mean ignoring NaN."""
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if x.size > 0 else float("nan")


def compute_multilabel_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    *,
    threshold: float = 0.5,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute multilabel metrics (e.g. 5 fingers): accuracy, P/R/F1 macro, AUPRC, AUROC, per-finger.

    Args:
        y_pred: (N, num_classes) continuous predictions (e.g. sigmoid).
        y_true: (N, num_classes) binary labels (0/1).
        threshold: Binarization threshold for precision/recall/F1.
        num_classes: Inferred from shape if None.

    Returns:
        dict with:
          - accuracy, precision_macro, recall_macro, f1_macro
          - auprc_macro, auroc_macro (optional auprc_micro, auroc_micro)
          - per_finger: {"finger_0": {precision, recall, f1, auprc, auroc}, ...}
        All scalar values are float; per-finger values may be nan for constant classes.
    """
    y_pred = _to_numpy(y_pred)
    y_true = _to_numpy(y_true)
    if num_classes is None:
        num_classes = y_true.shape[1]

    pred_binary = (y_pred >= threshold).astype(np.float64)

    # Overall accuracy (all labels correct per sample)
    correct = (pred_binary == y_true).all(axis=1).astype(np.float64)
    accuracy = float(correct.mean())

    # Per-finger binary accuracy (mean over samples per finger, then mean over fingers)
    finger_acc_per = (pred_binary == y_true).astype(np.float64).mean(axis=0)
    finger_accuracy = float(np.mean(finger_acc_per))

    # Macro P/R/F1 via sklearn (zero_division=0 to avoid warnings)
    precision_macro = float(
        precision_score(y_true, pred_binary, average="macro", zero_division=0)
    )
    recall_macro = float(
        recall_score(y_true, pred_binary, average="macro", zero_division=0)
    )
    f1_macro = float(f1_score(y_true, pred_binary, average="macro", zero_division=0))

    # AUPRC / AUROC macro (and optionally micro)
    auprc_per = [_safe_auprc(y_true[:, j], y_pred[:, j]) for j in range(num_classes)]
    auroc_per = [_safe_auroc(y_true[:, j], y_pred[:, j]) for j in range(num_classes)]
    auprc_macro = _nanmean(auprc_per)
    auroc_macro = _nanmean(auroc_per)
    try:
        auprc_micro = float(
            average_precision_score(y_true, y_pred, average="micro")
        )
    except Exception:
        auprc_micro = float("nan")
    try:
        auroc_micro = float(roc_auc_score(y_true, y_pred, average="micro"))
    except Exception:
        auroc_micro = float("nan")

    # Per-finger metrics
    per_finger: Dict[str, Dict[str, float]] = {}
    for j in range(num_classes):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        pb = pred_binary[:, j]
        tp = (pb * yt).sum()
        fp = (pb * (1 - yt)).sum()
        fn = ((1 - pb) * yt).sum()
        prec = float(tp / (tp + fp + 1e-10))
        rec = float(tp / (tp + fn + 1e-10))
        f1_j = float(2 * prec * rec / (prec + rec + 1e-10))
        per_finger[f"finger_{j}"] = {
            "precision": prec,
            "recall": rec,
            "f1": f1_j,
            "auprc": auprc_per[j],
            "auroc": auroc_per[j],
        }

    return {
        "accuracy": accuracy,
        "finger_accuracy": finger_accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "auprc_macro": auprc_macro,
        "auroc_macro": auroc_macro,
        "auprc_micro": auprc_micro,
        "auroc_micro": auroc_micro,
        "per_finger": per_finger,
    }


def compute_curves(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    *,
    num_classes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute Precision-Recall and ROC curve data for multilabel (macro and per-class).

    Args:
        y_pred: (N, num_classes) continuous predictions.
        y_true: (N, num_classes) binary labels.
        num_classes: Inferred from shape if None.

    Returns:
        dict with:
          - pr: { "precision_macro", "recall_macro", "thresholds_pr" (None for macro),
                  "per_finger": [ {"precision", "recall", "thresholds"}, ... ] }
          - roc: { "fpr_macro", "tpr_macro", "thresholds_roc",
                   "per_finger": [ {"fpr", "tpr", "thresholds"}, ... ] }
        Macro curves use micro-averaged P/R and macro-averaged FPR/TPR via sklearn.
    """
    y_pred = _to_numpy(y_pred)
    y_true = _to_numpy(y_true)
    if num_classes is None:
        num_classes = y_true.shape[1]

    # Per-finger PR and ROC
    pr_per = []
    roc_per = []
    for j in range(num_classes):
        prec, rec, th_pr = precision_recall_curve(y_true[:, j], y_pred[:, j])
        fpr, tpr, th_roc = roc_curve(y_true[:, j], y_pred[:, j])
        pr_per.append({"precision": prec, "recall": rec, "thresholds": th_pr})
        roc_per.append({"fpr": fpr, "tpr": tpr, "thresholds": th_roc})

    # Macro PR: average precision-recall across classes (interpolate to common recall grid or use micro)
    # sklearn's PrecisionRecallDisplay.from_predictions with average='macro' uses micro for macro curve
    prec_macro, rec_macro, th_pr_macro = precision_recall_curve(
        y_true.ravel(), y_pred.ravel()
    )
    fpr_macro, tpr_macro, th_roc_macro = roc_curve(
        y_true.ravel(), y_pred.ravel()
    )

    return {
        "pr": {
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "thresholds_pr_macro": th_pr_macro,
            "per_finger": pr_per,
        },
        "roc": {
            "fpr_macro": fpr_macro,
            "tpr_macro": tpr_macro,
            "thresholds_roc_macro": th_roc_macro,
            "per_finger": roc_per,
        },
    }
