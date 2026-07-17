from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from adapters.feature_to_sequence import feature_tensor_to_sequences
from datasets.loaders import LoaderConfig, make_dataloaders
from training.train_distill import build_student


def collect_probs(model, loader, device):
    model.eval()
    probs_all = []
    y_all = []

    with torch.no_grad():
        for x_raw, y in loader:
            x = feature_tensor_to_sequences(x_raw.to(device))
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_all.append(probs)
            y_all.append(y.cpu().numpy())

    return np.concatenate(probs_all, axis=0), np.concatenate(y_all, axis=0)


def metrics_at_thresholds(y_true, probs, thresholds):
    y_pred = (probs >= thresholds.reshape(1, -1)).astype(int)

    out = {
        "exact_accuracy": float(np.mean(np.all(y_pred == y_true, axis=1))),
        "finger_accuracy": float(np.mean(y_pred == y_true)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auprc_macro": float(np.mean([
            average_precision_score(y_true[:, i], probs[:, i])
            for i in range(y_true.shape[1])
        ])),
        "auroc_macro": float(np.mean([
            roc_auc_score(y_true[:, i], probs[:, i])
            for i in range(y_true.shape[1])
        ])),
        "per_finger": {},
    }

    for i in range(y_true.shape[1]):
        out["per_finger"][f"finger_{i}"] = {
            "threshold": float(thresholds[i]),
            "precision": float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "recall": float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "f1": float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)),
            "auprc": float(average_precision_score(y_true[:, i], probs[:, i])),
            "auroc": float(roc_auc_score(y_true[:, i], probs[:, i])),
        }

    return out


def tune_thresholds(y_true, probs, grid):
    thresholds = []

    for i in range(y_true.shape[1]):
        best_t = 0.5
        best_f1 = -1.0

        for t in grid:
            pred = (probs[:, i] >= t).astype(int)
            score = f1_score(y_true[:, i], pred, zero_division=0)

            if score > best_f1:
                best_f1 = score
                best_t = float(t)

        thresholds.append(best_t)

    return np.array(thresholds, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    payload = torch.load(ckpt_path, map_location=device)
    cfg = payload["config"]

    loader_cfg = LoaderConfig(
        batch_size=args.batch_size,
        num_workers=0,
        seed=int(cfg.get("seed", 42)),
        out_dim=int(cfg.get("out_dim", 5)),
    )

    train_loader, val_loader, test_loader = make_dataloaders(
        processed_dir=cfg.get("processed_dir", "datasets/processed/physiomio"),
        train_file=cfg.get("train_file", "train.pt"),
        val_file=cfg.get("val_file", "val.pt"),
        test_file=cfg.get("test_file", "test.pt"),
        cfg=loader_cfg,
    )

    x0_raw, _ = next(iter(train_loader))
    x0 = feature_tensor_to_sequences(x0_raw.to(device))

    model = build_student(
        student_name=cfg.get("student", "micro"),
        sample_x=x0,
        out_dim=int(cfg.get("out_dim", 5)),
    ).to(device)
    model.load_state_dict(payload["model_state"])

    val_probs, val_y = collect_probs(model, val_loader, device)
    test_probs, test_y = collect_probs(model, test_loader, device)

    default_thresholds = np.full(test_y.shape[1], 0.5, dtype=np.float32)
    grid = np.arange(0.05, 0.96, 0.01, dtype=np.float32)
    tuned_thresholds = tune_thresholds(val_y, val_probs, grid)

    default_metrics = metrics_at_thresholds(test_y, test_probs, default_thresholds)
    tuned_metrics = metrics_at_thresholds(test_y, test_probs, tuned_thresholds)

    result = {
        "checkpoint": str(ckpt_path),
        "seed": cfg.get("seed"),
        "default_thresholds": default_thresholds.tolist(),
        "tuned_thresholds": tuned_thresholds.tolist(),
        "default_test_metrics": default_metrics,
        "tuned_test_metrics": tuned_metrics,
    }

    print("\n[Thresholds]")
    print("default:", [round(float(x), 3) for x in default_thresholds])
    print("tuned: ", [round(float(x), 3) for x in tuned_thresholds])

    print("\n[Default 0.5 Test]")
    for k, v in default_metrics.items():
        if k != "per_finger":
            print(f"  {k}: {v:.4f}")

    print("\n[Tuned Threshold Test]")
    for k, v in tuned_metrics.items():
        if k != "per_finger":
            print(f"  {k}: {v:.4f}")

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = ckpt_path.parent / "threshold_tuning.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[Done] Saved threshold tuning to {out_path}")


if __name__ == "__main__":
    main()
