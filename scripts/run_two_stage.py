from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.train import (
    HEALTHY_PROCESSED_DIR,
    IMPAIRED_PROCESSED_DIR,
    TrainConfig,
    train_loop,
)


def _ensure_processed(skip_preprocess: bool) -> None:
    healthy_train = Path(HEALTHY_PROCESSED_DIR) / "train.pt"
    impaired_train = Path(IMPAIRED_PROCESSED_DIR) / "train.pt"
    if skip_preprocess or (healthy_train.exists() and impaired_train.exists()):
        return

    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_preprocess.py"), "--separate-arm-dirs"],
        check=True,
        cwd=str(REPO_ROOT),
    )


def _metric_line(metrics: dict) -> str:
    return (
        f"f1_macro={metrics.get('f1_macro', float('nan')):.4f}, "
        f"auprc_macro={metrics.get('auprc_macro', float('nan')):.4f}, "
        f"accuracy={metrics.get('accuracy', float('nan')):.4f}"
    )


def _base_cfg(args: argparse.Namespace) -> TrainConfig:
    cfg = TrainConfig(
        model_name=args.model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
        save_training_curves=True,
        early_stop_metric="finger_accuracy",
        tune_threshold=True,
    )
    if args.model == "lstm":
        cfg.lr = 5e-4
        cfg.batch_size = 64
    elif args.model == "cnn":
        cfg.lr = 1e-3
        cfg.batch_size = 64
        cfg.cnn_variant = "base"
    elif args.model == "gnn":
        cfg.lr = 1e-2
        cfg.batch_size = 64
    return cfg


def _stage_cfg(args: argparse.Namespace, stage: str, **overrides: object) -> TrainConfig:
    fields = _base_cfg(args).__dict__.copy()
    fields.update(overrides)
    fields["training_stage"] = stage
    return TrainConfig(**fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lstm", choices=["lstm", "gnn", "cnn"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=0,
                        help="LSTM finetune only: freeze backbone for this many epochs, then unfreeze.")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda, cpu, or cuda:0. Default: cuda")
    args = parser.parse_args()

    _ensure_processed(args.skip_preprocess)

    print(f"\n=== Pretrain ({args.model}) on healthy ===")
    pretrain_out = train_loop(
        _stage_cfg(args, "pretrain", processed_dir=HEALTHY_PROCESSED_DIR)
    )
    pretrain_ckpt = os.path.join(pretrain_out["results_path"], "checkpoint_best.pt")

    print(f"\n=== Finetune ({args.model}) on impaired ===")
    finetune_out = train_loop(
        _stage_cfg(
            args,
            "finetune",
            processed_dir=IMPAIRED_PROCESSED_DIR,
            pretrained_ckpt_path=pretrain_ckpt,
            freeze_backbone_epochs=args.freeze_backbone_epochs if args.model == "lstm" else 0,
        )
    )

    comparison: dict = {
        "model": args.model,
        "seed": args.seed,
        "pretrain_results": pretrain_out["results_path"],
        "finetune_results": finetune_out["results_path"],
        "finetune_test": finetune_out["test_metrics"],
    }

    if not args.skip_baseline:
        print(f"\n=== Direct baseline ({args.model}) on impaired ===")
        direct_out = train_loop(
            _stage_cfg(args, "direct", processed_dir=IMPAIRED_PROCESSED_DIR)
        )
        comparison["direct_results"] = direct_out["results_path"]
        comparison["direct_test"] = direct_out["test_metrics"]

    out_dir = Path("results") / "comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.model}_seed{args.seed}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\n[Comparison] {out_path}")
    print(f"  finetune: {_metric_line(comparison['finetune_test'])}")
    if not args.skip_baseline:
        print(f"  direct:   {_metric_line(comparison['direct_test'])}")


if __name__ == "__main__":
    main()
