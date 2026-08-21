from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.rp5_four_channel import (
    FourChannelRunConfig,
    summarize_runs,
    write_json,
    run_four_channel_experiment,
)


CONTEXTS = {
    "200ms": 1,
    "500ms": 4,
    "1s": 9,
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def run_logged(cfg: FourChannelRunConfig) -> dict:
    run_dir = Path(cfg.output_root) / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", asdict(cfg))
    with open(run_dir / "console.log", "w") as log_file:
        tee_out = Tee(sys.stdout, log_file)
        tee_err = Tee(sys.stderr, log_file)
        try:
            with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                return run_four_channel_experiment(cfg)
        except Exception as exc:
            write_json(
                run_dir / "status.json",
                {
                    "status": "failed",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            raise


def base_cfg(args: argparse.Namespace, *, run_id: str, mode: str, view: str, seed: int, context_windows: int) -> FourChannelRunConfig:
    return FourChannelRunConfig(
        run_id=run_id,
        mode=mode,
        view=view,
        processed_dir=args.processed_dir,
        full_processed_dir=args.full_processed_dir,
        output_root=args.output_root,
        seed=seed,
        context_windows=context_windows,
        context_stride=args.context_stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        patience=args.patience,
        device=args.device,
        transfer_checkpoint=args.transfer_checkpoint,
        teacher_checkpoint=args.teacher_checkpoint,
        distill_alpha=args.distill_alpha,
        distill_temperature=args.distill_temperature,
        synthetic_smoke=args.synthetic_smoke,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        notes={"stage": args.stage},
    )


def planned_configs(args: argparse.Namespace) -> List[FourChannelRunConfig]:
    configs: List[FourChannelRunConfig] = []

    if args.stage == "smoke":
        configs.append(
            base_cfg(
                args,
                run_id="smoke_synthetic_direct_left_ctx500_seed42",
                mode="direct",
                view="left",
                seed=42,
                context_windows=CONTEXTS["500ms"],
            )
        )
        return configs

    if args.stage in {"view", "all"}:
        for view in ("left", "right", "dual"):
            configs.append(
                base_cfg(
                    args,
                    run_id=f"view_{view}_direct_ctx500_seed42",
                    mode="direct",
                    view=view,
                    seed=42,
                    context_windows=CONTEXTS["500ms"],
                )
            )

    if args.stage in {"context", "all"}:
        for label, context_windows in CONTEXTS.items():
            configs.append(
                base_cfg(
                    args,
                    run_id=f"context_{label}_direct_{args.selected_view}_seed42",
                    mode="direct",
                    view=args.selected_view,
                    seed=42,
                    context_windows=context_windows,
                )
            )

    if args.stage in {"direct", "all"}:
        for seed in args.seeds:
            configs.append(
                base_cfg(
                    args,
                    run_id=f"direct_{args.selected_view}_ctx{args.selected_context}_seed{seed}",
                    mode="direct",
                    view=args.selected_view,
                    seed=seed,
                    context_windows=args.selected_context,
                )
            )

    if args.stage in {"transfer", "all"}:
        for seed in args.seeds:
            configs.append(
                base_cfg(
                    args,
                    run_id=f"transfer_{args.selected_view}_ctx{args.selected_context}_seed{seed}",
                    mode="transfer",
                    view=args.selected_view,
                    seed=seed,
                    context_windows=args.selected_context,
                )
            )

    if args.stage in {"distill", "all"}:
        for seed in args.seeds:
            configs.append(
                base_cfg(
                    args,
                    run_id=f"distill_{args.selected_view}_ctx{args.selected_context}_seed{seed}",
                    mode="distill",
                    view=args.selected_view,
                    seed=seed,
                    context_windows=args.selected_context,
                )
            )

    return configs


def collect_run_dirs(output_root: str | Path) -> Iterable[Path]:
    root = Path(output_root)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RP5 four-channel CNN-Micro experiments.")
    parser.add_argument("--stage", choices=["smoke", "view", "context", "direct", "transfer", "distill", "all", "aggregate"], default="smoke")
    parser.add_argument("--processed-dir", default="datasets/processed/physiomio_rp5_4ch")
    parser.add_argument("--full-processed-dir", default="datasets/processed/physiomio")
    parser.add_argument("--output-root", default="experiments/rp5_4ch/runs")
    parser.add_argument("--selected-view", choices=["left", "right", "dual"], default="dual")
    parser.add_argument("--selected-context", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--context-stride", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--transfer-checkpoint", default="")
    parser.add_argument("--teacher-checkpoint", default="results/distill_micro_from_cnn_a0.3_t2.0/checkpoint_best.pt")
    parser.add_argument("--distill-alpha", type=float, default=0.5)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument("--synthetic-smoke", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--summary-path", default="experiments/rp5_4ch/aggregate_summary.json")
    args = parser.parse_args()

    if args.device == "auto":
        try:
            import torch

            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"

    if args.stage == "smoke":
        args.synthetic_smoke = True
        args.epochs = min(args.epochs, 1)
        args.max_train_batches = args.max_train_batches or 2
        args.max_eval_batches = args.max_eval_batches or 1

    if args.stage == "aggregate":
        summary = summarize_runs(collect_run_dirs(args.output_root), args.summary_path)
        print(json.dumps(summary["groups"], indent=2))
        return

    results = []
    for cfg in planned_configs(args):
        results.append(run_logged(cfg))
    summarize_runs(collect_run_dirs(args.output_root), args.summary_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
