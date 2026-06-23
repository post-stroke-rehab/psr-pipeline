from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from adapters.feature_to_sequence import feature_tensor_to_sequences
from datasets.loaders import LoaderConfig, make_dataloaders
from training.train_distill import build_student, count_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    payload = torch.load(args.checkpoint, map_location=device)
    cfg = payload["config"]

    loader_cfg = LoaderConfig(
        batch_size=args.batch_size,
        num_workers=0,
        seed=int(cfg.get("seed", 42)),
        out_dim=int(cfg.get("out_dim", 5)),
    )

    train_loader, _, _ = make_dataloaders(
        processed_dir=cfg.get("processed_dir", "datasets/processed/physiomio"),
        train_file=cfg.get("train_file", "train.pt"),
        val_file=cfg.get("val_file", "val.pt"),
        test_file=cfg.get("test_file", "test.pt"),
        cfg=loader_cfg,
    )

    x_raw, _ = next(iter(train_loader))
    x = feature_tensor_to_sequences(x_raw.to(device))

    model = build_student(
        student_name=cfg.get("student", "micro"),
        sample_x=x,
        out_dim=int(cfg.get("out_dim", 5)),
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model(x)

        times = []
        for _ in range(args.iters):
            start = time.perf_counter()
            _ = model(x)
            end = time.perf_counter()
            times.append((end - start) * 1000.0)

    times_t = torch.tensor(times)
    result = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "batch_size": args.batch_size,
        "params": count_params(model),
        "mean_ms_per_batch": float(times_t.mean()),
        "median_ms_per_batch": float(times_t.median()),
        "p95_ms_per_batch": float(torch.quantile(times_t, 0.95)),
        "mean_ms_per_sample": float(times_t.mean() / args.batch_size),
        "median_ms_per_sample": float(times_t.median() / args.batch_size),
        "p95_ms_per_sample": float(torch.quantile(times_t, 0.95) / args.batch_size),
    }

    print(json.dumps(result, indent=2))

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = Path(args.checkpoint).parent / f"latency_batch{args.batch_size}.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[Done] Saved latency benchmark to {out_path}")


if __name__ == "__main__":
    main()
