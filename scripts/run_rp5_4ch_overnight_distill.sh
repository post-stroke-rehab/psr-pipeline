#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LOG="experiments/rp5_4ch/overnight_distill.log"
STATUS="experiments/rp5_4ch/overnight_status.json"
PYTHON=".venv-rp5/bin/python"
FULL_DIR="datasets/processed/physiomio"
FINAL_DIR="experiments/rp5_4ch/final"

mkdir -p "experiments/rp5_4ch" "$FINAL_DIR"

{
  echo "[$(date -Iseconds)] overnight watcher started"
  echo "{\"status\":\"waiting_for_full_preprocessing\",\"started_at\":\"$(date -Iseconds)\"}" > "$STATUS"

  while true; do
    if [[ -f "$FULL_DIR/train.pt" && -f "$FULL_DIR/val.pt" && -f "$FULL_DIR/test.pt" ]]; then
      break
    fi
    echo "[$(date -Iseconds)] waiting for $FULL_DIR/{train,val,test}.pt"
    sleep 300
  done

  echo "[$(date -Iseconds)] full preprocessing detected; launching distillation"
  echo "{\"status\":\"running_distillation\",\"started_at\":\"$(date -Iseconds)\"}" > "$STATUS"

  "$PYTHON" -u scripts/run_rp5_4ch_experiments.py \
    --stage distill \
    --selected-view right \
    --selected-context 9 \
    --seeds 0 1 2 3 4 \
    --epochs 60 \
    --batch-size 128 \
    --device cpu

  echo "[$(date -Iseconds)] distillation finished; aggregating"
  "$PYTHON" -u scripts/run_rp5_4ch_experiments.py --stage aggregate

  "$PYTHON" - <<'PY'
import json
import math
import shutil
from pathlib import Path

import numpy as np

root = Path("experiments/rp5_4ch/runs")
final = Path("experiments/rp5_4ch/final")
final.mkdir(parents=True, exist_ok=True)

runs = []
for seed in range(5):
    run = root / f"distill_right_ctx9_seed{seed}"
    status_path = run / "status.json"
    if not status_path.exists():
        continue
    status = json.loads(status_path.read_text())
    test = json.loads((run / "test_metrics.json").read_text())
    runs.append(
        {
            "seed": seed,
            "run_dir": str(run),
            "best_val_f1": float(status["best_val_f1"]),
            "checkpoint_sha256": status["checkpoint_sha256"],
            "test_metrics": test,
        }
    )

if not runs:
    raise SystemExit("No distillation runs found to finalize.")

best = max(runs, key=lambda row: row["best_val_f1"])
best_run = Path(best["run_dir"])
best_seed = best["seed"]

shutil.copy2(best_run / "checkpoint_best.pt", final / f"cnn_micro_4ch_right_ctx9_distill_seed{best_seed}.pt")
shutil.copy2(best_run / "model_config.json", final / "cnn_micro_4ch_right_ctx9_distill_model_config.json")
shutil.copy2(best_run / "thresholds.json", final / "cnn_micro_4ch_right_ctx9_distill_thresholds.json")

metric_keys = [
    ("subset_accuracy", "accuracy"),
    ("finger_accuracy", "finger_accuracy"),
    ("macro_f1", "f1_macro"),
    ("macro_auprc", "auprc_macro"),
    ("macro_auroc", "auroc_macro"),
]

summary = {}
for out_key, metric_key in metric_keys:
    vals = np.array([row["test_metrics"][metric_key] for row in runs], dtype=float)
    sd = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    summary[out_key] = {
        "mean": float(vals.mean()),
        "std": sd,
        "ci95": float(1.96 * sd / math.sqrt(len(vals))) if len(vals) > 1 else 0.0,
        "n": len(vals),
    }

payload = {
    "selected_run": best,
    "runs": runs,
    "summary": summary,
}
(final / "distillation_training_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=True))
PY

  echo "{\"status\":\"completed\",\"completed_at\":\"$(date -Iseconds)\"}" > "$STATUS"
  echo "[$(date -Iseconds)] overnight distillation pipeline completed"
} >> "$LOG" 2>&1
