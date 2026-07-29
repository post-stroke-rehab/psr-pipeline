# Figure and Table Sources

## Figure 1: Pipeline Overview

- Source basis: user-supplied pipeline image committed as `figures/pipeline_overview.png`, cross-checked against `README.md`, `datasets/loaders.py`, `data_processing/preprocess_config.py`, `adapters/feature_to_sequence.py`, `models/CNN/README.md`, `training/train_distill.py`, `scripts/eval_thresholds.py`, and `scripts/benchmark_student_latency.py`
- Construction: supplied image committed in the paper assets and cropped at LaTeX include time to emphasize the pipeline content and suppress excess margins

## Table 1: Direct Model Comparison

- Numeric source: `metrics/lstm/metrics.json`, `metrics/gnn/test/metrics.json`, `metrics/cnn/metrics.json`
- Construction: manually transcribed into `tables/model_comparison.tex`

## Figure 2: Aggregate Metric Plot

- Numeric source: `metrics/lstm/metrics.json`, `metrics/gnn/test/metrics.json`, `metrics/cnn/metrics.json`, `models/CNN/evaluations/teacher_resnet152/metrics.json`, `models/CNN/evaluations/optuna_large/metrics.json`
- Construction: manually transcribed into `figures/model_comparison_plot.tex`

## Table 2: CNN Family Comparison

- Numeric source: `models/CNN/README.md`, `models/CNN/evaluations/summary.json`, and per-model metrics under `models/CNN/evaluations/`
- Construction: manually transcribed into `tables/cnn_student_sweep.tex`

## Table 3: Transfer Learning

- Numeric source: `training/tuning/cnn/both_stages_summary.json`, `training/tuning/lstm/both_stages_summary.json`
- Construction: manually transcribed into `tables/transfer_learning.tex`

## Table 4 and Figure 3: Per-Finger F1

- Numeric source: `per_finger.*.f1` fields in `metrics/lstm/metrics.json`, `metrics/gnn/test/metrics.json`, and `models/CNN/evaluations/optuna_large/metrics.json`
- Construction: manually transcribed into `tables/per_finger_f1.tex` and `figures/per_finger_f1_plot.tex`

## Table 5: Literature Context

- Source basis: primary-source literature consulted on July 28, 2026 and logged in `artifacts/literature_search_log.md`
- Construction: manually summarized into `tables/literature_context.tex` with an explicit comparability caveat in the caption

## Textual Claims

- Preprocessing defaults: `data_processing/preprocess_config.py`
- Split policy and PhysioMio loader behavior: `datasets/loaders.py`
- Shared adapter behavior: `adapters/feature_to_sequence.py`
- Model-family construction: `training/train.py`, `models/lstm.py`, `models/gnn.py`, `models/CNN/*`
- CNN distillation and deployment utilities: `training/train_distill.py`, `training/teacher_ensemble.py`, `scripts/eval_thresholds.py`, `scripts/benchmark_student_latency.py`
- Two-stage transfer summary: `training/tuning/cnn/both_stages_summary.json`, `training/tuning/lstm/both_stages_summary.json`, `results/README.md`
