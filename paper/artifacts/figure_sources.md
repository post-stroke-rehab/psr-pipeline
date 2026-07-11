# Figure and Table Sources

## Figure 1: Pipeline Overview

- Source basis: user-supplied pipeline image committed as `figures/pipeline_overview.png`, cross-checked against `README.md`, `datasets/loaders.py`, `data_processing/preprocess_config.py`, `adapters/feature_to_sequence.py`, `training/train.py`, and open PRs `#56` and `#59`
- Construction: supplied image copied into the paper assets and referenced directly from `sections/method.tex`

## Figure 2 and Table 1: Model Comparison

- Numeric source: `metrics/lstm/metrics.json`, `metrics/gnn/test/metrics.json`, `metrics/cnn/metrics.json`
- Construction: manually transcribed into `tables/model_comparison.tex` and `figures/model_comparison_plot.tex`

## Figure 3 and Table 2: Per-Finger F1

- Numeric source: `per_finger.*.f1` fields in the same three metric JSON files
- Construction: manually transcribed into `tables/per_finger_f1.tex` and `figures/per_finger_f1_plot.tex`

## Textual Claims

- Preprocessing defaults: `data_processing/preprocess_config.py`
- Split policy and PhysioMio loader behavior: `datasets/loaders.py`
- Shared adapter behavior: `adapters/feature_to_sequence.py`
- Model-family construction: `training/train.py`, `models/lstm.py`, `models/gnn.py`, `models/CNN/*`
- Two-stage transfer summary: `scripts/run_two_stage.py`, `results/README.md`
