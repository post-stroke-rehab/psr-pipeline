# Figure And Table Sources

## Pipeline Overview

- Asset: `figures/pipeline_overview.png`
- Source: user-supplied project diagram, checked against preprocessing,
  training, evaluation, and deployment code
- Use: cropped at inclusion time to remove excess margins

## Direct Model Comparison

- Table: `tables/model_comparison.tex`
- Figure: `figures/model_comparison_plot.tex`
- Data: `metrics/lstm/metrics.json`, `metrics/cnn/metrics.json`,
  `metrics/gnn/test/metrics.json`, and selected optimized CNN/ResNet metrics

## CNN Student And Teacher Sweep

- Table: `tables/cnn_student_sweep.tex`
- Data: `models/CNN/evaluations/summary.json`, per-model metric files, and
  model sizes documented in `models/CNN/README.md`

## Healthy-To-Impaired Transfer

- Table: `tables/transfer_learning.tex`
- Data: `training/tuning/cnn/both_stages_summary.json` and
  `training/tuning/lstm/both_stages_summary.json`

## Four-Channel Comparison

- Table: `tables/four_channel_comparison.tex`
- Figure: `figures/four_channel_comparison_plot.tex`
- Data: `experiments/rp5_4ch/final/five_seed_comparison.json`
- Construction: five-seed means and sample standard deviations transcribed
  from the committed aggregate

## Per-Finger Comparison

- Table: `tables/per_finger_f1.tex`
- Figure: `figures/per_finger_f1_plot.tex`
- Data: per-finger F1 fields in the LSTM, GNN, and CNN-Large metric files

## Literature Context

- Table: `tables/literature_context.tex`
- Sources: primary literature recorded in `literature_search_log.md`
- Project row: updated with the original CNN-Large result and the completed
  four-channel distillation mean
