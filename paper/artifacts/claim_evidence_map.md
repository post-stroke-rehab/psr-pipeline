# Claim-Evidence Map

## Data And Representation

Claim: The study uses PhysioMio impaired-arm recordings with patient-disjoint
70/10/20 train/validation/test partitions.

Evidence: `datasets/loaders.py`, `datasets/physiomio_four_channel.py`, and
sanitized split metadata under `experiments/rp5_4ch/runs/`.

Claim: The original model-family experiments use the loader's 2,000 Hz
configuration, whereas four-channel retraining uses 2,048 Hz, 410-sample
windows, and a 205-sample stride.

Evidence: `datasets/loaders.py`, `data_processing/channel_selection.py`, and
four-channel run manifests.

Claim: Four active channels are ordered [ECRB, ECRL, FDS, FDP], the final right
map uses one-based channels [15, 16, 9, 1], and ground is excluded.

Evidence: `data_processing/channel_selection.py`,
`experiments/rp5_4ch/final/model_card.md`, and final model configuration.

## Model-Family Study

Claim: The direct baseline comparison has no uniform winner: LSTM leads subset
accuracy and macro AUROC, while GNN leads macro F1 and macro AUPRC.

Evidence: `metrics/lstm/metrics.json`, `metrics/cnn/metrics.json`,
`metrics/gnn/test/metrics.json`, and `paper/tables/model_comparison.tex`.

Claim: CNN-Large is the strongest single-split student in the optimized CNN
sweep, while the 158K-parameter CNN-Micro offers a smaller architecture with
nearby aggregate performance.

Evidence: `models/CNN/evaluations/summary.json`, per-model evaluation JSON,
`models/CNN/README.md`, and `paper/tables/cnn_student_sweep.tex`.

Claim: Healthy-to-impaired transfer improves the tuned CNN-Base metrics but
gives mixed LSTM results.

Evidence: `training/tuning/cnn/both_stages_summary.json`,
`training/tuning/lstm/both_stages_summary.json`, and
`paper/tables/transfer_learning.tex`.

## Four-Channel Study

Claim: Final direct, transfer, and distillation experiments each use five seeds
with the right-map, nine-window, 48-feature input.

Evidence: `experiments/rp5_4ch/aggregate_summary.json`, final comparison JSON,
run configurations, and run manifests.

Claim: Cross-channel distillation gives the strongest five-seed mean metrics
among the four-channel modes: 0.5219 subset accuracy, 0.7612 finger accuracy,
0.6095 macro F1, 0.7904 macro AUROC, and 0.6933 macro AUPRC.

Evidence: `experiments/rp5_4ch/final/five_seed_comparison.json`,
`experiments/rp5_4ch/final/distillation_training_summary.md`, and
`paper/tables/four_channel_comparison.tex`.

Claim: First-layer-sliced transfer does not outperform direct four-channel
training in the committed five-seed comparison.

Evidence: `experiments/rp5_4ch/final/transfer_training_summary.md` and final
comparison JSON.

Claim: The selected distilled seed-4 checkpoint was chosen by validation macro
F1, contains 123,317 parameters, accepts `(batch, 9, 48)`, and emits five
logits.

Evidence: final checkpoint payload, model configuration, model card,
`training/rp5_four_channel.py`, and checkpoint reload verification.

## Deployment Boundary

Claim: The selected four-channel checkpoint has a committed fixed-context ONNX
export and replay-compatible preprocessing interface.

Evidence: `deployment/artifacts/cnn_micro.onnx`,
`models/CNN/export_onnx.py`, deployment tests, and `deployment/README.md`.

Claim: The study does not report Raspberry Pi latency, hardware safety, or
clinical efficacy.

Evidence: no committed device timing artifact or human-subject hardware study;
the manuscript and model card limit their claims to software evaluation.
