# Adversarial Self-Review

## Scientific Story

The manuscript begins with the requested LSTM/CNN/GNN comparison and then
evaluates architecture search, healthy-to-impaired transfer, sensor reduction,
transfer initialization, and cross-channel distillation. It reads as a research
study rather than a repository inventory.

## Evidence Strength

Every quantitative value is linked to a committed metric or run summary. The
four-channel table reports five-seed mean and sample standard deviation; the
64-channel source is explicitly labelled as one seed. The older benchmark
families remain single-run results and are not given uncertainty estimates.

## Model Lineage

The legacy CNN, tuned CNN-Base transfer model, optimized 158K-parameter
CNN-Micro, compatible 64-channel `CNNMicroSequence`, and final
123K-parameter four-channel `CNNMicroSequence` are named as distinct
experimental objects. Their metrics are not merged into a single leaderboard.

## Selection And Leakage

Four-channel thresholds are tuned on validation predictions. Distillation is
selected by mean validation macro F1, and seed 4 is selected by validation
macro F1 within that mode. Test performance is reported after selection and is
not described as a selection criterion.

## Claims

The paper claims that distillation improves the tested four-channel means and
that the selected model has an ONNX-compatible interface. It does not claim
statistical significance, universal state of the art, measured Raspberry Pi
latency, therapeutic effectiveness, or clinical readiness.

## Remaining Weaknesses

The strongest outstanding limitations are one patient split, inferred rather
than hardware-validated channel placement, non-causal compatibility
preprocessing, no on-device timing, and no hardware or clinical evaluation.
Raw-signal models, causal preprocessing, sparse graphs, and alternative channel
selection remain useful ablations.
