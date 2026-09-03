# Reverse Outline

## Abstract

- Defines post-stroke five-finger intent decoding, summarizes the common
  representation, reports the three-model baseline, and closes with the
  five-seed four-channel distillation result and ONNX-ready model.

## Introduction

- Establishes sEMG intent decoding as the software bridge to rehabilitation
  assistance.
- Explains why paretic multilabel decoding differs from healthy gesture
  classification.
- Frames model choice, model improvement, and sensor reduction as three linked
  research questions.
- States the processing, comparative, transfer, and four-channel contributions.

## Related Work

- Connects sEMG rehabilitation systems to data and evaluation quality.
- Distinguishes Ninapro healthy-subject benchmarks from PhysioMio.
- Reviews stroke-specific decoding, deep architectures, transfer learning,
  distillation, and deployment-oriented optimization.

## Method

- Defines raw and feature notation for both 64- and four-channel inputs.
- Documents patient-level ingestion, signal processing, handcrafted features,
  and shared tensor adaptation.
- Specifies the four-muscle channel maps and exclusion of ground.
- Formalizes LSTM, GNN, CNN, BCE, Optuna, distillation, and both transfer paths.

## Experiments

- Defines the patient split and four experiment families.
- Separates single-run historical benchmarks from the five-seed four-channel
  comparison.
- States validation-only threshold and checkpoint selection.

## Results

- Reports complementary LSTM/GNN baseline behavior.
- Shows the optimized CNN/ResNet size-performance structure.
- Describes architecture-dependent healthy transfer.
- Demonstrates that cross-channel distillation is the strongest reduced-input
  method while the 64-channel source remains stronger.
- Retains per-finger and literature context without collapsing incompatible
  tasks into a leaderboard.

## Discussion

- Interprets the three inductive biases and non-monotonic CNN scaling.
- Explains the different outcomes of healthy transfer, sliced initialization,
  and distillation.
- Frames sensor reduction as a measured performance/implementability trade-off.
- Defines ONNX compatibility, hardware timing, robustness, and clinical
  evaluation boundaries.

## Conclusion

- Identifies the distilled 123K-parameter four-channel CNN-Micro as the
  software handoff and states the next hardware-in-the-loop research step.
