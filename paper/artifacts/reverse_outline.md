# Reverse Outline

## Abstract

- **Paragraph topic:** software research project for post-stroke finger-intent decoding, with baseline families first and improvement paths second.

## Introduction

- **Paragraph 1 topic:** finger-intent decoding is the software bridge between stroke-rehab motivation and assistive hardware.
- **Paragraph 2 topic:** reproducibility requires stable preprocessing, labeling, and evaluation, not just a classifier.
- **Paragraph 3 topic:** the current codebase now supports direct baselines, improved CNNs, transfer learning, and deployment utilities.
- **Paragraph 4 topic:** the manuscript is a research-project paper, not a novel-architecture claim.
- **Paragraph 5 topic:** explicit study contributions and evidence boundaries.

## Related Work

- **Paragraph 1 topic:** rehabilitation value depends on the whole sensing-and-decoding stack.
- **Paragraph 2 topic:** Ninapro and PhysioMio define very different benchmark regimes.
- **Paragraph 3 topic:** prior stroke studies show promise but remain small and heterogeneous.
- **Paragraph 4 topic:** deep models and healthy-dataset results provide context but are not directly comparable.
- **Paragraph 5 topic:** transfer learning is now one of the most relevant directions for stroke-oriented sEMG.
- **Paragraph 6 topic:** distillation and Optuna matter because deployment constraints are part of the project.

## Method

- **Overview paragraph topic:** four-band pipeline with explicit notation.
- **Ingestion paragraph topic:** patient-grouped PhysioMio ingestion and five-finger label mapping.
- **Preprocessing paragraph topic:** Butterworth filtering, wavelet denoising, and overlapping windows.
- **Feature paragraph topic:** twelve handcrafted features define the shared representation.
- **Adapter paragraph topic:** one tensor pathway supports all model families.
- **Predictor paragraph topic:** LSTM, GNN, and CNN equations encode different priors.
- **Training paragraph topic:** multilabel BCE, Optuna, thresholding, latency, and distillation.
- **Transfer paragraph topic:** healthy-to-impaired pretraining/finetuning path.

## Experiments

- **Paragraph 1 topic:** patient-level impaired-arm split.
- **Paragraph 2 topic:** three evidence tiers and historical-versus-new artifact organization.
- **Paragraph 3 topic:** evaluation is intentionally baseline-first, then improvement paths.

## Results

- **Paragraph 1 topic:** direct LSTM/CNN/GNN baselines show complementary strengths.
- **Paragraph 2 topic:** optimized CNN students and ResNet teachers shift the frontier.
- **Paragraph 3 topic:** compact students stay competitive despite small size.
- **Paragraph 4 topic:** transfer learning helps CNN but not LSTM consistently.
- **Paragraph 5 topic:** distillation is implemented but not yet a complete empirical result.
- **Paragraph 6 topic:** finger-level behavior remains architecture-dependent.
- **Paragraph 7 topic:** external literature context should be read carefully because tasks differ.

## Discussion

- **Paragraph 1 topic:** the project now functions as a credible software research framework.
- **Paragraph 2 topic:** the paper's first half remains a genuine three-family comparison.
- **Paragraph 3 topic:** healthy-to-impaired transfer is a real direction, especially for CNNs.
- **Paragraph 4 topic:** deployment-facing utilities exist, but deployment evidence is incomplete.
- **Paragraph 5 topic:** limitations.
- **Paragraph 6 topic:** next milestones.

## Conclusion

- **Paragraph 1 topic:** restate the software research contribution.
- **Paragraph 2 topic:** next evidence should come from distilled-student and Raspberry Pi validation.
