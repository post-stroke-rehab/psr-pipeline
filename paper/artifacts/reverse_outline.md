# Reverse Outline

## Abstract

- **Paragraph 1 topic:** the paper documents a reproducible repository workflow for post-stroke sEMG finger-intent decoding.
- **Support points:** reproducibility problem, pipeline modules, headline benchmark numbers, scope limitation.

## Introduction

- **Paragraph 1 topic:** finger-intent decoding matters for rehabilitation interfaces.
- **Paragraph 2 topic:** the main challenge is full-pipeline reproducibility, not just classifier choice.
- **Paragraph 3 topic:** the repository solves that engineering gap by co-locating the full workflow.
- **Paragraph 4 topic:** the paper's contribution is systems-oriented rather than novelty-oriented.
- **Paragraph 5 topic:** the manuscript contribution list updates the repo benchmark and states scope honestly.

## Related Work

- **Paragraph 1 topic:** sEMG interface work depends on both sensing and decoding.
- **Paragraph 2 topic:** sequence models are a natural baseline family for windowed sEMG.
- **Paragraph 3 topic:** graph models offer a structured alternative, while this paper's distinction is the unified benchmark scaffold.

## Method

- **Overview paragraph topic:** the method is a four-module pipeline ending in five-label prediction.
- **Preprocessing paragraph topics:** band-pass filtering, wavelet denoising, and overlapping windowing.
- **Representation paragraph topic:** 12 handcrafted descriptors define a consistent window-level tensor.
- **Adapter paragraph topic:** one shared sequence representation enables fair cross-model comparison.
- **Predictor paragraph topics:** LSTM, CNN, and GNN each consume the same representation with different inductive biases.
- **Training paragraph topic:** the training code standardizes loss, optimization, checkpointing, and metric reporting.

## Experiments

- **Dataset paragraph topic:** processed PhysioMio tensors are built from contiguous labeled segments.
- **Split paragraph topic:** the benchmark uses patient-level 70/10/20 partitioning and defaults to impaired-arm recordings.
- **Scope paragraph topic:** only evidence-complete artifacts under `metrics/` are used for headline claims.
- **Protocol paragraph topic:** the write-up maps each claim to an existing repository artifact.

## Results

- **Paragraph 1 topic:** the LSTM and GNN trade different strengths across aggregate metrics.
- **Paragraph 2 topic:** grouped metrics show the LSTM leads on exact-match behavior and AUROC.
- **Paragraph 3 topic:** higher GNN recall explains its F1 and AUPRC advantage.
- **Paragraph 4 topic:** per-finger scores show GNN thumb strength and more balanced LSTM performance elsewhere.
- **Paragraph 5 topic:** including the stored LSTM artifacts materially updates the README-level comparison.

## Discussion

- **Paragraph 1 topic:** the repository is already valuable as a reproducible benchmark scaffold.
- **Paragraph 2 topic:** the benchmark suggests complementary strengths for recurrent and graph formulations.
- **Paragraph 3 topic:** two-stage transfer artifacts are promising but currently secondary.
- **Paragraph 4 topic:** the repo still lacks broader empirical coverage and integrated evaluation for all branches.

## Conclusion

- **Paragraph 1 topic:** restate the repository-level contribution and the main benchmark outcome.
- **Paragraph 2 topic:** summarize limits and the next extension path.
