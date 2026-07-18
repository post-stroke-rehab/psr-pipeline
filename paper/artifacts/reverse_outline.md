# Reverse Outline

## Abstract

- **Paragraph 1 topic:** the paper documents the software portion of an edge-oriented post-stroke sEMG decoding stack.
- **Support points:** reproducibility problem, pipeline modules, strongest merged CNN results, deployment-oriented utilities, scope limitation.

## Introduction

- **Paragraph 1 topic:** finger-intent decoding matters for post-stroke rehabilitation and embedded assistive systems.
- **Paragraph 2 topic:** the central challenge is reproducible end-to-end software, not just classifier selection.
- **Paragraph 3 topic:** the updated repository now contains a fuller stack including optimized CNNs, transfer summaries, and deployment utilities.
- **Paragraph 4 topic:** the paper is a systems manuscript rather than an algorithmic novelty claim.
- **Paragraph 5 topic:** the manuscript contributions are pipeline documentation, consolidated evidence, deployment-context framing, and honest scope boundaries.

## Related Work

- **Paragraph 1 topic:** stroke-rehabilitation sEMG work links clinical relevance to reliable signal interpretation.
- **Paragraph 2 topic:** gesture-recognition surveys emphasize preprocessing, representation, and real-time constraints.
- **Paragraph 3 topic:** LSTM, CNN, and GNN families provide complementary inductive biases for windowed sEMG.
- **Paragraph 4 topic:** distillation and hyperparameter optimization matter because this paper is about deployable software, not only offline accuracy.

## Method

- **Overview paragraph topic:** the method is a four-band software stack from ingestion through deployment-facing analysis.
- **Ingestion paragraph topics:** label mapping, impaired-arm defaults, patient-level splits, and padded tensor caching define the benchmark.
- **Preprocessing paragraph topics:** band-pass filtering, wavelet denoising, and short overlapping windows are fixed in code.
- **Representation paragraph topic:** 12 handcrafted descriptors create a stable feature tensor with guarded spectral statistics.
- **Adapter paragraph topics:** a shared sequence adapter standardizes cross-model comparison, and the CNN branch reshapes that sequence for Conv1d input.
- **Predictor paragraph topics:** LSTM, GNN, and CNN families express different temporal and structural priors under one upstream pipeline.
- **Training paragraph topic:** loss functions, metrics, Optuna tuning, threshold search, distillation, and latency benchmarking define the practical workflow.
- **Transfer paragraph topic:** healthy-to-impaired staging is an explicit part of the repository's experimental surface.

## Experiments

- **Dataset paragraph topic:** processed PhysioMio tensors are built from contiguous labeled segments.
- **Split paragraph topic:** the benchmark uses patient-level 70/10/20 partitioning and defaults to impaired-arm recordings.
- **Scope paragraph topics:** direct metrics, auxiliary summaries, and code-only utilities are treated as distinct evidence tiers.
- **Protocol paragraph topic:** all claims are tied to committed repository artifacts rather than new unpublished experiments.

## Results

- **Paragraph 1 topic:** the optimized CNN-Large artifact is now the strongest committed benchmark result.
- **Paragraph 2 topic:** the gap from legacy CNN to optimized CNN-Large shows how much the merged CNN stack changed the repository story.
- **Paragraph 3 topic:** the CNN student sweep shows that compact models remain competitive at small INT8 footprints.
- **Paragraph 4 topic:** teacher evaluations show that a tuned student can rival or exceed deeper reference models.
- **Paragraph 5 topic:** transfer-learning evidence helps CNN more clearly than LSTM.
- **Paragraph 6 topic:** per-finger results show that aggregate improvements do not erase finger-specific trade-offs.
- **Paragraph 7 topic:** deployment utilities are present, but their evidence is not yet complete enough for full hardware claims.

## Discussion

- **Paragraph 1 topic:** the repository now qualifies as a coherent software stack rather than a loose experiment collection.
- **Paragraph 2 topic:** optimized CNN students materially shift the benchmark frontier.
- **Paragraph 3 topic:** transfer learning is a credible direction, especially for CNNs.
- **Paragraph 4 topic:** deployment-oriented code exists, but artifact completeness still lags the benchmark core.
- **Paragraph 5 topic:** the remaining limitations define the next empirical milestone for the project.

## Conclusion

- **Paragraph 1 topic:** restate the software-stack contribution and the headline CNN result.
- **Paragraph 2 topic:** state that the next step is better evidence, especially distilled-student and Raspberry Pi validation.
