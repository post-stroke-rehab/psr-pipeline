# Reviewer Response Summary

This artifact records how the manuscript revision addressed the external research review. Implementation-only requests and new-training requests are separated from paper edits.

## Accepted Paper Revisions

- Reframed the manuscript as a research study in post-stroke five-finger intent decoding rather than as a repository or pipeline report.
- Shortened the abstract to a problem, method, result, and deployment-choice structure.
- Moved the novelty statement from Related Work to the Introduction and made it more positive.
- Replaced rhetorical phrasing such as "moves the empirical frontier" with measured scientific language.
- Introduced "subset accuracy" / "exact match ratio" as the standard multilabel terminology.
- Clarified model lineage across direct CNN, optimized CNN students, ResNet teachers, and transfer-learning CNN-Base.
- Added the exact Optuna objective and latency-penalty function from the CNN optimization code.
- Clarified that distillation is implemented but not benchmarked as a completed result in the manuscript.
- Distinguished raw signal notation from feature tensor notation.
- Clarified the 200-sample segment filter versus 200 ms feature windows.
- Clarified Welch periodogram settings, including the effective 256-sample segment length and default 50% overlap.
- Added rationale for handcrafted features as a data-efficient and embedded-friendly representation.
- Added justification and limitations for CNN temporal convolution over the flattened channel-feature representation.
- Added benchmarked GNN details: default two-layer GCN, hidden dimension 64, ReLU, dropout 0.5, dense window graph, and mean pooling.
- Added interpretation for non-monotonic CNN scaling, especially CNN-XLarge underperformance relative to CNN-Large.
- Expanded per-finger F1 interpretation with cautious biomechanical hypotheses.
- Shortened the conclusion and focused the take-home message on CNN-Micro as the embedded deployment choice while retaining LSTM/GNN scientific value.

## Deferred Experimental Revisions

- Multiple random seeds, confidence intervals, and statistical tests are acknowledged as necessary future work but were not added because they require new training runs.
- Patient-level cross-validation is acknowledged as future work.
- Raw-signal CNN, 2D spatial-temporal CNN, sparse GNN adjacency, and graph k-nearest-neighbor ablations are acknowledged as future comparisons.
- Distilled-student performance and Raspberry Pi 5 on-device latency are identified as future deployment experiments.
