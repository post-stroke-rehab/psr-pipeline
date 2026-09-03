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
- Added the completed five-seed four-channel direct, transfer, and distillation comparison with mean, standard deviation, and confidence-interval artifacts.
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

- Five random seeds and confidence intervals are now reported for the four-channel study; the historical model-family experiments remain single-run results.
- Patient-level cross-validation is acknowledged as future work.
- Raw-signal CNN, 2D spatial-temporal CNN, sparse GNN adjacency, and graph k-nearest-neighbor ablations are acknowledged as future comparisons.
- Distilled-student performance is now evaluated; Raspberry Pi 5 on-device latency remains a future deployment experiment.
