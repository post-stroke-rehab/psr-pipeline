# Reviewer-Style Self Review

## 1. Contribution

- **Question:** What new value does this paper provide?
- **Answer:** It upgrades the repo into a technically grounded software paper, replaces the stale open-PR framing with merged evidence, and documents the edge-oriented pipeline in a way that another ML researcher can audit.
- **Status:** pass

- **Question:** Is the contribution algorithmically novel?
- **Answer:** No, and the paper does not claim that it is. The contribution is benchmark packaging and evidence-constrained documentation.
- **Status:** pass

- **Question:** Does the paper correctly reflect the updated repo state rather than the older draft's open-PR state?
- **Answer:** Yes. The manuscript now treats the CNN refactor, optimization sweep, and distillation utilities as merged software components while keeping unsupported result claims out.
- **Status:** pass

## 2. Writing Clarity

- **Question:** Can a technically literate reader reconstruct the workflow from the paper?
- **Answer:** Yes for ingestion, preprocessing, feature representation, tensor adaptation, model families, transfer-learning summaries, and deployment-facing utilities; the manuscript stays close to code-level facts.
- **Status:** pass

- **Question:** Are terms stable and paragraphs single-purpose?
- **Answer:** Yes. Core terms (`preprocessing`, `window`, `feature tensor`, `adapter`, `finger accuracy`) are used consistently.
- **Status:** pass

## 3. Experimental Strength

- **Question:** Are the reported improvements strong enough for a claim of clear method superiority?
- **Answer:** Partially. The optimized CNN-Large artifact is clearly strongest among committed results, but the paper still avoids a broad superiority claim because the benchmark scope remains narrow.
- **Status:** pass after claim weakening

- **Question:** Do we report failure modes honestly?
- **Answer:** Yes. The paper explicitly states the absence of ablations, single-dataset scope, missing distilled-student result bundles, and the lack of on-device Raspberry Pi timing evidence.
- **Status:** pass

## 4. Evaluation Completeness

- **Question:** Are all key design claims backed by ablations?
- **Answer:** No. The repo does not currently store a full ablation package, so the paper avoids causal claims about individual modules.
- **Status:** needs new experiment

- **Question:** Are all strong baselines covered?
- **Answer:** The current repo compares LSTM, GNN, legacy CNN, optimized CNN students, and CNN teachers, but not a broader external baseline set.
- **Status:** needs new experiment

## 5. Method Design Soundness

- **Question:** Is the method realistic for practical use?
- **Answer:** The preprocessing and benchmark workflow are realistic as an engineering starting point, and the compact CNN sizes strengthen the embedded-software story, but clinical or on-device deployment claims would need additional validation.
- **Status:** pass with scope limitation

- **Question:** Are there hidden technical risks?
- **Answer:** The dense GNN graph may scale poorly, transfer-learning evidence is not yet harmonized with the main benchmark tables, and the distillation utilities have not yet been mirrored by equally complete committed result packs.
- **Status:** pass after limitation paragraph
