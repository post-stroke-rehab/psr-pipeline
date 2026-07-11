# Reviewer-Style Self Review

## 1. Contribution

- **Question:** What new value does this paper provide?
- **Answer:** It upgrades the repo into a documented benchmark paper, consolidates the missing LSTM evidence, and makes the pipeline reproducible as a manuscript artifact.
- **Status:** pass

- **Question:** Is the contribution algorithmically novel?
- **Answer:** No, and the paper does not claim that it is. The contribution is benchmark packaging and evidence-constrained documentation.
- **Status:** pass

## 2. Writing Clarity

- **Question:** Can a technically literate reader reconstruct the workflow from the paper?
- **Answer:** Yes for preprocessing, feature representation, tensor adaptation, model families, and metric outputs; the manuscript intentionally stays close to code-level facts.
- **Status:** pass

- **Question:** Are terms stable and paragraphs single-purpose?
- **Answer:** Yes. Core terms (`preprocessing`, `window`, `feature tensor`, `adapter`, `finger accuracy`) are used consistently.
- **Status:** pass

## 3. Experimental Strength

- **Question:** Are the reported improvements strong enough for a claim of clear method superiority?
- **Answer:** Only partially. LSTM and GNN exchange wins across metrics, so the paper frames the result as a trade-off instead of a decisive victory.
- **Status:** pass after claim weakening

- **Question:** Do we report failure modes honestly?
- **Answer:** Yes. The paper explicitly states the absence of ablations, single-dataset scope, and uneven artifact maturity across branches.
- **Status:** pass

## 4. Evaluation Completeness

- **Question:** Are all key design claims backed by ablations?
- **Answer:** No. The repo does not currently store a full ablation package, so the paper avoids causal claims about individual modules.
- **Status:** needs new experiment

- **Question:** Are all strong baselines covered?
- **Answer:** The current repo compares LSTM, CNN, and GNN, but not a broader external baseline set.
- **Status:** needs new experiment

## 5. Method Design Soundness

- **Question:** Is the method realistic for practical use?
- **Answer:** The preprocessing and benchmark workflow are realistic as an engineering starting point, but clinical deployment claims would need additional validation.
- **Status:** pass with scope limitation

- **Question:** Are there hidden technical risks?
- **Answer:** The dense GNN graph may scale poorly, and transfer-learning evidence is not yet harmonized with the main benchmark tables.
- **Status:** pass after limitation paragraph
