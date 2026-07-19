# Paper Directory

This directory contains an evidence-constrained arXiv-style paper for the current state of the PSR research project as inspected on July 19, 2026.

## Build

From the repository root:

```bash
(cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory=build main.tex)
cp paper/build/main.pdf paper/paper.pdf
```

## Contents

- `main.tex`: manuscript entrypoint
- `sections/`: section-by-section source files
- `figures/`: committed image and LaTeX figure sources used in the manuscript
- `tables/`: reusable LaTeX table fragments
- `references.bib`: bibliography
- `artifacts/`: writing-process artifacts derived from the `Research-Paper-Writing-Skills` workflow and the relevant K-Dense scientific-writing skills
- `paper.pdf`: compiled output committed alongside sources

## Evidence Scope

The manuscript intentionally limits itself to evidence already present in the current project artifacts:

- preprocessing and tensor pipeline code under `data_processing/`, `adapters/`, and `datasets/`
- model implementations under `models/`
- saved benchmark metrics and plots under `metrics/` and `models/CNN/evaluations/`
- additional two-stage summaries under `training/tuning/` and `results/README.md`
- deployment-facing utilities under `training/train_distill.py`, `scripts/eval_thresholds.py`, and `scripts/benchmark_student_latency.py`
- repository documentation in `README.md`

Claims that would require new experiments, broader baselines, or external clinical validation are softened or explicitly deferred.

## Figure Notes

- `figures/pipeline_overview.png` is the user-supplied pipeline image aligned to the current repo and cropped during LaTeX inclusion.
- The pipeline figure reflects the current software stack and project direction; it should not be read as evidence that all depicted deployment blocks are already validated on hardware.

## Context Notes

- The broader project motivation includes Raspberry Pi 5-served inference for rehabilitation hardware, but this manuscript is intentionally limited to the software portion.
- The previously attached poster was used only as project-context background; no quantitative claims in the manuscript depend on it.
