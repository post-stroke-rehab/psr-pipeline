# Paper Directory

This directory contains an evidence-constrained research paper for the current state of `psr-pipeline` as inspected on July 11, 2026.

## Build

From the repository root:

```bash
(cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error -output-directory=build main.tex)
cp paper/build/main.pdf paper/paper.pdf
```

## Contents

- `main.tex`: manuscript entrypoint
- `sections/`: section-by-section source files
- `figures/`: LaTeX figure sources used in the manuscript
- `tables/`: reusable LaTeX table fragments
- `references.bib`: bibliography
- `artifacts/`: writing-process artifacts derived from the `Research-Paper-Writing-Skills` workflow
- `paper.pdf`: compiled output committed alongside sources

## Evidence Scope

The manuscript intentionally limits itself to evidence already present in this repository:

- preprocessing and tensor pipeline code under `data_processing/`, `adapters/`, and `datasets/`
- model implementations under `models/`
- saved benchmark metrics and plots under `metrics/`
- additional two-stage summaries under `results/README.md`
- repository documentation in `README.md`

Claims that would require new experiments, broader baselines, or external clinical validation are softened or explicitly deferred.
