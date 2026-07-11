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
- `figures/`: committed image and LaTeX figure sources used in the manuscript
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
- open pull requests `#56` and `#59` for under-review CNN and distillation extensions
- repository documentation in `README.md`

Claims that would require new experiments, broader baselines, or external clinical validation are softened or explicitly deferred.

## Figure Notes

- `figures/pipeline_overview.png` is the user-supplied pipeline image used in the paper revision dated July 11, 2026.
- The pipeline figure reflects the merged benchmark path plus open-PR extension details discussed in the manuscript; it should not be read as evidence that all depicted deployment blocks are already validated in `main`.
