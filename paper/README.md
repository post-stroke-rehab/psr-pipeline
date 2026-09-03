# Paper And ArXiv Sources

This directory contains the LaTeX manuscript, evidence artifacts, used figures
and tables, and the compiled preprint.

## Build

From the repository root:

```bash
make -C paper pdf
```

This runs `latexmk` and refreshes `paper/paper.pdf`. TeX Live with
`latexmk`, `pdfLaTeX`, BibTeX, TikZ, and PGFPlots is required.

## ArXiv Upload

```bash
make -C paper arxiv-source
```

Upload `paper/build/arxiv-source.tar.gz` to arXiv. The archive contains:

- `main.tex`;
- all manuscript files under `sections/`;
- all used table and figure sources under `tables/` and `figures/`;
- `references.bib`; and
- the matching generated `main.bbl`.

The archive intentionally excludes `paper.pdf`, this README, writing-process
artifacts, LaTeX logs, auxiliary files, Git metadata, code, checkpoints, and
unused assets. arXiv compiles from the archive root; select PDFLaTeX if the
processor is not detected automatically and inspect the generated PDF before
submitting.

## Contents

- `main.tex`: manuscript entrypoint and PDF metadata
- `sections/`: section-by-section manuscript source
- `figures/`: the pipeline image and PGFPlots figure sources used by the paper
- `tables/`: reusable table fragments used by the paper
- `references.bib`: bibliography database
- `artifacts/`: claim-evidence map, outlines, source notes, and self-review
- `paper.pdf`: compiled preprint matching the committed sources

## Evidence Scope

Quantitative claims are derived from committed metrics under `metrics/`,
`models/CNN/evaluations/`, `results/`, and `experiments/rp5_4ch/`. The
four-channel comparison reports five-seed means and standard deviations; the
older model-family and transfer tables remain single-run estimates.

The ONNX export establishes software compatibility with the selected
`(batch, 9, 48)` input contract. The paper does not report Raspberry Pi
latency, hardware safety, or clinical efficacy.
