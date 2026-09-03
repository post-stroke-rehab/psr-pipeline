# Reproducibility Guide

## Environment

The committed four-channel runs used Python 3.11.13 and PyTorch 2.13.0 on an
Apple-silicon CPU. A CUDA device can be selected for new runs.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

`requirements.txt` covers preprocessing and training. The development file adds
the test runner plus ONNX export and verification. Raspberry Pi inference uses
`deployment/requirements-rp5.txt`.

## Dataset

Download PhysioMio from Hugging Face:

```bash
python scripts/download_physiomio.py
```

The downloader writes parquet recordings beneath
`datasets/raw/physiomio/`. Raw and processed data are ignored by Git. Follow
the upstream PhysioMio license and access terms.

The original 64-channel benchmark loader was configured at 2,000 Hz. The later
hardware-targeted study follows PhysioMio's documented 2,048 Hz sampling rate.
Do not silently mix these settings when reproducing a reported table.

## Four-Channel Preprocessing

Build left, right, and dual channel-map views using identical patient splits:

```bash
python scripts/run_preprocess.py \
  --four-channel \
  --raw-dir datasets/raw/physiomio \
  --processed-dir datasets/processed/physiomio_rp5_4ch \
  --fs 2048 \
  --arm-split impaired \
  --force
```

The active muscle order is `[ECRB, ECRL, FDS, FDP]`. The ground/reference
electrode is not an input channel. The selected right-arm map uses one-based
PhysioMio channels `[15, 16, 9, 1]`, corresponding to zero-based indices
`[14, 15, 8, 0]`.

## Smoke Tests

The synthetic preprocessing and training checks do not require PhysioMio:

```bash
python scripts/run_preprocess.py \
  --four-channel \
  --synthetic-smoke \
  --processed-dir /tmp/physiomio_4ch_smoke \
  --fs 2048 \
  --force

python scripts/run_rp5_4ch_experiments.py \
  --stage smoke \
  --output-root /tmp/rp5_4ch_smoke_runs \
  --summary-path /tmp/rp5_4ch_smoke_summary.json \
  --device cpu
```

## Full Four-Channel Experiment Sequence

The committed study selected the right map and a nine-window context. The
following commands reproduce the final training stages after preprocessing:

```bash
# Train a compatible 64-channel source/reference model.
python scripts/run_rp5_4ch_experiments.py \
  --stage source --selected-context 9 --device auto

# Five direct four-channel seeds.
python scripts/run_rp5_4ch_experiments.py \
  --stage direct --selected-view right --selected-context 9 \
  --seeds 0 1 2 3 4 --device auto

# Five transfer-initialized seeds. Supply the retained source checkpoint.
python scripts/run_rp5_4ch_experiments.py \
  --stage transfer --selected-view right --selected-context 9 \
  --seeds 0 1 2 3 4 \
  --transfer-checkpoint /path/to/source_full64_checkpoint.pt \
  --device auto

# Five cross-channel distillation seeds.
python scripts/run_rp5_4ch_experiments.py \
  --stage distill --selected-view right --selected-context 9 \
  --seeds 0 1 2 3 4 \
  --teacher-checkpoint results/distill_micro_from_cnn_a0.3_t2.0/checkpoint_best.pt \
  --device auto

python scripts/run_rp5_4ch_experiments.py --stage aggregate
```

Each run stores its configuration, sanitized manifest, epoch history,
validation-selected thresholds, validation metrics, test metrics, and status.
Intermediate run checkpoints are ignored; selected final checkpoints and their
model cards are committed under `experiments/rp5_4ch/final/`.

## ONNX Export And Replay

Export the selected four-channel checkpoint with its fixed nine-window context:

```bash
python models/CNN/export_onnx.py \
  --checkpoint experiments/rp5_4ch/final/cnn_micro_4ch_right_ctx9_distill_seed4.pt \
  --output deployment/artifacts/cnn_micro.onnx \
  --windows 9 \
  --verify
```

Replay a right-arm PhysioMio recording through preprocessing and ONNX inference:

```bash
python scripts/run_rp5_inference.py \
  --model deployment/artifacts/cnn_micro.onnx \
  --input /path/to/impaired-arm-recording.parquet \
  --channels channel_15 channel_16 channel_09 channel_01 \
  --sample-rate 2048 \
  --context-windows 9 \
  --thresholds 0.50,0.70,0.60,0.65,0.60 \
  --unpaced
```

This validates the software path; it is not a Raspberry Pi latency or clinical
validation result.

## Paper And ArXiv Source

```bash
make -C paper pdf
make -C paper arxiv-source
```

The first command compiles and refreshes `paper/paper.pdf`. The second creates
`paper/build/arxiv-source.tar.gz` containing only the TeX entrypoint, section
files, bibliography, generated bibliography, and used figure/table sources.
Inspect arXiv's generated PDF before completing submission.
