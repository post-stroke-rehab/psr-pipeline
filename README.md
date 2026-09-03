# Deep Neural Networks for Post-Stroke Finger-Intent Decoding

This research project studies five-finger motor-intent decoding from post-stroke
surface electromyography (sEMG). It combines patient-level PhysioMio data
splits, signal conditioning, handcrafted time- and frequency-domain features,
and recurrent, convolutional, and graph neural networks. The software is
designed with a downstream rehabilitation device in mind: the selected compact
model accepts four active forearm channels and can be exported to ONNX for a
Raspberry Pi 5 software stack.

The accompanying preprint, LaTeX sources, evidence map, and compiled PDF are in
[`paper/`](paper/). The repository contains software research artifacts; it
does not establish clinical efficacy or hardware safety.

## Research Story

The study proceeds in three stages:

1. **Model-family comparison.** LSTM, CNN, and GNN decoders are evaluated on a
   common 64-channel PhysioMio feature representation. The LSTM leads the
   direct baselines on subset accuracy, while the GNN leads on macro F1 and
   macro AUPRC.
2. **Performance improvement.** An Optuna-tuned CNN family and 1D ResNet
   teachers establish the model-size/performance trade-off. Healthy-to-impaired
   transfer improves one CNN-Base experiment but gives mixed LSTM results.
3. **Hardware-targeted retraining.** CNN-Micro is retrained for four active
   channels. Cross-channel knowledge distillation gives the strongest
   five-seed mean performance among direct, transfer-initialized, and distilled
   four-channel students.

The original model-family benchmarks and the later four-channel study use
related but distinct CNN implementations and preprocessing configurations.
Their results are therefore reported separately rather than combined into one
leaderboard.

## Main Results

### Direct 64-channel model-family baselines

| Model | Subset accuracy | Finger accuracy | Macro F1 | Macro AUROC | Macro AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| LSTM | **0.545** | **0.784** | 0.705 | **0.858** | 0.754 |
| CNN legacy | 0.442 | 0.694 | 0.676 | 0.767 | 0.764 |
| GNN | 0.448 | 0.683 | **0.706** | 0.787 | **0.776** |

### Hardware-targeted CNN-Micro study

The four-channel rows report mean +/- standard deviation across five seeds.
The 64-channel source is a separately labelled, single-seed reference trained
on the same patient split.

| Training mode | Subset accuracy | Finger accuracy | Macro F1 | Macro AUROC | Macro AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 64-channel source, seed 42 | 0.5658 | 0.7846 | 0.6810 | 0.8520 | 0.7696 |
| Direct four-channel | 0.4801 +/- 0.0080 | 0.7326 +/- 0.0127 | 0.5822 +/- 0.0162 | 0.7683 +/- 0.0139 | 0.6401 +/- 0.0230 |
| Transfer four-channel | 0.4618 +/- 0.0089 | 0.7111 +/- 0.0078 | 0.5706 +/- 0.0109 | 0.7478 +/- 0.0064 | 0.5985 +/- 0.0164 |
| **Distilled four-channel** | **0.5219 +/- 0.0114** | **0.7612 +/- 0.0038** | **0.6095 +/- 0.0058** | **0.7904 +/- 0.0073** | **0.6933 +/- 0.0036** |

Full metric definitions, provenance, and result-family boundaries are in
[`docs/RESULTS.md`](docs/RESULTS.md).

## Selected Model Contract

The software handoff model is
[`cnn_micro_4ch_right_ctx9_distill_seed4.pt`](experiments/rp5_4ch/final/cnn_micro_4ch_right_ctx9_distill_seed4.pt).
It was selected by validation macro F1, not by test-set ranking.

| Item | Contract |
| --- | --- |
| Active inputs | Four RAW sEMG channels; the ground/reference electrode is excluded |
| Muscle order | `[ECRB, ECRL, FDS, FDP]` |
| Right-map channel IDs | One-based `[15, 16, 9, 1]`; zero-based `[14, 15, 8, 0]` |
| Raw input | `(samples, 4)` at 2,048 Hz |
| Feature tensor | `(batch, 4, windows, 12)` |
| CNN input | `(batch, 9, 48)` |
| Output | Five logits ordered `[thumb, index, middle, ring, little]` |
| Thresholds | `[0.50, 0.70, 0.60, 0.65, 0.60]`, selected on validation data |

The checkpoint SHA-256 is
`3512c065823487dc8b697bb1f0830ef7b0cd61fa5478a7343548a5fe2920d0d8`.
See the complete [model card](experiments/rp5_4ch/final/model_card.md) and
[deployment guide](deployment/README.md).

## Data And Preprocessing

Experiments use the
[PhysioMio dataset](https://huggingface.co/datasets/formove-ai/physiomio), a
bilateral HD-sEMG dataset collected from stroke patients. Raw recordings are
not committed to this repository.

The shared pipeline applies a 20--450 Hz fourth-order zero-phase Butterworth
filter, Symlet-4 wavelet denoising, 200 ms windows with 50% overlap, and 12
features per channel-window pair. Historical 64-channel experiments used the
project loader's 2,000 Hz configuration. The four-channel retraining follows
PhysioMio's documented 2,048 Hz rate, producing 410-sample windows with a
205-sample stride.

```bash
python scripts/download_physiomio.py
python scripts/run_preprocess.py --four-channel --fs 2048 --force
```

Dataset access, split construction, smoke tests, and full experiment commands
are documented in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Installation

Python 3.11 was used for the committed four-channel runs.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Install `requirements-dev.txt` for tests and ONNX export verification. The
Raspberry Pi runtime has a smaller dependency set in
[`deployment/requirements-rp5.txt`](deployment/requirements-rp5.txt).

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Repository Guide

| Path | Purpose |
| --- | --- |
| `data_processing/`, `datasets/`, `adapters/` | Signal conditioning, PhysioMio ingestion, channel selection, and tensor adaptation |
| `models/`, `training/`, `evaluation/` | LSTM/CNN/GNN models, tuning, distillation, and multilabel metrics |
| `metrics/` | Direct LSTM/CNN/GNN benchmark artifacts |
| `models/CNN/evaluations/` | Tuned CNN students and ResNet teacher artifacts |
| `results/` | Healthy-to-impaired transfer-learning artifacts and the legacy teacher checkpoint |
| `experiments/rp5_4ch/` | Four-channel run manifests, aggregate summaries, final checkpoints, and model card |
| `deployment/` | ONNX export/runtime software and replay-based inference path |
| `notebooks/` | Reproduction notebook and labelled exploratory notebooks |
| `paper/` | ArXiv-style LaTeX sources, evidence artifacts, figures, tables, and compiled PDF |

## Paper Build

```bash
make -C paper pdf
make -C paper arxiv-source
```

The second command writes an upload-ready source archive under `paper/build/`;
build products are intentionally ignored except for the committed
`paper/paper.pdf`.

## Citation And License

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Until an arXiv
identifier is assigned, cite the repository and the included preprint title.
The software is released under the [MIT License](LICENSE). Dataset licensing
and access remain governed by PhysioMio's upstream terms.
