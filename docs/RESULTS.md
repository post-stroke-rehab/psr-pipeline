# Results And Artifact Index

This document maps each reported result family to its committed evidence. The
families were produced at different stages of the research and should not be
merged into a single undifferentiated leaderboard.

## Direct Model-Family Baselines

The LSTM, legacy CNN, and GNN use the shared 64-channel feature representation
and one deterministic patient-level split. These are single-run estimates.

| Model | Subset accuracy | Finger accuracy | Macro F1 | Macro AUROC | Macro AUPRC | Evidence |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| LSTM | 0.545 | 0.784 | 0.705 | 0.858 | 0.754 | `metrics/lstm/metrics.json` |
| CNN legacy | 0.442 | 0.694 | 0.676 | 0.767 | 0.764 | `metrics/cnn/metrics.json` |
| GNN | 0.448 | 0.683 | 0.706 | 0.787 | 0.776 | `metrics/gnn/test/metrics.json` |

ROC, precision-recall, confusion, loss, and training-curve artifacts are stored
beside each metric file under `metrics/`.

## Tuned CNN Students And ResNet Teachers

The Optuna-tuned student family and internal 1D ResNet references are under
`models/CNN/evaluations/`. `summary.json` is the compact index; each model
directory contains its complete metrics and plots. CNN-Large is the strongest
single-split offline student in this family. The 158K-parameter CNN-Micro was
selected as the compact architecture before the later channel-reduction study.

These models are not the same implementation as the four-channel
`CNNMicroSequence` checkpoint. Parameter counts and metrics must be cited with
their experiment family.

## Healthy-To-Impaired Transfer

The two-stage CNN-Base and LSTM studies are summarized in `results/` and
`training/tuning/`. Healthy-arm recordings are used for pretraining and
impaired-arm recordings for finetuning. The CNN-Base metrics improve after
finetuning, while the LSTM improves thresholded accuracy but loses macro F1,
AUROC, and AUPRC.

These single-seed experiments are separate from the four-channel transfer
initialization study below.

## Four-Channel CNN-Micro Study

The hardware-targeted study uses four active channels in the fixed muscle order
`[ECRB, ECRL, FDS, FDP]`, 2,048 Hz preprocessing, 200 ms windows, 50% overlap,
and nine-window model contexts. The full comparison is stored in
`experiments/rp5_4ch/final/five_seed_comparison.json`.

| Training mode | Seeds | Subset accuracy | Finger accuracy | Macro F1 | Macro AUROC | Macro AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64-channel source/reference | 1 | 0.5658 | 0.7846 | 0.6810 | 0.8520 | 0.7696 |
| Direct four-channel | 5 | 0.4801 +/- 0.0080 | 0.7326 +/- 0.0127 | 0.5822 +/- 0.0162 | 0.7683 +/- 0.0139 | 0.6401 +/- 0.0230 |
| Transfer four-channel | 5 | 0.4618 +/- 0.0089 | 0.7111 +/- 0.0078 | 0.5706 +/- 0.0109 | 0.7478 +/- 0.0064 | 0.5985 +/- 0.0164 |
| Distilled four-channel | 5 | 0.5219 +/- 0.0114 | 0.7612 +/- 0.0038 | 0.6095 +/- 0.0058 | 0.7904 +/- 0.0073 | 0.6933 +/- 0.0036 |

Values for the five-seed rows are mean +/- sample standard deviation. The
committed JSON also provides 95% confidence-interval half-widths. No
patient-level significance test was committed, so differences are interpreted
descriptively.

The selected deployment checkpoint is distilled seed 4 because it attained the
highest validation macro F1 among the final distilled runs. It was not selected
by test performance. Its SHA-256 is
`3512c065823487dc8b697bb1f0830ef7b0cd61fa5478a7343548a5fe2920d0d8`.

## Metric Definitions

- **Subset accuracy** is the fraction of samples for which all five binary
  finger labels are correct.
- **Finger accuracy** is binary accuracy averaged over all samples and fingers.
- **Macro F1**, **macro AUROC**, and **macro AUPRC** average the corresponding
  per-finger metrics so each finger has equal weight.
- Four-channel decision thresholds are tuned using validation probabilities
  only and then applied to the held-out test split.
