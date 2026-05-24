# CNN Benchmark Summary

## Dataset

Retrained the CNN using the corrected MEGA processed split files:

- `datasets/processed/physiomio/train.pt`
- `datasets/processed/physiomio/val.pt`
- `datasets/processed/physiomio/test.pt`

Verified tensor structure:

- `X`: `(N, C, W, F)`
- `y`: `(N, 5)`

Verified split sizes:

- Train: 3696 samples
- Validation: 592 samples
- Test: 976 samples
- Total: 5264 samples

Approximate split ratio:

- Train: 70.2%
- Validation: 11.2%
- Test: 18.5%

Labels were verified to contain binary multi-label targets with values `{0, 1}`.

## Model

CNN benchmark using the integrated training pipeline:

`python -m training.train --model cnn --epochs 50 --batch_size 64 --save_training_curves`

## Training Configuration

- Model: CNN base
- Epochs requested: 50
- Early stopping patience: 25
- Actual stopping point: epoch 30
- Batch size: 64
- Learning rate: 0.001
- Weight decay: 0.0001
- Seed: 42
- Loss: BCE with logits
- Class imbalance handling: `pos_weight`
- Device: local CPU

## Final Test Metrics

- Loss: 0.4717
- Exact-match accuracy: 0.5430
- Finger-level accuracy: 0.7750
- Macro precision: 0.6493
- Macro recall: 0.7180
- Macro F1: 0.6817
- Macro AUPRC: 0.7319
- Macro AUROC: 0.8559
- Micro AUPRC: 0.7323
- Micro AUROC: 0.8563

## Per-Finger Test Metrics

| Finger | F1 | Precision | Recall | AUPRC | AUROC |
|---|---:|---:|---:|---:|---:|
| Finger 0 | 0.6984 | 0.6769 | 0.7213 | 0.7166 | 0.8204 |
| Finger 1 | 0.7171 | 0.6657 | 0.7770 | 0.7702 | 0.8807 |
| Finger 2 | 0.6719 | 0.6474 | 0.6984 | 0.7405 | 0.8682 |
| Finger 3 | 0.6803 | 0.6517 | 0.7115 | 0.7332 | 0.8687 |
| Finger 4 | 0.6410 | 0.6047 | 0.6820 | 0.6990 | 0.8416 |

## Artifacts

Saved under `results/cnn/`:

- `metrics.json`
- `summary.md`
- `best_checkpoint.pt`
- `pr_curve.png`
- `roc_curve.png`
- `confusion_matrices.png`
- `loss_curves.png`
- `training_curves.json`

