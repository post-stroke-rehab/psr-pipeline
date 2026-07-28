# GNN Benchmark Results

## Split Method
- **Strategy:** Patient-level 70 / 10 / 20 split (train / val / test)
- **No subject overlap** between splits — patients assigned exclusively to one split
- **Seed:** 42
- **Dataset:** PhysioMio (`formove-ai/physiomio`), pretrain stage, data from `datasets/processed/physiomio/healthy`

## Hyperparameters
| Parameter | Value |
|-----------|-------|
| Model | CNN |
| Training stage | pretrain |
| Optimizer | Adam |
| Learning rate | 0.00024972318836863155 |
| Weight decay | 0.0004192324803088369 |
| Batch size | 32 |
| Max epochs | 200 |
| Early stopping patience | 35 epochs |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=10) |
| Min LR | 1e-06 |
| Threshold | [0.5599999999999998, 0.5600000000000003, 0.6000000000000003, 0.34, 0.6800000000000004] |
| Device | cpu |

## Final Test Metrics
| Metric | Value |
|--------|-------|
| Accuracy (exact match) | 0.6094 |
| Finger Accuracy | 0.8023 |
| Precision (macro) | 0.6486 |
| Recall (macro) | 0.9239 |
| F1 (macro) | 0.7601 |
| AUROC (macro) | 0.9153 |
| AUPRC (macro) | 0.8406 |
| AUROC (micro) | 0.9109 |
| AUPRC (micro) | 0.8294 |

## Per-Finger Breakdown
| Finger | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
|--------|----------|-----------|--------|----|-------|-------|
| finger_0 | 0.7617 | 0.6645 | 0.9196 | 0.7715 | 0.8692 | 0.7957 |
| finger_1 | 0.7656 | 0.5746 | 0.9625 | 0.7196 | 0.9295 | 0.8675 |
| finger_2 | 0.8672 | 0.7347 | 0.9000 | 0.8090 | 0.9333 | 0.8541 |
| finger_3 | 0.7969 | 0.6148 | 0.9375 | 0.7426 | 0.9294 | 0.8566 |
| finger_4 | 0.8203 | 0.6545 | 0.9000 | 0.7579 | 0.9150 | 0.8292 |

## Outputs
- `metrics.json` — full test metrics
- `training_curves.json` + `loss_curves.png` — training history
- `pr_curve.png` + `roc_curve.png` — PR and ROC curves
- `confusion_matrices.png` — per-finger confusion matrices
- `checkpoint_best.pt` — best model checkpoint
- `config.json` — full training config
