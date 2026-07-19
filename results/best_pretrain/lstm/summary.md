# GNN Benchmark Results

## Split Method
- **Strategy:** Patient-level 70 / 10 / 20 split (train / val / test)
- **No subject overlap** between splits — patients assigned exclusively to one split
- **Seed:** 42
- **Dataset:** PhysioMio (`formove-ai/physiomio`), pretrain stage, data from `datasets/processed/physiomio/healthy`

## Hyperparameters
| Parameter | Value |
|-----------|-------|
| Model | LSTM |
| Training stage | pretrain |
| Optimizer | Adam |
| Learning rate | 0.00011597192109985082 |
| Weight decay | 1.7889107850235746e-06 |
| Batch size | 64 |
| Max epochs | 200 |
| Early stopping patience | 30 epochs |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=10) |
| Min LR | 1e-06 |
| Threshold | [0.31999999999999995, 0.4400000000000001, 0.4000000000000001, 0.5600000000000003, 0.5400000000000003] |
| Device | cpu |

## Final Test Metrics
| Metric | Value |
|--------|-------|
| Accuracy (exact match) | 0.5156 |
| Finger Accuracy | 0.7734 |
| Precision (macro) | 0.6258 |
| Recall (macro) | 0.8111 |
| F1 (macro) | 0.7055 |
| AUROC (macro) | 0.8629 |
| AUPRC (macro) | 0.7685 |
| AUROC (micro) | 0.8605 |
| AUPRC (micro) | 0.7534 |

## Per-Finger Breakdown
| Finger | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
|--------|----------|-----------|--------|----|-------|-------|
| finger_0 | 0.7266 | 0.6458 | 0.8304 | 0.7266 | 0.7966 | 0.6871 |
| finger_1 | 0.7812 | 0.6111 | 0.8250 | 0.7021 | 0.8915 | 0.7983 |
| finger_2 | 0.7930 | 0.6154 | 0.9000 | 0.7310 | 0.9092 | 0.8435 |
| finger_3 | 0.8008 | 0.6526 | 0.7750 | 0.7086 | 0.8793 | 0.7755 |
| finger_4 | 0.7656 | 0.6042 | 0.7250 | 0.6591 | 0.8380 | 0.7381 |

## Outputs
- `metrics.json` — full test metrics
- `training_curves.json` + `loss_curves.png` — training history
- `pr_curve.png` + `roc_curve.png` — PR and ROC curves
- `confusion_matrices.png` — per-finger confusion matrices
- `checkpoint_best.pt` — best model checkpoint
- `config.json` — full training config
