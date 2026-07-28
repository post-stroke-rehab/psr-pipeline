# GNN Benchmark Results

## Split Method
- **Strategy:** Patient-level 70 / 10 / 20 split (train / val / test)
- **No subject overlap** between splits — patients assigned exclusively to one split
- **Seed:** 42
- **Dataset:** PhysioMio (`formove-ai/physiomio`), direct stage, data from `datasets/processed/physiomio/impaired`

## Hyperparameters
| Parameter | Value |
|-----------|-------|
| Model | CNN |
| Training stage | direct |
| Optimizer | Adam |
| Learning rate | 0.00020054282947649357 |
| Weight decay | 1.0072239929233116e-05 |
| Batch size | 128 |
| Max epochs | 200 |
| Early stopping patience | 15 epochs |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=10) |
| Min LR | 1e-06 |
| Threshold | [0.26, 0.3, 0.5400000000000003, 0.4200000000000001, 0.38000000000000006] |
| Device | cpu |

## Final Test Metrics
| Metric | Value |
|--------|-------|
| Accuracy (exact match) | 0.5625 |
| Finger Accuracy | 0.7750 |
| Precision (macro) | 0.6690 |
| Recall (macro) | 0.6376 |
| F1 (macro) | 0.6510 |
| AUROC (macro) | 0.8400 |
| AUPRC (macro) | 0.7257 |
| AUROC (micro) | 0.8387 |
| AUPRC (micro) | 0.7234 |

## Per-Finger Breakdown
| Finger | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
|--------|----------|-----------|--------|----|-------|-------|
| finger_0 | 0.7681 | 0.7483 | 0.7079 | 0.7276 | 0.8336 | 0.7695 |
| finger_1 | 0.8042 | 0.6641 | 0.7556 | 0.7069 | 0.8688 | 0.7555 |
| finger_2 | 0.7708 | 0.6630 | 0.5422 | 0.5966 | 0.8413 | 0.7105 |
| finger_3 | 0.7653 | 0.6373 | 0.5778 | 0.6061 | 0.8350 | 0.7097 |
| finger_4 | 0.7667 | 0.6326 | 0.6044 | 0.6182 | 0.8212 | 0.6832 |

## Outputs
- `metrics.json` — full test metrics
- `training_curves.json` + `loss_curves.png` — training history
- `pr_curve.png` + `roc_curve.png` — PR and ROC curves
- `confusion_matrices.png` — per-finger confusion matrices
- `checkpoint_best.pt` — best model checkpoint
- `config.json` — full training config
