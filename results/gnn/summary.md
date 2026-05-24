# GNN Benchmark Results

## Split Method
- **Strategy:** Patient-level 70 / 10 / 20 split (train / val / test)
- **No subject overlap** between splits — patients assigned exclusively to one split
- **Seed:** 42
- **Dataset:** PhysioMio (`formove-ai/physiomio`), impaired arm recordings

## Hyperparameters
| Parameter | Value |
|-----------|-------|
| Model | GNN |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 128 |
| Max epochs | 200 |
| Early stopping patience | 25 epochs |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=10) |
| Min LR | 1e-06 |
| Threshold | 0.5 |
| Device | cuda |

## Final Test Metrics
| Metric | Value |
|--------|-------|
| Accuracy (exact match) | 0.5738 |
| Finger Accuracy | 0.7719 |
| Precision (macro) | 0.6584 |
| Recall (macro) | 0.6790 |
| F1 (macro) | 0.6672 |
| AUROC (macro) | 0.8367 |
| AUPRC (macro) | 0.7348 |
| AUROC (micro) | 0.8377 |
| AUPRC (micro) | 0.7369 |

## Per-Finger Breakdown
| Finger | Precision | Recall | F1 | AUROC | AUPRC |
|--------|-----------|--------|----|-------|-------|
| finger_0 | 0.6921 | 0.5948 | 0.6398 | 0.7681 | 0.6432 |
| finger_1 | 0.6758 | 0.7311 | 0.7024 | 0.8684 | 0.7891 |
| finger_2 | 0.6626 | 0.7082 | 0.6846 | 0.8588 | 0.7600 |
| finger_3 | 0.6687 | 0.7016 | 0.6848 | 0.8595 | 0.7581 |
| finger_4 | 0.5929 | 0.6590 | 0.6242 | 0.8288 | 0.7238 |

## Outputs
- `metrics.json` — full test metrics
- `training_curves.json` + `loss_curves.png` — training history
- `pr_curve.png` + `roc_curve.png` — PR and ROC curves
- `confusion_matrices.png` — per-finger confusion matrices
- `checkpoint_best.pt` — best model checkpoint
- `config.json` — full training config
