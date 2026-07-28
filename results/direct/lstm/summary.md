# GNN Benchmark Results

## Split Method
- **Strategy:** Patient-level 70 / 10 / 20 split (train / val / test)
- **No subject overlap** between splits — patients assigned exclusively to one split
- **Seed:** 42
- **Dataset:** PhysioMio (`formove-ai/physiomio`), direct stage, data from `datasets/processed/physiomio/impaired`

## Hyperparameters
| Parameter | Value |
|-----------|-------|
| Model | LSTM |
| Training stage | direct |
| Optimizer | Adam |
| Learning rate | 0.00011085122517311723 |
| Weight decay | 8.61257919259488e-06 |
| Batch size | 32 |
| Max epochs | 200 |
| Early stopping patience | 25 epochs |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=10) |
| Min LR | 1e-06 |
| Threshold | [0.27999999999999997, 0.4400000000000001, 0.5200000000000002, 0.46000000000000013, 0.5000000000000002] |
| Device | cpu |

## Final Test Metrics
| Metric | Value |
|--------|-------|
| Accuracy (exact match) | 0.5403 |
| Finger Accuracy | 0.7772 |
| Precision (macro) | 0.6663 |
| Recall (macro) | 0.6775 |
| F1 (macro) | 0.6716 |
| AUROC (macro) | 0.8223 |
| AUPRC (macro) | 0.7519 |
| AUROC (micro) | 0.8264 |
| AUPRC (micro) | 0.7545 |

## Per-Finger Breakdown
| Finger | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
|--------|----------|-----------|--------|----|-------|-------|
| finger_0 | 0.7194 | 0.6728 | 0.6984 | 0.6854 | 0.7661 | 0.6820 |
| finger_1 | 0.8056 | 0.6793 | 0.7156 | 0.6970 | 0.8508 | 0.8002 |
| finger_2 | 0.7903 | 0.6667 | 0.6578 | 0.6622 | 0.8341 | 0.7591 |
| finger_3 | 0.7903 | 0.6581 | 0.6844 | 0.6710 | 0.8385 | 0.7615 |
| finger_4 | 0.7806 | 0.6544 | 0.6311 | 0.6425 | 0.8222 | 0.7568 |

## Outputs
- `metrics.json` — full test metrics
- `training_curves.json` + `loss_curves.png` — training history
- `pr_curve.png` + `roc_curve.png` — PR and ROC curves
- `confusion_matrices.png` — per-finger confusion matrices
- `checkpoint_best.pt` — best model checkpoint
- `config.json` — full training config
