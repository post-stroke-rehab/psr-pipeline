# GNN Benchmark Results

## Split Method
- **Strategy:** Patient-level 70 / 10 / 20 split (train / val / test)
- **No subject overlap** between splits — patients assigned exclusively to one split
- **Seed:** 42
- **Dataset:** PhysioMio (`formove-ai/physiomio`), finetune stage, data from `datasets/processed/physiomio/impaired`

## Hyperparameters
| Parameter | Value |
|-----------|-------|
| Model | CNN |
| Training stage | finetune |
| Optimizer | Adam |
| Learning rate | 0.00013589328909885292 |
| Weight decay | 5.602106202215709e-05 |
| Batch size | 64 |
| Max epochs | 200 |
| Early stopping patience | 30 epochs |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=10) |
| Min LR | 1e-06 |
| Threshold | [0.5599999999999998, 0.6200000000000003, 0.5600000000000003, 0.46000000000000013, 0.5000000000000002] |
| Device | cpu |

## Final Test Metrics
| Metric | Value |
|--------|-------|
| Accuracy (exact match) | 0.6028 |
| Finger Accuracy | 0.8028 |
| Precision (macro) | 0.7722 |
| Recall (macro) | 0.6051 |
| F1 (macro) | 0.6740 |
| AUROC (macro) | 0.8746 |
| AUPRC (macro) | 0.8076 |
| AUROC (micro) | 0.8737 |
| AUPRC (micro) | 0.8033 |

## Per-Finger Breakdown
| Finger | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
|--------|----------|-----------|--------|----|-------|-------|
| finger_0 | 0.7347 | 0.8333 | 0.4921 | 0.6188 | 0.8636 | 0.8157 |
| finger_1 | 0.8681 | 0.8779 | 0.6711 | 0.7607 | 0.9062 | 0.8629 |
| finger_2 | 0.8097 | 0.7529 | 0.5822 | 0.6566 | 0.8653 | 0.7866 |
| finger_3 | 0.8000 | 0.7120 | 0.6044 | 0.6538 | 0.8697 | 0.7835 |
| finger_4 | 0.8014 | 0.6847 | 0.6756 | 0.6801 | 0.8680 | 0.7892 |

## Outputs
- `metrics.json` — full test metrics
- `training_curves.json` + `loss_curves.png` — training history
- `pr_curve.png` + `roc_curve.png` — PR and ROC curves
- `confusion_matrices.png` — per-finger confusion matrices
- `checkpoint_best.pt` — best model checkpoint
- `config.json` — full training config
