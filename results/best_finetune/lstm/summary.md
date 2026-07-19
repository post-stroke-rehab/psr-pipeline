# GNN Benchmark Results

## Split Method
- **Strategy:** Patient-level 70 / 10 / 20 split (train / val / test)
- **No subject overlap** between splits — patients assigned exclusively to one split
- **Seed:** 42
- **Dataset:** PhysioMio (`formove-ai/physiomio`), finetune stage, data from `datasets/processed/physiomio/impaired`

## Hyperparameters
| Parameter | Value |
|-----------|-------|
| Model | LSTM |
| Training stage | finetune |
| Optimizer | Adam |
| Learning rate | 0.0004048292801624781 |
| Weight decay | 4.40160541310955e-05 |
| Batch size | 128 |
| Max epochs | 200 |
| Early stopping patience | 25 epochs |
| LR scheduler | ReduceLROnPlateau (factor=0.5, patience=10) |
| Min LR | 1e-06 |
| Threshold | [0.4999999999999999, 0.5200000000000002, 0.6200000000000003, 0.5200000000000002, 0.5000000000000002] |
| Device | cpu |

## Final Test Metrics
| Metric | Value |
|--------|-------|
| Accuracy (exact match) | 0.5319 |
| Finger Accuracy | 0.7769 |
| Precision (macro) | 0.7535 |
| Recall (macro) | 0.5161 |
| F1 (macro) | 0.6109 |
| AUROC (macro) | 0.8071 |
| AUPRC (macro) | 0.7022 |
| AUROC (micro) | 0.8003 |
| AUPRC (micro) | 0.7049 |

## Per-Finger Breakdown
| Finger | Accuracy | Precision | Recall | F1 | AUROC | AUPRC |
|--------|----------|-----------|--------|----|-------|-------|
| finger_0 | 0.6875 | 0.6758 | 0.5492 | 0.6060 | 0.7538 | 0.6443 |
| finger_1 | 0.8250 | 0.8194 | 0.5644 | 0.6684 | 0.8534 | 0.7707 |
| finger_2 | 0.7958 | 0.8000 | 0.4622 | 0.5859 | 0.8264 | 0.7139 |
| finger_3 | 0.7917 | 0.7389 | 0.5156 | 0.6073 | 0.8125 | 0.7037 |
| finger_4 | 0.7847 | 0.7333 | 0.4889 | 0.5867 | 0.7894 | 0.6784 |

## Outputs
- `metrics.json` — full test metrics
- `training_curves.json` + `loss_curves.png` — training history
- `pr_curve.png` + `roc_curve.png` — PR and ROC curves
- `confusion_matrices.png` — per-finger confusion matrices
- `checkpoint_best.pt` — best model checkpoint
- `config.json` — full training config
