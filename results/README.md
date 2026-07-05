# Results — Pretrained → Finetuned (seed 42)

## Direct baseline (impaired, no pretrain)

| Metric | LSTM |
|--------|------|
| Finger accuracy | 78.4% |
| F1 macro | 70.5% |
| AUROC macro | 85.8% |
| AUPRC macro | 75.4% |

## CNN

### Pretrain (healthy test)

| Metric | Value |
|--------|-------|
| Val best finger acc | 81.8% |
| Finger accuracy | 76.0% |
| F1 macro | 66.4% |
| AUROC macro | 83.3% |
| AUPRC macro | 69.9% |

Per-finger accuracy: 71.9% / 80.5% / 78.1% / 77.3% / 72.3%

Plots: `best_pretrain/cnn/`

### Finetune (impaired test)

| Metric | Value |
|--------|-------|
| Val best finger acc | 83.2% |
| Finger accuracy | 80.0% |
| F1 macro | 68.0% |
| AUROC macro | 86.3% |
| AUPRC macro | 76.7% |

Per-finger accuracy: 73.2% / 84.3% / 80.8% / 81.1% / 80.4%

Plots: `best_finetune/cnn/`

## LSTM

### Pretrain (healthy test)

| Metric | Value |
|--------|-------|
| Val best finger acc | 85.5% |
| Finger accuracy | 75.2% |
| F1 macro | 67.9% |
| AUROC macro | 85.8% |
| AUPRC macro | 77.3% |

Plots: `best_pretrain/lstm/`

### Finetune (impaired test)

| Metric | Value |
|--------|-------|
| Val best finger acc | 81.9% |
| Finger accuracy | 77.7% |
| F1 macro | 66.0% |
| AUROC macro | 83.0% |
| AUPRC macro | 73.6% |

Plots: `best_finetune/lstm/`
