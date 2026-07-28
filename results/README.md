# Results — Optuna-tuned Pretrained → Finetuned (seed 42)

## Direct baseline (impaired, no pretrain)

| Metric | LSTM | CNN |
|--------|------|-----|
| Finger accuracy | 77.7% | 77.5% |
| F1 macro | 67.2% | 65.1% |
| AUROC macro | 82.2% | 84.0% |
| AUPRC macro | 75.2% | 72.6% |

Per-finger accuracy:

- LSTM: 71.9% / 80.6% / 79.0% / 79.0% / 78.1%
- CNN: 76.8% / 80.4% / 77.1% / 76.5% / 76.7%

Plots: `direct/lstm/`, `direct/cnn/`

## CNN

### Pretrain (healthy test)

| Metric | Value |
|--------|-------|
| Finger accuracy | 80.2% |
| F1 macro | 76.0% |
| AUROC macro | 91.5% |
| AUPRC macro | 84.1% |

Per-finger accuracy: 76.2% / 76.6% / 86.7% / 79.7% / 82.0%

Plots: `best_pretrain/cnn/`

### Finetune (impaired test)

| Metric | Value |
|--------|-------|
| Finger accuracy | 80.3% |
| F1 macro | 67.4% |
| AUROC macro | 87.5% |
| AUPRC macro | 80.8% |

Per-finger accuracy: 73.5% / 86.8% / 81.0% / 80.0% / 80.1%

Plots: `best_finetune/cnn/`

## LSTM

### Pretrain (healthy test)

| Metric | Value |
|--------|-------|
| Finger accuracy | 77.3% |
| F1 macro | 70.6% |
| AUROC macro | 86.3% |
| AUPRC macro | 76.9% |

Per-finger accuracy: 72.7% / 78.1% / 79.3% / 80.1% / 76.6%

Plots: `best_pretrain/lstm/`

### Finetune (impaired test)

| Metric | Value |
|--------|-------|
| Finger accuracy | 77.7% |
| F1 macro | 61.1% |
| AUROC macro | 80.7% |
| AUPRC macro | 70.2% |

Per-finger accuracy: 68.8% / 82.5% / 79.6% / 79.2% / 78.5%

Plots: `best_finetune/lstm/`