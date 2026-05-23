# LSTM Metrics

Best dual-metric LSTM run copied from:

`training/runs/lstm_tune_h128_h256_fc128_d05_lr0p0005_bs64_1779552307`

Checkpoint:

`training/ckpts/lstm_best_dual_metric.pt`

## Test Metrics

- f1_macro: 0.7051
- finger_accuracy: 0.7838
- auroc_macro: 0.8577
- auprc_macro: 0.7537

## Hyperparameters

- hidden1: 128
- hidden2: 256
- fc_hidden: 128
- dropout: 0.5
- lr: 0.0005
- batch_size: 64
