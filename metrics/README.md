# Direct Baseline Metrics

This directory contains the committed single-run evaluation artifacts for the
direct LSTM, legacy CNN, and GNN comparison reported in the paper. Each model
directory contains machine-readable metrics and available ROC,
precision-recall, confusion, loss, or training-curve plots.

These baseline artifacts are distinct from:

- tuned CNN students and ResNet teachers in `models/CNN/evaluations/`;
- healthy-to-impaired transfer results in `results/`; and
- five-seed four-channel experiments in `experiments/rp5_4ch/`.

See `docs/RESULTS.md` for the cross-directory index and metric definitions.
