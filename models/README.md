# Model Implementations

The maintained five-finger intent decoders are:

- `lstm.py`: recurrent baseline over windowed channel-feature sequences.
- `gnn.py`: GCN, GraphSAGE, and GAT variants over window graphs.
- `CNN/`: compact temporal CNN students, 1D ResNet teachers, Optuna tuning,
  knowledge distillation, evaluations, and ONNX export.

All classifiers produce five logits ordered `[thumb, index, middle, ring,
little]`. The 64-channel model-family code consumes 768 features per window.
The hardware-targeted `CNNMicroSequence` in `training/rp5_four_channel.py`
consumes 48 features per window from four active channels.

`Mamba.ipynb` and `lstm.ipynb` are exploratory notebooks retained for research
provenance; they are not part of the paper's reported benchmark path. Dataset
download and reproduction instructions are maintained in
`docs/REPRODUCIBILITY.md` rather than in external file-sharing links.
