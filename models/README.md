# Neural Signal Processing Models

This folder contains deep learning architectures for sEMG signal processing tasks. Each model supports end-to-end training, validation, and inference, with Bayesian hyperparameter optimization.

## Available Models

### Signal Translation
- **Mamba**: State space model for translating hand-to-nose grasping sEMG signals to finger intent sEMG signals
  - Input: `(batch, seq_len, input_dim)` — Hand-to-nose sEMG
  - Output: `(batch, seq_len, output_dim)` — Finger intent sEMG

### Finger Intent Prediction
- **LSTM**: Recurrent architecture for predicting finger intent from sEMG features
- **CNN**: Convolutional network for extracting temporal patterns from sEMG sequences
- **GNN**: Graph neural network for structured sEMG representations
  - Input: `(batch, seq_len, 3)` — sEMG features
  - Output: `(batch, 5)` — Binary predictions per finger (sigmoid activation)

### Dataset
Due to the size of the dataset, it has instead been uploaded to the mega link here:
Download it and put in the correct directory when needed to be used. https://mega.nz/folder/DZcXTQxR#u75Y5nCKy3Z7zPF4oTyURw