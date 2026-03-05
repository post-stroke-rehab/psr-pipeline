import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# Configuration for the LSTM model to easily adjust hyperparameters
@dataclass
class LSTMConfig:
    input_size: int = 3
    hidden1: int = 64
    hidden2: int = 128
    fc_hidden: int = 64
    dropout: float = 0.5
    out_dim: int = 5

# Architecture for the LSTM 
class LSTM_model(nn.Module):
    def __init__(self, config: LSTMConfig):
        super(LSTM_model, self).__init__()

        self.lstm1 = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden1,
            batch_first=True
        )

        self.lstm2 = nn.LSTM(
            input_size=config.hidden1,
            hidden_size=config.hidden2,
            batch_first=True
        )

        self.fc1 = nn.Linear(config.hidden2, config.fc_hidden)
        self.dropout_fc = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.fc_hidden, config.out_dim)  # 5 output for the fingers

    def forward(self, x):
        # Input validation to ensure correct shape and size
        assert x.ndim == 3, \
            f"LSTM_model expected input of shape (batch, seq_len, input_size), got {tuple(x.shape)}"
        assert x.size(-1) == self.lstm1.input_size, \
            f"Expected input_size={self.lstm1.input_size}, got {x.size(-1)}"

        # x: (batch, seq_len, input_size)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # Keep only last timestep (batch, hidden2)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = self.fc2(x)
        # x = torch.sigmoid(x)  # Sigmoid activation for each finger
        # Removed sigmoid activation for unified pipeline
        return x  # (batch, out_dim)