import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Iterable, Union

_BACKBONE_PREFIXES = ("lstm1.", "lstm2.")

# Configuration for the LSTM model to easily adjust hyperparameters
@dataclass
class LSTMConfig:
    input_size: int = 3
    hidden1: int = 128
    hidden2: int = 256
    fc_hidden: int = 128
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


def backbone_parameters(model: LSTM_model) -> Iterable[nn.Parameter]:
    for name, param in model.named_parameters():
        if name.startswith(_BACKBONE_PREFIXES):
            yield param


def head_parameters(model: LSTM_model) -> Iterable[nn.Parameter]:
    for name, param in model.named_parameters():
        if not name.startswith(_BACKBONE_PREFIXES):
            yield param


def _extract_backbone_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v for k, v in state.items() if k.startswith(_BACKBONE_PREFIXES)}


def load_pretrained_backbone(
    model: LSTM_model,
    ckpt_path_or_state: Union[str, Dict[str, torch.Tensor]],
) -> None:
    if isinstance(ckpt_path_or_state, str):
        payload = torch.load(ckpt_path_or_state, map_location="cpu", weights_only=True)
        state = payload["model_state"]
    else:
        state = ckpt_path_or_state

    backbone_state = _extract_backbone_state(state)
    if not backbone_state:
        raise ValueError("No LSTM backbone weights found in checkpoint.")

    model.load_state_dict(backbone_state, strict=False)


def set_backbone_trainable(model: LSTM_model, trainable: bool) -> None:
    for param in backbone_parameters(model):
        param.requires_grad = trainable