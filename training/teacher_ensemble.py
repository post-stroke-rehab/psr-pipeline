from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from training.train import TrainConfig, build_model


@dataclass
class TeacherSpec:
    name: str
    checkpoint_path: str
    weight: float = 1.0


def _load_checkpoint(path: str) -> dict:
    return torch.load(path, map_location="cpu")


def _build_lstm_from_checkpoint(sample_x: torch.Tensor, ckpt: dict) -> nn.Module:
    """Build LSTM teacher by inferring hidden sizes from checkpoint tensors."""
    from models.lstm import LSTM_model, LSTMConfig

    state = ckpt["model_state"]

    hidden1 = state["lstm1.weight_hh_l0"].shape[1]
    hidden2 = state["lstm2.weight_hh_l0"].shape[1]
    fc_hidden = state["fc1.weight"].shape[0]
    out_dim = state["fc2.weight"].shape[0]
    input_size = int(sample_x.size(-1))

    cfg = LSTMConfig(
        input_size=input_size,
        hidden1=hidden1,
        hidden2=hidden2,
        fc_hidden=fc_hidden,
        dropout=0.5,
        out_dim=out_dim,
    )
    return LSTM_model(cfg)


class TeacherEnsemble(nn.Module):
    """
    Training-time-only ensemble for knowledge distillation.

    Input:
        x: adapted feature sequence of shape (batch, seq_len, features)

    Output:
        averaged teacher logits of shape (batch, out_dim)
    """

    def __init__(
        self,
        specs: List[TeacherSpec],
        sample_x: torch.Tensor,
        device: torch.device,
        out_dim: int = 5,
        batch_size: int = 128,
    ):
        super().__init__()

        if not specs:
            raise ValueError("TeacherEnsemble requires at least one TeacherSpec.")

        self.specs = specs
        self.device = device

        teachers: Dict[str, nn.Module] = {}

        for spec in specs:
            name = spec.name.lower().strip()
            ckpt = _load_checkpoint(spec.checkpoint_path)

            if name == "lstm":
                model = _build_lstm_from_checkpoint(sample_x, ckpt)
            else:
                cfg = TrainConfig(
                    model_name=name,
                    batch_size=batch_size,
                    device=str(device),
                    out_dim=out_dim,
                )
                model = build_model(cfg, sample_x)

                # Initialize lazy CNN layers before loading weights.
                with torch.no_grad():
                    _ = model(sample_x.to(device))

            model.load_state_dict(ckpt["model_state"], strict=True)
            model.to(device)
            model.eval()

            for p in model.parameters():
                p.requires_grad = False

            teachers[name] = model

        self.teachers = nn.ModuleDict(teachers)

        weights = torch.tensor([float(spec.weight) for spec in specs], dtype=torch.float32)
        weights = weights / weights.sum()
        self.register_buffer("weights", weights)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = []

        for spec in self.specs:
            name = spec.name.lower().strip()
            teacher = self.teachers[name]
            logits.append(teacher(x))

        stacked = torch.stack(logits, dim=0)  # (num_teachers, batch, out_dim)
        weights = self.weights.view(-1, 1, 1).to(stacked.device)

        return (stacked * weights).sum(dim=0)
