"""Export the distilled AdaptiveCNNStudent checkpoint to ONNX for Raspberry Pi 5.

The PR #71 deployment checkpoint stores ``model_state`` from
``training.train_distill.AdaptiveCNNStudent``.  This exporter reconstructs the
architecture directly from checkpoint tensor shapes, exports model input as
``(batch, windows, features)``, and optionally verifies ONNX Runtime parity.

Example:
    python models/CNN/export_onnx.py \
        --checkpoint results/distill_micro_from_cnn_a0.3_t2.0/checkpoint_best.pt \
        --output deployment/artifacts/cnn_micro.onnx \
        --verify
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveCNNStudent(nn.Module):
    """Deployment reconstruction of training.train_distill.AdaptiveCNNStudent."""

    def __init__(
        self,
        in_features: int,
        out_dim: int = 5,
        width: int = 20,
        fc_hidden: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(width, width, kernel_size=5, padding=2)
        self.bn = nn.BatchNorm1d(width)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(width, fc_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Training/model-facing layout is (N, W, F).
        if x.dim() != 3:
            raise ValueError(f"Expected (N,W,F), got {tuple(x.shape)}")
        x = x.permute(0, 2, 1).contiguous()
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def load_checkpoint_model(path: Path) -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "model_state" not in payload:
        raise ValueError("Expected distilled checkpoint with a 'model_state' entry")

    state = payload["model_state"]
    conv1 = state["conv1.weight"]
    fc1 = state["fc1.weight"]
    fc2 = state["fc2.weight"]

    model = AdaptiveCNNStudent(
        in_features=int(conv1.shape[1]),
        width=int(conv1.shape[0]),
        fc_hidden=int(fc1.shape[0]),
        out_dim=int(fc2.shape[0]),
        dropout=float(payload.get("config", {}).get("dropout", 0.2)),
    )
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, payload


def export_onnx(model: nn.Module, output: Path, windows: int, opset: int) -> torch.Tensor:
    in_features = int(model.conv1.in_channels)
    dummy = torch.randn(1, windows, in_features, dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy,
            str(output),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={
                "features": {0: "batch", 1: "windows"},
                "logits": {0: "batch"},
            },
        )
    return dummy


def verify(model: nn.Module, output: Path, sample: torch.Tensor) -> None:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Install verification deps: pip install onnx onnxruntime") from exc

    onnx.checker.check_model(onnx.load(str(output)))
    with torch.inference_mode():
        pt = model(sample).cpu().numpy()

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    ort_out = session.run(["logits"], {"features": sample.numpy()})[0]
    np.testing.assert_allclose(pt, ort_out, rtol=1e-5, atol=1e-5)
    print(f"Parity passed; max abs error={np.max(np.abs(pt - ort_out)):.3e}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--windows", type=int, default=39)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    model, payload = load_checkpoint_model(args.checkpoint)
    sample = export_onnx(model, args.output, args.windows, args.opset)

    print(f"Exported {args.output}")
    print(f"Input:  float32 [batch, windows, {model.conv1.in_channels}]")
    print(f"Output: float32 [batch, {model.fc2.out_features}] logits")
    print(f"Student: {payload.get('config', {}).get('student', 'unknown')}")

    if args.verify:
        verify(model, args.output, sample)


if __name__ == "__main__":
    main()
