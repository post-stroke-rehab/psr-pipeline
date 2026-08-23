"""Export supported CNN checkpoints to ONNX for Raspberry Pi 5 deployment.

Supports both:
- the older distilled ``AdaptiveCNNStudent`` checkpoint from PR #71; and
- the four-channel ``CNNMicroSequence`` checkpoints produced by
  ``training.rp5_four_channel``.

Examples:
    python models/CNN/export_onnx.py \
        --checkpoint results/distill_micro_from_cnn_a0.3_t2.0/checkpoint_best.pt \
        --output deployment/artifacts/cnn_micro.onnx \
        --verify

    python models/CNN/export_onnx.py \
        --checkpoint experiments/rp5_4ch/final/cnn_micro_4ch_right_ctx9_distill_seed4.pt \
        --output deployment/artifacts/cnn_micro_4ch_right_ctx9.onnx \
        --verify
"""

from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.CNN.students import CNN_Micro


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
        if x.dim() != 3:
            raise ValueError(f"Expected (N,W,F), got {tuple(x.shape)}")
        x = x.permute(0, 2, 1).contiguous()
        x = F.relu(self.conv1(x))
        x = F.relu(self.bn(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class CNNMicroSequence(nn.Module):
    """Deployment reconstruction of training.rp5_four_channel.CNNMicroSequence."""

    def __init__(self, in_features: int = 48, out_dim: int = 5, dropout: float = 0.2) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_dim = int(out_dim)
        self.backbone = CNN_Micro(
            in_channels=self.in_features,
            num_classes=self.out_dim,
            dropout=float(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"CNNMicroSequence expects (N,W,F), got {tuple(x.shape)}")
        if x.shape[1] < 2:
            x = F.pad(x, (0, 0, 0, 2 - int(x.shape[1])))
        return self.backbone(x.permute(0, 2, 1).contiguous())


class StaticAdaptiveMaxPool1d(nn.Module):
    """ONNX-safe equivalent of AdaptiveMaxPool1d for a fixed input length.

    PyTorch allows adaptive max pooling to produce more output bins than input
    samples (the selected 9-window CNN-Micro reaches length 4 then pools to 8).
    ONNX exporters do not lower that case directly. For a fixed input length the
    adaptive pooling bin boundaries are static, so the exact operation can be
    written as fixed slices plus ``amax`` and concatenation.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        if input_size < 1 or output_size < 1:
            raise ValueError("input_size and output_size must be positive")
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.bins = tuple(
            (
                math.floor(i * self.input_size / self.output_size),
                math.ceil((i + 1) * self.input_size / self.output_size),
            )
            for i in range(self.output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = [
            torch.amax(x[..., start:end], dim=-1, keepdim=True)
            for start, end in self.bins
        ]
        return torch.cat(pooled, dim=-1)


def _checkpoint_state(payload: object) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError("Expected checkpoint payload to be a dictionary")
    state = payload.get("model_state", payload)
    if not isinstance(state, dict):
        raise ValueError("Checkpoint does not contain a valid state dict")
    return state


def _dropout_from_payload(payload: dict) -> float:
    config = payload.get("config", {})
    return float(config.get("dropout", 0.2)) if isinstance(config, dict) else 0.2


def load_checkpoint_model(path: Path) -> tuple[nn.Module, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Expected checkpoint payload to be a dictionary")

    state = _checkpoint_state(payload)
    keys = set(state)
    dropout = _dropout_from_payload(payload)

    if "conv1.weight" in keys and "fc1.weight" in keys and "fc2.weight" in keys:
        conv1 = state["conv1.weight"]
        fc1 = state["fc1.weight"]
        fc2 = state["fc2.weight"]
        model: nn.Module = AdaptiveCNNStudent(
            in_features=int(conv1.shape[1]),
            width=int(conv1.shape[0]),
            fc_hidden=int(fc1.shape[0]),
            out_dim=int(fc2.shape[0]),
            dropout=dropout,
        )
        model.load_state_dict(state, strict=True)
    else:
        normalized = {
            (key if key.startswith("backbone.") else f"backbone.{key}"): value
            for key, value in state.items()
        }
        proj_key = "backbone.proj.weight"
        fc2_key = "backbone.fc2.weight"
        if proj_key not in normalized or fc2_key not in normalized:
            preview = ", ".join(sorted(keys)[:8])
            raise ValueError(
                "Unsupported checkpoint architecture. Expected either "
                "AdaptiveCNNStudent keys (conv1.weight/fc1.weight/fc2.weight) "
                "or CNNMicroSequence/CNN_Micro keys containing proj.weight and "
                f"fc2.weight. First keys: {preview}"
            )

        model = CNNMicroSequence(
            in_features=int(normalized[proj_key].shape[1]),
            out_dim=int(normalized[fc2_key].shape[0]),
            dropout=dropout,
        )
        model.load_state_dict(normalized, strict=True)

    model.eval()
    return model, payload


def model_input_features(model: nn.Module) -> int:
    if isinstance(model, AdaptiveCNNStudent):
        return int(model.conv1.in_channels)
    if isinstance(model, CNNMicroSequence):
        return int(model.in_features)
    raise TypeError(f"Unsupported model type: {type(model).__name__}")


def model_output_features(model: nn.Module) -> int:
    if isinstance(model, AdaptiveCNNStudent):
        return int(model.fc2.out_features)
    if isinstance(model, CNNMicroSequence):
        return int(model.out_dim)
    raise TypeError(f"Unsupported model type: {type(model).__name__}")


def resolve_windows(payload: dict, override: int | None) -> int:
    if override is not None:
        if override < 1:
            raise ValueError("--windows must be at least 1")
        return int(override)

    config = payload.get("config", {})
    if isinstance(config, dict):
        context_windows = config.get("context_windows")
        if isinstance(context_windows, int) and context_windows > 0:
            return context_windows

    return 39


def prepare_export_model(model: nn.Module, windows: int) -> nn.Module:
    """Return an ONNX-friendly copy that is numerically identical at this context."""
    export_model = copy.deepcopy(model).eval()
    if not isinstance(export_model, CNNMicroSequence):
        return export_model

    pool = export_model.backbone.pool
    if not isinstance(pool, nn.AdaptiveMaxPool1d):
        return export_model

    effective_windows = max(int(windows), 2)
    # CNN_Micro applies max_pool1d(kernel_size=2, stride=2) before this pool.
    pooled_input_size = effective_windows // 2
    output_size = pool.output_size
    if isinstance(output_size, tuple):
        output_size = output_size[0]
    export_model.backbone.pool = StaticAdaptiveMaxPool1d(
        input_size=pooled_input_size,
        output_size=int(output_size),
    )
    return export_model


def export_onnx(model: nn.Module, output: Path, windows: int, opset: int) -> torch.Tensor:
    """Export with fixed temporal context and dynamic batch size."""
    in_features = model_input_features(model)
    dummy = torch.randn(1, windows, in_features, dtype=torch.float32)
    output.parent.mkdir(parents=True, exist_ok=True)

    export_model = prepare_export_model(model, windows)
    with torch.inference_mode():
        original_out = model(dummy)
        export_out = export_model(dummy)
    torch.testing.assert_close(original_out, export_out, rtol=1e-6, atol=1e-7)

    with torch.inference_mode():
        torch.onnx.export(
            export_model,
            dummy,
            str(output),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={
                "features": {0: "batch"},
                "logits": {0: "batch"},
            },
            dynamo=False,
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
    ap.add_argument(
        "--windows",
        type=int,
        default=None,
        help="Temporal context length. Defaults to checkpoint config.context_windows, or 39 for legacy checkpoints.",
    )
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    model, payload = load_checkpoint_model(args.checkpoint)
    windows = resolve_windows(payload, args.windows)
    sample = export_onnx(model, args.output, windows, args.opset)

    print(f"Exported {args.output}")
    print(f"Input:  float32 [batch, {windows}, {model_input_features(model)}]")
    print(f"Context windows: {windows} (fixed)")
    print(f"Output: float32 [batch, {model_output_features(model)}] logits")
    print(f"Architecture: {type(model).__name__}")

    if args.verify:
        verify(model, args.output, sample)


if __name__ == "__main__":
    main()
