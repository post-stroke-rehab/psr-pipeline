"""Export a trained CNN checkpoint to ONNX for Raspberry Pi 5 inference.

CNN_Micro is the default deployment target because it is the current best model.

Examples
--------
# Standard CNN Micro checkpoint/state_dict
python models/CNN/export_onnx.py \
    --checkpoint models/CNN/checkpoints/student_micro.pth \
    --output models/CNN/checkpoints/student_micro.onnx \
    --verify

# Optuna CNN Micro checkpoint; arch_kwargs are restored automatically
python models/CNN/export_onnx.py \
    --checkpoint models/CNN/checkpoints/optuna_micro.pth \
    --output models/CNN/checkpoints/optuna_micro.onnx \
    --verify

Use --model to export another student architecture if needed.

The exporter infers ``in_channels`` from ``proj.weight`` when possible, so a
model retrained for a reduced electrode set (for example 6 channels x 12
features = 72 input channels) does not need a hard-coded input size here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

try:
    from .students import CNN_Base, CNN_Large, CNN_Micro, CNN_Nano, CNN_XLarge
except ImportError:  # Allows: python models/CNN/export_onnx.py
    from students import CNN_Base, CNN_Large, CNN_Micro, CNN_Nano, CNN_XLarge


MODEL_REGISTRY = {
    "nano": CNN_Nano,
    "micro": CNN_Micro,
    "base": CNN_Base,
    "large": CNN_Large,
    "xlarge": CNN_XLarge,
}


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return state_dict


def _extract_state_dict(checkpoint: Any) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return (state_dict, arch_kwargs) from common PyTorch checkpoint formats."""
    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict(), {}

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint).__name__}")

    arch_kwargs = checkpoint.get("arch_kwargs", {}) or {}

    for key in ("state_dict", "model_state_dict", "model"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, nn.Module):
            return candidate.state_dict(), arch_kwargs
        if isinstance(candidate, dict) and candidate and all(
            isinstance(value, torch.Tensor) for value in candidate.values()
        ):
            return candidate, arch_kwargs

    if checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        return checkpoint, arch_kwargs

    raise ValueError(
        "Could not find model weights. Expected a plain state_dict or a checkpoint "
        "containing 'state_dict', 'model_state_dict', or 'model'."
    )


def _infer_in_channels(state_dict: dict[str, torch.Tensor]) -> int | None:
    weight = state_dict.get("proj.weight")
    if weight is not None and weight.ndim == 3:
        return int(weight.shape[1])
    return None


def load_model(checkpoint_path: Path, model_name: str, in_channels: int | None) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, nn.Module):
        model = checkpoint
        model.eval()
        return model

    state_dict, arch_kwargs = _extract_state_dict(checkpoint)
    state_dict = _strip_prefix(state_dict, "module.")
    state_dict = _strip_prefix(state_dict, "model.")

    inferred_channels = _infer_in_channels(state_dict)
    if in_channels is None:
        in_channels = inferred_channels
    elif inferred_channels is not None and in_channels != inferred_channels:
        raise ValueError(
            f"--in-channels={in_channels} does not match checkpoint weights "
            f"({inferred_channels})."
        )

    if in_channels is not None:
        arch_kwargs = {**arch_kwargs, "in_channels": in_channels}

    model = MODEL_REGISTRY[model_name](**arch_kwargs)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def export_onnx(
    model: nn.Module,
    output_path: Path,
    input_channels: int,
    window_count: int,
    opset: int,
) -> torch.Tensor:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, input_channels, window_count, dtype=torch.float32)

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={
                "input": {0: "batch", 2: "windows"},
                "logits": {0: "batch"},
            },
        )

    return dummy_input


def verify_onnx(model: nn.Module, onnx_path: Path, sample_input: torch.Tensor) -> None:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Verification requires: pip install onnx onnxruntime") from exc

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    with torch.inference_mode():
        torch_output = model(sample_input).cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_output = session.run(None, {"input": sample_input.cpu().numpy()})[0]

    np.testing.assert_allclose(torch_output, ort_output, rtol=1e-4, atol=1e-5)
    max_error = float(np.max(np.abs(torch_output - ort_output)))
    print(f"ONNX parity check passed (max abs error: {max_error:.3e})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Input .pt/.pth checkpoint")
    parser.add_argument("--output", type=Path, required=True, help="Output .onnx path")
    parser.add_argument(
        "--model",
        choices=MODEL_REGISTRY,
        default="micro",
        help="Student architecture (default: micro)",
    )
    parser.add_argument(
        "--in-channels",
        type=int,
        default=None,
        help="Override input feature channels; normally inferred from checkpoint",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=39,
        help="Dummy time-window count used during export (dynamic in the ONNX model)",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Validate ONNX and compare ONNX Runtime output with PyTorch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.windows < 1:
        raise ValueError("--windows must be >= 1")

    model = load_model(args.checkpoint, args.model, args.in_channels)

    if not hasattr(model, "proj") or not isinstance(model.proj, nn.Conv1d):
        if args.in_channels is None:
            raise ValueError("Could not infer input channels; pass --in-channels explicitly")
        input_channels = args.in_channels
    else:
        input_channels = int(model.proj.in_channels)

    sample_input = export_onnx(
        model=model,
        output_path=args.output,
        input_channels=input_channels,
        window_count=args.windows,
        opset=args.opset,
    )

    print(f"Exported: {args.output}")
    print(f"Model: {args.model}")
    print(f"Input: float32 [batch, {input_channels}, windows]")
    print("Output: float32 [batch, 5] raw logits")

    if args.verify:
        verify_onnx(model, args.output, sample_input)


if __name__ == "__main__":
    main()
