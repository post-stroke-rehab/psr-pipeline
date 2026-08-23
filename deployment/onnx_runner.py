"""Small ONNX Runtime wrapper for Raspberry Pi inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class InferenceResult:
    probabilities: np.ndarray
    intents: np.ndarray


class ONNXFingerIntentModel:
    def __init__(self, model_path: str | Path, thresholds: Sequence[float] | None = None) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("Install Raspberry Pi dependencies from deployment/requirements-rp5.txt") from exc

        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) < 1:
            raise ValueError("Expected an ONNX model with one input and at least one output")

        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.input_shape = inputs[0].shape

        t = np.asarray(thresholds if thresholds is not None else [0.5] * 5, dtype=np.float32)
        if t.shape != (5,):
            raise ValueError(f"thresholds must contain five values; got {t.shape}")
        self.thresholds = t

    @property
    def expected_feature_count(self) -> int | None:
        shape = self.input_shape
        if len(shape) == 3 and isinstance(shape[2], int):
            return int(shape[2])
        return None

    def predict(self, model_input: np.ndarray) -> InferenceResult:
        x = np.ascontiguousarray(model_input, dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(f"Expected model input (N,W,F); got {x.shape}")

        expected = self.expected_feature_count
        if expected is not None and x.shape[2] != expected:
            raise ValueError(
                f"Model expects {expected} features/window, preprocessing produced {x.shape[2]}"
            )

        logits = self.session.run([self.output_name], {self.input_name: x})[0]
        logits = np.asarray(logits, dtype=np.float32)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        intents = probabilities >= self.thresholds.reshape(1, -1)
        return InferenceResult(probabilities=probabilities, intents=intents)
