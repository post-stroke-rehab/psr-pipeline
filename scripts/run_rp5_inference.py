"""End-to-end replay -> real-time preprocessing -> ONNX inference runner.

Run this script on a Raspberry Pi 5 to benchmark the software deployment path
using recorded PhysioMio data as the acquisition source.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from deployment.onnx_runner import ONNXFingerIntentModel
from deployment.realtime_preprocess import RealtimePreprocessConfig, RealtimePreprocessor
from deployment.replay import load_physiomio_channels, replay_array


def _parse_thresholds(value: str) -> list[float]:
    vals = [float(v.strip()) for v in value.split(",") if v.strip()]
    if len(vals) != 5:
        raise argparse.ArgumentTypeError("thresholds must be five comma-separated values")
    return vals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True, help="FP32 ONNX model")
    ap.add_argument("--input", type=Path, required=True, help="Raw PhysioMio parquet recording")
    ap.add_argument("--channels", nargs="+", required=True, help="Ordered channel_* columns")
    ap.add_argument("--sample-rate", type=float, default=2000.0)
    ap.add_argument("--window-ms", type=float, default=200.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--context-windows", type=int, default=39)
    ap.add_argument("--chunk-ms", type=float, default=10.0)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--unpaced", action="store_true")
    ap.add_argument("--thresholds", type=_parse_thresholds, default=[0.5] * 5)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    model = ONNXFingerIntentModel(args.model, thresholds=args.thresholds)
    pre = RealtimePreprocessor(
        RealtimePreprocessConfig(
            sample_rate=args.sample_rate,
            channels=len(args.channels),
            window_size=args.window_ms / 1000.0,
            overlap=args.overlap,
            context_windows=args.context_windows,
        )
    )

    expected = model.expected_feature_count
    if expected is not None and expected != pre.cfg.model_features:
        raise ValueError(
            f"Model expects {expected} features/window, but {len(args.channels)} channels "
            f"produce {pre.cfg.model_features}. Use the channel set the model was trained on."
        )

    samples = load_physiomio_channels(args.input, args.channels)
    preprocess_ms: list[float] = []
    inference_ms: list[float] = []
    end_to_end_ms: list[float] = []
    predictions = 0

    def on_samples(chunk: np.ndarray) -> None:
        nonlocal predictions
        t0 = time.perf_counter()
        p0 = time.perf_counter()
        contexts = pre.push(chunk)
        p1 = time.perf_counter()
        if contexts:
            preprocess_ms.append((p1 - p0) * 1000.0)

        for model_input in contexts:
            i0 = time.perf_counter()
            result = model.predict(model_input)
            i1 = time.perf_counter()
            inference_ms.append((i1 - i0) * 1000.0)
            end_to_end_ms.append((i1 - t0) * 1000.0)
            predictions += 1

            if not args.quiet:
                probs = np.round(result.probabilities[0], 4).tolist()
                intents = result.intents[0].astype(int).tolist()
                print(f"prediction={predictions:05d} probs={probs} intents={intents}")

    replay_stats = replay_array(
        samples,
        sample_rate=args.sample_rate,
        callback=on_samples,
        chunk_ms=args.chunk_ms,
        realtime=not args.unpaced,
        speed=args.speed,
    )

    def stats(values: list[float]) -> dict:
        if not values:
            return {"count": 0, "mean_ms": None, "p95_ms": None, "max_ms": None}
        a = np.asarray(values, dtype=np.float64)
        return {
            "count": int(a.size),
            "mean_ms": float(np.mean(a)),
            "p95_ms": float(np.percentile(a, 95)),
            "max_ms": float(np.max(a)),
        }

    report = {
        "model": str(args.model),
        "input": str(args.input),
        "channels": args.channels,
        "feature_count": pre.cfg.model_features,
        "window_samples": pre.cfg.window_samples,
        "stride_samples": pre.cfg.stride_samples,
        "context_windows": pre.cfg.context_windows,
        "predictions": predictions,
        "replay": replay_stats,
        "preprocessing": stats(preprocess_ms),
        "inference": stats(inference_ms),
        "end_to_end": stats(end_to_end_ms),
    }

    print(json.dumps(report, indent=2))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
