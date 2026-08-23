# Raspberry Pi 5 inference deployment

This package wires the current distilled CNN-Micro deployment path together:

```text
recorded/raw sEMG chunks
        ↓
RealtimePreprocessor (runs on RP5)
        ↓
float32 model input (1, W, C*12)
        ↓
ONNX Runtime
        ↓
5 probabilities
        ↓
5 calibrated thresholds
        ↓
5 finger intents
```

## What runs on the Raspberry Pi

The Pi is responsible for preprocessing and inference. The model is **not** fed raw sEMG directly.

`deployment/realtime_preprocess.py` receives arbitrary `(samples, channels)` chunks and maintains a rolling raw-signal context. Every inference stride it reuses the repository's existing preprocessing implementation:

1. 20–450 Hz band-pass filter
2. wavelet denoising
3. 200 ms windows
4. 50% overlap
5. 12 features per channel
6. `(C,W,12) -> (1,W,C*12)` model adapter

By default a new model input is emitted every 100 ms. The context is right-padded with zeros during startup to preserve the fixed sequence length used by training, then becomes a rolling history once full.

This is a compatibility-first implementation. The offline pipeline currently uses zero-phase filtering and non-streaming wavelet denoising, so this module reruns that same implementation over the available rolling context rather than pretending those operations are inherently causal. Benchmark this on the Pi. If it is too slow, replace the DSP with a causal/stateful implementation only after measuring prediction impact.

## Current model contract

PR #71's checkpoint is the distilled `AdaptiveCNNStudent`, not `models/CNN/students.py::CNN_Micro`.

The current 64-channel model expects:

```text
input:  float32 [batch, windows, 768]
        768 = 64 channels × 12 features
output: float32 [batch, 5] logits
```

A future four-channel checkpoint will instead expect 48 features/window. The runner checks this automatically and fails rather than silently using the wrong channel count.

## Export checkpoint to ONNX

Run export on a development machine with PyTorch installed:

```bash
python models/CNN/export_onnx.py \
  --checkpoint results/distill_micro_from_cnn_a0.3_t2.0/checkpoint_best.pt \
  --output deployment/artifacts/cnn_micro.onnx \
  --verify
```

The Pi itself does not need PyTorch or the `.pt` file for normal ONNX inference.

## Install on Raspberry Pi 5

Clone/copy this repository or copy the deployment bundle, then create a virtual environment and install:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r deployment/requirements-rp5.txt
```

For normal hardware inference, the Pi needs:

```text
deployment/
  realtime_preprocess.py
  onnx_runner.py
  __init__.py

data_processing/
  preprocess.py
  preprocess_config.py
  bandpass_filter.py
  wavelet_denoise.py
  windowing.py
  feature_extraction.py

cnn_micro.onnx
preprocessing/deployment configuration
five thresholds
channel ordering/map
```

If the full repo is cloned onto the Pi, the Python source files above are already present; only the generated ONNX artifact and deployment-specific configuration need to be added/updated.

The original `.pt` checkpoint is useful for reproducibility and PyTorch-vs-ONNX verification but is not required by ONNX Runtime on the Pi.

## Replay benchmark

To emulate acquisition using a raw PhysioMio parquet file:

```bash
python scripts/run_rp5_inference.py \
  --model deployment/artifacts/cnn_micro.onnx \
  --input /path/to/recording.parquet \
  --channels channel_01 channel_02 ... channel_64 \
  --sample-rate 2000 \
  --chunk-ms 10 \
  --report deployment/report.json
```

The replay layer delivers small raw chunks at wall-clock acquisition speed. It does not pre-window samples for the model. The rolling preprocessor decides when a full inference update is available.

Use `--unpaced` for regression testing without sleeps.

## Production acquisition replacement

The emulator and future hardware receiver should both end at the same software boundary:

```python
raw_chunk: np.ndarray  # shape (N, C), ordered channels
preprocessor.push(raw_chunk)
```

When Pico/ADC acquisition is available, replace the parquet replay callback with the hardware receiver. `RealtimePreprocessor` and `ONNXFingerIntentModel` remain unchanged.

## Files that must accompany the ONNX model

The ONNX file contains the network graph and weights, but it does **not** define the raw-signal contract. Keep the following deployment metadata with it:

- ordered channel map;
- acquisition/sample rate;
- preprocessing parameters;
- model context length;
- feature ordering/version;
- five probability thresholds;
- model/checkpoint/version identifier.

These should eventually be frozen into a machine-readable deployment manifest so the RP5 runner can validate them at startup.
