# Raspberry Pi 5 Inference Software

This package implements the software boundary from four raw sEMG channels to
five finger-intent predictions:

```text
four RAW sEMG channels
        -> rolling preprocessing and 12 features/channel
        -> float32 tensor [batch, 9, 48]
        -> CNN-Micro ONNX inference
        -> five probabilities
        -> validation-selected thresholds
        -> five binary finger intents
```

## Current Model Contract

- Active muscle order: `[ECRB, ECRL, FDS, FDP]`
- Ground/reference electrode: excluded from model input
- Right-map PhysioMio channels: `15, 16, 9, 1` (one-based)
- Sampling rate: 2,048 Hz
- Window: 200 ms (410 samples), 50% overlap (205-sample stride)
- Context: 9 windows
- Model input: `float32 [batch, 9, 48]`
- Model output: `float32 [batch, 5]` logits
- Finger order: `[thumb, index, middle, ring, little]`
- Thresholds: `[0.50, 0.70, 0.60, 0.65, 0.60]`

The selected PyTorch checkpoint is
`experiments/rp5_4ch/final/cnn_micro_4ch_right_ctx9_distill_seed4.pt`. The
committed `artifacts/cnn_micro.onnx` is its fixed-context ONNX export.

## Runtime Behavior

`RealtimePreprocessor` receives `(samples, 4)` chunks, maintains rolling signal
history, and emits model contexts. It reuses the offline 20--450 Hz zero-phase
filter, Symlet-4 wavelet denoising, 200 ms windowing, and 12-feature extractor.

This compatibility-first preprocessing is computationally non-causal because
it reruns forward-backward filtering and wavelet denoising over available
history. It must be benchmarked on the Raspberry Pi before real-time claims are
made. The repository currently demonstrates export and replay compatibility,
not measured Raspberry Pi latency.

## Export

```bash
python models/CNN/export_onnx.py \
  --checkpoint experiments/rp5_4ch/final/cnn_micro_4ch_right_ctx9_distill_seed4.pt \
  --output deployment/artifacts/cnn_micro.onnx \
  --windows 9 \
  --verify
```

The export has dynamic batch size and fixed temporal context. ONNX verification
requires the development dependencies in `requirements-dev.txt`.

## Install On Raspberry Pi 5

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r deployment/requirements-rp5.txt
```

Normal inference uses ONNX Runtime and does not require PyTorch. Keep the model
with its channel order, acquisition rate, context length, feature order, and
threshold file.

## Recorded-Signal Replay

```bash
python scripts/run_rp5_inference.py \
  --model deployment/artifacts/cnn_micro.onnx \
  --input /path/to/impaired-arm-recording.parquet \
  --channels channel_15 channel_16 channel_09 channel_01 \
  --sample-rate 2048 \
  --context-windows 9 \
  --thresholds 0.50,0.70,0.60,0.65,0.60 \
  --unpaced
```

Remove `--unpaced` to replay at acquisition speed. The runner checks that the
number of selected channels produces the feature count expected by the ONNX
model and fails on mismatches.

Future acquisition code should deliver ordered `float32` chunks to the same
boundary:

```python
raw_chunk: np.ndarray  # shape (N, 4), order ECRB/ECRL/FDS/FDP
preprocessor.push(raw_chunk)
```

ADC acquisition, actuator control, hardware safety, and clinical validation are
outside this software package.
