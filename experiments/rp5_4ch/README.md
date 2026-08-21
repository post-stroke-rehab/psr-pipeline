# RP5 Four-Channel CNN-Micro Retraining

This folder is the local run log and handoff area for retraining CNN-Micro on four active sEMG inputs for Raspberry Pi 5 deployment.

## Input Contract

- Raw active signals: `(samples, 4)`
- Active muscle order: `[ECRB, ECRL, FDS, FDP]`
- Ground/reference electrode: used by hardware only; never included in model tensors
- Feature tensor: `(batch, 4, windows, 12)`
- CNN input after adapter: `(batch, windows, 48)`
- Output logits: `(batch, 5)`
- Output finger order: `[thumb, index, middle, ring, little]`
- Sampling rate for PhysioMio preprocessing: `2048 Hz`
- Default windowing: `200 ms`, `50%` overlap, `410` samples per window, `205` sample stride

## Channel Maps

| View | One-based channel IDs | Zero-based channel IDs |
| --- | --- | --- |
| Left | `[1, 3, 9, 14]` | `[0, 2, 8, 13]` |
| Right | `[15, 16, 9, 1]` | `[14, 15, 8, 0]` |

Because PhysioMio does not provide a reliable per-patient laterality map for this hardware placement, preprocessing emits left-map and right-map views. The dual-map policy trains with both views and evaluates left/right views separately.

## Run Commands

Local shape/logging smoke test:

```bash
python scripts/run_rp5_4ch_experiments.py --stage smoke
```

Colab run sequence after PhysioMio preprocessing:

```bash
python scripts/run_rp5_4ch_experiments.py --stage view --device cuda
python scripts/run_rp5_4ch_experiments.py --stage context --selected-view dual --device cuda
python scripts/run_rp5_4ch_experiments.py --stage direct --selected-view dual --selected-context 4 --device cuda
python scripts/run_rp5_4ch_experiments.py --stage transfer --selected-view left --selected-context 4 --transfer-checkpoint path/to/full64_cnn_micro.pt --device cuda
python scripts/run_rp5_4ch_experiments.py --stage distill --selected-view dual --selected-context 4 --device cuda
python scripts/run_rp5_4ch_experiments.py --stage aggregate
```

## Run Folder Contents

Each completed run writes:

- `config.json`
- `console.log`
- `history.json`
- `manifest.json`
- `model_config.json`
- `thresholds.json`
- `val_metrics.json`
- `test_metrics.json`
- `status.json`
- `checkpoint_best.pt`

The final selected checkpoint should be copied to `experiments/rp5_4ch/final/` with the matching `model_card.md`, `model_config.json`, `thresholds.json`, and `manifest.json`.
