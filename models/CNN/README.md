# CNN — Post-Stroke sEMG → Finger Activation

Trains 1D CNN models that predict which of 5 fingers are activated from sEMG. Supports baseline training (student or teacher) and single-step knowledge distillation from a ResNet teacher into a smaller student.

## Data Shape

The team dataloader in `datasets/loaders.py` yields:

- `X`: `(B, 64, W, 12)` float32 — 64 sEMG channels, `W` time-windows (padded to dataset-wide max, e.g. 39), 12 hand-crafted features per window (RMS, MAV, IAV, SSC, ZC, WAMP, slope, VAR, log-energy, spectral center, mean freq, median freq).
- `y`: `(B, 5)` float32 binary — one bit per finger (thumb, index, middle, ring, little).

In `main.get_loaders`, we permute and reshape this to `(B, 768, W)` so the 64 channels × 12 features become Conv1d input channels and the model convolves over time.

## Files

| File | Role |
|---|---|
| `main.py` | Entry point: `Config` dataclass, `train_student / train_teacher / train_student_with_kd`. |
| `students.py` | Five student CNN classes: `CNN_Nano`, `CNN_Micro`, `CNN_Base`, `CNN_Large`, `CNN_XLarge`. |
| `teachers.py` | Three 1D ResNet teacher classes: `ResNet50_1D`, `ResNet101_1D`, `ResNet152_1D` (share `Bottleneck1D`). |
| `training.py` | Generic `train_model` loop + `train_one_epoch` + `evaluate`. Uses `BCEWithLogitsLoss`. |
| `distillation.py` | Single-step KD: hard `BCEWithLogitsLoss` + soft `MSE(sigmoid(s/T), sigmoid(t/T)) * T²`, blended by `alpha`. |
| `optimization.py` | Optuna HPO over architecture + training hyperparams per student size, scored by `accuracy × speed_penalty`. |

All models output **raw logits** (no sigmoid). Loss and accuracy code apply `sigmoid` where needed.

## Student variants

Each student takes `in_channels=768, num_classes=5`. The first layer is a 1×1 Conv1d that projects the 768-channel input down to a small width so the rest of the stack is cheap (this is what makes the models Pi-friendly). The time axis is shrunk with `AdaptiveAvgPool1d` before the FC head, so any `W` is accepted.

| Model | Params | fp32 | int8 (post-quant) |
|---|---:|---:|---:|
| Nano | 54K | 213 KB | 53 KB |
| Micro | 158K | 617 KB | 154 KB |
| Base | 390K | 1.5 MB | 381 KB |
| Large | 803K | 3.1 MB | 784 KB |
| XLarge | 1.62M | 6.3 MB | 1.58 MB |

All five are sized for a Raspberry Pi 5 (4–8 GB) after int8 quantization. FC layers are bounded so they don't dominate memory.

## Teacher variants

Standard 1D ResNet with `Bottleneck1D` blocks; only the `layers` list differs:

- `ResNet50_1D`  → `[3, 4, 6, 3]`
- `ResNet101_1D` → `[3, 4, 23, 3]`
- `ResNet152_1D` → `[3, 8, 36, 3]`

## How to run

Edit the defaults of the `Config` dataclass at the top of `main.py`, then from the repo root:

```bash
python models/CNN/main.py
```

Modes:

- `mode="student"` — trains the student named by `student_name`.
- `mode="teacher"` — trains the teacher named by `teacher_name`.
- `mode="student_kd"` — trains the student via single-step KD from the teacher. If `teacher_ckpt` is set and the file exists, the teacher is loaded; otherwise the teacher is pretrained on the fly for `teacher_epochs` epochs.

Checkpoints are saved under `cfg.save_dir` (default `models/CNN/checkpoints/`):

- `student_<name>.pth`
- `teacher_<name>.pth`
- `student_kd_<student>_from_<teacher>.pth`

After training, the test split is scored once with `evaluation.metrics.compute_multilabel_metrics` (team standard) and a one-line summary (`acc / f1_macro / auprc_macro`) is printed.

## Hyperparameter optimization

`optimization.py` runs Optuna for each student size and saves the best model plus a `.txt` reproducibility file.

```bash
python models/CNN/optimization.py
```

Each student constructor accepts architecture kwargs (channel counts, kernel sizes, pool target, FC width, dropout). `SEARCH_SPACES` in `optimization.py` defines the per-size choices Optuna picks from. Training hyperparams `lr`, `weight_decay`, and `batch_size` are tuned globally via `TRAIN_SPACE`. Epochs is **fixed** at `HPOConfig.trial_epochs` (default 100, the cap) — Optuna already uses the best per-epoch val accuracy and the pruner kills slow learners, so tuning epochs would just optimize "train more = better". Lower `trial_epochs` for bigger models or KD runs.

**Per-trial early stopping.** To make HPO quick, each trial stops once val accuracy has plateaued — controlled by `HPOConfig.early_stop_patience` (default 5 epochs without significant improvement) and `HPOConfig.early_stop_min_delta` (default 2.0 percentage points). Trials that keep improving train all the way to `trial_epochs`. The final selected hyperparams should still be retrained for the full 100 epochs to squeeze out residual gains.

**Scoring.** Before each size's study, the default (current hardcoded) model is trained for 1 epoch and its inference latency is captured as the reference `t_ref`. Each trial's score is

```
score = val_accuracy × penalty(latency / t_ref)
penalty: 1.0 when at/under baseline, linear 1.0→0.4 between 1.0× and 1.2×, 0.05 beyond 1.2×
```

so trials that are no slower than the baseline aren't penalized, mildly slower trials lose points, and big regressions are crushed. A `MedianPruner` kills bad trials early based on val accuracy reported per epoch.

**Outputs** (per size, in `models/CNN/checkpoints/`):
- `optuna_<size>.pth` — `{"state_dict": ..., "arch_kwargs": {...}}`. Reload with:
  ```python
  ckpt = torch.load("models/CNN/checkpoints/optuna_nano.pth")
  model = CNN_Nano(**ckpt["arch_kwargs"]); model.load_state_dict(ckpt["state_dict"])
  ```
- `optuna_<size>.txt` — human-readable: trial count, best score / accuracy / latency, architecture kwargs, training hyperparams, and a "reproduce" snippet.

Tune scope and trial count via `HPOConfig` at the top of `optimization.py` (e.g. `sizes=["nano"]`, `n_trials_per_size=10` for a smoke test).

## Importing training functions

The three training functions in `main.py` are the public API for any external HPO/eval script:

```python
from models.CNN.main import Config, train_student, train_teacher, train_student_with_kd

cfg = Config(student_name="nano", epochs=10, lr=1e-3)
model = train_student(cfg)
```
