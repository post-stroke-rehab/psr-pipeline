import copy
import itertools
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import optuna
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from main import Config, set_seed, get_loaders, _summary_metrics
from students import CNN_Nano, CNN_Micro, CNN_Base, CNN_Large, CNN_XLarge
from training import train_one_epoch, evaluate


STUDENTS = {"nano": CNN_Nano, "micro": CNN_Micro, "base": CNN_Base,
            "large": CNN_Large, "xlarge": CNN_XLarge}


SEARCH_SPACES: Dict[str, Dict[str, Any]] = {
    "nano": {
        "proj_ch":  [16, 24, 32, 48, 64],
        "conv1_ch": [24, 32, 48, 64, 96],
        "conv1_k":  [3, 5, 7],
        "pool":     [4, 8, 16],
        "fc1":      [32, 48, 64, 96, 128],
        "dropout":  (0.0, 0.5),
    },
    "micro": {
        "proj_ch":  [24, 32, 48, 64, 96],
        "conv1_ch": [32, 48, 64, 96, 128],
        "conv1_k":  [3, 5, 7],
        "conv2_ch": [48, 64, 96, 128, 160],
        "conv2_k":  [3, 5, 7],
        "pool":     [4, 8, 16],
        "fc1":      [48, 64, 96, 128, 192],
        "dropout":  (0.0, 0.5),
    },
    "base": {
        "proj_ch":  [32, 48, 64, 96, 128],
        "conv1_ch": [48, 64, 96, 128, 160],
        "conv1_k":  [3, 5, 7],
        "conv2_ch": [64, 96, 128, 160, 192],
        "conv2_k":  [3, 5, 7],
        "conv3_ch": [96, 128, 160, 192, 256],
        "conv3_k":  [3, 5, 7],
        "conv4_ch": [96, 128, 160, 192, 256],
        "conv4_k":  [3, 5, 7],
        "pool":     [2, 4, 8],
        "fc1":      [64, 96, 128, 192, 256],
        "dropout":  (0.0, 0.5),
    },
    "large": {
        "proj_ch":  [48, 64, 96, 128, 192],
        "conv1_ch": [64, 96, 128, 192, 256],
        "conv1_k":  [3, 5, 7],
        "conv2_ch": [96, 128, 192, 256, 320],
        "conv2_k":  [3, 5, 7],
        "conv3_ch": [128, 192, 256, 320, 384],
        "conv3_k":  [3, 5, 7],
        "conv4_ch": [128, 192, 256, 320, 384],
        "conv4_k":  [3, 5, 7],
        "pool":     [2, 4, 8],
        "fc1":      [96, 128, 192, 256, 384],
        "dropout":  (0.0, 0.5),
    },
    "xlarge": {
        "proj_ch":  [64, 96, 128, 192, 256],
        "conv1_ch": [96, 128, 192, 256, 384],
        "conv1_k":  [3, 5, 7],
        "conv2_ch": [128, 192, 256, 384, 512],
        "conv2_k":  [3, 5, 7],
        "conv3_ch": [192, 256, 384, 512, 640],
        "conv3_k":  [3, 5, 7],
        "conv4_ch": [256, 384, 512, 640, 768],
        "conv4_k":  [3, 5, 7],
        "pool":     [2, 4, 8],
        "fc1":      [128, 192, 256, 384, 512],
        "dropout":  (0.0, 0.6),
    },
}


TRAIN_SPACE = {
    "lr":           (1e-5, 1e-2),   # log-uniform
    "weight_decay": (1e-6, 1e-3),   # log-uniform
    "batch_size":   [16, 32, 64, 128],
}


@dataclass
class HPOConfig:
    sizes: List[str] = field(default_factory=lambda: ["nano", "micro", "base", "large", "xlarge"])
    n_trials_per_size: int = 50
    trial_epochs: int = 100              # max epochs per trial (early stop usually triggers sooner)
    early_stop_patience: int = 5         # stop trial after this many epochs without a significant improvement
    early_stop_min_delta: float = 2.0    # percentage-points of val accuracy that counts as "significant"
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "models/CNN/checkpoints"
    baseline_epochs: int = 1
    latency_warmup: int = 5
    latency_iters: int = 20


def suggest_arch(trial: optuna.Trial, size: str) -> Dict[str, Any]:
    out = {}
    for name, space in SEARCH_SPACES[size].items():
        if isinstance(space, tuple):
            out[name] = trial.suggest_float(name, space[0], space[1])
        else:
            out[name] = trial.suggest_categorical(name, space)
    return out


def suggest_training(trial: optuna.Trial) -> Dict[str, Any]:
    return {
        "lr":           trial.suggest_float("lr",           *TRAIN_SPACE["lr"],           log=True),
        "weight_decay": trial.suggest_float("weight_decay", *TRAIN_SPACE["weight_decay"], log=True),
        "batch_size":   trial.suggest_categorical("batch_size", TRAIN_SPACE["batch_size"]),
    }


@torch.no_grad()
def measure_latency(model: nn.Module, loader, device: torch.device,
                    warmup: int = 5, iters: int = 20) -> float:
    """Median per-batch forward latency in milliseconds."""
    model.eval()
    cyc = itertools.cycle(loader)
    for _ in range(warmup):
        x, _ = next(cyc)
        model(x.to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        x, _ = next(cyc)
        x = x.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def speed_penalty(ratio: float) -> float:
    if ratio <= 1.0:
        return 1.0
    if ratio <= 1.2:
        return 1.0 - 3.0 * (ratio - 1.0)   # 1.0 -> 0.4 linearly across [1.0, 1.2]
    return 0.05


def _train_for_trial(model, train_loader, val_loader, lr, weight_decay, epochs,
                     device, trial: optuna.Trial, patience: int, min_delta: float):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    best_acc, best_state = -1.0, None
    last_milestone = -float("inf")   # last val_acc that reset the patience counter
    stall = 0
    for epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        val = evaluate(model, val_loader, criterion, device)
        acc = val["accuracy"]
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
        if acc >= last_milestone + min_delta:
            last_milestone = acc
            stall = 0
        else:
            stall += 1
        if stall >= patience:
            break
        trial.report(acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return best_acc, best_state, epoch + 1


def _loaders_for(batch_size: int, hcfg: HPOConfig):
    return get_loaders(Config(batch_size=batch_size, seed=hcfg.seed, device=hcfg.device))


def measure_baseline(size: str, hcfg: HPOConfig):
    """Train default architecture for `baseline_epochs`, then measure inference latency."""
    set_seed(hcfg.seed)
    device = torch.device(hcfg.device)
    model = STUDENTS[size]().to(device)
    train_loader, val_loader, _ = _loaders_for(32, hcfg)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(hcfg.baseline_epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
    val = evaluate(model, val_loader, criterion, device)
    latency_ms = measure_latency(model, val_loader, device, hcfg.latency_warmup, hcfg.latency_iters)
    return {"latency_ms": latency_ms, "val_accuracy": val["accuracy"]}


def run_study(size: str, hcfg: HPOConfig, baseline_latency_ms: float):
    device = torch.device(hcfg.device)
    cls = STUDENTS[size]

    best: Dict[str, Any] = {"score": -1.0}

    def objective(trial: optuna.Trial) -> float:
        set_seed(hcfg.seed + trial.number)
        arch_kwargs = suggest_arch(trial, size)
        train_kwargs = suggest_training(trial)

        train_loader, val_loader, _ = _loaders_for(train_kwargs["batch_size"], hcfg)

        model = cls(**arch_kwargs).to(device)
        best_acc, best_state, epochs_run = _train_for_trial(
            model, train_loader, val_loader,
            lr=train_kwargs["lr"],
            weight_decay=train_kwargs["weight_decay"],
            epochs=hcfg.trial_epochs,
            device=device, trial=trial,
            patience=hcfg.early_stop_patience,
            min_delta=hcfg.early_stop_min_delta,
        )
        model.load_state_dict(best_state)

        latency_ms = measure_latency(model, val_loader, device, hcfg.latency_warmup, hcfg.latency_iters)
        ratio = latency_ms / baseline_latency_ms
        score = (best_acc / 100.0) * speed_penalty(ratio)

        trial.set_user_attr("val_accuracy", best_acc)
        trial.set_user_attr("latency_ms", latency_ms)
        trial.set_user_attr("ratio", ratio)
        trial.set_user_attr("epochs_run", epochs_run)

        if score > best["score"]:
            best.update(score=score, state=best_state, arch_kwargs=arch_kwargs,
                        train_kwargs=train_kwargs, val_accuracy=best_acc,
                        latency_ms=latency_ms, ratio=ratio, epochs_run=epochs_run)
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=hcfg.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.optimize(objective, n_trials=hcfg.n_trials_per_size)
    return study, best


def _format_kwargs(d: Dict[str, Any]) -> str:
    return "\n".join(f"  {k:<14} = {v if not isinstance(v, float) else f'{v:.6g}'}" for k, v in d.items())


def _write_txt(path: Path, *, size: str, study: optuna.Study, best: Dict[str, Any],
               baseline_latency_ms: float, baseline_val_acc: float, trial_epochs: int):
    n_pruned = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED)
    n_complete = sum(1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE)
    arch_repr = ", ".join(f"{k}={v!r}" for k, v in best["arch_kwargs"].items())
    epoch_note = "early-stopped" if best["epochs_run"] < trial_epochs else "ran to cap"

    content = f"""=== optuna_{size} ===
trials run:         {len(study.trials)}  (complete={n_complete}, pruned={n_pruned})
best score:         {best['score']:.4f}
best val accuracy:  {best['val_accuracy']:.2f}%
inference latency:  {best['latency_ms']:.3f} ms  (baseline {baseline_latency_ms:.3f} ms, ratio {best['ratio']:.3f}x)
baseline val acc:   {baseline_val_acc:.2f}%
epochs trained:     {best['epochs_run']} / {trial_epochs} ({epoch_note})

Architecture kwargs:
{_format_kwargs(best['arch_kwargs'])}

Training hyperparams:
{_format_kwargs(best['train_kwargs'])}

Reproduce:
  from main import Config
  from students import {STUDENTS[size].__name__}
  model = {STUDENTS[size].__name__}({arch_repr})
  cfg = Config(student_name="{size}", lr={best['train_kwargs']['lr']:.6g}, weight_decay={best['train_kwargs']['weight_decay']:.6g}, batch_size={best['train_kwargs']['batch_size']}, epochs={trial_epochs})
"""
    path.write_text(content)


def run(hcfg: HPOConfig = None):
    hcfg = hcfg or HPOConfig()
    save_dir = Path(hcfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for size in hcfg.sizes:
        print(f"\n========== {size.upper()} ==========")
        print(f"[baseline] training default architecture for {hcfg.baseline_epochs} epoch(s)...")
        base = measure_baseline(size, hcfg)
        print(f"[baseline] latency={base['latency_ms']:.3f}ms  val_acc={base['val_accuracy']:.2f}%")

        print(f"[study] running {hcfg.n_trials_per_size} trials...")
        study, best = run_study(size, hcfg, base["latency_ms"])

        pth_path = save_dir / f"optuna_{size}.pth"
        torch.save({"state_dict": best["state"], "arch_kwargs": best["arch_kwargs"]}, pth_path)

        txt_path = save_dir / f"optuna_{size}.txt"
        _write_txt(txt_path, size=size, study=study, best=best,
                   baseline_latency_ms=base["latency_ms"], baseline_val_acc=base["val_accuracy"],
                   trial_epochs=hcfg.trial_epochs)

        print(f"[done] score={best['score']:.4f}  acc={best['val_accuracy']:.2f}%  "
              f"latency={best['latency_ms']:.3f}ms  ratio={best['ratio']:.3f}x  -> {pth_path.name}")
        summary.append({
            "size": size, "score": best["score"], "val_acc": best["val_accuracy"],
            "latency_ms": best["latency_ms"], "baseline_latency_ms": base["latency_ms"],
        })

    print("\n========== SUMMARY ==========")
    print(f"{'size':<8} {'score':>8} {'val_acc':>9} {'latency_ms':>12} {'baseline_ms':>12} {'ratio':>7}")
    for s in summary:
        ratio = s["latency_ms"] / s["baseline_latency_ms"]
        print(f"{s['size']:<8} {s['score']:>8.4f} {s['val_acc']:>8.2f}% "
              f"{s['latency_ms']:>12.3f} {s['baseline_latency_ms']:>12.3f} {ratio:>7.3f}x")


if __name__ == "__main__":
    run()
