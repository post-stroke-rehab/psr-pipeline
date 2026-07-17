from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import optuna
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.train import (
    HEALTHY_PROCESSED_DIR,
    IMPAIRED_PROCESSED_DIR,
    TRAINING_STAGES,
    TrainConfig,
    train_loop,
)

TUNING_ROOT = REPO_ROOT / "training" / "tuning"
SUPPORTED_MODELS = ("lstm", "cnn", "gnn")

ARCH_LOCK_FIELDS: Dict[str, List[str]] = {
    "lstm": ["lstm_hidden1", "lstm_hidden2", "lstm_fc_hidden"],
    "cnn": ["cnn_variant"],
    "gnn": [
        "gnn_type",
        "gnn_num_layers",
        "gnn_hid_dim",
        "gnn_aggregation",
        "gnn_readout_layers",
    ],
}


def _default_processed_dir(stage: str) -> str:
    if stage == "pretrain":
        return HEALTHY_PROCESSED_DIR
    return IMPAIRED_PROCESSED_DIR


def _pretrain_ckpt_path(model: str) -> Path:
    return REPO_ROOT / "results" / "best_pretrain" / model / "checkpoint_best.pt"


def _tuning_root(model: str) -> Path:
    return TUNING_ROOT / model


def _load_checkpoint_config(ckpt_path: str) -> Dict[str, Any]:
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    return payload.get("config", {})


def _locked_arch(model: str, ckpt_cfg: Dict[str, Any]) -> Dict[str, Any]:
    locked: Dict[str, Any] = {}
    for field in ARCH_LOCK_FIELDS[model]:
        if field in ckpt_cfg:
            locked[field] = ckpt_cfg[field]
    return locked


def _suggest_lstm(
    trial: optuna.Trial,
    stage: str,
    *,
    fixed_arch: Optional[Dict[str, Any]] = None,
    search_arch: bool = False,
) -> Dict[str, Any]:
    if fixed_arch is None or search_arch:
        params: Dict[str, Any] = {
            "lstm_hidden1": trial.suggest_categorical("hidden1", [64, 128, 192]),
            "lstm_hidden2": trial.suggest_categorical("hidden2", [128, 256, 384]),
            "lstm_fc_hidden": trial.suggest_categorical("fc_hidden", [64, 128, 192]),
        }
    else:
        params = dict(fixed_arch)

    params.update(
        {
            "lstm_dropout": trial.suggest_float("dropout", 0.30, 0.60, step=0.05),
            "lr": trial.suggest_float("lr", 1e-4, 2e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "patience": trial.suggest_int("patience", 15, 35, step=5),
        }
    )
    if stage == "finetune":
        params["freeze_backbone_epochs"] = trial.suggest_int("freeze_backbone_epochs", 0, 10, step=2)
        params["backbone_lr"] = trial.suggest_float("backbone_lr", 1e-5, 1e-4, log=True)
        params["head_lr"] = trial.suggest_float("head_lr", 1e-4, 2e-3, log=True)
        params["finger0_pos_weight_boost"] = trial.suggest_float(
            "finger0_pos_weight_boost", 1.0, 2.0, step=0.1
        )
        params["pretrained_init"] = "backbone"
    return params


def _suggest_cnn(
    trial: optuna.Trial,
    stage: str,
    *,
    fixed_arch: Optional[Dict[str, Any]] = None,
    search_arch: bool = False,
) -> Dict[str, Any]:
    if fixed_arch is None or search_arch:
        params: Dict[str, Any] = {
            "cnn_variant": trial.suggest_categorical(
                "cnn_variant", ["nano", "micro", "base", "large"]
            ),
        }
    else:
        params = dict(fixed_arch)

    params.update(
        {
            "cnn_dropout": trial.suggest_float("dropout", 0.10, 0.50, step=0.05),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "patience": trial.suggest_int("patience", 15, 35, step=5),
            "finger0_pos_weight_boost": trial.suggest_float(
                "finger0_pos_weight_boost", 1.0, 2.0, step=0.1
            ),
        }
    )
    if stage == "finetune":
        params["pretrained_init"] = "full"
    return params


def _suggest_gnn(
    trial: optuna.Trial,
    stage: str,
    *,
    fixed_arch: Optional[Dict[str, Any]] = None,
    search_arch: bool = False,
) -> Dict[str, Any]:
    if fixed_arch is None or search_arch:
        params: Dict[str, Any] = {
            "gnn_type": trial.suggest_categorical("gnn_type", ["gcn", "sage", "gat"]),
            "gnn_num_layers": trial.suggest_int("gnn_num_layers", 1, 4),
            "gnn_hid_dim": trial.suggest_categorical("gnn_hid_dim", [32, 64, 128, 256]),
            "gnn_aggregation": trial.suggest_categorical("gnn_aggregation", ["mean", "max", "add"]),
            "gnn_readout_layers": trial.suggest_int("gnn_readout_layers", 1, 3),
        }
    else:
        params = dict(fixed_arch)

    params.update(
        {
            "gnn_dropout": trial.suggest_float("dropout", 0.10, 0.70, step=0.1),
            "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
            "patience": trial.suggest_int("patience", 15, 40, step=5),
            "finger0_pos_weight_boost": trial.suggest_float(
                "finger0_pos_weight_boost", 1.0, 2.0, step=0.1
            ),
        }
    )
    if stage == "finetune":
        params["pretrained_init"] = "full"
    return params


def _suggest_params(
    trial: optuna.Trial,
    model: str,
    stage: str,
    *,
    fixed_arch: Optional[Dict[str, Any]] = None,
    search_arch: bool = False,
) -> Dict[str, Any]:
    if model == "lstm":
        return _suggest_lstm(trial, stage, fixed_arch=fixed_arch, search_arch=search_arch)
    if model == "cnn":
        return _suggest_cnn(trial, stage, fixed_arch=fixed_arch, search_arch=search_arch)
    if model == "gnn":
        return _suggest_gnn(trial, stage, fixed_arch=fixed_arch, search_arch=search_arch)
    raise ValueError(f"Unsupported model={model}")


def _build_cfg(args: argparse.Namespace, trial: optuna.Trial) -> TrainConfig:
    params = _suggest_params(
        trial,
        args.model,
        args.stage,
        fixed_arch=args.locked_arch,
        search_arch=args.search_arch,
    )
    processed_dir = args.processed_dir or _default_processed_dir(args.stage)
    root = _tuning_root(args.model)

    cfg = TrainConfig(
        model_name=args.model,
        training_stage=args.stage,
        processed_dir=processed_dir,
        device=args.device,
        seed=args.seed,
        epochs=args.epochs,
        early_stop_metric="finger_accuracy",
        tune_threshold=False,
        save_curves=False,
        save_training_curves=False,
        verbose_data_stats=False,
        pretrained_ckpt_path=args.pretrained_ckpt,
        ckpt_dir=str(root / "trials" / f"{args.stage}_trial_{trial.number:04d}" / "ckpts"),
        run_dir=str(root / "trials" / f"{args.stage}_trial_{trial.number:04d}" / "runs"),
        results_dir=str(root / "trials" / f"{args.stage}_trial_{trial.number:04d}" / "results"),
    )
    for key, value in params.items():
        setattr(cfg, key, value)
    return cfg


def _objective(trial: optuna.Trial, args: argparse.Namespace) -> float:
    cfg = _build_cfg(args, trial)
    if args.stage == "finetune" and not cfg.pretrained_ckpt_path:
        raise ValueError("finetune tuning requires --pretrained-ckpt")

    lr_msg = (
        f"head_lr={cfg.head_lr:.2e} backbone_lr={cfg.backbone_lr:.2e}"
        if args.model == "lstm" and args.stage == "finetune"
        else f"lr={cfg.lr:.2e}"
    )
    arch_msg = ""
    if args.model == "lstm":
        arch_msg = f"hidden=({cfg.lstm_hidden1},{cfg.lstm_hidden2},{cfg.lstm_fc_hidden})"
    elif args.model == "cnn":
        arch_msg = f"variant={cfg.cnn_variant}"
    else:
        arch_msg = f"type={cfg.gnn_type} hid={cfg.gnn_hid_dim}"

    print(
        f"\n[Trial {trial.number}] {args.model}/{args.stage} {arch_msg} "
        f"{lr_msg} bs={cfg.batch_size}"
    )

    out = train_loop(
        cfg,
        optuna_trial=trial,
        tune_max_epochs=args.tune_epochs,
        tune_skip_test=True,
        apply_defaults=False,
    )
    return float(out["val_score"])


def _cfg_from_best_params(args: argparse.Namespace, params: Dict[str, Any]) -> TrainConfig:
    processed_dir = args.processed_dir or _default_processed_dir(args.stage)
    cfg = TrainConfig(
        model_name=args.model,
        training_stage=args.stage,
        processed_dir=processed_dir,
        device=args.device,
        seed=args.seed,
        epochs=args.epochs,
        early_stop_metric="finger_accuracy",
        tune_threshold=True,
        save_curves=True,
        save_training_curves=True,
        verbose_data_stats=True,
        pretrained_ckpt_path=args.pretrained_ckpt,
        pretrained_init="backbone" if args.model == "lstm" else "full",
    )
    mapping = {
        "hidden1": "lstm_hidden1",
        "hidden2": "lstm_hidden2",
        "fc_hidden": "lstm_fc_hidden",
        "dropout": "lstm_dropout",
        "cnn_variant": "cnn_variant",
        "gnn_type": "gnn_type",
        "gnn_num_layers": "gnn_num_layers",
        "gnn_hid_dim": "gnn_hid_dim",
        "gnn_aggregation": "gnn_aggregation",
        "gnn_readout_layers": "gnn_readout_layers",
    }
    for key, value in params.items():
        if key == "dropout":
            if args.model == "lstm":
                setattr(cfg, "lstm_dropout", value)
            elif args.model == "cnn":
                setattr(cfg, "cnn_dropout", value)
            else:
                setattr(cfg, "gnn_dropout", value)
            continue
        attr = mapping.get(key, key)
        if hasattr(cfg, attr):
            setattr(cfg, attr, value)
    if args.locked_arch and not args.search_arch:
        for key, value in args.locked_arch.items():
            setattr(cfg, key, value)
    return cfg


def _write_summary(study: optuna.Study, args: argparse.Namespace, best_cfg: TrainConfig) -> None:
    root = _tuning_root(args.model)
    root.mkdir(parents=True, exist_ok=True)
    summary = {
        "model": args.model,
        "stage": args.stage,
        "seed": args.seed,
        "n_trials": len(study.trials),
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_config": asdict(best_cfg),
    }
    out_path = root / f"best_{args.stage}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    txt_path = root / f"best_{args.stage}.txt"
    txt_path.write_text(
        "\n".join(
            [
                f"=== {args.model.upper()} Optuna ({args.stage}) ===",
                f"trials: {len(study.trials)}",
                f"best val finger_accuracy: {study.best_value:.4f}",
                "best params:",
                *[f"  {k}: {v}" for k, v in study.best_params.items()],
            ]
        ),
        encoding="utf-8",
    )
    print(f"[Saved] {out_path}")
    print(f"[Saved] {txt_path}")


def _prepare_stage_args(args: argparse.Namespace) -> None:
    args.locked_arch = None
    if args.stage == "finetune" and not args.pretrained_ckpt:
        raise ValueError("finetune tuning requires --pretrained-ckpt")
    if args.stage == "finetune" and not args.search_arch:
        ckpt_cfg = _load_checkpoint_config(args.pretrained_ckpt)
        args.locked_arch = _locked_arch(args.model, ckpt_cfg)
        print(f"[Finetune] Locked {args.model} architecture: {args.locked_arch}")


def _run_stage(args: argparse.Namespace) -> Dict[str, Any]:
    _prepare_stage_args(args)
    root = _tuning_root(args.model)
    study_name = args.study_name or f"{args.model}_{args.stage}_finger_acc_v2"
    storage = f"sqlite:///{(root / f'{study_name}.db').as_posix()}"
    root.mkdir(parents=True, exist_ok=True)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )

    print(
        f"[Optuna] model={args.model} stage={args.stage} trials={args.n_trials} "
        f"tune_epochs={args.tune_epochs} metric=finger_accuracy"
    )
    study.optimize(
        lambda trial: _objective(trial, args),
        n_trials=args.n_trials,
        catch=(RuntimeError, ValueError),
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError(f"No {args.model}/{args.stage} trials completed successfully.")

    print(f"\n[Optuna:{args.model}/{args.stage}] Best trial:")
    print(f"  val finger_accuracy: {study.best_value:.4f}")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    best_cfg = _cfg_from_best_params(args, study.best_params)
    _write_summary(study, args, best_cfg)

    result: Dict[str, Any] = {
        "model": args.model,
        "stage": args.stage,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "best_config": asdict(best_cfg),
    }

    if args.skip_final_train:
        return result

    print(f"\n[Final train:{args.model}/{args.stage}] Best hyperparameters...")
    final_out = train_loop(best_cfg, apply_defaults=False)
    test_metrics = final_out["test_metrics"] or {}
    result["test_metrics"] = test_metrics
    result["results_path"] = final_out.get("results_path")
    print(
        f"[Final test:{args.model}/{args.stage}] finger_accuracy="
        f"{test_metrics.get('finger_accuracy', float('nan')):.4f} "
        f"f1_macro={test_metrics.get('f1_macro', float('nan')):.4f}"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna two-stage tuning for LSTM/CNN/GNN.")
    parser.add_argument("--model", default="lstm", choices=SUPPORTED_MODELS)
    parser.add_argument("--stage", default="finetune", choices=list(TRAINING_STAGES))
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--pretrained-ckpt", default=None)
    parser.add_argument(
        "--both-stages",
        action="store_true",
        help="Pretrain Optuna + final train, then finetune Optuna + final train.",
    )
    parser.add_argument(
        "--search-arch",
        action="store_true",
        help="Search architecture during finetune (LSTM: backbone init only).",
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--tune-epochs", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--skip-final-train", action="store_true")
    args = parser.parse_args()

    if args.both_stages:
        pretrain_args = argparse.Namespace(**vars(args))
        pretrain_args.stage = "pretrain"
        pretrain_args.pretrained_ckpt = None
        pretrain_args.search_arch = False
        pretrain_args.study_name = args.study_name or f"{args.model}_pretrain_finger_acc_v2"
        print(f"\n========== Stage 1/2: {args.model.upper()} Pretrain ==========")
        pretrain_result = _run_stage(pretrain_args)

        ckpt_path = _pretrain_ckpt_path(args.model)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Missing pretrain checkpoint: {ckpt_path}")

        finetune_args = argparse.Namespace(**vars(args))
        finetune_args.stage = "finetune"
        finetune_args.pretrained_ckpt = str(ckpt_path)
        finetune_args.search_arch = False
        finetune_args.study_name = None
        if args.study_name:
            finetune_args.study_name = f"{args.study_name}_finetune"
        else:
            finetune_args.study_name = f"{args.model}_finetune_finger_acc_v2"
        print(f"\n========== Stage 2/2: {args.model.upper()} Finetune ==========")
        finetune_result = _run_stage(finetune_args)

        summary_path = _tuning_root(args.model) / "both_stages_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                {"pretrain": pretrain_result, "finetune": finetune_result},
                f,
                indent=2,
                default=str,
            )
        print(f"\n[Done] {args.model} both stages complete. Summary: {summary_path}")
        return

    if args.stage == "finetune" and not args.pretrained_ckpt:
        parser.error("finetune tuning requires --pretrained-ckpt (or use --both-stages)")

    _run_stage(args)


if __name__ == "__main__":
    main()
