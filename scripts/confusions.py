
import os, json, argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from training.train import TrainConfig, build_model, load_checkpoint
from datasets.loaders import make_dataloaders, LoaderConfig
from adapters.feature_to_sequence import feature_tensor_to_sequences


def _filter_cfg_dict(d: dict) -> dict:
    """Keep only keys that exist in TrainConfig (so we can do TrainConfig(**d) safely)."""
    allowed = set(TrainConfig().__dict__.keys())
    return {k: v for k, v in d.items() if k in allowed}


@torch.no_grad()
def collect_probs_targets(model, test_loader, device, use_feature_adapter: bool):
    model.eval()
    all_probs, all_targets = [], []

    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device).float()
        if use_feature_adapter:
            x = feature_tensor_to_sequences(x)

        logits = model(x)
        probs = torch.sigmoid(logits)

        all_probs.append(probs.detach().cpu())
        all_targets.append(y.detach().cpu())

    probs = torch.cat(all_probs, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()
    return probs, targets


def confusion_counts(y_true_bin: np.ndarray, y_pred_bin: np.ndarray):
    # y_*: (N, 5) in {0,1}
    out = []
    for j in range(y_true_bin.shape[1]):
        yt = y_true_bin[:, j]
        yp = y_pred_bin[:, j]
        tp = int(((yp == 1) & (yt == 1)).sum())
        tn = int(((yp == 0) & (yt == 0)).sum())
        fp = int(((yp == 1) & (yt == 0)).sum())
        fn = int(((yp == 0) & (yt == 1)).sum())
        out.append({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    return out


def plot_cm(cm2x2, title: str, save_path: str): # emily made the colours prettier
    # cm2x2 = [[tn, fp],[fn, tp]]
    arr = np.array(cm2x2, dtype=int)

    plt.figure(figsize=(5, 4))

    plt.imshow(
        arr,
        cmap="Blues",  # cleaner sequential colormap
        vmin=0,
        interpolation="nearest"
    )
    plt.colorbar()

    plt.title(title, fontsize=14)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"], fontsize=11)
    plt.yticks([0, 1], ["True 0", "True 1"], fontsize=11)

    for (i, j), v in np.ndenumerate(arr):
        plt.text(
            j, i, str(v),
            ha="center", va="center",
            fontsize=13,
            color="white" if v > arr.max() / 2 else "black"
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="training/runs/cnn_XXXXXXX")
    ap.add_argument("--ckpt", default="training/ckpts/cnn_best.pt")
    ap.add_argument("--out_dir", default="results/cnn")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    run_path = Path(args.run)
    cfg_path = run_path / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing {cfg_path}")

    cfg_dict = json.loads(cfg_path.read_text())
    cfg = TrainConfig(**_filter_cfg_dict(cfg_dict))

    device = torch.device(cfg.device)
    loader_cfg = LoaderConfig(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
        out_dim=cfg.out_dim,
    )
    train_loader, val_loader, test_loader = make_dataloaders(
        processed_dir=cfg.processed_dir,
        train_file=cfg.train_file,
        val_file=cfg.val_file,
        test_file=cfg.test_file,
        cfg=loader_cfg,
    )

    x0, _ = next(iter(train_loader))
    if cfg.use_feature_adapter:
        x0 = feature_tensor_to_sequences(x0)
    model = build_model(cfg, x0).to(device)

    load_checkpoint(args.ckpt, model=model)

    probs, targets = collect_probs_targets(model, test_loader, device, cfg.use_feature_adapter)
    y_pred_bin = (probs >= args.threshold).astype(np.int32)
    y_true_bin = targets.astype(np.int32)

    counts = confusion_counts(y_true_bin, y_pred_bin)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # save raw counts
    (out_dir / "confusion_counts.json").write_text(json.dumps({
        "threshold": args.threshold,
        "per_finger": counts
    }, indent=2))

    # per-finger plots
    for j, c in enumerate(counts):
        cm = [[c["tn"], c["fp"]], [c["fn"], c["tp"]]]
        plot_cm(cm, f"finger_{j} confusion (thr={args.threshold})", str(out_dir / f"confusion_finger_{j}.png"))

    print("Wrote:")
    print(" -", out_dir / "confusion_counts.json")
    for j in range(len(counts)):
        print(" -", out_dir / f"confusion_finger_{j}.png")


if __name__ == "__main__":
    main()