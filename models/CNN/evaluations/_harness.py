"""Fast in-memory experiment harness for the CNN pipeline.

Loads the processed tensors once onto GPU, reshapes to (N, 768, W), and provides
controlled training/eval so we can A/B feature-standardization, threshold tuning,
multiclass framing, teacher quality, and KD -- without DataLoader overhead.

Run:  python models/CNN/evaluations/_harness.py <stage>
"""
import sys, time, json, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models" / "CNN"))

from students import CNN_Nano, CNN_Micro, CNN_Base, CNN_Large, CNN_XLarge
from teachers import ResNet50_1D, ResNet101_1D, ResNet152_1D
from evaluation.metrics import compute_multilabel_metrics

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STUDENTS = {"nano": CNN_Nano, "micro": CNN_Micro, "base": CNN_Base, "large": CNN_Large, "xlarge": CNN_XLarge}
TEACHERS = {"resnet50": ResNet50_1D, "resnet101": ResNet101_1D, "resnet152": ResNet152_1D}


def set_seed(s=42):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


_CACHE = {}
def load_data():
    if _CACHE:
        return _CACHE
    P = ROOT / "datasets" / "processed"
    for split in ["train", "val", "test"]:
        d = torch.load(P / f"{split}.pt", map_location="cpu", weights_only=True)
        X, y = d["X"].float(), d["y"].float()
        b, c, w, f = X.shape
        X = X.permute(0, 1, 3, 2).reshape(b, c * f, w).contiguous()   # (N, 768, W)
        _CACHE[split] = (X.to(DEV), y.to(DEV))
    # per-row (768) standardization stats from train
    Xtr = _CACHE["train"][0]
    mean = Xtr.mean(dim=(0, 2), keepdim=True)
    std = Xtr.std(dim=(0, 2), keepdim=True).clamp_min(1e-6)
    _CACHE["stats"] = (mean, std)
    return _CACHE


def get(split, standardize=False):
    d = load_data()
    X, y = d[split]
    if standardize:
        mean, std = d["stats"]
        X = (X - mean) / std
    return X, y


# ---- multiclass helpers ----
def discover_classes(y):
    seen, out = set(), []
    for row in (y > 0.5).int().tolist():
        k = tuple(row)
        if k not in seen:
            seen.add(k); out.append(k)
    out.sort()
    return torch.tensor(out, dtype=torch.float32, device=y.device)

def encode_targets(y, classes):
    match = ((y > 0.5).float().unsqueeze(1) == classes.unsqueeze(0)).all(-1).float()
    return match.argmax(1).long()

def to_probs5(task, logits, classes):
    if task == "multiclass":
        return logits.softmax(1) @ classes
    return torch.sigmoid(logits)


# ---- metrics ----
def macro_f1_np(probs, y, thr):
    pred = (probs >= thr).astype(np.float64)
    tp = (pred * y).sum(0); fp = (pred * (1 - y)).sum(0); fn = ((1 - pred) * y).sum(0)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-9)
    return f1.mean()

def tune_thresholds(val_probs, val_y):
    """Per-finger threshold maximizing val F1; returns (5,) thresholds."""
    grid = np.linspace(0.05, 0.95, 19)
    thr = np.full(val_probs.shape[1], 0.5)
    for j in range(val_probs.shape[1]):
        best_t, best_f = 0.5, -1
        yj = val_y[:, j]
        for t in grid:
            pred = (val_probs[:, j] >= t).astype(np.float64)
            tp = (pred * yj).sum(); fp = (pred * (1 - yj)).sum(); fn = ((1 - pred) * yj).sum()
            f = 2 * tp / (2 * tp + fp + fn + 1e-9)
            if f > best_f:
                best_f, best_t = f, t
        thr[j] = best_t
    return thr


# ---- batched train/eval over in-memory GPU tensors ----
def iterate(X, y, bs, shuffle):
    n = X.size(0)
    idx = torch.randperm(n, device=X.device) if shuffle else torch.arange(n, device=X.device)
    for i in range(0, n, bs):
        j = idx[i:i + bs]
        yield X[j], y[j]

def compute_pos_weight(y):
    pos = (y > 0.5).float().sum(0); n = y.size(0)
    return ((n - pos) / pos.clamp(min=1)).clamp(max=100.0)

def compute_class_weight(y, classes):
    k = classes.size(0)
    cnt = torch.bincount(encode_targets(y, classes), minlength=k).float()
    return y.size(0) / (k * cnt.clamp(min=1))


@torch.no_grad()
def predict_probs(model, X, bs=512, task="multilabel", classes=None):
    model.eval()
    out = []
    for i in range(0, X.size(0), bs):
        out.append(to_probs5(task, model(X[i:i + bs]), classes).cpu())
    return torch.cat(out).numpy()


def augment(x, scale_jitter=0.0, channel_dropout=0.0, noise_std=0.0):
    if scale_jitter > 0:
        x = x * (1.0 + (torch.rand(x.size(0), 1, 1, device=x.device) * 2 - 1) * scale_jitter)
    if channel_dropout > 0:
        keep = (torch.rand(x.size(0), x.size(1), 1, device=x.device) > channel_dropout).float()
        x = x * keep
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std * x.std(dim=2, keepdim=True)
    return x


@torch.no_grad()
def ensemble_probs(models, X, bs=512, task="multilabel", classes=None):
    acc = None
    for m in models:
        p = predict_probs(m, X, bs=bs, task=task, classes=classes)
        acc = p if acc is None else acc + p
    return acc / len(models)


def train(model, Xtr, ytr, Xva, yva, *, epochs=60, lr=1e-3, wd=1e-4, bs=32,
          task="multilabel", classes=None, use_weight=True, verbose=False, aug=None):
    model = model.to(DEV)
    if task == "multiclass":
        w = compute_class_weight(ytr, classes).to(DEV) if use_weight else None
        crit = nn.CrossEntropyLoss(weight=w)
    else:
        pw = compute_pos_weight(ytr).to(DEV) if use_weight else None
        crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_f1, best_state = -1, None
    yva_np = (yva > 0.5).float().cpu().numpy()
    for ep in range(epochs):
        model.train()
        for xb, yb in iterate(Xtr, ytr, bs, True):
            if aug:
                xb = augment(xb, **aug)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, encode_targets(yb, classes)) if task == "multiclass" else crit(logits, yb)
            loss.backward(); opt.step()
        sched.step()
        vp = predict_probs(model, Xva, task=task, classes=classes)
        f1 = macro_f1_np(vp, yva_np, 0.5)
        if f1 > best_f1:
            best_f1, best_state = f1, copy.deepcopy(model.state_dict())
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"   ep{ep:>3} val_f1@0.5={f1*100:.2f}")
    model.load_state_dict(best_state)
    return model, best_f1


def train_kd(student, teacher, Xtr, ytr, Xva, yva, *, epochs=60, lr=1e-3, wd=1e-4, bs=32,
             task="multilabel", classes=None, T=4.0, alpha=0.5, use_weight=True):
    student = student.to(DEV); teacher = teacher.to(DEV).eval()
    if task == "multiclass":
        w = compute_class_weight(ytr, classes).to(DEV) if use_weight else None
        crit = nn.CrossEntropyLoss(weight=w)
    else:
        pw = compute_pos_weight(ytr).to(DEV) if use_weight else None
        crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best_f1, best_state = -1, None
    yva_np = (yva > 0.5).float().cpu().numpy()
    for ep in range(epochs):
        student.train()
        for xb, yb in iterate(Xtr, ytr, bs, True):
            opt.zero_grad()
            s = student(xb)
            with torch.no_grad():
                t = teacher(xb)
            if task == "multiclass":
                hard = crit(s, encode_targets(yb, classes))
                soft = F.kl_div(F.log_softmax(s / T, 1), F.softmax(t / T, 1),
                                reduction="batchmean") * (T * T)
            else:
                hard = crit(s, yb)
                soft = F.binary_cross_entropy_with_logits(s / T, torch.sigmoid(t / T)) * (T * T)
            (alpha * hard + (1 - alpha) * soft).backward(); opt.step()
        sched.step()
        vp = predict_probs(student, Xva, task=task, classes=classes)
        f1 = macro_f1_np(vp, yva_np, 0.5)
        if f1 > best_f1:
            best_f1, best_state = f1, copy.deepcopy(student.state_dict())
    student.load_state_dict(best_state)
    return student, best_f1


def full_eval(model, *, standardize, task="multilabel", classes=None, tune=True, tag=""):
    Xva, yva = get("val", standardize); Xte, yte = get("test", standardize)
    vp = predict_probs(model, Xva, task=task, classes=classes)
    tp = predict_probs(model, Xte, task=task, classes=classes)
    yva_np = (yva > 0.5).float().cpu().numpy(); yte_np = (yte > 0.5).float().cpu().numpy()
    m05 = compute_multilabel_metrics(torch.tensor(tp), torch.tensor(yte_np), threshold=0.5)
    res = {"tag": tag, "f1@0.5": m05["f1_macro"], "acc@0.5": m05["accuracy"],
           "finger_acc": m05["finger_accuracy"], "auprc": m05["auprc_macro"],
           "auroc": m05["auroc_macro"], "recall@0.5": m05["recall_macro"], "prec@0.5": m05["precision_macro"]}
    if tune:
        thr = tune_thresholds(vp, yva_np)
        f1_tuned = macro_f1_np(tp, yte_np, thr)
        res["f1@tuned"] = float(f1_tuned)
        res["thr"] = [round(float(t), 2) for t in thr]
    return res


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def pr(res):
    base = (f"{res['tag']:<34} f1@0.5={res['f1@0.5']*100:5.2f}  "
            f"f1@tuned={res.get('f1@tuned', float('nan'))*100:5.2f}  "
            f"acc={res['acc@0.5']*100:5.2f}  fingacc={res['finger_acc']*100:5.2f}  "
            f"auprc={res['auprc']*100:5.2f}  auroc={res['auroc']*100:5.2f}  "
            f"R={res['recall@0.5']*100:5.2f} P={res['prec@0.5']*100:5.2f}")
    print(base)
    return res
