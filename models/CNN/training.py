import copy
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import trange


# ---- task helpers: "multilabel" (5 sigmoids) or "multiclass" (K-way softmax) ----

def discover_classes(loader):
    """Unique 5-bit label vectors seen in the data, as a (K, 5) tensor."""
    seen, out = set(), []
    for _, y in loader:
        for row in (y > 0.5).int():
            key = tuple(row.tolist())
            if key not in seen:
                seen.add(key)
                out.append(key)
    out.sort()
    return torch.tensor(out, dtype=torch.float32)


def encode_targets(y, classes):
    """(B, 5) multi-hot -> (B,) class index into `classes`."""
    match = ((y > 0.5).float().unsqueeze(1) == classes.unsqueeze(0)).all(-1).float()
    return match.argmax(1).long()


def to_probs5(task, logits, classes):
    """Model logits -> (B, 5) per-finger probabilities (for metrics/thresholding)."""
    if task == "multiclass":
        return logits.softmax(1) @ classes
    return torch.sigmoid(logits)


def task_loss(task, criterion, logits, y, classes):
    if task == "multiclass":
        return criterion(logits, encode_targets(y, classes))
    return criterion(logits, y)


# ---- imbalance weighting ----

def compute_pos_weight(loader, num_classes=5):
    pos, n = torch.zeros(num_classes), 0
    for _, y in loader:
        pos += (y > 0.5).float().sum(0)
        n += y.size(0)
    return ((n - pos) / pos.clamp(min=1)).clamp(max=100.0)


def compute_class_weight(loader, classes):
    k, n = classes.size(0), 0
    cnt = torch.zeros(k)
    for _, y in loader:
        cnt += torch.bincount(encode_targets(y, classes), minlength=k).float()
        n += y.size(0)
    return n / (k * cnt.clamp(min=1))


def setup_criterion(task, loader, classes, cfg, device):
    if task == "multiclass":
        weight = compute_class_weight(loader, classes).to(device) if cfg.use_pos_weight else None
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)
    pos_weight = compute_pos_weight(loader).to(device) if cfg.use_pos_weight else None
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


# ---- label-preserving augmentation on (B, C, W) feature tensors ----

def aug_from_cfg(cfg):
    a = {"noise_std": cfg.aug_noise_std, "channel_dropout": cfg.aug_channel_dropout,
         "scale_jitter": cfg.aug_scale_jitter}
    return a if any(v > 0 for v in a.values()) else None


def augment_batch(x, noise_std=0.0, channel_dropout=0.0, scale_jitter=0.0):
    if scale_jitter > 0:
        x = x * (1.0 + (torch.rand(x.size(0), 1, 1, device=x.device) * 2 - 1) * scale_jitter)
    if channel_dropout > 0:
        keep = (torch.rand(x.size(0), x.size(1), 1, device=x.device) > channel_dropout).float()
        x = x * keep
    if noise_std > 0:
        x = x + torch.randn_like(x) * noise_std * x.std(dim=2, keepdim=True)
    return x


# ---- EMA / LR schedule ----

class ModelEMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            s.copy_(s * self.decay + m * (1 - self.decay) if s.is_floating_point() else m)


def make_scheduler(optimizer, cfg):
    if getattr(cfg, "scheduler", "cosine") != "cosine" or cfg.epochs <= 0:
        return None
    warmup = getattr(cfg, "warmup_epochs", 0)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.epochs - warmup, 1))
    if warmup > 0:
        warm = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup)
        return torch.optim.lr_scheduler.SequentialLR(optimizer, [warm, cos], milestones=[warmup])
    return cos


# ---- train / eval ----

def _macro_f1(pred, target):
    tp = (pred * target).sum(0)
    fp = (pred * (1 - target)).sum(0)
    fn = ((1 - pred) * target).sum(0)
    return (2 * tp / (2 * tp + fp + fn + 1e-8)).mean().item()


def train_one_epoch(model, loader, optimizer, criterion, device, task="multilabel",
                    classes=None, aug=None, grad_clip=0.0, ema=None):
    model.train()
    total, n = 0.0, 0
    cls = classes.to(device) if classes is not None else None
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if aug:
            x = augment_batch(x, **aug)
        optimizer.zero_grad()
        loss = task_loss(task, criterion, model(x), y, cls)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        total += loss.item()
        n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, task="multilabel", classes=None):
    model.eval()
    total_loss, nb = 0.0, 0
    cls = classes.to(device) if classes is not None else None
    probs, tgts = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += task_loss(task, criterion, logits, y, cls).item()
        nb += 1
        probs.append(to_probs5(task, logits, cls).cpu())
        tgts.append(y.cpu())
    probs, tgts = torch.cat(probs), torch.cat(tgts)
    pred = (probs > 0.5).float()
    exact = (pred == tgts).all(dim=1).float().mean().item() * 100.0
    return {"loss": total_loss / max(nb, 1), "accuracy": exact, "f1": _macro_f1(pred, tgts) * 100.0}


def train_model(model, train_loader, val_loader, cfg, device, save_path=None,
                task="multilabel", classes=None):
    criterion = setup_criterion(task, train_loader, classes, cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = make_scheduler(optimizer, cfg)
    ema = ModelEMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None
    aug = aug_from_cfg(cfg)
    history, best = [], -1.0
    for epoch in trange(cfg.epochs, desc="train"):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device,
                                     task=task, classes=classes, aug=aug,
                                     grad_clip=cfg.grad_clip, ema=ema)
        eval_model = ema.shadow if ema is not None else model
        val = evaluate(eval_model, val_loader, criterion, device, task=task, classes=classes)
        if scheduler is not None:
            scheduler.step()
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}})
        print(f"[{epoch+1}/{cfg.epochs}] train_loss={train_loss:.4f}  val_loss={val['loss']:.4f}  "
              f"val_acc={val['accuracy']:.2f}%  val_f1={val['f1']:.2f}")
        if save_path and val["f1"] > best:
            best = val["f1"]
            torch.save(eval_model.state_dict(), save_path)
    if save_path and Path(save_path).exists():
        model.load_state_dict(torch.load(save_path, map_location=device))
    return model, history
