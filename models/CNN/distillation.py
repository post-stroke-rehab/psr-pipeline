from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from training import (evaluate, setup_criterion, make_scheduler, aug_from_cfg,
                      augment_batch, task_loss, ModelEMA)


def distillation_loss(student_logits, teacher_logits, y, T, alpha, task, classes, criterion):
    hard = task_loss(task, criterion, student_logits, y, classes)
    if task == "multiclass":
        soft = F.kl_div(F.log_softmax(student_logits / T, dim=1),
                        F.softmax(teacher_logits / T, dim=1),
                        reduction="batchmean") * (T * T)
    else:
        soft = F.binary_cross_entropy_with_logits(student_logits / T,
                                                  torch.sigmoid(teacher_logits / T)) * (T * T)
    return alpha * hard + (1 - alpha) * soft


def train_one_epoch_kd(student, teacher, loader, optimizer, criterion, device, cfg,
                       task, classes, aug, ema):
    student.train()
    teacher.eval()
    cls = classes.to(device) if classes is not None else None
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if aug:
            x = augment_batch(x, **aug)
        optimizer.zero_grad()
        s_logits = student(x)
        with torch.no_grad():
            t_logits = teacher(x)
        loss = distillation_loss(s_logits, t_logits, y, cfg.kd_temperature, cfg.kd_alpha,
                                 task, cls, criterion)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
        optimizer.step()
        if ema is not None:
            ema.update(student)
        total += loss.item()
        n += 1
    return total / max(n, 1)


def train_student_kd(student, teacher, train_loader, val_loader, cfg, device, save_path=None,
                     task="multilabel", classes=None):
    criterion = setup_criterion(task, train_loader, classes, cfg, device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = make_scheduler(optimizer, cfg)
    ema = ModelEMA(student, cfg.ema_decay) if cfg.ema_decay > 0 else None
    aug = aug_from_cfg(cfg)
    history, best = [], -1.0
    for epoch in trange(cfg.epochs, desc="kd"):
        train_loss = train_one_epoch_kd(student, teacher, train_loader, optimizer, criterion,
                                        device, cfg, task, classes, aug, ema)
        eval_model = ema.shadow if ema is not None else student
        val = evaluate(eval_model, val_loader, criterion, device, task=task, classes=classes)
        if scheduler is not None:
            scheduler.step()
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}})
        print(f"[{epoch+1}/{cfg.epochs}] kd_loss={train_loss:.4f}  val_loss={val['loss']:.4f}  "
              f"val_acc={val['accuracy']:.2f}%  val_f1={val['f1']:.2f}")
        if save_path and val["f1"] > best:
            best = val["f1"]
            torch.save(eval_model.state_dict(), save_path)
    if save_path and Path(save_path).exists():
        student.load_state_dict(torch.load(save_path, map_location=device))
    return student, history
