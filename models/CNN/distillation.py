import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from training import evaluate


def distillation_loss(student_logits, teacher_logits, y, T, alpha):
    hard = F.binary_cross_entropy_with_logits(student_logits, y)
    soft = F.mse_loss(torch.sigmoid(student_logits / T), torch.sigmoid(teacher_logits / T)) * (T * T)
    return alpha * hard + (1 - alpha) * soft


def train_one_epoch_kd(student, teacher, loader, optimizer, device, T, alpha):
    student.train()
    teacher.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        s_logits = student(x)
        with torch.no_grad():
            t_logits = teacher(x)
        loss = distillation_loss(s_logits, t_logits, y, T, alpha)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return total / max(n, 1)


def train_student_kd(student, teacher, train_loader, val_loader, cfg, device, save_path=None):
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    eval_criterion = nn.BCEWithLogitsLoss()
    history, best_acc = [], -1.0
    for epoch in trange(cfg.epochs, desc="kd"):
        train_loss = train_one_epoch_kd(student, teacher, train_loader, optimizer, device, cfg.kd_temperature, cfg.kd_alpha)
        val = evaluate(student, val_loader, eval_criterion, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}})
        print(f"[{epoch+1}/{cfg.epochs}] kd_loss={train_loss:.4f}  val_loss={val['loss']:.4f}  val_acc={val['accuracy']:.2f}%")
        if save_path and val["accuracy"] > best_acc:
            best_acc = val["accuracy"]
            torch.save(student.state_dict(), save_path)
    return student, history
