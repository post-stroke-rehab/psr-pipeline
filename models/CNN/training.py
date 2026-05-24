import torch
import torch.nn as nn
from tqdm import trange


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item()
        pred = (torch.sigmoid(logits) > 0.5).float()
        correct += (pred == y).all(dim=1).sum().item()
        total += y.size(0)
    return {"loss": total_loss / max(len(loader), 1), "accuracy": 100.0 * correct / max(total, 1)}


def train_model(model, train_loader, val_loader, cfg, device, save_path=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    history, best_acc = [], -1.0
    for epoch in trange(cfg.epochs, desc="train"):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val = evaluate(model, val_loader, criterion, device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}})
        print(f"[{epoch+1}/{cfg.epochs}] train_loss={train_loss:.4f}  val_loss={val['loss']:.4f}  val_acc={val['accuracy']:.2f}%")
        if save_path and val["accuracy"] > best_acc:
            best_acc = val["accuracy"]
            torch.save(model.state_dict(), save_path)
    return model, history
