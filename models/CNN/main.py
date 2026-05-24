import sys
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datasets.loaders import make_dataloaders, LoaderConfig
from evaluation.metrics import compute_multilabel_metrics

from students import CNN_Nano, CNN_Micro, CNN_Base, CNN_Large, CNN_XLarge
from teachers import ResNet50_1D, ResNet101_1D, ResNet152_1D
from training import train_model
from distillation import train_student_kd


STUDENTS = {"nano": CNN_Nano, "micro": CNN_Micro, "base": CNN_Base, "large": CNN_Large, "xlarge": CNN_XLarge}
TEACHERS = {"resnet50": ResNet50_1D, "resnet101": ResNet101_1D, "resnet152": ResNet152_1D}


@dataclass
class Config:
    mode: str = "student"                # "student" | "teacher" | "student_kd"
    student_name: str = "nano"           # nano | micro | base | large | xlarge
    teacher_name: str = "resnet50"       # resnet50 | resnet101 | resnet152

    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4

    kd_temperature: float = 4.0
    kd_alpha: float = 0.5
    teacher_ckpt: Optional[str] = None   # if None, teacher is pretrained on the fly
    teacher_epochs: int = 30

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    save_dir: str = "models/CNN/checkpoints"


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _reshape(loader):
    for x, y in loader:
        b, c, w, f = x.shape
        yield x.permute(0, 1, 3, 2).reshape(b, c * f, w).contiguous(), y


class _ReshapedLoader:
    def __init__(self, loader):
        self.loader = loader
    def __iter__(self):
        return _reshape(self.loader)
    def __len__(self):
        return len(self.loader)


def get_loaders(cfg: Config):
    loader_cfg = LoaderConfig(batch_size=cfg.batch_size, seed=cfg.seed,
                              pin_memory=(cfg.device == "cuda"))
    train, val, test = make_dataloaders(cfg=loader_cfg)
    return _ReshapedLoader(train), _ReshapedLoader(val), _ReshapedLoader(test)


def _save_path(cfg: Config, name: str) -> str:
    Path(cfg.save_dir).mkdir(parents=True, exist_ok=True)
    return f"{cfg.save_dir}/{name}.pth"


@torch.no_grad()
def _summary_metrics(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device)
        preds.append(torch.sigmoid(model(x)).cpu())
        targets.append(y)
    return compute_multilabel_metrics(torch.cat(preds), torch.cat(targets))


def train_student(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    train_loader, val_loader, test_loader = get_loaders(cfg)
    model = STUDENTS[cfg.student_name]().to(device)
    print(f"student={cfg.student_name}  params={sum(p.numel() for p in model.parameters()):,}")
    save_path = _save_path(cfg, f"student_{cfg.student_name}")
    model, _ = train_model(model, train_loader, val_loader, cfg, device, save_path=save_path)
    metrics = _summary_metrics(model, test_loader, device)
    print(f"test  acc={metrics['accuracy']:.3f}  f1_macro={metrics['f1_macro']:.3f}  auprc={metrics['auprc_macro']:.3f}")
    return model


def train_teacher(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    train_loader, val_loader, test_loader = get_loaders(cfg)
    teacher_epochs = cfg.teacher_epochs if cfg.mode == "student_kd" else cfg.epochs
    model = TEACHERS[cfg.teacher_name]().to(device)
    print(f"teacher={cfg.teacher_name}  params={sum(p.numel() for p in model.parameters()):,}")
    save_path = _save_path(cfg, f"teacher_{cfg.teacher_name}")
    teacher_cfg = Config(**{**cfg.__dict__, "epochs": teacher_epochs})
    model, _ = train_model(model, train_loader, val_loader, teacher_cfg, device, save_path=save_path)
    metrics = _summary_metrics(model, test_loader, device)
    print(f"test  acc={metrics['accuracy']:.3f}  f1_macro={metrics['f1_macro']:.3f}  auprc={metrics['auprc_macro']:.3f}")
    return model


def train_student_with_kd(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    train_loader, val_loader, test_loader = get_loaders(cfg)

    teacher = TEACHERS[cfg.teacher_name]().to(device)
    if cfg.teacher_ckpt and Path(cfg.teacher_ckpt).exists():
        teacher.load_state_dict(torch.load(cfg.teacher_ckpt, map_location=device))
        print(f"loaded teacher checkpoint: {cfg.teacher_ckpt}")
    else:
        print(f"pretraining teacher={cfg.teacher_name} for {cfg.teacher_epochs} epochs")
        teacher_cfg = Config(**{**cfg.__dict__, "epochs": cfg.teacher_epochs})
        teacher, _ = train_model(teacher, train_loader, val_loader, teacher_cfg, device,
                                 save_path=_save_path(cfg, f"teacher_{cfg.teacher_name}"))

    student = STUDENTS[cfg.student_name]().to(device)
    print(f"student={cfg.student_name}  params={sum(p.numel() for p in student.parameters()):,}")
    save_path = _save_path(cfg, f"student_kd_{cfg.student_name}_from_{cfg.teacher_name}")
    student, _ = train_student_kd(student, teacher, train_loader, val_loader, cfg, device, save_path=save_path)
    metrics = _summary_metrics(student, test_loader, device)
    print(f"test  acc={metrics['accuracy']:.3f}  f1_macro={metrics['f1_macro']:.3f}  auprc={metrics['auprc_macro']:.3f}")
    return student


def main():
    cfg = Config()
    {"student": train_student, "teacher": train_teacher, "student_kd": train_student_with_kd}[cfg.mode](cfg)


if __name__ == "__main__":
    main()
