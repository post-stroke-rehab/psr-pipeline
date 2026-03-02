#!/usr/bin/env python3
"""
SIMPLE USAGE:
    1. Edit the "USER CONFIGURATION" section below
    2. Edit config.py for model/training settings
    3. Run: python main_simple.py

That's it!
═══════════════════════════════════════════════════════════════════════════
"""

import torch
from pathlib import Path
from datetime import datetime

# Your config
from config import Config

# Data loading (your colleague's code)
from loaders import make_dataloaders, DataConfig, SplitConfig
from data_processing.preprocess_config import PreprocessConfig

# Model building and training
from utils import set_seed, build_student, get_device, save_model
from teachers import build_teacher
from training import train_model
from distillation import train_with_distillation, pretrain_teacher


# =============================================================================
# 📝 USER CONFIGURATION - EDIT THIS SECTION!
# =============================================================================

class UserConfig:
    """
    ═══════════════════════════════════════════════════════════════════
    EDIT THESE SETTINGS TO CUSTOMIZE YOUR TRAINING
    ═══════════════════════════════════════════════════════════════════
    """
    
    # ========== FILE PATHS & NAMING ==========
    # Where to find your raw data
    RAW_DATA_DIR = "datasets/raw"
    
    # Where processed data is cached
    PROCESSED_DATA_DIR = "datasets/processed"
    
    # Where to save trained models
    MODELS_DIR = "models"
    
    # Model file naming template (you can use these variables):
    # {model} = student model name (e.g., "nano")
    # {mode} = "baseline" or "distilled"
    # {timestamp} = current date/time
    # {teacher} = teacher model name
    # {lr} = learning rate
    # {epochs} = number of epochs
    
    # Example naming schemes (uncomment the one you want):
    
    # Simple naming (default):
    MODEL_NAME_TEMPLATE = "{model}_{mode}"
    # Result: nano_baseline.pth, nano_distilled.pth
    
    # With timestamp:
    # MODEL_NAME_TEMPLATE = "{model}_{mode}_{timestamp}"
    # Result: nano_baseline_20240302_143025.pth
    
    # With hyperparameters:
    # MODEL_NAME_TEMPLATE = "{model}_{mode}_lr{lr}_e{epochs}"
    # Result: nano_baseline_lr0.001_e50.pth
    
    # Fully descriptive:
    # MODEL_NAME_TEMPLATE = "{model}_{mode}_teacher{teacher}_lr{lr}_e{epochs}_{timestamp}"
    # Result: nano_distilled_teacherresnet50_lr0.001_e50_20240302_143025.pth
    
    # Custom experiment name (uncomment to use):
    # EXPERIMENT_NAME = "experiment_01"
    # MODEL_NAME_TEMPLATE = "{experiment}_{model}_{mode}"
    # Result: experiment_01_nano_baseline.pth
    
    # ========== DATA LOADING SETTINGS ==========
    # Data split ratios (must sum to 1.0)
    TRAIN_SPLIT = 0.8   # 80% for training
    VAL_SPLIT = 0.1     # 10% for validation
    TEST_SPLIT = 0.1    # 10% for testing
    
    # Number of CPU workers for data loading (0 = main thread only)
    NUM_WORKERS = 0     # Set to 2-4 if you have multiple CPU cores
    
    # ========== EXPERIMENT TRACKING ==========
    # Set to True to add timestamp to all saved files
    ADD_TIMESTAMP = False
    
    # Set to True to save both "best" and "final" models
    SAVE_FINAL_MODEL = True
    
    """
    ═══════════════════════════════════════════════════════════════════
    END OF USER CONFIGURATION
    Don't change anything below unless you know what you're doing!
    ═══════════════════════════════════════════════════════════════════
    """


# =============================================================================
# HELPER FUNCTIONS - Don't change these
# =============================================================================

def get_model_filename(cfg: Config, mode: str) -> str:
    """
    Generate model filename based on user template.
    
    Args:
        cfg: Config object
        mode: "baseline" or "distilled"
    
    Returns:
        Filename with .pth extension
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Build replacement dict
    replacements = {
        'model': cfg.student_model,
        'mode': mode,
        'timestamp': timestamp,
        'teacher': cfg.teacher_model if cfg.use_distillation else 'none',
        'lr': str(cfg.lr).replace('.', 'p'),  # 0.001 -> 0p001
        'epochs': cfg.epochs,
    }
    
    # Add experiment name if defined
    if hasattr(UserConfig, 'EXPERIMENT_NAME'):
        replacements['experiment'] = UserConfig.EXPERIMENT_NAME
    
    # Apply template
    filename = UserConfig.MODEL_NAME_TEMPLATE
    for key, value in replacements.items():
        filename = filename.replace(f"{{{key}}}", str(value))
    
    return f"{filename}.pth"


def get_model_path(cfg: Config, mode: str, final: bool = False) -> str:
    """Get full path for saving model."""
    filename = get_model_filename(cfg, mode)
    
    if final and UserConfig.SAVE_FINAL_MODEL:
        filename = filename.replace('.pth', '_final.pth')
    
    return f"{UserConfig.MODELS_DIR}/{filename}"


# =============================================================================
# DATA ADAPTER - Converts 4D spectrograms to 3D for your CNN models
# =============================================================================

class SpectrogramAdapter:

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.dataset = dataloader.dataset
        
    def __iter__(self):
        for X, y in self.dataloader:
            # X shape: (batch, C, W, F) where C=channels, W=time windows, F=frequencies
            # Flatten to: (batch, C, W*F) to make it compatible with Conv1d models
            batch, C, W, F = X.shape
            X_flat = X.reshape(batch, C, W * F)  # (batch, C, W*F)
            yield X_flat, y
    
    def __len__(self):
        return len(self.dataloader)


def load_data(cfg: Config):
    """
    Returns:
        train_loader, val_loader, test_loader adapted for CNN models
    """
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    # Configure data loading
    data_cfg = DataConfig(
        raw_dir=UserConfig.RAW_DATA_DIR,
        processed_dir=UserConfig.PROCESSED_DATA_DIR,
        out_dim=cfg.out_dim,
        batch_size=cfg.batch_size,
        num_workers=UserConfig.NUM_WORKERS,
        pin_memory=True if cfg.device == "cuda" else False
    )
    
    split_cfg = SplitConfig(
        train=UserConfig.TRAIN_SPLIT,
        val=UserConfig.VAL_SPLIT,
        test=UserConfig.TEST_SPLIT,
        seed=cfg.seed
    )
    
    preprocess_cfg = PreprocessConfig()
    
    # Get dataloaders (this will build processed splits if they don't exist)
    print("Loading/building processed data splits...")
    train_loader, val_loader, test_loader = make_dataloaders(
        data_cfg=data_cfg,
        split_cfg=split_cfg,
        preprocess_cfg=preprocess_cfg
    )
    
    # Wrap with adapter to convert 4D→3D for your CNN models
    train_loader = SpectrogramAdapter(train_loader)
    val_loader = SpectrogramAdapter(val_loader)
    test_loader = SpectrogramAdapter(test_loader)
    
    print(f"✓ Data loaded: {len(train_loader)} train batches, {len(val_loader)} val batches")
    print("="*70 + "\n")
    
    return train_loader, val_loader, test_loader


# =============================================================================
# MAIN TRAINING FUNCTIONS
# =============================================================================

def train_baseline(cfg: Config):
    """
    Train a student model WITHOUT knowledge distillation.
    
    Args:
        cfg: Configuration from config.py
    """
    # Setup
    set_seed(cfg.seed)
    device = get_device(cfg)
    
    # Load data
    train_loader, val_loader, test_loader = load_data(cfg)
    
    # Build model
    print(f"Building model: {cfg.student_model}")
    model = build_student(cfg.student_model, cfg).to(device)
    
    # Get save paths
    best_path = get_model_path(cfg, mode="baseline", final=False)
    final_path = get_model_path(cfg, mode="baseline", final=True)
    
    # Train
    model, history = train_model(
        model, train_loader, val_loader, cfg, device,
        save_best=True, 
        save_path=best_path
    )
    
    # Save final
    if UserConfig.SAVE_FINAL_MODEL:
        save_model(
            model, final_path, cfg=cfg,
            metadata={'training': 'baseline', 'final': True}
        )
    
    print(f"\n✓ Training complete!")
    print(f"  Best model: {best_path}")
    if UserConfig.SAVE_FINAL_MODEL:
        print(f"  Final model: {final_path}\n")
    
    return model, history


def train_with_kd(cfg: Config):
    """
    Train a student model WITH knowledge distillation.
    
    Args:
        cfg: Configuration from config.py (must have use_distillation=True)
    """
    # Setup
    set_seed(cfg.seed)
    device = get_device(cfg)
    
    # Load data
    train_loader, val_loader, test_loader = load_data(cfg)
    
    # Build student
    print(f"Building student model: {cfg.student_model}")
    student = build_student(cfg.student_model, cfg).to(device)
    
    # Build teacher
    print(f"Building teacher model: {cfg.teacher_model}")
    teacher = build_teacher(cfg.teacher_model, cfg.in_channels, cfg.out_dim).to(device)
    
    # Load or pretrain teacher
    if cfg.teacher_pretrained_path:
        print(f"Loading pretrained teacher from: {cfg.teacher_pretrained_path}")
        from utils import load_model
        teacher, _ = load_model(teacher, cfg.teacher_pretrained_path, device=str(device))
    else:
        print(f"Pretraining teacher for {cfg.teacher_pretrain_epochs} epochs...")
        teacher_path = f"{UserConfig.MODELS_DIR}/{cfg.teacher_model}_teacher.pth"
        teacher = pretrain_teacher(
            teacher, train_loader, val_loader, cfg, device,
            epochs=cfg.teacher_pretrain_epochs,
            save_path=teacher_path
        )
    
    # Build TA model if multi-step
    ta_model = None
    if cfg.distill_mode == "multi_step" and cfg.ta_model:
        print(f"Building TA model: {cfg.ta_model}")
        ta_model = build_student(cfg.ta_model, cfg).to(device)
        
        if cfg.ta_pretrained_path:
            print(f"Loading pretrained TA from: {cfg.ta_pretrained_path}")
            from utils import load_model
            ta_model, _ = load_model(ta_model, cfg.ta_pretrained_path, device=str(device))
        else:
            print(f"Pretraining TA for {cfg.teacher_pretrain_epochs} epochs...")
            ta_path = f"{UserConfig.MODELS_DIR}/{cfg.ta_model}_ta.pth"
            ta_model, _ = train_model(
                ta_model, train_loader, val_loader, cfg, device,
                save_best=True, save_path=ta_path
            )
    
    # Get save paths
    best_path = get_model_path(cfg, mode="distilled", final=False)
    final_path = get_model_path(cfg, mode="distilled", final=True)
    
    # Train student with distillation
    student, history = train_with_distillation(
        student, teacher, train_loader, val_loader, cfg, device,
        ta_model=ta_model, save_best=True, save_path=best_path
    )
    
    # Save final
    if UserConfig.SAVE_FINAL_MODEL:
        save_model(
            student, final_path, cfg=cfg,
            metadata={'training': 'distillation', 'final': True}
        )
    
    print(f"\n✓ Distillation complete!")
    print(f"  Best model: {best_path}")
    if UserConfig.SAVE_FINAL_MODEL:
        print(f"  Final model: {final_path}\n")
    
    return student, history


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main training function. Configure everything in config.py and UserConfig!
    
    To train:
        1. Edit UserConfig section above for file paths and naming
        2. Edit config.py for model/training settings
        3. Run: python main_simple.py
    """
    # Load configuration
    cfg = Config()
    
    # Create directories
    Path(UserConfig.MODELS_DIR).mkdir(exist_ok=True)
    Path(UserConfig.RAW_DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(UserConfig.PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*70)
    print("sEMG HAND MOVEMENT PREDICTION - TRAINING")
    print("="*70)
    print(f"Student Model: {cfg.student_model}")
    print(f"Distillation: {'Enabled' if cfg.use_distillation else 'Disabled'}")
    if cfg.use_distillation:
        print(f"Teacher Model: {cfg.teacher_model}")
        print(f"Distillation Mode: {cfg.distill_mode}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Batch Size: {cfg.batch_size}")
    print(f"Learning Rate: {cfg.lr}")
    print(f"Models will be saved to: {UserConfig.MODELS_DIR}/")
    print("="*70 + "\n")
    
    # Train based on configuration
    if cfg.use_distillation:
        model, history = train_with_kd(cfg)
        mode = "distilled"
    else:
        model, history = train_baseline(cfg)
        mode = "baseline"
    
    # Get final path for evaluation instructions
    final_path = get_model_path(cfg, mode=mode, final=False)
    
    print("\n" + "="*70)
    print("ALL DONE! 🎉")
    print("="*70)
    print(f"Final validation accuracy: {history[-1]['val_accuracy']:.2f}%")
    print(f"\nTo evaluate, run:")
    print(f"  python evaluation.py --model-path {final_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
