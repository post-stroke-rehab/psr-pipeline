import torch
import json
from dataclasses import dataclass, asdict
from typing import Optional, List
from enum import Enum

class DistillationMode(Enum):
    """Types of distillation strategies"""
    NONE = "none"                    # No distillation
    SINGLE_STEP = "single_step"      # Teacher -> Student
    MULTI_STEP = "multi_step"        # Teacher -> TA -> Student
    ENSEMBLE = "ensemble"            # Multiple Teachers -> Student

@dataclass
class Config:
    
    """ MAIN SETTINGS """
    
    student_model: str = "nano"             # Model to train: "nano", "micro", "base", "large", "xlarge"
    epochs: int = 50                        # Number of training epochs
    batch_size: int = 32                    # Batch size for training
    lr: float = 0.001                       # Learning rate
    
    use_distillation: bool = False          # Enable/disable knowledge distillation
    teacher_model: str = "resnet50"         # Teacher: "resnet50", "resnet101", "resnet152"
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"  # "cuda" or "cpu"
    seed: int = 42                          # Random seed for reproducibility
    
    """ OTHER SETTINGS """
    
    # Data dimensions
    in_channels: int = 6            # Number of sEMG muscle channels
    seq_len: int = 200              # Input sequence length (time steps)
    out_dim: int = 5                # Number of output classes (fingers)
    
    # Student Model Architecture (usually don't change)
    conv1_out: int = 24             # First conv layer output channels
    conv1_kernel: int = 3           # First conv kernel size
    conv2_out: int = 24             # Second conv layer output channels
    conv2_kernel: int = 5           # Second conv kernel size
    conv3_out: int = 48             # Third conv layer output channels
    conv3_kernel: int = 9           # Third conv kernel size
    conv4_out: int = 72             # Fourth conv layer output channels
    conv4_kernel: int = 16          # Fourth conv kernel size
    conv5_out: int = 128            # Fifth conv layer output channels
    conv5_kernel: int = 5           # Fifth conv kernel size
    pool_kernel: int = 2            # Max pooling kernel size
    fc1_out: int = 256              # First FC layer output size
    fc2_out: int = 512              # Second FC layer output size
    dropout: float = 0.2            # Dropout rate
    
    # Training hyperparameters
    weight_decay: float = 0.0001    # L2 regularization
    
    # Knowledge Distillation Settings
    distill_mode: str = "single_step"       # "single_step", "multi_step", "ensemble"
    teacher_pretrained_path: Optional[str] = None  # Path to pretrained teacher weights
    teacher_pretrain_epochs: int = 30       # Epochs to pretrain teacher if no weights provided
    ta_model: Optional[str] = None          # TA architecture (e.g., "large" or "xlarge" student model)
    ta_pretrained_path: Optional[str] = None
    distill_temperature: float = 4.0        # Temperature for softening distributions
    distill_alpha: float = 0.5              # Weight: alpha*hard_loss + (1-alpha)*soft_loss
    use_feature_distillation: bool = False  # Match intermediate representations
    feature_loss_weight: float = 0.1        # Weight for feature matching loss
    ta_temperature: float = 3.0             # Temperature for TA->Student distillation
    ta_alpha: float = 0.5                   # Weight for TA distillation
    ensemble_teachers: List[str] = None     # List of teacher models for ensemble
    ensemble_weights: List[float] = None    # Weights for each teacher (auto-normalized)
    use_adaptive_pool: bool = False         # Use adaptive pooling to reduce RAM


def save_config(cfg:Config, filename:str):
    cfg_dict = asdict(cfg)
    with open(filename, "w") as f:
        json.dump(cfg_dict, f, indent=4)
    print(f"Configuration saved to {filename}")
