'''
AYO, HEAR ME OUT TWIN

This code provides 5 different architectures to try, of increasing computational weight
The plan is to try which is the limit of computation of the Raspberry Pi, processor only
After choosing one or two models, using the config dataclass you can tune parameters easily to find the optimal model

I still have to add some stuff, like an actual data loader, but we're almost there dw
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple
from tqdm import tqdm, trange
import json


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """
    To create a variant:
        cfg = Config(dropout=0.3, lr=0.0001)
    
    To save configuration:
        with open('config.json', 'w') as f:
            json.dump(asdict(cfg), f, indent=2)
    """
    # Data dimensions
    in_channels: int = 6            # Number of sEMG muscle channels
    seq_len: int = 200              # Input sequence length (time steps)
    out_dim: int = 5                # Number of output classes (fingers)
    
    # Block 1
    conv1_out: int = 24             # First conv layer output channels
    conv1_kernel: int = 3           # First conv kernel size
    conv2_out: int = 24             # Second conv layer output channels
    conv2_kernel: int = 5           # Second conv kernel size
    
    # Block 2
    conv3_out: int = 48             # Third conv layer output channels
    conv3_kernel: int = 9           # Third conv kernel size
    conv4_out: int = 72             # Fourth conv layer output channels
    conv4_kernel: int = 16          # Fourth conv kernel size
    
    # Block 3 (only used by heavy model)
    conv5_out: int = 128            # Fifth conv layer output channels
    conv5_kernel: int = 5           # Fifth conv kernel size
    
    # Pooling
    pool_kernel: int = 2            # Max pooling kernel size
    
    # Fully connected layers
    fc1_out: int = 256              # First FC layer output size
    fc2_out: int = 512              # Second FC layer (only heavy model uses this)
    dropout: float = 0.2            # Dropout rate
    
    # Training hyperparameters
    lr: float = 0.001               # Learning rate
    weight_decay: float = 0.0001    # L2 regularization
    epochs: int = 5                 # Number of training epochs
    batch_size: int = 32            # Batch size
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Reproducibility
    seed: int = 67
    


    ## Don't use right now, distillation training is under development ##
    use_distillation: bool = False  # Enable knowledge distillation
    distill_temperature: float = 4.0
    distill_alpha: float = 0.5
    
    # For RAM reduction (optional)
    use_adaptive_pool: bool = False  # Use adaptive pooling to reduce RAM


def scale_channels(base_value: int, scale_factor: float) -> int: 
    """
    Scale number of channels by a factor, ensuring it's at least 1. Used for the different architectures' sizes
    Returns:
        Scaled value, minimum 1
    """
    return max(1, int(base_value * scale_factor))


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class CNN_Nano(nn.Module): #Parameters: ~100K 
    
    def __init__(self, cfg: Config, scale_factor: float = 0.67):
        super().__init__()
        self.cfg = cfg
        self.scale = scale_factor
        
        # Very small channels for speed
        conv1_out = scale_channels(cfg.conv1_out, scale_factor)  # 24 → 16
        fc1_out = scale_channels(cfg.fc1_out, 0.25)              # 256 → 64
        
        # Block 1
        self.conv1 = nn.Conv1d(cfg.in_channels, conv1_out, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(conv1_out)
        
        # MLP
        # After 1 pool: 200→198→99
        self.flattened_size = 99 * conv1_out
        self.fc1 = nn.Linear(self.flattened_size, fc1_out)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc2 = nn.Linear(fc1_out, cfg.out_dim)
    
    def forward(self, x):
        # Single conv block
        x = self.conv1(x)              # (B, 16, 198)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=self.cfg.pool_kernel)  # (B, 16, 99)
        
        # Minimal MLP
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


class CNN_Micro(nn.Module): #Parameters: ~250k
    def __init__(self, cfg: Config, scale_factor: float = 0.83):
        super().__init__()
        self.cfg = cfg
        self.scale = scale_factor
        
        # Scale channels moderately
        conv1_out = scale_channels(cfg.conv1_out, scale_factor)  # 24 → 20
        conv2_out = scale_channels(cfg.conv2_out, scale_factor)  # 24 → 20
        fc1_out = scale_channels(cfg.fc1_out, 0.5)               # 256 → 128
        
        # Block 1
        self.conv1 = nn.Conv1d(cfg.in_channels, conv1_out, kernel_size=cfg.conv1_kernel)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=cfg.conv2_kernel)
        self.bn1 = nn.BatchNorm1d(conv2_out)
        
        # MLP
        # After 1 pool: 200→198→194→97
        self.flattened_size = 97 * conv2_out
        self.fc1 = nn.Linear(self.flattened_size, fc1_out)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc2 = nn.Linear(fc1_out, cfg.out_dim)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1(x))      # (B, 20, 198)
        x = self.conv2(x)              # (B, 20, 194)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=self.cfg.pool_kernel)  # (B, 20, 97)
        
        # MLP
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


class CNN_Base(nn.Module): #Parameters: ~700k

    def __init__(self, cfg: Config, scale_factor: float = 1.0):
        super().__init__()
        self.cfg = cfg
        self.scale = scale_factor
        
        # Use base config values directly
        conv1_out = scale_channels(cfg.conv1_out, scale_factor)
        conv2_out = scale_channels(cfg.conv2_out, scale_factor)
        conv3_out = scale_channels(cfg.conv3_out, scale_factor)
        conv4_out = scale_channels(cfg.conv4_out, scale_factor)
        fc1_out = scale_channels(cfg.fc1_out, scale_factor)
        
        # Block 1
        self.conv1 = nn.Conv1d(cfg.in_channels, conv1_out, kernel_size=cfg.conv1_kernel)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=cfg.conv2_kernel)
        self.bn1 = nn.BatchNorm1d(conv2_out)
        
        # Block 2
        self.conv3 = nn.Conv1d(conv2_out, conv3_out, kernel_size=cfg.conv3_kernel)
        self.bn2 = nn.BatchNorm1d(conv3_out)
        self.conv4 = nn.Conv1d(conv3_out, conv4_out, kernel_size=cfg.conv4_kernel)
        self.bn3 = nn.BatchNorm1d(conv4_out)
        
        # MLP
        # After 2 pools: 200→198→194→97→89→74→37
        self.flattened_size = 37 * conv4_out
        self.fc1 = nn.Linear(self.flattened_size, fc1_out)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc2 = nn.Linear(fc1_out, cfg.out_dim)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1(x))      # (B, 24, 198)
        x = self.conv2(x)              # (B, 24, 194)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=self.cfg.pool_kernel)  # (B, 24, 97)
        
        # Block 2
        x = self.conv3(x)              # (B, 48, 89)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv4(x)              # (B, 72, 74)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=self.cfg.pool_kernel)  # (B, 72, 37)
        
        # MLP
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


class CNN_Large(nn.Module): #Parameters: ~1.5M

    def __init__(self, cfg: Config, scale_factor: float = 1.33):
        super().__init__()
        self.cfg = cfg
        self.scale = scale_factor
        
        # Increase channel counts
        conv1_out = scale_channels(cfg.conv1_out, scale_factor)  # 24 → 32
        conv2_out = scale_channels(cfg.conv2_out, scale_factor)  # 24 → 32
        conv3_out = scale_channels(cfg.conv3_out, scale_factor)  # 48 → 64
        conv4_out = scale_channels(cfg.conv4_out, scale_factor)  # 72 → 96
        conv5_out = scale_channels(cfg.conv5_out, scale_factor)  # 128 → 170
        fc1_out = scale_channels(cfg.fc2_out, scale_factor)      # 512 → 682
        
        # Block 1
        self.conv1 = nn.Conv1d(cfg.in_channels, conv1_out, kernel_size=cfg.conv1_kernel)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=cfg.conv2_kernel)
        self.bn1 = nn.BatchNorm1d(conv2_out)
        
        # Block 2
        self.conv3 = nn.Conv1d(conv2_out, conv3_out, kernel_size=cfg.conv3_kernel)
        self.bn2 = nn.BatchNorm1d(conv3_out)
        self.conv4 = nn.Conv1d(conv3_out, conv4_out, kernel_size=cfg.conv4_kernel)
        self.bn3 = nn.BatchNorm1d(conv4_out)
        
        # Block 3
        self.conv5 = nn.Conv1d(conv4_out, conv5_out, kernel_size=cfg.conv5_kernel)
        self.bn4 = nn.BatchNorm1d(conv5_out)
        
        # MLP
        # After 3 pools: 200→198→194→97→89→74→37→33→16
        self.flattened_size = 16 * conv5_out
        self.fc1 = nn.Linear(self.flattened_size, fc1_out)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc2 = nn.Linear(fc1_out, cfg.out_dim)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=self.cfg.pool_kernel)
        
        # Block 2
        x = self.conv3(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=self.cfg.pool_kernel)
        
        # Block 3
        x = self.conv5(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=self.cfg.pool_kernel)
        
        # MLP
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x


class CNN_XLarge(nn.Module): #Parameters: ~4M

    def __init__(self, cfg: Config, scale_factor: float = 2.0):
        super().__init__()
        self.cfg = cfg
        self.scale = scale_factor
        
        # Maximum channel counts
        conv1_out = scale_channels(cfg.conv1_out, scale_factor)  # 24 → 48
        conv2_out = scale_channels(cfg.conv2_out, scale_factor)  # 24 → 48
        conv3_out = scale_channels(cfg.conv3_out, scale_factor)  # 48 → 96
        conv4_out = scale_channels(cfg.conv4_out, scale_factor)  # 72 → 144
        conv5_out = scale_channels(cfg.conv5_out, scale_factor)  # 128 → 256
        
        # Additional layers for XLarge
        conv6_out = scale_channels(192, scale_factor)  # 384
        conv7_out = scale_channels(160, scale_factor)  # 320
        
        fc1_out = scale_channels(cfg.fc2_out, scale_factor * 2)  # 512 → 2048
        fc2_out = scale_channels(cfg.fc1_out, scale_factor)      # 256 → 512
        
        # Block 1
        self.conv1 = nn.Conv1d(cfg.in_channels, conv1_out, kernel_size=cfg.conv1_kernel)
        self.conv2 = nn.Conv1d(conv1_out, conv2_out, kernel_size=cfg.conv2_kernel)
        self.bn1 = nn.BatchNorm1d(conv2_out)
        
        # Block 2
        self.conv3 = nn.Conv1d(conv2_out, conv3_out, kernel_size=cfg.conv3_kernel)
        self.bn2 = nn.BatchNorm1d(conv3_out)
        self.conv4 = nn.Conv1d(conv3_out, conv4_out, kernel_size=7)  # Smaller kernel for depth
        self.bn3 = nn.BatchNorm1d(conv4_out)
        
        # Block 3
        self.conv5 = nn.Conv1d(conv4_out, conv5_out, kernel_size=cfg.conv5_kernel)
        self.bn4 = nn.BatchNorm1d(conv5_out)
        self.conv6 = nn.Conv1d(conv5_out, conv6_out, kernel_size=3)
        self.bn5 = nn.BatchNorm1d(conv6_out)
        
        # Block 4
        self.conv7 = nn.Conv1d(conv6_out, conv7_out, kernel_size=3)
        self.bn6 = nn.BatchNorm1d(conv7_out)
        
        # MLP
        # After 4 pools: 200→198→194→97→89→82→41→37→32→16→14→7
        self.flattened_size = 4 * conv7_out
        self.fc1 = nn.Linear(self.flattened_size, fc1_out)
        self.dropout1 = nn.Dropout(cfg.dropout)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.dropout2 = nn.Dropout(cfg.dropout)
        self.fc3 = nn.Linear(fc2_out, cfg.out_dim)
    
    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=self.cfg.pool_kernel)
        
        # Block 2
        x = self.conv3(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=self.cfg.pool_kernel)
        
        # Block 3
        x = self.conv5(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=self.cfg.pool_kernel)
        
        # Block 4
        x = self.conv7(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=self.cfg.pool_kernel)
        
        # Deep MLP
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x


# =============================================================================
# MODEL BUILDER, easy switch
# =============================================================================

def build_model(model_name: str, cfg: Config) -> nn.Module:

    model_configs = {
        "nano": (CNN_Nano, 0.67),        # 1 block - ultra-fast edge devices
        "micro": (CNN_Micro, 0.83),      # 1 block (2 conv) - mobile/embedded
        "base": (CNN_Base, 1.0),         # 2 blocks - balanced (original)
        "large": (CNN_Large, 1.33),      # 3 blocks - high accuracy
        "xlarge": (CNN_XLarge, 2.0),     # 4 blocks - maximum capacity
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_configs.keys())}")
    
    model_class, scale_factor = model_configs[model_name]
    model = model_class(cfg, scale_factor=scale_factor)
    
    # Print model size
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {model_name.upper():<10} - {n_params:,} trainable parameters (scale: {scale_factor}x)")
    
    return model


# =============================================================================
# LOSS FUNCTIONS, that's it yeah
# =============================================================================

def BCE(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: # binary_cross_entropy_loss
    return F.binary_cross_entropy(predictions, targets)

## Distilled learning is being developed, just ignore this function
def distillation_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                            true_labels: torch.Tensor, T: float = 4.0, alpha: float = 0.5) -> torch.Tensor:
    """
    Simplified distillation loss for multi-label classification.
    Uses MSE for soft targets (common in multi-label distillation).
    
    Args:
        student_logits: Student output (after sigmoid) - (batch, 5)
        teacher_logits: Teacher output (after sigmoid) - (batch, 5)
        true_labels: Ground truth (batch, 5)
        T: Temperature (controls smoothness)
        alpha: Weight between hard/soft loss
    """
    # Hard loss: student vs ground truth
    hard_loss = F.binary_cross_entropy(student_logits, true_labels)
    
    # Soft loss: student vs teacher (MSE is common for multi-label)
    soft_loss = F.mse_loss(student_logits, teacher_logits) * (T * T)
    
    return alpha * hard_loss + (1 - alpha) * soft_loss


# =============================================================================
# TRAINING LOOP - Clean and organized
# =============================================================================

def train_one_epoch(model: nn.Module, train_loader, optimizer, criterion,
                   device: torch.device, teacher: Optional[nn.Module] = None,
                   cfg: Optional[Config] = None) -> float:
    
    model.train()
    if teacher is not None:
        teacher.eval()
    
    total_loss = 0.0
    n_batches = 0
    
    for sequences, labels in train_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        if teacher is not None and cfg.use_distillation:
            # Knowledge distillation mode
            student_logits = model(sequences)
            with torch.no_grad():
                teacher_logits = teacher(sequences)
            
            loss = distillation_loss(
                student_logits, teacher_logits, labels,
                T=cfg.distill_temperature,
                alpha=cfg.distill_alpha
            )
        else:
            # Standard training
            predictions = model(sequences)
            loss = criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


@torch.no_grad()
def evaluate(model: nn.Module, data_loader, criterion, device: torch.device) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Returns:
        Dictionary with metrics: loss, accuracy
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for sequences, labels in data_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        predictions = model(sequences)
        loss = criterion(predictions, labels)
        
        total_loss += loss.item()
        
        # Calculate accuracy (assuming multi-class)
        _, predicted = predictions.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return {
        "loss": total_loss / len(data_loader),
        "accuracy": 100.0 * correct / total
    }


# =============================================================================
# MAIN TRAINING FUNCTION - CURRENTLY MISSING SOME CODE
# =============================================================================

def main(cfg: Config, model_name: str ):

    # Set seed for reproducibility
    set_seed(cfg.seed)
    
    # Setup device
    device = torch.device(cfg.device)
    print(f"[Device] Using: {device}")
    
    # Build model (uses same config with different scaling)
    model = build_model(model_name, cfg).to(device)
    
    # Setup optimizer and criterion
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    # ~~~~~~~~~ DON'T FORGET YOU'RE MISSING THE DATALOADER AHHH ~~~~~~~~~~~~~~~~~~~ (That was kind of a note to myself for later)
    # train_loader = ...
    # val_loader = ...
    
    print(f"\n[Training] Starting training for {cfg.epochs} epochs...")
    print(f"[Config] lr={cfg.lr}, dropout={cfg.dropout}, batch_size={cfg.batch_size}")
    
    # Training loop
    for epoch in trange(cfg.epochs, desc="Epochs"):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Print progress every epoch
        if epoch % 1 == 0:
            print(f"\n[Epoch {epoch+1}/{cfg.epochs}] Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.2f}%")
    
    print("\n[Training] Complete!")
    return model
