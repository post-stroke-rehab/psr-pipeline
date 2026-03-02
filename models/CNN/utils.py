import torch
import torch.nn as nn
import numpy as np
import random
from config import Config
from students import CNN_Nano, CNN_Micro, CNN_Base, CNN_Large, CNN_XLarge
from teachers import build_teacher


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_student(model_name: str, cfg: Config) -> nn.Module:
    """
    Build a student model.
    
    Args:
        model_name: Name of the student architecture
        cfg: Configuration object
    
    Returns:
        Student model instance
    """
    model_configs = {
        "nano": (CNN_Nano, 0.67),
        "micro": (CNN_Micro, 0.83),
        "base": (CNN_Base, 1.0),
        "large": (CNN_Large, 1.33),
        "xlarge": (CNN_XLarge, 2.0),
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Unknown student model: {model_name}. Choose from {list(model_configs.keys())}")
    
    model_class, scale_factor = model_configs[model_name]
    model = model_class(cfg, scale_factor=scale_factor)
    
    # Print model size
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Student] {model_name.upper():<10} - {n_params:,} trainable parameters")
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model: nn.Module, filepath: str, cfg: Config = None, metadata: dict = None):
    """
    Save model weights and optionally config/metadata.
    
    Args:
        model: Model to save
        filepath: Path to save file (e.g., 'checkpoints/model.pth')
        cfg: Optional configuration to save alongside model
        metadata: Optional metadata dict (epoch, accuracy, etc.)
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
    }
    
    if cfg is not None:
        from dataclasses import asdict
        save_dict['config'] = asdict(cfg)
    
    if metadata is not None:
        save_dict['metadata'] = metadata
    
    torch.save(save_dict, filepath)
    print(f"[Save] Model saved to {filepath}")


def load_model(model: nn.Module, filepath: str, device: str = 'cpu'):
    """
    Load model weights.
    
    Args:
        model: Model instance to load weights into
        filepath: Path to saved model file
        device: Device to load model onto
    
    Returns:
        model: Model with loaded weights
        metadata: Saved metadata (if any)
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    # Handle different save formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        metadata = checkpoint.get('metadata', {})
    else:
        # Assume direct state_dict
        model.load_state_dict(checkpoint)
        metadata = {}
    
    print(f"[Load] Model loaded from {filepath}")
    
    return model, metadata


def get_device(cfg: Config) -> torch.device:
    """Get the appropriate device (CPU/CUDA)."""
    device = torch.device(cfg.device)
    if device.type == 'cuda':
        print(f"[Device] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print(f"[Device] Using CPU")
    return device
