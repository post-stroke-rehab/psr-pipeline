import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
from tqdm import tqdm, trange
from config import Config


def train_one_epoch(model: nn.Module, 
                   train_loader, 
                   optimizer, 
                   criterion,
                   device: torch.device) -> Dict[str, float]:
    """
    Train for one epoch (standard training, no distillation).
    
    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    total_loss = 0.0
    n_batches = 0
    
    for sequences, labels in train_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(sequences)
        loss = criterion(predictions, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    avg_loss = total_loss / n_batches
    
    return {
        "loss": avg_loss,
    }


@torch.no_grad()
def evaluate(model: nn.Module, 
            data_loader, 
            criterion, 
            device: torch.device) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        criterion: Loss function
        device: Device to evaluate on
    
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
        
        # Calculate accuracy for multi-label classification
        # Threshold predictions at 0.5
        predicted = (predictions > 0.5).float()
        # Count samples where all labels match
        correct += (predicted == labels).all(dim=1).sum().item()
        total += labels.size(0)
    
    return {
        "loss": total_loss / len(data_loader),
        "accuracy": 100.0 * correct / total
    }


def train_model(model: nn.Module,
               train_loader,
               val_loader,
               cfg: Config,
               device: torch.device,
               save_best: bool = True,
               save_path: str = None) -> Tuple[nn.Module, List[Dict]]:
    """
    Complete training loop for standard training (no distillation).
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        cfg: Configuration object
        device: Device to train on
        save_best: Whether to save best model
        save_path: Path to save best model
    
    Returns:
        model: Trained model
        history: List of dicts containing metrics for each epoch
    """
    # Setup optimizer and criterion
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    
    # Binary Cross Entropy for multi-label classification
    criterion = nn.BCELoss()
    
    # Training history
    history = []
    best_val_acc = 0.0
    
    print(f"\n{'='*70}")
    print(f"STARTING TRAINING")
    print(f"{'='*70}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Learning Rate: {cfg.lr}")
    print(f"Batch Size: {cfg.batch_size}")
    print(f"Dropout: {cfg.dropout}")
    print(f"{'='*70}\n")
    
    # Training loop
    for epoch in trange(cfg.epochs, desc="Training"):
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Save history
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy']
        }
        history.append(epoch_data)
        
        # Save best model
        if save_best and val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            if save_path:
                from utils import save_model
                save_model(
                    model, 
                    save_path,
                    cfg=cfg,
                    metadata={
                        'epoch': epoch + 1,
                        'val_accuracy': best_val_acc,
                        'val_loss': val_metrics['loss']
                    }
                )
        
        # Print progress
        if (epoch + 1) % 1 == 0:
            print(f"\n[Epoch {epoch+1}/{cfg.epochs}]")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  Val Acc:    {val_metrics['accuracy']:.2f}%")
            if save_best:
                print(f"  Best Val Acc: {best_val_acc:.2f}%")
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*70}\n")
    
    return model, history
