import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm, trange
from config import Config


def distillation_loss(student_output: torch.Tensor, 
                     teacher_output: torch.Tensor,
                     true_labels: torch.Tensor, 
                     T: float = 4.0, 
                     alpha: float = 0.5) -> torch.Tensor:
    """
    Knowledge distillation loss for multi-label classification.
    
    Args:
        student_output: Student predictions (after sigmoid) - (batch, num_classes)
        teacher_output: Teacher predictions (logits, before sigmoid) - (batch, num_classes)
        true_labels: Ground truth labels - (batch, num_classes)
        T: Temperature for softening distributions
        alpha: Weight between hard loss and soft loss
    
    Returns:
        Combined distillation loss
    """
    # Hard loss: student vs ground truth
    hard_loss = F.binary_cross_entropy(student_output, true_labels)
    
    # Soft loss: student vs teacher
    # Apply sigmoid to teacher logits with temperature
    teacher_soft = torch.sigmoid(teacher_output / T)
    
    # MSE is common for multi-label distillation
    soft_loss = F.mse_loss(student_output, teacher_soft) * (T * T)
    
    # Combined loss
    return alpha * hard_loss + (1 - alpha) * soft_loss


def feature_distillation_loss(student_features: List[torch.Tensor],
                              teacher_features: List[torch.Tensor]) -> torch.Tensor:
    """
    Feature-based distillation loss matching intermediate representations.
    
    Args:
        student_features: List of student intermediate feature maps
        teacher_features: List of teacher intermediate feature maps
    
    Returns:
        Feature matching loss
    """
    loss = 0.0
    num_matches = min(len(student_features), len(teacher_features))
    
    if num_matches == 0:
        return torch.tensor(0.0)
    
    # Match features at corresponding depths
    step_teacher = len(teacher_features) // num_matches if num_matches > 0 else 1
    step_student = len(student_features) // num_matches if num_matches > 0 else 1
    
    for i in range(num_matches):
        s_feat = student_features[min(i * step_student, len(student_features) - 1)]
        t_feat = teacher_features[min(i * step_teacher, len(teacher_features) - 1)]
        
        # Adapt spatial dimensions if needed
        if len(s_feat.shape) == 3 and len(t_feat.shape) == 3:  # Conv features
            if s_feat.shape[2] != t_feat.shape[2]:
                t_feat = F.adaptive_avg_pool1d(t_feat, s_feat.shape[2])
        
        # Adapt channel dimensions if needed
        if s_feat.shape[1] != t_feat.shape[1]:
            # Use 1x1 conv to match channels
            adapter = nn.Conv1d(t_feat.shape[1], s_feat.shape[1], 1).to(s_feat.device)
            t_feat = adapter(t_feat)
        
        loss += F.mse_loss(s_feat, t_feat.detach())
    
    return loss / num_matches


def train_one_epoch_distill(student: nn.Module,
                           teacher: nn.Module,
                           train_loader,
                           optimizer,
                           device: torch.device,
                           cfg: Config,
                           ta_model: Optional[nn.Module] = None) -> Dict[str, float]:
    """
    Train for one epoch with knowledge distillation.
    
    Args:
        student: Student model
        teacher: Teacher model
        train_loader: Training data loader
        optimizer: Optimizer for student
        device: Device to train on
        cfg: Configuration with distillation parameters
        ta_model: Optional TA model for multi-step distillation
    
    Returns:
        Dictionary with training metrics
    """
    student.train()
    teacher.eval()
    if ta_model is not None:
        ta_model.eval()
    
    total_loss = 0.0
    total_hard_loss = 0.0
    total_soft_loss = 0.0
    total_feature_loss = 0.0
    n_batches = 0
    
    for sequences, labels in train_loader:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Get student output
        return_features = cfg.use_feature_distillation
        student_output = student(sequences, return_features=return_features)
        
        with torch.no_grad():
            # Multi-step distillation: Student learns from TA
            if cfg.distill_mode == "multi_step" and ta_model is not None:
                ta_output = ta_model(sequences, return_features=return_features)
                distill_output = ta_output
                T = cfg.ta_temperature
                alpha = cfg.ta_alpha
                teacher_features = ta_model.intermediate_features if return_features else []
            
            # Single-step distillation: Student learns from Teacher
            else:
                teacher_output = teacher(sequences, return_features=return_features)
                distill_output = teacher_output
                T = cfg.distill_temperature
                alpha = cfg.distill_alpha
                teacher_features = teacher.intermediate_features if return_features else []
        
        # Compute distillation loss
        loss = distillation_loss(student_output, distill_output, labels, T=T, alpha=alpha)
        
        # Track components
        hard_loss = F.binary_cross_entropy(student_output, labels)
        soft_loss = F.mse_loss(student_output, torch.sigmoid(distill_output / T)) * (T * T)
        total_hard_loss += hard_loss.item()
        total_soft_loss += soft_loss.item()
        
        # Add feature distillation if enabled
        if cfg.use_feature_distillation and len(student.intermediate_features) > 0:
            feature_loss = feature_distillation_loss(
                student.intermediate_features,
                teacher_features
            )
            loss += cfg.feature_loss_weight * feature_loss
            total_feature_loss += feature_loss.item()
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    metrics = {
        "loss": total_loss / n_batches,
        "hard_loss": total_hard_loss / n_batches,
        "soft_loss": total_soft_loss / n_batches,
    }
    
    if cfg.use_feature_distillation:
        metrics["feature_loss"] = total_feature_loss / n_batches
    
    return metrics


@torch.no_grad()
def evaluate(model: nn.Module, 
            data_loader, 
            device: torch.device) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: Data loader
        device: Device to evaluate on
    
    Returns:
        Dictionary with metrics: loss, accuracy
    """
    model.eval()
    criterion = nn.BCELoss()
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
        predicted = (predictions > 0.5).float()
        correct += (predicted == labels).all(dim=1).sum().item()
        total += labels.size(0)
    
    return {
        "loss": total_loss / len(data_loader),
        "accuracy": 100.0 * correct / total
    }


def train_with_distillation(student: nn.Module,
                          teacher: nn.Module,
                          train_loader,
                          val_loader,
                          cfg: Config,
                          device: torch.device,
                          ta_model: Optional[nn.Module] = None,
                          save_best: bool = True,
                          save_path: str = None) -> Tuple[nn.Module, List[Dict]]:
    """
    Complete training loop with knowledge distillation.
    
    Args:
        student: Student model to train
        teacher: Teacher model (pretrained)
        train_loader: Training data loader
        val_loader: Validation data loader
        cfg: Configuration with distillation parameters
        device: Device to train on
        ta_model: Optional TA model for multi-step distillation
        save_best: Whether to save best model
        save_path: Path to save best model
    
    Returns:
        student: Trained student model
        history: List of dicts containing metrics for each epoch
    """
    # Setup optimizer
    optimizer = torch.optim.Adam(
        student.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    
    # Training history
    history = []
    best_val_acc = 0.0
    
    print(f"\n{'='*70}")
    print(f"KNOWLEDGE DISTILLATION TRAINING")
    print(f"{'='*70}")
    print(f"Mode: {cfg.distill_mode}")
    print(f"Temperature: {cfg.distill_temperature}")
    print(f"Alpha: {cfg.distill_alpha}")
    if cfg.use_feature_distillation:
        print(f"Feature Distillation: Enabled (weight={cfg.feature_loss_weight})")
    if cfg.distill_mode == "multi_step" and ta_model is not None:
        print(f"TA Temperature: {cfg.ta_temperature}")
        print(f"TA Alpha: {cfg.ta_alpha}")
    print(f"Epochs: {cfg.epochs}")
    print(f"{'='*70}\n")
    
    # Training loop
    for epoch in trange(cfg.epochs, desc="Training"):
        # Train with distillation
        train_metrics = train_one_epoch_distill(
            student, teacher, train_loader, optimizer, device, cfg, ta_model
        )
        
        # Validation
        val_metrics = evaluate(student, val_loader, device)
        
        # Save history
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_hard_loss': train_metrics['hard_loss'],
            'train_soft_loss': train_metrics['soft_loss'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy']
        }
        
        if cfg.use_feature_distillation:
            epoch_data['train_feature_loss'] = train_metrics['feature_loss']
        
        history.append(epoch_data)
        
        # Save best model
        if save_best and val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            if save_path:
                from utils import save_model
                save_model(
                    student,
                    save_path,
                    cfg=cfg,
                    metadata={
                        'epoch': epoch + 1,
                        'val_accuracy': best_val_acc,
                        'val_loss': val_metrics['loss'],
                        'distillation_mode': cfg.distill_mode
                    }
                )
        
        # Print progress
        if (epoch + 1) % 1 == 0:
            print(f"\n[Epoch {epoch+1}/{cfg.epochs}]")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"    Hard Loss: {train_metrics['hard_loss']:.4f}")
            print(f"    Soft Loss: {train_metrics['soft_loss']:.4f}")
            if cfg.use_feature_distillation:
                print(f"    Feature Loss: {train_metrics['feature_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Acc:  {val_metrics['accuracy']:.2f}%")
            if save_best:
                print(f"  Best Val Acc: {best_val_acc:.2f}%")
    
    print(f"\n{'='*70}")
    print(f"DISTILLATION TRAINING COMPLETE")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"{'='*70}\n")
    
    return student, history


def pretrain_teacher(teacher: nn.Module,
                    train_loader,
                    val_loader,
                    cfg: Config,
                    device: torch.device,
                    epochs: int = None,
                    save_path: str = None) -> nn.Module:
    """
    Pretrain a teacher model before distillation.
    
    Args:
        teacher: Teacher model to pretrain
        train_loader: Training data loader
        val_loader: Validation data loader
        cfg: Configuration object
        device: Device to train on
        epochs: Number of epochs (overrides cfg.epochs)
        save_path: Path to save trained teacher
    
    Returns:
        Trained teacher model
    """
    from training import train_model
    
    print(f"\n{'='*70}")
    print(f"PRETRAINING TEACHER MODEL")
    print(f"{'='*70}\n")
    
    # Use specified epochs or default from config
    if epochs is not None:
        original_epochs = cfg.epochs
        cfg.epochs = epochs
    
    # Train teacher
    teacher, history = train_model(
        teacher, train_loader, val_loader, cfg, device,
        save_best=True, save_path=save_path
    )
    
    # Restore original epochs
    if epochs is not None:
        cfg.epochs = original_epochs
    
    return teacher
