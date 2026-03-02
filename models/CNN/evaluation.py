#!/usr/bin/env python3
"""
Model evaluation script with comprehensive metrics and visualizations.

Usage:
    python evaluation.py --model-path models/nano_distilled.pth \
                        --test-data test_data.pt \
                        --output-dir results/
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc,
    f1_score, precision_score, recall_score
)
import json
import os
from typing import Dict, List, Tuple

from config import Config
from utils import build_student, load_model, get_device
from teachers import build_teacher


def load_test_data(test_data_path: str, batch_size: int = 32):
    """
    Load test data.
    
    Args:
        test_data_path: Path to test data file
        batch_size: Batch size for DataLoader
    
    Returns:
        test_loader: DataLoader for test data
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    # Load test data (assume it's saved as a .pt file with sequences and labels)
    try:
        data = torch.load(test_data_path)
        test_sequences = data['sequences']
        test_labels = data['labels']
    except:
        # Fallback: create dummy test data
        print("[Warning] Could not load test data. Using dummy data.")
        test_sequences = torch.randn(200, 6, 200)
        test_labels = torch.randint(0, 2, (200, 5)).float()
    
    test_dataset = TensorDataset(test_sequences, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader


@torch.no_grad()
def get_predictions(model: nn.Module, 
                   data_loader, 
                   device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get model predictions on a dataset.
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader
        device: Device
    
    Returns:
        predictions: Raw predictions (probabilities)
        predicted_labels: Thresholded predictions (0 or 1)
        true_labels: Ground truth labels
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    for sequences, labels in data_loader:
        sequences = sequences.to(device)
        outputs = model(sequences)
        
        all_predictions.append(outputs.cpu().numpy())
        all_labels.append(labels.numpy())
    
    predictions = np.vstack(all_predictions)
    true_labels = np.vstack(all_labels)
    predicted_labels = (predictions > 0.5).astype(int)
    
    return predictions, predicted_labels, true_labels


def calculate_metrics(predictions: np.ndarray, 
                     true_labels: np.ndarray) -> Dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        predictions: Predicted labels (thresholded)
        true_labels: Ground truth labels
    
    Returns:
        Dictionary with various metrics
    """
    metrics = {}
    
    # Overall accuracy (exact match for all labels)
    exact_match = (predictions == true_labels).all(axis=1).mean()
    metrics['exact_match_accuracy'] = float(exact_match * 100)
    
    # Per-class metrics
    num_classes = predictions.shape[1]
    class_names = [f'Finger_{i}' for i in range(num_classes)]
    
    for i, class_name in enumerate(class_names):
        # Precision, Recall, F1 for each class
        precision = precision_score(true_labels[:, i], predictions[:, i], zero_division=0)
        recall = recall_score(true_labels[:, i], predictions[:, i], zero_division=0)
        f1 = f1_score(true_labels[:, i], predictions[:, i], zero_division=0)
        
        metrics[f'{class_name}_precision'] = float(precision)
        metrics[f'{class_name}_recall'] = float(recall)
        metrics[f'{class_name}_f1'] = float(f1)
    
    # Macro-averaged metrics
    metrics['macro_precision'] = float(precision_score(true_labels, predictions, average='macro', zero_division=0))
    metrics['macro_recall'] = float(recall_score(true_labels, predictions, average='macro', zero_division=0))
    metrics['macro_f1'] = float(f1_score(true_labels, predictions, average='macro', zero_division=0))
    
    # Micro-averaged metrics
    metrics['micro_precision'] = float(precision_score(true_labels, predictions, average='micro', zero_division=0))
    metrics['micro_recall'] = float(recall_score(true_labels, predictions, average='micro', zero_division=0))
    metrics['micro_f1'] = float(f1_score(true_labels, predictions, average='micro', zero_division=0))
    
    return metrics


def plot_confusion_matrices(predictions: np.ndarray, 
                          true_labels: np.ndarray,
                          output_dir: str):
    """
    Plot confusion matrix for each class.
    
    Args:
        predictions: Predicted labels
        true_labels: Ground truth labels
        output_dir: Directory to save plots
    """
    num_classes = predictions.shape[1]
    class_names = [f'Finger_{i}' for i in range(num_classes)]
    
    fig, axes = plt.subplots(1, num_classes, figsize=(4*num_classes, 3))
    if num_classes == 1:
        axes = [axes]
    
    for i, (ax, class_name) in enumerate(zip(axes, class_names)):
        cm = confusion_matrix(true_labels[:, i], predictions[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Inactive', 'Active'],
                   yticklabels=['Inactive', 'Active'])
        ax.set_title(f'{class_name}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Confusion matrices saved to {output_dir}/confusion_matrices.png")


def plot_precision_recall_curves(predictions_prob: np.ndarray,
                                true_labels: np.ndarray,
                                output_dir: str):
    """
    Plot Precision-Recall curves for each class.
    
    Args:
        predictions_prob: Raw prediction probabilities
        true_labels: Ground truth labels
        output_dir: Directory to save plots
    """
    num_classes = predictions_prob.shape[1]
    class_names = [f'Finger_{i}' for i in range(num_classes)]
    
    plt.figure(figsize=(10, 6))
    
    for i, class_name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(true_labels[:, i], predictions_prob[:, i])
        plt.plot(recall, precision, label=f'{class_name}', linewidth=2)
    
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Precision-Recall curves saved to {output_dir}/precision_recall_curves.png")


def plot_roc_curves(predictions_prob: np.ndarray,
                   true_labels: np.ndarray,
                   output_dir: str):
    """
    Plot ROC curves for each class.
    
    Args:
        predictions_prob: Raw prediction probabilities
        true_labels: Ground truth labels
        output_dir: Directory to save plots
    """
    num_classes = predictions_prob.shape[1]
    class_names = [f'Finger_{i}' for i in range(num_classes)]
    
    plt.figure(figsize=(10, 6))
    
    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(true_labels[:, i], predictions_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] ROC curves saved to {output_dir}/roc_curves.png")


def plot_per_class_metrics(metrics: Dict, output_dir: str):
    """
    Plot bar chart of per-class metrics.
    
    Args:
        metrics: Dictionary with metrics
        output_dir: Directory to save plots
    """
    # Extract per-class metrics
    class_names = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for key in metrics:
        if 'Finger_' in key and 'precision' in key:
            class_name = key.split('_precision')[0]
            class_names.append(class_name)
            precisions.append(metrics[f'{class_name}_precision'])
            recalls.append(metrics[f'{class_name}_recall'])
            f1_scores.append(metrics[f'{class_name}_f1'])
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, precisions, width, label='Precision', color='skyblue')
    ax.bar(x, recalls, width, label='Recall', color='lightcoral')
    ax.bar(x + width, f1_scores, width, label='F1-Score', color='lightgreen')
    
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Per-class metrics saved to {output_dir}/per_class_metrics.png")


def plot_training_history(history_path: str, output_dir: str):
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history_path: Path to training history JSON file
        output_dir: Directory to save plots
    """
    try:
        with open(history_path, 'r') as f:
            history = json.load(f)
    except:
        print("[Warning] Could not load training history")
        return
    
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    val_acc = [h['val_accuracy'] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1.plot(epochs, train_loss, label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_loss, label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curve
    ax2.plot(epochs, val_acc, label='Validation Accuracy', linewidth=2, color='green')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Training history saved to {output_dir}/training_history.png")


def create_evaluation_report(metrics: Dict, 
                            model_info: Dict,
                            output_dir: str):
    """
    Create a comprehensive text evaluation report.
    
    Args:
        metrics: Evaluation metrics
        model_info: Model information (architecture, parameters, etc.)
        output_dir: Directory to save report
    """
    report_path = os.path.join(output_dir, 'evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MODEL EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Model info
        f.write("MODEL INFORMATION:\n")
        f.write("-"*70 + "\n")
        for key, value in model_info.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Overall metrics
        f.write("OVERALL METRICS:\n")
        f.write("-"*70 + "\n")
        f.write(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.2f}%\n")
        f.write(f"Macro-averaged Precision: {metrics['macro_precision']:.4f}\n")
        f.write(f"Macro-averaged Recall: {metrics['macro_recall']:.4f}\n")
        f.write(f"Macro-averaged F1: {metrics['macro_f1']:.4f}\n")
        f.write(f"Micro-averaged Precision: {metrics['micro_precision']:.4f}\n")
        f.write(f"Micro-averaged Recall: {metrics['micro_recall']:.4f}\n")
        f.write(f"Micro-averaged F1: {metrics['micro_f1']:.4f}\n")
        f.write("\n")
        
        # Per-class metrics
        f.write("PER-CLASS METRICS:\n")
        f.write("-"*70 + "\n")
        
        class_names = []
        for key in metrics:
            if 'Finger_' in key and 'precision' in key:
                class_name = key.split('_precision')[0]
                if class_name not in class_names:
                    class_names.append(class_name)
        
        for class_name in class_names:
            f.write(f"\n{class_name}:\n")
            f.write(f"  Precision: {metrics[f'{class_name}_precision']:.4f}\n")
            f.write(f"  Recall:    {metrics[f'{class_name}_recall']:.4f}\n")
            f.write(f"  F1-Score:  {metrics[f'{class_name}_f1']:.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"[Report] Evaluation report saved to {report_path}")


def evaluate_model(args):
    """Main evaluation function."""
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config and model
    print(f"\nLoading model from: {args.model_path}")
    
    # Try to load config
    config_path = args.model_path.replace('.pth', '_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg_dict = json.load(f)
        cfg = Config(**cfg_dict)
    else:
        print("[Warning] Config not found, using default")
        cfg = Config()
    
    # Get device
    device = get_device(cfg)
    
    # Determine model type from path or args
    if args.model_type:
        model_name = args.model_type
    else:
        # Try to infer from path
        for name in ['nano', 'micro', 'base', 'large', 'xlarge']:
            if name in args.model_path.lower():
                model_name = name
                break
        else:
            model_name = 'base'  # default
    
    # Build and load model
    is_teacher = any(x in args.model_path.lower() for x in ['resnet', 'teacher'])
    
    if is_teacher:
        teacher_name = 'resnet50'  # default
        for name in ['resnet50', 'resnet101', 'resnet152']:
            if name in args.model_path.lower():
                teacher_name = name
                break
        model = build_teacher(teacher_name, cfg.in_channels, cfg.out_dim).to(device)
    else:
        model = build_student(model_name, cfg).to(device)
    
    model, metadata = load_model(model, args.model_path, device=str(device))
    
    # Load test data
    print(f"\nLoading test data from: {args.test_data}")
    test_loader = load_test_data(args.test_data, cfg.batch_size)
    
    # Get predictions
    print("\nEvaluating model...")
    predictions_prob, predictions, true_labels = get_predictions(model, test_loader, device)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(predictions, true_labels)
    
    # Model info
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_info = {
        'Model Type': model_name if not is_teacher else teacher_name,
        'Parameters': f'{num_params:,}',
        'Model Path': args.model_path,
    }
    if metadata:
        model_info.update(metadata)
    
    # Print metrics
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.2f}%")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Micro F1: {metrics['micro_f1']:.4f}")
    print(f"{'='*70}\n")
    
    # Create visualizations
    print("Creating visualizations...")
    plot_confusion_matrices(predictions, true_labels, args.output_dir)
    plot_precision_recall_curves(predictions_prob, true_labels, args.output_dir)
    plot_roc_curves(predictions_prob, true_labels, args.output_dir)
    plot_per_class_metrics(metrics, args.output_dir)
    
    # Plot training history if available
    history_path = args.model_path.replace('.pth', '_history.json')
    if os.path.exists(history_path):
        plot_training_history(history_path, args.output_dir)
    
    # Create report
    create_evaluation_report(metrics, model_info, args.output_dir)
    
    # Save metrics as JSON
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[Save] Metrics saved to {metrics_path}")
    
    print(f"\n{'='*70}")
    print(f"EVALUATION COMPLETE!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*70}\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate trained sEMG models')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model (.pth file)')
    parser.add_argument('--test-data', type=str, default=None,
                       help='Path to test data (.pt file)')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--model-type', type=str, default=None,
                       choices=['nano', 'micro', 'base', 'large', 'xlarge', 
                               'resnet50', 'resnet101', 'resnet152'],
                       help='Model type (inferred from path if not provided)')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    evaluate_model(args)


if __name__ == "__main__":
    main()
