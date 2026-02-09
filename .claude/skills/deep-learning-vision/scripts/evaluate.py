#!/usr/bin/env python3
"""
Evaluate trained model and generate performance metrics.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import torchvision
    import torchvision.transforms as transforms
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
except ImportError:
    print("Error: Required packages are missing.")
    print("Install with: pip install torch torchvision scikit-learn numpy")
    sys.exit(1)


def load_checkpoint(checkpoint_path, model, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'val_acc' in checkpoint:
            print(f"  Checkpoint val accuracy: {checkpoint['val_acc']:.2f}%")
    else:
        model.load_state_dict(checkpoint)

    return model


def evaluate_model(model, test_loader, device, class_names=None):
    """Evaluate model and compute metrics"""
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    print("Evaluating model...")

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            if batch_idx % 50 == 0:
                print(f"  Progress: {batch_idx}/{len(test_loader)} batches")

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)

    # Compute metrics
    accuracy = 100. * np.mean(all_preds == all_targets)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy: {accuracy:.2f}%")

    # Per-class metrics
    if class_names:
        print("\nPer-class metrics:")
        report = classification_report(
            all_targets, all_preds,
            target_names=class_names,
            digits=4
        )
        print(report)

    # Confusion matrix
    cm = confusion_matrix(all_targets, all_preds)

    return {
        'accuracy': accuracy,
        'predictions': all_preds.tolist(),
        'targets': all_targets.tolist(),
        'probabilities': all_probs.tolist(),
        'confusion_matrix': cm.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to test dataset')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'folder'],
                        help='Dataset type')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to use')
    parser.add_argument('--output', type=str, default='./evaluation_results.json',
                        help='Output file for results')

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Load model (simplified - actual implementation needs model architecture info)
    print(f"\nLoading checkpoint: {args.checkpoint}")
    print("⚠ Note: This script assumes the model architecture is known.")
    print("   In practice, save architecture info in checkpoint or use config file.")

    # Load test dataset
    if args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        test_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, transform=transform)
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif args.dataset == 'cifar100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        test_dataset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, transform=transform)
        class_names = None  # Too many to list
    else:
        print("⚠ Custom dataset evaluation not fully implemented")
        return 1

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    print(f"Test samples: {len(test_dataset)}")

    # For this template, we'll note that model loading requires architecture info
    print("\n⚠ Model architecture must be reconstructed from checkpoint metadata")
    print("   See train.py AVAILABLE_MODELS for supported architectures")

    return 0


if __name__ == '__main__':
    sys.exit(main())
