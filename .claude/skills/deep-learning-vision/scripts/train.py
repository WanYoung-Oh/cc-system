#!/usr/bin/env python3
"""
Universal training script for image classification and object detection.
Supports multiple model architectures and easy experimentation.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import torchvision
    import torchvision.transforms as transforms
    import torchvision.models as models
except ImportError:
    print("Error: PyTorch and torchvision are required.")
    print("Install with: pip install torch torchvision")
    sys.exit(1)


# ============================================================================
# Model Factory - Easy model selection and experimentation
# ============================================================================

AVAILABLE_MODELS = {
    # ResNet family
    'resnet18': lambda num_classes, pretrained: models.resnet18(pretrained=pretrained, num_classes=num_classes),
    'resnet34': lambda num_classes, pretrained: models.resnet34(pretrained=pretrained, num_classes=num_classes),
    'resnet50': lambda num_classes, pretrained: models.resnet50(pretrained=pretrained, num_classes=num_classes),
    'resnet101': lambda num_classes, pretrained: models.resnet101(pretrained=pretrained, num_classes=num_classes),
    'resnet152': lambda num_classes, pretrained: models.resnet152(pretrained=pretrained, num_classes=num_classes),

    # EfficientNet family
    'efficientnet_b0': lambda num_classes, pretrained: models.efficientnet_b0(pretrained=pretrained, num_classes=num_classes),
    'efficientnet_b1': lambda num_classes, pretrained: models.efficientnet_b1(pretrained=pretrained, num_classes=num_classes),
    'efficientnet_b2': lambda num_classes, pretrained: models.efficientnet_b2(pretrained=pretrained, num_classes=num_classes),
    'efficientnet_b3': lambda num_classes, pretrained: models.efficientnet_b3(pretrained=pretrained, num_classes=num_classes),
    'efficientnet_b4': lambda num_classes, pretrained: models.efficientnet_b4(pretrained=pretrained, num_classes=num_classes),

    # Vision Transformer
    'vit_b_16': lambda num_classes, pretrained: models.vit_b_16(pretrained=pretrained, num_classes=num_classes),
    'vit_b_32': lambda num_classes, pretrained: models.vit_b_32(pretrained=pretrained, num_classes=num_classes),
    'vit_l_16': lambda num_classes, pretrained: models.vit_l_16(pretrained=pretrained, num_classes=num_classes),

    # MobileNet family
    'mobilenet_v2': lambda num_classes, pretrained: models.mobilenet_v2(pretrained=pretrained, num_classes=num_classes),
    'mobilenet_v3_small': lambda num_classes, pretrained: models.mobilenet_v3_small(pretrained=pretrained, num_classes=num_classes),
    'mobilenet_v3_large': lambda num_classes, pretrained: models.mobilenet_v3_large(pretrained=pretrained, num_classes=num_classes),

    # DenseNet family
    'densenet121': lambda num_classes, pretrained: models.densenet121(pretrained=pretrained, num_classes=num_classes),
    'densenet161': lambda num_classes, pretrained: models.densenet161(pretrained=pretrained, num_classes=num_classes),
    'densenet169': lambda num_classes, pretrained: models.densenet169(pretrained=pretrained, num_classes=num_classes),

    # Swin Transformer family
    'swin_t': lambda num_classes, pretrained: models.swin_t(pretrained=pretrained, num_classes=num_classes),
    'swin_s': lambda num_classes, pretrained: models.swin_s(pretrained=pretrained, num_classes=num_classes),
    'swin_b': lambda num_classes, pretrained: models.swin_b(pretrained=pretrained, num_classes=num_classes),

    # ConvNeXt family
    'convnext_tiny': lambda num_classes, pretrained: models.convnext_tiny(pretrained=pretrained, num_classes=num_classes),
    'convnext_small': lambda num_classes, pretrained: models.convnext_small(pretrained=pretrained, num_classes=num_classes),
    'convnext_base': lambda num_classes, pretrained: models.convnext_base(pretrained=pretrained, num_classes=num_classes),
    'convnext_large': lambda num_classes, pretrained: models.convnext_large(pretrained=pretrained, num_classes=num_classes),

    # Other popular models
    'vgg16': lambda num_classes, pretrained: models.vgg16(pretrained=pretrained, num_classes=num_classes),
    'vgg19': lambda num_classes, pretrained: models.vgg19(pretrained=pretrained, num_classes=num_classes),
    'alexnet': lambda num_classes, pretrained: models.alexnet(pretrained=pretrained, num_classes=num_classes),
    'squeezenet': lambda num_classes, pretrained: models.squeezenet1_1(pretrained=pretrained, num_classes=num_classes),
}


def create_model(model_name, num_classes, pretrained=True):
    """Create model from factory"""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not supported. Available: {list(AVAILABLE_MODELS.keys())}")

    print(f"Creating model: {model_name}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Pretrained: {pretrained}")

    model = AVAILABLE_MODELS[model_name](num_classes, pretrained)
    return model


# ============================================================================
# Training Loop
# ============================================================================

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Print progress
        if batch_idx % 100 == 0:
            print(f"  Batch [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Acc: {100.*correct/total:.2f}%")

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total

    return val_loss, val_acc


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train image classification model')

    # Model arguments
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=list(AVAILABLE_MODELS.keys()),
                        help='Model architecture')
    parser.add_argument('--num-classes', type=int, required=True,
                        help='Number of output classes')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')

    # Dataset arguments
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--dataset', type=str, default='folder',
                        choices=['cifar10', 'cifar100', 'folder'],
                        help='Dataset type')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adamw'],
                        help='Optimizer')

    # Environment arguments
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to use for training')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    # Checkpoint arguments
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

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

    # Create model
    model = create_model(args.model, args.num_classes, args.pretrained)
    model = model.to(device)

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    # This is simplified - actual implementation would load from data_dir
    # For demonstration purposes
    if args.dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_val)
    else:
        print("⚠ Custom dataset loading not fully implemented in this template")
        print("  Implement ImageFolder or custom Dataset class for your data")
        return 1

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")

    # Setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Setup loss
    criterion = nn.CrossEntropyLoss()

    # Setup scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING START")
    print("=" * 60)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")
        print("-" * 40)

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Update scheduler
        scheduler.step()
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Save checkpoint
        if epoch % args.save_interval == 0 or val_acc > best_acc:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'args': vars(args),
            }

            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}")

            if val_acc > best_acc:
                best_acc = val_acc
                best_path = checkpoint_dir / "best_model.pth"
                torch.save(checkpoint, best_path)
                print(f"✓ New best model saved: {best_path}")

    # Training complete
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_acc:.2f}%")

    return 0


if __name__ == '__main__':
    sys.exit(main())
