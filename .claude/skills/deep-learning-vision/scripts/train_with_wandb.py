#!/usr/bin/env python3
"""
Training script with Weights & Biases (WandB) integration for experiment tracking.
Logs metrics, hyperparameters, model architecture, and visualizations.
"""

import argparse
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

try:
    import wandb
except ImportError:
    print("Error: Weights & Biases (wandb) is required.")
    print("Install with: pip install wandb")
    print("Then login with: wandb login")
    sys.exit(1)


# Import model factory from train.py
AVAILABLE_MODELS = {
    # ResNet family
    'resnet18': lambda num_classes, pretrained: models.resnet18(pretrained=pretrained, num_classes=num_classes),
    'resnet34': lambda num_classes, pretrained: models.resnet34(pretrained=pretrained, num_classes=num_classes),
    'resnet50': lambda num_classes, pretrained: models.resnet50(pretrained=pretrained, num_classes=num_classes),
    'resnet101': lambda num_classes, pretrained: models.resnet101(pretrained=pretrained, num_classes=num_classes),

    # EfficientNet family
    'efficientnet_b0': lambda num_classes, pretrained: models.efficientnet_b0(pretrained=pretrained, num_classes=num_classes),
    'efficientnet_b1': lambda num_classes, pretrained: models.efficientnet_b1(pretrained=pretrained, num_classes=num_classes),
    'efficientnet_b2': lambda num_classes, pretrained: models.efficientnet_b2(pretrained=pretrained, num_classes=num_classes),
    'efficientnet_b3': lambda num_classes, pretrained: models.efficientnet_b3(pretrained=pretrained, num_classes=num_classes),

    # Vision Transformer
    'vit_b_16': lambda num_classes, pretrained: models.vit_b_16(pretrained=pretrained, num_classes=num_classes),
    'vit_b_32': lambda num_classes, pretrained: models.vit_b_32(pretrained=pretrained, num_classes=num_classes),

    # Swin Transformer family
    'swin_t': lambda num_classes, pretrained: models.swin_t(pretrained=pretrained, num_classes=num_classes),
    'swin_s': lambda num_classes, pretrained: models.swin_s(pretrained=pretrained, num_classes=num_classes),
    'swin_b': lambda num_classes, pretrained: models.swin_b(pretrained=pretrained, num_classes=num_classes),

    # ConvNeXt family
    'convnext_tiny': lambda num_classes, pretrained: models.convnext_tiny(pretrained=pretrained, num_classes=num_classes),
    'convnext_small': lambda num_classes, pretrained: models.convnext_small(pretrained=pretrained, num_classes=num_classes),
    'convnext_base': lambda num_classes, pretrained: models.convnext_base(pretrained=pretrained, num_classes=num_classes),

    # MobileNet
    'mobilenet_v2': lambda num_classes, pretrained: models.mobilenet_v2(pretrained=pretrained, num_classes=num_classes),
    'mobilenet_v3_large': lambda num_classes, pretrained: models.mobilenet_v3_large(pretrained=pretrained, num_classes=num_classes),

    # DenseNet
    'densenet121': lambda num_classes, pretrained: models.densenet121(pretrained=pretrained, num_classes=num_classes),
    'densenet161': lambda num_classes, pretrained: models.densenet161(pretrained=pretrained, num_classes=num_classes),
}


def create_model(model_name, num_classes, pretrained=True):
    """Create model from factory"""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model {model_name} not supported. Available: {list(AVAILABLE_MODELS.keys())}")
    return AVAILABLE_MODELS[model_name](num_classes, pretrained)


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch with logging"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    batch_losses = []

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        batch_losses.append(loss.item())
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Log to WandB every 100 batches
        if batch_idx % 100 == 0:
            wandb.log({
                'batch': epoch * len(train_loader) + batch_idx,
                'train/batch_loss': loss.item(),
                'train/batch_acc': 100. * correct / total,
            })

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    # For confusion matrix
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total

    return val_loss, val_acc, all_preds, all_targets


def main():
    parser = argparse.ArgumentParser(description='Train with WandB integration')

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
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='Dataset type')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam', 'adamw'],
                        help='Optimizer')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay')

    # WandB arguments
    parser.add_argument('--wandb-project', type=str, required=True,
                        help='WandB project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='WandB entity (username or team)')
    parser.add_argument('--wandb-name', type=str, default=None,
                        help='WandB run name (default: auto-generated)')
    parser.add_argument('--wandb-tags', type=str, nargs='+', default=[],
                        help='WandB tags for this run')
    parser.add_argument('--wandb-notes', type=str, default=None,
                        help='WandB run notes')

    # Environment arguments
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Device to use')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')

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

    # Initialize WandB
    config = vars(args)
    config['device'] = str(device)

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        config=config,
        tags=args.wandb_tags,
        notes=args.wandb_notes,
    )

    print(f"✓ WandB initialized: {wandb.run.name}")
    print(f"  Project: {args.wandb_project}")
    print(f"  Run URL: {wandb.run.url}")

    # Create model
    model = create_model(args.model, args.num_classes, args.pretrained)
    model = model.to(device)

    # Log model architecture to WandB
    wandb.watch(model, log='all', log_freq=100)

    # Load dataset
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
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif args.dataset == 'cifar100':
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
        train_dataset = torchvision.datasets.CIFAR100(root=args.data_dir, train=True, download=True, transform=transform_train)
        val_dataset = torchvision.datasets.CIFAR100(root=args.data_dir, train=False, transform=transform_val)
        class_names = None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir) / wandb.run.id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print("\n" + "=" * 60)
    print("TRAINING START")
    print("=" * 60)

    best_acc = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch [{epoch}/{args.epochs}]")

        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")

        # Validate
        val_loss, val_acc, val_preds, val_targets = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Log to WandB
        wandb.log({
            'epoch': epoch,
            'train/loss': train_loss,
            'train/acc': train_acc,
            'val/loss': val_loss,
            'val/acc': val_acc,
            'learning_rate': scheduler.get_last_lr()[0],
        })

        # Log confusion matrix every 10 epochs
        if epoch % 10 == 0 and class_names:
            wandb.log({
                'confusion_matrix': wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=val_targets,
                    preds=val_preds,
                    class_names=class_names
                )
            })

        scheduler.step()

        # Save checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
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
            checkpoint_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, checkpoint_path)

            # Save to WandB
            wandb.save(str(checkpoint_path))
            print(f"✓ Best model saved and uploaded to WandB")

    # Training complete
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_acc:.2f}%")

    # Log final summary
    wandb.run.summary['best_val_acc'] = best_acc
    wandb.run.summary['total_time_hours'] = elapsed_time / 3600

    wandb.finish()
    print(f"\n✓ WandB run completed: {wandb.run.url}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
