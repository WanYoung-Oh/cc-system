#!/usr/bin/env python3
"""
Training script using PyTorch Lightning with WandB integration.
Best of both worlds: Lightning's clean code + WandB's experiment tracking.
"""

import argparse
import sys
from pathlib import Path

try:
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
    from lightning.pytorch.loggers import WandbLogger
except ImportError:
    try:
        import pytorch_lightning as L
        from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
        from pytorch_lightning.loggers import WandbLogger
    except ImportError:
        print("Error: PyTorch Lightning is required.")
        print("Install with: pip install lightning")
        sys.exit(1)

try:
    import wandb
except ImportError:
    print("Error: Weights & Biases (wandb) is required.")
    print("Install with: pip install wandb")
    print("Then login with: wandb login")
    sys.exit(1)

# Import custom modules
from lightning_module import VisionLightningModule, AVAILABLE_MODELS
from lightning_data import VisionDataModule


def main():
    parser = argparse.ArgumentParser(
        description='Train vision models with PyTorch Lightning and WandB',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

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
                        choices=['cifar10', 'cifar100', 'mnist', 'fashionmnist', 'folder'],
                        help='Dataset type')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--augmentation', type=str, default='default',
                        choices=['none', 'default', 'strong', 'autoaugment'],
                        help='Augmentation strategy')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio')

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
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'plateau', 'none'],
                        help='Learning rate scheduler')

    # Dataloader arguments
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')

    # Trainer arguments
    parser.add_argument('--accelerator', type=str, default='auto',
                        choices=['auto', 'cpu', 'gpu', 'mps', 'cuda'],
                        help='Accelerator type')
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of devices')
    parser.add_argument('--precision', type=str, default='32',
                        choices=['32', '16', 'bf16'],
                        help='Training precision')
    parser.add_argument('--gradient-clip-val', type=float, default=0.0,
                        help='Gradient clipping value (0 = disabled)')

    # WandB arguments
    parser.add_argument('--wandb-project', type=str, required=True,
                        help='WandB project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='WandB entity (username or team)')
    parser.add_argument('--wandb-name', type=str, default=None,
                        help='WandB run name')
    parser.add_argument('--wandb-tags', type=str, nargs='+', default=[],
                        help='WandB tags for this run')
    parser.add_argument('--wandb-notes', type=str, default=None,
                        help='WandB run notes')
    parser.add_argument('--wandb-mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'],
                        help='WandB mode')

    # Checkpoint arguments
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save-top-k', type=int, default=3,
                        help='Save top k checkpoints')

    # Other arguments
    parser.add_argument('--early-stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--fast-dev-run', action='store_true',
                        help='Run 1 train/val batch for debugging')

    args = parser.parse_args()

    # Set seed
    L.seed_everything(args.seed)

    print("=" * 70)
    print("PYTORCH LIGHTNING + WANDB TRAINING")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Num Classes: {args.num_classes}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Scheduler: {args.scheduler}")
    print(f"Augmentation: {args.augmentation}")
    print(f"Accelerator: {args.accelerator}")
    print(f"Precision: {args.precision}")
    print(f"WandB Project: {args.wandb_project}")
    print("=" * 70)

    # Create data module
    data_module = VisionDataModule(
        data_dir=args.data_dir,
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        augmentation=args.augmentation,
        val_split=args.val_split,
    )

    # Create model
    model = VisionLightningModule(
        model_name=args.model,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        learning_rate=args.lr,
        optimizer=args.optimizer,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        scheduler=args.scheduler,
        max_epochs=args.epochs,
    )

    # Create callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=Path(args.checkpoint_dir) / args.model,
        filename='{epoch}-{val/acc:.4f}',
        monitor='val/acc',
        mode='max',
        save_top_k=args.save_top_k,
        save_last=True,
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    # Early stopping
    if args.early_stopping:
        early_stop_callback = EarlyStopping(
            monitor='val/acc',
            mode='max',
            patience=args.patience,
            verbose=True,
        )
        callbacks.append(early_stop_callback)

    # WandB Logger
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        tags=args.wandb_tags,
        notes=args.wandb_notes,
        mode=args.wandb_mode,
        log_model=True,  # Log checkpoints to WandB
    )

    # Watch model with WandB (log gradients and model topology)
    wandb_logger.watch(model, log='all', log_freq=100)

    print(f"\n✓ WandB initialized: {wandb_logger.experiment.name}")
    print(f"  Project: {args.wandb_project}")
    print(f"  Run URL: {wandb_logger.experiment.url}")

    # Create trainer
    trainer = L.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_val=args.gradient_clip_val if args.gradient_clip_val > 0 else None,
        fast_dev_run=args.fast_dev_run,
        deterministic=True,
        log_every_n_steps=50,
    )

    # Train
    print("\nStarting training...")
    trainer.fit(model, data_module)

    # Test
    print("\nRunning test...")
    test_results = trainer.test(model, data_module)

    # Log best metrics to WandB summary
    wandb_logger.experiment.summary['best_val_acc'] = checkpoint_callback.best_model_score
    wandb_logger.experiment.summary['best_checkpoint'] = checkpoint_callback.best_model_path

    # Print results
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val/acc: {checkpoint_callback.best_model_score:.4f}")
    print(f"WandB run: {wandb_logger.experiment.url}")
    print("\n✓ All metrics and checkpoints logged to WandB")

    # Finish WandB run
    wandb.finish()

    return 0


if __name__ == '__main__':
    sys.exit(main())
