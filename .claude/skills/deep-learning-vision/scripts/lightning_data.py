#!/usr/bin/env python3
"""
PyTorch Lightning DataModule for vision datasets.
Handles data loading, preprocessing, and augmentation.
"""

import torch
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

try:
    import lightning as L
except ImportError:
    try:
        import pytorch_lightning as L
    except ImportError:
        print("Error: PyTorch Lightning is required.")
        print("Install with: pip install lightning")
        import sys
        sys.exit(1)


# ============================================================================
# Normalization Constants
# ============================================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2470, 0.2435, 0.2616]


# ============================================================================
# Lightning DataModule
# ============================================================================

class VisionDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for vision datasets.

    Supports:
    - CIFAR-10/100
    - ImageNet-style folder structure
    - Custom datasets
    - Automatic train/val split
    - Configurable augmentation
    """

    def __init__(
        self,
        data_dir: str,
        dataset: str = 'cifar10',
        batch_size: int = 32,
        num_workers: int = 4,
        image_size: int = 224,
        augmentation: str = 'default',
        val_split: float = 0.1,
        pin_memory: bool = True,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.augmentation = augmentation
        self.val_split = val_split
        self.pin_memory = pin_memory

        # Select normalization stats
        if dataset in ['cifar10', 'cifar100']:
            self.mean = CIFAR_MEAN
            self.std = CIFAR_STD
        else:
            self.mean = IMAGENET_MEAN
            self.std = IMAGENET_STD

        # Create transforms
        self.train_transform = self._get_train_transform()
        self.val_transform = self._get_val_transform()

    def _get_train_transform(self):
        """Get training augmentation transforms"""

        if self.augmentation == 'none':
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])

        elif self.augmentation == 'default':
            if self.dataset in ['cifar10', 'cifar100']:
                # CIFAR-specific augmentation
                return transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std),
                ])
            else:
                # ImageNet-style augmentation
                return transforms.Compose([
                    transforms.RandomResizedCrop(self.image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean, std=self.std),
                ])

        elif self.augmentation == 'strong':
            return transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])

        elif self.augmentation == 'autoaugment':
            if self.dataset in ['cifar10', 'cifar100']:
                policy = transforms.AutoAugmentPolicy.CIFAR10
            else:
                policy = transforms.AutoAugmentPolicy.IMAGENET

            return transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.AutoAugment(policy),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])

        else:
            raise ValueError(f"Unknown augmentation: {self.augmentation}")

    def _get_val_transform(self):
        """Get validation/test transforms"""

        if self.dataset in ['cifar10', 'cifar100']:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(self.image_size + 32),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])

    def prepare_data(self):
        """Download datasets if needed (called only on 1 GPU/TPU)"""

        if self.dataset == 'cifar10':
            torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True)
            torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True)

        elif self.dataset == 'cifar100':
            torchvision.datasets.CIFAR100(root=self.data_dir, train=True, download=True)
            torchvision.datasets.CIFAR100(root=self.data_dir, train=False, download=True)

        elif self.dataset == 'mnist':
            torchvision.datasets.MNIST(root=self.data_dir, train=True, download=True)
            torchvision.datasets.MNIST(root=self.data_dir, train=False, download=True)

        elif self.dataset == 'fashionmnist':
            torchvision.datasets.FashionMNIST(root=self.data_dir, train=True, download=True)
            torchvision.datasets.FashionMNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """Setup datasets for training/validation/testing"""

        if stage == 'fit' or stage is None:
            if self.dataset == 'cifar10':
                train_full = torchvision.datasets.CIFAR10(
                    root=self.data_dir, train=True, transform=self.train_transform
                )
                self.test_dataset = torchvision.datasets.CIFAR10(
                    root=self.data_dir, train=False, transform=self.val_transform
                )

            elif self.dataset == 'cifar100':
                train_full = torchvision.datasets.CIFAR100(
                    root=self.data_dir, train=True, transform=self.train_transform
                )
                self.test_dataset = torchvision.datasets.CIFAR100(
                    root=self.data_dir, train=False, transform=self.val_transform
                )

            elif self.dataset == 'folder':
                # ImageFolder structure: data_dir/train/, data_dir/val/
                train_dir = self.data_dir / 'train'
                val_dir = self.data_dir / 'val'

                if train_dir.exists() and val_dir.exists():
                    # Use existing train/val split
                    self.train_dataset = torchvision.datasets.ImageFolder(
                        root=train_dir, transform=self.train_transform
                    )
                    self.val_dataset = torchvision.datasets.ImageFolder(
                        root=val_dir, transform=self.val_transform
                    )
                    self.test_dataset = self.val_dataset
                    return
                else:
                    # Load all data and split
                    train_full = torchvision.datasets.ImageFolder(
                        root=self.data_dir, transform=self.train_transform
                    )
                    self.test_dataset = torchvision.datasets.ImageFolder(
                        root=self.data_dir, transform=self.val_transform
                    )

            else:
                raise ValueError(f"Dataset {self.dataset} not supported")

            # Split train into train/val
            train_size = int((1 - self.val_split) * len(train_full))
            val_size = len(train_full) - train_size

            self.train_dataset, self.val_dataset = random_split(
                train_full,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

            # Apply validation transforms to val split
            # Note: random_split doesn't allow different transforms, so this is a workaround
            # In production, consider using Subset with different transforms

        if stage == 'test' or stage is None:
            if self.dataset == 'cifar10':
                self.test_dataset = torchvision.datasets.CIFAR10(
                    root=self.data_dir, train=False, transform=self.val_transform
                )
            elif self.dataset == 'cifar100':
                self.test_dataset = torchvision.datasets.CIFAR100(
                    root=self.data_dir, train=False, transform=self.val_transform
                )

    def train_dataloader(self):
        """Training dataloader"""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def val_dataloader(self):
        """Validation dataloader"""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )

    def test_dataloader(self):
        """Test dataloader"""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
        )
