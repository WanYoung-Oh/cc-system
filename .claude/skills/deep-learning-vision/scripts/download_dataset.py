#!/usr/bin/env python3
"""
Download popular computer vision datasets.
Supports: CIFAR-10, CIFAR-100, ImageNet (via kaggle), COCO, VOC
"""

import argparse
import os
import sys
from pathlib import Path

try:
    import torch
    import torchvision
    import torchvision.datasets as datasets
except ImportError:
    print("Error: PyTorch and torchvision are required.")
    print("Install with: pip install torch torchvision")
    sys.exit(1)


SUPPORTED_DATASETS = {
    'cifar10': 'CIFAR-10 (60k 32x32 color images in 10 classes)',
    'cifar100': 'CIFAR-100 (60k 32x32 color images in 100 classes)',
    'mnist': 'MNIST (70k 28x28 grayscale handwritten digits)',
    'fashionmnist': 'Fashion-MNIST (70k 28x28 grayscale fashion items)',
    'coco': 'COCO 2017 (requires manual download or fiftyone)',
    'voc': 'PASCAL VOC (requires manual download)',
}


def download_cifar10(data_dir):
    """Download CIFAR-10 dataset"""
    print("Downloading CIFAR-10...")
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True)
    print(f"✓ Downloaded CIFAR-10 to {data_dir}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    return True


def download_cifar100(data_dir):
    """Download CIFAR-100 dataset"""
    print("Downloading CIFAR-100...")
    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True)
    print(f"✓ Downloaded CIFAR-100 to {data_dir}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    return True


def download_mnist(data_dir):
    """Download MNIST dataset"""
    print("Downloading MNIST...")
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True)
    print(f"✓ Downloaded MNIST to {data_dir}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    return True


def download_fashionmnist(data_dir):
    """Download Fashion-MNIST dataset"""
    print("Downloading Fashion-MNIST...")
    train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True)
    test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True)
    print(f"✓ Downloaded Fashion-MNIST to {data_dir}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    return True


def download_coco(data_dir):
    """Instructions for downloading COCO dataset"""
    print("\nCOCO dataset requires manual download or fiftyone package.")
    print("\nOption 1: Using fiftyone (recommended):")
    print("  pip install fiftyone")
    print("  python -c \"import fiftyone.zoo as foz; foz.load_zoo_dataset('coco-2017', split='train')\"")
    print("\nOption 2: Manual download:")
    print("  Visit: https://cocodataset.org/#download")
    print("  Download train2017.zip, val2017.zip, and annotations")
    print(f"  Extract to: {data_dir}/coco/")
    return False


def download_voc(data_dir):
    """Instructions for downloading PASCAL VOC dataset"""
    print("\nPASCAL VOC dataset download:")
    print("  Visit: http://host.robots.ox.ac.uk/pascal/VOC/")
    print("  Download VOC2012 or VOC2007")
    print(f"  Extract to: {data_dir}/voc/")
    print("\nOr use torchvision:")
    print("  from torchvision.datasets import VOCDetection")
    print(f"  VOCDetection('{data_dir}', year='2012', download=True)")
    return False


def main():
    parser = argparse.ArgumentParser(
        description='Download popular computer vision datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='\n'.join([f"  {k}: {v}" for k, v in SUPPORTED_DATASETS.items()])
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=list(SUPPORTED_DATASETS.keys()),
        help='Dataset to download'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data',
        help='Directory to save dataset (default: ./data)'
    )

    args = parser.parse_args()

    # Create data directory
    data_dir = Path(args.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {data_dir}")
    print("-" * 50)

    # Download dataset
    downloaders = {
        'cifar10': download_cifar10,
        'cifar100': download_cifar100,
        'mnist': download_mnist,
        'fashionmnist': download_fashionmnist,
        'coco': download_coco,
        'voc': download_voc,
    }

    success = downloaders[args.dataset](str(data_dir))

    if success:
        print("\n✓ Download completed successfully!")
    else:
        print("\n⚠ Please follow the instructions above to complete download.")

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
