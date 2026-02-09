#!/usr/bin/env python3
"""
Data preprocessing and augmentation utilities.
Handles image resizing, normalization, and augmentation strategies.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import torch
    import torchvision.transforms as transforms
except ImportError:
    print("Error: PyTorch and torchvision are required.")
    print("Install with: pip install torch torchvision")
    sys.exit(1)


# Standard ImageNet normalization
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# CIFAR-10/100 normalization
CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2470, 0.2435, 0.2616]


def get_augmentation_config(preset='default'):
    """Get augmentation configuration presets"""

    configs = {
        'none': {
            'description': 'No augmentation (only resize and normalize)',
            'train': ['resize', 'center_crop', 'to_tensor', 'normalize'],
            'val': ['resize', 'center_crop', 'to_tensor', 'normalize'],
        },
        'default': {
            'description': 'Standard augmentation for classification',
            'train': ['resize', 'random_crop', 'random_horizontal_flip', 'to_tensor', 'normalize'],
            'val': ['resize', 'center_crop', 'to_tensor', 'normalize'],
        },
        'strong': {
            'description': 'Strong augmentation with color jittering and rotation',
            'train': [
                'resize', 'random_crop', 'random_horizontal_flip',
                'color_jitter', 'random_rotation', 'to_tensor', 'normalize'
            ],
            'val': ['resize', 'center_crop', 'to_tensor', 'normalize'],
        },
        'autoaugment': {
            'description': 'AutoAugment policy for CIFAR/ImageNet',
            'train': [
                'resize', 'random_crop', 'random_horizontal_flip',
                'autoaugment', 'to_tensor', 'normalize'
            ],
            'val': ['resize', 'center_crop', 'to_tensor', 'normalize'],
        },
    }

    return configs.get(preset, configs['default'])


def build_transforms(config, image_size=224, dataset_type='imagenet', is_training=True):
    """Build torchvision transforms from config"""

    # Select normalization stats
    if dataset_type == 'cifar':
        mean, std = CIFAR_MEAN, CIFAR_STD
    else:  # imagenet or custom
        mean, std = IMAGENET_MEAN, IMAGENET_STD

    # Get transform list
    transform_list = config['train'] if is_training else config['val']

    # Build transforms
    ops = []

    for op in transform_list:
        if op == 'resize':
            ops.append(transforms.Resize(image_size))
        elif op == 'random_crop':
            ops.append(transforms.RandomCrop(image_size, padding=4))
        elif op == 'center_crop':
            ops.append(transforms.CenterCrop(image_size))
        elif op == 'random_horizontal_flip':
            ops.append(transforms.RandomHorizontalFlip())
        elif op == 'color_jitter':
            ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
        elif op == 'random_rotation':
            ops.append(transforms.RandomRotation(15))
        elif op == 'autoaugment':
            if dataset_type == 'cifar':
                ops.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10))
            else:
                ops.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))
        elif op == 'to_tensor':
            ops.append(transforms.ToTensor())
        elif op == 'normalize':
            ops.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(ops)


def save_config(config, output_path):
    """Save preprocessing config to JSON file"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✓ Configuration saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate data preprocessing configuration')
    parser.add_argument(
        '--preset',
        type=str,
        choices=['none', 'default', 'strong', 'autoaugment'],
        default='default',
        help='Augmentation preset (default: default)'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Target image size (default: 224)'
    )
    parser.add_argument(
        '--dataset-type',
        type=str,
        choices=['imagenet', 'cifar', 'custom'],
        default='imagenet',
        help='Dataset type for normalization stats (default: imagenet)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./config/preprocessing.json',
        help='Output configuration file path'
    )
    parser.add_argument(
        '--list-presets',
        action='store_true',
        help='List all available presets'
    )

    args = parser.parse_args()

    # List presets
    if args.list_presets:
        print("Available augmentation presets:\n")
        for preset, config in [
            ('none', get_augmentation_config('none')),
            ('default', get_augmentation_config('default')),
            ('strong', get_augmentation_config('strong')),
            ('autoaugment', get_augmentation_config('autoaugment')),
        ]:
            print(f"  {preset}:")
            print(f"    {config['description']}")
            print(f"    Train: {', '.join(config['train'])}")
            print(f"    Val: {', '.join(config['val'])}\n")
        return 0

    # Generate config
    aug_config = get_augmentation_config(args.preset)

    config = {
        'preset': args.preset,
        'description': aug_config['description'],
        'image_size': args.image_size,
        'dataset_type': args.dataset_type,
        'normalization': {
            'mean': CIFAR_MEAN if args.dataset_type == 'cifar' else IMAGENET_MEAN,
            'std': CIFAR_STD if args.dataset_type == 'cifar' else IMAGENET_STD,
        },
        'transforms': {
            'train': aug_config['train'],
            'val': aug_config['val'],
        }
    }

    print(f"Preprocessing configuration:")
    print(f"  Preset: {args.preset}")
    print(f"  Description: {aug_config['description']}")
    print(f"  Image size: {args.image_size}")
    print(f"  Dataset type: {args.dataset_type}")
    print("-" * 50)

    # Save config
    save_config(config, args.output)

    print("\nTo use this config in training:")
    print(f"  python scripts/train.py --preprocessing-config {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
