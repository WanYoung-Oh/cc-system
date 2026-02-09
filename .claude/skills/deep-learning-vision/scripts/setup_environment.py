#!/usr/bin/env python3
"""
Detect and configure training environment (Local, AWS, GCP, Colab, etc.)
"""

import argparse
import json
import os
import platform
import sys
from pathlib import Path

try:
    import torch
except ImportError:
    print("Error: PyTorch is required.")
    print("Install with: pip install torch")
    sys.exit(1)


def detect_apple_chip():
    """Detect specific Apple Silicon chip"""
    if platform.system() != 'Darwin':
        return None

    try:
        import subprocess
        result = subprocess.run(
            ['sysctl', '-n', 'machdep.cpu.brand_string'],
            capture_output=True,
            text=True
        )
        chip_name = result.stdout.strip()

        # Detect chip generation
        if 'M1' in chip_name:
            return 'M1'
        elif 'M2' in chip_name:
            return 'M2'
        elif 'M3' in chip_name:
            return 'M3'
        elif 'M4' in chip_name:
            return 'M4'
        else:
            return 'Apple Silicon'
    except:
        return 'Apple Silicon'


def get_recommended_batch_size(chip_type):
    """Get recommended batch size for Apple Silicon chips"""
    recommendations = {
        'M1': 16,
        'M2': 32,
        'M3': 48,
        'M4': 64,
    }
    return recommendations.get(chip_type, 32)


def detect_environment():
    """Automatically detect the current environment"""

    env_info = {
        'type': 'local',
        'platform': platform.system(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpu_names': [],
        'mps_available': False,
        'apple_chip': None,
    }

    # Get GPU names
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            env_info['gpu_names'].append(torch.cuda.get_device_name(i))

    # Check for Apple Silicon MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        env_info['mps_available'] = True
        env_info['apple_chip'] = detect_apple_chip()
        if env_info['apple_chip']:
            env_info['recommended_batch_size'] = get_recommended_batch_size(env_info['apple_chip'])

    # Detect specific environments
    if 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ:
        env_info['type'] = 'colab'
        env_info['colab_type'] = 'gpu' if 'COLAB_GPU' in os.environ else 'tpu'
    elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        env_info['type'] = 'kaggle'
    elif 'SAGEMAKER_TRAINING_MODULE' in os.environ:
        env_info['type'] = 'aws_sagemaker'
    elif os.path.exists('/opt/ml/metadata/resource-metadata.json'):
        env_info['type'] = 'aws_sagemaker'
    elif 'GCP_PROJECT' in os.environ or 'GOOGLE_CLOUD_PROJECT' in os.environ:
        env_info['type'] = 'gcp'

    return env_info


def configure_distributed(env_info):
    """Configure distributed training settings"""

    dist_config = {
        'backend': 'nccl' if env_info['cuda_available'] else 'gloo',
        'world_size': env_info['gpu_count'],
        'enabled': env_info['gpu_count'] > 1,
    }

    # Environment-specific settings
    if env_info['type'] == 'aws_sagemaker':
        dist_config['init_method'] = 'env://'
    elif env_info['type'] == 'gcp':
        dist_config['init_method'] = 'env://'
    else:
        dist_config['init_method'] = 'tcp://localhost:23456'

    return dist_config


def get_data_path_recommendations(env_info):
    """Recommend data storage paths based on environment"""

    recommendations = {}

    if env_info['type'] == 'local':
        recommendations['data_dir'] = './data'
        recommendations['checkpoint_dir'] = './checkpoints'
        recommendations['log_dir'] = './logs'

    elif env_info['type'] == 'colab':
        recommendations['data_dir'] = '/content/drive/MyDrive/data'
        recommendations['checkpoint_dir'] = '/content/drive/MyDrive/checkpoints'
        recommendations['log_dir'] = '/content/drive/MyDrive/logs'
        recommendations['note'] = 'Mount Google Drive for persistent storage'

    elif env_info['type'] == 'kaggle':
        recommendations['data_dir'] = '/kaggle/input'
        recommendations['checkpoint_dir'] = '/kaggle/working/checkpoints'
        recommendations['log_dir'] = '/kaggle/working/logs'

    elif env_info['type'] == 'aws_sagemaker':
        recommendations['data_dir'] = '/opt/ml/input/data/training'
        recommendations['checkpoint_dir'] = '/opt/ml/checkpoints'
        recommendations['log_dir'] = '/opt/ml/output'
        recommendations['note'] = 'Use S3 for data input and model output'

    elif env_info['type'] == 'gcp':
        recommendations['data_dir'] = '/gcs/bucket/data'
        recommendations['checkpoint_dir'] = '/gcs/bucket/checkpoints'
        recommendations['log_dir'] = '/gcs/bucket/logs'
        recommendations['note'] = 'Use GCS bucket for storage'

    return recommendations


def print_environment_info(env_info, dist_config, path_recommendations):
    """Pretty print environment information"""

    print("=" * 60)
    print("ENVIRONMENT DETECTION")
    print("=" * 60)

    print(f"\nEnvironment Type: {env_info['type'].upper()}")
    print(f"Platform: {env_info['platform']}")
    print(f"Python: {env_info['python_version']}")
    print(f"PyTorch: {env_info['pytorch_version']}")

    print("\nGPU Information:")
    if env_info['cuda_available']:
        print(f"  CUDA Available: Yes")
        print(f"  CUDA Version: {env_info['cuda_version']}")
        print(f"  GPU Count: {env_info['gpu_count']}")
        for i, name in enumerate(env_info['gpu_names']):
            print(f"  GPU {i}: {name}")
    elif env_info['mps_available']:
        print(f"  MPS (Metal) Available: Yes")
        if env_info['apple_chip']:
            print(f"  Apple Chip: {env_info['apple_chip']}")
            if 'recommended_batch_size' in env_info:
                print(f"  Recommended Batch Size: {env_info['recommended_batch_size']}")
        print(f"  Note: Using Apple Silicon GPU acceleration")
    else:
        print(f"  GPU Available: No (CPU only)")

    print("\nDistributed Training:")
    print(f"  Enabled: {dist_config['enabled']}")
    if dist_config['enabled']:
        print(f"  Backend: {dist_config['backend']}")
        print(f"  World Size: {dist_config['world_size']}")

    print("\nRecommended Paths:")
    for key, value in path_recommendations.items():
        if key != 'note':
            print(f"  {key}: {value}")
    if 'note' in path_recommendations:
        print(f"\nNote: {path_recommendations['note']}")

    print("\n" + "=" * 60)


def save_environment_config(env_info, dist_config, path_recommendations, output_path):
    """Save environment configuration to JSON file"""

    config = {
        'environment': env_info,
        'distributed': dist_config,
        'paths': path_recommendations,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Configuration saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Detect and configure training environment'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./config/environment.json',
        help='Output configuration file path (default: ./config/environment.json)'
    )
    parser.add_argument(
        '--force-type',
        type=str,
        choices=['local', 'colab', 'kaggle', 'aws_sagemaker', 'gcp'],
        help='Force specific environment type (override auto-detection)'
    )

    args = parser.parse_args()

    # Detect environment
    env_info = detect_environment()

    # Override if specified
    if args.force_type:
        env_info['type'] = args.force_type
        print(f"⚠ Environment type forced to: {args.force_type}")

    # Configure distributed training
    dist_config = configure_distributed(env_info)

    # Get path recommendations
    path_recommendations = get_data_path_recommendations(env_info)

    # Print info
    print_environment_info(env_info, dist_config, path_recommendations)

    # Save config
    save_environment_config(env_info, dist_config, path_recommendations, args.output)

    print("\nTo use this config in training:")
    print(f"  python scripts/train.py --env-config {args.output}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
