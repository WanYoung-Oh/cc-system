---
name: deep-learning-vision
description: Complete PyTorch/Lightning computer vision pipeline for image classification and object detection. Use when users need to (1) download and preprocess image datasets, (2) train deep learning models (ResNet, EfficientNet, ViT, Swin Transformer, ConvNeXt, etc.) with easy model experimentation, (3) set up training environments (local GPU with CUDA/Apple M1-M4, AWS, GCP, Colab), (4) track experiments with WandB, or (5) evaluate and optimize vision models. Includes both vanilla PyTorch and PyTorch Lightning implementations, 25+ model architectures, optimized Apple Silicon support for M3/M4, and document understanding models (DiT, LayoutLMv3). Supports full workflow from data collection to model deployment with clean, production-ready code.
---

# Deep Learning Vision

End-to-end PyTorch/Lightning workflow for computer vision tasks with emphasis on easy model experimentation, clean code structure, and multi-environment support.

## Training Frameworks

This skill provides **two training frameworks**:

1. **Vanilla PyTorch** (`train.py`, `train_with_wandb.py`) - Full control, explicit training loops
2. **PyTorch Lightning** (`train_lightning.py`, `train_lightning_wandb.py`) - Clean code, less boilerplate, production-ready

See [references/pytorch_lightning.md](references/pytorch_lightning.md) for detailed comparison and migration guide.

## Prerequisites

**Essential packages:**
```bash
pip install -r assets/project_template/requirements.txt
```

**For WandB tracking:**
```bash
wandb login
```

## Quick Start

### 1. Setup Environment

Detect and configure your training environment (local, Colab, AWS, GCP):

```bash
python scripts/setup_environment.py --output config/environment.json
```

Output includes:
- Environment type and GPU availability
- Recommended data paths
- Distributed training configuration

### 2. Download Dataset

Download popular vision datasets:

```bash
# CIFAR-10 (60k images, 10 classes)
python scripts/download_dataset.py --dataset cifar10 --data-dir ./data

# CIFAR-100 (60k images, 100 classes)
python scripts/download_dataset.py --dataset cifar100 --data-dir ./data

# MNIST
python scripts/download_dataset.py --dataset mnist --data-dir ./data

# List all available datasets
python scripts/download_dataset.py --help
```

For COCO or custom datasets, see script output for instructions.

### 3. Configure Preprocessing

Generate preprocessing configuration:

```bash
python scripts/preprocess_data.py \
  --preset default \
  --image-size 224 \
  --dataset-type imagenet \
  --output config/preprocessing.json
```

**Augmentation presets:**
- `none`: Resize and normalize only
- `default`: Standard augmentation (random crop, flip)
- `strong`: Color jittering + rotation
- `autoaugment`: AutoAugment policy

List all presets: `python scripts/preprocess_data.py --list-presets`

### 4. Train Model

**Choose your training framework:**

#### A. PyTorch Lightning (Recommended - Less code, more features)

```bash
# Basic Lightning training
python scripts/train_lightning.py \
  --model resnet50 \
  --num-classes 10 \
  --data-dir ./data \
  --dataset cifar10 \
  --epochs 100 \
  --batch-size 32

# Lightning with WandB (cleanest approach)
python scripts/train_lightning_wandb.py \
  --model convnext_small \
  --num-classes 10 \
  --data-dir ./data \
  --dataset cifar10 \
  --wandb-project vision-experiments \
  --wandb-name convnext-baseline \
  --precision 16  # Mixed precision!
  --devices 4  # Multi-GPU automatic!
```

**Lightning benefits:**
- 80% less boilerplate code
- Automatic multi-GPU/TPU support
- Built-in mixed precision training
- Automatic checkpointing and logging
- Production-ready code structure

#### B. Vanilla PyTorch (Full control)

```bash
# Basic PyTorch training
python scripts/train.py \
  --model resnet50 \
  --num-classes 10 \
  --data-dir ./data \
  --dataset cifar10 \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.001

# PyTorch with WandB
python scripts/train_with_wandb.py \
  --model resnet50 \
  --num-classes 10 \
  --data-dir ./data \
  --dataset cifar10 \
  --wandb-project vision-experiments \
  --wandb-name resnet50-cifar10
```

**Both frameworks log:**
- Training/validation metrics (loss, accuracy)
- Learning rate schedule
- Model architecture
- Confusion matrices (WandB)
- Hyperparameters

### 5. Evaluate Model

```bash
python scripts/evaluate.py \
  --checkpoint ./checkpoints/best_model.pth \
  --data-dir ./data \
  --dataset cifar10
```

## Model Experimentation

### Available Models

**Quick reference** (see [references/model_architectures.md](references/model_architectures.md) for complete guide):

| Model | Best For | Speed | Accuracy |
|-------|----------|-------|----------|
| resnet18/34/50/101 | General purpose, baseline | Fast | Good |
| efficientnet_b0-b4 | Efficient deployment | Medium | Excellent |
| vit_b_16/b_32 | Large datasets, SOTA | Slow | Best |
| swin_t/s/b | Hierarchical, multi-scale | Medium | Excellent |
| convnext_tiny/small/base | Modern CNN, efficient | Fast | Excellent |
| mobilenet_v2/v3 | Mobile, real-time | Very Fast | Moderate |
| densenet121/161 | Feature reuse, medical | Medium | Good |
| dit_base, layoutlmv3 | Document understanding | Medium | N/A |

### Experiment Workflow

**Try different models easily:**

```bash
# With Lightning (recommended)
python scripts/train_lightning_wandb.py --model resnet18 --wandb-name exp-resnet18
python scripts/train_lightning_wandb.py --model convnext_small --wandb-name exp-convnext
python scripts/train_lightning_wandb.py --model swin_s --wandb-name exp-swin

# With vanilla PyTorch
python scripts/train_with_wandb.py --model efficientnet_b0 --wandb-name exp-efficientnet
python scripts/train_with_wandb.py --model vit_b_16 --wandb-name exp-vit
```

**Compare hyperparameters:**

```bash
# Experiment with learning rates
python scripts/train_with_wandb.py --lr 0.1 --wandb-name lr-0.1
python scripts/train_with_wandb.py --lr 0.01 --wandb-name lr-0.01
python scripts/train_with_wandb.py --lr 0.001 --wandb-name lr-0.001

# Experiment with optimizers
python scripts/train_with_wandb.py --optimizer sgd --wandb-name opt-sgd
python scripts/train_with_wandb.py --optimizer adam --wandb-name opt-adam
python scripts/train_with_wandb.py --optimizer adamw --wandb-name opt-adamw
```

View all experiments in WandB dashboard with comparison charts.

## Multi-Environment Support

### Local GPU (CUDA)

Automatically detected. Use `--device cuda` or `--device auto`.

**Optimize for NVIDIA GPUs:**
- Maximize batch size until GPU memory 80-90% full
- Use all GPUs: Will auto-detect and use DataParallel
- Enable mixed precision for faster training (future feature)

### Apple Silicon (M1/M2/M3/M4)

Automatically detected. Use `--device mps` or `--device auto`.

**Setup:**
```bash
# Detect your chip and get recommendations
python scripts/setup_environment.py

# Outputs:
#   Apple Chip: M3
#   Recommended Batch Size: 48
```

**Optimization by chip:**

| Chip | Recommended Batch Size | Recommended Models |
|------|------------------------|-------------------|
| M1/M2 | 16-32 | resnet18, mobilenet_v3, efficientnet_b0 |
| M3/M3 Max | 32-48 | resnet50, convnext_small, swin_t |
| M3 Ultra/M4 | 48-64 | resnet101, convnext_base, swin_b |

**Training example:**
```bash
# M3 optimized training
python scripts/train_with_wandb.py \
  --model convnext_small \
  --batch-size 48 \
  --device mps \
  --wandb-project m3-experiments
```

**Performance notes:**
- M3/M4 GPUs are 5-12x faster than CPU
- Unified memory allows larger models than discrete GPUs
- Monitor Activity Monitor → GPU History for utilization
- See [references/environment_setup.md](references/environment_setup.md) for detailed optimization

### Google Colab

**Setup:**

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone your code or install skill
!git clone https://github.com/your-repo.git
%cd your-repo

# Run setup
!python scripts/setup_environment.py
```

**Training:**
```bash
!python scripts/train_with_wandb.py \
  --data-dir /content/drive/MyDrive/data \
  --checkpoint-dir /content/drive/MyDrive/checkpoints \
  --wandb-project colab-experiments
```

Save checkpoints to Drive to survive session restarts.

### AWS SageMaker

**Use managed training:**

```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='scripts/train.py',
    source_dir='.',
    role=role,
    instance_type='ml.p3.2xlarge',  # 1x V100
    instance_count=1,
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'model': 'resnet50',
        'num-classes': 10,
        'epochs': 100,
        'batch-size': 64,
    }
)

estimator.fit({'training': 's3://bucket/data'})
```

See [references/environment_setup.md](references/environment_setup.md) for instance types and pricing.

### Google Cloud Platform

**Vertex AI training:**

```python
from google.cloud import aiplatform

job = aiplatform.CustomTrainingJob(
    display_name='vision-training',
    script_path='scripts/train.py',
    container_uri='gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13:latest',
    requirements=['torchvision', 'wandb'],
)

job.run(
    args=[
        '--model', 'resnet50',
        '--num-classes', '10',
        '--epochs', '100',
    ],
    replica_count=1,
    machine_type='n1-standard-8',
    accelerator_type='NVIDIA_TESLA_V100',
    accelerator_count=1,
)
```

## Typical Workflows

### Workflow 1: Quick Classification Baseline (Lightning)

```bash
# 1. Setup environment
python scripts/setup_environment.py

# 2. Download data
python scripts/download_dataset.py --dataset cifar10 --data-dir ./data

# 3. Train with Lightning (minimal code!)
python scripts/train_lightning_wandb.py \
  --model resnet18 \
  --num-classes 10 \
  --data-dir ./data \
  --dataset cifar10 \
  --epochs 50 \
  --precision 16 \
  --wandb-project quick-baseline
```

**Vanilla PyTorch alternative:**
```bash
python scripts/train_with_wandb.py \
  --model resnet18 \
  --num-classes 10 \
  --data-dir ./data \
  --dataset cifar10 \
  --epochs 50 \
  --wandb-project quick-baseline
```

### Workflow 2: Model Comparison Study (Lightning)

```bash
# Train multiple models with Lightning
models=("resnet50" "convnext_small" "swin_s" "efficientnet_b2")

for model in "${models[@]}"; do
  python scripts/train_lightning_wandb.py \
    --model $model \
    --num-classes 10 \
    --data-dir ./data \
    --dataset cifar10 \
    --wandb-project model-comparison \
    --wandb-name $model \
    --wandb-tags comparison study-1 \
    --precision 16 \
    --early-stopping
done
```

Compare in WandB dashboard with parallel coordinates plots and metric comparisons.

### Workflow 3: Hyperparameter Tuning

**Option 1: Manual sweeps**

```bash
for lr in 0.1 0.01 0.001; do
  for bs in 32 64 128; do
    python scripts/train_with_wandb.py \
      --lr $lr \
      --batch-size $bs \
      --wandb-name lr-${lr}-bs-${bs}
  done
done
```

**Option 2: WandB Sweeps (recommended)**

Create `sweep_config.yaml`:
```yaml
program: scripts/train_with_wandb.py
method: bayes
metric:
  name: val/acc
  goal: maximize
parameters:
  lr:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.1
  batch_size:
    values: [32, 64, 128]
  model:
    values: [resnet18, resnet50, efficientnet_b0]
```

Run sweep:
```bash
wandb sweep sweep_config.yaml
wandb agent <sweep-id>
```

### Workflow 4: Custom Dataset

For ImageFolder-compatible datasets:

```
data/
├── train/
│   ├── class1/
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── class2/
└── val/
    ├── class1/
    └── class2/
```

Then:
```bash
python scripts/train_with_wandb.py \
  --model resnet50 \
  --num-classes <NUM_CLASSES> \
  --data-dir ./data \
  --dataset folder
```

Note: Current scripts have simplified dataset loading. For custom datasets, modify `train.py` or `train_with_wandb.py` to add `ImageFolder` loading.

## Advanced Configuration

### Using Config Files

Copy template:
```bash
cp assets/project_template/config.yaml my_config.yaml
```

Edit `my_config.yaml` with your settings, then (future feature):
```bash
python scripts/train.py --config my_config.yaml
```

### Model Selection Guide

**Quick decision tree:**

1. **Need fast inference?** → mobilenet_v3, convnext_tiny, efficientnet_b0
2. **Need high accuracy?** → convnext_base, swin_b, efficientnet_b3, vit_b_16
3. **Large dataset (>100k)?** → swin_b, vit_b_16, convnext_large
4. **Small dataset (<10k)?** → resnet18, convnext_tiny with pretrained weights
5. **Limited GPU memory?** → mobilenet_v2, resnet18, efficientnet_b0
6. **Medical/scientific imaging?** → densenet121, swin_b, resnet50
7. **Multi-scale/hierarchical tasks?** → swin_s, swin_b, convnext_base
8. **Document understanding?** → DiT-base, LayoutLMv3 (requires HuggingFace)
9. **Using Apple M3/M4?** → convnext_small, swin_s, resnet50

See [references/model_architectures.md](references/model_architectures.md) for detailed recommendations.

### Environment-Specific Best Practices

See [references/environment_setup.md](references/environment_setup.md) for:
- GPU setup (CUDA, MPS)
- Cloud platform configuration
- Distributed training
- Cost optimization
- Troubleshooting

## Bundled Resources

**PyTorch Lightning Scripts (Recommended):**
- `lightning_module.py`: LightningModule with 25+ models
- `lightning_data.py`: DataModule for data loading
- `train_lightning.py`: Clean Lightning training
- `train_lightning_wandb.py`: Lightning + WandB integration

**Vanilla PyTorch Scripts:**
- `train.py`: Basic PyTorch training
- `train_with_wandb.py`: PyTorch + WandB integration
- `evaluate.py`: Model evaluation

**Utilities:**
- `download_dataset.py`: Download popular datasets
- `preprocess_data.py`: Configure augmentation
- `setup_environment.py`: Environment detection

**References:**
- `pytorch_lightning.md`: Lightning guide, comparison, best practices
- `model_architectures.md`: Complete model guide with selection criteria
- `environment_setup.md`: Multi-platform setup and optimization

**Assets:**
- `project_template/`: Starter files (requirements.txt, config.yaml, README template)

## Common Use Cases

**User request:** "Train a model to classify my images"
1. Run `setup_environment.py` to detect GPU
2. Run `download_dataset.py` or prepare custom data
3. Run `train_with_wandb.py` with resnet50 baseline
4. Iterate on model selection based on results

**User request:** "Compare ResNet50 vs EfficientNet"
1. Train both models with same config using `train_with_wandb.py`
2. Use different `--wandb-name` for each
3. Compare metrics in WandB dashboard

**User request:** "Set up training on AWS"
1. Read `references/environment_setup.md` for SageMaker setup
2. Create SageMaker estimator with `scripts/train.py`
3. Configure S3 data paths

**User request:** "My training is slow"
1. Check GPU utilization: `nvidia-smi`
2. Review `environment_setup.md` troubleshooting section
3. Try smaller model or larger batch size

## Tips

- **Use PyTorch Lightning** for cleaner code and automatic features (multi-GPU, mixed precision)
- **Always use pretrained weights** unless dataset is very different from ImageNet
- **Start with resnet50 or convnext_small** as baseline, then experiment
- **Use WandB** to track all experiments systematically
- **Enable mixed precision** with `--precision 16` for 2-3x faster training
- **Monitor GPU usage** to optimize batch size (target 80-90% utilization)
- **Use early stopping** to prevent overtraining and save compute
- **Save checkpoints frequently** especially on cloud platforms with time limits
- **Leverage cloud spot instances** for cost savings (70-80% cheaper)

### Quick Comparison: When to use what?

| Use Case | Framework | Script |
|----------|-----------|--------|
| Production training | Lightning | `train_lightning_wandb.py` |
| Multi-GPU training | Lightning | `train_lightning.py --devices 4` |
| Learning fundamentals | PyTorch | `train.py` |
| Quick experiments | Lightning | `train_lightning.py --fast-dev-run` |
| Custom training loop | PyTorch | `train.py` (modify as needed) |
