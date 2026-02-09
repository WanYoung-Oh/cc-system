# Environment Setup Guide

Guide for setting up training environments across different platforms.

## Local GPU Setup

### NVIDIA GPU (CUDA)

**Prerequisites:**
- NVIDIA GPU with CUDA support (compute capability >= 3.5)
- NVIDIA drivers installed
- CUDA Toolkit (11.8 or 12.1 recommended)

**Installation:**

```bash
# Check CUDA availability
nvidia-smi

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(torch.cuda.is_available())"
```

**Optimal settings:**
- Use all available GPUs with DataParallel or DistributedDataParallel
- Maximize batch size until GPU memory is 80-90% full
- Enable TF32 for faster training: `torch.backends.cuda.matmul.allow_tf32 = True`

### Apple Silicon (MPS) - M1/M2/M3/M4

**Prerequisites:**
- Apple Silicon Mac (M1, M2, M3, M4, or later)
- macOS 12.3+ (macOS 13+ recommended for M3/M4)

**Installation:**

```bash
# Install PyTorch with MPS support
pip install torch torchvision

# Verify MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"
python -c "import torch; print(torch.backends.mps.is_built())"
```

**Performance notes by chip:**
- **M1/M2:** 3-5x faster than CPU, good for medium models
- **M3/M3 Max:** 5-8x faster than CPU, GPU ray tracing cores improve performance
- **M3 Ultra/M4:** 8-12x faster than CPU, competitive with entry-level NVIDIA GPUs
- **Unified memory advantage:** Can handle larger batch sizes than discrete GPUs with same memory

**Optimization tips:**
```python
# Enable MPS
device = torch.device("mps")

# Optimize for Apple Silicon
torch.set_num_threads(8)  # Use all performance cores

# For M3+: Larger batch sizes benefit from improved GPU
# M1/M2: batch_size 16-32
# M3/M3 Max: batch_size 32-64
# M3 Ultra/M4: batch_size 64-128
```

**Known limitations:**
- Some operations fall back to CPU (slower)
- Mixed precision (FP16) support limited compared to CUDA
- No bfloat16 support yet
- Distributed training not supported

**Best practices for M3+:**
1. Use larger batch sizes than M1/M2 (improved GPU performance)
2. Monitor Activity Monitor → GPU History to ensure GPU utilization
3. Keep macOS updated for latest MPS improvements
4. Close other GPU-intensive apps (Final Cut, games) during training

**Recommended models for Apple Silicon:**
- **M1/M2:** resnet18, mobilenet_v3, efficientnet_b0
- **M3/M3 Max:** resnet50, convnext_small, swin_t, efficientnet_b2
- **M3 Ultra/M4:** resnet101, convnext_base, swin_b, vit_b_16

### CPU Only

**Installation:**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**Optimal settings:**
- Use smaller models (resnet18, mobilenet)
- Reduce batch size to 8-16
- Expect 10-50x slower than GPU

## Google Colab

**Advantages:**
- Free GPU/TPU access
- No setup required
- Jupyter notebook interface

**Setup:**

```python
# Check GPU allocation
!nvidia-smi

# Mount Google Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# Set data paths
DATA_DIR = '/content/drive/MyDrive/data'
CHECKPOINT_DIR = '/content/drive/MyDrive/checkpoints'
```

**Best practices:**
- Save checkpoints to Google Drive frequently
- Use Colab Pro for longer sessions (24h vs 12h)
- Monitor resource usage to avoid disconnection
- Upload large datasets to Drive beforehand

**Limitations:**
- Session timeout (12h free, 24h Pro)
- Idle timeout (~90 min)
- Daily usage limits on free tier

## Kaggle Notebooks

**Advantages:**
- Free GPU (30h/week) or TPU access
- Large public datasets available
- Persistent output (500MB)

**Setup:**

```python
# Check GPU
!nvidia-smi

# Data typically in /kaggle/input
DATA_DIR = '/kaggle/input/your-dataset'
CHECKPOINT_DIR = '/kaggle/working/checkpoints'

# Output saved to /kaggle/working (up to 500MB)
```

**Best practices:**
- Use Kaggle datasets for faster data loading
- Save checkpoints within 500MB limit
- Enable internet for pip installs

**Limitations:**
- 30h GPU/week (free tier)
- 9h max session time
- 500MB output limit

## AWS SageMaker

**Advantages:**
- Scalable compute
- Managed infrastructure
- Distributed training support
- S3 integration

**Setup:**

```python
# Use SageMaker PyTorch estimator
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point='train.py',
    role=role,
    instance_type='ml.p3.2xlarge',  # 1x V100 GPU
    instance_count=1,
    framework_version='2.0.0',
    py_version='py310',
    hyperparameters={
        'epochs': 100,
        'batch-size': 64,
        'lr': 0.001,
    }
)

# Train with S3 data
estimator.fit({'training': 's3://bucket/data'})
```

**Instance types:**

| Instance | GPUs | GPU Memory | vCPUs | RAM | $/hour |
|----------|------|------------|-------|-----|--------|
| ml.p3.2xlarge | 1x V100 | 16GB | 8 | 61GB | ~$3.06 |
| ml.p3.8xlarge | 4x V100 | 64GB | 32 | 244GB | ~$12.24 |
| ml.p4d.24xlarge | 8x A100 | 320GB | 96 | 1152GB | ~$32.77 |
| ml.g4dn.xlarge | 1x T4 | 16GB | 4 | 16GB | ~$0.71 |

**Best practices:**
- Use spot instances for 70% cost savings
- Store data in S3 in same region
- Use distributed training for multi-GPU instances
- Enable checkpointing for spot interruptions
- Use SageMaker Debugger for monitoring

**Data paths:**
- Input: `/opt/ml/input/data/training`
- Output: `/opt/ml/output`
- Checkpoints: `/opt/ml/checkpoints` (synced to S3)

## Google Cloud Platform (GCP)

**Advantages:**
- TPU access
- Flexible instance configuration
- Good for large-scale training
- GCS integration

**Setup (Vertex AI):**

```python
from google.cloud import aiplatform

job = aiplatform.CustomTrainingJob(
    display_name='vision-training',
    script_path='train.py',
    container_uri='gcr.io/cloud-aiplatform/training/pytorch-gpu.1-13:latest',
    requirements=['torchvision', 'wandb'],
)

job.run(
    replica_count=1,
    machine_type='n1-standard-8',
    accelerator_type='NVIDIA_TESLA_V100',
    accelerator_count=1,
)
```

**Machine types:**

| Machine | GPUs | GPU Memory | vCPUs | RAM | $/hour |
|---------|------|------------|-------|-----|--------|
| n1-standard-8 + V100 | 1x V100 | 16GB | 8 | 30GB | ~$2.48 |
| n1-standard-16 + V100 | 1x V100 | 16GB | 16 | 60GB | ~$3.36 |
| a2-highgpu-1g | 1x A100 | 40GB | 12 | 85GB | ~$3.67 |
| a2-highgpu-8g | 8x A100 | 320GB | 96 | 680GB | ~$26.64 |

**Best practices:**
- Use preemptible VMs for 80% cost savings
- Store data in GCS in same region
- Use GKE for distributed training
- Enable GPU monitoring

**Data paths:**
- Mount GCS bucket: `gcsfuse bucket-name /gcs/bucket`

## Distributed Training

### Single Node, Multiple GPUs

**DataParallel (simple but slower):**

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.to(device)
```

**DistributedDataParallel (recommended):**

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

# Initialize process group
dist.init_process_group(backend='nccl')
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)

# Wrap model
model = model.to(local_rank)
model = DistributedDataParallel(model, device_ids=[local_rank])

# Run with: torchrun --nproc_per_node=NUM_GPUS train.py
```

### Multiple Nodes

**AWS SageMaker:**
```python
estimator = PyTorch(
    instance_type='ml.p3.8xlarge',
    instance_count=4,  # 4 nodes = 16 GPUs
    distribution={'pytorchddp': {'enabled': True}}
)
```

**GCP Vertex AI:**
```python
job.run(
    replica_count=4,  # 4 nodes
    machine_type='n1-standard-16',
    accelerator_type='NVIDIA_TESLA_V100',
    accelerator_count=4,  # 4 GPUs per node
)
```

## Environment Detection

Use `scripts/setup_environment.py` to auto-detect environment:

```bash
python scripts/setup_environment.py --output config/environment.json
```

This generates a config with:
- Environment type (local, colab, sagemaker, gcp)
- GPU availability and count
- Recommended data paths
- Distributed training settings

## Cost Optimization

### Strategy 1: Spot/Preemptible Instances
- 70-80% cost savings
- Risk of interruption
- Enable checkpointing

### Strategy 2: Right-size Instances
- Start with smaller instances
- Profile GPU utilization
- Scale up if GPU underutilized

### Strategy 3: Efficient Training
- Use mixed precision training (cuts memory by 50%)
- Gradient accumulation (simulate larger batch sizes)
- Early stopping (avoid unnecessary epochs)

### Strategy 4: Data Pipeline Optimization
- Prefetch data to GPU
- Use faster data formats (LMDB, TFRecord vs raw images)
- Multi-worker data loading

## Monitoring

### GPU Utilization

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Log to file
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv -l 1 > gpu_utilization.log
```

### Training Metrics

**WandB (recommended):**
```python
import wandb
wandb.init(project='vision-training')
wandb.log({'loss': loss, 'accuracy': acc})
```

**TensorBoard:**
```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('logs')
writer.add_scalar('Loss/train', loss, epoch)
```

## Troubleshooting

### Out of Memory (OOM)

**Solutions:**
1. Reduce batch size
2. Use gradient accumulation
3. Enable mixed precision training
4. Use gradient checkpointing
5. Reduce model size

### Slow Training

**Check:**
1. GPU utilization (should be >80%)
2. Data loading (use multiple workers)
3. Move data augmentation to GPU
4. Enable cudnn.benchmark

### Environment Issues

**CUDA version mismatch:**
```bash
# Check installed CUDA
nvcc --version

# Reinstall PyTorch with matching CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Library conflicts:**
```bash
# Use virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```
