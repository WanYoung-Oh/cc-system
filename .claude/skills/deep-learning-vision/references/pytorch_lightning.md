## PyTorch Lightning Integration

PyTorch Lightning provides a high-level interface that reduces boilerplate code while maintaining flexibility. This skill includes both vanilla PyTorch and Lightning implementations.

### Why PyTorch Lightning?

**Benefits:**
- **Less boilerplate:** No manual training loops, device management, or distributed training setup
- **Better organized code:** Separate concerns (model, data, training logic)
- **Built-in best practices:** Automatic mixed precision, gradient clipping, checkpointing
- **Easy distributed training:** Multi-GPU/TPU with minimal code changes
- **Integrated logging:** TensorBoard, WandB, MLflow out of the box
- **Reproducibility:** Deterministic training, automatic seed setting

**When to use Lightning:**
- Complex training workflows
- Multi-GPU/distributed training
- Experimentation with many hyperparameters
- Production ML pipelines
- Teams with varying PyTorch experience

**When to use vanilla PyTorch:**
- Learning fundamentals
- Custom training loops with unique requirements
- Minimal dependencies preferred

### Architecture Overview

Lightning separates concerns into three components:

1. **LightningModule** (`lightning_module.py`) - Model + training logic
2. **LightningDataModule** (`lightning_data.py`) - Data loading + preprocessing
3. **Trainer** - Training orchestration, hardware management

```
┌─────────────────────┐
│  LightningModule    │  ← Model + training/val/test steps
└──────────┬──────────┘
           │
┌──────────┴──────────┐
│ LightningDataModule │  ← Data loading + transforms
└──────────┬──────────┘
           │
┌──────────┴──────────┐
│      Trainer        │  ← Training loop + hardware
└─────────────────────┘
```

## Quick Start

### Basic Training

```bash
python scripts/train_lightning.py \
  --model resnet50 \
  --num-classes 10 \
  --data-dir ./data \
  --dataset cifar10 \
  --epochs 100 \
  --batch-size 32 \
  --lr 0.001
```

### Training with WandB

```bash
python scripts/train_lightning_wandb.py \
  --model convnext_small \
  --num-classes 10 \
  --data-dir ./data \
  --dataset cifar10 \
  --wandb-project lightning-experiments \
  --wandb-name convnext-baseline \
  --epochs 100
```

## Key Features

### 1. Automatic Device Management

Lightning handles device placement automatically:

```python
# No need for manual .to(device)
model = VisionLightningModule(model_name='resnet50')
trainer = L.Trainer(accelerator='auto')  # Detects CUDA/MPS/CPU
trainer.fit(model, datamodule)
```

### 2. Multi-GPU Training

**Single node, multiple GPUs:**

```bash
# Automatic DataParallel
python scripts/train_lightning.py \
  --model resnet50 \
  --num-classes 10 \
  --data-dir ./data \
  --devices 4 \
  --accelerator gpu
```

**Distributed Data Parallel (recommended for multi-GPU):**

```bash
# Use DDP strategy for better performance
python scripts/train_lightning.py \
  --model resnet50 \
  --devices 4 \
  --accelerator gpu \
  --strategy ddp
```

### 3. Mixed Precision Training

```bash
# 16-bit mixed precision (faster, less memory)
python scripts/train_lightning.py \
  --model resnet50 \
  --precision 16 \
  --batch-size 64  # Can use larger batch size

# bfloat16 (if supported by hardware)
python scripts/train_lightning.py \
  --precision bf16
```

**Benefits:**
- 2-3x faster training
- 50% less memory usage
- Minimal accuracy loss

### 4. Gradient Accumulation

Simulate larger batch sizes with limited memory:

```bash
python scripts/train_lightning.py \
  --batch-size 16 \
  --accumulate-grad-batches 4  # Effective batch size: 16 * 4 = 64
```

### 5. Early Stopping

```bash
python scripts/train_lightning.py \
  --model resnet50 \
  --early-stopping \
  --patience 10  # Stop if no improvement for 10 epochs
```

### 6. Automatic Checkpointing

Lightning saves checkpoints automatically:

```bash
python scripts/train_lightning.py \
  --checkpoint-dir ./checkpoints \
  --save-top-k 3  # Save top 3 models
```

**Checkpoints include:**
- Model weights
- Optimizer state
- Hyperparameters
- Training epoch

**Resume training:**

```bash
python scripts/train_lightning.py \
  --resume-from-checkpoint ./checkpoints/model/epoch=50-val_acc=0.85.ckpt
```

### 7. Learning Rate Finder

Find optimal learning rate automatically:

```python
from lightning.pytorch.tuner import Tuner

trainer = L.Trainer(...)
tuner = Tuner(trainer)

# Find optimal LR
lr_finder = tuner.lr_find(model, datamodule)
print(f"Optimal LR: {lr_finder.suggestion()}")

# Plot LR vs loss
fig = lr_finder.plot(suggest=True)
fig.show()
```

### 8. WandB Integration

Lightning + WandB provides comprehensive experiment tracking:

```python
from lightning.pytorch.loggers import WandbLogger

wandb_logger = WandbLogger(
    project='my-project',
    name='experiment-1',
    log_model=True,  # Log checkpoints to WandB
)

trainer = L.Trainer(logger=wandb_logger)
```

**Automatic logging:**
- Training/validation metrics
- Learning rate schedule
- Model architecture
- System metrics (GPU/CPU usage)
- Checkpoints

## Advanced Usage

### Custom Callbacks

Create custom training behaviors:

```python
from lightning.pytorch.callbacks import Callback

class CustomCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # Custom logic at end of each epoch
        pass

trainer = L.Trainer(callbacks=[CustomCallback()])
```

### Profiling

Profile training to find bottlenecks:

```bash
python scripts/train_lightning.py \
  --profiler simple  # or 'advanced', 'pytorch'
```

### Fast Dev Run

Test code with 1 batch per epoch:

```bash
python scripts/train_lightning.py \
  --fast-dev-run
```

## Comparison: PyTorch vs Lightning

### Vanilla PyTorch (`train.py`, `train_with_wandb.py`)

**Pros:**
- Full control over training loop
- Easier to understand for beginners
- Minimal dependencies

**Cons:**
- More boilerplate code
- Manual device management
- Manual distributed training setup
- More code to maintain

**Use when:**
- Learning PyTorch fundamentals
- Unique training requirements
- Minimal dependencies needed

### PyTorch Lightning (`train_lightning.py`, `train_lightning_wandb.py`)

**Pros:**
- Clean, modular code
- Automatic device management
- Built-in distributed training
- Easy multi-GPU support
- Integrated logging
- Less code to write and maintain

**Cons:**
- Additional abstraction layer
- Slight learning curve
- Less control over low-level details

**Use when:**
- Production training pipelines
- Multi-GPU/distributed training
- Extensive experimentation
- Team collaboration

## Migration Guide

### From vanilla PyTorch to Lightning

**1. Convert model to LightningModule:**

```python
# Before (vanilla PyTorch)
model = models.resnet50(pretrained=True)
model = model.to(device)

# After (Lightning)
model = VisionLightningModule(
    model_name='resnet50',
    pretrained=True
)
# No .to(device) needed!
```

**2. Convert training loop:**

```python
# Before (vanilla PyTorch)
for epoch in range(epochs):
    for batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

# After (Lightning)
# All in training_step() in LightningModule
def training_step(self, batch, batch_idx):
    x, y = batch
    loss = self.criterion(self(x), y)
    return loss
# Lightning handles optimizer, backward, etc.
```

**3. Start training:**

```python
# Before (vanilla PyTorch)
# 50+ lines of training loop code

# After (Lightning)
trainer = L.Trainer(max_epochs=100)
trainer.fit(model, datamodule)
```

## Best Practices

### 1. Use DataModules

Encapsulate all data logic in `LightningDataModule`:

```python
# Good: All data logic in one place
datamodule = VisionDataModule(data_dir='./data', dataset='cifar10')
trainer.fit(model, datamodule)

# Avoid: Scattered data logic
```

### 2. Log Frequently

Log all important metrics:

```python
self.log('train/loss', loss, on_step=True, on_epoch=True)
self.log('train/acc', acc, on_step=True, on_epoch=True)
self.log('learning_rate', self.optimizers().param_groups[0]['lr'])
```

### 3. Use Validation

Always have a validation set to monitor overfitting:

```bash
python scripts/train_lightning.py \
  --val-split 0.1  # Hold out 10% for validation
```

### 4. Deterministic Training

For reproducibility:

```python
L.seed_everything(42)
trainer = L.Trainer(deterministic=True)
```

### 5. Monitor GPU Usage

Check GPU utilization during training:

```bash
# Terminal 1: Start training
python scripts/train_lightning.py ...

# Terminal 2: Monitor GPU
watch -n 1 nvidia-smi
```

Target 80-90% GPU utilization for optimal training speed.

## Troubleshooting

### OOM (Out of Memory)

**Solutions:**
1. Reduce batch size: `--batch-size 16`
2. Use mixed precision: `--precision 16`
3. Enable gradient accumulation: `--accumulate-grad-batches 2`
4. Use smaller model: `--model resnet18`

### Slow Training

**Check:**
1. GPU utilization (should be >80%)
2. Data loading: Increase `--num-workers`
3. Mixed precision enabled: `--precision 16`
4. Batch size: Try larger if memory allows

### WandB Not Logging

**Solutions:**
1. Login: `wandb login`
2. Check internet connection
3. Use offline mode: `--wandb-mode offline`
4. Sync later: `wandb sync ./wandb/offline-run-xxx`

## Resources

**Official Documentation:**
- PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/
- WandB + Lightning: https://docs.wandb.ai/guides/integrations/lightning

**Example Commands:**

```bash
# Quick baseline
python scripts/train_lightning.py \
  --model resnet18 \
  --num-classes 10 \
  --data-dir ./data \
  --dataset cifar10 \
  --epochs 50 \
  --fast-dev-run  # Test first!

# Production training
python scripts/train_lightning_wandb.py \
  --model convnext_base \
  --num-classes 100 \
  --data-dir ./data \
  --dataset cifar100 \
  --epochs 200 \
  --batch-size 64 \
  --precision 16 \
  --devices 4 \
  --accelerator gpu \
  --wandb-project production-models \
  --early-stopping \
  --patience 20
```
