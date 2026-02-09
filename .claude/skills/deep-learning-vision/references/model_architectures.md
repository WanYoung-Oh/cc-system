# Model Architecture Guide

Quick reference for available model architectures and when to use them.

## ResNet Family

**Best for:** General-purpose image classification, transfer learning baseline

| Model | Params | ImageNet Top-1 | Speed | Use Case |
|-------|--------|----------------|-------|----------|
| resnet18 | 11.7M | 69.8% | Fast | Quick experiments, limited resources |
| resnet34 | 21.8M | 73.3% | Fast | Balanced speed/accuracy |
| resnet50 | 25.6M | 76.1% | Medium | Standard baseline |
| resnet101 | 44.5M | 77.4% | Slow | High accuracy needed |
| resnet152 | 60.2M | 78.3% | Slow | Maximum ResNet accuracy |

**Strengths:** Proven architecture, excellent transfer learning, skip connections prevent vanishing gradients

**Weaknesses:** Not the most parameter-efficient

**Recommended starting point:** resnet50 with pretrained weights

## EfficientNet Family

**Best for:** Resource-constrained environments, mobile deployment, maximum accuracy per parameter

| Model | Params | ImageNet Top-1 | Speed | Use Case |
|-------|--------|----------------|-------|----------|
| efficientnet_b0 | 5.3M | 77.7% | Fast | Mobile, edge devices |
| efficientnet_b1 | 7.8M | 79.8% | Fast | Balanced efficiency |
| efficientnet_b2 | 9.2M | 80.5% | Medium | Better accuracy, still efficient |
| efficientnet_b3 | 12M | 81.7% | Medium | High accuracy, reasonable size |
| efficientnet_b4 | 19M | 83.4% | Slow | Maximum accuracy |

**Strengths:** Best accuracy per parameter, compound scaling, excellent for deployment

**Weaknesses:** Slightly slower than ResNet at equivalent parameter count

**Recommended starting point:** efficientnet_b0 for deployment, b3 for accuracy

## Vision Transformer (ViT)

**Best for:** Large datasets, when compute is available, cutting-edge performance

| Model | Params | ImageNet Top-1 | Speed | Use Case |
|-------|--------|----------------|-------|----------|
| vit_b_16 | 86M | 81.1% | Slow | Standard ViT |
| vit_b_32 | 88M | 75.9% | Medium | Faster ViT variant |
| vit_l_16 | 304M | 82.6% | Very Slow | Maximum accuracy |

**Strengths:** State-of-the-art with sufficient data, attention mechanisms, scalable

**Weaknesses:** Requires large datasets or strong pretrained weights, memory intensive

**Recommended starting point:** vit_b_16 with pretrained weights

## MobileNet Family

**Best for:** Mobile deployment, real-time inference, minimal latency

| Model | Params | ImageNet Top-1 | Speed | Use Case |
|-------|--------|----------------|-------|----------|
| mobilenet_v2 | 3.5M | 71.9% | Very Fast | Legacy mobile apps |
| mobilenet_v3_small | 2.5M | 67.7% | Very Fast | Extreme efficiency |
| mobilenet_v3_large | 5.5M | 75.0% | Very Fast | Best MobileNet accuracy |

**Strengths:** Fastest inference, smallest models, designed for mobile

**Weaknesses:** Lower accuracy than larger models

**Recommended starting point:** mobilenet_v3_large

## DenseNet Family

**Best for:** Feature reuse, gradient flow, medical imaging

| Model | Params | ImageNet Top-1 | Speed | Use Case |
|-------|--------|----------------|-------|----------|
| densenet121 | 8.0M | 74.4% | Medium | Efficient DenseNet |
| densenet161 | 28.7M | 77.1% | Slow | Better accuracy |
| densenet169 | 14.1M | 75.6% | Medium | Balanced |

**Strengths:** Efficient feature reuse, strong gradient flow, fewer parameters

**Weaknesses:** Memory intensive during training, slower than ResNet

## Swin Transformer Family

**Best for:** Hierarchical vision tasks, high-resolution images, object detection

| Model | Params | ImageNet Top-1 | Speed | Use Case |
|-------|--------|----------------|-------|----------|
| swin_t | 28M | 81.5% | Medium | Efficient transformer |
| swin_s | 50M | 83.2% | Medium | Balanced performance |
| swin_b | 88M | 83.6% | Slow | High accuracy |

**Strengths:** Hierarchical architecture like CNNs, efficient attention (shifted windows), excellent for dense prediction tasks

**Weaknesses:** More complex than standard ViT, requires more tuning

**Recommended starting point:** swin_s for balanced performance, swin_b for maximum accuracy

**Note:** Swin excels at multi-scale tasks (object detection, segmentation) compared to standard ViT

## ConvNeXt Family

**Best for:** Modern CNN architecture, combines CNN efficiency with Transformer design principles

| Model | Params | ImageNet Top-1 | Speed | Use Case |
|-------|--------|----------------|-------|----------|
| convnext_tiny | 28M | 82.5% | Fast | Efficient modern CNN |
| convnext_small | 50M | 83.6% | Fast | Balanced |
| convnext_base | 89M | 84.1% | Medium | High accuracy |
| convnext_large | 198M | 84.6% | Slow | Maximum accuracy |

**Strengths:** CNN efficiency with Transformer-level accuracy, faster training than ViT, excellent transfer learning

**Weaknesses:** Larger models than ResNet for similar performance

**Recommended starting point:** convnext_small for efficiency, convnext_base for accuracy

**Note:** ConvNeXt modernizes ResNet design with Transformer insights, achieving SOTA CNN performance

## Document Understanding Models

### Document Image Transformer (DiT) and LayoutLMv3

**Best for:** Document analysis, OCR, document layout understanding, form extraction

These models are specialized for document understanding tasks and require additional libraries:

```bash
# Install HuggingFace transformers and timm
pip install transformers timm
```

**DiT (Document Image Transformer):**
- **Architecture:** ViT-based model pretrained on document images
- **Use case:** Document classification, layout analysis
- **Implementation:**
```python
from transformers import AutoModel, AutoImageProcessor

# Load DiT model
processor = AutoImageProcessor.from_pretrained("microsoft/dit-base")
model = AutoModel.from_pretrained("microsoft/dit-base")
```

**LayoutLMv3:**
- **Architecture:** Multimodal (text + vision + layout) transformer
- **Use case:** Document understanding with text, visual, and spatial information
- **Best for:** Invoice processing, form understanding, document QA
- **Implementation:**
```python
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification

processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForSequenceClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=num_classes
)
```

**Note:** These models are not included in the default training scripts as they require different preprocessing pipelines (OCR + layout). Consult HuggingFace documentation for fine-tuning guides.

**Key differences from standard vision models:**
- LayoutLMv3: Requires text input (OCR results) + image + bounding boxes
- DiT: Works on document images directly but pretrained on document-specific data
- Both excel at structured documents (invoices, forms, receipts) over natural images

## Selection Guide

### By Use Case

**Quick prototyping:**
- resnet18, mobilenet_v3_large, convnext_tiny

**Best accuracy (no constraints):**
- convnext_large, swin_b, vit_l_16, efficientnet_b4

**Production deployment:**
- efficientnet_b0, mobilenet_v3_large, convnext_tiny

**Transfer learning:**
- resnet50, efficientnet_b3, convnext_base

**Medical/scientific imaging:**
- densenet121, resnet50, swin_b

**Multi-scale/hierarchical tasks:**
- swin_s, swin_b, convnext_base

**Document understanding:**
- DiT-base, LayoutLMv3-base (requires HuggingFace)

### By Dataset Size

**Small (<10k images):**
- Use pretrained: resnet18, efficientnet_b0
- Strong augmentation required

**Medium (10k-100k images):**
- resnet50, efficientnet_b1/b2
- Pretrained recommended

**Large (>100k images):**
- Any model, vit_b_16 becomes competitive
- Consider training from scratch

### By Compute Budget

**Limited GPU memory (<8GB):**
- resnet18, mobilenet_v3, efficientnet_b0
- Reduce batch size

**Standard GPU (8-16GB):**
- resnet50, efficientnet_b2, densenet121
- Batch size 32-128

**High-end GPU (>16GB):**
- resnet101, efficientnet_b4, vit_b_16
- Batch size 128+

## Hyperparameter Recommendations

### Learning Rate by Model

| Model Family | Initial LR (SGD) | Initial LR (Adam) |
|--------------|------------------|-------------------|
| ResNet | 0.1 | 0.001 |
| EfficientNet | 0.05 | 0.0005 |
| ViT | 0.001 | 0.0001 |
| MobileNet | 0.1 | 0.001 |
| DenseNet | 0.1 | 0.001 |

### Optimizer Selection

**SGD with momentum:**
- Default choice for most vision tasks
- Better generalization
- Requires more epochs

**Adam/AdamW:**
- Faster convergence
- Good for ViT
- May overfit on small datasets

### Batch Size Guidelines

- **Too small (<8):** Unstable training, poor batch norm
- **Small (8-32):** Good for limited memory, may need LR adjustment
- **Medium (32-128):** Standard choice
- **Large (>128):** Requires learning rate scaling, more stable training

## Fine-tuning Strategies

### Full Fine-tuning
Unfreeze all layers, train entire model
- **When:** Dataset similar to pretraining, sufficient data (>10k)
- **LR:** 10x smaller than training from scratch

### Partial Fine-tuning
Freeze early layers, train later layers
- **When:** Limited data (<10k), dataset similar to pretraining
- **LR:** Standard fine-tuning LR for unfrozen layers

### Feature Extraction
Freeze all layers except final classifier
- **When:** Very limited data (<1k), quick baseline
- **LR:** Standard training LR for classifier

## Common Pitfalls

1. **Using wrong image size:** Models expect specific input sizes (check architecture)
2. **Not using pretrained weights:** Always use pretrained unless dataset is very different
3. **Learning rate too high:** Start conservative, especially for fine-tuning
4. **Insufficient augmentation:** Vision models benefit greatly from augmentation
5. **Wrong normalization:** Use ImageNet stats unless training from scratch
