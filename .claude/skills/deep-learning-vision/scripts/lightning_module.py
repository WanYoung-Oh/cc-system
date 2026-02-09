#!/usr/bin/env python3
"""
PyTorch Lightning module for vision models.
Encapsulates model, training, validation, and optimization logic.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

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
# Model Factory
# ============================================================================

AVAILABLE_MODELS = {
    # ResNet family
    'resnet18': lambda num_classes, pretrained: models.resnet18(weights='IMAGENET1K_V1' if pretrained else None),
    'resnet34': lambda num_classes, pretrained: models.resnet34(weights='IMAGENET1K_V1' if pretrained else None),
    'resnet50': lambda num_classes, pretrained: models.resnet50(weights='IMAGENET1K_V2' if pretrained else None),
    'resnet101': lambda num_classes, pretrained: models.resnet101(weights='IMAGENET1K_V2' if pretrained else None),
    'resnet152': lambda num_classes, pretrained: models.resnet152(weights='IMAGENET1K_V2' if pretrained else None),

    # EfficientNet family
    'efficientnet_b0': lambda num_classes, pretrained: models.efficientnet_b0(weights='IMAGENET1K_V1' if pretrained else None),
    'efficientnet_b1': lambda num_classes, pretrained: models.efficientnet_b1(weights='IMAGENET1K_V1' if pretrained else None),
    'efficientnet_b2': lambda num_classes, pretrained: models.efficientnet_b2(weights='IMAGENET1K_V1' if pretrained else None),
    'efficientnet_b3': lambda num_classes, pretrained: models.efficientnet_b3(weights='IMAGENET1K_V1' if pretrained else None),
    'efficientnet_b4': lambda num_classes, pretrained: models.efficientnet_b4(weights='IMAGENET1K_V1' if pretrained else None),

    # Vision Transformer
    'vit_b_16': lambda num_classes, pretrained: models.vit_b_16(weights='IMAGENET1K_V1' if pretrained else None),
    'vit_b_32': lambda num_classes, pretrained: models.vit_b_32(weights='IMAGENET1K_V1' if pretrained else None),
    'vit_l_16': lambda num_classes, pretrained: models.vit_l_16(weights='IMAGENET1K_V1' if pretrained else None),

    # Swin Transformer family
    'swin_t': lambda num_classes, pretrained: models.swin_t(weights='IMAGENET1K_V1' if pretrained else None),
    'swin_s': lambda num_classes, pretrained: models.swin_s(weights='IMAGENET1K_V1' if pretrained else None),
    'swin_b': lambda num_classes, pretrained: models.swin_b(weights='IMAGENET1K_V1' if pretrained else None),

    # ConvNeXt family
    'convnext_tiny': lambda num_classes, pretrained: models.convnext_tiny(weights='IMAGENET1K_V1' if pretrained else None),
    'convnext_small': lambda num_classes, pretrained: models.convnext_small(weights='IMAGENET1K_V1' if pretrained else None),
    'convnext_base': lambda num_classes, pretrained: models.convnext_base(weights='IMAGENET1K_V1' if pretrained else None),
    'convnext_large': lambda num_classes, pretrained: models.convnext_large(weights='IMAGENET1K_V1' if pretrained else None),

    # MobileNet family
    'mobilenet_v2': lambda num_classes, pretrained: models.mobilenet_v2(weights='IMAGENET1K_V2' if pretrained else None),
    'mobilenet_v3_small': lambda num_classes, pretrained: models.mobilenet_v3_small(weights='IMAGENET1K_V1' if pretrained else None),
    'mobilenet_v3_large': lambda num_classes, pretrained: models.mobilenet_v3_large(weights='IMAGENET1K_V2' if pretrained else None),

    # DenseNet family
    'densenet121': lambda num_classes, pretrained: models.densenet121(weights='IMAGENET1K_V1' if pretrained else None),
    'densenet161': lambda num_classes, pretrained: models.densenet161(weights='IMAGENET1K_V1' if pretrained else None),
    'densenet169': lambda num_classes, pretrained: models.densenet169(weights='IMAGENET1K_V1' if pretrained else None),

    # Other models
    'vgg16': lambda num_classes, pretrained: models.vgg16(weights='IMAGENET1K_V1' if pretrained else None),
    'vgg19': lambda num_classes, pretrained: models.vgg19(weights='IMAGENET1K_V1' if pretrained else None),
}


def modify_classifier(model, model_name, num_classes):
    """Modify the final classifier layer for custom number of classes"""

    if 'resnet' in model_name or 'resnext' in model_name:
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif 'efficientnet' in model_name:
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif 'vit' in model_name:
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    elif 'swin' in model_name:
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)

    elif 'convnext' in model_name:
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    elif 'mobilenet' in model_name:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    elif 'densenet' in model_name:
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    elif 'vgg' in model_name:
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)

    return model


# ============================================================================
# Lightning Module
# ============================================================================

class VisionLightningModule(L.LightningModule):
    """
    PyTorch Lightning module for vision classification tasks.

    Handles:
    - Model creation and initialization
    - Training and validation steps
    - Optimizer and scheduler configuration
    - Metric computation and logging
    """

    def __init__(
        self,
        model_name: str = 'resnet50',
        num_classes: int = 10,
        pretrained: bool = True,
        learning_rate: float = 0.001,
        optimizer: str = 'sgd',
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        scheduler: str = 'cosine',
        max_epochs: int = 100,
    ):
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Create model
        if model_name not in AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} not supported. Available: {list(AVAILABLE_MODELS.keys())}")

        self.model = AVAILABLE_MODELS[model_name](num_classes, pretrained)

        # Modify classifier for custom num_classes
        if num_classes != 1000:
            self.model = modify_classifier(self.model, model_name, num_classes)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = []
        self.val_acc = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """Training step"""
        images, labels = batch

        # Forward pass
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Compute accuracy
        _, predicted = outputs.max(1)
        acc = (predicted == labels).float().mean()

        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        images, labels = batch

        # Forward pass
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Compute accuracy
        _, predicted = outputs.max(1)
        acc = (predicted == labels).float().mean()

        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        """Test step"""
        images, labels = batch

        # Forward pass
        outputs = self(images)
        loss = self.criterion(outputs, labels)

        # Compute accuracy
        _, predicted = outputs.max(1)
        acc = (predicted == labels).float().mean()

        # Log metrics
        self.log('test/loss', loss)
        self.log('test/acc', acc)

        return {'test_loss': loss, 'test_acc': acc}

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""

        # Optimizer
        if self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=self.hparams.momentum,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        elif self.hparams.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay
            )
        else:
            raise ValueError(f"Optimizer {self.hparams.optimizer} not supported")

        # Scheduler
        if self.hparams.scheduler == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.hparams.max_epochs)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                }
            }
        elif self.hparams.scheduler == 'step':
            scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                }
            }
        elif self.hparams.scheduler == 'plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/acc',
                    'interval': 'epoch',
                }
            }
        else:
            return optimizer
