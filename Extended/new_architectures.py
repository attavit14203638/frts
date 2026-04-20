#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
New Architecture Wrappers for Journal Extension.

This module provides wrapper classes for OneFormer and UperNet+Swin architectures,
adapted to work with the training pipeline (train-time upsampling, custom losses).

Architectures:
- OneFormerWrapper: Task-conditioned query-based segmentation using HuggingFace's OneFormer
- UperNetSwinWrapper: UperNet with Swin Transformer backbone via HuggingFace
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Union
import logging

# Setup module logger
try:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Core'))
    from utils import get_logger
    logger = get_logger()
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Import custom loss functions
try:
    from loss_functions import create_loss_function, CombinedLoss
    CUSTOM_LOSSES_AVAILABLE = True
except ImportError:
    CUSTOM_LOSSES_AVAILABLE = False


class _SimpleOutput:
    """Lightweight output container with .loss and .logits to mimic HF outputs."""
    def __init__(self, loss: Optional[torch.Tensor], logits: torch.Tensor) -> None:
        self.loss = loss
        self.logits = logits


def _align_logits_labels_for_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    upsample_logits: bool = True,
    mode: str = "bilinear",
    align_corners: Optional[bool] = False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Align logits and labels to the same spatial resolution for loss computation.

    When upsample_logits=True (train-time upsampling enabled):
        Upsamples logits to match label resolution.
    When upsample_logits=False:
        Downscales labels to match logit resolution using nearest interpolation.
    """
    if logits.shape[-2:] == labels.shape[-2:]:
        return logits, labels
    if upsample_logits:
        align_param = align_corners if mode != "nearest" else None
        resized_logits = F.interpolate(logits, size=labels.shape[-2:], mode=mode, align_corners=align_param)
        return resized_logits, labels
    else:
        resized_labels = F.interpolate(
            labels.unsqueeze(1).float(), size=logits.shape[-2:], mode="nearest"
        ).squeeze(1).long()
        return logits, resized_labels


# =============================================================================
# OneFormer Wrapper
# =============================================================================

class OneFormerWrapper(nn.Module):
    """
    Wrapper for OneFormer for semantic segmentation.
    
    OneFormer is a task-conditioned query-based architecture that unifies
    semantic, instance, and panoptic segmentation with a single model.
    This wrapper adapts it for pixel-wise semantic segmentation by:
    1. Providing a "semantic" task token via OneFormerProcessor
    2. Converting query-based outputs to dense pixel predictions
    3. Applying train-time upsampling strategy
    
    Uses HuggingFace's transformers library implementation.
    """
    
    def __init__(
        self,
        model_name: str = "shi-labs/oneformer_ade20k_swin_large",
        num_classes: int = 2,
        ignore_index: int = 255,
        project_config: Any = None,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.project_config = project_config
        self.loss_type = project_config.get("loss_type", "cross_entropy") if project_config else "cross_entropy"
        self.loss_params = project_config.get("loss_params", {}) if project_config else {}
        
        # Build id2label and label2id if not provided
        if id2label is None:
            id2label = {i: f"class_{i}" for i in range(num_classes)}
        if label2id is None:
            label2id = {v: k for k, v in id2label.items()}
        
        try:
            from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor, AutoConfig
            
            # Load config and modify for our task
            config = AutoConfig.from_pretrained(model_name)
            config.num_labels = num_classes
            config.id2label = id2label
            config.label2id = label2id
            
            # Disable auxiliary losses to reduce memory usage
            # This saves ~15-20% GPU memory by only computing loss at final decoder layer
            use_aux_loss = project_config.get("oneformer_use_auxiliary_loss", False) if project_config else False
            config.use_auxiliary_loss = use_aux_loss
            config.output_auxiliary_logits = use_aux_loss
            if not use_aux_loss:
                logger.info("OneFormerWrapper: Auxiliary losses DISABLED for memory efficiency")
            
            # Load model
            self.model = OneFormerForUniversalSegmentation.from_pretrained(
                model_name,
                config=config,
                ignore_mismatched_sizes=True
            )
            
            # Processor wraps image processor + CLIP tokenizer for task tokens
            self.processor = OneFormerProcessor.from_pretrained(model_name)
            
            logger.info(f"OneFormerWrapper: Loaded '{model_name}' with {num_classes} classes")
            
        except ImportError as e:
            raise RuntimeError(
                "transformers library with OneFormer support is required. "
                "Install with: pip install transformers>=4.30.0"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load OneFormer model '{model_name}': {e}") from e
        
        # Pre-compute task inputs for "semantic" task (reused every forward pass)
        # The processor tokenizes the task string for the contrastive query head
        self._task_inputs = self.processor.tokenizer(
            ["the task is semantic"],
            padding="max_length",
            max_length=self.model.config.task_seq_len if hasattr(self.model.config, 'task_seq_len') else 77,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]  # (1, seq_len)
        
        # Gradient checkpointing support
        self.supports_gradient_checkpointing = True
        
        self._init_loss_function()
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the underlying model.
        
        Note: OneFormerForUniversalSegmentation doesn't natively support gradient checkpointing.
        This is a no-op that logs a warning.
        """
        logger.warning(
            "OneFormerWrapper: OneFormer does not natively support gradient checkpointing. "
            "To reduce memory, use a smaller batch size with gradient accumulation instead."
        )

    def _init_loss_function(self):
        """Initialize the loss function based on configuration."""
        if CUSTOM_LOSSES_AVAILABLE and self.loss_type != "cross_entropy":
            logger.info(f"OneFormerWrapper: Using custom loss type: {self.loss_type}")
            self.loss_fn = create_loss_function(
                loss_type=self.loss_type,
                loss_params=self.loss_params,
                ignore_index=self.ignore_index
            )
        else:
            if self.loss_type != "cross_entropy" and not CUSTOM_LOSSES_AVAILABLE:
                logger.warning(f"Custom losses not available, falling back to cross_entropy")
            self.loss_fn = nn.CrossEntropyLoss(
                ignore_index=self.ignore_index
            )
    
    def forward(
        self, 
        pixel_values: torch.Tensor, 
        labels: Optional[torch.LongTensor] = None
    ) -> _SimpleOutput:
        """
        Forward pass for OneFormer.
        
        Args:
            pixel_values: Input images (B, C, H, W)
            labels: Ground truth segmentation masks (B, H, W)
            
        Returns:
            _SimpleOutput with loss and logits
        """
        B = pixel_values.shape[0]
        device = pixel_values.device
        
        # Expand pre-computed task inputs to match batch size and move to device
        task_inputs = self._task_inputs.expand(B, -1).to(device)
        
        # Forward through OneFormer with task conditioning
        outputs = self.model(pixel_values=pixel_values, task_inputs=task_inputs)
        
        # Convert query-based outputs to dense predictions
        # Get mask predictions: (B, num_queries, H/4, W/4) typically
        mask_logits = outputs.masks_queries_logits  # (B, Q, H', W')
        class_logits = outputs.class_queries_logits  # (B, Q, num_classes+1)
        
        # Compute semantic segmentation by weighting masks by class probabilities
        # Remove the "no object" class (last class)
        class_probs = F.softmax(class_logits, dim=-1)[..., :-1]  # (B, Q, num_classes)
        
        # Always compute semantic logits at native reduced resolution (~H/4)
        _, Q, H, W = mask_logits.shape
        C = self.num_classes
        masks_flat = mask_logits.view(B, Q, -1)  # (B, Q, H*W)
        
        # Weighted combination: (B, Q, C).T @ (B, Q, H*W) -> (B, C, H*W)
        semantic_logits = torch.einsum('bqc,bqn->bcn', class_probs, masks_flat.sigmoid())
        semantic_logits = semantic_logits.view(B, C, H, W)
        
        # Cast to float32 for mixed precision safety (fp16 can flush small values to 0)
        semantic_logits = semantic_logits.float()
        
        loss = None
        if labels is not None:
            upsample = self.project_config.get("train_time_upsample", False) if self.project_config else False
            mode = self.project_config.get("interpolation_mode", "bilinear") if self.project_config else "bilinear"
            align = self.project_config.get("interpolation_align_corners", False) if self.project_config else False
            logits_for_loss, labels_for_loss = _align_logits_labels_for_loss(
                semantic_logits, labels, upsample_logits=upsample, mode=mode, align_corners=align
            )
            if isinstance(self.loss_fn, CombinedLoss) if CUSTOM_LOSSES_AVAILABLE else False:
                loss, _ = self.loss_fn(logits_for_loss, labels_for_loss)
            else:
                if hasattr(self.loss_fn, 'weight') and self.loss_fn.weight is not None:
                    self.loss_fn.weight = self.loss_fn.weight.to(logits_for_loss.device)
                loss = self.loss_fn(logits_for_loss, labels_for_loss)
        
        return _SimpleOutput(loss=loss, logits=semantic_logits)


# =============================================================================
# UperNet+Swin Wrapper
# =============================================================================

class UperNetSwinWrapper(nn.Module):
    """
    Wrapper for UperNet with Swin Transformer backbone (CNN-Transformer hybrid).
    
    Uses HuggingFace's UperNetForSemanticSegmentation with a Swin backbone.
    Outputs at ~H/4 resolution natively. Accepts arbitrary input sizes.
    Implements train-time upsampling strategy.
    """
    
    def __init__(
        self,
        model_name: str = "openmmlab/upernet-swin-base",
        num_classes: int = 2,
        ignore_index: int = 255,
        project_config: Any = None,
        id2label: Optional[Dict[int, str]] = None,
        label2id: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.project_config = project_config
        self.loss_type = project_config.get("loss_type", "cross_entropy") if project_config else "cross_entropy"
        self.loss_params = project_config.get("loss_params", {}) if project_config else {}
        
        # Build id2label and label2id if not provided
        if id2label is None:
            id2label = {i: f"class_{i}" for i in range(num_classes)}
        if label2id is None:
            label2id = {v: k for k, v in id2label.items()}
        
        try:
            from transformers import UperNetForSemanticSegmentation, AutoConfig
            
            # Load config and modify for our task
            config = AutoConfig.from_pretrained(model_name)
            config.num_labels = num_classes
            config.id2label = id2label
            config.label2id = label2id
            
            # Load model
            self.net = UperNetForSemanticSegmentation.from_pretrained(
                model_name,
                config=config,
                ignore_mismatched_sizes=True
            )
            
            logger.info(f"UperNetSwinWrapper: Loaded '{model_name}' with {num_classes} classes")
            
        except ImportError as e:
            raise RuntimeError(
                "transformers library with UperNet support is required. "
                "Install with: pip install transformers>=4.30.0"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load UperNet+Swin model '{model_name}': {e}") from e
        
        # Expose HF config for downstream compatibility (id2label, label2id, etc.)
        self.config = config
        
        # Gradient checkpointing support
        self.supports_gradient_checkpointing = True
        
        self._init_loss_function()
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Enable gradient checkpointing for the underlying model."""
        self.net.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def _init_loss_function(self):
        """Initialize the loss function based on configuration."""
        if CUSTOM_LOSSES_AVAILABLE and self.loss_type != "cross_entropy":
            logger.info(f"UperNetSwinWrapper: Using custom loss type: {self.loss_type}")
            self.loss_fn = create_loss_function(
                loss_type=self.loss_type,
                loss_params=self.loss_params,
                ignore_index=self.ignore_index
            )
        else:
            if self.loss_type != "cross_entropy" and not CUSTOM_LOSSES_AVAILABLE:
                logger.warning(f"Custom losses not available, falling back to cross_entropy")
            self.loss_fn = nn.CrossEntropyLoss(
                ignore_index=self.ignore_index
            )
    
    # Delegate save/load to the inner HF model for checkpoint compatibility
    def state_dict(self, *args, **kwargs):
        return self.net.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.net.load_state_dict(state_dict, strict=strict)

    def save_pretrained(self, *args, **kwargs):
        return self.net.save_pretrained(*args, **kwargs)
    
    def forward(
        self, 
        pixel_values: torch.Tensor, 
        labels: Optional[torch.LongTensor] = None
    ) -> _SimpleOutput:
        """
        Forward pass for UperNet+Swin.
        
        Args:
            pixel_values: Input images (B, C, H, W)
            labels: Ground truth segmentation masks (B, H, W)
            
        Returns:
            _SimpleOutput with loss and logits
        """
        # Get logits from HF model without its internal loss computation
        outputs = self.net(pixel_values=pixel_values, labels=None)
        logits = outputs.logits  # native decoder resolution (~H/4 × W/4)
        
        loss = None
        if labels is not None:
            upsample = self.project_config.get("train_time_upsample", False) if self.project_config else False
            mode = self.project_config.get("interpolation_mode", "bilinear") if self.project_config else "bilinear"
            align = self.project_config.get("interpolation_align_corners", False) if self.project_config else False
            logits_for_loss, labels_for_loss = _align_logits_labels_for_loss(
                logits, labels, upsample_logits=upsample, mode=mode, align_corners=align
            )
            if isinstance(self.loss_fn, CombinedLoss) if CUSTOM_LOSSES_AVAILABLE else False:
                loss, _ = self.loss_fn(logits_for_loss, labels_for_loss)
            else:
                if hasattr(self.loss_fn, 'weight') and self.loss_fn.weight is not None:
                    self.loss_fn.weight = self.loss_fn.weight.to(logits_for_loss.device)
                loss = self.loss_fn(logits_for_loss, labels_for_loss)
        
        return _SimpleOutput(loss=loss, logits=logits)


# =============================================================================
# Factory Function
# =============================================================================

def create_new_architecture(
    architecture: str,
    num_classes: int = 2,
    ignore_index: int = 255,
    project_config: Any = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create new architecture wrappers.
    
    Args:
        architecture: One of 'oneformer', 'upernet_swin'
        num_classes: Number of output classes
        ignore_index: Label index to ignore in loss computation
        project_config: Project configuration object
        **kwargs: Architecture-specific parameters
        
    Returns:
        Initialized model wrapper
        
    Raises:
        ValueError: If architecture is not supported
    """
    architecture = architecture.lower()
    
    if architecture == "oneformer":
        model_name = kwargs.get("model_name", "shi-labs/oneformer_ade20k_swin_large")
        id2label = kwargs.get("id2label", None)
        label2id = kwargs.get("label2id", None)
        
        return OneFormerWrapper(
            model_name=model_name,
            num_classes=num_classes,
            ignore_index=ignore_index,
            project_config=project_config,
            id2label=id2label,
            label2id=label2id,
        )
    
    elif architecture == "upernet_swin":
        model_name = kwargs.get("model_name", "openmmlab/upernet-swin-base")
        id2label = kwargs.get("id2label", None)
        label2id = kwargs.get("label2id", None)
        
        return UperNetSwinWrapper(
            model_name=model_name,
            num_classes=num_classes,
            ignore_index=ignore_index,
            project_config=project_config,
            id2label=id2label,
            label2id=label2id,
        )
    
    else:
        raise ValueError(
            f"Unsupported architecture: '{architecture}'. "
            f"Supported new architectures: 'oneformer', 'upernet_swin'"
        )


# =============================================================================
# Architecture Information
# =============================================================================

ARCHITECTURE_INFO = {
    "oneformer": {
        "description": "Task-conditioned query-based universal segmentation model (SHI-Labs)",
        "default_model": "shi-labs/oneformer_ade20k_swin_large",
        "output_resolution": "~H/4 × W/4",
        "pretrained_options": [
            "shi-labs/oneformer_ade20k_swin_tiny",
            "shi-labs/oneformer_ade20k_swin_large",
            "shi-labs/oneformer_ade20k_dinat_large",
            "shi-labs/oneformer_coco_swin_large",
        ],
        "notes": "Requires transformers>=4.30.0. Task-conditioned joint training for universal segmentation."
    },
    "upernet_swin": {
        "description": "UperNet with Swin Transformer backbone (CNN-Transformer hybrid)",
        "default_model": "openmmlab/upernet-swin-base",
        "output_resolution": "~H/4 × W/4",
        "pretrained_options": [
            "openmmlab/upernet-swin-tiny",
            "openmmlab/upernet-swin-small",
            "openmmlab/upernet-swin-base",
            "openmmlab/upernet-swin-large",
        ],
        "notes": "Requires transformers>=4.30.0. Accepts arbitrary input resolution."
    }
}


def get_architecture_info(architecture: str) -> Dict[str, Any]:
    """Get information about a supported architecture."""
    architecture = architecture.lower()
    if architecture in ARCHITECTURE_INFO:
        return ARCHITECTURE_INFO[architecture]
    raise ValueError(f"Unknown architecture: {architecture}")


def list_new_architectures() -> list:
    """List all supported new architectures."""
    return list(ARCHITECTURE_INFO.keys())
