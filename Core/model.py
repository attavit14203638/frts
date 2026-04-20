#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model definition and configuration for TCD segmentation models.
"""

import torch
import torch.nn as nn
import os # Added for os.path.exists
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, AutoConfig, SchedulerType as HfSchedulerType

from transformers.optimization import get_scheduler

from typing import Dict, Tuple, Optional, Union, List, Any
import logging
from enum import Enum

# Import from refactored modules
from config import Config
from utils import get_logger

# Import custom loss functions from Extended
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Extended'))
try:
    from loss_functions import create_loss_function, CombinedLoss
    CUSTOM_LOSSES_AVAILABLE = True
except ImportError:
    CUSTOM_LOSSES_AVAILABLE = False

# Import new architectures from Extended
try:
    from new_architectures import OneFormerWrapper, UperNetSwinWrapper, create_new_architecture, _align_logits_labels_for_loss
    NEW_ARCHITECTURES_AVAILABLE = True
except ImportError:
    NEW_ARCHITECTURES_AVAILABLE = False

# Setup module logger
logger = get_logger()

class SchedulerType(Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"

class _SimpleOutput:
    """Lightweight output container with .loss and .logits to mimic HF outputs."""
    def __init__(self, loss: Optional[torch.Tensor], logits: torch.Tensor) -> None:
        self.loss = loss
        self.logits = logits


class SegFormerWrapper(nn.Module):
    """
    Unified SegFormer wrapper with toggleable train-time upsampling.

    Wraps HuggingFace's SegformerForSemanticSegmentation:
    - Configurable train-time upsampling: when enabled, upsamples logits to input
      resolution for loss; when disabled, downscales labels to match decoder output (~H/4).
    - Support for custom loss functions
    - Support for custom loss types (dice, boundary, hausdorff, combined)
    - Always returns native decoder logits (~H/4 × W/4); pipeline handles final upsampling
    """
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        ignore_index: int,
        project_config: Config,
        id2label: Optional[Dict] = None,
        label2id: Optional[Dict] = None,
    ):
        super().__init__()
        self.project_config = project_config
        self.ignore_index = ignore_index

        # Gradient checkpointing support
        self.supports_gradient_checkpointing = True

        # Build HF config
        if id2label is None:
            id2label = {i: f"class_{i}" for i in range(num_classes)}
        if label2id is None:
            label2id = {v: k for k, v in id2label.items()}

        hf_config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            semantic_loss_ignore_index=ignore_index,
        )

        try:
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                model_name, config=hf_config, ignore_mismatched_sizes=True
            )
            logger.info(f"SegFormerWrapper: Loaded pre-trained '{model_name}'")
        except Exception as e:
            logger.warning(f"Could not load pre-trained weights for '{model_name}': {e}. Initializing from config.")
            self.net = SegformerForSemanticSegmentation(config=hf_config)

        # Expose HF config for downstream compatibility (id2label, label2id, etc.)
        self.config = hf_config

        # Initialize loss function based on config
        self.loss_type = project_config.get("loss_type", "cross_entropy")
        self.loss_params = project_config.get("loss_params", {})
        self._init_loss_function()

    def _init_loss_function(self):
        """Initialize the loss function based on configuration."""
        if CUSTOM_LOSSES_AVAILABLE and self.loss_type != "cross_entropy":
            logger.info(f"SegFormerWrapper: Using custom loss type: {self.loss_type}")
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

    def _set_gradient_checkpointing(self, enable=False, gradient_checkpointing_func=None):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.net, 'segformer') and hasattr(self.net.segformer, 'encoder'):
            self.net.segformer.encoder.gradient_checkpointing = enable

    # Delegate save/load to the inner HF model for checkpoint compatibility
    def state_dict(self, *args, **kwargs):
        return self.net.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.net.load_state_dict(state_dict, strict=strict)

    def save_pretrained(self, *args, **kwargs):
        return self.net.save_pretrained(*args, **kwargs)

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> _SimpleOutput:
        # Get logits from HF model without its internal loss computation
        outputs = self.net(
            pixel_values=pixel_values,
            labels=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        logits = outputs.logits  # native decoder resolution (~H/4 × W/4)

        loss = None
        if labels is not None:
            upsample = self.project_config.get("train_time_upsample", False)
            mode = self.project_config.get("interpolation_mode", "bilinear")
            align = self.project_config.get("interpolation_align_corners", False)
            logits_for_loss, labels_for_loss = _align_logits_labels_for_loss(
                logits, labels, upsample_logits=upsample, mode=mode, align_corners=align
            )

            # Compute loss using configured loss function
            if isinstance(self.loss_fn, CombinedLoss) if CUSTOM_LOSSES_AVAILABLE else False:
                loss, _ = self.loss_fn(logits_for_loss, labels_for_loss)
            else:
                if hasattr(self.loss_fn, 'weight') and self.loss_fn.weight is not None:
                    self.loss_fn.weight = self.loss_fn.weight.to(logits_for_loss.device)
                loss = self.loss_fn(logits_for_loss, labels_for_loss)

        return _SimpleOutput(loss=loss, logits=logits)


# --- Train-time Upsampling Helper Functions ---
# _align_logits_labels_for_loss is the canonical definition in new_architectures.py.
# It is imported above; a local fallback is defined here only if that import failed.
if not NEW_ARCHITECTURES_AVAILABLE:
    def _align_logits_labels_for_loss(
        logits: torch.Tensor,
        labels: torch.Tensor,
        upsample_logits: bool = True,
        mode: str = "bilinear",
        align_corners: Optional[bool] = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Align logits and labels to the same spatial resolution for loss computation."""
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


def _build_smp_deeplabv3(backbone: str, num_classes: int) -> nn.Module:
    try:
        import segmentation_models_pytorch as smp
    except Exception as e:
        raise RuntimeError(
            "segmentation-models-pytorch is required for architecture='deeplabv3'. Install with 'pip install segmentation-models-pytorch timm'"
        ) from e
    model = smp.DeepLabV3(
        encoder_name=backbone,
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
        upsampling=1,
    )
    return model




class SETRDecoder(nn.Module):
    """Simple decoder for SETR"""
    def __init__(self, embed_dim: int, num_classes: int, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, embed_dim // 4, 3, padding=1),
            nn.BatchNorm2d(embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 4, num_classes, 1)
        )
    
    def forward(self, x):
        # x shape: (B, L, D) where L = (H/patch_size) * (W/patch_size)
        B, L, D = x.shape
        
        # Calculate spatial dimensions
        spatial_dim = int(L ** 0.5)
        H_patches = W_patches = spatial_dim
        
        # Reshape to spatial format
        x = x.transpose(1, 2).contiguous()  # (B, D, L)
        x = x.view(B, D, H_patches, W_patches)  # (B, D, H_patches, W_patches)
        
        # Apply decoder
        output = self.decoder(x)  # (B, num_classes, H_patches, W_patches)
        
        return output


def _build_simple_setr(num_classes: int, embed_dim: int = 768, patch_size: int = 16, input_size: int = 1024) -> nn.Module:
    """Build a simple SETR model using timm's Vision Transformer with configurable input size"""
    try:
        import timm
    except Exception as e:
        raise RuntimeError(
            "timm is required for architecture='setr'. Install with 'pip install timm'"
        ) from e
    
    # Use a pre-trained Vision Transformer backbone with flexible input size
    # The augreg variant has better pre-training and supports input size override
    try:
        backbone = timm.create_model(
            'vit_base_patch16_224.augreg_in21k_ft_in1k',
            pretrained=True,
            num_classes=0,
            img_size=input_size,  # Override default 224 to support native resolution
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        logger.info(f"Created SETR backbone with native {input_size}×{input_size} support using augreg variant")
    except Exception as e:
        logger.warning(f"Failed to load augreg variant: {e}. Falling back to standard vit_base_patch16_224")
        # Fallback to standard model with input size override
        backbone = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=0,
            img_size=input_size,  # Override default 224
            patch_size=patch_size,
            embed_dim=embed_dim
        )
        logger.info(f"Created SETR backbone with native {input_size}×{input_size} support using standard variant")
    
    decoder = SETRDecoder(embed_dim=embed_dim, num_classes=num_classes, patch_size=patch_size)
    
    class SETRModel(nn.Module):
        def __init__(self, backbone, decoder):
            super().__init__()
            self.backbone = backbone
            self.decoder = decoder
            
        def forward(self, x):
            # Extract features using ViT backbone at native resolution
            features = self.backbone.forward_features(x)  # (B, L, D)
            
            # Remove the CLS token for segmentation (first token is CLS)
            features = features[:, 1:]
            
            # Apply decoder
            output = self.decoder(features)
            return output
    
    return SETRModel(backbone, decoder)


class DeepLabV3Wrapper(nn.Module):
    def __init__(
        self,
        backbone: str,
        num_classes: int,
        ignore_index: int,
        project_config: Config
    ) -> None:
        super().__init__()
        self.net = _build_smp_deeplabv3(backbone=backbone, num_classes=num_classes)
        self.ignore_index = ignore_index
        self.project_config = project_config
        self.loss_type = project_config.get("loss_type", "cross_entropy")
        self.loss_params = project_config.get("loss_params", {})
        self._init_loss_function()

    def _init_loss_function(self):
        """Initialize the loss function based on configuration."""
        if CUSTOM_LOSSES_AVAILABLE and self.loss_type != "cross_entropy":
            logger.info(f"DeepLabV3Wrapper: Using custom loss type: {self.loss_type}")
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

    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.LongTensor] = None) -> _SimpleOutput:
        logits = self.net(pixel_values)
        loss = None
        if labels is not None:
            upsample = self.project_config.get("train_time_upsample", False)
            mode = self.project_config.get("interpolation_mode", "bilinear")
            align = self.project_config.get("interpolation_align_corners", False)
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


class SETRWrapper(nn.Module):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int,
        project_config: Config,
        embed_dim: int = 768,
        patch_size: int = 16
    ) -> None:
        super().__init__()
        # Get input size from config, default to 1024 for native high-resolution processing
        input_size = project_config.get("setr_input_size", 1024)
        self.net = _build_simple_setr(
            num_classes=num_classes, 
            embed_dim=embed_dim, 
            patch_size=patch_size,
            input_size=input_size
        )
        self.ignore_index = ignore_index
        self.project_config = project_config
        self.input_size = input_size
        self.loss_type = project_config.get("loss_type", "cross_entropy")
        self.loss_params = project_config.get("loss_params", {})
        self._init_loss_function()
        logger.info(f"SETRWrapper initialized for native {input_size}×{input_size} processing")

    def _init_loss_function(self):
        """Initialize the loss function based on configuration."""
        if CUSTOM_LOSSES_AVAILABLE and self.loss_type != "cross_entropy":
            logger.info(f"SETRWrapper: Using custom loss type: {self.loss_type}")
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

    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.LongTensor] = None) -> _SimpleOutput:
        # SETR decoder outputs at (H/patch_size × W/patch_size) resolution
        logits = self.net(pixel_values)

        loss = None
        if labels is not None:
            upsample = self.project_config.get("train_time_upsample", False)
            mode = self.project_config.get("interpolation_mode", "bilinear")
            align = self.project_config.get("interpolation_align_corners", False)
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

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: Union[str, SchedulerType], 
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5, 
    power: float = 1.0,      
    patience: int = 3,       
    factor: float = 0.1,     
    min_lr_val: float = 0,
    config_obj: Optional[Config] = None
) -> Union[torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau, None]:
    
    current_scheduler_type_enum: Optional[SchedulerType] = None
    hf_scheduler_type_enum: Optional[HfSchedulerType] = None 

    if isinstance(scheduler_type, str):
        try:
            current_scheduler_type_enum = SchedulerType(scheduler_type.lower())
        except ValueError:
            logger.info(f"Scheduler type '{scheduler_type}' not in custom SchedulerType enum. Trying Hugging Face types.")
            try:
                hf_scheduler_type_enum = HfSchedulerType(scheduler_type.lower())
            except ValueError:
                logger.error(f"Unknown scheduler type string: '{scheduler_type}'. Must be one of {[e.value for e in SchedulerType]} or {[e.value for e in HfSchedulerType]}. Defaulting to Hugging Face linear.")
                # Default to Hugging Face linear if string is not recognized by either enum
                hf_scheduler_type_enum = HfSchedulerType.LINEAR 
    elif isinstance(scheduler_type, SchedulerType): 
        current_scheduler_type_enum = scheduler_type
    elif isinstance(scheduler_type, HfSchedulerType): 
         hf_scheduler_type_enum = scheduler_type
    else:
        logger.error(f"Invalid scheduler_type type: {type(scheduler_type)}. Defaulting to Hugging Face linear.")
        hf_scheduler_type_enum = HfSchedulerType.LINEAR

    if current_scheduler_type_enum == SchedulerType.REDUCE_ON_PLATEAU:
        # For ReduceLROnPlateau, it is expected to monitor a metric where smaller is better (e.g., loss).
        # The actual metric value (e.g., validation_loss) is passed during scheduler.step() in the training loop.
        scheduler_mode = 'min'
        monitored_metric_for_log = "validation_loss" # Clarify in log what it's *intended* for.
        
        # The 'verbose' argument for ReduceLROnPlateau is deprecated in PyTorch 1.10+ 
        # and will be removed in 2.0. PyTorch issues a warning if used.
        # It's better to rely on standard logging for LR updates if using newer PyTorch.
        # If compatibility with PyTorch <1.10 is critical, one might add it conditionally:
        # verbose_kwarg = {'verbose': True} if torch.__version__ < '1.10.0' else {}
        # and then **verbose_kwarg in the constructor call.

        logger.info(f"Creating ReduceLROnPlateau scheduler: patience={patience}, factor={factor}, min_lr={min_lr_val}, mode='{scheduler_mode}' (intended to monitor '{monitored_metric_for_log}')")
        
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=factor,
            patience=patience,
            threshold=0.0001,
            threshold_mode='rel',
            cooldown=0,
            min_lr=min_lr_val, 
            eps=1e-8 
        )
    
    # Determine the scheduler name for Hugging Face's get_scheduler
    scheduler_name_for_hf: Union[str, HfSchedulerType]
    if hf_scheduler_type_enum: # Prioritize if it was directly an HfSchedulerType or resolved to one
        scheduler_name_for_hf = hf_scheduler_type_enum
    elif current_scheduler_type_enum: # If it was our custom enum, try to map its value
        try:
            scheduler_name_for_hf = HfSchedulerType(current_scheduler_type_enum.value)
        except ValueError:
            logger.error(f"Custom scheduler type '{current_scheduler_type_enum.value}' not directly mappable to Hugging Face schedulers. Defaulting to Hugging Face linear.")
            scheduler_name_for_hf = HfSchedulerType.LINEAR
    else: # Should not happen if logic above is correct, but as a fallback
        logger.error("Scheduler type determination failed unexpectedly. Defaulting to Hugging Face linear.")
        scheduler_name_for_hf = HfSchedulerType.LINEAR

    # Log warnings if specific parameters might not be used by the chosen scheduler type in transformers
    # Ensure scheduler_name_for_hf is a string for comparison if it's an enum
    scheduler_name_str = scheduler_name_for_hf.value if isinstance(scheduler_name_for_hf, Enum) else str(scheduler_name_for_hf)

    if scheduler_name_str == "cosine_with_restarts" and num_cycles != 0.5: # HfSchedulerType.COSINE_WITH_RESTARTS.value
         logger.warning(f"num_cycles={num_cycles} provided, but the standard transformers get_scheduler might use a default value for cosine_with_restarts.")
    if scheduler_name_str == "polynomial" and power != 1.0: # HfSchedulerType.POLYNOMIAL.value
         logger.warning(f"power={power} provided, but the standard transformers get_scheduler might use a default value for polynomial.")

    try:
        scheduler = get_scheduler(
            name=scheduler_name_for_hf, 
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        log_name = scheduler_name_for_hf.value if isinstance(scheduler_name_for_hf, Enum) else str(scheduler_name_for_hf)
        logger.info(f"Created Hugging Face scheduler: type={log_name}, warmup_steps={num_warmup_steps}, total_steps={num_training_steps}")
        return scheduler
    except Exception as e:
        logger.error(f"Failed to create Hugging Face scheduler '{scheduler_name_for_hf}': {e}. Falling back to Hugging Face Linear.")
        return get_scheduler(
            name=HfSchedulerType.LINEAR, 
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    return None # Should be unreachable if defaults work

def create_model(
    config: Config,
    num_training_steps: int,
    logger: Optional[logging.Logger] = None
) -> Tuple[nn.Module, torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    if logger is None: # Fallback logger if not provided
        logger = get_logger()

    logger.info(f"--- Model Creation ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device} for model creation and potential checkpoint loading.")

    architecture = config.get("architecture", "segformer")
    model_name = config.get("model_name")
    backbone = config.get("backbone", "resnet34")
    
    # Build architecture-specific identifier for logging
    if architecture == "segformer":
        model_identifier = model_name
        logger.info(f"Attempting to load architecture: {architecture} (model_name={model_name})")
    elif architecture == "deeplabv3":
        model_identifier = backbone
        logger.info(f"Attempting to load architecture: {architecture} (backbone={backbone})")
    elif architecture == "setr":
        model_identifier = f"SETR (embed_dim={config.get('setr_embed_dim', 768)}, patch_size={config.get('setr_patch_size', 16)})"
        logger.info(f"Attempting to load architecture: {architecture} (embed_dim={config.get('setr_embed_dim', 768)}, patch_size={config.get('setr_patch_size', 16)})")
    elif architecture == "oneformer":
        model_identifier = config.get("oneformer_model", "shi-labs/oneformer_ade20k_swin_large")
        logger.info(f"Attempting to load architecture: {architecture} (model={model_identifier})")
    elif architecture == "upernet_swin":
        model_identifier = config.get("upernet_swin_model", "openmmlab/upernet-swin-base")
        logger.info(f"Attempting to load architecture: {architecture} (model={model_identifier})")
    else:
        model_identifier = architecture
        logger.info(f"Attempting to load architecture: {architecture}")

    learning_rate = config["learning_rate"]
    weight_decay = config["weight_decay"]
    id2label = config["id2label"]
    label2id = config["label2id"]
    num_labels = len(id2label)

    if architecture == "segformer":
        logger.info(f"Initializing SegFormerWrapper for '{model_name}'")
        model = SegFormerWrapper(
            model_name=model_name,
            num_classes=num_labels,
            ignore_index=config.get("ignore_index", 255),
            project_config=config,
            id2label=id2label,
            label2id=label2id,
        )
    elif architecture == "deeplabv3":
        logger.info(f"Initializing DeepLabV3 (backbone={backbone})")
        model = DeepLabV3Wrapper(
            backbone=backbone,
            num_classes=num_labels,
            ignore_index=config.get("ignore_index", 255),
            project_config=config
        )
    elif architecture == "setr":
        embed_dim = config.get("setr_embed_dim", 768)
        patch_size = config.get("setr_patch_size", 16)
        logger.info(f"Initializing SETR (embed_dim={embed_dim}, patch_size={patch_size})")
        model = SETRWrapper(
            num_classes=num_labels,
            ignore_index=config.get("ignore_index", 255),
            project_config=config,
            embed_dim=embed_dim,
            patch_size=patch_size
        )
    elif architecture == "oneformer":
        if not NEW_ARCHITECTURES_AVAILABLE:
            raise RuntimeError(
                "New architectures module not available. Ensure Extended/new_architectures.py exists "
                "and transformers>=4.30.0 is installed."
            )
        oneformer_model = config.get("oneformer_model", "shi-labs/oneformer_ade20k_swin_large")
        logger.info(f"Initializing OneFormer (model={oneformer_model})")
        model = OneFormerWrapper(
            model_name=oneformer_model,
            num_classes=num_labels,
            ignore_index=config.get("ignore_index", 255),
            project_config=config,
            id2label=id2label,
            label2id=label2id,
        )
    elif architecture == "upernet_swin":
        if not NEW_ARCHITECTURES_AVAILABLE:
            raise RuntimeError(
                "New architectures module not available. Ensure Extended/new_architectures.py exists "
                "and transformers>=4.30.0 is installed."
            )
        upernet_swin_model = config.get("upernet_swin_model", "openmmlab/upernet-swin-base")
        logger.info(f"Initializing UperNet+Swin (model={upernet_swin_model})")
        model = UperNetSwinWrapper(
            model_name=upernet_swin_model,
            num_classes=num_labels,
            ignore_index=config.get("ignore_index", 255),
            project_config=config,
            id2label=id2label,
            label2id=label2id,
        )
    else:
        raise ValueError(f"Unsupported architecture: {architecture}. Supported: segformer, deeplabv3, setr, oneformer, upernet_swin")

    model.to(device) # Move model to device before optimizer creation and checkpoint loading

    # --- Checkpoint Loading (Model Weights) ---
    if config.get("resume_from_checkpoint", False):
        model_weights_path = config.get("resume_from_checkpoint_path")
        if model_weights_path and os.path.exists(model_weights_path):
            logger.info(f"Attempting to load model weights from: {model_weights_path}")
            try:
                state_dict = torch.load(model_weights_path, map_location=device)
                # Handle potential nested state_dict (e.g. if saved with 'model_state_dict' key)
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                elif 'state_dict' in state_dict: # another common key
                    state_dict = state_dict['state_dict']
                
                model.load_state_dict(state_dict, strict=True) # strict=True for fine-tuning same architecture
                logger.info(f"Successfully loaded model weights from {model_weights_path}.")
            except Exception as e:
                logger.error(f"Error loading model weights from {model_weights_path}: {e}. Model will use initial/pretrained weights.")
        else:
            logger.warning(f"resume_from_checkpoint is True, but path '{model_weights_path}' is invalid or not found. Model will use initial/pretrained weights.")
    else:
        logger.info(f"Not resuming model weights from checkpoint. Model uses initial/pretrained weights for {model_identifier}.")


    optimizer = create_optimizer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=weight_decay
    )
    logger.info(f"Optimizer created with initial LR: {learning_rate}, Weight Decay: {weight_decay}")

    # --- Optimizer State Loading ---
    if config.get("resume_from_checkpoint", False):
        optimizer_state_path = config.get("resume_optimizer_path") # Get path from config
        if optimizer_state_path and os.path.exists(optimizer_state_path): # Check if path is not None/empty and exists
            logger.info(f"Attempting to load optimizer state from: {optimizer_state_path}")
            try:
                optimizer.load_state_dict(torch.load(optimizer_state_path, map_location=device))
                logger.info(f"Successfully loaded optimizer state from {optimizer_state_path}.")
                
                # CRITICAL: Apply the new learning rate from the fine-tuning config
                logger.info(f"Applying new learning rate ({config.get('learning_rate', 'KEY_NOT_FOUND')}) to all parameter groups in the loaded optimizer.")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config["learning_rate"]
                logger.info(f"New learning rate {config.get('learning_rate', 'KEY_NOT_FOUND')} applied to optimizer.")
            except Exception as e:
                logger.error(f"Failed to load optimizer state from {optimizer_state_path}: {e}. Optimizer will use initial state with new LR.")
        else:
            logger.warning(f"Optimizer state file not found at {optimizer_state_path}. Optimizer will use initial state with new LR: {config.get('learning_rate', 'KEY_NOT_FOUND')}.")


    # --- Scheduler Instantiation (New Scheduler) ---
    scheduler = None
    scheduler_type_name = config.get("scheduler_type", "linear").lower()
    logger.info(f"Attempting to create scheduler of type: '{scheduler_type_name}'.")

    if scheduler_type_name == "reduce_on_plateau":
        scheduler_monitor_metric = config.get("scheduler_monitor", "val_loss") # Default to val_loss if not specified
        scheduler_mode = config.get("scheduler_mode", "min") # Default to min if not specified
        
        logger.info(f"Creating ReduceLROnPlateau scheduler: "
                    f"Monitoring '{scheduler_monitor_metric}' in '{scheduler_mode}' mode. "
                    f"Patience={config.get('scheduler_patience', 10)}, Factor={config.get('scheduler_factor', 0.1)}, MinLR={config.get('min_lr_scheduler', 0)}.")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_mode,
            factor=config.get('scheduler_factor', 0.1),
            patience=config.get('scheduler_patience', 10),
            min_lr=config.get('min_lr_scheduler', 0),
            verbose=True # Logs when LR changes
        )
    else:
        # For other schedulers, use the existing Hugging Face get_scheduler logic
        num_warmup_steps = int(config.get("warmup_ratio", 0.0) * num_training_steps) # Default warmup_ratio to 0.0 if not present
        logger.info(f"Using Hugging Face get_scheduler for '{scheduler_type_name}' with {num_warmup_steps} warmup steps for {num_training_steps} total training steps.")
        scheduler = create_scheduler( # Call the existing helper for HF schedulers
            optimizer=optimizer,
            scheduler_type=scheduler_type_name, # Pass the string name
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=config.get("num_cycles", 0.5),
            power=config.get("power", 1.0),
            # These are not directly used by HF get_scheduler but are passed to our create_scheduler helper
            patience=config.get("scheduler_patience", 3),
            factor=config.get("scheduler_factor", 0.1),
            min_lr_val=config.get("min_lr_scheduler", 0),
            config_obj=config
        )

    if config.get("gradient_checkpointing", False):
        logger.info("Configuration 'gradient_checkpointing' is True. Attempting to enable...")
        try:
            if hasattr(model, 'supports_gradient_checkpointing') and model.supports_gradient_checkpointing:
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
                    logger.info(f"Called model.gradient_checkpointing_enable() for {model.__class__.__name__}.")
                elif hasattr(model, '_set_gradient_checkpointing'): 
                     model._set_gradient_checkpointing(enable=True)
                     logger.info(f"Called model._set_gradient_checkpointing(enable=True) for {model.__class__.__name__}.")
                elif hasattr(model, 'segformer') and hasattr(model.segformer, 'encoder'): 
                    model.segformer.encoder.gradient_checkpointing = True
                    logger.info("Manually set model.segformer.encoder.gradient_checkpointing = True.")
                else:
                    logger.warning(f"Model {model.__class__.__name__} claims support but no known method to enable gradient checkpointing was found.")
            elif isinstance(model, SegformerForSemanticSegmentation):
                if hasattr(model, 'segformer') and hasattr(model.segformer, 'encoder'):
                    model.segformer.encoder.gradient_checkpointing = True
                    logger.info("Manually set model.segformer.encoder.gradient_checkpointing = True for SegFormer.")
                else:
                    logger.warning(f"SegFormer {model_identifier} does not have .segformer.encoder. Cannot enable gradient checkpointing.")
            else: 
                 logger.warning(f"Gradient checkpointing not configured for model type: {model.__class__.__name__}. Check model's `supports_gradient_checkpointing` or structure.")
        except Exception as e:
            logger.error(f"Error during gradient checkpointing setup: {e}")
    else:
        logger.info("Configuration 'gradient_checkpointing' is False. Skipping gradient checkpointing setup.")
    logger.info(f"--- Model Creation Complete ---")
    return model, optimizer, scheduler

def create_optimizer(
    model: nn.Module,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8
) -> torch.optim.Optimizer:
    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        betas=betas,
        eps=eps
    )
    
    return optimizer

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, module in model.named_modules():
        for param_name, _ in module.named_parameters():
            if param_name == "weight" and not any(isinstance(module, layer_type) for layer_type in forbidden_layer_types):
                if name:
                    result.append(f"{name}.{param_name}")
                else:
                    result.append(param_name)
    return result
