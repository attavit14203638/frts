#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Checkpoint management utilities for TCD segmentation models.

This module provides functions for saving, loading, and managing model checkpoints
for all supported architectures (SegFormer, DeepLabV3, SETR, UperNet+Swin, OneFormer).
"""

import os
import json
import torch
import logging
import shutil
from typing import Dict, Any, Optional, Union, Tuple, List
from datetime import datetime
from types import SimpleNamespace

from exceptions import FileError, FileNotFoundError
from utils import get_logger

# Setup module logger
logger = get_logger()


def save_checkpoint(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    global_step: int,
    epoch: int,
    output_dir: str,
    max_to_keep: int = 3,
    metrics: Optional[Dict[str, float]] = None,
    metric_to_monitor: Optional[str] = None,
    monitor_mode: str = 'max',
    is_final: bool = False,
    config: Optional[Any] = None  # Add config parameter
):
    """
    Saves a checkpoint of the model, optimizer, and scheduler state.

    Args:
        model: The model to save.
        optimizer: The optimizer state to save.
        scheduler: The scheduler state to save.
        global_step: The current global training step.
        epoch: The current epoch number.
        output_dir: Directory where checkpoints will be saved.
        max_to_keep: Maximum number of recent checkpoints to keep.
        metrics: Optional dictionary of metrics from evaluation.
        metric_to_monitor: Optional metric key from 'metrics' to track for best checkpoint.
        monitor_mode: 'max' or 'min'. How to compare the monitored metric.
        is_final: If True, marks this as the final checkpoint, potentially skipping cleanup.
    """
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        # Save model state
        model_to_save = model.module if hasattr(model, 'module') else model
        model_path = os.path.join(checkpoint_dir, 'pytorch_model.bin')
        torch.save(model_to_save.state_dict(), model_path)
        logger.info(f"Model state saved to {model_path}")

        # Save optimizer state
        optimizer_path = os.path.join(checkpoint_dir, 'optimizer.pt')
        torch.save(optimizer.state_dict(), optimizer_path)
        logger.info(f"Optimizer state saved to {optimizer_path}")

        # Save scheduler state
        scheduler_path = os.path.join(checkpoint_dir, 'scheduler.pt')
        torch.save(scheduler.state_dict(), scheduler_path)
        logger.info(f"Scheduler state saved to {scheduler_path}")

        # Save the full configuration
        if config:
            config_path = os.path.join(checkpoint_dir, 'config.json')
            try:
                # If it's a class with to_dict, use it, otherwise convert directly
                config_dict = config.to_dict() if hasattr(config, 'to_dict') else config
                with open(config_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
                logger.info(f"Configuration saved to {config_path}")
            except Exception as e:
                logger.warning(f"Could not save configuration to {config_path}: {e}")

        # Save training state (step, epoch, etc.)
        state_path = os.path.join(checkpoint_dir, 'trainer_state.json')
        state = {'global_step': global_step, 'epoch': epoch}
        if metrics:
            state['metrics'] = metrics
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"Training state saved to {state_path}")

        # --- Start: Add logic for saving best checkpoint ---
        if metric_to_monitor and metrics and metric_to_monitor in metrics:
            current_metric_value = metrics[metric_to_monitor]
            best_checkpoint_dir = os.path.join(output_dir, "best_checkpoint")
            best_metric_file = os.path.join(best_checkpoint_dir, "best_metric.json")

            previous_best_metric = -float('inf') if monitor_mode == 'max' else float('inf')

            # Check if a previous best checkpoint exists and read its metric
            if os.path.exists(best_metric_file):
                try:
                    with open(best_metric_file, 'r') as f:
                        best_metric_data = json.load(f)
                        # Ensure the metric name matches what we are currently monitoring
                        if best_metric_data.get('metric_name') == metric_to_monitor:
                            previous_best_metric = best_metric_data.get('value', previous_best_metric)
                except Exception as e:
                    logger.warning(f"Could not read previous best metric from {best_metric_file}: {e}")

            # Compare current metric with previous best
            is_better = False
            if monitor_mode == 'max' and current_metric_value > previous_best_metric:
                is_better = True
            elif monitor_mode == 'min' and current_metric_value < previous_best_metric:
                is_better = True

            if is_better:
                logger.info(f"New best metric '{metric_to_monitor}': {current_metric_value:.4f} (previous: {previous_best_metric:.4f}). Saving to {best_checkpoint_dir}")

                # Remove old best checkpoint directory if it exists
                if os.path.exists(best_checkpoint_dir):
                    try:
                        shutil.rmtree(best_checkpoint_dir)
                    except Exception as e:
                        logger.error(f"Failed to remove previous best checkpoint {best_checkpoint_dir}: {e}")

                # Copy the current checkpoint to the best checkpoint directory
                try:
                    # Use dirs_exist_ok=True for Python 3.8+
                    shutil.copytree(checkpoint_dir, best_checkpoint_dir, dirs_exist_ok=True)

                    # Save the new best metric info
                    best_metric_data = {
                        'metric_name': metric_to_monitor,
                        'value': current_metric_value,
                        'global_step': global_step,
                        'epoch': epoch,
                        'timestamp': datetime.now().isoformat()
                    }
                    with open(best_metric_file, 'w') as f:
                        json.dump(best_metric_data, f, indent=2)

                    # Save full performance metrics in the same format as final_metrics.txt
                    if metrics:
                        best_metrics_file = os.path.join(best_checkpoint_dir, "best_metrics.txt")
                        with open(best_metrics_file, 'w') as f:
                            for metric_name, metric_value in metrics.items():
                                f.write(f"{metric_name} = {metric_value:.4f}\n")
                        logger.info(f"Best performance metrics saved to {best_metrics_file}")

                except Exception as e:
                    logger.error(f"Failed to copy or save best checkpoint to {best_checkpoint_dir}: {e}")
        # --- End: Add logic for saving best checkpoint ---

        # Clean up old checkpoints
        cleanup_old_checkpoints(output_dir, max_to_keep)

    except Exception as e:
        logger.error(f"Failed to save checkpoint at step {global_step}: {e}")
        # Clean up partially created checkpoint directory if saving failed
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)
            logger.warning(f"Removed partially created checkpoint directory: {checkpoint_dir}")


def cleanup_old_checkpoints(output_dir: str, max_to_keep: int):
    """
    Removes older checkpoints, keeping only the `max_to_keep` most recent ones.
    The 'best_checkpoint' directory, if exists, is always preserved.

    Args:
        output_dir: Directory containing the checkpoints.
        max_to_keep: Maximum number of recent checkpoints to retain.
    """
    # Get all checkpoint-{step} directories
    checkpoints = []
    best_checkpoint_name = "best_checkpoint" # Define the name
    try:
        for d in os.listdir(output_dir):
            dir_path = os.path.join(output_dir, d)
            # Exclude the best checkpoint directory from cleanup consideration
            if d == best_checkpoint_name or not os.path.isdir(dir_path):
                continue

            if d.startswith("checkpoint-"):
                try:
                    step = int(d.split("-")[1])
                    checkpoints.append((step, dir_path))
                except (IndexError, ValueError):
                    logger.warning(f"Could not parse step from checkpoint directory name: {d}")
                    continue
    except FileNotFoundError:
        logger.warning(f"Checkpoint directory {output_dir} not found during cleanup.")
        return

    if len(checkpoints) <= max_to_keep:
        logger.debug(f"No old checkpoints to remove. Found {len(checkpoints)}, max_to_keep={max_to_keep}")
        return

    # Sort checkpoints by step number (ascending)
    checkpoints.sort(key=lambda x: x[0])

    # Determine checkpoints to remove
    num_to_remove = len(checkpoints) - max_to_keep
    to_remove = checkpoints[:num_to_remove]

    logger.info(f"Found {len(checkpoints)} checkpoints. Keeping {max_to_keep}, removing {num_to_remove}." )

    for step, dir_path in to_remove:
        try:
            shutil.rmtree(dir_path)
            logger.info(f"Removed old checkpoint: {dir_path}")
        except OSError as e:
            logger.error(f"Error removing checkpoint {dir_path}: {e}")

def find_best_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find the best checkpoint directory saved based on the monitored metric.

    Args:
        output_dir: Directory where checkpoints are saved.

    Returns:
        Path to the best checkpoint directory or None if not found.
    """
    best_checkpoint_dir = os.path.join(output_dir, "best_checkpoint")
    if os.path.exists(best_checkpoint_dir) and os.path.isdir(best_checkpoint_dir):
        # Optionally, verify it contains the best_metric.json file
        if os.path.exists(os.path.join(best_checkpoint_dir, "best_metric.json")):
            return best_checkpoint_dir
        else:
            logger.warning(f"Found {best_checkpoint_dir} but it's missing 'best_metric.json'.")
            # Depending on requirements, you might return the path anyway or be strict:
            return None # Strict: require the metric file
            # return best_checkpoint_dir # Less strict: return path if directory exists
    return None


def load_checkpoint(
    checkpoint_dir: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Load model checkpoint with optimizer and scheduler state.
    
    Args:
        checkpoint_dir: Directory with saved checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        map_location: Device to map tensors to
        strict: Whether to strictly enforce that the keys in state_dict match
        
    Returns:
        Dictionary with loaded training state
        
    Raises:
        FileNotFoundError: If checkpoint directory or files don't exist
        FileError: If loading fails
    """
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory {checkpoint_dir} not found")
    
    # Load model weights
    try:
        model.from_pretrained(checkpoint_dir, map_location=map_location)
    except Exception as e:
        error_msg = f"Failed to load model weights from {checkpoint_dir}: {str(e)}"
        logger.error(error_msg)
        raise FileError(error_msg) from e
    
    # Load optimizer and scheduler
    opt_sch_path = os.path.join(checkpoint_dir, "optimizer_scheduler.pt")
    training_state = {}
    
    if os.path.exists(opt_sch_path):
        try:
            checkpoint = torch.load(opt_sch_path, map_location=map_location)
            
            # Load optimizer state
            if optimizer is not None and 'optimizer' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
            
            # Load scheduler state
            if scheduler is not None and 'scheduler' in checkpoint and checkpoint['scheduler']:
                scheduler.load_state_dict(checkpoint['scheduler'])
            
            # Get training state
            training_state = {
                'global_step': checkpoint.get('global_step', 0),
                'epoch': checkpoint.get('epoch', 0),
                'metrics': checkpoint.get('metrics', {}),
                'timestamp': checkpoint.get('timestamp', '')
            }
            
            logger.info(f"Loaded checkpoint from {checkpoint_dir} (step: {training_state['global_step']})")
            
        except Exception as e:
            error_msg = f"Failed to load optimizer/scheduler from {opt_sch_path}: {str(e)}"
            logger.error(error_msg)
            raise FileError(error_msg) from e
    
    # Load training state from JSON if available
    json_path = os.path.join(checkpoint_dir, "training_state.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                json_state = json.load(f)
                for k, v in json_state.items():
                    if k not in training_state:
                        training_state[k] = v
        except Exception as e:
            logger.warning(f"Failed to load training state from {json_path}: {str(e)}")
    
    return training_state


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in output directory.
    
    Args:
        output_dir: Directory with checkpoints
        
    Returns:
        Path to latest checkpoint directory or None if no checkpoints found
    """
    if not os.path.exists(output_dir):
        return None
    
    # Look for "final_checkpoint" first
    final_ckpt = os.path.join(output_dir, "final_checkpoint")
    if os.path.exists(final_ckpt) and os.path.isdir(final_ckpt):
        return final_ckpt
    
    # Look for checkpoint-{step} directories
    checkpoints = []
    for d in os.listdir(output_dir):
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d)):
            try:
                step = int(d.split("-")[1])
                checkpoints.append((step, os.path.join(output_dir, d)))
            except (IndexError, ValueError):
                continue
    
    # Find checkpoint with highest step
    if checkpoints:
        _, latest_checkpoint = max(checkpoints, key=lambda x: x[0])
        return latest_checkpoint
    
    return None


def verify_checkpoint(checkpoint_dir: str) -> bool:
    """
    Verify that a checkpoint directory contains valid files.
    
    Args:
        checkpoint_dir: Directory with checkpoint
        
    Returns:
        True if checkpoint is valid, False otherwise
    """
    if not os.path.exists(checkpoint_dir) or not os.path.isdir(checkpoint_dir):
        logger.warning(f"Checkpoint directory {checkpoint_dir} not found")
        return False
    
    # Support two layouts:
    #   1) HF-style: config.json + (pytorch_model.bin | model.safetensors)
    #   2) Generic:  model.pt + config.json (our custom baseline format)

    # Check for HF-style files
    required_files = ["config.json"]
    model_files = ["pytorch_model.bin", "model.safetensors"]
    has_hf_model_file = any(os.path.exists(os.path.join(checkpoint_dir, f)) for f in model_files)
    has_config = os.path.exists(os.path.join(checkpoint_dir, "config.json"))

    # Check for generic baseline style
    has_generic_model = os.path.exists(os.path.join(checkpoint_dir, "model.pt")) and has_config
    
    if (has_hf_model_file and has_config) or has_generic_model:
        return True
    else:
        logger.warning(f"No valid checkpoint layout found in {checkpoint_dir}")
        return False


def extract_checkpoint_step(checkpoint_dir: str) -> Optional[int]:
    """
    Extract step number from checkpoint directory name.
    
    Args:
        checkpoint_dir: Checkpoint directory path
        
    Returns:
        Step number or None if not found
    """
    basename = os.path.basename(checkpoint_dir)
    if basename.startswith("checkpoint-"):
        try:
            return int(basename.split("-")[1])
        except (IndexError, ValueError):
            pass
    
    # Try to get step from training_state.json
    json_path = os.path.join(checkpoint_dir, "training_state.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                state = json.load(f)
                if 'global_step' in state:
                    return state['global_step']
        except Exception:
            pass
    
    return None


def create_checkpoint_metadata(
    model_name: str,
    global_step: int,
    epoch: int,
    metrics: Optional[Dict[str, float]] = None,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create metadata dictionary for checkpoint.
    
    Args:
        model_name: Name of the model
        global_step: Current global step
        epoch: Current epoch
        metrics: Evaluation metrics
        config: Model configuration
        
    Returns:
        Metadata dictionary
    """
    return {
        "model_name": model_name,
        "global_step": global_step,
        "epoch": epoch,
        "metrics": metrics or {},
        "config": config or {},
        "timestamp": datetime.now().isoformat(),
        "pytorch_version": torch.__version__
    }


def _resolve_checkpoint_config(model_path: str, config: Optional[Any], logger: logging.Logger) -> dict:
    """
    Read and merge checkpoint config.json with the passed config dict.

    Priority: checkpoint config.json values are used as defaults, but the
    passed ``config`` dict (from the pipeline run) takes precedence when
    both provide the same key.

    Returns a plain dict with all configuration values needed for model
    construction.
    """
    merged: dict = {}

    # 1. Read config.json from the checkpoint directory (or parent)
    for candidate in [
        os.path.join(model_path, "config.json"),
        os.path.join(os.path.dirname(model_path), "config.json"),
    ]:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r") as f:
                    merged = json.load(f)
                logger.info(f"Loaded checkpoint config from {candidate}")
            except Exception as e:
                logger.warning(f"Error reading {candidate}: {e}")
            break

    # 2. Overlay the runtime config (takes precedence)
    if config is not None:
        if hasattr(config, "items"):
            for k, v in config.items():
                if v is not None:
                    merged[k] = v
        elif hasattr(config, "to_dict"):
            for k, v in config.to_dict().items():
                if v is not None:
                    merged[k] = v

    return merged


def _load_weights_into_model(model, model_path: str, logger: logging.Logger) -> None:
    """
    Load ``pytorch_model.bin`` (preferred) or ``model.pt`` into *model*.
    """
    candidates = [
        os.path.join(model_path, "pytorch_model.bin"),
        os.path.join(model_path, "model.pt"),
    ]
    for weight_path in candidates:
        if os.path.exists(weight_path):
            state_dict = torch.load(weight_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=True)
            logger.info(f"Loaded weights from {weight_path}")
            return
    logger.warning(f"No pytorch_model.bin or model.pt found in {model_path}")


def _attach_config_shim(model, cfg: dict, logger: logging.Logger) -> None:
    """
    Attach a minimal HF-like ``.config`` namespace so downstream code can
    access ``id2label`` / ``label2id`` on non-HF wrapper models.
    """
    try:
        id2label_map = cfg.get("id2label", {})
        id2label_map = {int(k): v for k, v in id2label_map.items()} if isinstance(id2label_map, dict) else {}
        label2id_map = cfg.get("label2id", {})
        model.config = SimpleNamespace(id2label=id2label_map, label2id=label2id_map)
    except Exception as e:
        logger.warning(f"Failed to attach config shim: {e}")


def load_model_for_evaluation(
    model_path: str,
    config: Optional[Any] = None,
    device: Optional[Union[str, torch.device]] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[Any, Any]:
    """
    Load a model for evaluation from a checkpoint directory or HuggingFace
    model name.

    Architecture is determined from the ``"architecture"`` key in the
    checkpoint's ``config.json`` (or the passed *config* dict).  Supported
    values: ``segformer``, ``deeplabv3``, ``setr``, ``upernet_swin``,
    ``oneformer``.

    Args:
        model_path: Path to checkpoint directory or HuggingFace model name.
        config: Optional runtime configuration dict / Config object.
        device: Device to place the model on (defaults to CPU).
        logger: Optional logger instance.

    Returns:
        Tuple of (model, image_processor).
    """
    if logger is None:
        logger = get_logger()

    logger.info(f"Loading model from: {model_path}")

    if device is None:
        device = torch.device("cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    try:
        is_local_path = os.path.exists(model_path)

        if not is_local_path:
            # ---- Remote HuggingFace model name ----
            from transformers import SegformerForSemanticSegmentation
            model = SegformerForSemanticSegmentation.from_pretrained(model_path)
            logger.info(f"Loaded model from HuggingFace: {model_path}")

        else:
            # ---- Local checkpoint ----
            # 1. Read config once
            cfg = _resolve_checkpoint_config(model_path, config, logger)
            architecture = cfg.get("architecture", "segformer")
            logger.info(f"Detected architecture: '{architecture}'")

            # Common fields used by most wrapper models
            num_classes = len(cfg.get("id2label", {"0": "__background__", "1": "tree"}))
            ignore_index = cfg.get("semantic_loss_ignore_index", 255)
            eval_config = config if config is not None else cfg

            # 2. Dispatch by architecture
            if architecture == "segformer":
                model = _load_segformer(model_path, cfg, logger)

            elif architecture == "deeplabv3":
                from model import DeepLabV3Wrapper
                backbone = cfg.get("backbone", "resnet50")
                logger.info(f"Loading DeepLabV3Wrapper (backbone={backbone})")
                model = DeepLabV3Wrapper(
                    backbone=backbone,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    project_config=eval_config,
                )
                _load_weights_into_model(model, model_path, logger)
                _attach_config_shim(model, cfg, logger)

            elif architecture == "setr":
                from model import SETRWrapper
                embed_dim = cfg.get("setr_embed_dim", 768)
                patch_size = cfg.get("setr_patch_size", 16)
                logger.info(f"Loading SETRWrapper (embed_dim={embed_dim}, patch_size={patch_size})")
                model = SETRWrapper(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    project_config=eval_config,
                    embed_dim=embed_dim,
                    patch_size=patch_size,
                )
                _load_weights_into_model(model, model_path, logger)
                _attach_config_shim(model, cfg, logger)

            elif architecture == "upernet_swin":
                from new_architectures import UperNetSwinWrapper
                upernet_swin_model = cfg.get("upernet_swin_model", "openmmlab/upernet-swin-base")
                id2label_map = cfg.get("id2label", {0: "background", 1: "tree_crown"})
                id2label_map = {int(k): v for k, v in id2label_map.items()} if isinstance(id2label_map, dict) else {}
                label2id_map = cfg.get("label2id", {"background": 0, "tree_crown": 1})
                logger.info(f"Loading UperNetSwinWrapper (model={upernet_swin_model})")
                model = UperNetSwinWrapper(
                    model_name=upernet_swin_model,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    project_config=eval_config,
                    id2label=id2label_map,
                    label2id=label2id_map,
                )
                _load_weights_into_model(model, model_path, logger)
                _attach_config_shim(model, cfg, logger)

            elif architecture == "oneformer":
                from new_architectures import OneFormerWrapper
                oneformer_model_name = cfg.get("oneformer_model", "shi-labs/oneformer_ade20k_swin_large")
                id2label_map = cfg.get("id2label", {0: "background", 1: "tree_crown"})
                id2label_map = {int(k): v for k, v in id2label_map.items()} if isinstance(id2label_map, dict) else {}
                label2id_map = cfg.get("label2id", {"background": 0, "tree_crown": 1})
                logger.info(f"Loading OneFormerWrapper (model={oneformer_model_name})")
                model = OneFormerWrapper(
                    model_name=oneformer_model_name,
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    project_config=eval_config,
                    id2label=id2label_map,
                    label2id=label2id_map,
                )
                _load_weights_into_model(model, model_path, logger)
                _attach_config_shim(model, cfg, logger)

            else:
                raise ValueError(
                    f"Unsupported architecture '{architecture}' in checkpoint at {model_path}. "
                    f"Supported: segformer, deeplabv3, setr, upernet_swin, oneformer."
                )

        # Move to device and create image processor
        model = model.to(device)
        from transformers import SegformerImageProcessor
        image_processor = SegformerImageProcessor(
            do_resize=False,
            do_rescale=True,
            do_normalize=True,
        )
        return model, image_processor

    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}", exc_info=True)
        raise


def _load_segformer(model_path: str, cfg: dict, logger: logging.Logger):
    """
    Load a SegFormer from a local checkpoint.

    Uses the HuggingFace ``from_pretrained`` API.  If a ``model_name``
    backbone identifier is available in *cfg* it is used to explicitly
    construct the ``SegformerConfig`` so that architecture mismatches are
    caught early.
    """
    from transformers import SegformerForSemanticSegmentation, SegformerConfig

    hf_backbone_name = cfg.get("model_name")  # e.g. "nvidia/mit-b5"

    if not hf_backbone_name:
        logger.info(f"Loading SegFormer directly from checkpoint (no backbone override): {model_path}")
        model = SegformerForSemanticSegmentation.from_pretrained(model_path)
    else:
        logger.info(f"Loading SegFormer with backbone '{hf_backbone_name}' and weights from '{model_path}'")

        # Resolve num_labels
        id2label = cfg.get("id2label")
        label2id = cfg.get("label2id")
        num_labels = cfg.get("num_labels")
        if num_labels is None:
            if id2label and isinstance(id2label, dict):
                num_labels = len(id2label)
            elif label2id and isinstance(label2id, dict):
                num_labels = len(label2id)

        seg_config = SegformerConfig.from_pretrained(
            hf_backbone_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        model = SegformerForSemanticSegmentation.from_pretrained(
            model_path,
            config=seg_config,
        )

    logger.info("Successfully loaded SegFormer model")
    return model
