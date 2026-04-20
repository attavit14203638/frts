#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Centralized pipeline for TCD segmentation.

This module provides unified entry points for training, evaluation, and prediction
of TCD segmentation models, consolidating functionality that was previously spread
across multiple files.
"""

import os
import json
import torch
import numpy as np
import logging
from PIL import Image
import matplotlib.pyplot as plt
import cv2
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from typing import Dict, Optional, Tuple, Any, Union, List
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# Import from refactored modules
from config import Config
from dataset import load_and_shuffle_dataset, create_dataloaders, create_dataset_from_masks
from model import create_model
from checkpoint import save_checkpoint, load_checkpoint, verify_checkpoint
from metrics import (
    calculate_metrics, calculate_boundary_iou,
    categorize_errors, calculate_error_statistics,
    ERROR_CATEGORIES, calculate_confusion_matrix)
from visualization import (
    plot_training_metrics,
    plot_prediction_confidence,
    plot_confusion_matrix,
    plot_error_analysis_map,
    tensor_to_image,
    plot_prediction_comparison_with_confidence,
    visualize_segmentation,
    plot_learning_rate_schedule
)
from image_utils import (
    ensure_rgb,
    get_tiles,
    stitch_tiles
)
from utils import set_seed, log_or_print

def ensure_mask_dimensions(mask_np, target_shape, logger=None, context="", is_label_mask=True):
    """
    Ensures the mask (numpy array) matches the target shape (H, W).
    Uses OpenCV for resizing as the standard method.
    """
    import cv2
    
    orig_shape = mask_np.shape
    target_h, target_w = target_shape
    log_prefix = f"[ensure_mask_dimensions]{'['+context+']' if context else ''}"

    if orig_shape[:2] == (target_h, target_w):
        return mask_np

    # Log warning about resizing
    if logger:
        logger.warning(f"{log_prefix} Resizing mask from {orig_shape} to {target_shape}")

    try:
        interp = cv2.INTER_NEAREST if is_label_mask else cv2.INTER_LINEAR
        # OpenCV uses (width, height)
        resized_np = cv2.resize(mask_np, (target_w, target_h), interpolation=interp)
        return resized_np
    except Exception as e:
        if logger:
            logger.error(f"{log_prefix} OpenCV resize failed: {e}. Returning original mask.")
        return mask_np

def run_training_pipeline(
    config: Config, 
    logger: Optional[Any] = None, 
    is_notebook: bool = False,
    fold_dataset_dict: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Run the complete TCD training pipeline.

    Args:
        config: Configuration object
        logger: Optional logger for logging messages
        is_notebook: Whether running in a notebook environment
        fold_dataset_dict: Optional pre-split dataset dict for cross-validation
        
    Returns:
        Dictionary with training results including trained model and metrics
    """
    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)

    # Set random seed for reproducibility
    set_seed(config["seed"])

    # Create TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(config["output_dir"], "tensorboard"))

    # Log basic information — only architecture-relevant params, not the full config dump
    arch = config.get("architecture", "segformer")
    _arch_param_keys = {
        "segformer": ["model_name"],
        "deeplabv3": ["backbone"],
        "setr": ["setr_embed_dim", "setr_patch_size", "setr_input_size"],
        "oneformer": ["oneformer_model"],
        "upernet_swin": ["upernet_swin_model"],
    }
    _shared_keys = [
        "architecture", "dataset_name", "num_epochs", "train_batch_size",
        "learning_rate", "weight_decay", "gradient_accumulation_steps",
        "mixed_precision", "loss_type", "train_time_upsample",
        "scheduler_type", "output_dir", "seed",
    ]
    _active_keys = _shared_keys + _arch_param_keys.get(arch, [])
    _active_config = {k: config.get(k) for k in _active_keys if config.get(k) is not None}
    log_or_print(f"Starting training with config: {_active_config}", logger, logging.INFO, is_notebook)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_or_print(f"Using device: {device}", logger, logging.INFO, is_notebook)

    # Log train-time upsampling status for all architectures
    upsample_status = "enabled" if config.get('train_time_upsample', False) else "disabled"
    log_or_print(f"Train-time upsampling for {arch.upper()} is {upsample_status}", logger, logging.INFO, is_notebook)

    # Load dataset - either use provided fold dataset or load a new one
    if fold_dataset_dict is not None:
        # Use the pre-split dataset for cross-validation
        log_or_print("Using pre-split dataset for cross-validation", logger, logging.INFO, is_notebook)
        dataset_dict = fold_dataset_dict
    else:
        # Load and shuffle dataset once to ensure consistent ordering
        log_or_print(f"Loading dataset with consistent seed {config['seed']}...", logger, logging.INFO, is_notebook)
        dataset_dict = load_and_shuffle_dataset(
            dataset_name=config["dataset_name"],
            seed=config["seed"]
        )

    # Create dataloaders
    log_or_print("Creating dataloaders...", logger, logging.INFO, is_notebook)

    # For cross-validation, we want to skip the validation split since we already have one
    validation_split = None if fold_dataset_dict is not None else config["validation_split"]
    
    train_dataloader, eval_dataloader, id2label, label2id = create_dataloaders(
        dataset_dict=dataset_dict,
        image_processor=None,  # Will be created internally
        config=config, # Pass the config object
        train_batch_size=config["train_batch_size"],
        eval_batch_size=config["eval_batch_size"],
        num_workers=config["num_workers"],
        validation_split=validation_split,
        seed=config["seed"]
        # Removed: image_size=config["image_size"]
    )

    # Update config with id2label and label2id
    config["id2label"] = id2label
    config["label2id"] = label2id

    # Create model using the corrected function signature
    model, optimizer, scheduler = create_model(
        config=config,  # Pass config object
        num_training_steps=len(train_dataloader) * config["num_epochs"] // config["gradient_accumulation_steps"],
        logger=logger # Pass logger for checkpoint loading messages
    )

    # Plot and log learning rate schedule
    try:
        lr_plot_path = os.path.join(config["output_dir"], "lr_schedule.png")
        lr_fig = plot_learning_rate_schedule(
            scheduler=scheduler,
            num_training_steps=len(train_dataloader) * config["num_epochs"] // config["gradient_accumulation_steps"],
            optimizer=optimizer,
            save_path=lr_plot_path
        )
        if writer:
            writer.add_figure("Training/LearningRateSchedule", lr_fig)
        plt.close(lr_fig) # Close figure to free memory
        if logger:
            logger.info(f"Learning rate schedule plot saved to {lr_plot_path}")
    except Exception as e:
        if logger:
            logger.warning(f"Could not plot learning rate schedule: {e}")

    # Train the model
    log_or_print("Starting training...", logger, logging.INFO, is_notebook)

    trained_model = train_model(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        logger=logger,
        writer=writer,
        is_notebook=is_notebook
    )

    # Evaluate the model
    log_or_print("Evaluating final model...", logger, logging.INFO, is_notebook)

    metrics = evaluate_model(
        model=trained_model,
        eval_dataloader=eval_dataloader,
        device=device,
        output_dir=config["output_dir"],
        id2label=id2label,
        analyze_errors=config.get("analyze_errors", False),
        logger=logger,
        is_notebook=is_notebook
    )

    # Log and save final metrics
    log_or_print("Final evaluation metrics:", logger, logging.INFO, is_notebook)
    for metric_name, metric_value in metrics.items():
        log_or_print(f"  {metric_name} = {metric_value:.4f}", logger, logging.INFO, is_notebook)

    with open(os.path.join(config["output_dir"], "final_metrics.txt"), "w") as f:
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name} = {metric_value:.4f}\n")

    # Save final model
    final_model_dir = os.path.join(config["output_dir"], "final_model")
    os.makedirs(final_model_dir, exist_ok=True)

    # Save depending on architecture
    arch = config.get("architecture", "segformer")
    if arch == "segformer":
        trained_model.save_pretrained(final_model_dir)
        # Also save optimizer and scheduler state for completeness
        torch.save({
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }, os.path.join(final_model_dir, "optimizer_scheduler.pt"))
    else:
        # For DeepLabV3, SETR, and other custom models, save a standardized checkpoint
        # This ensures 'pytorch_model.bin' and 'config.json' are created correctly
        final_step = len(train_dataloader) * config["num_epochs"] // config["gradient_accumulation_steps"]
        save_checkpoint(
            model=trained_model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=final_step,
            epoch=config["num_epochs"],
            output_dir=config["output_dir"],
            is_final=True,  # Mark as final to avoid cleanup
            metrics=metrics,
            config=config # Pass the full config to be saved
        )
        # Rename the final checkpoint folder to 'final_model' for consistency
        final_checkpoint_path = os.path.join(config["output_dir"], f"checkpoint-{final_step}")
        if os.path.exists(final_checkpoint_path):
            if os.path.exists(final_model_dir):
                import shutil
                shutil.rmtree(final_model_dir) # Remove old final_model dir if it exists
            os.rename(final_checkpoint_path, final_model_dir)
            log_or_print(f"Final model checkpoint saved and renamed to {final_model_dir}", logger, logging.INFO, is_notebook)
        else:
            log_or_print("Final checkpoint for custom architecture not found after saving.", logger, logging.WARNING, is_notebook)

    log_or_print(f"Final model saved to {final_model_dir}", logger, logging.INFO, is_notebook)

    # Close TensorBoard writer
    writer.close()

    # Return results
    return {
        "model": trained_model,
        "metrics": metrics,
        "model_dir": final_model_dir,
        "id2label": id2label,
        "label2id": label2id
    }

def train_model(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    eval_dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    config: Config,
    logger: Optional[Any] = None,
    writer: Optional[SummaryWriter] = None,
    is_notebook: bool = False
) -> torch.nn.Module:
    """
    Train the model with standardized parameters from config.

    Args:
        model: The model to train
        train_dataloader: DataLoader for training data
        eval_dataloader: DataLoader for evaluation data
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler
        device: Device to train on (cuda or cpu)
        config: Configuration object
        logger: Optional logger for logging messages
        writer: TensorBoard writer for logging metrics
        is_notebook: Whether running in a notebook environment

    Returns:
        Trained model
    """
    # Extract training parameters from config
    num_epochs = config["num_epochs"]
    logging_steps = config["logging_steps"]
    eval_steps = config["eval_steps"]
    save_steps = config["save_steps"]
    mixed_precision = config["mixed_precision"]
    gradient_accumulation_steps = config["gradient_accumulation_steps"]
    output_dir = config["output_dir"]
    # --- Get best checkpoint config ---
    save_best = config.get("save_best_checkpoint", True)
    metric_to_monitor = config.get("best_checkpoint_metric", "f1_score_class_1")
    monitor_mode = config.get("best_checkpoint_monitor_mode", "max")
    max_to_keep = config.get("max_checkpoints_to_keep", 3)
    # --- End get best checkpoint config ---

    # Move model to device
    model = model.to(device)

    # Set up mixed precision training
    scaler = None
    amp_context = lambda: torch.no_op()
    amp_enabled = False
    pytorch_version = torch.__version__

    if mixed_precision and torch.cuda.is_available():
        try:
            from torch.amp import GradScaler, autocast
            scaler = GradScaler()
            amp_context = lambda: autocast(device_type='cuda', dtype=torch.float16)
            amp_enabled = True
            log_or_print(f"Using torch.amp (PyTorch {pytorch_version}) for mixed precision.", logger, logging.INFO, is_notebook)
        except ImportError as e:
            amp_enabled = False
            log_or_print(f"Failed to import torch.amp ({e}). Disabling mixed precision.", logger, logging.ERROR, is_notebook)
    elif mixed_precision and not torch.cuda.is_available():
        log_or_print("Mixed precision requested but CUDA is not available. Using regular precision on CPU.", logger, logging.WARNING, is_notebook)

    # Training loop
    global_step = 0
    total_loss = 0.0
    nan_detected = False  # Flag to track NaN detection for early termination
    
    consecutive_nan_count = 0  # Track consecutive NaN batches
    max_consecutive_nan = 5  # Halt after this many consecutive NaN losses

    for epoch in range(num_epochs):
        # Log epoch start
        log_or_print(f"Starting epoch {epoch + 1}/{num_epochs}", logger, logging.INFO, is_notebook)

        model.train()

        # Create progress bar
        from tqdm import tqdm
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass with mixed precision if enabled
            if amp_enabled and scaler is not None:
                with amp_context():
                    outputs = model(**batch)
                    loss = outputs.loss / gradient_accumulation_steps

                # NaN/Inf detection before backward pass
                if torch.isnan(loss) or torch.isinf(loss):
                    consecutive_nan_count += 1
                    log_or_print(f"WARNING: NaN/Inf loss detected at epoch {epoch+1}, step {step}, global_step {global_step}. "
                                 f"Consecutive NaN count: {consecutive_nan_count}/{max_consecutive_nan}", 
                                 logger, logging.WARNING, is_notebook)
                    optimizer.zero_grad()  # Clear any accumulated gradients
                    if consecutive_nan_count >= max_consecutive_nan:
                        log_or_print(f"CRITICAL: {max_consecutive_nan} consecutive NaN losses detected. Halting training.", 
                                     logger, logging.ERROR, is_notebook)
                        nan_detected = True
                        break
                    continue  # Skip this batch
                else:
                    consecutive_nan_count = 0  # Reset counter on valid loss

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()

                # Update weights if we've accumulated enough gradients
                if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    # Step scheduler only if it's not ReduceLROnPlateau
                    if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Log training metrics (MOVED INSIDE OPTIMIZER STEP)
                    if global_step > 0 and global_step % logging_steps == 0:
                        avg_loss = total_loss / (logging_steps * gradient_accumulation_steps) # Adjust for accumulated loss
                        lr = scheduler.get_last_lr()[0]
                        log_or_print(f"Step {global_step}: loss = {avg_loss:.4f}, lr = {lr:.8f}", logger, logging.INFO, is_notebook)
                        if writer:
                            writer.add_scalar("train/loss", avg_loss, global_step)
                            writer.add_scalar("train/learning_rate", lr, global_step)
                        total_loss = 0.0 # Reset accumulated loss for logging
                    
                    # Evaluate model (MOVED INSIDE OPTIMIZER STEP)
                    if global_step > 0 and global_step % eval_steps == 0:
                        log_or_print(f"Evaluating model at step {global_step}", logger, logging.INFO, is_notebook)
                        metrics = evaluate_model(
                            model=model, eval_dataloader=eval_dataloader, device=device,
                            logger=logger, is_notebook=is_notebook, output_dir=output_dir, # Pass output_dir for visualizations
                            id2label=config.get("id2label") # Pass id2label for visualizations
                        )
                        for metric_name, metric_value in metrics.items():
                            log_or_print(f"{metric_name} = {metric_value:.4f}", logger, logging.INFO, is_notebook)
                            if writer:
                                writer.add_scalar(f"eval/{metric_name}", metric_value, global_step)
                        
                        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            metric_to_monitor_for_scheduler = config.get("scheduler_monitor", "val_loss")
                            metric_value_for_scheduler = metrics.get(metric_to_monitor_for_scheduler)
                            if metric_value_for_scheduler is not None:
                                scheduler.step(metric_value_for_scheduler)
                                log_or_print(f"ReduceLROnPlateau scheduler step with {metric_to_monitor_for_scheduler}: {metric_value_for_scheduler:.4f}", logger, logging.INFO, is_notebook)
                                current_lr = optimizer.param_groups[0]['lr']
                                log_or_print(f"Current learning rate after scheduler step: {current_lr:.8f}", logger, logging.INFO, is_notebook)
                                if writer: writer.add_scalar("train/learning_rate", current_lr, global_step)
                            else:
                                log_or_print(f"Metric '{metric_to_monitor_for_scheduler}' not found. Scheduler cannot step.", logger, logging.WARNING, is_notebook)
                        
                        if save_best:
                            save_checkpoint(
                                model=model, optimizer=optimizer, scheduler=scheduler,
                                global_step=global_step, epoch=epoch, output_dir=output_dir,
                                metrics=metrics, metric_to_monitor=metric_to_monitor,
                                monitor_mode=monitor_mode, max_to_keep=max_to_keep, config=config
                            )
                        model.train()

                    # Save model checkpoint (periodic save, MOVED INSIDE OPTIMIZER STEP)
                    if global_step > 0 and global_step % save_steps == 0:
                        save_checkpoint(
                            model=model, optimizer=optimizer, scheduler=scheduler,
                            global_step=global_step, epoch=epoch, output_dir=output_dir,
                            max_to_keep=max_to_keep, config=config
                        )
                        log_or_print(f"Saved model checkpoint at step {global_step}", logger, logging.INFO, is_notebook)

            else: # Standard forward pass (no mixed precision)
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps

                # NaN/Inf detection before backward pass
                if torch.isnan(loss) or torch.isinf(loss):
                    consecutive_nan_count += 1
                    log_or_print(f"WARNING: NaN/Inf loss detected at epoch {epoch+1}, step {step}, global_step {global_step}. "
                                 f"Consecutive NaN count: {consecutive_nan_count}/{max_consecutive_nan}", 
                                 logger, logging.WARNING, is_notebook)
                    optimizer.zero_grad()  # Clear any accumulated gradients
                    if consecutive_nan_count >= max_consecutive_nan:
                        log_or_print(f"CRITICAL: {max_consecutive_nan} consecutive NaN losses detected. Halting training.", 
                                     logger, logging.ERROR, is_notebook)
                        nan_detected = True
                        break
                    continue  # Skip this batch
                else:
                    consecutive_nan_count = 0  # Reset counter on valid loss

                loss.backward()

                if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    if scheduler is not None and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # Log training metrics (MOVED INSIDE OPTIMIZER STEP)
                    if global_step > 0 and global_step % logging_steps == 0:
                        avg_loss = total_loss / (logging_steps * gradient_accumulation_steps) # Adjust for accumulated loss
                        lr = scheduler.get_last_lr()[0]
                        log_or_print(f"Step {global_step}: loss = {avg_loss:.4f}, lr = {lr:.8f}", logger, logging.INFO, is_notebook)
                        if writer:
                            writer.add_scalar("train/loss", avg_loss, global_step)
                            writer.add_scalar("train/learning_rate", lr, global_step)
                        total_loss = 0.0 # Reset accumulated loss for logging
                    
                    # Evaluate model (MOVED INSIDE OPTIMIZER STEP)
                    if global_step > 0 and global_step % eval_steps == 0:
                        log_or_print(f"Evaluating model at step {global_step}", logger, logging.INFO, is_notebook)
                        metrics = evaluate_model(
                            model=model, eval_dataloader=eval_dataloader, device=device,
                            logger=logger, is_notebook=is_notebook, output_dir=output_dir, # Pass output_dir for visualizations
                            id2label=config.get("id2label") # Pass id2label for visualizations
                        )
                        for metric_name, metric_value in metrics.items():
                            log_or_print(f"{metric_name} = {metric_value:.4f}", logger, logging.INFO, is_notebook)
                            if writer:
                                writer.add_scalar(f"eval/{metric_name}", metric_value, global_step)

                        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            metric_to_monitor_for_scheduler = config.get("scheduler_monitor", "val_loss")
                            metric_value_for_scheduler = metrics.get(metric_to_monitor_for_scheduler)
                            if metric_value_for_scheduler is not None:
                                scheduler.step(metric_value_for_scheduler)
                                log_or_print(f"ReduceLROnPlateau scheduler step with {metric_to_monitor_for_scheduler}: {metric_value_for_scheduler:.4f}", logger, logging.INFO, is_notebook)
                                current_lr = optimizer.param_groups[0]['lr']
                                log_or_print(f"Current learning rate after scheduler step: {current_lr:.8f}", logger, logging.INFO, is_notebook)
                                if writer: writer.add_scalar("train/learning_rate", current_lr, global_step)
                            else:
                                log_or_print(f"Metric '{metric_to_monitor_for_scheduler}' not found. Scheduler cannot step.", logger, logging.WARNING, is_notebook)
                        
                        if save_best:
                            save_checkpoint(
                                model=model, optimizer=optimizer, scheduler=scheduler,
                                global_step=global_step, epoch=epoch, output_dir=output_dir,
                                metrics=metrics, metric_to_monitor=metric_to_monitor,
                                monitor_mode=monitor_mode, max_to_keep=max_to_keep, config=config
                            )
                        model.train()

                    # Save model checkpoint (periodic save, MOVED INSIDE OPTIMIZER STEP)
                    if global_step > 0 and global_step % save_steps == 0:
                        save_checkpoint(
                            model=model, optimizer=optimizer, scheduler=scheduler,
                            global_step=global_step, epoch=epoch, output_dir=output_dir,
                            max_to_keep=max_to_keep, config=config
                        )
                        log_or_print(f"Saved model checkpoint at step {global_step}", logger, logging.INFO, is_notebook)
            
            # Update total loss (accumulates per-batch loss before averaging for logging)
            total_loss += loss.item() * gradient_accumulation_steps # Store non-divided loss for accumulation

            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})

            # Evaluate model
            # MOVED INSIDE THE OPTIMIZER STEP BLOCK

            # Save model checkpoint (periodic save)
            # MOVED INSIDE THE OPTIMIZER STEP BLOCK

        # Check if NaN was detected and break out of epoch loop
        if nan_detected:
            log_or_print(f"Training halted at epoch {epoch + 1} due to NaN detection. Saving emergency checkpoint...", 
                         logger, logging.ERROR, is_notebook)
            # Save emergency checkpoint before exiting
            save_checkpoint(
                model=model, optimizer=optimizer, scheduler=scheduler,
                global_step=global_step, epoch=epoch, output_dir=output_dir,
                is_final=True, config=config
            )
            log_or_print(f"Emergency checkpoint saved. Best checkpoint from before NaN may be usable.", 
                         logger, logging.INFO, is_notebook)
            break  # Exit the epoch loop

    # Save final model checkpoint (only if training completed normally)
    if not nan_detected:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            epoch=num_epochs,
            output_dir=output_dir,
            is_final=True,  # Mark as final to potentially skip cleanup logic if needed
            config=config  # Pass config to the final save
        )
        log_or_print("Saved final model checkpoint", logger, logging.INFO, is_notebook)
    else:
        log_or_print("Training was halted due to NaN detection. Check best_checkpoint for usable weights.", 
                     logger, logging.WARNING, is_notebook)

    return model


# --- Evaluation Helper Functions ---

def _run_evaluation_loop(
    model: torch.nn.Module,
    eval_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    logger: Optional[Any] = None,
    is_notebook: bool = False
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Runs the evaluation loop, collects predictions, labels, and loss."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(eval_dataloader, desc="Evaluating")
        for batch_idx, batch in enumerate(progress_bar):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            labels_tensor = batch["labels"]
            target_h, target_w = labels_tensor.shape[-2:]

            resized_logits = F.interpolate(
                logits,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False
            )

            preds = torch.argmax(resized_logits, dim=1).detach().cpu().numpy()
            labels = labels_tensor.detach().cpu().numpy()

            all_preds.append(preds)
            all_labels.append(labels)

    avg_loss = total_loss / len(eval_dataloader)
    all_preds_np = np.concatenate(all_preds, axis=0)
    all_labels_np = np.concatenate(all_labels, axis=0)

    return avg_loss, all_preds_np, all_labels_np


def _calculate_and_log_metrics(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    avg_loss: float,
    logger: Optional[Any] = None,
    is_notebook: bool = False
) -> Dict[str, float]:
    """Calculates and logs evaluation metrics with 'val_' prefix."""
    raw_metrics = calculate_metrics(all_preds, all_labels)
    
    # Add 'val_' prefix to all metrics and store avg_loss as 'val_loss'
    metrics = {}
    for key, value in raw_metrics.items():
        if not key.startswith("val_"):
            metrics[f"val_{key}"] = value
        else:
            metrics[key] = value # Already has prefix
    metrics["val_loss"] = avg_loss

    log_or_print("Evaluation metrics (prefixed with 'val_'):", logger, logging.INFO, is_notebook)
    for metric_name, metric_value in metrics.items(): # Iterate over the new prefixed metrics
        log_or_print(f"  {metric_name} = {metric_value:.4f}", logger, logging.INFO, is_notebook)

    return metrics


def _perform_error_analysis(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    output_dir: str,
    id2label: Dict[int, str],
    logger: Optional[Any] = None,
    is_notebook: bool = False
) -> Dict[str, float]:
    """Performs detailed error analysis."""
    error_stats = {}
    if not output_dir:
        return error_stats

    error_analysis_dir = os.path.join(output_dir, "error_analysis")
    os.makedirs(error_analysis_dir, exist_ok=True)
    log_or_print("Performing detailed error analysis...", logger, logging.INFO, is_notebook)

    try:
        num_classes = len(id2label)
        use_detailed_mode = num_classes > 2

        if use_detailed_mode:
            log_or_print(f"Using multi-class error analysis with {num_classes} classes.", logger, logging.INFO, is_notebook)

        full_error_map = categorize_errors(
            all_preds,
            all_labels,
            num_classes=num_classes,
            detailed_mode=use_detailed_mode
        )

        error_stats = calculate_error_statistics(full_error_map)
        log_or_print("Overall Error Statistics:", logger, logging.INFO, is_notebook)
        for stat_name, stat_value in error_stats.items():
            log_or_print(f"  {stat_name} = {stat_value:.2f}%", logger, logging.INFO, is_notebook)

    except Exception as e_analyze:
        log_or_print(f"Error during detailed error analysis: {e_analyze}", logger, logging.ERROR, is_notebook)

    return error_stats


def _plot_confusion_matrices(
    all_preds: np.ndarray,
    all_labels: np.ndarray,
    output_dir: str,
    id2label: Dict[int, str],
    logger: Optional[Any] = None,
    is_notebook: bool = False
):
    """Calculates and plots raw and normalized confusion matrices."""
    if not output_dir or not id2label:
        return

    try:
        num_classes = len(id2label)
        class_names = [id2label[i] for i in range(num_classes)]

        cm = calculate_confusion_matrix(preds=all_preds, labels=all_labels, num_classes=num_classes)

        # Plot non-normalized
        cm_path_raw = os.path.join(output_dir, "confusion_matrix_raw.png")
        plot_confusion_matrix(cm=cm, class_names=class_names, normalize=False, title='Confusion Matrix (Counts)', save_path=cm_path_raw)
        plt.close('all')
        log_or_print(f"Saved raw confusion matrix plot to {cm_path_raw}", logger, logging.INFO, is_notebook)

        # Plot normalized
        cm_path_norm = os.path.join(output_dir, "confusion_matrix_normalized.png")
        plot_confusion_matrix(cm=cm, class_names=class_names, normalize=True, title='Normalized Confusion Matrix (%)', save_path=cm_path_norm)
        plt.close('all')
        log_or_print(f"Saved normalized confusion matrix plot to {cm_path_norm}", logger, logging.INFO, is_notebook)

    except Exception as e_cm:
        log_or_print(f"Failed to generate confusion matrix plot: {e_cm}", logger, logging.WARNING, is_notebook)


# --- Main Evaluation Function ---

def evaluate_model(
    model: torch.nn.Module,
    eval_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: Optional[str] = None,
    id2label: Optional[Dict[int, str]] = None,
    analyze_errors: bool = False,
    logger: Optional[Any] = None,
    is_notebook: bool = False
) -> Dict[str, float]:
    """
    Evaluate the model and calculate metrics, optionally analyzing errors.

    Args:
        model: The model to evaluate
        eval_dataloader: DataLoader for evaluation data
        device: Device to evaluate on (cuda or cpu)
        output_dir: Directory to save visualizations
        id2label: Mapping from class indices to class names
        analyze_errors: Whether to perform detailed error analysis.
        logger: Optional logger for logging messages
        is_notebook: Whether running in a notebook environment

    Returns:
        Dictionary of evaluation metrics (including error stats if analyze_errors is True)
    """
    model = model.to(device)
    model.eval()

    # Run evaluation loop
    avg_loss, all_preds, all_labels = _run_evaluation_loop(
        model=model,
        eval_dataloader=eval_dataloader,
        device=device,
        logger=logger,
        is_notebook=is_notebook
    )

    # Calculate and log metrics
    metrics = _calculate_and_log_metrics(
        all_preds=all_preds,
        all_labels=all_labels,
        avg_loss=avg_loss,
        logger=logger,
        is_notebook=is_notebook
    )

    # Perform error analysis if requested
    if analyze_errors and output_dir and id2label:
        error_stats = _perform_error_analysis(
            all_preds=all_preds,
            all_labels=all_labels,
            output_dir=output_dir,
            id2label=id2label,
            logger=logger,
            is_notebook=is_notebook
        )
        metrics.update(error_stats)

    # Plot confusion matrices if possible
    if output_dir and id2label:
        _plot_confusion_matrices(
            all_preds=all_preds,
            all_labels=all_labels,
            output_dir=output_dir,
            id2label=id2label,
            logger=logger,
            is_notebook=is_notebook
        )

    return metrics


def run_prediction_pipeline(
    config: Config,
    image_paths: Union[str, List[str]],
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    batch_size: int = 1,
    visualize: bool = True,
    show_confidence: bool = False,
    show_class_activation_maps: bool = False,
    logger: Optional[Any] = None,
    is_notebook: bool = False
) -> Dict[str, Any]:
    """
    Run the complete TCD prediction pipeline.

    Args:
        config: Configuration object
        image_paths: Path(s) to input image(s). Can be a single string or a list of strings.
        model_path: Path to the saved model. If None, uses config["output_dir"]/final_model
        output_dir: Directory to save prediction visualizations. If None, uses config["output_dir"]/predictions
        batch_size: Batch size for prediction (primarily for multiple images)
        visualize: Whether to visualize the predictions
        show_confidence: Whether to visualize prediction confidence
        show_class_activation_maps: Whether to visualize class activation maps
        logger: Optional logger for logging messages
        is_notebook: Whether running in a notebook environment

    Returns:
        Dictionary with prediction results including segmentation masks and visualizations
    """
    # Set up paths
    if model_path is None:
        model_path = os.path.join(config["output_dir"], "final_model")

    if output_dir is None:
        output_dir = os.path.join(config["output_dir"], "predictions")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Log basic information
    log_msg = f"Starting prediction using model from {model_path}"
    log_or_print(log_msg, logger, logging.INFO, is_notebook)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_msg = f"Using device: {device}"
    log_or_print(log_msg, logger, logging.INFO, is_notebook)

    # Verify model path
    if not verify_checkpoint(model_path):
        error_msg = f"Invalid model checkpoint at {model_path}"
        log_or_print(error_msg, logger, logging.ERROR, is_notebook)
        raise ValueError(error_msg)

    # Load model using the new safe load_model_for_evaluation function
    try:
        log_or_print(f"Prediction: Loading model from {model_path} using load_model_for_evaluation.", logger, logging.INFO, is_notebook)
        
        # Import the improved model loading function
        from checkpoint import load_model_for_evaluation
        
        # Load model and image processor - handles all supported architectures
        model, _ = load_model_for_evaluation(
            model_path=model_path,
            config=config,  # Pass the current project config
            device=device,
            logger=logger
        )
        
        model.eval()

        log_msg = f"Model ({model.__class__.__name__}) loaded successfully from {model_path}"
        log_or_print(log_msg, logger, logging.INFO, is_notebook)
    except Exception as e:
        error_msg = f"Failed to load model from {model_path}: {str(e)}"
        log_or_print(error_msg, logger, logging.ERROR, is_notebook)
        raise

    # Load id2label/label2id mapping with fallback for non-HF models (e.g., DeepLabV3Wrapper)
    try:
        id2label = model.config.id2label  # type: ignore[attr-defined]
        label2id = model.config.label2id  # type: ignore[attr-defined]
    except Exception:
        # Fallback: read from checkpoint's config.json
        cfg_path = os.path.join(model_path, "config.json")
        with open(cfg_path, 'r') as f:
            _cfg_json = json.load(f)
        id2label = {int(k): v for k, v in _cfg_json.get('id2label', {}).items()}
        label2id = _cfg_json.get('label2id', {})

    # Create image processor with custom configuration to prevent automatic downsizing
    # This is the key fix for preventing segmentation outputs from appearing shrunken
    image_processor = SegformerImageProcessor(
        do_resize=False,  # Prevent automatic resizing which causes segmentation to be clustered
        do_rescale=True,  # Still normalize pixel values
        do_normalize=True  # Still perform ImageNet normalization
    )
    log_or_print("Using modified image processor configuration with do_resize=False to maintain original dimensions",
                logger, logging.INFO, is_notebook)

    # Handle single image input
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    # Load images
    images = []
    image_nps = []

    for image_path in image_paths:
        try:
            image = Image.open(image_path).convert("RGB")
            image_np = np.array(image)
            images.append(image)
            image_nps.append(image_np)
        except Exception as e:
            error_msg = f"Failed to load image {image_path}: {str(e)}"
            if logger:
                logger.error(error_msg)
            elif is_notebook:
                print(f"Error: {error_msg}")
            # Skip this image and continue with others
            continue

    if not images:
        error_msg = "No valid images found for prediction"
        log_or_print(error_msg, logger, logging.ERROR, is_notebook)
        raise ValueError(error_msg)

    # Process images in batches
    results = {
        "images": [],
        "segmentation_maps": [],
        "visualizations": [],
        "confidence_maps": []
    }

    # Get tiling parameters from config
    tile_size = config.get("inference_tile_size", None)
    overlap = config.get("inference_tile_overlap", 64)

    # --- Process Images ---
    for idx, (image_pil, image_np) in enumerate(zip(images, image_nps)):
        img_name = os.path.basename(image_paths[idx])
        base_name, _ = os.path.splitext(img_name)

        log_msg = f"Processing image {idx + 1}/{len(images)}: {img_name}"
        log_or_print(log_msg, logger, logging.INFO, is_notebook)

        # --- Tiled Inference ---
        if tile_size is not None and tile_size > 0:
            log_msg = f"Using tiled inference with tile_size={tile_size}, overlap={overlap}"
            log_or_print(log_msg, logger, logging.INFO, is_notebook)

            original_h, original_w = image_np.shape[:2]
            predicted_logits_tiles = {}

            # Generate and process tiles
            tile_generator = get_tiles(image_np, tile_size, overlap)

            # Use tqdm for progress tracking if not in notebook
            tile_iterator = tile_generator if is_notebook else tqdm(tile_generator, desc=f"Processing tiles for {img_name}")

            for y_start, x_start, tile_np in tile_iterator:
                # Preprocess tile
                # Note: Image processor expects PIL Image or list of PIL Images
                tile_pil = Image.fromarray(ensure_rgb(tile_np))
                inputs = image_processor(images=tile_pil, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Store original tile dimensions before processing
                original_tile_h, original_tile_w = tile_np.shape[:2]

                # Make prediction on tile
                with torch.no_grad():
                    outputs = model(**inputs)
                    # Get logits shape BEFORE any potential model downsampling
                    tile_logits = outputs.logits # Shape: (1, num_classes, tile_h, tile_w)

                    # If logits are smaller than the original tile, upsample them to match
                    if (tile_logits.shape[2] != original_tile_h or
                        tile_logits.shape[3] != original_tile_w):
                        # Log that we're upsampling the tile
                        log_or_print(
                            f"Upsampling tile logits from {tile_logits.shape[2]}x{tile_logits.shape[3]} to {original_tile_h}x{original_tile_w}",
                            logger, logging.DEBUG, is_notebook
                        )
                        # Upsample to match original tile dimensions using configured settings
                        interp_mode = config.get("interpolation_mode", "bilinear")
                        align_corners = config.get("interpolation_align_corners", False)
                        align_corners_param = align_corners if interp_mode != 'nearest' else None
                        tile_logits = F.interpolate(
                            tile_logits,
                            size=(original_tile_h, original_tile_w),
                            mode=interp_mode,
                            align_corners=align_corners_param
                        )

                # Convert logits to numpy and permute to HWC
                tile_logits_np = tile_logits.squeeze(0).permute(1, 2, 0).cpu().numpy()
                tile_h, tile_w, num_classes = tile_logits_np.shape

                # Log tile shape and coordinates
                log_or_print(
                    f"Tile at (y={y_start}, x={x_start}) - logits shape: {tile_logits_np.shape}",
                    logger, logging.DEBUG, is_notebook
                )

                # Pad tile if needed to (tile_size, tile_size, num_classes)
                pad_h = tile_size - tile_h
                pad_w = tile_size - tile_w
                if pad_h > 0 or pad_w > 0:
                    tile_logits_np = np.pad(
                        tile_logits_np,
                        ((0, pad_h), (0, pad_w), (0, 0)),
                        mode='constant',
                        constant_values=0
                    )
                    log_or_print(
                        f"Padded tile at (y={y_start}, x={x_start}) to shape: {tile_logits_np.shape}",
                        logger, logging.DEBUG, is_notebook
                    )

                predicted_logits_tiles[(y_start, x_start)] = tile_logits_np

            # Stitch predicted tiles (logits)
            stitched_logits_np = stitch_tiles(
                tiles=predicted_logits_tiles,
                target_shape=(original_h, original_w),
                tile_size=tile_size,
                overlap=overlap
            ) # Shape: (H, W, num_classes)

            # <<< Upsample stitched logits BEFORE argmax >>>
            stitched_logits_tensor = torch.from_numpy(stitched_logits_np).permute(2, 0, 1).unsqueeze(0).float().to(device) # B, C, H, W
            interp_mode = config.get("interpolation_mode", "bilinear")
            align_corners = config.get("interpolation_align_corners", False)
            align_corners_param = align_corners if interp_mode != 'nearest' else None
            upsampled_logits = F.interpolate(
                stitched_logits_tensor,
                size=(original_h, original_w),
                mode=interp_mode,
                align_corners=align_corners_param
            )

            # Get final prediction by taking argmax on upsampled logits
            pred_np = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy() # Shape: (original_h, original_w)

            # Calculate confidence map from upsampled logits if needed
            if show_confidence:
                 # Convert upsampled logits to probabilities
                 upsampled_probs = torch.softmax(upsampled_logits.squeeze(0), dim=0).cpu().numpy() # Shape: (num_classes, original_h, original_w)
                 confidence = np.max(upsampled_probs, axis=0) # Shape: (original_h, original_w)

                 # <<< Explicitly resize confidence map to original image dimensions >>>
                 # Note: Confidence is now calculated from already upsampled logits, but we keep this check just in case.
                 if confidence.shape != (original_h, original_w):
                     log_msg = f"Warning: Stitched confidence shape {confidence.shape} differs from original image shape {(original_h, original_w)}. Resizing confidence map..."
                     log_or_print(log_msg, logger, logging.WARNING, is_notebook)
                     conf_tensor = torch.from_numpy(confidence).unsqueeze(0).unsqueeze(0).float().to(device) # Move tensor to device
                     # Use configured settings for resizing confidence map (typically bilinear)
                     interp_mode_conf = config.get("interpolation_mode", "bilinear") # Use main mode, usually bilinear for continuous data
                     align_corners_conf = config.get("interpolation_align_corners", False)
                     align_corners_param_conf = align_corners_conf if interp_mode_conf != 'nearest' else None
                     resized_conf = F.interpolate(
                         conf_tensor,
                         size=(original_h, original_w),
                         mode=interp_mode_conf,
                         align_corners=align_corners_param_conf
                     )
                     confidence = resized_conf.squeeze().cpu().numpy() # Move back to CPU

        # --- Full Image Inference ---
        else:
            # Preprocess full image
            inputs = image_processor(images=image_pil, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Make prediction
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits # Shape: (1, num_classes, H, W)

            # <<< Upsample logits BEFORE argmax >>>
            original_h, original_w = image_np.shape[:2] # Get original dims
            interp_mode = config.get("interpolation_mode", "bilinear")
            align_corners = config.get("interpolation_align_corners", False)
            align_corners_param = align_corners if interp_mode != 'nearest' else None
            upsampled_logits = F.interpolate(
                logits,
                size=(original_h, original_w),
                mode=interp_mode,
                align_corners=align_corners_param
            )

            # Get final prediction by taking argmax on upsampled logits
            pred_np = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy() # Shape: (original_h, original_w)

            # Calculate confidence map from upsampled logits if needed
            if show_confidence:
                 # Convert upsampled logits to probabilities
                 upsampled_probs = torch.softmax(upsampled_logits.squeeze(0), dim=0).cpu().numpy() # Shape: (num_classes, original_h, original_w)
                 confidence = np.max(upsampled_probs, axis=0) # Shape: (original_h, original_w)

                 # <<< Explicitly resize confidence map to original image dimensions >>>
                 # Note: Confidence is now calculated from already upsampled logits, but we keep this check just in case.
                 if confidence.shape != (original_h, original_w):
                     log_msg = f"Warning: Full image confidence shape {confidence.shape} differs from original image shape {(original_h, original_w)}. Resizing confidence map..."
                     log_or_print(log_msg, logger, logging.WARNING, is_notebook)
                     conf_tensor = torch.from_numpy(confidence).unsqueeze(0).unsqueeze(0).float().to(device) # Move tensor to device
                     # Use configured settings for resizing confidence map (typically bilinear)
                     interp_mode_conf = config.get("interpolation_mode", "bilinear") # Use main mode, usually bilinear for continuous data
                     align_corners_conf = config.get("interpolation_align_corners", False)
                     align_corners_param_conf = align_corners_conf if interp_mode_conf != 'nearest' else None
                     resized_conf = F.interpolate(
                         conf_tensor,
                         size=(original_h, original_w),
                         mode=interp_mode_conf,
                         align_corners=align_corners_param_conf
                     )
                     confidence = resized_conf.squeeze().cpu().numpy() # Move back to CPU

        # --- Store Results and Visualize ---
        results["images"].append(image_np)
        results["segmentation_maps"].append(pred_np)

        # Create visualization
        if visualize:
            vis = visualize_segmentation(
                image=image_np,
                segmentation_map=pred_np,
                id2label=id2label,
                alpha=0.5
            )
            results["visualizations"].append(vis)

            # Save visualization - ensure proper size matching
            vis_path = os.path.join(output_dir, f"{base_name}_prediction.png")

            # Use exact figure dimensions to maintain aspect ratio
            img_height, img_width = image_np.shape[:2]
            dpi = 100  # Standard DPI
            figsize = (img_width*2/dpi, img_height/dpi)  # 2x width for side-by-side

            # Create the figure with fixed size
            fig = plt.figure(figsize=figsize, dpi=dpi)

            # Create the subplots with exact relative positions
            ax1 = fig.add_axes([0, 0, 0.5, 1])  # Left half - original image
            ax2 = fig.add_axes([0.5, 0, 0.5, 1])  # Right half - segmentation

            # Display images without interpolation to maintain pixel accuracy
            ax1.imshow(image_np, interpolation='nearest')
            ax1.set_title("Original Image")
            ax1.axis("off")

            ax2.imshow(vis, interpolation='nearest')
            ax2.set_title("Segmentation Prediction")
            ax2.axis("off")

            # Save without padding to maintain exact dimensions
            plt.savefig(vis_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
            plt.close(fig)

            log_msg = f"Saved prediction visualization to {vis_path}"
            log_or_print(log_msg, logger, logging.INFO, is_notebook)

            # --- Save predicted mask as PNG, resized to input image size ---
            try:
                mask_save_path = os.path.join(output_dir, f"{base_name}_prediction_mask.png")
                input_h, input_w = image_np.shape[:2]
                # Log original mask shape
                log_or_print(f"Original predicted mask shape: {pred_np.shape}, target: {(input_h, input_w)}", logger, logging.INFO, is_notebook)
                # Ensure mask dimensions
                pred_np_resized = ensure_mask_dimensions(pred_np, (input_h, input_w), logger=logger, context=f"{base_name}_mask", is_label_mask=True)
                # Log after resizing
                log_or_print(f"Final mask shape before saving: {pred_np_resized.shape}", logger, logging.INFO, is_notebook)
                # Save as mode 'L' - scale binary mask values (0,1) to full range (0,255) for visibility
                mask_img = Image.fromarray((pred_np_resized * 255).astype(np.uint8), mode='L')
                mask_img.save(mask_save_path)
                log_msg = f"Saved predicted mask (mode 'L', exact size) to {mask_save_path}"
                log_or_print(log_msg, logger, logging.INFO, is_notebook)
            except Exception as e:
                log_or_print(f"Failed to save resized predicted mask: {e}", logger, logging.WARNING, is_notebook)

        # Generate confidence visualization if requested
        if show_confidence:
            # Visualize confidence
            confidence_vis = plot_prediction_confidence(
                image=image_np,
                prediction=pred_np,
                confidence_map=confidence, # Calculated in either tiled or full image path
                class_names=list(id2label.values())
            )
            results["confidence_maps"].append(confidence_vis)

            # Save confidence visualization (matplotlib version, may have different size)
            conf_path = os.path.join(output_dir, f"{base_name}_confidence.png")
            Image.fromarray(confidence_vis).save(conf_path)
            log_msg = f"Saved confidence visualization to {conf_path}"
            log_or_print(log_msg, logger, logging.INFO, is_notebook)

            # --- Save raw confidence map as PNG with same dimensions as input image ---
            raw_conf_path = os.path.join(output_dir, f"{base_name}_confidence_raw.png")
            input_h, input_w = image_np.shape[:2]
            # Log original confidence shape
            log_or_print(f"Original confidence map shape: {confidence.shape}, target: {(input_h, input_w)}", logger, logging.INFO, is_notebook)
            # Ensure confidence map dimensions
            confidence_resized = ensure_mask_dimensions(confidence, (input_h, input_w), logger=logger, context=f"{base_name}_confidence", is_label_mask=False)
            # Log after resizing
            log_or_print(f"Final confidence map shape before saving: {confidence_resized.shape}", logger, logging.INFO, is_notebook)
            confidence_uint8 = (np.clip(confidence_resized, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(confidence_uint8).save(raw_conf_path)
            log_msg = f"Saved raw confidence map to {raw_conf_path}"
            log_or_print(log_msg, logger, logging.INFO, is_notebook)

            # --- Save overlay PNG (heatmap blended with input image, same size, no colorbar/axes) ---
            try:
                # Ensure image is uint8 and 3-channel
                img_for_overlay = image_np
                if img_for_overlay.dtype != np.uint8:
                    img_for_overlay = (np.clip(img_for_overlay, 0, 1) * 255).astype(np.uint8)
                if img_for_overlay.shape[2] != 3:
                    img_for_overlay = img_for_overlay[..., :3]
                conf_uint8 = (np.clip(confidence, 0, 1) * 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(conf_uint8, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(img_for_overlay, 0.6, heatmap, 0.4, 0)
                overlay_path = os.path.join(output_dir, f"{base_name}_confidence_overlay.png")
                Image.fromarray(overlay).save(overlay_path)
                log_msg = f"Saved confidence overlay (same size as input) to {overlay_path}"
                log_or_print(log_msg, logger, logging.INFO, is_notebook)
            except Exception as e:
                log_or_print(f"Failed to save confidence overlay: {e}", logger, logging.WARNING, is_notebook)

        # Generate class activation maps if requested
        # Note: CAM generation might be complex with tiling.
        # Current implementation assumes full image processing.
        if show_class_activation_maps and hasattr(model, 'segformer'):
            if tile_size is not None and tile_size > 0:
                 log_msg = "Warning: CAM generation with tiling is not directly supported. Generating CAM for the full image (might cause memory issues)."
                 log_or_print(log_msg, logger, logging.WARNING, is_notebook)

            try:
                # Define save path for CAM
                cam_path = os.path.join(output_dir, f"{base_name}_cam.png")

                # Lazy import to avoid hard dependency if function isn't available
                from visualization import plot_class_activation_map as _plot_cam
                _ = _plot_cam(
                    model=model,
                    image=image_pil,
                    image_processor=image_processor,
                    device=device,
                    save_path=cam_path
                )

                # Log saving (or attempt)
                log_msg = f"Saved class activation map to {cam_path}"
                log_or_print(log_msg, logger, logging.INFO, is_notebook)
            except Exception as e:
                error_msg = f"Failed to generate class activation map for {img_name}: {str(e)}"
                log_or_print(error_msg, logger, logging.WARNING, is_notebook)

    return results


def predict_image(
    model: torch.nn.Module,
    image_path: str,
    image_processor: Optional[SegformerImageProcessor] = None,
    device: Optional[torch.device] = None,
    id2label: Optional[Dict[int, str]] = None,
    return_confidence: bool = False,
    logger: Optional[Any] = None, # Added logger
    is_notebook: bool = False # Added is_notebook
) -> Dict[str, Any]:
    """
    Predict segmentation for a single image.

    Args:
        model: The model to use for prediction
        image_path: Path to the input image
        image_processor: Image processor for preprocessing
        device: Device to run inference on (cuda or cpu)
        id2label: Mapping from class indices to class names
        return_confidence: Whether to return confidence scores

    Returns:
        Dictionary with prediction results
    """
    # Set device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load image
    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image)

    # Create image processor if not provided
    if image_processor is None:
        # Use same configuration as in run_prediction_pipeline for consistency
        image_processor = SegformerImageProcessor(
            do_resize=False,  # Prevent automatic resizing
            do_rescale=True,  # Still normalize pixel values
            do_normalize=True  # Still perform ImageNet normalization
        )
        # Ensure logger is available for the log_or_print function
        effective_logger_pred_img = logger if logger is not None else logging.getLogger(__name__) # Fallback logger
        log_or_print("Using modified image processor configuration with do_resize=False to maintain original dimensions", effective_logger_pred_img, logging.INFO, is_notebook)


    # Get id2label from model if not provided (already handled by model.config.id2label above)
    if id2label is None and hasattr(model, 'config') and hasattr(model.config, 'id2label'):
         id2label = model.config.id2label # This line is now redundant but harmless

    # Preprocess image
    inputs = image_processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Get predicted segmentation map
    predicted_segmentation = logits.argmax(dim=1).squeeze().cpu().numpy()

    # Visualize prediction
    visualization = visualize_segmentation(
        image=image_np,
        segmentation_map=predicted_segmentation,
        id2label=id2label,
        alpha=0.5
    )

    # Prepare result
    result = {
        "image": image_np,
        "segmentation_map": predicted_segmentation,
        "visualization": visualization
    }

    # Add confidence information if requested
    if return_confidence:
        # Get softmax probabilities
        probs = torch.softmax(logits, dim=1).squeeze() # (NumClasses, H, W)

        # Get confidence as the maximum probability
        confidence = torch.max(probs, dim=0)[0].cpu().numpy() # (H, W)

        # Add to result
        result["confidence"] = confidence

        # Create confidence visualization (using the existing 3-panel plot for single image prediction)
        # Note: This uses plot_prediction_confidence, not the new 4-panel one.
        try:
            confidence_vis_array = plot_prediction_confidence(
                image=image_np,
                prediction=predicted_segmentation,
                confidence_map=confidence,
                class_names=list(id2label.values())
            )
            result["confidence_visualization"] = confidence_vis_array
        except Exception as e:
             # Use the standardized logging function instead of direct logging call
             log_or_print(f"Failed to generate confidence visualization for {image_path}: {e}", logger, logging.WARNING, is_notebook) # Pass logger and is_notebook
             result["confidence_visualization"] = None # Indicate failure

    return result
