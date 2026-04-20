#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Custom loss functions for journal extension.

This module provides advanced loss functions for semantic segmentation,
including boundary-aware and distance-based losses for loss function ablation studies.

Classes:
    - BoundaryLoss: Boundary loss from Kervadec et al.
    - HausdorffDistanceLoss: Differentiable approximation of Hausdorff distance
    - DiceLoss: Standard Dice loss for overlap optimization
    - CombinedLoss: Flexible combination of multiple losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Union, Tuple
from scipy.ndimage import distance_transform_edt

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Core'))

from utils import get_logger

logger = get_logger()


def compute_distance_map(mask: np.ndarray) -> np.ndarray:
    """
    Compute signed distance transform for a binary mask.
    
    Negative values inside the object, positive values outside.
    
    Args:
        mask: Binary mask array (H, W) or batch (N, H, W)
        
    Returns:
        Signed distance map with same shape as input
    """
    if mask.ndim == 2:
        mask = mask[np.newaxis, ...]
        squeeze = True
    else:
        squeeze = False
    
    distance_maps = np.zeros_like(mask, dtype=np.float32)
    
    for i in range(mask.shape[0]):
        m = mask[i].astype(bool)
        pos_dist = distance_transform_edt(m)
        neg_dist = distance_transform_edt(~m)
        distance_maps[i] = neg_dist - pos_dist
    
    if squeeze:
        distance_maps = distance_maps[0]
    
    return distance_maps


def compute_distance_map_batch(masks: torch.Tensor) -> torch.Tensor:
    """
    Compute distance maps for a batch of masks.
    
    Args:
        masks: Binary masks (N, H, W) or (N, C, H, W)
        
    Returns:
        Distance maps as torch tensor on same device as input
    """
    device = masks.device
    masks_np = masks.detach().cpu().numpy()
    
    if masks_np.ndim == 4:
        N, C, H, W = masks_np.shape
        distance_maps = np.zeros((N, C, H, W), dtype=np.float32)
        for n in range(N):
            for c in range(C):
                distance_maps[n, c] = compute_distance_map(masks_np[n, c])
    else:
        distance_maps = compute_distance_map(masks_np)
    
    return torch.from_numpy(distance_maps).to(device)


class BoundaryLoss(nn.Module):
    """
    Boundary loss from Kervadec et al.
    
    Loss = ∫ φ_G(q) · s_θ(q) dq
    
    Where φ_G is the signed distance map and s_θ is the softmax output.
    This loss penalizes predictions that are far from the ground truth boundary.
    
    Reference:
        Kervadec et al., "Boundary loss for highly unbalanced segmentation", MIDL 2019
    """
    
    def __init__(self, include_background: bool = False):
        """
        Initialize BoundaryLoss.
        
        Args:
            include_background: Whether to include background class in loss computation
        """
        super().__init__()
        self.include_background = include_background
    
    def forward(
        self, 
        softmax_output: torch.Tensor, 
        target: torch.Tensor,
        distance_maps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute boundary loss.
        
        Args:
            softmax_output: (N, C, H, W) softmax probabilities
            target: (N, H, W) ground truth labels or (N, C, H, W) one-hot encoded
            distance_maps: (N, C, H, W) precomputed distance maps (optional)
            
        Returns:
            Scalar loss value
        """
        N, C, H, W = softmax_output.shape
        
        if target.ndim == 3:
            target_onehot = F.one_hot(target.long(), num_classes=C)
            target_onehot = target_onehot.permute(0, 3, 1, 2).float()
        else:
            target_onehot = target.float()
        
        if distance_maps is None:
            distance_maps = compute_distance_map_batch(target_onehot)
        
        start_idx = 0 if self.include_background else 1
        
        loss = torch.einsum(
            'nchw,nchw->nc', 
            softmax_output[:, start_idx:], 
            distance_maps[:, start_idx:]
        )
        
        return loss.mean()


class HausdorffDistanceLoss(nn.Module):
    """
    Differentiable approximation of Hausdorff distance loss.
    
    Uses distance transforms to approximate HD in a differentiable manner.
    Minimizes the maximum distance between prediction and ground truth boundaries.
    
    Reference:
        Karimi et al., "Reducing the Hausdorff Distance in Medical Image Segmentation", 2019
    """
    
    def __init__(self, alpha: float = 2.0, include_background: bool = False):
        """
        Initialize HausdorffDistanceLoss.
        
        Args:
            alpha: Exponent controlling sharpness of approximation (higher = sharper)
            include_background: Whether to include background class
        """
        super().__init__()
        self.alpha = alpha
        self.include_background = include_background
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Hausdorff distance loss.
        
        Args:
            pred: (N, C, H, W) predicted probabilities (softmax output)
            target: (N, H, W) ground truth labels or (N, C, H, W) one-hot
            
        Returns:
            Scalar loss value
        """
        N, C, H, W = pred.shape
        
        if target.ndim == 3:
            target_onehot = F.one_hot(target.long(), num_classes=C)
            target_onehot = target_onehot.permute(0, 3, 1, 2).float()
        else:
            target_onehot = target.float()
        
        target_np = target_onehot.detach().cpu().numpy()
        
        start_idx = 0 if self.include_background else 1
        batch_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        for n in range(N):
            for c in range(start_idx, C):
                gt_mask = target_np[n, c]
                
                dt_from_gt = distance_transform_edt(gt_mask == 0)
                dt_from_gt = torch.from_numpy(dt_from_gt).to(pred.device).float()
                
                pred_c = pred[n, c]
                weighted_pred = pred_c * (dt_from_gt ** self.alpha)
                batch_loss = batch_loss + weighted_pred.mean()
                
                dt_from_pred = distance_transform_edt(
                    (pred[n, c].detach().cpu().numpy() < 0.5).astype(np.float32)
                )
                dt_from_pred = torch.from_numpy(dt_from_pred).to(pred.device).float()
                
                gt_c = target_onehot[n, c]
                weighted_gt = gt_c * (dt_from_pred ** self.alpha)
                batch_loss = batch_loss + weighted_gt.mean()
        
        num_classes = C - start_idx
        return batch_loss / (2 * N * num_classes)


class DiceLoss(nn.Module):
    """
    Dice loss for semantic segmentation.
    
    Optimizes the Dice coefficient (F1-score) directly.
    Effective for handling class imbalance.
    """
    
    def __init__(
        self, 
        smooth: float = 1e-6, 
        include_background: bool = False,
        reduction: str = 'mean'
    ):
        """
        Initialize DiceLoss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
            include_background: Whether to include background class
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background
        self.reduction = reduction
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            pred: (N, C, H, W) predicted logits or probabilities
            target: (N, H, W) ground truth labels
            
        Returns:
            Scalar loss value (1 - Dice coefficient)
        """
        N, C, H, W = pred.shape
        
        if pred.requires_grad:
            pred_soft = F.softmax(pred, dim=1)
        else:
            pred_soft = pred
        
        if target.ndim == 3:
            target_onehot = F.one_hot(target.long(), num_classes=C)
            target_onehot = target_onehot.permute(0, 3, 1, 2).float()
        else:
            target_onehot = target.float()
        
        start_idx = 0 if self.include_background else 1
        
        pred_flat = pred_soft[:, start_idx:].contiguous().view(N, -1)
        target_flat = target_onehot[:, start_idx:].contiguous().view(N, -1)
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        cardinality = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice_score
        
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        else:
            return dice_loss


class CombinedLoss(nn.Module):
    """
    Flexible combination of multiple loss functions.
    
    Allows weighted combination of CE, Dice, Boundary, and Hausdorff losses.
    """
    
    def __init__(
        self,
        losses: Dict[str, nn.Module],
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize CombinedLoss.
        
        Args:
            losses: Dictionary of loss name -> loss module
            weights: Dictionary of loss name -> weight (default: equal weights)
        """
        super().__init__()
        self.losses = nn.ModuleDict(losses)
        
        if weights is None:
            weights = {name: 1.0 for name in losses.keys()}
        self.weights = weights
        
        logger.info(f"CombinedLoss initialized with losses: {list(losses.keys())}, weights: {weights}")
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted logits or probabilities
            target: Ground truth labels
            **kwargs: Additional arguments for specific losses
            
        Returns:
            Tuple of (total_loss, dict of individual losses)
        """
        individual_losses = {}
        total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        for name, loss_fn in self.losses.items():
            weight = self.weights.get(name, 1.0)
            
            if isinstance(loss_fn, BoundaryLoss):
                pred_soft = F.softmax(pred, dim=1)
                loss_value = loss_fn(pred_soft, target, kwargs.get('distance_maps'))
            elif isinstance(loss_fn, HausdorffDistanceLoss):
                pred_soft = F.softmax(pred, dim=1)
                loss_value = loss_fn(pred_soft, target)
            elif isinstance(loss_fn, nn.CrossEntropyLoss):
                loss_value = loss_fn(pred, target)
            else:
                loss_value = loss_fn(pred, target)
            
            individual_losses[name] = loss_value
            total_loss = total_loss + weight * loss_value
        
        return total_loss, individual_losses


def create_loss_function(
    loss_type: str,
    loss_params: Optional[Dict[str, Any]] = None,
    ignore_index: int = 255
) -> nn.Module:
    """
    Factory function to create loss functions based on configuration.
    
    Args:
        loss_type: Type of loss ('cross_entropy', 'dice', 'boundary', 'hausdorff', 'combined')
        loss_params: Parameters for the loss function
        ignore_index: Index to ignore in loss computation
        
    Returns:
        Loss function module
    """
    if loss_params is None:
        loss_params = {}
    
    logger.info(f"Creating loss function: {loss_type} with params: {loss_params}")
    
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(
            ignore_index=ignore_index
        )
    
    elif loss_type == 'dice':
        return DiceLoss(
            smooth=loss_params.get('smooth', 1e-6),
            include_background=loss_params.get('include_background', False)
        )
    
    elif loss_type == 'boundary':
        return BoundaryLoss(
            include_background=loss_params.get('include_background', False)
        )
    
    elif loss_type == 'hausdorff':
        return HausdorffDistanceLoss(
            alpha=loss_params.get('alpha', 2.0),
            include_background=loss_params.get('include_background', False)
        )
    
    elif loss_type == 'dice_ce':
        ce_weight = loss_params.get('ce_weight', 0.5)
        dice_weight = loss_params.get('dice_weight', 0.5)
        
        return CombinedLoss(
            losses={
                'cross_entropy': nn.CrossEntropyLoss(
                    ignore_index=ignore_index
                ),
                'dice': DiceLoss(
                    smooth=loss_params.get('smooth', 1e-6),
                    include_background=loss_params.get('include_background', False)
                )
            },
            weights={
                'cross_entropy': ce_weight,
                'dice': dice_weight
            }
        )
    
    elif loss_type == 'boundary_ce':
        ce_weight = loss_params.get('ce_weight', 0.5)
        boundary_weight = loss_params.get('boundary_weight', 0.5)
        
        return CombinedLoss(
            losses={
                'cross_entropy': nn.CrossEntropyLoss(
                    ignore_index=ignore_index
                ),
                'boundary': BoundaryLoss(
                    include_background=loss_params.get('include_background', False)
                )
            },
            weights={
                'cross_entropy': ce_weight,
                'boundary': boundary_weight
            }
        )
    
    elif loss_type == 'combined':
        losses = {}
        weights = {}
        
        if loss_params.get('use_ce', True):
            losses['cross_entropy'] = nn.CrossEntropyLoss(
                ignore_index=ignore_index
            )
            weights['cross_entropy'] = loss_params.get('ce_weight', 1.0)
        
        if loss_params.get('use_dice', False):
            losses['dice'] = DiceLoss(
                smooth=loss_params.get('smooth', 1e-6),
                include_background=loss_params.get('include_background', False)
            )
            weights['dice'] = loss_params.get('dice_weight', 1.0)
        
        if loss_params.get('use_boundary', False):
            losses['boundary'] = BoundaryLoss(
                include_background=loss_params.get('include_background', False)
            )
            weights['boundary'] = loss_params.get('boundary_weight', 1.0)
        
        if loss_params.get('use_hausdorff', False):
            losses['hausdorff'] = HausdorffDistanceLoss(
                alpha=loss_params.get('alpha', 2.0),
                include_background=loss_params.get('include_background', False)
            )
            weights['hausdorff'] = loss_params.get('hausdorff_weight', 1.0)
        
        if not losses:
            logger.warning("No losses specified in combined loss, defaulting to CrossEntropyLoss")
            return nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        return CombinedLoss(losses=losses, weights=weights)
    
    else:
        logger.warning(f"Unknown loss type '{loss_type}', defaulting to CrossEntropyLoss")
        return nn.CrossEntropyLoss(
            ignore_index=ignore_index
        )
