#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Statistical validation utilities for journal extension.

This module provides tools for statistical significance testing and confidence
interval estimation to ensure journal-level rigor.

Functions:
    - run_friedman_test: Non-parametric test for comparing multiple algorithms
    - bootstrap_confidence_interval: Non-parametric confidence interval estimation
    - compute_all_metrics_with_ci: Compute metrics with bootstrap CIs
    - paired_ttest: Paired t-test for comparing two algorithms
"""

import sys
import os
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from scipy.stats import friedmanchisquare

# Add Core to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Core'))

from utils import get_logger

logger = get_logger()


def run_friedman_test(results_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Run Friedman test on algorithm performance matrix.
    
    The Friedman test is a non-parametric alternative to repeated-measures ANOVA
    for comparing multiple algorithms across multiple datasets or folds.
    
    Args:
        results_matrix: Shape (n_datasets, n_algorithms)
                       Each row is a dataset/fold, each column is an algorithm
    
    Returns:
        Dictionary with statistic, p-value, and interpretation
    """
    if results_matrix.shape[0] < 2:
        logger.warning("Friedman test requires at least 2 datasets/folds")
        return {
            "friedman_statistic": None,
            "p_value": None,
            "significant": False,
            "interpretation": "Insufficient data for Friedman test"
        }
    
    if results_matrix.shape[1] < 3:
        logger.warning("Friedman test is designed for 3+ algorithms. Consider paired t-test for 2 algorithms.")
    
    try:
        stat, p_value = friedmanchisquare(*results_matrix.T)
        
        ranks = np.zeros_like(results_matrix, dtype=float)
        for i in range(results_matrix.shape[0]):
            ranks[i] = len(results_matrix[i]) + 1 - np.argsort(np.argsort(results_matrix[i])) - 1
        
        avg_ranks = np.mean(ranks, axis=0)
        
        return {
            "friedman_statistic": float(stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "average_ranks": avg_ranks.tolist(),
            "interpretation": "Reject H0: algorithms differ significantly" if p_value < 0.05 
                             else "Fail to reject H0: no significant difference detected"
        }
    except Exception as e:
        logger.error(f"Friedman test failed: {e}")
        return {
            "friedman_statistic": None,
            "p_value": None,
            "significant": False,
            "interpretation": f"Friedman test failed: {e}"
        }


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_fn: Callable = np.mean,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """
    Calculate bootstrap confidence interval for any statistic.
    
    Uses resampling with replacement to estimate the sampling distribution
    of a statistic and compute confidence intervals.
    
    Args:
        data: Array of metric values (e.g., per-image IoU scores)
        statistic_fn: Function to compute (default: mean)
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level (default: 0.95 for 95% CI)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with point estimate and CI bounds
    """
    if len(data) == 0:
        return {
            "point_estimate": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "confidence_level": confidence_level,
            "n_bootstrap": n_bootstrap,
            "formatted": "N/A (empty data)"
        }
    
    np.random.seed(seed)
    n = len(data)
    
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        resample_indices = np.random.randint(0, n, size=n)
        resample = data[resample_indices]
        bootstrap_stats.append(statistic_fn(resample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_lower = float(np.percentile(bootstrap_stats, lower_percentile))
    ci_upper = float(np.percentile(bootstrap_stats, upper_percentile))
    point_estimate = float(statistic_fn(data))
    
    return {
        "point_estimate": point_estimate,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap,
        "formatted": f"{point_estimate:.4f} ({confidence_level*100:.0f}% CI: {ci_lower:.4f} - {ci_upper:.4f})"
    }


def compute_all_metrics_with_ci(
    predictions: List[np.ndarray],
    ground_truths: List[np.ndarray],
    metric_functions: Optional[Dict[str, Callable]] = None,
    n_bootstrap: int = 1000
) -> Dict[str, Dict]:
    """
    Compute IoU, B-IoU, F1 with bootstrap CIs for all test images.
    
    Args:
        predictions: List of prediction arrays
        ground_truths: List of ground truth arrays
        metric_functions: Optional dict of metric name -> function(pred, gt) -> float
        n_bootstrap: Number of bootstrap iterations
        
    Returns:
        Dictionary with metrics and their confidence intervals
    """
    from metrics import calculate_metrics, calculate_boundary_iou
    
    per_image_metrics = {
        "iou": [],
        "boundary_iou": [],
        "f1_score": [],
        "precision": [],
        "recall": []
    }
    
    for pred, gt in zip(predictions, ground_truths):
        metrics = calculate_metrics(pred, gt)
        
        per_image_metrics["iou"].append(metrics.get("iou_class_1", 0.0))
        per_image_metrics["f1_score"].append(metrics.get("f1_score_class_1", 0.0))
        per_image_metrics["precision"].append(metrics.get("precision_class_1", 0.0))
        per_image_metrics["recall"].append(metrics.get("recall_class_1", 0.0))
        
        biou = calculate_boundary_iou(pred, gt, dilation_ratio=0.02)
        per_image_metrics["boundary_iou"].append(biou.get("boundary_iou_class_1", 0.0))
    
    results = {}
    for metric_name, values in per_image_metrics.items():
        if values:
            results[metric_name] = bootstrap_confidence_interval(
                np.array(values),
                n_bootstrap=n_bootstrap
            )
    
    if metric_functions:
        for name, func in metric_functions.items():
            custom_values = []
            for pred, gt in zip(predictions, ground_truths):
                try:
                    custom_values.append(func(pred, gt))
                except Exception as e:
                    logger.warning(f"Custom metric {name} failed: {e}")
            if custom_values:
                results[name] = bootstrap_confidence_interval(
                    np.array(custom_values),
                    n_bootstrap=n_bootstrap
                )
    
    return results


def paired_ttest(
    scores_a: np.ndarray,
    scores_b: np.ndarray
) -> Dict[str, Any]:
    """
    Paired t-test for comparing two algorithms.
    
    Use this instead of Friedman test when comparing exactly 2 methods.
    
    Args:
        scores_a: Scores for algorithm A
        scores_b: Scores for algorithm B
        
    Returns:
        Dictionary with t-statistic, p-value, and interpretation
    """
    from scipy.stats import ttest_rel
    
    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have the same length")
    
    if len(scores_a) < 2:
        return {
            "t_statistic": None,
            "p_value": None,
            "significant": False,
            "interpretation": "Insufficient data for paired t-test"
        }
    
    t_stat, p_value = ttest_rel(scores_a, scores_b)
    
    mean_diff = np.mean(scores_a - scores_b)
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "mean_difference": float(mean_diff),
        "a_better": mean_diff > 0 and p_value < 0.05,
        "b_better": mean_diff < 0 and p_value < 0.05,
        "interpretation": f"Mean difference: {mean_diff:.4f}. " +
                         ("Significant difference." if p_value < 0.05 else "No significant difference.")
    }

