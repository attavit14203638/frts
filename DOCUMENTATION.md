# Technical Documentation

This document provides implementation-level notes that complement the [README](README.md). For a high-level overview, usage, and citation info, see the README. For a log of changes across papers, see the [CHANGELOG](CHANGELOG.md).

## Table of Contents

1. [Core Module Layout](#1-core-module-layout)
2. [Full-Resolution Supervision Mechanics](#2-full-resolution-supervision-mechanics)
3. [Data Processing](#3-data-processing)
4. [Training Pipeline](#4-training-pipeline)
5. [Optimization Notes](#5-optimization-notes)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Configuration Management](#7-configuration-management)
8. [Error Handling](#8-error-handling)
9. [Extending with New Architectures or Datasets](#9-extending-with-new-architectures-or-datasets)

---

## 1. Core Module Layout

| Module | Purpose |
|---|---|
| `Core/pipeline.py` | Centralised train / evaluate / predict entry points |
| `Core/config.py` | `Config` dataclass with validation, JSON (de)serialisation |
| `Core/dataset.py` | HuggingFace dataset loading, augmentation, DataLoader wiring |
| `Core/model.py` | Conference-era architecture wrappers (SegFormer, PSPNet, SETR) |
| `Core/metrics.py` | IoU, F1, precision, recall, pixel accuracy, Boundary IoU |
| `Core/checkpoint.py` | Checkpoint save/restore with config metadata |
| `Core/image_utils.py` | Tiling, stitching, image I/O helpers |
| `Core/exceptions.py` | Custom exception hierarchy |
| `Core/utils.py` | Logger setup and small shared utilities |
| `Core/main.py` | CLI entry point (subcommands: `train`, `predict`, `evaluate`) |
| `Extended/new_architectures.py` | Journal-era architectures: DeepLabV3, UperNet+Swin, OneFormer |
| `Extended/datasets_loader.py` | External eval datasets: SelvaMask, BAMFORESTS, SavannaTreeAI, Quebec |
| `Extended/statistical_validation.py` | Friedman test and bootstrap 95% confidence intervals |
| `Extended/loss_functions.py` | Loss implementations (cross-entropy default) |
| `Extended/tensorboard_utils.py` | TensorBoard log parsing and plotting |
| `Extended/visualization.py` | Figures used in the manuscript |
| `Extended/inspect_dataset.py` | Standalone dataset sanity-check utility |

`Extended/` modules import from `Core/` via `sys.path.insert` at the top of each file. Run scripts from the repository root or set `PYTHONPATH=Core:Extended` for reliable resolution.

---

## 2. Full-Resolution Supervision Mechanics

The central modification is captured in a single function: before computing loss, the model's reduced-resolution logits are bilinearly upsampled to match ground-truth dimensions. Given logits $L \in \mathbb{R}^{C \times H/R \times W/R}$ and label $Y \in \{0, \ldots, C-1\}^{H \times W}$:

```python
if config.train_time_upsample:
    logits_full = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    loss = F.cross_entropy(logits_full, Y)
else:
    Y_reduced = F.interpolate(Y.unsqueeze(1).float(), size=logits.shape[-2:],
                              mode="nearest").squeeze(1).long()
    loss = F.cross_entropy(logits, Y_reduced)
```

**Design invariants:**

- Upsampling is applied during training only; the inference path is unchanged.
- No new learnable parameters are introduced.
- Supported on any encoder-decoder model whose decoder outputs logits at a reduced spatial resolution.
- The flag `train_time_upsample` (in `config.py`) controls which branch runs.

---

## 3. Data Processing

The primary dataset is `restor/tcd` (OAM-TCD) via HuggingFace Datasets.

1. **Loading and shuffling.** `load_and_shuffle_dataset` seeds a deterministic shuffle so that all pipeline components see the same order.
2. **Preprocessing.** Images are normalised to ImageNet statistics and resized (default crop 1024×1024). Grayscale inputs are auto-converted to RGB.
3. **Mask handling.** Annotations under the `annotation` field are converted to binary masks (0 = background, 1 = tree crown). Multi-channel annotations are reduced deterministically. Invalid samples are logged and skipped.
4. **DataLoader.** Standard PyTorch `DataLoader` with `pin_memory`, `persistent_workers`, and `prefetch_factor` configurable via `Config`.

External evaluation datasets (SelvaMask, BAMFORESTS, SavannaTreeAI, Quebec) are loaded via dataset-specific classes in `Extended/datasets_loader.py`. All loaders return `{'image': PIL.Image, 'mask': PIL.Image}` dictionaries for a uniform downstream interface.

---

## 4. Training Pipeline

Centralised in `Core/pipeline.run_training_pipeline`. Responsibilities:

- Build model, dataset, optimizer, scheduler from a single `Config`.
- Run forward/backward with gradient accumulation and AMP autocast.
- Periodic validation and checkpoint saving (best-on-B-IoU by default).
- TensorBoard logging under `outputs/<run>/tensorboard/`.

The multi-seed protocol used in the journal paper is implemented as three independent invocations with seeds `{42, 123, 2024}`. Each seed controls weight initialisation, data shuffling, and stochastic regularisation.

---

## 5. Optimization Notes

- **Mixed precision** (FP16 autocast + GradScaler). Enabled via `config.mixed_precision = True`. Reduces memory substantially on V100-class GPUs; required for UperNet+Swin and OneFormer at 1024×1024 crops.
- **Gradient accumulation** supports effective batch sizes larger than GPU memory allows. Journal paper uses `per_device_batch_size = 2` with `accumulation_steps = 4` (effective batch = 8).
- **Learning-rate scheduling**: cosine annealing with warm restarts, 10% warmup ratio over 50 epochs (journal default). Linear warmup + linear decay is retained as an alternative.
- **Gradient checkpointing** (`config.gradient_checkpointing = True`) trades compute for memory. Relevant for large-backbone architectures (OneFormer Swin-L).
- **DataLoader perf knobs**: `dataloader_pin_memory`, `dataloader_persistent_workers`, `dataloader_prefetch_factor` all exposed in `Config`.

---

## 6. Evaluation Metrics

Computed in `Core/metrics.py`:

- **IoU** (Intersection over Union), per-class and mean.
- **F1 / Dice coefficient.**
- **Precision and recall** (for the foreground class).
- **Pixel accuracy.**
- **Boundary IoU (B-IoU)** — primary metric for the journal paper. Dilation radius defaults to `d = 0.02 * sqrt(H^2 + W^2)` following Cheng et al. (2021). B-IoU is computed as the IoU of the two contour bands $(G_d \cap G)$ and $(P_d \cap P)$.

Statistical validation (`Extended/statistical_validation.py`) adds:

- **Friedman test** (`scipy.stats.friedmanchisquare`) for cross-architecture comparison over seeds.
- **Bootstrap 95% confidence intervals** using the percentile method (default $B = 10\,000$ resamples).

---

## 7. Configuration Management

`Core/config.py` exposes the `Config` dataclass. Conventions:

- Every CLI flag corresponds to a `Config` field.
- `Config.save(path)` and `Config.load(path)` round-trip via JSON so that each run directory contains `effective_train_config.json` for reproducibility.
- Invalid combinations (e.g. `mixed_precision=True` on CPU) raise `ConfigurationError` at construction time.

---

## 8. Error Handling

Custom exceptions in `Core/exceptions.py` form a hierarchy rooted at `TCDError`. Subclasses include `DatasetError`, `InvalidSampleError`, `EmptyMaskError`, `ShapeMismatchError`, `ClassImbalanceError`, and `ConfigurationError`. Invalid samples are skipped rather than aborting training; shape/dtype mismatches at model I/O boundaries raise immediately with contextual information.

---

## 9. Extending with New Architectures or Datasets

### Add a new architecture

1. Create a wrapper class in `Extended/new_architectures.py` (or a new module) exposing:
   - `forward(image) -> logits` with logits at a known reduction factor $R$.
   - Any architecture-specific init arguments.
2. Register the wrapper in the factory function in `Core/model.py`.
3. Ensure the wrapper respects `config.train_time_upsample` by either (a) returning raw reduced-resolution logits and letting `pipeline.py` upsample, or (b) short-circuiting to full-resolution when the flag is set. Approach (a) is preferred for uniformity.

### Add a new external dataset

1. Create a new loader class in `Extended/datasets_loader.py` subclassing the base abstract class.
2. Implement the `__iter__` (or `__getitem__` + `__len__`) protocol returning `{'image': PIL.Image, 'mask': PIL.Image}`.
3. Normalise label polarity (tree = 1, background = 0).
4. Register the dataset in the loader factory so it can be referenced by name from the CLI.

---

For paper-facing claims, empirical results, and citations, see the journal manuscript (see the [README](README.md) for citation details).
