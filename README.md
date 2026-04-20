# Full-Resolution Training Supervision for Tree Crown Delineation

This repository hosts the reference implementation for two related papers on boundary-accurate tree crown delineation (TCD) in aerial imagery:

1. **BARE** (AusDM 2025, conference): "Boundary-Aware with Resolution Enhancement for Tree Crown Delineation".
2. **FRTS** (Journal of Big Data, 2026, journal extension): "Full-Resolution Training Supervision: Architecture-Agnostic Boundary Enhancement for Tree Crown Delineation".

The journal paper significantly extends the conference paper with two additional architectures (five in total), multi-seed statistical validation, bootstrap confidence intervals, cross-dataset zero-shot evaluation on four external datasets, and a corrected SETR baseline (see [CHANGELOG](CHANGELOG.md) for details).

> **Note on repository history.** This repository was previously named `bare`; it was renamed to `frts` when the journal extension was released. Old URLs redirect automatically. Historical snapshots are preserved as git tags (`paper/ausdm2025`, `paper/jbd2026`).

## Key Contributions

- **Full-Resolution Training Supervision** (FRTS): a training-time-only modification that bilinearly upsamples decoder logits to ground-truth resolution for loss computation. No parameters added, no architectural changes, no inference overhead.
- **Architecture-agnostic validation** across five paradigms: CNN-based (DeepLabV3), pure transformer (SETR), hierarchical transformer (SegFormer), hybrid transformer (UperNet + Swin), and query-based (OneFormer).
- **Boundary IoU (B-IoU)** established as a complementary evaluation metric for TCD, with consistent 20-30 percentage-point gaps exposed between standard IoU and B-IoU across architectures.
- **Statistical rigor**: three-seed experiments, Friedman tests, and bootstrap 95% confidence intervals on all reported architecture comparisons.
- **Cross-dataset evaluation**: zero-shot transfer on SelvaMask, BAMFORESTS, SavannaTreeAI, and Quebec datasets.

## Repository Layout

```
Core/           # Original conference codebase (BARE, AusDM 2025):
                # SegFormer, PSPNet, SETR with full-resolution supervision
Extended/       # Journal extensions (FRTS, JBD 2026):
                #   new_architectures.py    - DeepLabV3, UperNet+Swin, OneFormer
                #   datasets_loader.py      - SelvaMask, BAMFORESTS, SavannaTreeAI, Quebec loaders
                #   statistical_validation.py - Friedman test and bootstrap CIs
                #   loss_functions.py       - loss functions (cross-entropy default)
                #   visualization.py        - figures used in the manuscript
                #   tensorboard_utils.py    - training diagnostics
Notebook/       # Interactive training and analysis notebooks
```

## Getting Started

### Installation

```bash
git clone https://github.com/attavit14203638/frts.git
cd frts
pip install -r Core/requirements.txt
```

### Training

```bash
# Baseline training (no full-resolution supervision)
python Core/main.py train \
    --architecture segformer \
    --model_name nvidia/mit-b5 \
    --dataset_name restor/tcd \
    --output_dir ./outputs/segformer_baseline \
    --num_epochs 50 \
    --train_time_upsample False

# Full-resolution supervision
python Core/main.py train \
    --architecture segformer \
    --model_name nvidia/mit-b5 \
    --dataset_name restor/tcd \
    --output_dir ./outputs/segformer_fullres \
    --num_epochs 50 \
    --train_time_upsample True
```

See `python Core/main.py train --help` for the full argument list.

### Evaluation

```bash
python Core/main.py evaluate \
    --config_path ./outputs/segformer_fullres/effective_train_config.json \
    --model_path ./outputs/segformer_fullres/final_model \
    --output_dir ./eval/segformer_fullres
```

### Cross-Dataset Zero-Shot Evaluation

Loaders for the four external datasets are in `Extended/datasets_loader.py`. See the journal paper (Section 5.2) for the evaluation protocol.

## Supported Architectures

Ordered by architecture family (CNN, Hybrid, Transformer) and then by reduction factor R within each family.

| Architecture | Backbone | Decoder | R (reduction) | Params | Family | Source |
|---|---|---|---|---|---|---|
| DeepLabV3 | ResNet-50 | ASPP + linear | 8 | ~39M | CNN | Extended |
| PSPNet | ResNet-50 | PSP | 8 | ~49M | CNN | Core (conference only) |
| UperNet+Swin | Swin-Base | UperNet (PPM+FPN) | 4 | ~110M | Hybrid | Extended |
| OneFormer | Swin-Large | Mask Transformer | 4 | ~220M | Hybrid | Extended |
| SegFormer | MiT-B5 | All-MLP | 4 | ~82M | Transformer | Core |
| SETR | ViT-Base | PUP | 16 | ~90M | Transformer | Core |

PSPNet is retained for reproducibility of the AusDM 2025 paper but is not part of the journal evaluation.

## Papers

### Journal (primary)

```bibtex
@article{wilaiwongsakul_frts_2026,
  author  = {Wilaiwongsakul, Attavit and Liang, Bin and Zheng, Bryan and Chen, Fang},
  title   = {Full-Resolution Training Supervision: Architecture-Agnostic Boundary Enhancement for Tree Crown Delineation},
  journal = {Journal of Big Data},
  year    = {2026},
  note    = {Under review}
}
```

### Conference

```bibtex
@incollection{wilaiwongsakul_bare_2026,
  author    = {Wilaiwongsakul, Attavit and Liang, Bin and Jia, Wenfeng and Zheng, Bryan and Chen, Fang},
  title     = {{BARE}: {Boundary}-{Aware} with {Resolution} {Enhancement} for {Tree} {Crown} {Delineation}},
  booktitle = {Data Science and Machine Learning. AusDM 2025},
  editor    = {Nguyen, Quang V. and Li, Yan and Kwan, Paul and Zhao, Yanchun and Boo, Yee Ling and Nayak, Richi},
  series    = {Communications in Computer and Information Science},
  volume    = {2765},
  pages     = {302--315},
  year      = {2026},
  publisher = {Springer, Singapore},
  doi       = {10.1007/978-981-95-6786-7_20}
}
```

> **Important correction.** The SETR baseline results in the AusDM 2025 conference paper contained an implementation error: the baseline configuration inadvertently applied logit upsampling during training, making the baseline behave like full-resolution supervision. This led to near-identical B-IoU scores (0.643 baseline vs. 0.644 full-res). The corrected implementation in the journal extension reveals substantially larger B-IoU improvements for SETR (+0.1045), consistent with the resolution bottleneck hypothesis. See the [CHANGELOG](CHANGELOG.md) for the commit fix.

## Historical Snapshots

Use git tags to check out the exact code state corresponding to each paper:

```bash
# AusDM 2025 conference paper (BARE)
git checkout paper/ausdm2025

# Journal of Big Data extension (FRTS)
git checkout paper/jbd2026
```

## License

MIT License — see [LICENSE](LICENSE).

## Citation

If you use this code, please cite both papers above, or use the `CITATION.cff` file via GitHub's "Cite this repository" button.
