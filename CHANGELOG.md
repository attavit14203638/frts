# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] - Journal extension (FRTS, Journal of Big Data 2026)

### Added

- **New architectures** (`Extended/new_architectures.py`): DeepLabV3, UperNet+Swin, and OneFormer wrappers with a unified full-resolution-supervision hook.
- **External dataset loaders** (`Extended/datasets_loader.py`): SelvaMask, BAMFORESTS, SavannaTreeAI, and Quebec loaders for zero-shot cross-dataset evaluation.
- **Statistical validation** (`Extended/statistical_validation.py`): Friedman test and bootstrap 95% confidence intervals for multi-seed experiments.
- **Multi-seed training protocol**: three random seeds `{42, 123, 2024}` controlling weight initialisation, data shuffling, and stochastic regularisation.
- **Boundary IoU as primary metric** with configurable dilation radius (default `d = 0.02 * sqrt(H^2 + W^2)`, following Cheng et al. 2021).
- `CITATION.cff` and `CHANGELOG.md` for repository metadata.

### Changed

- **Repository renamed** from `bare` to `frts`. GitHub preserves automatic redirects from the previous URL.
- **Five architectures** are now the primary evaluation set (DeepLabV3, SETR, SegFormer, UperNet+Swin, OneFormer); PSPNet is retained for AusDM 2025 reproducibility but is not part of the journal evaluation.
- **Terminology**: "Upsampling Framework" / "Train-Time Upsampling" replaced with **Full-Resolution Training Supervision (FRTS)** to match the journal paper's vocabulary. Configuration flag `train_time_upsample` is retained as an alias for backward compatibility.

### Fixed

- **SETR baseline bug (conference-only regression).** In the AusDM 2025 version, the SETR baseline configuration inadvertently applied logit upsampling during training, making it behave identically to full-resolution supervision. Reported conference B-IoU values were baseline 0.643 vs. full-res 0.644 — essentially equal. The corrected baseline (tag `paper/jbd2026`) reveals the true conference result would have been approximately baseline 0.5364, full-res 0.6410 — an improvement of +0.1045 B-IoU, consistent with SETR's aggressive `R = 16` reduction factor. See Section 1, final paragraph, of the journal paper for the full disclosure.

## [paper/ausdm2025] - AusDM 2025 conference submission

### Added

- Initial release for the BARE conference paper (AusDM 2025, Springer CCIS vol. 2765, pp. 302-315, DOI 10.1007/978-981-95-6786-7_20).
- Three architectures: SegFormer, PSPNet, SETR.
- Training, evaluation, and prediction pipelines with optional train-time upsampling.
- Dataset support for `restor/tcd` (OAM-TCD).
- Class-weighting experiments (later shown to be architecture-dependent and dropped from the journal).

### Known issues (fixed in the journal extension)

- SETR baseline training unintentionally applied logit upsampling. See the Fixed section above for the corrected behaviour.
