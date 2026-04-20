#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cross-Dataset Loader for Journal Extension.

This module provides dataset loaders for external tree crown segmentation datasets,
enabling cross-dataset validation of the train-time upsampling methodology.

Supported Datasets:
- SelvaMask: Tropical tree crown segmentation (HuggingFace, COCO polygons)
- BAMFORESTS: Bavarian forest/park tree crowns (local, COCO format)
- SavannaTreeAI: Northern Australian savanna tree species (local/Zenodo, COCO format)
- Quebec: Boreal/temperate forest tree crowns (local, GeoTIFF + GeoPackage)
- OAM-TCD: Open Aerial Map Tree Crown Delineation (via HuggingFace)

All datasets return consistent dictionaries: {'image': PIL.Image, 'mask': PIL.Image}
"""

import os
import glob
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
import logging
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

# Setup module logger
try:
    import sys
    _codebase_dir = os.path.join(os.path.dirname(__file__), '..')
    if _codebase_dir not in sys.path:
        sys.path.insert(0, _codebase_dir)
    from Core.utils import get_logger
    logger = get_logger()
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

# Optional imports for geospatial data
try:
    import rasterio  # type: ignore[import-not-found]
    from rasterio.windows import Window
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.transform import from_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    logger.warning("rasterio not available. Some geospatial datasets may not load correctly.")

# Optional imports for geospatial vector data
try:
    import geopandas as gpd  # type: ignore[import-not-found]
    from shapely.geometry import box as shapely_box
    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    logger.warning("geopandas not available. Quebec dataset will not load.")

# Optional imports for COCO-format annotations
try:
    from pycocotools.coco import COCO  # type: ignore[import-not-found]
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    logger.warning("pycocotools not available. COCO-format datasets require: pip install pycocotools")

# Optional imports for HuggingFace datasets
try:
    from datasets import load_dataset, Dataset as HFDataset, DatasetDict
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    logger.warning("HuggingFace datasets not available. OAM-TCD loading will fail.")


# =============================================================================
# Base Dataset Class
# =============================================================================

class BaseTreeDataset(ABC):
    """
    Abstract base class for tree crown segmentation datasets.
    
    All subclasses must implement __len__ and __getitem__ methods that return
    consistent dictionary format: {'image': PIL.Image, 'mask': PIL.Image}
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            root_dir: Root directory of the dataset
            split: Dataset split ('train', 'val', 'test')
            transform: Optional transform to apply to samples
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root directory not found: {root_dir}")
    
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with 'image' and 'mask' keys containing PIL Images
        """
        pass
    
    def _ensure_rgb(self, image: Image.Image) -> Image.Image:
        """Ensure image is in RGB format."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    
    def _ensure_binary_mask(self, mask: Image.Image) -> Image.Image:
        """Ensure mask is binary (0 = background, 1 = tree)."""
        mask_array = np.array(mask)
        
        # Handle different mask formats
        if mask_array.ndim == 3:
            # Multi-channel mask, take first channel or convert
            mask_array = mask_array[:, :, 0]
        
        # Binarize if not already
        if mask_array.max() > 1:
            # Assume values > 0 are tree class
            mask_array = (mask_array > 0).astype(np.uint8)
        
        return Image.fromarray(mask_array, mode='L')


# =============================================================================
# COCO Annotation Helper
# =============================================================================

class COCOAnnotationMixin:
    """Mixin for datasets that use COCO-format polygon annotations."""

    @staticmethod
    def _coco_anns_to_binary_mask(coco: "COCO", img_id: int, height: int, width: int) -> np.ndarray:
        """Rasterise all COCO polygon annotations for a given image into a binary mask."""
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        mask = np.zeros((height, width), dtype=np.uint8)
        for ann in anns:
            ann_mask = coco.annToMask(ann)
            mask = np.maximum(mask, ann_mask)
        return mask


# =============================================================================
# SelvaMask Dataset (HuggingFace)
# =============================================================================

class SelvaMaskDataset(BaseTreeDataset, COCOAnnotationMixin):
    """
    Loader for the SelvaMask dataset (HuggingFace: selvamask/SelvaMask).

    SelvaMask provides UAV orthomosaics of tropical forests from Panama, Brazil,
    and Ecuador at ~3.5 cm GSD with COCO-format polygon annotations for
    individual tree crowns.

    The HuggingFace version stores COCO polygon annotations inline under the
    'annotations' key (with 'segmentation' sub-field).  If a pre-rasterised
    'label' column exists it is used directly; otherwise polygons are
    rasterised on the fly.  A local COCO JSON can also be supplied.
    """

    def __init__(
        self,
        root_dir: Optional[str] = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        dataset_name: str = "selvamask/SelvaMask",
        cache_dir: Optional[str] = None,
        annotation_file: Optional[str] = None,
    ):
        if not HUGGINGFACE_AVAILABLE:
            raise RuntimeError("HuggingFace datasets library required. Install with: pip install datasets")

        self.split = split
        self.transform = transform
        self.dataset_name = dataset_name
        self._local_coco = None

        logger.info(f"Loading SelvaMask dataset from HuggingFace: {dataset_name}")
        self.dataset: Any = load_dataset(dataset_name, split=split, cache_dir=cache_dir)  # type: ignore[possibly-undefined]
        logger.info(f"SelvaMaskDataset: Loaded {len(self.dataset)} samples for split '{split}'")

        if annotation_file is not None:
            if not PYCOCOTOOLS_AVAILABLE:
                raise RuntimeError("pycocotools required for COCO annotations. pip install pycocotools")
            self._local_coco = COCO(annotation_file)  # type: ignore[possibly-undefined]

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _rasterise_inline_annotations(annotations: Dict[str, Any], height: int, width: int) -> np.ndarray:
        """Rasterise COCO polygon annotations stored inline in HuggingFace rows."""
        from PIL import ImageDraw
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        segmentations = annotations.get("segmentation", [])
        for poly_group in segmentations:
            # Each poly_group is a list of polygons; each polygon is [x,y,x,y,...]
            for poly in poly_group:
                if len(poly) >= 6:  # Need at least 3 points
                    coords = list(zip(poly[0::2], poly[1::2]))
                    draw.polygon(coords, fill=1)
        return np.array(mask, dtype=np.uint8)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))

        if "label" in item and item["label"] is not None:
            mask = item["label"]
            if not isinstance(mask, Image.Image):
                mask = Image.fromarray(np.array(mask).astype(np.uint8))
        elif "annotations" in item and item["annotations"] is not None:
            w, h = image.size
            mask_array = self._rasterise_inline_annotations(item["annotations"], h, w)
            mask = Image.fromarray(mask_array, mode="L")
        elif self._local_coco is not None:
            img_id = item.get("image_id", idx)
            w, h = image.size
            mask_array = self._coco_anns_to_binary_mask(self._local_coco, img_id, h, w)
            mask = Image.fromarray(mask_array, mode="L")
        else:
            raise RuntimeError(
                "No mask available. Provide annotation_file for COCO polygon rasterisation."
            )

        image = self._ensure_rgb(image)
        mask = self._ensure_binary_mask(mask)
        sample = {"image": image, "mask": mask}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _ensure_rgb(self, image: Image.Image) -> Image.Image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def _ensure_binary_mask(self, mask: Image.Image) -> Image.Image:
        mask_array = np.array(mask)
        if mask_array.ndim == 3:
            mask_array = mask_array[:, :, 0]
        if mask_array.max() > 1:
            mask_array = (mask_array > 0).astype(np.uint8)
        return Image.fromarray(mask_array, mode="L")


# =============================================================================
# BAMFORESTS Dataset (COCO format, local)
# =============================================================================

class BAMFORESTSDataset(BaseTreeDataset, COCOAnnotationMixin):
    """
    Loader for the BAMFORESTS dataset (Troles et al., 2024).

    BAMFORESTS provides 27,160 individually delineated tree crowns across 105 ha
    of VHR UAV imagery from coniferous, mixed, and deciduous forests and city
    parks in Bamberg, Germany. Annotations are in COCO instance segmentation
    format.

    Expected directory structure:
        root_dir/
            images/
                <image_file>.png
                ...
            annotations/
                instances_train.json   (or a single annotations.json)
                instances_val.json
                ...
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        annotation_file: Optional[str] = None,
    ):
        super().__init__(root_dir, split, transform)

        if not PYCOCOTOOLS_AVAILABLE:
            raise RuntimeError("pycocotools required for BAMFORESTS. pip install pycocotools")

        if annotation_file is not None:
            ann_path = Path(annotation_file)
        else:
            ann_dir = self.root_dir / "annotations"
            candidates = [
                ann_dir / f"instances_{split}.json",
                ann_dir / f"{split}.json",
                ann_dir / "annotations.json",
                self.root_dir / "annotations.json",
            ]
            ann_path = next((c for c in candidates if c.exists()), None)
            if ann_path is None:
                raise FileNotFoundError(
                    f"No COCO annotation file found for split '{split}' in {self.root_dir}. "
                    f"Tried: {[str(c) for c in candidates]}"
                )

        self.coco = COCO(str(ann_path))  # type: ignore[possibly-undefined]
        self.image_ids = sorted(self.coco.getImgIds())
        self._image_dir = self.root_dir / "images"
        if not self._image_dir.exists():
            self._image_dir = self.root_dir

        logger.info(
            f"BAMFORESTSDataset: Loaded {len(self.image_ids)} images from {ann_path}"
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = self._image_dir / img_info["file_name"]

        image = Image.open(image_path)
        image = self._ensure_rgb(image)

        w, h = image.size
        mask_array = self._coco_anns_to_binary_mask(self.coco, img_id, h, w)
        mask = Image.fromarray(mask_array, mode="L")
        mask = self._ensure_binary_mask(mask)

        sample = {"image": image, "mask": mask}
        if self.transform:
            sample = self.transform(sample)
        return sample


# =============================================================================
# SavannaTreeAI Dataset (Jansen et al. 2023, COCO format, local)
# =============================================================================

class SavannaTreeAIDataset(BaseTreeDataset, COCOAnnotationMixin):
    """
    Loader for the SavannaTreeAI dataset (Jansen et al., 2023).

    Provides RPAS imagery of Northern Australian savanna woodlands with 2547
    polygon annotations spanning 36 tree species, stored in COCO instance
    segmentation format. Available from Zenodo (DOI: 10.5281/zenodo.7094916).

    Expected directory structure:
        root_dir/
            images/                 (tiled 1024x1024 images)
                <tile>.png
                ...
            annotations.json        (COCO-format annotations)
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        annotation_file: Optional[str] = None,
    ):
        super().__init__(root_dir, split, transform)

        if not PYCOCOTOOLS_AVAILABLE:
            raise RuntimeError("pycocotools required for SavannaTreeAI. pip install pycocotools")

        if annotation_file is not None:
            ann_path = Path(annotation_file)
        else:
            candidates = [
                self.root_dir / "annotations.json",
                self.root_dir / "annotations" / f"instances_{split}.json",
                self.root_dir / "annotations" / f"{split}.json",
                self.root_dir / "annotations" / "annotations.json",
            ]
            ann_path = next((c for c in candidates if c.exists()), None)
            if ann_path is None:
                raise FileNotFoundError(
                    f"No COCO annotation file found in {self.root_dir}. "
                    f"Tried: {[str(c) for c in candidates]}"
                )

        self.coco = COCO(str(ann_path))  # type: ignore[possibly-undefined]
        self.image_ids = sorted(self.coco.getImgIds())
        # Check split-based image directory first (e.g. images/val/), then flat images/
        self._image_dir = self.root_dir / "images" / split
        if not self._image_dir.exists():
            self._image_dir = self.root_dir / "images"
        if not self._image_dir.exists():
            self._image_dir = self.root_dir

        logger.info(
            f"SavannaTreeAIDataset: Loaded {len(self.image_ids)} images from {ann_path} "
            f"(image_dir={self._image_dir})"
        )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        image_path = self._image_dir / img_info["file_name"]

        image = Image.open(image_path)
        image = self._ensure_rgb(image)

        w, h = image.size
        mask_array = self._coco_anns_to_binary_mask(self.coco, img_id, h, w)
        mask = Image.fromarray(mask_array, mode="L")
        mask = self._ensure_binary_mask(mask)

        sample = {"image": image, "mask": mask}
        if self.transform:
            sample = self.transform(sample)
        return sample


# =============================================================================
# Quebec Dataset (GeoTIFF + GeoPackage, local)
# =============================================================================

class QuebecDataset(BaseTreeDataset):
    """
    Loader for the Quebec tree crown dataset.

    Quebec provides large-scale GeoTIFF rasters with GeoPackage polygon
    annotations covering individual tree crowns. Annotations only cover a
    **partial region** of the raster.

    This loader:
    1. Tiles the raster into fixed-size crops.
    2. Rasterizes GeoPackage polygons into binary masks per tile.
    3. Marks pixels **outside** the annotation bounding box as ignore_index
       (255) so that evaluation metrics skip unannotated regions.

    Expected directory structure:
        root_dir/
            images/
                zone1/
                    <raster>.tif
                zone2/
                    ...
            annotations/
                Z1_polygons.gpkg
                Z2_polygons.gpkg
                ...
    """

    # Mapping from zone directory name to annotation file
    _ZONE_ANNOTATION_MAP = {
        "zone1": "Z1_polygons.gpkg",
        "zone2": "Z2_polygons.gpkg",
        "zone3": "Z3_polygons.gpkg",
    }

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        tile_size: int = 512,
        zones: Optional[List[str]] = None,
        ignore_index: int = 255,
    ):
        super().__init__(root_dir, split, transform)

        if not RASTERIO_AVAILABLE:
            raise RuntimeError("rasterio required for Quebec dataset. pip install rasterio")
        if not GEOPANDAS_AVAILABLE:
            raise RuntimeError("geopandas required for Quebec dataset. pip install geopandas")

        self.tile_size = tile_size
        self.ignore_index = ignore_index

        # Determine which zones to load
        if zones is None:
            zones = sorted(self._ZONE_ANNOTATION_MAP.keys())

        # Build tile index: list of (tif_path, gpkg_path, col_off, row_off)
        self._tiles: List[Tuple[str, str, int, int]] = []
        self._annotation_bounds: Dict[str, Any] = {}  # zone -> annotation bbox

        for zone_name in zones:
            gpkg_name = self._ZONE_ANNOTATION_MAP.get(zone_name)
            if gpkg_name is None:
                logger.warning(f"Unknown zone '{zone_name}', skipping.")
                continue

            gpkg_path = self.root_dir / "annotations" / gpkg_name
            zone_dir = self.root_dir / "images" / zone_name

            if not gpkg_path.exists():
                logger.warning(f"Annotation file not found: {gpkg_path}")
                continue
            if not zone_dir.exists():
                logger.warning(f"Image directory not found: {zone_dir}")
                continue

            tif_files = sorted(zone_dir.glob("*.tif"))
            if not tif_files:
                logger.warning(f"No .tif files found in {zone_dir}")
                continue

            tif_path = str(tif_files[0])  # Use the first raster in the zone

            # Read GeoPackage and compute the annotation bounding box
            gdf = gpd.read_file(str(gpkg_path))

            with rasterio.open(tif_path) as src:
                # Reproject annotations to raster CRS if needed
                if gdf.crs != src.crs:
                    gdf = gdf.to_crs(src.crs)

                ann_bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
                self._annotation_bounds[zone_name] = {
                    "bounds": ann_bounds,
                    "gdf": gdf,
                    "src_crs": src.crs,
                    "src_transform": src.transform,
                }

                # Generate tile grid
                for row_off in range(0, src.height, tile_size):
                    for col_off in range(0, src.width, tile_size):
                        self._tiles.append(
                            (tif_path, str(gpkg_path), col_off, row_off)
                        )

        logger.info(
            f"QuebecDataset: {len(self._tiles)} tiles across "
            f"{len(self._annotation_bounds)} zone(s), tile_size={tile_size}"
        )

    def __len__(self) -> int:
        return len(self._tiles)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tif_path, gpkg_path, col_off, row_off = self._tiles[idx]

        with rasterio.open(tif_path) as src:
            # Clamp window to raster bounds
            win_w = min(self.tile_size, src.width - col_off)
            win_h = min(self.tile_size, src.height - row_off)
            window = Window(col_off, row_off, win_w, win_h)

            img_arr = src.read(window=window)
            win_transform = src.window_transform(window)
            win_bounds = rasterio.windows.bounds(window, src.transform)

            # Convert to RGB HWC
            if img_arr.shape[0] >= 3:
                img_arr = np.transpose(img_arr[:3], (1, 2, 0))  # CHW -> HWC
            else:
                img_arr = np.transpose(img_arr, (1, 2, 0))

            # Determine the zone for this tile
            zone_name = None
            for zn, info in self._annotation_bounds.items():
                zone_dir = self.root_dir / "images" / zn
                zone_tifs = [str(t) for t in zone_dir.glob("*.tif")]
                if tif_path in zone_tifs:
                    zone_name = zn
                    break

            if zone_name is None:
                # Fallback: no annotation info, mark everything as ignore
                mask_arr = np.full((win_h, win_w), self.ignore_index, dtype=np.uint8)
            else:
                info = self._annotation_bounds[zone_name]
                ann_bounds = info["bounds"]  # [minx, miny, maxx, maxy]
                gdf = info["gdf"]

                # Create the annotation bounding box as a Shapely geometry
                ann_region = shapely_box(*ann_bounds)

                # Create the tile bounding box
                tile_region = shapely_box(win_bounds[0], win_bounds[1],
                                          win_bounds[2], win_bounds[3])

                if not tile_region.intersects(ann_region):
                    # Tile is entirely outside annotation region -> all ignore
                    mask_arr = np.full((win_h, win_w), self.ignore_index, dtype=np.uint8)
                else:
                    # Filter polygons that intersect this tile
                    gdf_tile = gdf[gdf.geometry.intersects(tile_region)]

                    # Rasterize tree polygons: 1 = tree
                    if not gdf_tile.empty:
                        shapes = [(geom, 1) for geom in gdf_tile.geometry]
                        mask_arr = rio_rasterize(
                            shapes,
                            out_shape=(win_h, win_w),
                            transform=win_transform,
                            fill=0,
                            dtype=np.uint8,
                        )
                    else:
                        mask_arr = np.zeros((win_h, win_w), dtype=np.uint8)

                    # Mark pixels outside the annotation bounding box as ignore
                    # Build a validity mask: 1 = inside annotation bbox
                    validity_shapes = [(ann_region, 1)]
                    validity_mask = rio_rasterize(
                        validity_shapes,
                        out_shape=(win_h, win_w),
                        transform=win_transform,
                        fill=0,
                        dtype=np.uint8,
                    )
                    mask_arr[validity_mask == 0] = self.ignore_index

        # Convert to PIL
        image = Image.fromarray(img_arr.astype(np.uint8)).convert("RGB")
        mask = Image.fromarray(mask_arr, mode="L")

        sample = {"image": image, "mask": mask}
        if self.transform:
            sample = self.transform(sample)
        return sample


# =============================================================================
# OAM-TCD Dataset (HuggingFace)
# =============================================================================

class OAMTCDDataset(BaseTreeDataset):
    """
    Dataset loader for OAM-TCD (Open Aerial Map Tree Crown Delineation) via HuggingFace.
    
    This is a wrapper around the HuggingFace dataset 'restor/tcd'.
    """
    
    def __init__(
        self,
        root_dir: Optional[str] = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        dataset_name: str = "restor/tcd",
        cache_dir: Optional[str] = None,
    ):
        if not HUGGINGFACE_AVAILABLE:
            raise RuntimeError("HuggingFace datasets library required. Install with: pip install datasets")
        
        # Don't call super().__init__ since we don't need root_dir
        self.split = split
        self.transform = transform
        self.dataset_name = dataset_name
        
        # Load dataset
        logger.info(f"Loading OAM-TCD dataset from HuggingFace: {dataset_name}")
        self.dataset: Any = load_dataset(dataset_name, split=split, cache_dir=cache_dir)  # type: ignore[possibly-undefined]
        logger.info(f"OAMTCDDataset: Loaded {len(self.dataset)} samples for split '{split}'")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        
        # HuggingFace TCD: try common mask key names
        image = item['image']
        _MASK_KEYS = ('label', 'mask', 'annotation', 'labels')
        mask = None
        for _k in _MASK_KEYS:
            if _k in item:
                mask = item[_k]
                break
        if mask is None:
            raise KeyError(
                f"Could not find mask in OAM-TCD sample. "
                f"Available keys: {list(item.keys())}. Expected one of {_MASK_KEYS}."
            )
        
        # Ensure PIL format
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.array(image))
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(np.array(mask).astype(np.uint8))
        
        image = self._ensure_rgb(image)
        mask = self._ensure_binary_mask(mask)
        
        sample = {'image': image, 'mask': mask}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def _ensure_rgb(self, image: Image.Image) -> Image.Image:
        """Ensure image is in RGB format."""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    
    def _ensure_binary_mask(self, mask: Image.Image) -> Image.Image:
        """Ensure mask is binary (0 = background, 1 = tree)."""
        mask_array = np.array(mask)
        if mask_array.ndim == 3:
            mask_array = mask_array[:, :, 0]
        if mask_array.max() > 1:
            mask_array = (mask_array > 0).astype(np.uint8)
        return Image.fromarray(mask_array, mode='L')


# =============================================================================
# Unified Dataset Wrapper
# =============================================================================

class UnifiedTreeDataset:
    """
    Unified wrapper that provides consistent access to all tree crown datasets.
    
    Usage:
        dataset = UnifiedTreeDataset(
            dataset_name='bamforests',
            root_dir='/path/to/data',
            split='train'
        )
        sample = dataset[0]  # {'image': PIL.Image, 'mask': PIL.Image}
    """
    
    SUPPORTED_DATASETS = {
        'oam_tcd': OAMTCDDataset,
        'tcd': OAMTCDDataset,
        'restor/tcd': OAMTCDDataset,
        'selvamask': SelvaMaskDataset,
        'bamforests': BAMFORESTSDataset,
        'savannatreeai': SavannaTreeAIDataset,
        'quebec': QuebecDataset,
    }
    
    def __init__(
        self,
        dataset_name: str,
        root_dir: Optional[str] = None,
        split: str = "train",
        transform: Optional[Callable] = None,
        **kwargs
    ):
        """
        Initialize unified dataset.
        
        Args:
            dataset_name: Name of the dataset (oam_tcd, selvamask, bamforests, savannatreeai)
            root_dir: Root directory for local datasets (ignored for HuggingFace datasets)
            split: Dataset split
            transform: Optional transform
            **kwargs: Additional arguments passed to specific dataset class
        """
        dataset_name_lower = dataset_name.lower()
        
        if dataset_name_lower not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: '{dataset_name}'. "
                f"Supported: {list(self.SUPPORTED_DATASETS.keys())}"
            )
        
        dataset_class = self.SUPPORTED_DATASETS[dataset_name_lower]
        
        # HuggingFace datasets don't need root_dir
        if dataset_class in (OAMTCDDataset, SelvaMaskDataset):
            self.dataset = dataset_class(
                split=split,
                transform=transform,
                **kwargs
            )
        else:
            if root_dir is None:
                raise ValueError(f"root_dir is required for dataset '{dataset_name}'")
            self.dataset = dataset_class(
                root_dir=root_dir,
                split=split,
                transform=transform,
                **kwargs
            )
        
        self.dataset_name = dataset_name
        self.split = split
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.dataset[idx]
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


# =============================================================================
# Factory Function
# =============================================================================

def load_tree_dataset(
    dataset_name: str,
    root_dir: Optional[str] = None,
    split: str = "train",
    transform: Optional[Callable] = None,
    **kwargs
) -> UnifiedTreeDataset:
    """
    Factory function to load any supported tree crown dataset.
    
    Args:
        dataset_name: Name of the dataset
        root_dir: Root directory for local datasets
        split: Dataset split
        transform: Optional transform
        **kwargs: Additional arguments
        
    Returns:
        UnifiedTreeDataset instance
    """
    return UnifiedTreeDataset(
        dataset_name=dataset_name,
        root_dir=root_dir,
        split=split,
        transform=transform,
        **kwargs
    )


def list_supported_datasets() -> List[str]:
    """List all supported dataset names."""
    return list(UnifiedTreeDataset.SUPPORTED_DATASETS.keys())


# =============================================================================
# Dataset Information
# =============================================================================

DATASET_INFO = {
    "oam_tcd": {
        "name": "Open Aerial Map Tree Crown Delineation",
        "source": "HuggingFace (restor/tcd)",
        "description": "Global tree crown segmentation from aerial imagery",
        "num_classes": 2,
        "resolution": "Variable (10-30cm GSD)",
        "requires_download": False,
    },
    "selvamask": {
        "name": "SelvaMask",
        "source": "HuggingFace (selvamask/SelvaMask)",
        "description": "Tropical tree crown segmentation (Panama, Brazil, Ecuador)",
        "num_classes": 2,
        "resolution": "4.5 cm GSD (UAV)",
        "requires_download": False,
    },
    "bamforests": {
        "name": "BAMFORESTS",
        "source": "DLR (local COCO files)",
        "description": "Bavarian forest/park tree crowns, 27k instances",
        "num_classes": 2,
        "resolution": "VHR UAV",
        "requires_download": True,
    },
    "savannatreeai": {
        "name": "SavannaTreeAI",
        "source": "Zenodo (DOI: 10.5281/zenodo.7094916)",
        "description": "Northern Australian savanna woodland, 2547 polygons, COCO format",
        "num_classes": 2,
        "resolution": "VHR RPAS",
        "requires_download": True,
    },
    "quebec": {
        "name": "Quebec Tree Crown",
        "source": "Local GeoTIFF + GeoPackage",
        "description": "Quebec forest tree crowns from large-scale orthomosaics with partial annotations",
        "num_classes": 2,
        "resolution": "VHR UAV",
        "requires_download": True,
        "notes": "Annotations cover only a subregion; unannotated pixels are marked ignore_index=255",
    },
}

def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Get information about a dataset."""
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower in DATASET_INFO:
        return DATASET_INFO[dataset_name_lower]
    raise ValueError(f"Unknown dataset: {dataset_name}")
