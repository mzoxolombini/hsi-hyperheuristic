"""
Dataset loader module
Execution Order: 9
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import h5py
import scipy.io
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import urllib.request
import tarfile
import zipfile
import warnings

from .preprocessing import Preprocessor
from .meta_features import MetaFeatureExtractor
from config.config_loader import DatasetConfig

logger = logging.getLogger(__name__)


class HSIDataset(Dataset):
    """
    Hyperspectral Image Dataset
    
    Loads standard HSI datasets (Indian Pines, Pavia University, Salinas)
    with full preprocessing pipeline.
    """
    
    # Dataset URLs
    DATASET_URLS = {
        'Indian_Pines': {
            'data': 'https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
            'gt': 'https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat'
        },
        'Pavia_University': {
            'data': 'https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
            'gt': 'https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat'
        },
        'Salinas': {
            'data': 'https://www.ehu.eus/ccwintco/uploads/f/f1/Salinas_corrected.mat',
            'gt': 'https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat'
        }
    }
    
    def __init__(
        self,
        dataset_config: DatasetConfig,
        preprocessor: Preprocessor,
        mode: str = "train",
        download: bool = True,
        transform: Optional[Any] = None
    ):
        """
        Initialize dataset
        
        Args:
            dataset_config: Dataset configuration
            preprocessor: Preprocessor instance
            mode: "train", "val", or "test"
            download: Whether to download dataset if not exists
            transform: Optional data augmentation transforms
        """
        self.config = dataset_config
        self.preprocessor = preprocessor
        self.mode = mode
        self.transform = transform
        
        # Ensure data directory exists
        self.data_dir = Path(self.config.get("data_dir", "./data"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download dataset if needed
        if download:
            self._download_dataset()
        
        # Load data
        self.data, self.gt = self._load_data()
        
        # Preprocess
        self.data = self.preprocessor.process(self.data, mode=mode)
        
        # Extract patches
        self.patches, self.patch_coords = self._extract_patches()
        
        # Extract meta-features
        self.meta_features = self._extract_meta_features()
        
        logger.info(f"Loaded {self.config.name} dataset: {len(self.patches)} patches")
    
    def _download_dataset(self) -> None:
        """Download dataset if not exists"""
        dataset_path = self.data_dir / self.config.name
        dataset_path.mkdir(exist_ok=True)
        
        data_file = dataset_path / f"{self.config.name}_data.mat"
        gt_file = dataset_path / f"{self.config.name}_gt.mat"
        
        # Download data file
        if not data_file.exists():
            logger.info(f"Downloading {self.config.name} data...")
            urllib.request.urlretrieve(
                self.DATASET_URLS[self.config.name]['data'],
                data_file
            )
        
        # Download ground truth file
        if not gt_file.exists():
            logger.info(f"Downloading {self.config.name} ground truth...")
            urllib.request.urlretrieve(
                self.DATASET_URLS[self.config.name]['gt'],
                gt_file
            )
    
    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from MATLAB files"""
        dataset_path = self.data_dir / self.config.name
        
        # Load data
        data_file = dataset_path / f"{self.config.name}_data.mat"
        gt_file = dataset_path / f"{self.config.name}_gt.mat"
        
        # Load MATLAB files
        if self.config.name == "Indian_Pines":
            data = scipy.io.loadmat(data_file)['indian_pines_corrected']
            gt = scipy.io.loadmat(gt_file)['indian_pines_gt']
        elif self.config.name == "Pavia_University":
            data = scipy.io.loadmat(data_file)['paviaU']
            gt = scipy.io.loadmat(gt_file)['paviaU_gt']
        elif self.config.name == "Salinas":
            data = scipy.io.loadmat(data_file)['salinas_corrected']
            gt = scipy.io.loadmat(gt_file)['salinas_gt']
        else:
            raise ValueError(f"Unknown dataset: {self.config.name}")
        
        # Convert to float32
        data = data.astype(np.float32)
        gt = gt.astype(np.int32)
        
        # Remove bad bands if specified
        if hasattr(self.config, 'bad_bands') and self.config.bad_bands:
            data = self._remove_bad_bands(data)
        
        return data, gt
    
    def _remove_bad_bands(self, data: np.ndarray) -> np.ndarray:
        """Remove water absorption and noisy bands"""
        keep_bands = []
        for band_idx in range(data.shape[2]):
            keep = True
            for bad_range in self.config.bad_bands:
                if bad_range[0] <= band_idx <= bad_range[1]:
                    keep = False
                    break
            if keep:
                keep_bands.append(band_idx)
        
        return data[:, :, keep_bands]
    
    def _extract_patches(self) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Extract overlapping patches"""
        patch_size = self.preprocessor.config.patch_size
        stride = self.preprocessor.config.stride
        
        patches = []
        coords = []
        
        h, w, _ = self.data.shape
        
        for i in range(0, h - patch_size, stride):
            for j in range(0, w - patch_size, stride):
                patch = self.data[i:i + patch_size, j:j + patch_size, :]
                patches.append(patch)
                coords.append((i, j))
        
        # Limit number of patches based on mode
        if self.mode == "train" and len(patches) > self.preprocessor.config.train_patches:
            indices = np.random.choice(len(patches), self.preprocessor.config.train_patches, replace=False)
            patches = [patches[i] for i in indices]
            coords = [coords[i] for i in indices]
        elif self.mode in ["val", "test"] and len(patches) > self.preprocessor.config.test_patches:
            indices = np.random.choice(len(patches), self.preprocessor.config.test_patches, replace=False)
            patches = [patches[i] for i in indices]
            coords = [coords[i] for i in indices]
        
        return patches, coords
    
    def _extract_meta_features(self) -> List[Dict[str, float]]:
        """Extract meta-features for each patch"""
        extractor = MetaFeatureExtractor()
        meta_features = []
        
        for patch in self.patches:
            features = extractor.extract(patch)
            meta_features.append(features)
        
        return meta_features
    
    def __len__(self) -> int:
        return len(self.patches)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get dataset item"""
        patch = self.patches[idx]
        meta_features = self.meta_features[idx]
        coords = self.patch_coords[idx]
        
        # Extract corresponding ground truth
        i, j = coords
        patch_size = self.preprocessor.config.patch_size
        gt_patch = self.gt[i:i + patch_size, j:j + patch_size]
        
        # Apply transforms if any
        if self.transform:
            patch, gt_patch = self.transform(patch, gt_patch)
        
        # Convert to tensors
        patch_tensor = torch.FloatTensor(patch).permute(2, 0, 1)  # [C, H, W]
        gt_tensor = torch.LongTensor(gt_patch)
        
        # Convert meta-features to tensor
        meta_tensor = torch.FloatTensor(list(meta_features.values()))
        
        return {
            'patch': patch_tensor,
            'meta_features': meta_tensor,
            'ground_truth': gt_tensor,
            'coords': torch.tensor(coords),
            'patch_idx': idx
        }
    
    def get_full_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get full image and ground truth"""
        return self.data, self.gt


class DatasetManager:
    """Manager for multiple datasets"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.datasets: Dict[str, HSIDataset] = {}
        self.preprocessor = Preprocessor(config['preprocessing'])
    
    def load_dataset(self, dataset_name: str, mode: str = "train") -> HSIDataset:
        """Load specific dataset"""
        if dataset_name not in self.config['datasets']:
            raise ValueError(f"Dataset {dataset_name} not in config")
        
        dataset_config = self.config['datasets'][dataset_name]
        
        # Create dataset
        dataset = HSIDataset(
            dataset_config=dataset_config,
            preprocessor=self.preprocessor,
            mode=mode
        )
        
        self.datasets[f"{dataset_name}_{mode}"] = dataset
        return dataset
    
    def create_dataloader(
        self,
        dataset_name: str,
        mode: str = "train",
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4
    ) -> DataLoader:
        """Create DataLoader for dataset"""
        dataset_key = f"{dataset_name}_{mode}"
        
        if dataset_key not in self.datasets:
            dataset = self.load_dataset(dataset_name, mode)
        else:
            dataset = self.datasets[dataset_key]
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle and mode == "train",
            num_workers=num_workers,
            pin_memory=True
        )
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get dataset information"""
        if dataset_name not in self.config['datasets']:
            raise ValueError(f"Dataset {dataset_name} not in config")
        
        config = self.config['datasets'][dataset_name]
        return {
            'name': dataset_name,
            'height': config.height,
            'width': config.width,
            'bands': config.bands,
            'classes': config.classes,
            'bad_bands': config.bad_bands if hasattr(config, 'bad_bands') else []
        }
