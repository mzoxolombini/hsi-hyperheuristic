"""
Meta-feature extraction module
Execution Order: 12
"""

import numpy as np
from scipy.ndimage import sobel, uniform_filter
from scipy import stats
from scipy.signal import welch
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings

logger = logging.getLogger(__name__)


class MetaFeatureExtractor:
    """
    Extract 12 meta-features from hyperspectral patches
    
    Features (from Table 6.1):
    1. Fractal dimension
    2. Spectral gradient variance
    3. SNR
    4. Spatial homogeneity
    5. Spectral entropy
    6. Mean reflectance
    7. Band correlation
    8. Texture energy
    9. Local contrast
    10. Edge density
    11. Spectral variance
    12. Spatial variance
    """
    
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.feature_names = [
            'fractal_dimension',
            'spectral_gradient_variance',
            'snr',
            'spatial_homogeneity',
            'spectral_entropy',
            'mean_reflectance',
            'band_correlation',
            'texture_energy',
            'local_contrast',
            'edge_density',
            'spectral_variance',
            'spatial_variance'
        ]
        
    def extract(self, patch: np.ndarray) -> Dict[str, float]:
        """
        Extract all meta-features from patch
        
        Args:
            patch: Hyperspectral patch [H, W, B]
            
        Returns:
            Dictionary of meta-features
        """
        features = {}
        
        try:
            # 1. Fractal dimension
            features['fractal_dimension'] = self._fractal_dimension(patch)
            
            # 2. Spectral gradient variance
            features['spectral_gradient_variance'] = self._spectral_gradient_variance(patch)
            
            # 3. SNR
            features['snr'] = self._calculate_snr(patch)
            
            # 4. Spatial homogeneity
            features['spatial_homogeneity'] = self._spatial_homogeneity(patch)
            
            # 5. Spectral entropy
            features['spectral_entropy'] = self._spectral_entropy(patch)
            
            # 6. Mean reflectance
            features['mean_reflectance'] = self._mean_reflectance(patch)
            
            # 7. Band correlation
            features['band_correlation'] = self._band_correlation(patch)
            
            # 8. Texture energy
            features['texture_energy'] = self._texture_energy(patch)
            
            # 9. Local contrast
            features['local_contrast'] = self._local_contrast(patch)
            
            # 10. Edge density
            features['edge_density'] = self._edge_density(patch)
            
            # 11. Spectral variance
            features['spectral_variance'] = self._spectral_variance(patch)
            
            # 12. Spatial variance
            features['spatial_variance'] = self._spatial_variance(patch)
            
        except Exception as e:
            logger.warning(f"Error extracting meta-features: {e}")
            # Return default values
            for name in self.feature_names:
                features[name] = 0.5  # Default middle value
        
        # Normalize if requested
        if self.normalize:
            features = self._normalize_features(features)
        
        return features
    
    def _fractal_dimension(self, patch: np.ndarray) -> float:
        """Box-counting fractal dimension"""
        # Use mean band for 2D calculation
        img_2d = np.mean(patch, axis=2)
        
        # Ensure positive values
        img_2d = (img_2d - np.min(img_2d)) / (np.max(img_2d) - np.min(img_2d) + 1e-10)
        
        # Binarize
        threshold = np.mean(img_2d)
        binary = (img_2d > threshold).astype(np.uint8)
        
        # Box counting algorithm
        sizes = [2, 4, 8, 16, 32]
        counts = []
        
        for size in sizes:
            if size < binary.shape[0] and size < binary.shape[1]:
                # Count boxes containing foreground
                count = 0
                for i in range(0, binary.shape[0] - size, size):
                    for j in range(0, binary.shape[1] - size, size):
                        box = binary[i:i+size, j:j+size]
                        if np.any(box):
                            count += 1
                counts.append(count)
        
        if len(counts) >= 3:
            # Linear fit in log-log space
            log_sizes = np.log(sizes[:len(counts)])
            log_counts = np.log(counts)
            coeffs = np.polyfit(log_sizes, log_counts, 1)
            return -coeffs[0]  # Negative slope is fractal dimension
        else:
            return 1.5  # Default value
    
    def _spectral_gradient_variance(self, patch: np.ndarray) -> float:
        """Variance of spectral gradients"""
        h, w, b = patch.shape
        
        if b < 2:
            return 0.0
        
        # Compute spectral gradients
        spectral_grad = np.gradient(patch, axis=2)
        
        # Compute variance
        return np.var(spectral_grad)
    
    def _calculate_snr(self, patch: np.ndarray) -> float:
        """Signal-to-Noise Ratio"""
        # Estimate noise using spatial differences
        h, w, b = patch.shape
        
        if h < 2 or w < 2:
            return 25.0  # Default
        
        # Horizontal differences
        diff_h = patch[1:, :, :] - patch[:-1, :, :]
        
        # Vertical differences
        diff_v = patch[:, 1:, :] - patch[:, :-1, :]
        
        # Noise estimate
        noise_est = 0.5 * (np.std(diff_h) + np.std(diff_v))
        
        # Signal estimate
        signal_est = np.std(patch)
        
        # SNR in dB
        if noise_est > 0:
            snr_db = 20 * np.log10(signal_est / noise_est)
            return max(0.0, min(50.0, snr_db))  # Clip to [0, 50]
        else:
            return 50.0
    
    def _spatial_homogeneity(self, patch: np.ndarray) -> float:
        """Spatial homogeneity measure"""
        # Use GLCM contrast as inverse of homogeneity
        from skimage.feature import graycomatrix, graycoprops
        
        img_2d = np.mean(patch, axis=2)
        img_2d = (img_2d * 255).astype(np.uint8)
        
        try:
            glcm = graycomatrix(
                img_2d,
                distances=[1],
                angles=[0],
                levels=256,
                symmetric=True,
                normed=True
            )
            
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            homogeneity = 1.0 / (1.0 + contrast)
            return float(homogeneity)
        except:
            # Fallback: variance-based homogeneity
            variance = np.var(img_2d)
            return 1.0 / (1.0 + variance)
    
    def _spectral_entropy(self, patch: np.ndarray) -> float:
        """Entropy of mean spectrum"""
        spectrum = np.mean(patch, axis=(0, 1))
        
        # Ensure positive values
        spectrum = spectrum - np.min(spectrum) + 1e-10
        
        # Normalize to probability distribution
        spectrum = spectrum / (np.sum(spectrum) + 1e-10)
        
        # Calculate entropy
        entropy = -np.sum(spectrum * np.log(spectrum + 1e-10))
        
        # Normalize by log(bands)
        max_entropy = np.log(patch.shape[2])
        if max_entropy > 0:
            entropy = entropy / max_entropy
        
        return float(entropy)
    
    def _mean_reflectance(self, patch: np.ndarray) -> float:
        """Mean reflectance value"""
        return float(np.mean(patch))
    
    def _band_correlation(self, patch: np.ndarray) -> float:
        """Average band correlation"""
        h, w, b = patch.shape
        
        if b < 2:
            return 0.7  # Default
        
        # Reshape to [pixels, bands]
        reshaped = patch.reshape(-1, b)
        
        # Compute correlation matrix
        corr_matrix = np.corrcoef(reshaped.T)
        
        # Average of upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        avg_corr = np.mean(corr_matrix[mask])
        
        return float(avg_corr) if not np.isnan(avg_corr) else 0.7
    
    def _texture_energy(self, patch: np.ndarray) -> float:
        """Texture energy using Laws' masks"""
        img_2d = np.mean(patch, axis=2)
        
        # Laws' texture masks
        L5 = np.array([1, 4, 6, 4, 1]) / 16
        E5 = np.array([-1, -2, 0, 2, 1]) / 6
        S5 = np.array([-1, 0, 2, 0, -1]) / 4
        R5 = np.array([1, -4, 6, -4, 1]) / 16
        
        # Create 2D masks
        masks = []
        for mask1 in [L5, E5, S5, R5]:
            for mask2 in [L5, E5, S5, R5]:
                masks.append(np.outer(mask1, mask2))
        
        # Convolve with each mask
        energies = []
        for mask in masks:
            from scipy.signal import convolve2d
            response = convolve2d(img_2d, mask, mode='same', boundary='symm')
            energy = np.mean(response ** 2)
            energies.append(energy)
        
        return float(np.mean(energies))
    
    def _local_contrast(self, patch: np.ndarray) -> float:
        """Local contrast measure"""
        img_2d = np.mean(patch, axis=2)
        
        # Compute local standard deviation
        local_std = uniform_filter(img_2d ** 2, size=3)
        local_std -= uniform_filter(img_2d, size=3) ** 2
        local_std = np.sqrt(np.maximum(local_std, 0))
        
        return float(np.mean(local_std))
    
    def _edge_density(self, patch: np.ndarray) -> float:
        """Edge density using Sobel filter"""
        img_2d = np.mean(patch, axis=2)
        
        # Compute gradient magnitude
        grad_x = sobel(img_2d, axis=0)
        grad_y = sobel(img_2d, axis=1)
        grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # Threshold to detect edges
        threshold = np.percentile(grad_mag, 75)
        edges = grad_mag > threshold
        
        # Edge density = proportion of edge pixels
        density = np.mean(edges)
        
        return float(density)
    
    def _spectral_variance(self, patch: np.ndarray) -> float:
        """Average spectral variance"""
        h, w, b = patch.shape
        
        if b < 2:
            return 0.1  # Default
        
        # Variance per band
        band_variances = np.var(patch, axis=(0, 1))
        
        return float(np.mean(band_variances))
    
    def _spatial_variance(self, patch: np.ndarray) -> float:
        """Average spatial variance"""
        h, w, b = patch.shape
        
        if h < 2 or w < 2:
            return 0.1  # Default
        
        # Variance per pixel across bands
        spatial_variances = np.var(patch, axis=2)
        
        return float(np.mean(spatial_variances))
    
    def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Normalize features to [0, 1] range"""
        # Predefined ranges based on dataset statistics
        ranges = {
            'fractal_dimension': (1.0, 2.5),      # Fractal dimension range
            'spectral_gradient_variance': (0.0, 1.0),
            'snr': (0.0, 50.0),                   # dB
            'spatial_homogeneity': (0.0, 1.0),
            'spectral_entropy': (0.0, 1.0),
            'mean_reflectance': (0.0, 1.0),
            'band_correlation': (0.0, 1.0),
            'texture_energy': (0.0, 1.0),
            'local_contrast': (0.0, 0.5),
            'edge_density': (0.0, 0.5),
            'spectral_variance': (0.0, 0.5),
            'spatial_variance': (0.0, 0.5)
        }
        
        normalized = {}
        for name, value in features.items():
            if name in ranges:
                min_val, max_val = ranges[name]
                if max_val > min_val:
                    norm_val = (value - min_val) / (max_val - min_val)
                    normalized[name] = float(np.clip(norm_val, 0.0, 1.0))
                else:
                    normalized[name] = 0.5  # Default
            else:
                normalized[name] = value
        
        return normalized
    
    def features_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """Convert features dictionary to numpy array"""
        return np.array([features[name] for name in self.feature_names], dtype=np.float32)
    
    def features_to_tensor(self, features: Dict[str, float]) -> 'torch.Tensor':
        """Convert features dictionary to PyTorch tensor"""
        import torch
        return torch.FloatTensor(self.features_to_array(features))
