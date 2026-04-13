"""
Meta-Feature Extractor for Hyperspectral Images
Implements Table 6.1 from the thesis: 12 meta-features φ_t
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage
from skimage.filters import gabor
from skimage.feature import graycomatrix, graycoprops
from typing import Tuple


class MetaFeatureExtractor(nn.Module):
    """
    Extract 12 meta-features φ_t as defined in Table 6.1 of the thesis.
    
    Meta-features:
    1. Local fractal dimension (D_f) - texture complexity
    2. Spectral gradient variance (σ²∇λ) - material transition rate
    3. Signal-to-noise ratio (SNR) - data quality
    4. Spatial homogeneity (H_s) - spatial autocorrelation
    5. Spectral entropy (H_λ) - spectral disorder
    6. Mean reflectance (ρ̄) - average brightness
    7. Band correlation (ρ̄_avg) - spectral redundancy
    8. Texture energy (E_tex) - Gabor filter response
    9. Edge density - boundary frequency
    10. Spectral contrast - inter-band differences
    11. Local binary pattern variance - micro-texture
    12. Absorption depth - spectral feature strength
    """
    
    def __init__(self, num_spectral_bands: int = 34, patch_size: int = 64):
        super().__init__()
        self.num_bands = num_spectral_bands
        self.patch_size = patch_size
        self._create_gabor_bank()
        
        # Running statistics for normalization
        self.register_buffer('feature_mean', torch.zeros(12))
        self.register_buffer('feature_std', torch.ones(12))
        self.n_samples = 0
    
    def _create_gabor_bank(self):
        """Create Gabor filter bank for texture energy (E_tex)."""
        self.gabor_filters = []
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            for sigma in [1, 3]:
                for frequency in [0.1, 0.3]:
                    filt_real, filt_imag = gabor(
                        frequency, theta, sigma_x=sigma, sigma_y=sigma, 
                        n_stds=3, fill=True
                    )
                    self.gabor_filters.append(filt_real.astype(np.float32))
    
    def _fractal_dimension(self, patch: np.ndarray) -> float:
        """
        Local fractal dimension D_f using box-counting method.
        For 2D image patch, D_f ∈ [1.0, 2.0]
        """
        if patch.ndim == 3:
            patch = patch.mean(axis=0)  # Average across bands
        
        patch = (patch * 255).astype(np.uint8)
        
        # Box-counting algorithm
        sizes = [2, 4, 8, 16, 32]
        counts = []
        
        for size in sizes:
            count = 0
            for i in range(0, patch.shape[0], size):
                for j in range(0, patch.shape[1], size):
                    box = patch[i:i+size, j:j+size]
                    if box.max() - box.min() > 0:
                        count += 1
            counts.append(max(count, 1))
        
        if len(sizes) < 2:
            return 1.5
        
        # Linear regression in log-log space
        log_sizes = np.log(np.array(sizes))
        log_counts = np.log(np.array(counts))
        
        slope = np.polyfit(log_sizes, log_counts, 1)[0]
        return max(1.0, min(2.0, -slope))
    
    def _spectral_entropy(self, spectral_vector: np.ndarray) -> float:
        """Spectral entropy H_λ = -Σ p(λ) log p(λ)"""
        # Normalize to probability distribution
        spectral_vector = spectral_vector - spectral_vector.min()
        prob = spectral_vector / (spectral_vector.sum() + 1e-8)
        return -np.sum(prob * np.log2(prob + 1e-8))
    
    def _homogeneity(self, patch: np.ndarray) -> float:
        """Spatial homogeneity H_s using GLCM contrast."""
        if patch.ndim == 3:
            patch = patch.mean(axis=0)
        
        patch = (patch * 255).astype(np.uint8)
        glcm = graycomatrix(patch, distances=[1], angles=[0], levels=256, symmetric=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        
        # Normalize contrast to [0, 1]
        return 1.0 / (1.0 + contrast / 100.0)
    
    def _local_binary_pattern_variance(self, patch: np.ndarray) -> float:
        """Local Binary Pattern variance for micro-texture."""
        if patch.ndim == 3:
            patch = patch.mean(axis=0)
        
        # Simple LBP approximation using local differences
        from skimage.feature import local_binary_pattern
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(patch, n_points, radius, method='uniform')
        return lbp.var()
    
    def _absorption_depth(self, spectral_vector: np.ndarray) -> float:
        """
        Absorption depth - strength of spectral absorption features.
        Simulated using local minima detection.
        """
        # Find prominent local minima
        from scipy.signal import find_peaks
        # Negative for absorption (peaks in negative space)
        valleys, properties = find_peaks(-spectral_vector, prominence=0.05)
        if len(valleys) > 0:
            return np.mean(properties['peak_heights'])
        return 0.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input patch (B, C, H, W) - C is spectral dimension
        Returns:
            meta_features: (B, 12) tensor of normalized meta-features
        """
        batch_size = x.shape[0]
        features = []
        
        for i in range(batch_size):
            # Convert to numpy for feature extraction
            patch_np = x[i].cpu().numpy()  # (C, H, W)
            
            # 1. Local fractal dimension (D_f)
            df = self._fractal_dimension(patch_np)
            
            # 2. Spectral gradient variance (σ²∇λ)
            spectral_grad = np.diff(patch_np, axis=0)
            spectral_grad_var = spectral_grad.var()
            
            # 3. Signal-to-noise ratio (SNR)
            signal = patch_np.mean()
            noise = patch_np.std()
            snr = 20 * np.log10(signal / (noise + 1e-8))
            snr = max(0, min(50, snr)) / 50.0  # Normalize to [0,1]
            
            # 4. Spatial homogeneity (H_s)
            hs = self._homogeneity(patch_np)
            
            # 5. Spectral entropy (H_λ)
            mean_spectrum = patch_np.mean(axis=(1, 2))
            h_lambda = self._spectral_entropy(mean_spectrum)
            h_lambda = h_lambda / np.log2(self.num_bands)  # Normalize
            
            # 6. Mean reflectance (ρ̄)
            mean_ref = patch_np.mean()
            
            # 7. Band correlation (ρ̄_avg)
            correlation = np.corrcoef(patch_np.reshape(self.num_bands, -1))
            avg_corr = (correlation.sum() - self.num_bands) / (self.num_bands * (self.num_bands - 1))
            
            # 8. Texture energy (E_tex) - Gabor filter response variance
            spatial_patch = patch_np.mean(axis=0)
            texture_energies = []
            for filt in self.gabor_filters:
                filtered = ndimage.convolve(spatial_patch, filt, mode='constant')
                texture_energies.append(filtered.var())
            etex = np.mean(texture_energies) / 100.0  # Normalize
            
            # 9. Edge density
            from scipy import ndimage as ndi
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = sobel_x.T
            grad_x = ndi.convolve(spatial_patch, sobel_x)
            grad_y = ndi.convolve(spatial_patch, sobel_y)
            edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            edge_density = (edge_magnitude > 0.1).mean()
            
            # 10. Spectral contrast
            spectral_contrast = mean_spectrum.max() - mean_spectrum.min()
            
            # 11. Local binary pattern variance
            lbp_var = self._local_binary_pattern_variance(patch_np)
            lbp_var = min(1.0, lbp_var / 100.0)
            
            # 12. Absorption depth
            abs_depth = self._absorption_depth(mean_spectrum)
            
            # Collect all 12 features
            feat_vec = np.array([
                df, spectral_grad_var, snr, hs, h_lambda, mean_ref,
                avg_corr, etex, edge_density, spectral_contrast, lbp_var, abs_depth
            ], dtype=np.float32)
            
            # Clip to reasonable ranges
            feat_vec = np.clip(feat_vec, 0.0, 1.0)
            features.append(feat_vec)
        
        features = torch.tensor(np.array(features), dtype=torch.float32, device=x.device)
        
        # Z-score normalization using running statistics
        if self.training:
            self._update_running_stats(features)
        
        normalized = (features - self.feature_mean) / (self.feature_std + 1e-8)
        return torch.sigmoid(normalized)  # Bound to [0, 1]
    
    def _update_running_stats(self, features: torch.Tensor):
        """Update running mean and std for normalization."""
        batch_mean = features.mean(dim=0)
        batch_std = features.std(dim=0)
        
        if self.n_samples == 0:
            self.feature_mean = batch_mean
            self.feature_std = batch_std
        else:
            self.feature_mean = (self.feature_mean * self.n_samples + batch_mean * features.shape[0]) / (self.n_samples + features.shape[0])
            self.feature_std = torch.sqrt(
                (self.feature_std**2 * self.n_samples + batch_std**2 * features.shape[0]) / (self.n_samples + features.shape[0])
            )
        self.n_samples += features.shape[0]