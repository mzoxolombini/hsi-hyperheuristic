"""
Preprocessing pipeline module
Execution Order: 10
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA, FastICA
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings

from config.config_loader import PreprocessingConfig

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Seven-stage preprocessing pipeline for hyperspectral images
    
    Stages:
    1. Radiometric correction
    2. Bad band removal
    3. Atmospheric correction
    4. Spectral-spatial denoising
    5. Dimensionality reduction
    6. Normalization
    7. Data augmentation (training only)
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.fitted_pca = None
        self.fitted_ica = None
        
    def process(self, data: np.ndarray, mode: str = "train") -> np.ndarray:
        """
        Apply full preprocessing pipeline
        
        Args:
            data: Input hyperspectral image [H, W, B]
            mode: "train", "val", or "test"
            
        Returns:
            Preprocessed image
        """
        logger.info(f"Starting preprocessing for {mode} mode")
        
        # Stage 1: Radiometric correction
        data = self._radiometric_correction(data)
        
        # Stage 2: Bad band removal (handled in dataset loader)
        
        # Stage 3: Atmospheric correction
        data = self._atmospheric_correction(data)
        
        # Stage 4: Spectral-spatial denoising
        data = self._denoise_data(data)
        
        # Stage 5: Dimensionality reduction
        data = self._reduce_dimensionality(data, mode)
        
        # Stage 6: Normalization
        data = self._normalize_data(data)
        
        # Stage 7: Data augmentation (training only)
        if mode == "train":
            data = self._augment_data(data)
        
        logger.info(f"Preprocessing completed. Output shape: {data.shape}")
        return data
    
    def _radiometric_correction(self, data: np.ndarray) -> np.ndarray:
        """Convert DN to reflectance (simplified)"""
        # In practice, use sensor-specific calibration coefficients
        # Here we use linear scaling to [0, 1]
        data_min = np.min(data)
        data_max = np.max(data)
        
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        
        return np.clip(data, 0.0, 1.0)
    
    def _atmospheric_correction(self, data: np.ndarray) -> np.ndarray:
        """QUAC/FLAASH approximation"""
        # Subtract dark pixel (5th percentile) per band
        h, w, b = data.shape
        
        corrected_data = np.zeros_like(data)
        for band in range(b):
            band_data = data[:, :, band]
            dark_value = np.percentile(band_data, 5)
            corrected_data[:, :, band] = band_data - dark_value
        
        # Ensure non-negative values
        corrected_data = np.maximum(corrected_data, 0.0)
        
        return corrected_data
    
    def _denoise_data(self, data: np.ndarray) -> np.ndarray:
        """Optimized spectral-spatial denoising"""
        method = self.config.denoising_method
        
        if method == "wavelet":
            return self._wavelet_denoise(data)
        elif method == "bm3d":
            return self._bm3d_denoise(data)
        elif method == "tv":
            return self._tv_denoise(data)
        elif method == "savgol":
            return self._savgol_denoise(data)
        else:
            logger.warning(f"Unknown denoising method: {method}. Using default.")
            return self._savgol_denoise(data)
    
    def _savgol_denoise(self, data: np.ndarray) -> np.ndarray:
        """Savitzky-Golay spectral smoothing"""
        h, w, b = data.shape
        denoised = np.zeros_like(data)
        
        for i in range(h):
            for j in range(w):
                spectrum = data[i, j, :]
                denoised[i, j, :] = savgol_filter(
                    spectrum,
                    window_length=7,
                    polyorder=3
                )
        
        return denoised
    
    def _wavelet_denoise(self, data: np.ndarray) -> np.ndarray:
        """Wavelet-based denoising"""
        try:
            import pywt
            
            h, w, b = data.shape
            denoised = np.zeros_like(data)
            
            # Process each band independently
            for band in range(b):
                band_data = data[:, :, band]
                
                # Perform 2D wavelet transform
                coeffs = pywt.dwt2(band_data, 'db4')
                
                # Threshold coefficients (hard thresholding)
                cA, (cH, cV, cD) = coeffs
                threshold = np.std(cD) * np.sqrt(2 * np.log(h * w))
                
                cH = pywt.threshold(cH, threshold, mode='hard')
                cV = pywt.threshold(cV, threshold, mode='hard')
                cD = pywt.threshold(cD, threshold, mode='hard')
                
                # Reconstruct
                denoised_band = pywt.idwt2((cA, (cH, cV, cD)), 'db4')
                denoised[:, :, band] = denoised_band[:h, :w]  # Handle padding
            
            return denoised
            
        except ImportError:
            logger.warning("PyWavelets not installed. Using Gaussian filter.")
            return gaussian_filter(data, sigma=1.0)
    
    def _bm3d_denoise(self, data: np.ndarray) -> np.ndarray:
        """BM3D denoising approximation"""
        # Use PCA + Gaussian filter as approximation
        if data.shape[2] > 10:
            pca = PCA(n_components=10)
            h, w, b = data.shape
            reshaped = data.reshape(-1, b)
            pca_comps = pca.fit_transform(reshaped).reshape(h, w, 10)
            
            # Denoise each component
            for c in range(10):
                pca_comps[:, :, c] = gaussian_filter(pca_comps[:, :, c], sigma=1.0)
            
            # Reconstruct
            denoised = pca.inverse_transform(pca_comps.reshape(-1, 10)).reshape(h, w, b)
            return denoised
        else:
            return gaussian_filter(data, sigma=1.0)
    
    def _tv_denoise(self, data: np.ndarray) -> np.ndarray:
        """Total Variation denoising"""
        from skimage.restoration import denoise_tv_chambolle
        
        if data.ndim == 3:
            h, w, b = data.shape
            denoised = np.zeros_like(data)
            
            for band in range(b):
                denoised[:, :, band] = denoise_tv_chambolle(
                    data[:, :, band],
                    weight=0.1,
                    eps=1e-4,
                    max_num_iter=100
                )
            return denoised
        else:
            return denoise_tv_chambolle(data, weight=0.1)
    
    def _reduce_dimensionality(self, data: np.ndarray, mode: str) -> np.ndarray:
        """Dimensionality reduction (PCA/ICA)"""
        method = self.config.dimensionality_reduction
        
        if method == "pca":
            return self._pca_reduction(data, mode)
        elif method == "ica":
            return self._ica_reduction(data, mode)
        elif method == "mfa":
            return self._mfa_reduction(data, mode)
        else:
            logger.warning(f"Unknown reduction method: {method}. Using PCA.")
            return self._pca_reduction(data, mode)
    
    def _pca_reduction(self, data: np.ndarray, mode: str) -> np.ndarray:
        """PCA dimensionality reduction"""
        h, w, b = data.shape
        reshaped = data.reshape(-1, b)
        
        if mode == "train" or self.fitted_pca is None:
            # Determine number of components for variance threshold
            pca_temp = PCA()
            pca_temp.fit(reshaped)
            
            cumulative_var = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumulative_var >= self.config.variance_threshold) + 1
            
            # Fit PCA with determined components
            self.fitted_pca = PCA(n_components=n_components)
            self.fitted_pca.fit(reshaped)
            logger.info(f"PCA fitted with {n_components} components "
                       f"({self.config.variance_threshold*100:.1f}% variance)")
        
        # Transform data
        reduced = self.fitted_pca.transform(reshaped).reshape(h, w, -1)
        return reduced
    
    def _ica_reduction(self, data: np.ndarray, mode: str) -> np.ndarray:
        """ICA dimensionality reduction"""
        h, w, b = data.shape
        reshaped = data.reshape(-1, b)
        
        if mode == "train" or self.fitted_ica is None:
            n_components = min(30, b // 2)
            self.fitted_ica = FastICA(
                n_components=n_components,
                max_iter=200,
                random_state=42
            )
            self.fitted_ica.fit(reshaped)
            logger.info(f"ICA fitted with {n_components} components")
        
        reduced = self.fitted_ica.transform(reshaped).reshape(h, w, -1)
        return reduced
    
    def _mfa_reduction(self, data: np.ndarray, mode: str) -> np.ndarray:
        """Minimum Noise Fraction (MNF) approximation"""
        # MNF = PCA on noise-whitened data
        h, w, b = data.shape
        
        # Estimate noise covariance
        noise_est = self._estimate_noise(data)
        
        # Whiten data
        data_whitened = self._whiten_data(data, noise_est)
        
        # Apply PCA
        return self._pca_reduction(data_whitened, mode)
    
    def _estimate_noise(self, data: np.ndarray) -> np.ndarray:
        """Estimate noise covariance matrix"""
        h, w, b = data.shape
        
        # Use spatial differences for noise estimation
        noise_samples = []
        
        for i in range(h-1):
            for j in range(w-1):
                # Differences with neighbors
                diff1 = data[i+1, j, :] - data[i, j, :]
                diff2 = data[i, j+1, :] - data[i, j, :]
                noise_samples.extend([diff1, diff2])
        
        noise_samples = np.array(noise_samples)
        noise_cov = np.cov(noise_samples.T)
        
        return noise_cov
    
    def _whiten_data(self, data: np.ndarray, noise_cov: np.ndarray) -> np.ndarray:
        """Whiten data using noise covariance"""
        # Compute eigenvalues/vectors of noise covariance
        eigvals, eigvecs = np.linalg.eigh(noise_cov)
        
        # Remove near-zero eigenvalues
        mask = eigvals > 1e-10
        eigvals = eigvals[mask]
        eigvecs = eigvecs[:, mask]
        
        # Whitening matrix
        whitening_mat = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        
        # Whiten data
        h, w, b = data.shape
        reshaped = data.reshape(-1, b)
        whitened = reshaped @ whitening_mat.T
        
        return whitened.reshape(h, w, -1)
    
    def _normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize data to zero mean, unit variance per band"""
        if data.ndim == 3:
            for band in range(data.shape[2]):
                band_data = data[:, :, band]
                mean = np.mean(band_data)
                std = np.std(band_data)
                
                if std > 1e-6:
                    data[:, :, band] = (band_data - mean) / std
        
        return data
    
    def _augment_data(self, data: np.ndarray) -> np.ndarray:
        """Data augmentation for training"""
        if np.random.random() < self.config.augmentation_prob:
            # Spectral mixing
            alpha = np.random.beta(0.4, 0.4)
            data = alpha * data + (1 - alpha) * np.roll(data, shift=10, axis=2)
        
        if np.random.random() < self.config.augmentation_prob:
            # Spatial rotation
            k = np.random.randint(0, 4)
            data = np.rot90(data, k=k, axes=(0, 1))
        
        if np.random.random() < self.config.augmentation_prob:
            # Flip
            if np.random.random() < 0.5:
                data = np.flip(data, axis=0)  # Vertical flip
            else:
                data = np.flip(data, axis=1)  # Horizontal flip
        
        if np.random.random() < self.config.augmentation_prob:
            # Add noise
            noise_level = np.random.uniform(0.01, 0.05)
            data = data + np.random.randn(*data.shape) * noise_level
        
        return data
    
    def save_state(self, path: str) -> None:
        """Save preprocessor state"""
        import joblib
        joblib.dump({
            'fitted_pca': self.fitted_pca,
            'fitted_ica': self.fitted_ica,
            'config': self.config
        }, path)
        logger.info(f"Preprocessor state saved to {path}")
    
    def load_state(self, path: str) -> None:
        """Load preprocessor state"""
        import joblib
        state = joblib.load(path)
        self.fitted_pca = state['fitted_pca']
        self.fitted_ica = state['fitted_ica']
        self.config = state['config']
        logger.info(f"Preprocessor state loaded from {path}")
