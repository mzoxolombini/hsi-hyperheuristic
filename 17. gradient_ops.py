"""
Gradient operators LLH module
Execution Order: 15
"""

import numpy as np
from scipy.ndimage import gaussian_filter, sobel, gaussian_gradient_magnitude
from scipy import signal
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field

from .base import BaseLLH, register_llh

logger = logging.getLogger(__name__)


@dataclass
class GradientConfig:
    """Gradient operator configuration"""
    scale: str = "medium"  # "fine", "medium", "coarse"
    method: str = "sobel"  # "sobel", "scharr", "prewitt", "gaussian"
    sigma_spatial: float = 1.5
    sigma_spectral: float = 3.0
    alpha: float = 0.4  # Spatial weight
    beta: float = 0.4   # Spectral weight
    gamma: float = 0.2  # Mixed weight
    threshold_method: str = "percentile"  # "percentile", "otsu", "adaptive"
    threshold_value: float = 0.7  # For percentile


@register_llh("gradient_fine")
@register_llh("gradient_medium")
@register_llh("gradient_coarse")
class HolisticGradientOperator(BaseLLH):
    """
    Multi-scale spectral-spatial gradient operator
    
    Implements holistic gradient magnitude computation with
    meta-feature conditioned weights (Section 7.5.1)
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
        # Parse configuration
        self.gradient_config = GradientConfig(**config)
        
        # Set scale-specific parameters
        self._set_scale_parameters()
        
        # Precompute kernels for efficiency
        self._precompute_kernels()
        
        self.supports_meta_features = True
        self.requires_training = False
        
        logger.info(f"Initialized Gradient Operator: {name}, scale: {self.gradient_config.scale}")
    
    def _set_scale_parameters(self) -> None:
        """Set parameters based on scale"""
        scale_params = {
            "fine": (1.0, 0.8),    # sigma_lambda, sigma_spatial
            "medium": (3.0, 1.5),
            "coarse": (7.0, 2.5)
        }
        
        if self.gradient_config.scale in scale_params:
            sigma_lambda, sigma_spatial = scale_params[self.gradient_config.scale]
            self.gradient_config.sigma_spectral = sigma_lambda
            self.gradient_config.sigma_spatial = sigma_spatial
    
    def _precompute_kernels(self) -> None:
        """Precompute convolution kernels"""
        method = self.gradient_config.method
        
        if method == "sobel":
            self.kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            self.kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        elif method == "scharr":
            self.kernel_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
            self.kernel_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])
        elif method == "prewitt":
            self.kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            self.kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        else:  # gaussian
            self.kernel_x = None
            self.kernel_y = None
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply holistic gradient operator
        
        Args:
            data: Input data [H, W, B]
            **kwargs: Additional parameters
                - meta_features: Meta-features for conditioning
                - return_gradient: Return gradient magnitude instead of segmentation
                
        Returns:
            Segmentation map [H, W] or gradient magnitude
        """
        if not self.validate_input(data):
            return np.zeros(data.shape[:2], dtype=np.int32)
        
        # Extract parameters
        meta_features = kwargs.get('meta_features', None)
        return_gradient = kwargs.get('return_gradient', False)
        
        # Condition weights on meta-features
        if meta_features is not None:
            self.condition_on_meta_features(meta_features)
        
        # Compute holistic gradient magnitude
        gradient_magnitude = self._compute_holistic_gradient(data)
        
        if return_gradient:
            return gradient_magnitude
        
        # Convert gradient to segmentation via thresholding
        segmentation = self._threshold_gradient(gradient_magnitude)
        
        return segmentation
    
    def _compute_holistic_gradient(self, data: np.ndarray) -> np.ndarray:
        """
        Compute holistic gradient magnitude (Eq. 7.2)
        
        ∇_holistic = √(α·∇_spatial² + β·∇_spectral² + γ·∇_mixed²)
        """
        # Compute gradient components
        grad_spatial = self._compute_spatial_gradient(data)
        grad_spectral = self._compute_spectral_gradient(data)
        grad_mixed = self._compute_mixed_gradient(data)
        
        # Combine with conditioned weights
        gradient_magnitude = np.sqrt(
            self.gradient_config.alpha * grad_spatial**2 +
            self.gradient_config.beta * grad_spectral**2 +
            self.gradient_config.gamma * grad_mixed**2
        )
        
        # Normalize
        if gradient_magnitude.max() > 0:
            gradient_magnitude = gradient_magnitude / gradient_magnitude.max()
        
        return gradient_magnitude
    
    def _compute_spatial_gradient(self, data: np.ndarray) -> np.ndarray:
        """Compute spatial gradient ∇_xy I"""
        h, w, b = data.shape
        
        if b >= 3:
            # Use first 3 PCA components for spatial gradient
            from sklearn.decomposition import PCA
            pca = PCA(n_components=3)
            reshaped = data.reshape(-1, b)
            spatial_data = pca.fit_transform(reshaped).reshape(h, w, 3)
        else:
            spatial_data = data
        
        # Compute gradient using selected method
        if self.gradient_config.method == "gaussian":
            # Gaussian gradient magnitude
            grad_magnitude = np.zeros((h, w))
            for c in range(spatial_data.shape[2]):
                grad = gaussian_gradient_magnitude(
                    spatial_data[:, :, c],
                    sigma=self.gradient_config.sigma_spatial,
                    mode='reflect'
                )
                grad_magnitude += grad
            grad_magnitude /= spatial_data.shape[2]
        else:
            # Convolution-based gradient
            grad_x = np.zeros((h, w))
            grad_y = np.zeros((h, w))
            
            for c in range(spatial_data.shape[2]):
                if self.kernel_x is not None and self.kernel_y is not None:
                    gx = signal.convolve2d(
                        spatial_data[:, :, c],
                        self.kernel_x,
                        mode='same',
                        boundary='symm'
                    )
                    gy = signal.convolve2d(
                        spatial_data[:, :, c],
                        self.kernel_y,
                        mode='same',
                        boundary='symm'
                    )
                else:
                    gx = sobel(spatial_data[:, :, c], axis=0)
                    gy = sobel(spatial_data[:, :, c], axis=1)
                
                grad_x += gx
                grad_y += gy
            
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2) / spatial_data.shape[2]
        
        return grad_magnitude
    
    def _compute_spectral_gradient(self, data: np.ndarray) -> np.ndarray:
        """Compute spectral gradient ∇_λ I"""
        h, w, b = data.shape
        
        if b < 2:
            return np.zeros((h, w))
        
        grad_spectral = np.zeros((h, w))
        
        # Gaussian derivative along spectral dimension
        for i in range(h):
            for j in range(w):
                spectrum = data[i, j, :]
                
                # Apply Gaussian smoothing
                smoothed = gaussian_filter(spectrum, sigma=self.gradient_config.sigma_spectral)
                
                # Compute gradient
                gradient = np.gradient(smoothed)
                
                # Use L2 norm of gradient
                grad_spectral[i, j] = np.linalg.norm(gradient)
        
        # Normalize
        if grad_spectral.max() > 0:
            grad_spectral = grad_spectral / grad_spectral.max()
        
        return grad_spectral
    
    def _compute_mixed_gradient(self, data: np.ndarray) -> np.ndarray:
        """Compute mixed gradient ∇_xy,λ I (second-order derivatives)"""
        h, w, b = data.shape
        
        if b < 2:
            return np.zeros((h, w))
        
        grad_mixed = np.zeros((h, w))
        
        # Limit computation to first N bands for efficiency
        n_bands = min(10, b)
        
        for band in range(n_bands):
            # Compute spatial gradient for this band
            band_data = data[:, :, band]
            
            if self.gradient_config.method == "gaussian":
                spatial_grad = gaussian_gradient_magnitude(
                    band_data,
                    sigma=self.gradient_config.sigma_spatial,
                    mode='reflect'
                )
            else:
                if self.kernel_x is not None and self.kernel_y is not None:
                    gx = signal.convolve2d(band_data, self.kernel_x, mode='same', boundary='symm')
                    gy = signal.convolve2d(band_data, self.kernel_y, mode='same', boundary='symm')
                else:
                    gx = sobel(band_data, axis=0)
                    gy = sobel(band_data, axis=1)
                
                spatial_grad = np.sqrt(gx**2 + gy**2)
            
            # Compute spectral derivative of spatial gradient
            # Approximate by difference with neighboring bands
            if band < n_bands - 1:
                next_band_data = data[:, :, band + 1]
                
                if self.gradient_config.method == "gaussian":
                    next_spatial_grad = gaussian_gradient_magnitude(
                        next_band_data,
                        sigma=self.gradient_config.sigma_spatial,
                        mode='reflect'
                    )
                else:
                    if self.kernel_x is not None and self.kernel_y is not None:
                        gx = signal.convolve2d(next_band_data, self.kernel_x, mode='same', boundary='symm')
                        gy = signal.convolve2d(next_band_data, self.kernel_y, mode='same', boundary='symm')
                    else:
                        gx = sobel(next_band_data, axis=0)
                        gy = sobel(next_band_data, axis=1)
                    
                    next_spatial_grad = np.sqrt(gx**2 + gy**2)
                
                # Mixed gradient as spectral derivative of spatial gradient
                mixed = np.abs(next_spatial_grad - spatial_grad)
                grad_mixed += mixed
        
        grad_mixed /= n_bands
        
        # Normalize
        if grad_mixed.max() > 0:
            grad_mixed = grad_mixed / grad_mixed.max()
        
        return grad_mixed
    
    def _threshold_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """Convert gradient magnitude to segmentation via thresholding"""
        method = self.gradient_config.threshold_method
        
        if method == "percentile":
            threshold = np.percentile(gradient, self.gradient_config.threshold_value * 100)
            segmentation = (gradient > threshold).astype(np.int32)
        
        elif method == "otsu":
            from skimage.filters import threshold_otsu
            try:
                threshold = threshold_otsu(gradient)
                segmentation = (gradient > threshold).astype(np.int32)
            except:
                # Fallback to percentile
                threshold = np.percentile(gradient, 70)
                segmentation = (gradient > threshold).astype(np.int32)
        
        elif method == "adaptive":
            from skimage.filters import threshold_local
            try:
                block_size = min(35, gradient.shape[0] // 4, gradient.shape[1] // 4)
                block_size = max(block_size, 3)
                if block_size % 2 == 0:
                    block_size += 1
                
                adaptive_thresh = threshold_local(gradient, block_size, offset=0.1)
                segmentation = (gradient > adaptive_thresh).astype(np.int32)
            except:
                # Fallback to percentile
                threshold = np.percentile(gradient, 70)
                segmentation = (gradient > threshold).astype(np.int32)
        
        else:
            # Default: median threshold
            threshold = np.median(gradient)
            segmentation = (gradient > threshold).astype(np.int32)
        
        # Label connected components
        from skimage.measure import label
        segmentation = label(segmentation, connectivity=2)
        
        return segmentation
    
    def condition_on_meta_features(self, meta_features: Dict[str, float]) -> None:
        """Condition weights on meta-features (Section 7.5.1)"""
        complexity = meta_features.get('fractal_dimension', 1.5)
        spectral_ent = meta_features.get('spectral_entropy', 0.5)
        
        # Adjust weights based on complexity
        if complexity > 1.5:
            # Complex regions: emphasize spectral gradient
            self.gradient_config.alpha = 0.3
            self.gradient_config.beta = 0.5
            self.gradient_config.gamma = 0.2
        elif complexity < 1.2:
            # Simple regions: emphasize spatial gradient
            self.gradient_config.alpha = 0.6
            self.gradient_config.beta = 0.2
            self.gradient_config.gamma = 0.2
        
        # Adjust based on spectral entropy
        if spectral_ent > 0.7:  # High spectral diversity
            self.gradient_config.gamma = min(0.4, self.gradient_config.gamma + 0.1)
            self.gradient_config.beta = max(0.3, self.gradient_config.beta - 0.05)
        
        # Ensure weights sum to 1
        total = self.gradient_config.alpha + self.gradient_config.beta + self.gradient_config.gamma
        if total > 0:
            self.gradient_config.alpha /= total
            self.gradient_config.beta /= total
            self.gradient_config.gamma /= total
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters"""
        return self.gradient_config.__dict__
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set parameters"""
        for key, value in params.items():
            if hasattr(self.gradient_config, key):
                setattr(self.gradient_config, key, value)
    
    def get_complexity(self) -> float:
        """Get computational complexity estimate"""
        # O(H * W * B) for spectral gradient dominates
        base_complexity = 1.0
        
        # Adjust based on scale
        scale_complexity = {
            "fine": 0.7,    # Smaller sigma = faster
            "medium": 1.0,
            "coarse": 1.3   # Larger sigma = slower
        }
        
        multiplier = scale_complexity.get(self.gradient_config.scale, 1.0)
        return base_complexity * multiplier
