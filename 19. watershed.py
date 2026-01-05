"""
Watershed segmentation LLH module
Execution Order: 17
"""

import numpy as np
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import disk, dilation, erosion
from scipy import ndimage as ndi
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field

from .base import BaseLLH, register_llh
from .gradient_ops import HolisticGradientOperator

logger = logging.getLogger(__name__)


@dataclass
class WatershedConfig:
    """Watershed segmentation configuration"""
    marker_source: str = "gradient"  # "gradient", "distance", "hmin", "intensity"
    compactness: float = 0.3
    gradient_scale: str = "medium"   # For gradient-based markers
    min_distance: int = 10
    threshold_abs: float = 0.1
    footprint_radius: int = 3
    connectivity: int = 2


@register_llh("watershed")
class WatershedSegmentation(BaseLLH):
    """
    Marker-controlled watershed segmentation
    
    Implements watershed segmentation with different marker
    generation strategies.
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
        # Parse configuration
        self.watershed_config = WatershedConfig(**config)
        
        # Initialize gradient operator for marker generation
        self.gradient_op = HolisticGradientOperator(
            f"gradient_{self.watershed_config.gradient_scale}",
            {"scale": self.watershed_config.gradient_scale}
        )
        
        self.supports_meta_features = False
        self.requires_training = False
        
        logger.info(f"Initialized Watershed LLH: {name}")
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply watershed segmentation
        
        Args:
            data: Input data [H, W, B]
            **kwargs: Additional parameters
                - markers: Precomputed markers (optional)
                - gradient_image: Precomputed gradient (optional)
                
        Returns:
            Segmentation labels
        """
        if not self.validate_input(data):
            return np.zeros(data.shape[:2], dtype=np.int32)
        
        # Ensure 3D input
        if data.ndim != 3:
            raise ValueError("Watershed requires 3D input [H, W, B]")
        
        # Extract parameters
        markers = kwargs.get('markers', None)
        gradient_image = kwargs.get('gradient_image', None)
        
        # Use first band or mean for intensity image
        if data.shape[2] > 1:
            intensity_image = np.mean(data, axis=2)
        else:
            intensity_image = data[:, :, 0]
        
        # Normalize intensity image
        intensity_image = (intensity_image - intensity_image.min()) / \
                         (intensity_image.max() - intensity_image.min() + 1e-10)
        
        # Generate markers if not provided
        if markers is None:
            markers = self._generate_markers(intensity_image, data)
        
        # Generate gradient image if not provided
        if gradient_image is None:
            gradient_image = self._generate_gradient_image(data)
        
        # Apply watershed
        labels = watershed(
            gradient_image,
            markers=markers,
            compactness=self.watershed_config.compactness,
            connectivity=self.watershed_config.connectivity
        )
        
        return labels
    
    def _generate_markers(self, intensity_image: np.ndarray, 
                         data: np.ndarray) -> np.ndarray:
        """Generate markers for watershed"""
        method = self.watershed_config.marker_source
        
        if method == "gradient":
            return self._gradient_based_markers(data)
        elif method == "distance":
            return self._distance_based_markers(intensity_image)
        elif method == "hmin":
            return self._hmin_based_markers(intensity_image)
        elif method == "intensity":
            return self._intensity_based_markers(intensity_image)
        else:
            logger.warning(f"Unknown marker method: {method}. Using gradient.")
            return self._gradient_based_markers(data)
    
    def _gradient_based_markers(self, data: np.ndarray) -> np.ndarray:
        """Generate markers from gradient minima"""
        # Compute gradient magnitude
        gradient_mag = self.gradient_op.apply(data, return_gradient=True)
        
        # Find local minima in gradient (potential object centers)
        from skimage.feature import peak_local_max
        
        # Invert gradient for minima detection (watershed expects basins)
        gradient_inv = gradient_mag.max() - gradient_mag
        
        # Find local maxima in inverted gradient = minima in original
        coordinates = peak_local_max(
            gradient_inv,
            min_distance=self.watershed_config.min_distance,
            threshold_abs=self.watershed_config.threshold_abs,
            footprint=disk(self.watershed_config.footprint_radius),
            exclude_border=False
        )
        
        # Create marker image
        markers = np.zeros(gradient_mag.shape, dtype=np.int32)
        for i, (y, x) in enumerate(coordinates):
            markers[y, x] = i + 1
        
        # Label connected components
        from skimage.measure import label
        markers = label(markers > 0, connectivity=2)
        
        return markers
    
    def _distance_based_markers(self, intensity_image: np.ndarray) -> np.ndarray:
        """Generate markers using distance transform"""
        # Threshold image to create binary mask
        threshold = np.percentile(intensity_image, 70)
        binary = intensity_image > threshold
        
        # Compute distance transform
        distance = ndi.distance_transform_edt(binary)
        
        # Find peaks in distance transform
        from skimage.feature import peak_local_max
        coordinates = peak_local_max(
            distance,
            min_distance=self.watershed_config.min_distance,
            footprint=disk(self.watershed_config.footprint_radius)
        )
        
        # Create marker image
        markers = np.zeros(distance.shape, dtype=np.int32)
        for i, (y, x) in enumerate(coordinates):
            markers[y, x] = i + 1
        
        # Label connected components
        from skimage.measure import label
        markers = label(markers > 0, connectivity=2)
        
        return markers
    
    def _hmin_based_markers(self, intensity_image: np.ndarray) -> np.ndarray:
        """Generate markers using h-minima transform"""
        try:
            from skimage.morphology import h_minima
            
            # Apply h-minima transform
            h = np.percentile(intensity_image, 30)
            markers = h_minima(intensity_image, h)
            
            # Label markers
            from skimage.measure import label
            markers = label(markers, connectivity=2)
            
            return markers
            
        except ImportError:
            logger.warning("h_minima not available. Using distance-based markers.")
            return self._distance_based_markers(intensity_image)
    
    def _intensity_based_markers(self, intensity_image: np.ndarray) -> np.ndarray:
        """Generate markers from intensity peaks"""
        # Find local maxima in intensity
        from skimage.feature import peak_local_max
        
        coordinates = peak_local_max(
            intensity_image,
            min_distance=self.watershed_config.min_distance,
            threshold_abs=self.watershed_config.threshold_abs,
            footprint=disk(self.watershed_config.footprint_radius)
        )
        
        # Create marker image
        markers = np.zeros(intensity_image.shape, dtype=np.int32)
        for i, (y, x) in enumerate(coordinates):
            markers[y, x] = i + 1
        
        # Label connected components
        from skimage.measure import label
        markers = label(markers > 0, connectivity=2)
        
        return markers
    
    def _generate_gradient_image(self, data: np.ndarray) -> np.ndarray:
        """Generate gradient image for watershed"""
        # Compute holistic gradient magnitude
        gradient_mag = self.gradient_op.apply(data, return_gradient=True)
        
        # Normalize for watershed
        if gradient_mag.max() > gradient_mag.min():
            gradient_mag = (gradient_mag - gradient_mag.min()) / \
                          (gradient_mag.max() - gradient_mag.min())
        
        return gradient_mag
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters"""
        return self.watershed_config.__dict__
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set parameters"""
        for key, value in params.items():
            if hasattr(self.watershed_config, key):
                setattr(self.watershed_config, key, value)
        
        # Update gradient operator if scale changed
        if 'gradient_scale' in params:
            self.gradient_op = HolisticGradientOperator(
                f"gradient_{self.watershed_config.gradient_scale}",
                {"scale": self.watershed_config.gradient_scale}
            )
    
    def get_complexity(self) -> float:
        """Get computational complexity estimate"""
        # Watershed + marker generation
        base_complexity = 1.2
        
        # Adjust based on marker source
        marker_complexity = {
            "gradient": 1.3,    # Includes gradient computation
            "distance": 1.1,
            "hmin": 1.0,
            "intensity": 0.9
        }
        
        multiplier = marker_complexity.get(self.watershed_config.marker_source, 1.0)
        return base_complexity * multiplier
