"""
Clustering LLHs module
Execution Order: 16
"""

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field

from .base import BaseLLH, register_llh

logger = logging.getLogger(__name__)


@dataclass
class KMeansConfig:
    """K-means configuration"""
    n_clusters: int = 16
    init: str = "k-means++"  # "k-means++", "random"
    n_init: int = 10
    max_iter: int = 300
    tol: float = 1e-4
    random_state: int = 42


@dataclass
class SpectralClusteringConfig:
    """Spectral clustering configuration"""
    n_clusters: int = 16
    affinity: str = "nearest_neighbors"  # "nearest_neighbors", "rbf"
    n_neighbors: int = 10
    gamma: float = 1.0
    random_state: int = 42


@dataclass  
class GaussianMixtureConfig:
    """Gaussian Mixture Model configuration"""
    n_components: int = 16
    covariance_type: str = "full"  # "full", "tied", "diag", "spherical"
    max_iter: int = 100
    tol: float = 1e-3
    random_state: int = 42


@register_llh("kmeans")
class KMeansClustering(BaseLLH):
    """K-means clustering LLH"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
        # Parse configuration
        self.kmeans_config = KMeansConfig(**config)
        
        # Initialize model
        self.model = KMeans(
            n_clusters=self.kmeans_config.n_clusters,
            init=self.kmeans_config.init,
            n_init=self.kmeans_config.n_init,
            max_iter=self.kmeans_config.max_iter,
            tol=self.kmeans_config.tol,
            random_state=self.kmeans_config.random_state
        )
        
        self.supports_meta_features = False
        self.requires_training = True
        
        logger.info(f"Initialized K-means LLH: {name}")
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply K-means clustering
        
        Args:
            data: Input data [H, W, B] or [N, B]
            **kwargs: Additional parameters
                - n_clusters: Override default number of clusters
                
        Returns:
            Cluster labels
        """
        if not self.validate_input(data):
            return np.zeros(data.shape[:2], dtype=np.int32)
        
        # Extract parameters
        n_clusters = kwargs.get('n_clusters', self.kmeans_config.n_clusters)
        
        # Reshape if 3D
        if data.ndim == 3:
            h, w, b = data.shape
            data_2d = data.reshape(-1, b)
            return_3d = True
        else:
            data_2d = data
            return_3d = False
        
        # Update model if n_clusters changed
        if n_clusters != self.model.n_clusters:
            self.model = KMeans(
                n_clusters=n_clusters,
                init=self.kmeans_config.init,
                n_init=self.kmeans_config.n_init,
                max_iter=self.kmeans_config.max_iter,
                tol=self.kmeans_config.tol,
                random_state=self.kmeans_config.random_state
            )
        
        # Fit and predict
        labels = self.model.fit_predict(data_2d)
        
        # Reshape back if needed
        if return_3d:
            labels = labels.reshape(h, w)
        
        return labels
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters"""
        return self.kmeans_config.__dict__
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set parameters"""
        for key, value in params.items():
            if hasattr(self.kmeans_config, key):
                setattr(self.kmeans_config, key, value)
        
        # Reinitialize model if parameters changed
        self.model = KMeans(
            n_clusters=self.kmeans_config.n_clusters,
            init=self.kmeans_config.init,
            n_init=self.kmeans_config.n_init,
            max_iter=self.kmeans_config.max_iter,
            tol=self.kmeans_config.tol,
            random_state=self.kmeans_config.random_state
        )
    
    def get_complexity(self) -> float:
        """Get computational complexity estimate"""
        # O(n_samples * n_clusters * n_features * n_iterations)
        return 0.8  # Relatively fast


@register_llh("spectral_clustering")
class SpectralClusteringLLH(BaseLLH):
    """Spectral clustering LLH"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
        # Parse configuration
        self.spectral_config = SpectralClusteringConfig(**config)
        
        # Initialize model
        self.model = SpectralClustering(
            n_clusters=self.spectral_config.n_clusters,
            affinity=self.spectral_config.affinity,
            n_neighbors=self.spectral_config.n_neighbors,
            gamma=self.spectral_config.gamma,
            random_state=self.spectral_config.random_state,
            assign_labels='kmeans'
        )
        
        self.supports_meta_features = False
        self.requires_training = True
        
        logger.info(f"Initialized Spectral Clustering LLH: {name}")
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply spectral clustering
        
        Args:
            data: Input data [H, W, B] or [N, B]
            **kwargs: Additional parameters
                - n_clusters: Override default number of clusters
                
        Returns:
            Cluster labels
        """
        if not self.validate_input(data):
            return np.zeros(data.shape[:2], dtype=np.int32)
        
        # Spectral clustering can be memory intensive for large datasets
        # Use subsampling for large images
        if data.ndim == 3:
            h, w, b = data.shape
            if h * w > 10000:  # Too many pixels for spectral clustering
                logger.warning("Image too large for spectral clustering, using K-means instead")
                from .clustering import KMeansClustering
                kmeans = KMeansClustering("kmeans_fallback", {"n_clusters": self.spectral_config.n_clusters})
                return kmeans.apply(data, **kwargs)
            
            data_2d = data.reshape(-1, b)
            return_3d = True
        else:
            data_2d = data
            if data_2d.shape[0] > 10000:
                # Subsample
                indices = np.random.choice(data_2d.shape[0], 10000, replace=False)
                data_2d = data_2d[indices]
                subsampled = True
            else:
                subsampled = False
            return_3d = False
        
        # Update model if n_clusters changed
        n_clusters = kwargs.get('n_clusters', self.spectral_config.n_clusters)
        if n_clusters != self.model.n_clusters:
            self.model = SpectralClustering(
                n_clusters=n_clusters,
                affinity=self.spectral_config.affinity,
                n_neighbors=self.spectral_config.n_neighbors,
                gamma=self.spectral_config.gamma,
                random_state=self.spectral_config.random_state,
                assign_labels='kmeans'
            )
        
        # Fit and predict
        labels = self.model.fit_predict(data_2d)
        
        # Handle subsampling
        if subsampled and not return_3d:
            # Assign remaining points to nearest cluster centers
            from sklearn.neighbors import NearestNeighbors
            centers = np.array([data_2d[labels == i].mean(axis=0) for i in range(n_clusters)])
            nbrs = NearestNeighbors(n_neighbors=1).fit(centers)
            all_indices = np.arange(data.shape[0])
            sampled_mask = np.isin(all_indices, indices)
            unsampled_indices = all_indices[~sampled_mask]
            
            if len(unsampled_indices) > 0:
                unsampled_data = data[unsampled_indices]
                distances, cluster_assignments = nbrs.kneighbors(unsampled_data)
                
                # Create full labels array
                full_labels = np.zeros(data.shape[0], dtype=np.int32)
                full_labels[indices] = labels
                full_labels[unsampled_indices] = cluster_assignments.flatten()
                labels = full_labels
        
        # Reshape back if needed
        if return_3d:
            labels = labels.reshape(h, w)
        
        return labels
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters"""
        return self.spectral_config.__dict__
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set parameters"""
        for key, value in params.items():
            if hasattr(self.spectral_config, key):
                setattr(self.spectral_config, key, value)
        
        # Reinitialize model
        self.model = SpectralClustering(
            n_clusters=self.spectral_config.n_clusters,
            affinity=self.spectral_config.affinity,
            n_neighbors=self.spectral_config.n_neighbors,
            gamma=self.spectral_config.gamma,
            random_state=self.spectral_config.random_state,
            assign_labels='kmeans'
        )
    
    def get_complexity(self) -> float:
        """Get computational complexity estimate"""
        # O(n_samples²) for affinity matrix, very expensive
        return 2.5  # Very high complexity


@register_llh("gmm")
class GaussianMixtureClustering(BaseLLH):
    """Gaussian Mixture Model clustering LLH"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
        # Parse configuration
        self.gmm_config = GaussianMixtureConfig(**config)
        
        # Initialize model
        self.model = GaussianMixture(
            n_components=self.gmm_config.n_components,
            covariance_type=self.gmm_config.covariance_type,
            max_iter=self.gmm_config.max_iter,
            tol=self.gmm_config.tol,
            random_state=self.gmm_config.random_state
        )
        
        self.supports_meta_features = False
        self.requires_training = True
        
        logger.info(f"Initialized GMM LLH: {name}")
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply Gaussian Mixture Model clustering
        
        Args:
            data: Input data [H, W, B] or [N, B]
            **kwargs: Additional parameters
                - n_components: Override default number of components
                
        Returns:
            Cluster labels
        """
        if not self.validate_input(data):
            return np.zeros(data.shape[:2], dtype=np.int32)
        
        # Reshape if 3D
        if data.ndim == 3:
            h, w, b = data.shape
            data_2d = data.reshape(-1, b)
            return_3d = True
        else:
            data_2d = data
            return_3d = False
        
        # Update model if n_components changed
        n_components = kwargs.get('n_components', self.gmm_config.n_components)
        if n_components != self.model.n_components:
            self.model = GaussianMixture(
                n_components=n_components,
                covariance_type=self.gmm_config.covariance_type,
                max_iter=self.gmm_config.max_iter,
                tol=self.gmm_config.tol,
                random_state=self.gmm_config.random_state
            )
        
        # Fit and predict
        labels = self.model.fit_predict(data_2d)
        
        # Reshape back if needed
        if return_3d:
            labels = labels.reshape(h, w)
        
        return labels
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters"""
        return self.gmm_config.__dict__
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set parameters"""
        for key, value in params.items():
            if hasattr(self.gmm_config, key):
                setattr(self.gmm_config, key, value)
        
        # Reinitialize model
        self.model = GaussianMixture(
            n_components=self.gmm_config.n_components,
            covariance_type=self.gmm_config.covariance_type,
            max_iter=self.gmm_config.max_iter,
            tol=self.gmm_config.tol,
            random_state=self.gmm_config.random_state
        )
    
    def get_complexity(self) -> float:
        """Get computational complexity estimate"""
        # O(n_samples * n_components * n_features²) for full covariance
        covariance_multiplier = {
            "full": 1.5,
            "tied": 1.2,
            "diag": 1.0,
            "spherical": 0.8
        }
        return 1.2 * covariance_multiplier.get(self.gmm_config.covariance_type, 1.0)
