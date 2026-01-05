"""
SS-PSO LLH module
Execution Order: 14
"""

import numpy as np
from scipy import spatial
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass, field

from .base import BaseLLH, register_llh

logger = logging.getLogger(__name__)


@dataclass
class SSPSOConfig:
    """SS-PSO configuration"""
    n_particles: int = 50
    max_iterations: int = 200
    inertia_base: float = 0.9
    inertia_min: float = 0.4
    cognitive_weight: float = 1.5
    social_weight: float = 1.5
    spatial_weight: float = 1.0
    variant: str = "accurate"  # "fast", "accurate", "spatial", "spectral"
    n_clusters: int = 16
    convergence_tol: float = 1e-6
    verbose: bool = False


@register_llh("ss_pso_fast")
@register_llh("ss_pso_accurate")
@register_llh("ss_pso_spatial")
@register_llh("ss_pso_spectral")
class SpectralSpatialPSO(BaseLLH):
    """
    Spectral-Spatial PSO for hyperspectral clustering
    
    Implements SS-PSO with convergence guarantees and
    spatial regularization (Theorem 6.3.1)
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
        # Parse configuration
        self.sspso_config = SSPSOConfig(**config)
        
        # Set variant-specific parameters
        self._set_variant_parameters()
        
        # State variables
        self.particles = None
        self.velocities = None
        self.personal_best = None
        self.personal_best_fitness = None
        self.global_best = None
        self.global_best_fitness = None
        
        self.supports_meta_features = True
        self.requires_training = False
        
        logger.info(f"Initialized SS-PSO LLH: {name}, variant: {self.sspso_config.variant}")
    
    def _set_variant_parameters(self) -> None:
        """Set parameters based on variant"""
        variant_params = {
            "fast": (20, 50, 0.5),      # particles, iterations, spatial_weight
            "accurate": (100, 300, 1.0),
            "spatial": (50, 200, 2.0),
            "spectral": (50, 200, 0.1)
        }
        
        if self.sspso_config.variant in variant_params:
            n_particles, max_iter, spatial_weight = variant_params[self.sspso_config.variant]
            self.sspso_config.n_particles = n_particles
            self.sspso_config.max_iterations = max_iter
            self.sspso_config.spatial_weight = spatial_weight
    
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply SS-PSO clustering
        
        Args:
            data: Input data [H, W, B] or [N, B]
            **kwargs: Additional parameters
                - n_clusters: Number of clusters (optional)
                - meta_features: Meta-features for conditioning (optional)
                
        Returns:
            Segmentation labels
        """
        if not self.validate_input(data):
            return np.zeros(data.shape[:2], dtype=np.int32)
        
        # Extract parameters
        n_clusters = kwargs.get('n_clusters', self.sspso_config.n_clusters)
        meta_features = kwargs.get('meta_features', None)
        
        # Condition parameters on meta-features
        if meta_features is not None:
            self.condition_on_meta_features(meta_features)
        
        # Reshape if 3D
        if data.ndim == 3:
            h, w, b = data.shape
            data_2d = data.reshape(-1, b)
            return_3d = True
        else:
            data_2d = data
            return_3d = False
        
        # Run SS-PSO optimization
        centroids, labels = self._optimize(data_2d, n_clusters)
        
        # Reshape back if needed
        if return_3d:
            labels = labels.reshape(h, w)
        
        return labels
    
    def _optimize(self, data: np.ndarray, n_clusters: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        SS-PSO optimization with spatial regularization
        
        Args:
            data: Input data [N, B]
            n_clusters: Number of clusters
            
        Returns:
            Tuple of (centroids, labels)
        """
        n_samples, n_features = data.shape
        
        # Initialize particles
        self._initialize_particles(data, n_clusters, n_features)
        
        # Optimization loop
        for iteration in range(self.sspso_config.max_iterations):
            # Adaptive inertia weight
            inertia = self._adaptive_inertia(iteration)
            
            # Evaluate fitness and update best positions
            self._evaluate_and_update(data, n_clusters)
            
            # Check convergence
            if self._check_convergence():
                logger.debug(f"SS-PSO converged at iteration {iteration}")
                break
            
            # Update velocities and positions
            self._update_velocities_and_positions(inertia, data, n_clusters, iteration)
            
            # Log progress
            if self.sspso_config.verbose and (iteration + 1) % 50 == 0:
                logger.debug(f"Iteration {iteration + 1}: "
                           f"Best fitness = {self.global_best_fitness:.4f}")
        
        # Final clustering
        distances = spatial.distance.cdist(data, self.global_best, 'euclidean')
        labels = np.argmin(distances, axis=1)
        
        return self.global_best, labels
    
    def _initialize_particles(self, data: np.ndarray, n_clusters: int, n_features: int) -> None:
        """Initialize particles and velocities"""
        # Initialize particles using k-means++ initialization
        from sklearn.cluster import KMeans
        
        self.particles = np.zeros((self.sspso_config.n_particles, n_clusters, n_features))
        self.velocities = np.random.randn(*self.particles.shape) * 0.1
        
        # Initialize first particle with k-means++ centers
        kmeans = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=1,
            random_state=42
        )
        kmeans.fit(data)
        self.particles[0] = kmeans.cluster_centers_
        
        # Initialize remaining particles with random perturbations
        for i in range(1, self.sspso_config.n_particles):
            self.particles[i] = self.particles[0] + np.random.randn(n_clusters, n_features) * 0.5
        
        # Initialize best positions
        self.personal_best = self.particles.copy()
        self.personal_best_fitness = np.full(self.sspso_config.n_particles, -np.inf)
        self.global_best = self.particles[0].copy()
        self.global_best_fitness = -np.inf
    
    def _adaptive_inertia(self, iteration: int) -> float:
        """Calculate adaptive inertia weight"""
        # Linearly decreasing inertia
        inertia_range = self.sspso_config.inertia_base - self.sspso_config.inertia_min
        inertia = self.sspso_config.inertia_base - (inertia_range * iteration / self.sspso_config.max_iterations)
        return max(inertia, self.sspso_config.inertia_min)
    
    def _evaluate_and_update(self, data: np.ndarray, n_clusters: int) -> None:
        """Evaluate fitness and update best positions"""
        for i in range(self.sspso_config.n_particles):
            fitness = self._calculate_fitness(data, self.particles[i], n_clusters)
            
            # Update personal best
            if fitness > self.personal_best_fitness[i]:
                self.personal_best[i] = self.particles[i].copy()
                self.personal_best_fitness[i] = fitness
            
            # Update global best
            if fitness > self.global_best_fitness:
                self.global_best = self.particles[i].copy()
                self.global_best_fitness = fitness
    
    def _calculate_fitness(self, data: np.ndarray, centroids: np.ndarray, n_clusters: int) -> float:
        """
        Calculate fitness: -J + λ·𝓁_spat
        
        J: Reconstruction error
        𝓁_spat: Spatial regularization term
        λ: Spatial weight
        """
        # Assign points to clusters
        distances = spatial.distance.cdist(data, centroids, 'euclidean')
        labels = np.argmin(distances, axis=1)
        
        # Reconstruction error
        reconstruction_error = 0.0
        for k in range(n_clusters):
            cluster_points = data[labels == k]
            if len(cluster_points) > 0:
                error = np.sum(np.linalg.norm(cluster_points - centroids[k], axis=1) ** 2)
                reconstruction_error += error
        
        # Spatial regularization (if data has spatial structure)
        spatial_reg = self._spatial_regularization(labels, data.shape[0])
        
        # Fitness = negative total cost
        total_cost = reconstruction_error + self.sspso_config.spatial_weight * spatial_reg
        return -total_cost
    
    def _spatial_regularization(self, labels: np.ndarray, n_samples: int) -> float:
        """
        Spatial regularization term 𝓁_spat
        
        Args:
            labels: Cluster labels
            n_samples: Number of samples
            
        Returns:
            Spatial regularization loss
        """
        # Check if data can be arranged in a grid
        grid_size = int(np.sqrt(n_samples))
        if grid_size ** 2 != n_samples:
            return 0.0  # No spatial regularization for non-grid data
        
        # Arrange labels in grid
        label_grid = labels.reshape(grid_size, grid_size)
        
        # Calculate spatial discontinuities
        spatial_loss = 0.0
        
        for i in range(grid_size):
            for j in range(grid_size):
                current_label = label_grid[i, j]
                
                # Check 4-connected neighbors
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid_size and 0 <= nj < grid_size:
                        if label_grid[ni, nj] != current_label:
                            spatial_loss += 1.0
        
        return spatial_loss
    
    def _check_convergence(self) -> bool:
        """Check convergence criteria"""
        if self.global_best_fitness == -np.inf:
            return False
        
        # Check if fitness hasn't improved significantly
        recent_fitness = self.personal_best_fitness[:10]  # Last 10 personal bests
        if len(recent_fitness) >= 10:
            improvement = np.max(recent_fitness) - np.min(recent_fitness)
            if improvement < self.sspso_config.convergence_tol:
                return True
        
        return False
    
    def _update_velocities_and_positions(self, inertia: float, data: np.ndarray,
                                        n_clusters: int, iteration: int) -> None:
        """Update velocities and positions"""
        for i in range(self.sspso_config.n_particles):
            # Cognitive component
            r1 = np.random.rand()
            cognitive = (self.sspso_config.cognitive_weight * r1 * 
                        (self.personal_best[i] - self.particles[i]))
            
            # Social component
            r2 = np.random.rand()
            social = (self.sspso_config.social_weight * r2 * 
                     (self.global_best - self.particles[i]))
            
            # Spatial regularization gradient
            spatial_grad = self._spatial_gradient(data, self.particles[i], n_clusters)
            
            # Apply convergence constraint
            max_spatial_coeff = (1 - self.sspso_config.inertia_min) / 2
            spatial_coeff = min(self.sspso_config.spatial_weight, max_spatial_coeff)
            
            # Velocity update
            self.velocities[i] = (
                inertia * self.velocities[i] +
                cognitive +
                social +
                spatial_coeff * spatial_grad
            )
            
            # Position update
            self.particles[i] += self.velocities[i]
            
            # Boundary handling
            self._handle_boundaries(self.particles[i])
    
    def _spatial_gradient(self, data: np.ndarray, centroids: np.ndarray, n_clusters: int) -> np.ndarray:
        """Numerical gradient of spatial regularization term"""
        epsilon = 1e-6
        
        # Random perturbation
        perturbation = np.random.randn(*centroids.shape) * epsilon
        
        # Current loss
        current_loss = self._spatial_regularization_for_centroids(data, centroids, n_clusters)
        
        # Perturbed loss
        perturbed_centroids = centroids + epsilon * perturbation
        perturbed_loss = self._spatial_regularization_for_centroids(data, perturbed_centroids, n_clusters)
        
        # Numerical gradient
        gradient = (perturbed_loss - current_loss) / epsilon * perturbation
        
        return gradient
    
    def _spatial_regularization_for_centroids(self, data: np.ndarray, centroids: np.ndarray,
                                             n_clusters: int) -> float:
        """Compute spatial loss for given centroids"""
        distances = spatial.distance.cdist(data, centroids, 'euclidean')
        labels = np.argmin(distances, axis=1)
        return self._spatial_regularization(labels, len(data))
    
    def _handle_boundaries(self, centroids: np.ndarray) -> None:
        """Handle boundary conditions for centroids"""
        # Clip to reasonable range
        np.clip(centroids, -10.0, 10.0, out=centroids)
    
    def condition_on_meta_features(self, meta_features: Dict[str, float]) -> None:
        """Condition parameters on meta-features"""
        complexity = meta_features.get('fractal_dimension', 1.5)
        
        # Adjust parameters based on complexity
        if complexity > 1.5:
            # Complex regions: more exploration
            self.sspso_config.spatial_weight = min(1.5, self.sspso_config.spatial_weight * 1.2)
            self.sspso_config.cognitive_weight = 1.8
            self.sspso_config.social_weight = 1.8
        elif complexity < 1.2:
            # Simple regions: more exploitation
            self.sspso_config.spatial_weight = max(0.5, self.sspso_config.spatial_weight * 0.8)
            self.sspso_config.cognitive_weight = 1.2
            self.sspso_config.social_weight = 1.2
        
        # Enforce convergence constraint
        max_spatial_coeff = (1 - self.sspso_config.inertia_min) / 2
        if self.sspso_config.spatial_weight > max_spatial_coeff:
            self.sspso_config.spatial_weight = max_spatial_coeff
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters"""
        return self.sspso_config.__dict__
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set parameters"""
        for key, value in params.items():
            if hasattr(self.sspso_config, key):
                setattr(self.sspso_config, key, value)
    
    def get_complexity(self) -> float:
        """Get computational complexity estimate"""
        # O(n_particles * max_iterations * n_samples * n_clusters * n_features)
        base_complexity = 1.0
        
        # Adjust based on variant
        variant_complexity = {
            "fast": 0.3,
            "accurate": 2.0,
            "spatial": 1.2,
            "spectral": 1.2
        }
        
        multiplier = variant_complexity.get(self.sspso_config.variant, 1.0)
        return base_complexity * multiplier
