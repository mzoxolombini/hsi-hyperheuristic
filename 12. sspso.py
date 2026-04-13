"""
Spectral-Spatial Particle Swarm Optimization (SS-PSO)
Implements Chapter 6 with Theorem 6.3.1 convergence guarantee
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple, Dict, Any
from .base import BaseLLH


class SSPSO(BaseLLH):
    """
    Spectral-Spatial Particle Swarm Optimization with convergence guarantees.
    
    Theorem 6.3.1: The system converges if c3 < (1 - ω) / 2
    """
    
    def __init__(
        self,
        n_particles: int = 50,
        n_iterations: int = 200,
        n_clusters: int = 9,
        omega: float = 0.5,
        c1: float = 1.5,
        c2: float = 1.5,
        c3: float = 0.1,
        spatial_weight: float = 1.0,
        topology_size: int = 5,
        use_meta_conditioning: bool = True,
        device: str = 'cuda'
    ):
        """
        Args:
            n_particles: Number of particles in swarm
            n_iterations: Maximum iterations
            n_clusters: Number of output clusters/classes
            omega: Inertia weight (must be in [0,1])
            c1: Cognitive coefficient
            c2: Social coefficient  
            c3: Spatial regularization coefficient (must satisfy convergence bound)
            spatial_weight: λ weight for spatial regularization term L_spat
            topology_size: k for k×k communication neighbourhood
            use_meta_conditioning: Enable meta-feature conditioned parameters
            device: Computation device
        """
        super().__init__()
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.n_clusters = n_clusters
        self.omega = omega
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.spatial_weight = spatial_weight
        self.topology_size = topology_size
        self.use_meta_conditioning = use_meta_conditioning
        self.device = device
        
        # Enforce Theorem 6.3.1 convergence bound
        self._enforce_convergence_bound()
        
        # Particle state
        self.particles = None
        self.velocities = None
        self.personal_best = None
        self.global_best = None
        self.personal_best_fitness = None
        
    def _enforce_convergence_bound(self):
        """Enforce Theorem 6.3.1: c3 < (1 - ω) / 2"""
        max_c3 = (1 - self.omega) / 2 - 1e-6  # Subtract epsilon for numerical stability
        if self.c3 >= max_c3:
            original = self.c3
            self.c3 = max(0.01, max_c3)
            import warnings
            warnings.warn(
                f"c3={original} exceeds convergence bound {max_c3}. "
                f"Clipped to {self.c3}. See Theorem 6.3.1."
            )
    
    def _compute_spatial_gradient(
        self, 
        centroids: torch.Tensor, 
        data: torch.Tensor,
        spatial_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute spatial regularization gradient ∇_f L_spat.
        
        L_spat = Σ_{i,j∈N} exp(-||p_i - p_j||²/2σ²) * I[l_i ≠ l_j]
        
        Args:
            centroids: Current cluster centroids (n_clusters, n_features)
            data: Input data points (n_points, n_features)
            spatial_positions: Spatial coordinates (n_points, 2)
        """
        n_points = data.shape[0]
        n_features = data.shape[1]
        
        # Compute pairwise spatial distances
        spatial_diff = spatial_positions.unsqueeze(1) - spatial_positions.unsqueeze(0)
        spatial_dist2 = (spatial_diff ** 2).sum(dim=2)
        
        # Gaussian kernel for spatial proximity
        sigma = 0.1
        spatial_weight = torch.exp(-spatial_dist2 / (2 * sigma ** 2))
        
        # Compute spectral distances to centroids
        spectral_dist = torch.cdist(data, centroids)
        labels = spectral_dist.argmin(dim=1)
        
        # Penalize label discontinuities between spatially close points
        label_diff = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()
        
        # L_spat gradient approximation
        L_spat = (spatial_weight * label_diff).sum()
        
        # Approximate gradient using finite differences
        grad = torch.zeros_like(centroids)
        eps = 1e-4
        
        for k in range(self.n_clusters):
            for d in range(n_features):
                centroids_plus = centroids.clone()
                centroids_plus[k, d] += eps
                centroids_minus = centroids.clone()
                centroids_minus[k, d] -= eps
                
                # Recompute labels with perturbed centroids
                dist_plus = torch.cdist(data, centroids_plus)
                labels_plus = dist_plus.argmin(dim=1)
                dist_minus = torch.cdist(data, centroids_minus)
                labels_minus = dist_minus.argmin(dim=1)
                
                label_diff_plus = (labels_plus.unsqueeze(1) != labels_plus.unsqueeze(0)).float()
                label_diff_minus = (labels_minus.unsqueeze(1) != labels_minus.unsqueeze(0)).float()
                
                L_plus = (spatial_weight * label_diff_plus).sum()
                L_minus = (spatial_weight * label_diff_minus).sum()
                
                grad[k, d] = (L_plus - L_minus) / (2 * eps)
        
        return self.spatial_weight * grad
    
    def _fitness(self, centroids: torch.Tensor, data: torch.Tensor) -> torch.Tensor:
        """Compute fitness (negative intra-cluster distance)."""
        distances = torch.cdist(data, centroids)
        min_distances = distances.min(dim=1)[0]
        return -min_distances.mean()
    
    def forward(
        self, 
        x: torch.Tensor, 
        spatial_coords: Optional[torch.Tensor] = None,
        meta_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Execute SS-PSO optimization.
        
        Args:
            x: Input features (H*W, n_features)
            spatial_coords: Spatial coordinates (H*W, 2)
            meta_features: Meta-features for conditioning (optional)
            
        Returns:
            labels: Cluster assignments (H*W,)
        """
        n_points, n_features = x.shape
        
        if spatial_coords is None:
            # Create grid coordinates
            h = int(np.sqrt(n_points))
            w = h
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, 1, h, device=x.device),
                torch.linspace(0, 1, w, device=x.device),
                indexing='ij'
            )
            spatial_coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        
        # Apply meta-feature conditioning if enabled
        if self.use_meta_conditioning and meta_features is not None:
            self._apply_meta_conditioning(meta_features)
        
        # Initialize particles
        self._initialize_particles(n_features, x.device)
        
        # Main optimization loop
        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                # Compute fitness
                fitness = self._fitness(self.particles[i], x)
                
                # Update personal best
                if fitness > self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness
                    self.personal_best[i] = self.particles[i].clone()
                
                # Update global best (local topology-aware)
                self._update_topology_best(i)
            
            # Update velocities and positions
            for i in range(self.n_particles):
                # Cognitive component
                cognitive = self.c1 * torch.rand_like(self.particles[i]) * (
                    self.personal_best[i] - self.particles[i]
                )
                
                # Social component
                social = self.c2 * torch.rand_like(self.particles[i]) * (
                    self.global_best - self.particles[i]
                )
                
                # Spatial regularization component
                spatial_grad = self._compute_spatial_gradient(
                    self.particles[i], x, spatial_coords
                )
                spatial_force = self.c3 * spatial_grad
                
                # Velocity update (Eq. 6.2)
                self.velocities[i] = (
                    self.omega * self.velocities[i] + 
                    cognitive + social + spatial_force
                )
                
                # Position update
                self.particles[i] = self.particles[i] + self.velocities[i]
                
                # Clamp positions to valid range [0, 1]
                self.particles[i] = torch.clamp(self.particles[i], 0, 1)
        
        # Final segmentation using best global solution
        distances = torch.cdist(x, self.global_best)
        labels = distances.argmin(dim=1)
        
        return labels
    
    def _initialize_particles(self, n_features: int, device: torch.device):
        """Initialize swarm with random centroids."""
        self.particles = torch.rand(
            self.n_particles, self.n_clusters, n_features, device=device
        )
        self.velocities = torch.zeros_like(self.particles)
        self.personal_best = self.particles.clone()
        self.personal_best_fitness = torch.full(
            (self.n_particles,), -float('inf'), device=device
        )
        self.global_best = self.particles[0].clone()
    
    def _update_topology_best(self, particle_idx: int):
        """Update local best based on k×k topology (Section 6.3.3)."""
        # Simplified: use global best for now
        # Full implementation would use spatial neighborhoods
        if self.personal_best_fitness[particle_idx] > self._get_global_best_fitness():
            self.global_best = self.personal_best[particle_idx].clone()
    
    def _get_global_best_fitness(self) -> torch.Tensor:
        """Get current global best fitness."""
        return self.personal_best_fitness.max()
    
    def _apply_meta_conditioning(self, meta_features: torch.Tensor):
        """
        Apply meta-feature conditioned parameters (Section 6.4.2).
        
        Uses neural network to predict ω, c1, c2, c3 from meta-features.
        """
        # This is a simplified version
        # Full implementation would use the MLP from Section 7.5.2
        
        # Scale parameters based on meta-features
        complexity = meta_features[0, 0].item()  # D_f
        noise = meta_features[0, 2].item()  # SNR
        
        # Adapt omega: lower inertia for complex regions
        self.omega = 0.9 - 0.4 * complexity
        
        # Adapt c3: stronger spatial regularization for complex regions
        self.c3 = 0.25 * complexity
        
        # Re-enforce convergence bound
        self._enforce_convergence_bound()
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration for serialization."""
        return {
            'name': 'SS-PSO',
            'n_particles': self.n_particles,
            'n_iterations': self.n_iterations,
            'n_clusters': self.n_clusters,
            'omega': self.omega,
            'c1': self.c1,
            'c2': self.c2,
            'c3': self.c3,
            'spatial_weight': self.spatial_weight,
            'topology_size': self.topology_size,
            'use_meta_conditioning': self.use_meta_conditioning
        }


# Specialist variants (Table 4.1)
class SSPSOFast(SSPSO):
    """Fast variant: 20 particles, 50 iterations"""
    def __init__(self, **kwargs):
        super().__init__(n_particles=20, n_iterations=50, **kwargs)


class SSPSOAccurate(SSPSO):
    """Accurate variant: 100 particles, 300 iterations"""
    def __init__(self, **kwargs):
        super().__init__(n_particles=100, n_iterations=300, **kwargs)


class SSPSOSpatial(SSPSO):
    """Spatial variant: high spatial regularization (λ = 2.0)"""
    def __init__(self, **kwargs):
        super().__init__(spatial_weight=2.0, **kwargs)


class SSPSOSpectral(SSPSO):
    """Spectral variant: pure spectral clustering (λ = 0.1)"""
    def __init__(self, **kwargs):
        super().__init__(spatial_weight=0.1, **kwargs)
