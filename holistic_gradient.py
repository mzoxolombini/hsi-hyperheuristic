"""
Holistic Spectral-Spatial Gradient Operator (Chapter 7)
Implements: |∇_holistic| = sqrt(α|∇_{x,y}I|² + β|∇_λI|² + γ|∇_{x,y,λ}I|²)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from ..features.meta_features import MetaFeatureExtractor


class HolisticGradient(nn.Module):
    """
    Holistic edge detection operator for hyperspectral imagery.
    
    Combines spatial, spectral, and mixed partial derivatives with
    meta-feature-conditioned weights.
    """
    
    def __init__(
        self,
        meta_extractor: MetaFeatureExtractor,
        spectral_scales: Tuple[float, float, float] = (1.0, 3.0, 7.0),
        device: str = 'cuda'
    ):
        """
        Args:
            meta_extractor: Meta-feature extractor for conditioning
            spectral_scales: (fine, medium, coarse) σ_λ values
            device: Computation device
        """
        super().__init__()
        self.meta_extractor = meta_extractor
        self.spectral_scales = spectral_scales
        self.device = device
        
        # Weight prediction MLP (Section 7.5.2)
        self.weight_mlp = nn.Sequential(
            nn.Linear(12, 24),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(24, 12),  # 3 scales × 3 weights = 9? Actually 3 weights total
            nn.Sigmoid()
        )
        
        # Spatial gradient Sobel filters
        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))
        
        # Multi-scale spectral gradient filters
        self._create_spectral_filters()
    
    def _create_spectral_filters(self):
        """Create Gaussian derivative filters for multiple scales."""
        self.spectral_filters = nn.ModuleList()
        for sigma in self.spectral_scales:
            # 1D Gaussian derivative kernel
            kernel_size = int(6 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
            kernel = -x / (sigma ** 3) * torch.exp(-x ** 2 / (2 * sigma ** 2))
            kernel = kernel / kernel.abs().sum()  # Normalize
            self.spectral_filters.append(
                nn.Conv1d(1, 1, kernel_size, padding=kernel_size//2, bias=False)
            )
            self.spectral_filters[-1].weight.data = kernel.view(1, 1, -1)
            self.spectral_filters[-1].requires_grad_(False)
    
    def _spatial_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial gradient magnitude |∇_{x,y}I|.
        
        Args:
            x: Input (B, C, H, W) - C is spectral dimension
        Returns:
            spatial_grad: (B, 1, H, W)
        """
        # Use mean spectrum for spatial gradient
        x_mean = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        
        grad_x = F.conv2d(x_mean, self.sobel_x, padding=1)
        grad_y = F.conv2d(x_mean, self.sobel_y, padding=1)
        
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
    
    def _spectral_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral gradient magnitude |∇_λI|.
        
        Uses multi-scale Gaussian derivative filters.
        
        Args:
            x: Input (B, C, H, W)
        Returns:
            spectral_grad: (B, 1, H, W) - aggregated across scales
        """
        batch_size, channels, height, width = x.shape
        
        # Reshape for 1D convolution along spectral dimension
        x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, 1, channels)  # (B*H*W, 1, C)
        
        scale_outputs = []
        for filt in self.spectral_filters:
            # Apply spectral filter
            filtered = filt(x_reshaped)  # (B*H*W, 1, C)
            # Take gradient magnitude (absolute value)
            grad_mag = torch.abs(filtered)
            # Reshape back
            grad_mag = grad_mag.mean(dim=2)  # Average across spectral dimension
            grad_mag = grad_mag.view(batch_size, height, width).unsqueeze(1)
            scale_outputs.append(grad_mag)
        
        # Aggregate across scales (max fusion)
        spectral_grad = torch.stack(scale_outputs, dim=0).max(dim=0)[0]
        return spectral_grad
    
    def _mixed_gradient(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute mixed partial derivative |∇_{x,y,λ}I|.
        
        Mixed gradient = sqrt(|∇_x(∇_λI)|² + |∇_y(∇_λI)|²)
        
        Args:
            x: Input (B, C, H, W)
        Returns:
            mixed_grad: (B, 1, H, W)
        """
        # First compute spectral gradient
        batch_size, channels, height, width = x.shape
        x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, 1, channels)
        
        # Use medium scale filter for mixed gradients
        medium_filt = self.spectral_filters[1]  # σ_λ = 3
        
        spectral_grad_vol = medium_filt(x_reshaped)  # (B*H*W, 1, C)
        spectral_grad_vol = spectral_grad_vol.abs()
        spectral_grad_vol = spectral_grad_vol.view(batch_size, height, width, channels)
        spectral_grad_vol = spectral_grad_vol.permute(0, 3, 1, 2)  # (B, C, H, W)
        
        # Spatial gradients of spectral gradient volume
        mixed_x = F.conv2d(spectral_grad_vol, self.sobel_x, padding=1, groups=channels)
        mixed_y = F.conv2d(spectral_grad_vol, self.sobel_y, padding=1, groups=channels)
        
        # Aggregate across spectral dimension
        mixed_grad = torch.sqrt(mixed_x ** 2 + mixed_y ** 2 + 1e-8)
        mixed_grad = mixed_grad.mean(dim=1, keepdim=True)
        
        return mixed_grad
    
    def forward(
        self, 
        x: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute holistic gradient magnitude.
        
        Args:
            x: Input hyperspectral patch (B, C, H, W)
            return_components: If True, also return individual components
        Returns:
            holistic_grad: (B, 1, H, W) Holistic gradient magnitude map
        """
        # Extract meta-features for conditioning
        meta_features = self.meta_extractor(x)  # (B, 12)
        
        # Predict conditioned weights α, β, γ (Eq. 7.3)
        weights = self.weight_mlp(meta_features)  # (B, 3)
        alpha = weights[:, 0:1]  # (B, 1)
        beta = weights[:, 1:2]
        gamma = weights[:, 2:3]
        
        # Reshape weights for broadcasting
        alpha = alpha.view(-1, 1, 1, 1)
        beta = beta.view(-1, 1, 1, 1)
        gamma = gamma.view(-1, 1, 1, 1)
        
        # Compute individual gradient components
        spatial_grad = self._spatial_gradient(x)  # (B, 1, H, W)
        spectral_grad = self._spectral_gradient(x)  # (B, 1, H, W)
        mixed_grad = self._mixed_gradient(x)  # (B, 1, H, W)
        
        # Fuse components (Eq. 7.3 from thesis)
        holistic_grad = torch.sqrt(
            alpha * spatial_grad ** 2 +
            beta * spectral_grad ** 2 +
            gamma * mixed_grad ** 2 + 1e-8
        )
        
        if return_components:
            return holistic_grad, spatial_grad, spectral_grad, mixed_grad
        
        return holistic_grad
    
    def get_optimal_scale(self, meta_features: torch.Tensor) -> int:
        """
        Complexity-guided scale selection (Section 7.5.4).
        
        Returns index of recommended scale based on fractal dimension.
        """
        df = meta_features[0, 0].item()  # Local fractal dimension
        if df > 1.5:
            return 2  # Use all scales (complex)
        elif df < 1.2:
            return 0  # Use only coarse scale (simple)
        else:
            return 1  # Use medium scale