#!/usr/bin/env python3
"""
Transfer Learning Adapter for Holistic Gradient Operator
Supports cross-task adaptation for anomaly detection, spectral unmixing, 
material mapping, and change detection using frozen spectral gradient layers.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class GradientTransferAdapter:
    """
    Adapts the holistic gradient operator to new hyperspectral tasks by fine-tuning
    only the fusion MLP while keeping spectral gradient layers frozen.
    """
    
    def __init__(self, source_model_path: Path, device: torch.device = None):
        """
        Initialize adapter with source segmentation model.
        
        Args:
            source_model_path: Path to trained gradient operator checkpoint
            device: Target device for inference/training
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load source model (trained on segmentation)
        logger.info(f"Loading source model from {source_model_path}")
        checkpoint = torch.load(source_model_path, map_location=self.device)
        
        # Rebuild gradient operator architecture
        from src.operators.gradient import HolisticGradientOperator
        self.model = HolisticGradientOperator(
            n_bands=checkpoint['config']['n_bands'],
            n_meta_features=checkpoint['config']['n_meta_features']
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()
        
        # Freeze spectral gradient layers (universal feature extractors)
        self._freeze_gradient_layers()
        
        # Retrainable components
        self.trainable_params = self.model.fusion_mlp.parameters()
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.trainable_params):,}")
        
        # Task-specific configurations
        self.task_configs = {
            'anomaly': {'loss': 'bce', 'weight': 0.4},
            'unmixing': {'loss': 'mse', 'weight': 1.0},
            'material': {'loss': 'ce', 'weight': 1.0},
            'change': {'loss': 'mse', 'weight': 0.8}
        }
        
    def _freeze_gradient_layers(self):
        """Freeze all spectral gradient computation layers."""
        frozen_layers = ['spatial_conv', 'spectral_conv', 'mixed_conv', 
                        'scale_fusion', 'gradient_norm']
        
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in frozen_layers):
                param.requires_grad = False
                logger.debug(f"Frozen layer: {name}")
            else:
                logger.debug(f"Trainable layer: {name}")
    
    def _task_specific_fusion(self, spatial_grad: torch.Tensor, 
                             spectral_grad: torch.Tensor, 
                             mixed_grad: torch.Tensor, 
                             weights: torch.Tensor, 
                             task: str) -> torch.Tensor:
        """
        Apply task-specific gradient fusion weights.
        
        Args:
            spatial_grad: Spatial gradient map [B, H, W]
            spectral_grad: Spectral gradient map [B, H, W]
            mixed_grad: Mixed gradient map [B, H, W]
            weights: Meta-feature conditioned weights [B, 3]
            task: Target task name
        """
        # Base weighted sum
        fused = (weights[:, 0:1] * spatial_grad + 
                 weights[:, 1:2] * spectral_grad + 
                 weights[:, 2:3] * mixed_grad)
        
        # Task-specific post-processing
        if task == 'anomaly':
            # Enhance discontinuities via sigmoid scaling
            fused = torch.sigmoid(fused / 0.1)  # Temperature scaling
            
        elif task == 'unmixing':
            # Normalize to abundance constraints [0,1]
            fused = torch.clamp(fused, 0, 1)
            
        elif task == 'change':
            # Highlight magnitude changes
            fused = torch.abs(fused - fused.mean())
            
        return fused
    
    def task_loss(self, prediction: torch.Tensor, 
                  target: torch.Tensor, 
                  task: str) -> torch.Tensor:
        """Compute task-specific loss."""
        if task in ['anomaly', 'material']:
            # Binary or multi-class segmentation
            return nn.functional.binary_cross_entropy_with_logits(
                prediction, target
            )
        elif task in ['unmixing', 'change']:
            # Regression tasks
            return nn.functional.mse_loss(prediction, target)
        else:
            raise ValueError(f"Unknown task: {task}")
    
    def fine_tune(self, target_loader: torch.utils.data.DataLoader, 
                  task: str, 
                  epochs: int = 5, 
                  lr: float = 1e-4,
                  val_loader: Optional[torch.utils.data.DataLoader] = None) -> Dict[str, List[float]]:
        """
        Fine-tune the fusion MLP on target task.
        
        Args:
            target_loader: DataLoader for target task
            task: Task name ('anomaly', 'unmixing', 'material', 'change')
            epochs: Number of fine-tuning epochs
            lr: Learning rate
            val_loader: Optional validation DataLoader
            
        Returns:
            Training history dictionary
        """
        logger.info(f"Fine-tuning for task: {task} over {epochs} epochs")
        
        optimizer = Adam(self.trainable_params, lr=lr)
        history = {'loss': [], 'val_loss': []}
        
        self.model.train()
        
        for epoch in range(epochs):
            epoch_losses = []
            
            for batch_idx, batch in enumerate(target_loader):
                # Expect batch: {'tile': tensor, 'meta': tensor, 'label': tensor}
                tiles = batch['tile'].to(self.device)
                meta_features = batch['meta'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass through frozen layers
                spatial_grad, spectral_grad, mixed_grad = self.model.gradient_layers(tiles)
                
                # Get meta-feature weights
                weights = self.model.fusion_mlp(meta_features)
                
                # Task-specific fusion
                fused = self._task_specific_fusion(
                    spatial_grad, spectral_grad, mixed_grad, weights, task
                )
                
                # Compute loss
                loss = self.task_loss(fused, labels, task)
                
                # Backward pass (only fusion MLP)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_losses.append(loss.item())
            
            avg_loss = torch.tensor(epoch_losses).mean().item()
            history['loss'].append(avg_loss)
            
            # Validation
            if val_loader:
                val_loss = self._validate(val_loader, task)
                history['val_loss'].append(val_loss)
                logger.info(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, val_loss={val_loss:.4f}")
            else:
                logger.info(f"Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}")
        
        self.model.eval()
        return history
    
    def _validate(self, val_loader: torch.utils.data.DataLoader, task: str) -> float:
        """Compute validation loss."""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for batch in val_loader:
                tiles = batch['tile'].to(self.device)
                meta_features = batch['meta'].to(self.device)
                labels = batch['label'].to(self.device)
                
                spatial_grad, spectral_grad, mixed_grad = self.model.gradient_layers(tiles)
                weights = self.model.fusion_mlp(meta_features)
                fused = self._task_specific_fusion(spatial_grad, spectral_grad, mixed_grad, weights, task)
                
                loss = self.task_loss(fused, labels, task)
                val_losses.append(loss.item())
        
        self.model.train()
        return torch.tensor(val_losses).mean().item()
    
    def save_adapter(self, save_path: Path, task: str, metrics: Dict = None):
        """Save fine-tuned adapter state."""
        save_dict = {
            'model_state': self.model.state_dict(),
            'task': task,
            'frozen_layers': ['gradient_layers'],
            'trainable_layers': ['fusion_mlp'],
            'metrics': metrics or {}
        }
        torch.save(save_dict, save_path)
        logger.info(f"Saved adapter to {save_path}")
    
    def load_adapter(self, load_path: Path):
        """Load fine-tuned adapter state."""
        checkpoint = torch.load(load_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        logger.info(f"Loaded adapter from {load_path} for task: {checkpoint['task']}")

# Example usage
if __name__ == "__main__":
    # Initialize source model (segmentation)
    adapter = GradientTransferAdapter(
        source_model_path=Path("checkpoints/gradient_segmentation.pt"),
        device=torch.device("cuda")
    )
    
    # Load target data (anomaly detection)
    from src.data.hyperspectral_loader import AnomalyDetectionLoader
    
    target_loader = AnomalyDetectionLoader(
        data_path=Path("data/botswana_anomalies.h5"),
        batch_size=32
    )
    
    # Fine-tune
    history = adapter.fine_tune(
        target_loader=target_loader,
        task='anomaly',
        epochs=5,
        lr=1e-4
    )
    
    # Save adapted model
    adapter.save_adapter(
        save_path=Path("checkpoints/gradient_anomaly.pt"),
        task='anomaly',
        metrics={'final_loss': history['loss'][-1]}
    )
