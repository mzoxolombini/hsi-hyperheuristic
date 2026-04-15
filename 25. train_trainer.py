"""
Policy network trainer module
Execution Order: 27
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import logging
import numpy as np
from tqdm import tqdm
import time
import os

from .network import PolicyNetwork
from data.dataset_loader import HSIDataset
from llhs.base import LLHRegistry
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


class PolicyTrainer:
    """
    Trainer for policy network
    
    Implements:
    1. Hindsight experience replay
    2. Cross-entropy with entropy regularization
    3. Gradient clipping
    4. Learning rate scheduling
    """
    
    def __init__(self, network: PolicyNetwork, config: Dict[str, Any]):
        """
        Initialize trainer
        
        Args:
            network: Policy network
            config: Training configuration
        """
        self.network = network
        self.config = config
        
        # Training parameters
        self.n_epochs = config.get('epochs', 50)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 1e-3)
        self.weight_decay = config.get('weight_decay', 1e-4)
        self.entropy_coef = config.get('entropy_coef', 0.1)
        
        # Setup optimizer
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # Loss function
        self.criterion = nn.BCELoss()  # Binary cross-entropy for multi-label
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network.to(self.device)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        logger.info(f"Policy trainer initialized on {self.device}")
    
    def train(self, train_dataset: HSIDataset, val_dataset: Optional[HSIDataset] = None,
              n_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Train policy network
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset (optional)
            n_epochs: Number of epochs (overrides config)
            
        Returns:
            Training statistics
        """
        if n_epochs is not None:
            self.n_epochs = n_epochs
        
        logger.info(f"Starting policy training for {self.n_epochs} epochs")
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # Training loop
        best_val_accuracy = 0.0
        best_model_state = None
        
        for epoch in range(self.n_epochs):
            epoch_start = time.time()
            
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            if val_dataset:
                val_metrics = self._validate_epoch(val_loader)
                
                # Update learning rate
                self.scheduler.step(val_metrics['accuracy'])
                
                # Save best model
                if val_metrics['accuracy'] > best_val_accuracy:
                    best_val_accuracy = val_metrics['accuracy']
                    best_model_state = self.network.state_dict().copy()
                    logger.info(f"  New best validation accuracy: {best_val_accuracy:.4f}")
            else:
                val_metrics = {'loss': 0.0, 'accuracy': 0.0}
            
            # Log epoch results
            epoch_time = time.time() - epoch_start
            
            self._log_epoch_results(epoch, train_metrics, val_metrics, epoch_time)
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch + 1, train_metrics, val_metrics)
        
        # Load best model
        if best_model_state is not None:
            self.network.load_state_dict(best_model_state)
            logger.info(f"Loaded best model with validation accuracy: {best_val_accuracy:.4f}")
        
        # Save final model
        self._save_model()
        
        return self.history
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.network.train()
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.n_epochs} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Get batch data
            patches = batch['patch'].to(self.device)
            meta_features = batch['meta_features'].to(self.device)
            ground_truth = batch['ground_truth'].to(self.device)
            
            # Get hindsight optimal LLH (simplified - in practice, use actual evaluation)
            optimal_llh = self._get_hindsight_optimal_llh(patches, ground_truth)
            
            # Convert to target probabilities
            target = self._llh_to_target(optimal_llh)
            
            # Forward pass
            self.optimizer.zero_grad()
            policy_probs, _ = self.network(meta_features)
            
            # Calculate loss
            bce_loss = self.criterion(policy_probs, target)
            
            # Entropy regularization
            entropy = -torch.mean(torch.sum(policy_probs * torch.log(policy_probs + 1e-10), dim=1))
            total_loss = bce_loss - self.entropy_coef * entropy
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Calculate accuracy
            predicted_llh = torch.argmax(policy_probs, dim=1)
            actual_llh = torch.argmax(target, dim=1)
            accuracy = (predicted_llh == actual_llh).float().mean().item()
            
            # Update metrics
            epoch_loss += total_loss.item()
            epoch_accuracy += accuracy
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss.item(),
                'acc': accuracy
            })
        
        # Calculate epoch averages
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_accuracy = epoch_accuracy / max(n_batches, 1)
        
        return {'loss': avg_loss, 'accuracy': avg_accuracy}
    
    def _validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.network.eval()
        
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Get batch data
                patches = batch['patch'].to(self.device)
                meta_features = batch['meta_features'].to(self.device)
                ground_truth = batch['ground_truth'].to(self.device)
                
                # Get hindsight optimal LLH
                optimal_llh = self._get_hindsight_optimal_llh(patches, ground_truth)
                
                # Convert to target probabilities
                target = self._llh_to_target(optimal_llh)
                
                # Forward pass
                policy_probs, _ = self.network(meta_features)
                
                # Calculate loss
                bce_loss = self.criterion(policy_probs, target)
                
                # Entropy regularization
                entropy = -torch.mean(torch.sum(policy_probs * torch.log(policy_probs + 1e-10), dim=1))
                total_loss = bce_loss - self.entropy_coef * entropy
                
                # Calculate accuracy
                predicted_llh = torch.argmax(policy_probs, dim=1)
                actual_llh = torch.argmax(target, dim=1)
                accuracy = (predicted_llh == actual_llh).float().mean().item()
                
                # Update metrics
                epoch_loss += total_loss.item()
                epoch_accuracy += accuracy
                n_batches += 1
        
        # Calculate epoch averages
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_accuracy = epoch_accuracy / max(n_batches, 1)
        
        return {'loss': avg_loss, 'accuracy': avg_accuracy}
    
    def _get_hindsight_optimal_llh(self, patches: torch.Tensor, 
                                  ground_truth: torch.Tensor) -> List[str]:
        """
        Get hindsight optimal LLH for each patch
        
        Note: In practice, this would evaluate each LLH on the patch
        and select the one with highest IoU. This is a simplified version.
        
        Args:
            patches: Batch of patches [B, C, H, W]
            ground_truth: Ground truth labels [B, H, W]
            
        Returns:
            List of optimal LLH names for each patch
        """
        batch_size = patches.shape[0]
        optimal_llhs = []
        
        # Simplified: use meta-features to decide optimal LLH
        for i in range(batch_size):
            patch_np = patches[i].cpu().numpy()
            gt_np = ground_truth[i].cpu().numpy()
            
            # Extract simple features to decide LLH
            mean_intensity = np.mean(patch_np)
            texture_complexity = np.std(patch_np)
            
            # Simple rules (in practice, use actual evaluation)
            if texture_complexity > 0.3:
                # Complex texture: use gradient-based method
                optimal_llhs.append('gradient_medium')
            elif mean_intensity > 0.5:
                # High intensity: use watershed
                optimal_llhs.append('watershed')
            else:
                # Default: use K-means
                optimal_llhs.append('kmeans')
        
        return optimal_llhs
    
    def _llh_to_target(self, llh_names: List[str]) -> torch.Tensor:
        """
        Convert LLH names to target probability tensor
        
        Args:
            llh_names: List of LLH names
            
        Returns:
            Target tensor [B, n_llhs]
        """
        # Get all available LLHs
        all_llhs = LLHRegistry.list_available()
        n_llhs = len(all_llhs)
        
        batch_size = len(llh_names)
        target = torch.zeros((batch_size, n_llhs), device=self.device)
        
        for i, llh_name in enumerate(llh_names):
            if llh_name in all_llhs:
                idx = all_llhs.index(llh_name)
                target[i, idx] = 1.0
            else:
                # Default: uniform distribution
                target[i, :] = 1.0 / n_llhs
        
        return target
    
    def _log_epoch_results(self, epoch: int, train_metrics: Dict[str, float],
                          val_metrics: Dict[str, float], epoch_time: float) -> None:
        """Log epoch results"""
        # Update history
        self.history['train_loss'].append(train_metrics['loss'])
        self.history['train_accuracy'].append(train_metrics['accuracy'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_accuracy'].append(val_metrics['accuracy'])
        self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
        
        # Log to console
        logger.info(f"Epoch {epoch+1}/{self.n_epochs}: "
                   f"Train Loss={train_metrics['loss']:.4f}, "
                   f"Train Acc={train_metrics['accuracy']:.4f}, "
                   f"Val Loss={val_metrics['loss']:.4f}, "
                   f"Val Acc={val_metrics['accuracy']:.4f}, "
                   f"Time={epoch_time:.1f}s, "
                   f"LR={self.optimizer.param_groups[0]['lr']:.6f}")
    
    def _save_checkpoint(self, epoch: int, train_metrics: Dict[str, float],
                        val_metrics: Dict[str, float]) -> None:
        """Save training checkpoint"""
        checkpoint_dir = os.path.join(self.config.get('checkpoints_dir', './checkpoints'), 'policy')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f'policy_epoch_{epoch}.pt')
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'history': self.history,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def _save_model(self) -> None:
        """Save final model"""
        model_dir = self.config.get('models_dir', './models')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, 'policy_network_final.pt')
        
        torch.save({
            'model_state_dict': self.network.state_dict(),
            'config': self.config,
            'history': self.history
        }, model_path)
        
        logger.info(f"Final model saved to {model_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint['epoch']})")
    
    def plot_training_history(self, save_path: Optional[str] = None) -> None:
        """Plot training history"""
        import matplotlib.pyplot as plt
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot losses
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot accuracies
        axes[0, 1].plot(epochs, self.history['train_accuracy'], 'b-', label='Train Accuracy')
        axes[0, 1].plot(epochs, self.history['val_accuracy'], 'r-', label='Val Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot learning rate
        axes[1, 0].plot(epochs, self.history['learning_rates'], 'g-')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # Plot entropy (if available)
        if 'entropy' in self.history:
            axes[1, 1].plot(epochs, self.history['entropy'], 'm-')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Entropy')
            axes[1, 1].set_title('Policy Entropy')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        if not self.history['train_loss']:
            return {}
        
        stats = {
            'final_train_loss': self.history['train_loss'][-1],
            'final_train_accuracy': self.history['train_accuracy'][-1],
            'final_val_loss': self.history['val_loss'][-1],
            'final_val_accuracy': self.history['val_accuracy'][-1],
            'best_val_accuracy': max(self.history['val_accuracy']) if self.history['val_accuracy'] else 0.0,
            'n_epochs': len(self.history['train_loss']),
            'final_learning_rate': self.history['learning_rates'][-1]
        }
        
        return stats
