"""
LLH selection module
Execution Order: 28
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import random

from .network import PolicyNetwork
from llhs.base import LLHRegistry, BaseLLH
from data.meta_features import MetaFeatureExtractor
from config.constants import LLHType

logger = logging.getLogger(__name__)


class LLHSelector:
    """
    LLH selection engine
    
    Implements:
    1. Policy network-based selection
    2. Uncertainty-aware selection
    3. Exploration vs exploitation trade-off
    4. Context-aware LLH parameter conditioning
    """
    
    def __init__(self, policy_network: PolicyNetwork, config: Dict[str, Any]):
        """
        Initialize selector
        
        Args:
            policy_network: Trained policy network
            config: Selection configuration
        """
        self.policy_network = policy_network
        self.config = config
        
        # Get available LLHs
        self.available_llhs = LLHRegistry.list_available()
        self.n_llhs = len(self.available_llhs)
        
        # LLH instances cache
        self.llh_instances: Dict[str, BaseLLH] = {}
        
        # Selection strategy
        self.selection_strategy = config.get('selection_strategy', 'probabilistic')
        self.exploration_rate = config.get('exploration_rate', 0.1)
        self.temperature = config.get('temperature', 1.0)
        
        # Meta-feature extractor
        self.meta_feature_extractor = MetaFeatureExtractor()
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.policy_network.to(self.device)
        self.policy_network.eval()
        
        # Selection history
        self.selection_history = []
        
        logger.info(f"LLH selector initialized with {self.n_llhs} available LLHs")
    
    def select_llh(self, data: np.ndarray, meta_features: Optional[Dict[str, float]] = None,
                  exploration: bool = True) -> Tuple[str, Dict[str, Any]]:
        """
        Select LLH for given data
        
        Args:
            data: Input data [H, W, B]
            meta_features: Pre-computed meta-features (optional)
            exploration: Whether to use exploration
            
        Returns:
            Tuple of (selected LLH name, selection metadata)
        """
        # Extract meta-features if not provided
        if meta_features is None:
            meta_features = self.meta_feature_extractor.extract(data)
        
        # Convert meta-features to tensor
        meta_tensor = self._meta_features_to_tensor(meta_features)
        
        # Get policy probabilities
        with torch.no_grad():
            policy_probs, value = self.policy_network(meta_tensor)
            policy_probs = policy_probs.cpu().numpy().flatten()
        
        # Apply temperature scaling
        if self.temperature != 1.0:
            policy_probs = self._apply_temperature(policy_probs, self.temperature)
        
        # Selection strategy
        if self.selection_strategy == 'greedy':
            selected_idx = np.argmax(policy_probs)
            selected_llh = self.available_llhs[selected_idx]
        elif self.selection_strategy == 'probabilistic':
            # Exploration-exploitation trade-off
            if exploration and random.random() < self.exploration_rate:
                # Exploration: random selection
                selected_idx = random.randint(0, self.n_llhs - 1)
                selected_llh = self.available_llhs[selected_idx]
                selection_type = 'exploration'
            else:
                # Exploitation: sample from policy distribution
                selected_idx = np.random.choice(self.n_llhs, p=policy_probs)
                selected_llh = self.available_llhs[selected_idx]
                selection_type = 'exploitation'
        elif self.selection_strategy == 'ucb':
            # Upper Confidence Bound
            selected_idx = self._ucb_selection(policy_probs)
            selected_llh = self.available_llhs[selected_idx]
            selection_type = 'ucb'
        else:
            raise ValueError(f"Unknown selection strategy: {self.selection_strategy}")
        
        # Get or create LLH instance
        llh_instance = self._get_llh_instance(selected_llh)
        
        # Condition LLH parameters on meta-features
        if llh_instance.supports_meta_features:
            llh_instance.condition_on_meta_features(meta_features)
        
        # Prepare metadata
        metadata = {
            'selected_llh': selected_llh,
            'llh_instance': llh_instance,
            'policy_probs': policy_probs.tolist(),
            'value_estimate': value.item() if value is not None else 0.0,
            'meta_features': meta_features,
            'selection_type': selection_type if 'selection_type' in locals() else 'greedy',
            'selected_probability': policy_probs[selected_idx],
            'entropy': self._calculate_entropy(policy_probs),
            'confidence': np.max(policy_probs)
        }
        
        # Update selection history
        self.selection_history.append({
            'selected_llh': selected_llh,
            'meta_features': meta_features,
            'policy_probs': policy_probs.tolist(),
            'timestamp': np.datetime64('now')
        })
        
        # Limit history size
        if len(self.selection_history) > 1000:
            self.selection_history = self.selection_history[-1000:]
        
        logger.debug(f"Selected LLH: {selected_llh} (prob: {policy_probs[selected_idx]:.3f})")
        
        return selected_llh, metadata
    
    def select_llh_batch(self, data_batch: List[np.ndarray],
                        meta_features_batch: Optional[List[Dict[str, float]]] = None,
                        exploration: bool = True) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Select LLHs for batch of data
        
        Args:
            data_batch: List of input data
            meta_features_batch: List of meta-features (optional)
            exploration: Whether to use exploration
            
        Returns:
            List of (selected LLH name, selection metadata)
        """
        batch_size = len(data_batch)
        results = []
        
        # Extract meta-features if not provided
        if meta_features_batch is None:
            meta_features_batch = []
            for data in data_batch:
                meta_features = self.meta_feature_extractor.extract(data)
                meta_features_batch.append(meta_features)
        
        # Convert to batch tensor
        meta_tensors = []
        for meta_features in meta_features_batch:
            meta_tensor = self._meta_features_to_tensor(meta_features)
            meta_tensors.append(meta_tensor)
        
        meta_batch = torch.stack(meta_tensors).to(self.device)
        
        # Get batch policy probabilities
        with torch.no_grad():
            policy_probs_batch, values_batch = self.policy_network(meta_batch)
            policy_probs_batch = policy_probs_batch.cpu().numpy()
            values_batch = values_batch.cpu().numpy().flatten() if values_batch is not None else None
        
        # Process each sample in batch
        for i in range(batch_size):
            policy_probs = policy_probs_batch[i]
            meta_features = meta_features_batch[i]
            
            # Apply temperature scaling
            if self.temperature != 1.0:
                policy_probs = self._apply_temperature(policy_probs, self.temperature)
            
            # Selection
            if self.selection_strategy == 'greedy':
                selected_idx = np.argmax(policy_probs)
                selected_llh = self.available_llhs[selected_idx]
                selection_type = 'greedy'
            elif self.selection_strategy == 'probabilistic':
                if exploration and random.random() < self.exploration_rate:
                    selected_idx = random.randint(0, self.n_llhs - 1)
                    selected_llh = self.available_llhs[selected_idx]
                    selection_type = 'exploration'
                else:
                    selected_idx = np.random.choice(self.n_llhs, p=policy_probs)
                    selected_llh = self.available_llhs[selected_idx]
                    selection_type = 'exploitation'
            else:
                selected_idx = 0  # Default
                selected_llh = self.available_llhs[selected_idx]
                selection_type = 'default'
            
            # Get LLH instance
            llh_instance = self._get_llh_instance(selected_llh)
            
            # Condition parameters
            if llh_instance.supports_meta_features:
                llh_instance.condition_on_meta_features(meta_features)
            
            # Prepare metadata
            metadata = {
                'selected_llh': selected_llh,
                'llh_instance': llh_instance,
                'policy_probs': policy_probs.tolist(),
                'value_estimate': values_batch[i] if values_batch is not None else 0.0,
                'meta_features': meta_features,
                'selection_type': selection_type,
                'selected_probability': policy_probs[selected_idx],
                'entropy': self._calculate_entropy(policy_probs),
                'confidence': np.max(policy_probs)
            }
            
            results.append((selected_llh, metadata))
        
        return results
    
    def _meta_features_to_tensor(self, meta_features: Dict[str, float]) -> torch.Tensor:
        """Convert meta-features dictionary to tensor"""
        # Get feature names in consistent order
        feature_names = self.meta_feature_extractor.feature_names
        feature_vector = [meta_features.get(name, 0.0) for name in feature_names]
        
        return torch.FloatTensor(feature_vector).unsqueeze(0).to(self.device)
    
    def _apply_temperature(self, probs: np.ndarray, temperature: float) -> np.ndarray:
        """Apply temperature scaling to probabilities"""
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        # Apply temperature
        scaled_probs = np.exp(np.log(probs + 1e-10) / temperature)
        
        # Renormalize
        scaled_probs = scaled_probs / (scaled_probs.sum() + 1e-10)
        
        return scaled_probs
    
    def _ucb_selection(self, probs: np.ndarray) -> int:
        """Upper Confidence Bound selection"""
        # Calculate UCB scores
        ucb_scores = []
        total_selections = sum(1 for h in self.selection_history if h.get('count', 0) > 0)
        
        for i in range(self.n_llhs):
            # Count selections for this LLH
            llh_selections = sum(1 for h in self.selection_history 
                               if h.get('selected_llh') == self.available_llhs[i])
            
            # Calculate UCB
            if llh_selections == 0:
                ucb = float('inf')  # Encourage exploration of unselected LLHs
            else:
                # UCB = mean_reward + exploration_bonus
                mean_reward = probs[i]
                exploration_bonus = np.sqrt(2 * np.log(total_selections + 1) / llh_selections)
                ucb = mean_reward + self.exploration_rate * exploration_bonus
            
            ucb_scores.append(ucb)
        
        return np.argmax(ucb_scores)
    
    def _get_llh_instance(self, llh_name: str) -> BaseLLH:
        """Get or create LLH instance"""
        if llh_name not in self.llh_instances:
            # Create new instance with default config
            config = self.config.get('llh_configs', {}).get(llh_name, {})
            llh_instance = LLHRegistry.create(llh_name, config)
            self.llh_instances[llh_name] = llh_instance
        
        return self.llh_instances[llh_name]
    
    def _calculate_entropy(self, probs: np.ndarray) -> float:
        """Calculate entropy of probability distribution"""
        # Clip probabilities to avoid log(0)
        probs = np.clip(probs, 1e-10, 1.0)
        return -np.sum(probs * np.log(probs))
    
    def update_with_feedback(self, selected_llh: str, performance: float,
                           meta_features: Dict[str, float]) -> None:
        """
        Update selector with performance feedback
        
        Args:
            selected_llh: LLH that was selected
            performance: Performance metric (e.g., IoU)
            meta_features: Meta-features of the data
        """
        # Store feedback for potential online learning
        feedback = {
            'selected_llh': selected_llh,
            'performance': performance,
            'meta_features': meta_features,
            'timestamp': np.datetime64('now')
        }
        
        # In a more advanced implementation, this would update the policy network
        # For now, just log the feedback
        logger.debug(f"Feedback received: {selected_llh} -> {performance:.4f}")
    
    def get_selection_statistics(self) -> Dict[str, Any]:
        """Get selection statistics"""
        if not self.selection_history:
            return {}
        
        # Count selections per LLH
        selection_counts = {}
        for llh_name in self.available_llhs:
            count = sum(1 for h in self.selection_history 
                       if h.get('selected_llh') == llh_name)
            selection_counts[llh_name] = count
        
        # Calculate average entropy and confidence
        entropies = []
        confidences = []
        
        for h in self.selection_history:
            if 'policy_probs' in h:
                probs = np.array(h['policy_probs'])
                entropies.append(self._calculate_entropy(probs))
                confidences.append(np.max(probs))
        
        stats = {
            'total_selections': len(self.selection_history),
            'selection_counts': selection_counts,
            'avg_entropy': np.mean(entropies) if entropies else 0.0,
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'exploration_rate': self.exploration_rate,
            'selection_strategy': self.selection_strategy
        }
        
        return stats
    
    def reset_history(self) -> None:
        """Reset selection history"""
        self.selection_history = []
        logger.info("Selection history reset")
    
    def save_state(self, filepath: str) -> None:
        """Save selector state"""
        import pickle
        
        state = {
            'selection_history': self.selection_history,
            'llh_instances': {},  # Don't save LLH instances (recreate as needed)
            'config': self.config,
            'available_llhs': self.available_llhs
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Selector state saved to {filepath}")
    
    def load_state(self, filepath: str) -> None:
        """Load selector state"""
        import pickle
        
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.selection_history = state.get('selection_history', [])
        self.config = state.get('config', self.config)
        
        logger.info(f"Selector state loaded from {filepath}")
