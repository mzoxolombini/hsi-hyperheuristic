"""
Base LLH module
Execution Order: 13
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)


class BaseLLH(ABC):
    """
    Base class for all Low-Level Heuristics
    
    All LLHs must implement:
    1. __init__: Initialize with configuration
    2. apply: Apply the heuristic to data
    3. get_parameters: Return current parameters
    4. set_parameters: Update parameters
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        Initialize LLH
        
        Args:
            name: LLH name
            config: Configuration dictionary
        """
        self.name = name
        self.config = config
        self.requires_training = False
        self.supports_meta_features = False
        
    @abstractmethod
    def apply(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply LLH to data
        
        Args:
            data: Input data [H, W, B] or [N, B]
            **kwargs: Additional parameters
            
        Returns:
            Segmentation map [H, W] or labels [N]
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters"""
        pass
    
    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set parameters"""
        pass
    
    def condition_on_meta_features(self, meta_features: Dict[str, float]) -> None:
        """
        Condition LLH parameters on meta-features
        
        Args:
            meta_features: Dictionary of meta-features
        """
        if self.supports_meta_features:
            self._update_parameters_from_meta_features(meta_features)
    
    def _update_parameters_from_meta_features(self, meta_features: Dict[str, float]) -> None:
        """Update parameters based on meta-features (to be implemented by subclasses)"""
        pass
    
    def get_complexity(self) -> float:
        """
        Get computational complexity estimate
        
        Returns:
            Complexity score (higher = more complex)
        """
        return 1.0  # Default
    
    def get_memory_usage(self, data_shape: Tuple[int, ...]) -> float:
        """
        Estimate memory usage in MB
        
        Args:
            data_shape: Shape of input data
            
        Returns:
            Estimated memory usage in MB
        """
        h, w, b = data_shape if len(data_shape) == 3 else (data_shape[0], 1, data_shape[1])
        return (h * w * b * 4) / (1024 * 1024)  # 4 bytes per float32
    
    def validate_input(self, data: np.ndarray) -> bool:
        """
        Validate input data
        
        Args:
            data: Input data
            
        Returns:
            True if valid, False otherwise
        """
        if data is None:
            logger.error("Input data is None")
            return False
        
        if not isinstance(data, np.ndarray):
            logger.error(f"Input data must be numpy array, got {type(data)}")
            return False
        
        if data.ndim not in [2, 3]:
            logger.error(f"Input data must be 2D or 3D, got shape {data.shape}")
            return False
        
        if np.any(np.isnan(data)):
            logger.error("Input data contains NaN values")
            return False
        
        if np.any(np.isinf(data)):
            logger.error("Input data contains Inf values")
            return False
        
        return True


class LLHRegistry:
    """
    Registry for LLH classes
    """
    
    _registry = {}
    
    @classmethod
    def register(cls, name: str, llh_class: type) -> None:
        """
        Register an LLH class
        
        Args:
            name: LLH name
            llh_class: LLH class
        """
        if not issubclass(llh_class, BaseLLH):
            raise TypeError(f"{llh_class} must be a subclass of BaseLLH")
        
        cls._registry[name] = llh_class
        logger.info(f"Registered LLH: {name}")
    
    @classmethod
    def create(cls, name: str, config: Dict[str, Any]) -> BaseLLH:
        """
        Create an LLH instance
        
        Args:
            name: LLH name
            config: Configuration
            
        Returns:
            LLH instance
        """
        if name not in cls._registry:
            raise ValueError(f"LLH {name} not registered")
        
        llh_class = cls._registry[name]
        return llh_class(name, config)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """
        List all registered LLHs
        
        Returns:
            List of LLH names
        """
        return list(cls._registry.keys())
    
    @classmethod
    def get_config_template(cls, name: str) -> Dict[str, Any]:
        """
        Get configuration template for LLH
        
        Args:
            name: LLH name
            
        Returns:
            Configuration template
        """
        if name not in cls._registry:
            raise ValueError(f"LLH {name} not registered")
        
        llh_class = cls._registry[name]
        
        # Create instance with default config to get parameters
        try:
            instance = llh_class(name, {})
            return instance.get_parameters()
        except:
            return {}


def register_llh(name: str):
    """
    Decorator to register LLH classes
    
    Args:
        name: LLH name
        
    Returns:
        Decorator function
    """
    def decorator(cls):
        LLHRegistry.register(name, cls)
        return cls
    return decorator
