"""
Framework tests
Test Execution Order
"""

import unittest
import numpy as np
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config_loader import ConfigLoader
from src.framework.hyperheuristic import HyperHeuristicFramework
from src.utils.reproducibility import ReproducibilityManager


class TestHyperHeuristicFramework(unittest.TestCase):
    """Test hyper-heuristic framework"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = ConfigLoader.load_config()
        self.config.framework.n_runs = 1  # Reduce for testing
        self.config.gp.generations = 2    # Reduce for testing
        self.config.policy.epochs = 2     # Reduce for testing
        
        # Create synthetic data for testing
        self.test_image = np.random.randn(64, 64, 10).astype(np.float32)
        self.test_gt = np.random.randint(0, 5, (64, 64))
    
    def test_framework_initialization(self):
        """Test framework initialization"""
        framework = HyperHeuristicFramework(self.config)
        self.assertIsNotNone(framework)
        self.assertFalse(framework.is_trained)
    
    def test_segmentation_modes(self):
        """Test different segmentation modes"""
        framework = HyperHeuristicFramework(self.config)
        
        # Test baseline segmentation
        result = framework.segment(self.test_image, mode="baseline")
        self.assertIn('segmentation', result)
        self.assertEqual(result['segmentation'].shape, (64, 64))
    
    def test_reproducibility(self):
        """Test reproducibility manager"""
        repro = ReproducibilityManager(self.config.framework.__dict__)
        self.assertIsNotNone(repro.execution_id)
        self.assertIn('execution_id', repro.hardware_config)


if __name__ == '__main__':
    unittest.main()
