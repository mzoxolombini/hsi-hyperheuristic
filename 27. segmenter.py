"""
Segmentation engine module
Execution Order: 32
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from llhs.base import LLHRegistry, BaseLLH
from policy.selector import LLHSelector
from data.meta_features import MetaFeatureExtractor
from gp.individual import Individual
from utils.metrics import SegmentationMetrics

logger = logging.getLogger(__name__)


class AdaptiveSegmenter:
    """
    Adaptive segmentation engine
    
    Implements:
    1. Patch-based adaptive LLH selection
    2. Multi-scale segmentation
    3. Result fusion and refinement
    4. Uncertainty estimation
    """
    
    def __init__(self, llh_registry: LLHRegistry, policy_network: Any,
                 config: Dict[str, Any]):
        """
        Initialize segmenter
        
        Args:
            llh_registry: LLH registry
            policy_network: Trained policy network
            config: Configuration
        """
        self.llh_registry = llh_registry
        self.config = config
        
        # Initialize selector
        selector_config = config.get('selector', {})
        self.selector = LLHSelector(policy_network, selector_config)
        
        # Meta-feature extractor
        self.meta_extractor = MetaFeatureExtractor()
        
        # Segmentation parameters
        self.patch_size = config.get('patch_size', 64)
        self.stride = config.get('stride', 32)
        self.n_classes = config.get('n_classes', 16)
        
        # Fusion parameters
        self.fusion_method = config.get('fusion_method', 'weighted_average')
        self.refinement_enabled = config.get('refinement_enabled', True)
        
        # Performance tracking
        self.segmentation_stats = []
        
        logger.info("Adaptive segmenter initialized")
    
    def segment_adaptive(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Perform adaptive segmentation
        
        Args:
            image: Input hyperspectral image [H, W, B]
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with segmentation results
        """
        segmentation_start = time.time()
        
        h, w, b = image.shape
        
        # Extract patches
        patches, patch_coords = self._extract_patches(image)
        
        # Process patches in parallel
        patch_segmentations = self._process_patches_parallel(patches, patch_coords, **kwargs)
        
        # Fuse patch segmentations
        fused_segmentation = self._fuse_segmentations(patch_segmentations, h, w)
        
        # Apply refinement
        if self.refinement_enabled:
            fused_segmentation = self._refine_segmentation(fused_segmentation, image)
        
        # Calculate uncertainty
        uncertainty = self._calculate_uncertainty(patch_segmentations)
        
        # Create result dictionary
        segmentation_time = time.time() - segmentation_start
        
        result = {
            'segmentation': fused_segmentation,
            'uncertainty': uncertainty,
            'execution_time': segmentation_time,
            'patch_segmentations': patch_segmentations,
            'image_shape': (h, w),
            'n_patches': len(patches),
            'method': 'adaptive'
        }
        
        # Update statistics
        self._update_statistics(result)
        
        logger.info(f"Adaptive segmentation completed: {h}x{w} image, "
                   f"{len(patches)} patches, {segmentation_time:.2f}s")
        
        return result
    
    def segment_with_pipeline(self, image: np.ndarray, pipeline: Individual,
                            **kwargs) -> Dict[str, Any]:
        """
        Segment using evolved pipeline
        
        Args:
            image: Input image
            pipeline: Evolved pipeline individual
            **kwargs: Additional parameters
            
        Returns:
            Segmentation results
        """
        segmentation_start = time.time()
        
        # Execute pipeline
        from gp.evaluation import MultiObjectiveEvaluator
        
        evaluator = MultiObjectiveEvaluator(self.config)
        
        # Create dummy ground truth for evaluation
        dummy_gt = np.zeros(image.shape[:2], dtype=np.int32)
        
        # Execute pipeline
        execution_result = evaluator._execute_pipeline(pipeline.pipeline, image)
        
        if not execution_result['success']:
            logger.warning("Pipeline execution failed")
            segmentation = np.zeros(image.shape[:2], dtype=np.int32)
        else:
            segmentation = execution_result['result']
        
        segmentation_time = time.time() - segmentation_start
        
        result = {
            'segmentation': segmentation,
            'execution_time': segmentation_time,
            'pipeline': str(pipeline),
            'method': 'evolved'
        }
        
        return result
    
    def segment_baseline(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Baseline segmentation (K-means)
        
        Args:
            image: Input image
            **kwargs: Additional parameters
            
        Returns:
            Segmentation results
        """
        segmentation_start = time.time()
        
        n_clusters = kwargs.get('n_clusters', self.n_classes)
        
        # Reshape for clustering
        h, w, b = image.shape
        reshaped = image.reshape(-1, b)
        
        # Apply K-means
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = kmeans.fit_predict(reshaped)
        
        segmentation = labels.reshape(h, w)
        
        segmentation_time = time.time() - segmentation_start
        
        result = {
            'segmentation': segmentation,
            'execution_time': segmentation_time,
            'method': 'baseline',
            'algorithm': 'kmeans',
            'n_clusters': n_clusters
        }
        
        return result
    
    def _extract_patches(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """Extract overlapping patches"""
        patches = []
        coords = []
        
        h, w, _ = image.shape
        patch_size = self.patch_size
        stride = self.stride
        
        for i in range(0, h - patch_size, stride):
            for j in range(0, w - patch_size, stride):
                patch = image[i:i + patch_size, j:j + patch_size, :]
                patches.append(patch)
                coords.append((i, j))
        
        # Add border patches if needed
        if (h - patch_size) % stride != 0:
            for j in range(0, w - patch_size, stride):
                patch = image[h - patch_size:h, j:j + patch_size, :]
                patches.append(patch)
                coords.append((h - patch_size, j))
        
        if (w - patch_size) % stride != 0:
            for i in range(0, h - patch_size, stride):
                patch = image[i:i + patch_size, w - patch_size:w, :]
                patches.append(patch)
                coords.append((i, w - patch_size))
        
        # Add corner patch
        if (h - patch_size) % stride != 0 and (w - patch_size) % stride != 0:
            patch = image[h - patch_size:h, w - patch_size:w, :]
            patches.append(patch)
            coords.append((h - patch_size, w - patch_size))
        
        return patches, coords
    
    def _process_patches_parallel(self, patches: List[np.ndarray],
                                patch_coords: List[Tuple[int, int]],
                                **kwargs) -> List[Dict[str, Any]]:
        """Process patches in parallel"""
        max_workers = self.config.get('max_workers', 4)
        patch_results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit patch processing tasks
            future_to_patch = {}
            for idx, (patch, coords) in enumerate(zip(patches, patch_coords)):
                future = executor.submit(
                    self._process_single_patch,
                    patch, coords, idx, **kwargs
                )
                future_to_patch[future] = idx
            
            # Collect results as they complete
            for future in as_completed(future_to_patch):
                idx = future_to_patch[future]
                try:
                    result = future.result()
                    patch_results.append((idx, result))
                except Exception as e:
                    logger.error(f"Patch {idx} processing failed: {e}")
                    # Add empty result
                    patch_results.append((idx, {
                        'segmentation': np.zeros((self.patch_size, self.patch_size), dtype=np.int32),
                        'selected_llh': 'error',
                        'meta_features': {},
                        'execution_time': 0.0
                    }))
        
        # Sort by patch index
        patch_results.sort(key=lambda x: x[0])
        return [result for _, result in patch_results]
    
    def _process_single_patch(self, patch: np.ndarray, coords: Tuple[int, int],
                            patch_idx: int, **kwargs) -> Dict[str, Any]:
        """Process single patch"""
        patch_start = time.time()
        
        # Extract meta-features
        meta_features = self.meta_extractor.extract(patch)
        
        # Select LLH
        selected_llh, selection_metadata = self.selector.select_llh(
            patch, meta_features, exploration=True
        )
        
        # Apply selected LLH
        llh_instance = selection_metadata['llh_instance']
        
        # Prepare LLH parameters
        llh_kwargs = {'meta_features': meta_features}
        if 'n_clusters' in kwargs:
            llh_kwargs['n_clusters'] = kwargs['n_clusters']
        
        # Apply LLH
        segmentation = llh_instance.apply(patch, **llh_kwargs)
        
        # Ensure segmentation has correct shape
        if segmentation.shape != (self.patch_size, self.patch_size):
            segmentation = segmentation[:self.patch_size, :self.patch_size]
        
        patch_time = time.time() - patch_start
        
        result = {
            'segmentation': segmentation,
            'selected_llh': selected_llh,
            'meta_features': meta_features,
            'selection_metadata': selection_metadata,
            'coords': coords,
            'patch_idx': patch_idx,
            'execution_time': patch_time
        }
        
        return result
    
    def _fuse_segmentations(self, patch_results: List[Dict[str, Any]],
                          h: int, w: int) -> np.ndarray:
        """Fuse patch segmentations into full image"""
        if self.fusion_method == 'weighted_average':
            return self._weighted_average_fusion(patch_results, h, w)
        elif self.fusion_method == 'majority_voting':
            return self._majority_voting_fusion(patch_results, h, w)
        elif self.fusion_method == 'confidence_weighted':
            return self._confidence_weighted_fusion(patch_results, h, w)
        else:
            logger.warning(f"Unknown fusion method: {self.fusion_method}. Using weighted average.")
            return self._weighted_average_fusion(patch_results, h, w)
    
    def _weighted_average_fusion(self, patch_results: List[Dict[str, Any]],
                               h: int, w: int) -> np.ndarray:
        """Weighted average fusion"""
        # Initialize accumulation arrays
        segmentation_sum = np.zeros((h, w, self.n_classes), dtype=np.float32)
        weight_sum = np.zeros((h, w), dtype=np.float32)
        
        for result in patch_results:
            seg = result['segmentation']
            coords = result['coords']
            i, j = coords
            ph, pw = seg.shape
            
            # Convert segmentation to one-hot
            seg_one_hot = np.eye(self.n_classes)[seg]  # [ph, pw, n_classes]
            
            # Create weight map (Gaussian centered at patch center)
            weight_map = self._create_weight_map(ph, pw)
            
            # Accumulate
            segmentation_sum[i:i+ph, j:j+pw] += seg_one_hot * weight_map[:, :, np.newaxis]
            weight_sum[i:i+ph, j:j+pw] += weight_map
        
        # Avoid division by zero
        weight_sum = np.maximum(weight_sum, 1e-10)
        
        # Normalize
        segmentation_prob = segmentation_sum / weight_sum[:, :, np.newaxis]
        
        # Convert to labels
        segmentation = np.argmax(segmentation_prob, axis=2)
        
        return segmentation
    
    def _majority_voting_fusion(self, patch_results: List[Dict[str, Any]],
                              h: int, w: int) -> np.ndarray:
        """Majority voting fusion"""
        # Initialize voting array
        votes = np.zeros((h, w, self.n_classes), dtype=np.int32)
        
        for result in patch_results:
            seg = result['segmentation']
            coords = result['coords']
            i, j = coords
            ph, pw = seg.shape
            
            # Add votes
            for pi in range(ph):
                for pj in range(pw):
                    if i + pi < h and j + pj < w:
                        class_idx = seg[pi, pj]
                        if 0 <= class_idx < self.n_classes:
                            votes[i + pi, j + pj, class_idx] += 1
        
        # Majority vote
        segmentation = np.argmax(votes, axis=2)
        
        return segmentation
    
    def _confidence_weighted_fusion(self, patch_results: List[Dict[str, Any]],
                                  h: int, w: int) -> np.ndarray:
        """Confidence-weighted fusion"""
        # Initialize accumulation arrays
        segmentation_sum = np.zeros((h, w, self.n_classes), dtype=np.float32)
        confidence_sum = np.zeros((h, w), dtype=np.float32)
        
        for result in patch_results:
            seg = result['segmentation']
            coords = result['coords']
            confidence = result['selection_metadata'].get('confidence', 0.5)
            i, j = coords
            ph, pw = seg.shape
            
            # Convert segmentation to one-hot
            seg_one_hot = np.eye(self.n_classes)[seg]
            
            # Weight by confidence
            weighted_seg = seg_one_hot * confidence
            
            # Accumulate
            segmentation_sum[i:i+ph, j:j+pw] += weighted_seg
            confidence_sum[i:i+ph, j:j+pw] += confidence
        
        # Normalize
        confidence_sum = np.maximum(confidence_sum, 1e-10)
        segmentation_prob = segmentation_sum / confidence_sum[:, :, np.newaxis]
        
        # Convert to labels
        segmentation = np.argmax(segmentation_prob, axis=2)
        
        return segmentation
    
    def _create_weight_map(self, h: int, w: int) -> np.ndarray:
        """Create Gaussian weight map for patch"""
        # Create coordinate grids
        y, x = np.mgrid[0:h, 0:w]
        
        # Center of patch
        center_y, center_x = h / 2, w / 2
        
        # Calculate distances
        distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        
        # Gaussian weight
        sigma = min(h, w) / 4
        weights = np.exp(-distance ** 2 / (2 * sigma ** 2))
        
        return weights
    
    def _refine_segmentation(self, segmentation: np.ndarray,
                           image: np.ndarray) -> np.ndarray:
        """Apply post-processing refinement"""
        if not self.refinement_enabled:
            return segmentation
        
        try:
            # Simple morphological refinement
            from skimage.morphology import opening, closing, disk
            
            refined = segmentation.copy()
            
            # Remove small regions
            from skimage.morphology import remove_small_objects
            min_size = max(1, (segmentation.shape[0] * segmentation.shape[1]) // 1000)
            refined = remove_small_objects(refined, min_size=min_size)
            
            # Fill holes
            from skimage.morphology import remove_small_holes
            refined = remove_small_holes(refined, area_threshold=min_size)
            
            # Apply median filter to smooth boundaries
            from scipy.ndimage import median_filter
            refined = median_filter(refined, size=3)
            
            return refined
            
        except Exception as e:
            logger.warning(f"Refinement failed: {e}")
            return segmentation
    
    def _calculate_uncertainty(self, patch_results: List[Dict[str, Any]]) -> np.ndarray:
        """Calculate segmentation uncertainty"""
        if not patch_results:
            return np.array([])
        
        # For each pixel, calculate variance across patches
        # This is simplified - in practice, use more sophisticated uncertainty measures
        h, w = patch_results[0]['segmentation'].shape
        uncertainties = []
        
        for result in patch_results:
            seg = result['segmentation']
            
            # Simple uncertainty: inverse of confidence
            confidence = result['selection_metadata'].get('confidence', 0.5)
            uncertainty = 1.0 - confidence
            
            uncertainties.append(np.full((h, w), uncertainty))
        
        # Average uncertainty
        avg_uncertainty = np.mean(uncertainties, axis=0)
        
        return avg_uncertainty
    
    def _update_statistics(self, result: Dict[str, Any]) -> None:
        """Update segmentation statistics"""
        stats = {
            'timestamp': time.time(),
            'image_shape': result['image_shape'],
            'n_patches': result['n_patches'],
            'execution_time': result['execution_time'],
            'method': result['method']
        }
        
        self.segmentation_stats.append(stats)
        
        # Limit history size
        if len(self.segmentation_stats) > 100:
            self.segmentation_stats = self.segmentation_stats[-100:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get segmentation statistics"""
        if not self.segmentation_stats:
            return {}
        
        stats = {
            'total_segmentations': len(self.segmentation_stats),
            'avg_execution_time': np.mean([s['execution_time'] for s in self.segmentation_stats]),
            'avg_image_size': np.mean([s['image_shape'][0] * s['image_shape'][1] 
                                      for s in self.segmentation_stats]),
            'methods_used': list(set(s['method'] for s in self.segmentation_stats))
        }
        
        return stats
    
    def reset_statistics(self) -> None:
        """Reset segmentation statistics"""
        self.segmentation_stats = []
        logger.info("Segmentation statistics reset")
