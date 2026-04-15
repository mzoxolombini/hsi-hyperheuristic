"""
Fitness evaluation module for GP
Execution Order: 23
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import psutil
import torch

from llhs.base import LLHRegistry
from data.meta_features import MetaFeatureExtractor
from utils.metrics import SegmentationMetrics

logger = logging.getLogger(__name__)


class MultiObjectiveEvaluator:
    """
    Multi-objective fitness evaluation
    
    Evaluates individuals across three objectives:
    1. Accuracy (mIoU)
    2. Efficiency (computational cost)
    3. Complexity (pipeline complexity)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluator
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.metrics_calculator = SegmentationMetrics()
        self.meta_feature_extractor = MetaFeatureExtractor()
        
        # Cache for pipeline execution results
        self.execution_cache: Dict[str, Dict[str, Any]] = {}
        
        # Performance profiling
        self.enable_profiling = config.get('enable_profiling', False)
        
        logger.info("Multi-objective evaluator initialized")
    
    def evaluate(self, individual, data: np.ndarray, ground_truth: np.ndarray,
                meta_features: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Evaluate individual fitness
        
        Args:
            individual: Individual to evaluate
            data: Input data [H, W, B]
            ground_truth: Ground truth labels [H, W]
            meta_features: Meta-features for conditioning (optional)
            
        Returns:
            Dictionary of fitness values
        """
        # Check cache
        individual_hash = individual.get_hash()
        if individual_hash in self.execution_cache:
            cached_result = self.execution_cache[individual_hash]
            
            # Check if evaluation parameters are similar
            if self._check_cache_validity(cached_result, data.shape, ground_truth.shape):
                logger.debug(f"Using cached evaluation for {individual_hash}")
                return cached_result['fitness']
        
        try:
            # Execute pipeline
            execution_result = self._execute_pipeline(
                individual.pipeline,
                data,
                meta_features
            )
            
            # Calculate objectives
            fitness = self._calculate_fitness(
                execution_result,
                ground_truth,
                individual
            )
            
            # Cache result
            self.execution_cache[individual_hash] = {
                'fitness': fitness,
                'execution_result': execution_result,
                'data_shape': data.shape,
                'gt_shape': ground_truth.shape,
                'timestamp': time.time()
            }
            
            return fitness
            
        except Exception as e:
            logger.warning(f"Evaluation failed for individual {individual.id}: {e}")
            return self._get_default_fitness()
    
    def _execute_pipeline(self, pipeline_node, data: np.ndarray,
                         meta_features: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Execute pipeline node recursively
        
        Args:
            pipeline_node: PipelineNode to execute
            data: Input data
            meta_features: Meta-features for conditioning
            
        Returns:
            Execution results
        """
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss
        
        try:
            # Execute node
            result = self._execute_node(pipeline_node, data, meta_features)
            
            execution_time = time.time() - start_time
            memory_after = psutil.Process().memory_info().rss
            memory_used = (memory_after - memory_before) / (1024 * 1024)  # MB
            
            return {
                'result': result,
                'execution_time': execution_time,
                'memory_used_mb': memory_used,
                'success': True,
                'node_type': pipeline_node.operation
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Pipeline execution failed at node {pipeline_node.operation}: {e}")
            
            return {
                'result': None,
                'execution_time': execution_time,
                'memory_used_mb': 0,
                'success': False,
                'error': str(e),
                'node_type': pipeline_node.operation
            }
    
    def _execute_node(self, node, data: np.ndarray,
                     meta_features: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Execute single node"""
        operation = node.operation
        params = node.parameters
        
        # Terminal operations
        if operation == 'PCA':
            return self._execute_pca(data, params)
        elif operation == 'ICA':
            return self._execute_ica(data, params)
        elif operation == 'Denoise':
            return self._execute_denoise(data, params)
        elif operation == 'Normalize':
            return self._execute_normalize(data, params)
        elif operation == 'SS_PSO':
            return self._execute_sspso(data, params, meta_features)
        elif operation == 'KMeans':
            return self._execute_kmeans(data, params)
        elif operation == 'SpectralClustering':
            return self._execute_spectral_clustering(data, params)
        elif operation == 'Watershed':
            return self._execute_watershed(data, params)
        elif operation == 'Gradient':
            return self._execute_gradient(data, params, meta_features)
        elif operation == 'FCM':
            return self._execute_fcm(data, params)
        elif operation == 'MRF':
            return self._execute_mrf(data, params)
        elif operation == 'CNN_Refine':
            return self._execute_cnn_refine(data, params)
        elif operation == 'Morphology':
            return self._execute_morphology(data, params)
        elif operation == 'CRF':
            return self._execute_crf(data, params)
        elif operation == 'None':
            return data  # No operation
        else:
            # Non-terminal: execute children sequentially
            result = data
            for child in node.children:
                result = self._execute_node(child, result, meta_features)
            return result
    
    def _execute_pca(self, data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Execute PCA dimensionality reduction"""
        from sklearn.decomposition import PCA
        
        n_components = params.get('n_components', 34)
        n_components = min(n_components, data.shape[2])
        
        h, w, b = data.shape
        reshaped = data.reshape(-1, b)
        
        pca = PCA(n_components=n_components)
        result = pca.fit_transform(reshaped).reshape(h, w, n_components)
        
        return result
    
    def _execute_ica(self, data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Execute ICA dimensionality reduction"""
        from sklearn.decomposition import FastICA
        
        n_components = params.get('n_components', 20)
        n_components = min(n_components, data.shape[2])
        
        h, w, b = data.shape
        reshaped = data.reshape(-1, b)
        
        ica = FastICA(n_components=n_components, max_iter=100, random_state=42)
        result = ica.fit_transform(reshaped).reshape(h, w, n_components)
        
        return result
    
    def _execute_denoise(self, data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Execute denoising"""
        method = params.get('method', 'wavelet')
        
        if method == 'wavelet':
            return self._wavelet_denoise(data, params)
        elif method == 'bm3d':
            return self._bm3d_denoise(data, params)
        elif method == 'tv':
            return self._tv_denoise(data, params)
        else:
            return data
    
    def _wavelet_denoise(self, data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Wavelet denoising"""
        try:
            import pywt
            
            wavelet, level = params.get('parameters', ('db4', 3))
            h, w, b = data.shape
            denoised = np.zeros_like(data)
            
            for band in range(b):
                coeffs = pywt.wavedec2(data[:, :, band], wavelet, level=level)
                
                # Threshold coefficients
                coeffs_thresh = []
                coeffs_thresh.append(coeffs[0])  # Approximation
                
                for detail in coeffs[1:]:
                    threshold = np.std(detail) * np.sqrt(2 * np.log(h * w))
                    detail_thresh = pywt.threshold(detail, threshold, mode='soft')
                    coeffs_thresh.append(detail_thresh)
                
                denoised_band = pywt.waverec2(coeffs_thresh, wavelet)
                denoised[:, :, band] = denoised_band[:h, :w]  # Handle padding
            
            return denoised
            
        except ImportError:
            logger.warning("PyWavelets not installed. Skipping denoising.")
            return data
    
    def _bm3d_denoise(self, data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """BM3D denoising approximation"""
        from scipy.ndimage import gaussian_filter
        
        sigma, _ = params.get('parameters', (1.0, 50))
        
        denoised = np.zeros_like(data)
        for band in range(data.shape[2]):
            denoised[:, :, band] = gaussian_filter(data[:, :, band], sigma=sigma)
        
        return denoised
    
    def _tv_denoise(self, data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Total Variation denoising"""
        from skimage.restoration import denoise_tv_chambolle
        
        weight, _ = params.get('parameters', (0.1, 10))
        
        denoised = np.zeros_like(data)
        for band in range(data.shape[2]):
            denoised[:, :, band] = denoise_tv_chambolle(
                data[:, :, band],
                weight=weight,
                eps=1e-4,
                max_num_iter=100
            )
        
        return denoised
    
    def _execute_normalize(self, data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Normalize data"""
        # Min-max normalization per band
        normalized = np.zeros_like(data)
        
        for band in range(data.shape[2]):
            band_data = data[:, :, band]
            min_val = band_data.min()
            max_val = band_data.max()
            
            if max_val > min_val:
                normalized[:, :, band] = (band_data - min_val) / (max_val - min_val)
            else:
                normalized[:, :, band] = band_data
        
        return normalized
    
    def _execute_sspso(self, data: np.ndarray, params: Dict[str, Any],
                      meta_features: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Execute SS-PSO clustering"""
        from llhs.sspso import SpectralSpatialPSO
        
        variant = params.get('variant', 'accurate')
        n_clusters = params.get('n_clusters', 16)
        
        # Create SS-PSO instance
        config = {
            'variant': variant,
            'n_clusters': n_clusters,
            **self.config.get('pso', {})
        }
        
        sspso = SpectralSpatialPSO('sspso_eval', config)
        
        # Apply with meta-feature conditioning
        kwargs = {'n_clusters': n_clusters}
        if meta_features:
            kwargs['meta_features'] = meta_features
        
        result = sspso.apply(data, **kwargs)
        
        return result
    
    def _execute_kmeans(self, data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Execute K-means clustering"""
        from sklearn.cluster import KMeans
        
        n_clusters = params.get('n_clusters', 16)
        init = params.get('init', 'k-means++')
        
        h, w, b = data.shape
        reshaped = data.reshape(-1, b)
        
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=10,
            random_state=42
        )
        
        labels = kmeans.fit_predict(reshaped)
        return labels.reshape(h, w)
    
    def _execute_spectral_clustering(self, data: np.ndarray, 
                                    params: Dict[str, Any]) -> np.ndarray:
        """Execute spectral clustering"""
        from sklearn.cluster import SpectralClustering
        
        n_clusters = params.get('n_clusters', 16)
        affinity = params.get('affinity', 'nearest_neighbors')
        
        h, w, b = data.shape
        
        # Spectral clustering is memory intensive
        if h * w > 5000:
            logger.warning("Image too large for spectral clustering. Using K-means.")
            return self._execute_kmeans(data, {'n_clusters': n_clusters})
        
        reshaped = data.reshape(-1, b)
        
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            random_state=42,
            assign_labels='kmeans'
        )
        
        labels = spectral.fit_predict(reshaped)
        return labels.reshape(h, w)
    
    def _execute_watershed(self, data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Execute watershed segmentation"""
        from llhs.watershed import WatershedSegmentation
        
        config = {
            'marker_source': params.get('marker_source', 'gradient'),
            'compactness': params.get('compactness', 0.3)
        }
        
        watershed = WatershedSegmentation('watershed_eval', config)
        return watershed.apply(data)
    
    def _execute_gradient(self, data: np.ndarray, params: Dict[str, Any],
                         meta_features: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Execute gradient-based segmentation"""
        from llhs.gradient_ops import HolisticGradientOperator
        
        scale = params.get('scale', 'medium')
        config = {'scale': scale}
        
        gradient_op = HolisticGradientOperator(f'gradient_{scale}', config)
        
        kwargs = {}
        if meta_features:
            kwargs['meta_features'] = meta_features
        
        return gradient_op.apply(data, **kwargs)
    
    def _execute_fcm(self, data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Execute Fuzzy C-means clustering"""
        try:
            from skfuzzy import cmeans
            
            n_clusters = params.get('n_clusters', 16)
            m = params.get('m', 2.0)
            
            h, w, b = data.shape
            reshaped = data.reshape(-1, b).T  # Transpose for skfuzzy
            
            # Fuzzy C-means
            cntr, u, _, _, _, _, _ = cmeans(
                reshaped,
                c=n_clusters,
                m=m,
                error=0.005,
                maxiter=100,
                init=None
            )
            
            # Hard clustering from membership matrix
            labels = np.argmax(u, axis=0)
            return labels.reshape(h, w)
            
        except ImportError:
            logger.warning("skfuzzy not installed. Using K-means instead.")
            return self._execute_kmeans(data, {'n_clusters': params.get('n_clusters', 16)})
    
    def _execute_mrf(self, data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Execute MRF refinement"""
        # Simple MRF approximation using TV denoising
        from skimage.restoration import denoise_tv_bregman
        
        weight = params.get('spatial_weight', 1.0)
        
        if data.ndim == 3:
            data_2d = data[:, :, 0]
        else:
            data_2d = data
        
        refined = denoise_tv_bregman(data_2d.astype(np.float64), weight=weight)
        return refined
    
    def _execute_cnn_refine(self, data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Execute CNN refinement"""
        # Simplified CNN refinement (placeholder)
        # In production, implement proper U-Net
        return data
    
    def _execute_morphology(self, data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Execute morphological operations"""
        from skimage.morphology import opening, closing, dilation, erosion, disk
        
        operation = params.get('operation', 'closing')
        size = params.get('size', 3)
        
        selem = disk(size)
        
        if operation == 'opening':
            result = opening(data, selem)
        elif operation == 'closing':
            result = closing(data, selem)
        elif operation == 'dilation':
            result = dilation(data, selem)
        elif operation == 'erosion':
            result = erosion(data, selem)
        else:
            result = data
        
        return result
    
    def _execute_crf(self, data: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Execute CRF refinement"""
        # Simplified CRF (placeholder)
        # In production, implement dense CRF
        return data
    
    def _calculate_fitness(self, execution_result: Dict[str, Any],
                          ground_truth: np.ndarray,
                          individual) -> Dict[str, float]:
        """
        Calculate multi-objective fitness
        
        Args:
            execution_result: Pipeline execution results
            ground_truth: Ground truth labels
            individual: Individual being evaluated
            
        Returns:
            Fitness dictionary
        """
        if not execution_result['success'] or execution_result['result'] is None:
            return self._get_default_fitness()
        
        result = execution_result['result']
        
        # Objective 1: Accuracy (mIoU)
        if result.shape != ground_truth.shape:
            accuracy = 0.0
        else:
            accuracy = self.metrics_calculator.calculate_miou(result, ground_truth)
        
        # Objective 2: Efficiency (inverse of execution time)
        exec_time = execution_result['execution_time']
        if exec_time > 0:
            efficiency = 1.0 / (exec_time + 0.001)  # Add small constant
        else:
            efficiency = 0.0
        
        # Objective 3: Complexity (inverse of pipeline complexity)
        pipeline_complexity = individual.get_complexity()
        if pipeline_complexity > 0:
            complexity_fitness = 1.0 / (pipeline_complexity + 1.0)
        else:
            complexity_fitness = 1.0
        
        # Additional objective: Memory efficiency
        memory_used = execution_result.get('memory_used_mb', 0)
        if memory_used > 0:
            memory_fitness = 100.0 / (memory_used + 1.0)  # MB
        else:
            memory_fitness = 0.0
        
        # Combine objectives
        fitness = {
            'accuracy': accuracy,
            'efficiency': efficiency,
            'complexity': complexity_fitness,
            'memory': memory_fitness,
            'valid': True,
            'execution_time': exec_time,
            'memory_used_mb': memory_used,
            'pipeline_complexity': pipeline_complexity
        }
        
        return fitness
    
    def _get_default_fitness(self) -> Dict[str, float]:
        """Get default fitness for invalid individuals"""
        return {
            'accuracy': 0.0,
            'efficiency': 0.0,
            'complexity': 0.0,
            'memory': 0.0,
            'valid': False,
            'execution_time': 0.0,
            'memory_used_mb': 0.0,
            'pipeline_complexity': 0.0
        }
    
    def _check_cache_validity(self, cached_result: Dict[str, Any],
                             data_shape: Tuple[int, ...],
                             gt_shape: Tuple[int, ...]) -> bool:
        """Check if cached result is still valid"""
        # Check data shapes
        if cached_result['data_shape'] != data_shape:
            return False
        
        if cached_result['gt_shape'] != gt_shape:
            return False
        
        # Check cache age (1 hour expiry)
        cache_age = time.time() - cached_result.get('timestamp', 0)
        if cache_age > 3600:  # 1 hour
            return False
        
        return True
    
    def clear_cache(self) -> None:
        """Clear evaluation cache"""
        self.execution_cache.clear()
        logger.info("Evaluation cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'cache_size': len(self.execution_cache),
            'cache_hits': sum(1 for v in self.execution_cache.values() if v.get('hits', 0) > 0),
            'total_evaluations': len(self.execution_cache)
        }
