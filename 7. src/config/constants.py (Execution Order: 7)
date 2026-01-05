"""
Constants module
Execution Order: 7
"""

from enum import Enum
from typing import List, Tuple, Dict, Any


class LLHType(Enum):
    """Low-Level Heuristic Types"""
    SS_PSO_FAST = "ss_pso_fast"
    SS_PSO_ACCURATE = "ss_pso_accurate"
    SS_PSO_SPATIAL = "ss_pso_spatial"
    SS_PSO_SPECTRAL = "ss_pso_spectral"
    GRADIENT_FINE = "gradient_fine"
    GRADIENT_MEDIUM = "gradient_medium"
    GRADIENT_COARSE = "gradient_coarse"
    KMEANS = "kmeans"
    SPECTRAL_CLUSTERING = "spectral_clustering"
    WATERSHED = "watershed"
    MRF = "mrf"
    CNN_REFINE = "cnn_refine"
    CRF = "crf"
    FCM = "fcm"


class DatasetType(Enum):
    """Dataset types"""
    INDIAN_PINES = "Indian_Pines"
    PAVIA_UNIVERSITY = "Pavia_University"
    SALINAS = "Salinas"
    HOUSTON = "Houston"
    BOTSWANA = "Botswana"


class OperationType(Enum):
    """Operation types in grammar"""
    PREPROCESS = "Preprocess"
    SEGMENT = "Segment"
    POSTPROCESS = "Postprocess"


class FitnessObjective(Enum):
    """Fitness objectives"""
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    COMPLEXITY = "complexity"
    ENERGY = "energy"


# Predefined grammar production rules
GRAMMAR_RULES = {
    'S': ['Preprocess Segment Postprocess'],
    'Preprocess': ['PCA', 'ICA', 'Denoise', 'Normalize', 'None'],
    'Segment': ['SS_PSO', 'KMeans', 'SpectralClustering', 'Watershed', 'Gradient', 'FCM'],
    'Postprocess': ['MRF', 'CNN_Refine', 'Morphology', 'CRF', 'None']
}

# Terminal parameters
TERMINAL_PARAMS = {
    'PCA': {'n_components': [10, 20, 30, 50, 100]},
    'ICA': {'n_components': [10, 20, 30]},
    'Denoise': {
        'method': ['wavelet', 'bm3d', 'tv'],
        'parameters': [('db4', 3), (1.0, 50), (0.1, 10)]
    },
    'SS_PSO': {
        'variant': ['fast', 'accurate', 'spatial', 'spectral'],
        'n_clusters': [8, 16, 20, 32]
    },
    'KMeans': {
        'n_clusters': [8, 16, 20, 32],
        'init': ['k-means++', 'random']
    },
    'SpectralClustering': {
        'n_clusters': [8, 16, 20],
        'affinity': ['nearest_neighbors', 'rbf']
    },
    'Watershed': {
        'marker_source': ['gradient', 'distance', 'hmin'],
        'compactness': [0.1, 0.3, 0.5, 0.7]
    },
    'Gradient': {
        'scale': ['fine', 'medium', 'coarse'],
        'method': ['sobel', 'scharr', 'prewitt']
    },
    'FCM': {
        'n_clusters': [8, 16, 20],
        'm': [1.5, 2.0, 2.5]
    },
    'MRF': {
        'spatial_weight': [0.5, 1.0, 2.0, 5.0],
        'iterations': [5, 10, 20]
    },
    'CNN_Refine': {
        'layers': [2, 3, 4],
        'channels': [16, 32, 64]
    },
    'Morphology': {
        'operation': ['opening', 'closing', 'dilation', 'erosion'],
        'size': [3, 5, 7]
    },
    'CRF': {
        'sigma': [1.0, 2.0, 3.0],
        'iterations': [5, 10]
    }
}

# Meta-feature names
META_FEATURE_NAMES = [
    'fractal_dimension',
    'spectral_gradient_variance',
    'snr',
    'spatial_homogeneity',
    'spectral_entropy',
    'mean_reflectance',
    'band_correlation',
    'texture_energy',
    'local_contrast',
    'edge_density',
    'spectral_variance',
    'spatial_variance'
]

# Dataset statistics (for normalization)
DATASET_STATS = {
    'Indian_Pines': {
        'mean_spectrum': None,  # Will be calculated
        'std_spectrum': None,
        'min_val': 0.0,
        'max_val': 1.0
    },
    'Pavia_University': {
        'mean_spectrum': None,
        'std_spectrum': None,
        'min_val': 0.0,
        'max_val': 1.0
    },
    'Salinas': {
        'mean_spectrum': None,
        'std_spectrum': None,
        'min_val': 0.0,
        'max_val': 1.0
    }
}

# Performance benchmarks (from thesis)
THESIS_BENCHMARKS = {
    'accuracy_mIoU': 0.874,
    'parameters_millions': 0.047,
    'training_time_minutes': 14.3,
    'energy_efficiency': 7.1,
    'generalization_rate': 0.903
}

# Color maps for visualization
CLASS_COLORMAPS = {
    'Indian_Pines': [
        '#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00',
        '#FF00FF', '#00FFFF', '#800000', '#008000', '#000080',
        '#808000', '#800080', '#008080', '#808080', '#C0C0C0',
        '#FFA500'
    ],
    'Pavia_University': [
        '#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00',
        '#FF00FF', '#00FFFF', '#800000', '#008000', '#000080'
    ]
}
