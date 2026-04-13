"""
Low-Level Heuristics (LLH) Registry
Implements Table 4.1 from the thesis
"""

from .base import BaseLLH, LLHRegistry
from .sspso import SSPSO, SSPSOFast, SSPSOAccurate, SSPSOSpatial, SSPSOSpectral
from .clustering import KMeansLLH, FCM_S, GMM
from .refinement import MRFRefinement, CRFRefinement
from .morphology import MorphologyLLH


# Register all LLHs (Table 4.1)
LLHRegistry.register('K-means++', KMeansLLH)
LLHRegistry.register('FCM-S', FCM_S)
LLHRegistry.register('GMM', GMM)

# Swarm Intelligence LLHs
LLHRegistry.register('SS-PSO-fast', SSPSOFast)
LLHRegistry.register('SS-PSO-accurate', SSPSOAccurate)
LLHRegistry.register('SS-PSO-spatial', SSPSOSpatial)
LLHRegistry.register('SS-PSO-spectral', SSPSOSpectral)

# Edge Detection LLHs (from Chapter 7 - to be added)
# LLHRegistry.register('Gradient-fine', GradientFine)
# LLHRegistry.register('Gradient-medium', GradientMedium)
# LLHRegistry.register('Gradient-coarse', GradientCoarse)

# Post-processing LLHs
LLHRegistry.register('MRF', MRFRefinement)
LLHRegistry.register('CRF', CRFRefinement)
LLHRegistry.register('Morphology', MorphologyLLH)

# Note: Watershed and CNN-refine are excluded (negative results, Table 4.2)
# Inactive terminals documented in thesis Section 4.3.2


def get_llh(name: str, **kwargs) -> BaseLLH:
    """Factory function to instantiate LLH by name."""
    return LLHRegistry.create(name, **kwargs)


def list_available_llhs() -> list:
    """Return list of registered LLH names."""
    return list(LLHRegistry._registry.keys())


__all__ = [
    'BaseLLH', 'LLHRegistry', 'get_llh', 'list_available_llhs',
    'SSPSO', 'SSPSOFast', 'SSPSOAccurate', 'SSPSOSpatial', 'SSPSOSpectral',
    'KMeansLLH', 'FCM_S', 'GMM', 'MRFRefinement', 'CRFRefinement', 'MorphologyLLH'
]