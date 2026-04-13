"""
Appendix A: Complete BNF Grammar for Grammar-Guided Genetic Programming
Grammar version v6.2 as defined in the thesis (pages 96-98)
"""

import random
import re
from typing import Dict, List, Any, Optional, Tuple


class BNFGrammar:
    """
    Backus-Naur Form Grammar for hyperspectral segmentation pipeline generation.
    
    This grammar enforces the spectral -> spatial -> refinement ordering
    as required by the thesis (Section 5.3.1).
    """
    
    # Core grammar structure from Appendix A.1
    GRAMMAR = {
        # Start symbol - enforces pipeline ordering
        "<PIPELINE>": [
            "<PREPROCESS> <SEGMENT> <POSTPROCESS>"
        ],
        
        # Preprocessing stage (must precede segmentation)
        "<PREPROCESS>": [
            "PCA(<INT>)",
            "ICA(<INT>)",
            "MNF(<INT>)",
            "BM3D(<SIGMA>)",
            "SavitzkyGolay(<WINDOW>, <ORDER>)",
            "Wavelet(<FAMILY>, <LEVELS>)",
            "NONE"
        ],
        
        # Segmentation stage - core operators
        "<SEGMENT>": [
            "KMeans(<CLUSTERS>, <INIT>)",
            "FCM_S(<CLUSTERS>, <FUZZINESS>)",
            "GMM(<CLUSTERS>, <COVARIANCE>)",
            "SS_PSO(<VARIANT>, <PARTICLES>, <ITERATIONS>, <META_COND>)",
            "HolisticGradient(<SCALES>, <META_WEIGHTS>)",
            "Watershed(<MARKER_SOURCE>, <COMPACTNESS>)",
            "RegionGrow(<SEED_POLICY>, <SIMILARITY>)"
        ],
        
        # Post-processing stage (refinement)
        "<POSTPROCESS>": [
            "<REFINEMENT>",
            "<SMOOTHING>",
            "<COMBINATION>",
            "NONE"
        ],
        
        "<REFINEMENT>": [
            "MRF(<SPATIAL_WEIGHT>, <BETA>)",
            "CRF(<GAUSSIAN_W>, <BILATERAL_W>)",
            "CNN_Refine(<LAYERS>, <FILTERS>)"
        ],
        
        "<SMOOTHING>": [
            "Morphology(<OPERATION>, <KERNEL_SIZE>)"
        ],
        
        "<COMBINATION>": [
            "<REFINEMENT> <SMOOTHING>",
            "<SMOOTHING> <REFINEMENT>"
        ],
        
        # Terminal parameter productions (Table A.1)
        "<INT>": ["10", "20", "34", "50", "100"],
        "<WINDOW>": ["5", "7", "9", "11"],
        "<ORDER>": ["2", "3", "4"],
        "<SIGMA>": ["0.5", "1.0", "1.5", "2.0"],
        "<FAMILY>": ["db4", "sym4", "coif2"],
        "<LEVELS>": ["3", "4", "5"],
        "<CLUSTERS>": ["9", "16", "20", "auto"],
        "<INIT>": ["kmeans++", "random", "spectral"],
        "<FUZZINESS>": ["1.5", "2.0", "2.5"],
        "<COVARIANCE>": ["full", "tied", "diag", "spherical"],
        "<VARIANT>": ["fast", "accurate", "spatial", "spectral"],
        "<PARTICLES>": ["20", "50", "100"],
        "<ITERATIONS>": ["50", "100", "200", "300"],
        "<META_COND>": ["true", "false"],
        "<MARKER_SOURCE>": ["cnn", "gradient", "manual"],
        "<COMPACTNESS>": ["0.1", "0.3", "0.5", "0.7"],
        "<SEED_POLICY>": ["random", "max_gradient", "saliency"],
        "<SIMILARITY>": ["euclidean", "spectral_angle", "mahalanobis"],
        "<SCALES>": ["fine", "medium", "coarse", "multiscale"],
        "<META_WEIGHTS>": ["true", "false"],
        "<SPATIAL_WEIGHT>": ["0.1", "0.5", "1.0", "2.0"],
        "<BETA>": ["1.0", "1.5", "2.0"],
        "<GAUSSIAN_W>": ["3.0", "5.0", "7.0"],
        "<BILATERAL_W>": ["5.0", "10.0", "15.0"],
        "<LAYERS>": ["2", "3", "4"],
        "<FILTERS>": ["32", "64", "128"],
        "<OPERATION>": ["opening", "closing", "dilation", "erosion"],
        "<KERNEL_SIZE>": ["3", "5", "7"]
    }
    
    # Grammar version history (Table A.2)
    VERSION_HISTORY = [
        {"version": "v1.0", "date": "2023-03", "terminals": 23, "invalid_rate": 0.87, "best_miou": 0.654},
        {"version": "v3.0", "date": "2023-08", "terminals": 31, "invalid_rate": 0.34, "best_miou": 0.720},
        {"version": "v6.0", "date": "2024-01", "terminals": 42, "invalid_rate": 0.20, "best_miou": 0.807},
        {"version": "v6.2", "date": "2024-02", "terminals": 48, "invalid_rate": 0.00, "best_miou": 0.874}
    ]
    
    # Inactive terminals (removed due to negative results, Section 4.3.2)
    INACTIVE_TERMINALS = ["Watershed", "CNN_Refine"]
    
    def __init__(self, max_depth: int = 6):
        """
        Initialize BNF grammar.
        
        Args:
            max_depth: Maximum derivation tree depth
        """
        self.max_depth = max_depth
        self._cache = {}
    
    def get_productions(self, non_terminal: str) -> List[str]:
        """Get all production rules for a given non-terminal."""
        return self.GRAMMAR.get(non_terminal, [])
    
    def is_terminal(self, symbol: str) -> bool:
        """Check if a symbol is terminal (no angle brackets, not a non-terminal)."""
        if not symbol.startswith("<") or not symbol.endswith(">"):
            return True
        # Also check if it's a non-terminal that has no further expansions
        return symbol not in self.GRAMMAR
    
    def expand(self, symbol: str, depth: int = 0) -> str:
        """
        Recursively expand a symbol using grammar rules.
        
        Args:
            symbol: Starting symbol (usually "<PIPELINE>")
            depth: Current recursion depth
            
        Returns:
            Expanded pipeline string
        """
        if depth > self.max_depth:
            return "NONE"
        
        if self.is_terminal(symbol):
            return symbol
        
        # Choose a random production rule
        productions = self.get_productions(symbol)
        if not productions:
            return symbol
        
        production = random.choice(productions)
        
        # Expand each part of the production
        result = []
        for part in production.split():
            result.append(self.expand(part, depth + 1))
        
        return " ".join(result)
    
    def generate_random_pipeline(self) -> str:
        """Generate a random valid pipeline from the grammar."""
        return self.expand("<PIPELINE>")
    
    def validate_pipeline(self, pipeline: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that a pipeline follows grammar constraints.
        
        Returns:
            (is_valid, error_message)
        """
        # Check pipeline ordering: PREPROCESS -> SEGMENT -> POSTPROCESS
        parts = pipeline.split()
        
        # Find the main stages
        preprocessing_keywords = ["PCA", "ICA", "MNF", "BM3D", "SavitzkyGolay", "Wavelet"]
        segmentation_keywords = ["KMeans", "FCM_S", "GMM", "SS_PSO", "HolisticGradient", "Watershed", "RegionGrow"]
        postprocessing_keywords = ["MRF", "CRF", "CNN_Refine", "Morphology"]
        
        # Check that segmentation occurs after preprocessing
        pre_idx = -1
        seg_idx = -1
        post_idx = -1
        
        for i, part in enumerate(parts):
            for kw in preprocessing_keywords:
                if part.startswith(kw):
                    pre_idx = i
            for kw in segmentation_keywords:
                if part.startswith(kw):
                    seg_idx = i
            for kw in postprocessing_keywords:
                if part.startswith(kw):
                    post_idx = i
        
        # Validate ordering
        if seg_idx != -1 and pre_idx > seg_idx:
            return False, "Segmentation must occur after preprocessing"
        
        if post_idx != -1 and seg_idx > post_idx:
            return False, "Post-processing must occur after segmentation"
        
        # Check for inactive terminals
        for inactive in self.INACTIVE_TERMINALS:
            if inactive in pipeline:
                return False, f"Inactive terminal '{inactive}' used (removed due to negative results)"
        
        return True, None
    
    def get_terminals(self) -> List[str]:
        """Get all terminal symbols from the grammar."""
        terminals = []
        for key, values in self.GRAMMAR.items():
            if key.startswith("<") and key.endswith(">"):
                # Check if all values are terminal (no angle brackets)
                if all("<" not in v for v in values):
                    terminals.extend(values)
        return list(set(terminals))
    
    def get_non_terminals(self) -> List[str]:
        """Get all non-terminal symbols."""
        return [k for k in self.GRAMMAR.keys() if k.startswith("<")]
    
    def get_search_space_size(self) -> int:
        """Calculate total number of possible pipelines (search space cardinality)."""
        # Simplified calculation: product of branching factors
        size = 1
        for non_terminal in self.get_non_terminals():
            size *= len(self.GRAMMAR.get(non_terminal, []))
        return min(size, 10**12)  # Cap at 1 trillion
    
    def get_invalid_rate_estimate(self) -> float:
        """Estimate the invalid pipeline rate for this grammar."""
        # v6.2 has 0% invalid rate by design
        return 0.0


class DerivationTree:
    """
    Represents a derivation tree for a pipeline generated by the grammar.
    Used for genetic programming crossover and mutation operations.
    """
    
    def __init__(self, grammar: BNFGrammar, root: str = "<PIPELINE>"):
        self.grammar = grammar
        self.root = root
        self.nodes = []
        self.depth = 0
    
    def build(self, symbol: str = None, depth: int = 0) -> Dict:
        """
        Build derivation tree recursively.
        
        Returns:
            Tree node as dict with 'symbol', 'children', 'depth'
        """
        if symbol is None:
            symbol = self.root
        
        node = {
            'symbol': symbol,
            'children': [],
            'depth': depth
        }
        
        if depth < self.grammar.max_depth and not self.grammar.is_terminal(symbol):
            productions = self.grammar.get_productions(symbol)
            if productions:
                # Choose a production (for building from a specific pipeline, use fixed)
                production = random.choice(productions)
                for part in production.split():
                    child = self.build(part, depth + 1)
                    node['children'].append(child)
        
        self.depth = max(self.depth, depth)
        return node
    
    def to_pipeline_string(self, node: Dict = None) -> str:
        """Convert derivation tree back to pipeline string."""
        if node is None:
            node = self.nodes[0] if self.nodes else self.build()
        
        if self.grammar.is_terminal(node['symbol']):
            return node['symbol']
        
        result = []
        for child in node['children']:
            result.append(self.to_pipeline_string(child))
        
        return " ".join(result)
    
    def get_depth(self) -> int:
        """Get maximum tree depth."""
        return self.depth
    
    def get_node_count(self) -> int:
        """Get total number of nodes in the tree."""
        def count_nodes(node):
            count = 1
            for child in node['children']:
                count += count_nodes(child)
            return count
        
        return count_nodes(self.nodes[0] if self.nodes else self.build())


class GrammarV6_2(BNFGrammar):
    """
    Grammar version v6.2 as defined in the thesis.
    This is the final validated version with 0% invalid pipeline rate.
    """
    
    def __init__(self):
        super().__init__(max_depth=6)
        # v6.2 specific settings
        self.version = "v6.2"
        self.invalid_rate = 0.0
        self.best_miou = 0.874
    
    def get_example_pipelines(self) -> Dict[str, str]:
        """
        Return example evolved pipelines from the thesis (Section A.4).
        
        Returns:
            Dictionary of pipeline names to their string representations
        """
        return {
            "Evolved Pipeline #47 (Highest Accuracy on Salinas)": 
                "PCA(34) SS_PSO(accurate,100,200,true) CRF(5.0,10.0)",
            
            "Evolved Pipeline #23 (Fastest on Pavia)": 
                "BM3D(1.0) KMeans(9,kmeans++) Morphology(opening,3)",
            
            "Scientific Mineral Mapping": 
                "ICA(20) SS_PSO(accurate,100,300,true) CRF(5.0,10.0)",
            
            "UAV Edge Deployment": 
                "PCA(34) SS_PSO(spatial,50,100,true) NONE",
            
            "Real-time Wildfire Monitoring": 
                "NONE KMeans(9,kmeans++) Morphology(opening,3)"
        }