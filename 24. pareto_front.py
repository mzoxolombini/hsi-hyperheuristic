"""
Pareto front management module for GP
Execution Order: 25
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
import json
from dataclasses import dataclass, field

from .individual import Individual

logger = logging.getLogger(__name__)


@dataclass
class ParetoFront:
    """
    Manages Pareto-optimal front of individuals
    
    Implements:
    1. Fast non-dominated sorting
    2. Crowding distance calculation
    3. Front maintenance and update
    """
    
    front: List[Individual] = field(default_factory=list)
    objectives: List[str] = field(default_factory=lambda: ['accuracy', 'efficiency', 'complexity'])
    max_size: int = 50
    
    def update(self, individuals: List[Individual]) -> None:
        """
        Update Pareto front with new individuals
        
        Args:
            individuals: List of individuals to consider
        """
        # Filter valid individuals
        valid_individuals = [ind for ind in individuals if ind.fitness.get('valid', False)]
        
        if not valid_individuals:
            return
        
        # Combine current front with new individuals
        all_individuals = self.front + valid_individuals
        
        # Fast non
