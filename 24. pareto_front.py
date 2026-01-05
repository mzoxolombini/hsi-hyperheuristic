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
        
        # Fast non-dominated sorting
        fronts = self._fast_non_dominated_sort(all_individuals)
        
        # Update Pareto front (first front)
        if fronts:
            self.front = fronts[0]
            
            # Limit front size using crowding distance
            if len(self.front) > self.max_size:
                self._reduce_front_size()
        
        logger.debug(f"Pareto front updated. Size: {len(self.front)}")
    
    def _fast_non_dominated_sort(self, individuals: List[Individual]) -> List[List[Individual]]:
        """
        Fast non-dominated sorting (NSGA-II algorithm)
        
        Args:
            individuals: List of individuals
            
        Returns:
            List of fronts (each front is list of non-dominated individuals)
        """
        if not individuals:
            return []
        
        # Initialize data structures
        n = len(individuals)
        S = [[] for _ in range(n)]  # Individuals dominated by i
        n_dominated = [0] * n       # Number of individuals dominating i
        fronts = [[]]               # Pareto fronts
        rank = [0] * n              # Front rank for each individual
        
        # Compute domination relationships
        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(individuals[i], individuals[j]):
                    S[i].append(j)
                    n_dominated[j] += 1
                elif self._dominates(individuals[j], individuals[i]):
                    S[j].append(i)
                    n_dominated[i] += 1
        
        # Find first front (non-dominated individuals)
        current_front = []
        for i in range(n):
            if n_dominated[i] == 0:
                rank[i] = 0
                current_front.append(i)
        
        fronts[0] = [individuals[i] for i in current_front]
        
        # Build subsequent fronts
        current_front_idx = 0
        while fronts[current_front_idx]:
            next_front_indices = []
            
            for i in fronts[current_front_idx]:
                idx = individuals.index(i)
                for j in S[idx]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        rank[j] = current_front_idx + 1
                        next_front_indices.append(j)
            
            current_front_idx += 1
            if next_front_indices:
                fronts.append([individuals[i] for i in next_front_indices])
            else:
                break
        
        return fronts
    
    def _dominates(self, ind1: Individual, ind2: Individual) -> bool:
        """
        Check if ind1 dominates ind2
        
        Args:
            ind1: First individual
            ind2: Second individual
            
        Returns:
            True if ind1 dominates ind2
        """
        # Get fitness values for common objectives
        better_in_any = False
        
        for obj in self.objectives:
            val1 = ind1.fitness.get(obj, 0.0)
            val2 = ind2.fitness.get(obj, 0.0)
            
            if val1 < val2:
                return False  # ind1 is worse in at least one objective
            elif val1 > val2:
                better_in_any = True
        
        return better_in_any
    
    def _reduce_front_size(self) -> None:
        """Reduce front size using crowding distance"""
        if len(self.front) <= self.max_size:
            return
        
        # Calculate crowding distance for each individual
        self._calculate_crowding_distance()
        
        # Sort by crowding distance (descending)
        self.front.sort(key=lambda ind: ind.fitness.get('crowding_distance', 0), reverse=True)
        
        # Keep only top individuals
        self.front = self.front[:self.max_size]
        
        logger.debug(f"Reduced Pareto front to {len(self.front)} individuals")
    
    def _calculate_crowding_distance(self) -> None:
        """Calculate crowding distance for individuals in front"""
        n = len(self.front)
        
        if n <= 2:
            # All individuals get maximum distance
            for ind in self.front:
                ind.fitness['crowding_distance'] = float('inf')
            return
        
        # Initialize distances
        for ind in self.front:
            ind.fitness['crowding_distance'] = 0.0
        
        # Calculate for each objective
        for obj in self.objectives:
            # Sort by objective value
            sorted_front = sorted(self.front, key=lambda ind: ind.fitness.get(obj, 0.0))
            
            # Boundary points get infinite distance
            sorted_front[0].fitness['crowding_distance'] = float('inf')
            sorted_front[-1].fitness['crowding_distance'] = float('inf')
            
            # Normalize objective values
            obj_min = sorted_front[0].fitness.get(obj, 0.0)
            obj_max = sorted_front[-1].fitness.get(obj, 0.0)
            obj_range = obj_max - obj_min
            
            if obj_range > 0:
                # Calculate crowding distance for intermediate points
                for i in range(1, n - 1):
                    prev_val = sorted_front[i - 1].fitness.get(obj, 0.0)
                    next_val = sorted_front[i + 1].fitness.get(obj, 0.0)
                    
                    distance = (next_val - prev_val) / obj_range
                    sorted_front[i].fitness['crowding_distance'] += distance
    
    def get_diverse_individuals(self, n: int) -> List[Individual]:
        """
        Get diverse individuals from Pareto front
        
        Args:
            n: Number of individuals to return
            
        Returns:
            List of diverse individuals
        """
        if not self.front:
            return []
        
        if len(self.front) <= n:
            return self.front.copy()
        
        # Calculate crowding distance if not already calculated
        if 'crowding_distance' not in self.front[0].fitness:
            self._calculate_crowding_distance()
        
        # Sort by crowding distance (descending)
        sorted_front = sorted(self.front, 
                             key=lambda ind: ind.fitness.get('crowding_distance', 0), 
                             reverse=True)
        
        # Also consider objective space diversity
        diverse_individuals = []
        selected_indices = set()
        
        # Select individuals with highest crowding distance
        for i in range(min(n, len(sorted_front))):
            diverse_individuals.append(sorted_front[i])
            selected_indices.add(i)
        
        # If we need more diverse individuals, select from different regions
        if len(diverse_individuals) < n:
            # Cluster individuals in objective space
            from sklearn.cluster import KMeans
            
            # Create feature matrix from objective values
            features = []
            for ind in self.front:
                feature = [ind.fitness.get(obj, 0.0) for obj in self.objectives]
                features.append(feature)
            
            features = np.array(features)
            
            # Perform clustering
            n_clusters = min(n, len(self.front))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features)
            
            # Select one individual from each cluster
            cluster_representatives = {}
            for idx, (ind, label) in enumerate(zip(self.front, cluster_labels)):
                if label not in cluster_representatives:
                    cluster_representatives[label] = idx
            
            # Add representatives not already selected
            for cluster_idx, ind_idx in cluster_representatives.items():
                if ind_idx not in selected_indices and len(diverse_individuals) < n:
                    diverse_individuals.append(self.front[ind_idx])
                    selected_indices.add(ind_idx)
        
        return diverse_individuals
    
    def get_extreme_individuals(self) -> Dict[str, Individual]:
        """
        Get individuals with extreme values in each objective
        
        Returns:
            Dictionary mapping objective to extreme individual
        """
        if not self.front:
            return {}
        
        extremes = {}
        
        for obj in self.objectives:
            # Maximization for all objectives
            extreme_ind = max(self.front, key=lambda ind: ind.fitness.get(obj, 0.0))
            extremes[obj] = extreme_ind
        
        return extremes
    
    def get_hypervolume(self, reference_point: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate hypervolume indicator
        
        Args:
            reference_point: Reference point for hypervolume calculation
            
        Returns:
            Hypervolume value
        """
        if not self.front:
            return 0.0
        
        # Default reference point (nadir point)
        if reference_point is None:
            reference_point = {}
            for obj in self.objectives:
                # Get worst value for each objective
                worst_val = min(ind.fitness.get(obj, 0.0) for ind in self.front)
                reference_point[obj] = worst_val - 0.1  # Slightly worse than worst
        
        # For 2D or 3D objectives, calculate hypervolume
        if len(self.objectives) == 2:
            return self._calculate_2d_hypervolume(reference_point)
        elif len(self.objectives) == 3:
            return self._calculate_3d_hypervolume(reference_point)
        else:
            # Approximate for higher dimensions
            return self._approximate_hypervolume(reference_point)
    
    def _calculate_2d_hypervolume(self, reference_point: Dict[str, float]) -> float:
        """Calculate hypervolume for 2 objectives"""
        # Get points sorted by first objective
        points = []
        for ind in self.front:
            point = (ind.fitness.get(self.objectives[0], 0.0),
                    ind.fitness.get(self.objectives[1], 0.0))
            points.append(point)
        
        # Sort by first objective (descending)
        points.sort(key=lambda x: x[0], reverse=True)
        
        # Calculate hypervolume using Lebesgue measure
        hypervolume = 0.0
        ref_x = reference_point.get(self.objectives[0], 0.0)
        ref_y = reference_point.get(self.objectives[1], 0.0)
        
        last_y = ref_y
        for x, y in points:
            if y > last_y:  # Non-dominated in y direction
                hypervolume += (x - ref_x) * (y - last_y)
                last_y = y
        
        return hypervolume
    
    def _calculate_3d_hypervolume(self, reference_point: Dict[str, float]) -> float:
        """Calculate hypervolume for 3 objectives (simplified)"""
        # Use Monte Carlo approximation for 3D
        n_samples = 1000
        count_inside = 0
        
        # Get bounds
        bounds = {}
        for obj in self.objectives:
            values = [ind.fitness.get(obj, 0.0) for ind in self.front]
            bounds[obj] = (min(values), max(values))
        
        ref_point = [reference_point.get(obj, bounds[obj][0] - 0.1) for obj in self.objectives]
        
        # Generate random points
        for _ in range(n_samples):
            # Sample random point
            point = []
            for obj in self.objectives:
                low, high = bounds[obj]
                point.append(np.random.uniform(ref_point[0], high))
            
            # Check if point is dominated by any front individual
            dominated = False
            for ind in self.front:
                if all(point[i] <= ind.fitness.get(self.objectives[i], 0.0) 
                      for i in range(3)):
                    dominated = True
                    break
            
            if dominated:
                count_inside += 1
        
        # Calculate volume
        volume = 1.0
        for obj in self.objectives:
            low, high = bounds[obj]
            volume *= (high - ref_point[0])
        
        hypervolume = volume * (count_inside / n_samples)
        return hypervolume
    
    def _approximate_hypervolume(self, reference_point: Dict[str, float]) -> float:
        """Approximate hypervolume for higher dimensions"""
        # Simple approximation: product of normalized distances to reference
        hypervolume = 1.0
        
        for obj in self.objectives:
            # Get best value for this objective
            best_val = max(ind.fitness.get(obj, 0.0) for ind in self.front)
            ref_val = reference_point.get(obj, 0.0)
            
            if best_val > ref_val:
                hypervolume *= (best_val - ref_val)
        
        return hypervolume
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get Pareto front statistics"""
        if not self.front:
            return {'size': 0}
        
        stats = {
            'size': len(self.front),
            'objectives': self.objectives,
            'objective_ranges': {},
            'extreme_values': {}
        }
        
        # Calculate ranges for each objective
        for obj in self.objectives:
            values = [ind.fitness.get(obj, 0.0) for ind in self.front]
            stats['objective_ranges'][obj] = {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values)
            }
        
        # Get extreme individuals
        extremes = self.get_extreme_individuals()
        stats['extreme_values'] = {
            obj: ind.fitness for obj, ind in extremes.items()
        }
        
        # Calculate hypervolume
        stats['hypervolume'] = self.get_hypervolume()
        
        return stats
    
    def save(self, filepath: str) -> None:
        """Save Pareto front to file"""
        data = {
            'front': [ind.to_dict() for ind in self.front],
            'objectives': self.objectives,
            'max_size': self.max_size,
            'statistics': self.get_statistics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Pareto front saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load Pareto front from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Load individuals
        self.front = []
        for ind_data in data['front']:
            individual = Individual.from_dict(ind_data)
            self.front.append(individual)
        
        # Load configuration
        self.objectives = data.get('objectives', ['accuracy', 'efficiency', 'complexity'])
        self.max_size = data.get('max_size', 50)
        
        logger.info(f"Pareto front loaded from {filepath}. Size: {len(self.front)}")
    
    def plot_front_2d(self, obj1: str, obj2: str, 
                     save_path: Optional[str] = None) -> None:
        """
        Plot 2D projection of Pareto front
        
        Args:
            obj1: First objective
            obj2: Second objective
            save_path: Path to save plot (optional)
        """
        if not self.front:
            logger.warning("Pareto front is empty")
            return
        
        import matplotlib.pyplot as plt
        
        # Extract objective values
        x_vals = [ind.fitness.get(obj1, 0.0) for ind in self.front]
        y_vals = [ind.fitness.get(obj2, 0.0) for ind in self.front]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        plt.scatter(x_vals, y_vals, c='blue', alpha=0.6, s=100, edgecolors='black')
        
        # Add labels for extreme points
        extremes = self.get_extreme_individuals()
        for obj, ind in extremes.items():
            if obj in [obj1, obj2]:
                x = ind.fitness.get(obj1, 0.0)
                y = ind.fitness.get(obj2, 0.0)
                plt.annotate(f"Best {obj}", (x, y), 
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=10, fontweight='bold')
        
        plt.xlabel(obj1, fontsize=12)
        plt.ylabel(obj2, fontsize=12)
        plt.title(f"Pareto Front: {obj1} vs {obj2}", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pareto front plot saved to {save_path}")
        
        plt.show()
    
    def __len__(self) -> int:
        return len(self.front)
    
    def __iter__(self):
        return iter(self.front)
    
    def __getitem__(self, idx):
        return self.front[idx]
