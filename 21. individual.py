"""
Individual representation module for GP
Execution Order: 22
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import random
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class PipelineNode:
    """
    Node in grammar-guided pipeline tree
    
    Represents either:
    1. Terminal operation (e.g., PCA, KMeans) with parameters
    2. Non-terminal operation (e.g., Segment) with children
    """
    operation: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    children: List['PipelineNode'] = field(default_factory=list)
    depth: int = 0
    
    def __repr__(self) -> str:
        """String representation"""
        if self.parameters:
            params_str = ', '.join(f"{k}={v}" for k, v in self.parameters.items())
            return f"{self.operation}({params_str})"
        return self.operation
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary"""
        return {
            'operation': self.operation,
            'parameters': self.parameters,
            'children': [child.to_dict() for child in self.children],
            'depth': self.depth
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineNode':
        """Create node from dictionary"""
        node = cls(
            operation=data['operation'],
            parameters=data.get('parameters', {}),
            depth=data.get('depth', 0)
        )
        
        for child_data in data.get('children', []):
            child_node = cls.from_dict(child_data)
            node.children.append(child_node)
        
        return node
    
    def get_hash(self) -> str:
        """Get unique hash for node"""
        # Convert to JSON string and hash
        json_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()[:8]
    
    def get_size(self) -> int:
        """Get total number of nodes in subtree"""
        size = 1  # Count this node
        for child in self.children:
            size += child.get_size()
        return size
    
    def get_depth(self) -> int:
        """Get maximum depth of subtree"""
        if not self.children:
            return self.depth
        
        child_depths = [child.get_depth() for child in self.children]
        return max(child_depths)
    
    def get_operations(self) -> List[str]:
        """Get list of all operations in subtree"""
        operations = [self.operation]
        for child in self.children:
            operations.extend(child.get_operations())
        return operations
    
    def count_operation(self, operation: str) -> int:
        """Count occurrences of specific operation"""
        count = 1 if self.operation == operation else 0
        for child in self.children:
            count += child.count_operation(operation)
        return count
    
    def get_terminal_nodes(self) -> List['PipelineNode']:
        """Get all terminal nodes in subtree"""
        terminals = []
        
        # Terminal node (no children)
        if not self.children:
            terminals.append(self)
        else:
            # Non-terminal: check children
            for child in self.children:
                terminals.extend(child.get_terminal_nodes())
        
        return terminals
    
    def get_non_terminal_nodes(self) -> List['PipelineNode']:
        """Get all non-terminal nodes in subtree"""
        non_terminals = []
        
        # Non-terminal node (has children)
        if self.children:
            non_terminals.append(self)
            for child in self.children:
                non_terminals.extend(child.get_non_terminal_nodes())
        
        return non_terminals
    
    def find_node(self, node_hash: str) -> Optional['PipelineNode']:
        """Find node by hash"""
        if self.get_hash() == node_hash:
            return self
        
        for child in self.children:
            found = child.find_node(node_hash)
            if found:
                return found
        
        return None
    
    def replace_subtree(self, old_hash: str, new_subtree: 'PipelineNode') -> bool:
        """
        Replace subtree with given hash
        
        Args:
            old_hash: Hash of subtree to replace
            new_subtree: New subtree
            
        Returns:
            True if replacement successful
        """
        # Check if this node should be replaced
        if self.get_hash() == old_hash:
            # Can't replace root node in this method
            return False
        
        # Check children
        for i, child in enumerate(self.children):
            if child.get_hash() == old_hash:
                self.children[i] = new_subtree
                # Update depth of new subtree
                self._update_depth(new_subtree, self.depth + 1)
                return True
            
            # Recursively check child's children
            if child.replace_subtree(old_hash, new_subtree):
                return True
        
        return False
    
    def _update_depth(self, node: 'PipelineNode', new_depth: int) -> None:
        """Update depth of node and all children"""
        node.depth = new_depth
        for child in node.children:
            self._update_depth(child, new_depth + 1)


@dataclass
class Individual:
    """
    Individual in GP population
    
    Contains:
    1. Pipeline tree
    2. Fitness values (multi-objective)
    3. Metadata
    """
    pipeline: PipelineNode
    fitness: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"ind_{random.randint(0, 1000000)}")
    
    def __post_init__(self):
        """Post-initialization setup"""
        if not self.metadata:
            self.metadata = {
                'creation_time': None,
                'generation': 0,
                'parent_ids': [],
                'mutation_count': 0,
                'crossover_count': 0
            }
    
    def __repr__(self) -> str:
        """String representation"""
        fitness_str = ', '.join(f"{k}={v:.4f}" for k, v in self.fitness.items())
        return f"Individual(id={self.id}, fitness=[{fitness_str}], pipeline={str(self.pipeline)})"
    
    def dominates(self, other: 'Individual') -> bool:
        """
        Check Pareto dominance
        
        Args:
            other: Other individual
            
        Returns:
            True if this individual dominates the other
        """
        # Check if all objectives are at least as good
        all_not_worse = True
        at_least_one_better = False
        
        # Check common objectives
        common_objectives = set(self.fitness.keys()) & set(other.fitness.keys())
        
        for obj in common_objectives:
            if self.fitness[obj] < other.fitness[obj]:
                all_not_worse = False
                break
            elif self.fitness[obj] > other.fitness[obj]:
                at_least_one_better = True
        
        # Also check if we have objectives that the other doesn't
        if not at_least_one_better:
            unique_objectives = set(self.fitness.keys()) - set(other.fitness.keys())
            at_least_one_better = len(unique_objectives) > 0
        
        return all_not_worse and at_least_one_better
    
    def weakly_dominates(self, other: 'Individual') -> bool:
        """
        Check weak Pareto dominance
        
        Args:
            other: Other individual
            
        Returns:
            True if this individual weakly dominates the other
        """
        # All objectives are at least as good
        common_objectives = set(self.fitness.keys()) & set(other.fitness.keys())
        
        for obj in common_objectives:
            if self.fitness[obj] < other.fitness[obj]:
                return False
        
        return True
    
    def is_incomparable(self, other: 'Individual') -> bool:
        """
        Check if individuals are incomparable
        
        Args:
            other: Other individual
            
        Returns:
            True if neither dominates the other
        """
        return not self.dominates(other) and not other.dominates(self)
    
    def get_total_fitness(self, weights: Dict[str, float] = None) -> float:
        """
        Get weighted sum of fitness values
        
        Args:
            weights: Objective weights (default: equal weights)
            
        Returns:
            Weighted fitness sum
        """
        if not self.fitness:
            return 0.0
        
        if weights is None:
            # Equal weights
            weights = {obj: 1.0 / len(self.fitness) for obj in self.fitness.keys()}
        
        total = 0.0
        for obj, value in self.fitness.items():
            if obj in weights:
                total += weights[obj] * value
        
        return total
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert individual to dictionary"""
        return {
            'id': self.id,
            'pipeline': self.pipeline.to_dict(),
            'fitness': self.fitness,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Individual':
        """Create individual from dictionary"""
        pipeline = PipelineNode.from_dict(data['pipeline'])
        individual = cls(
            pipeline=pipeline,
            fitness=data.get('fitness', {}),
            metadata=data.get('metadata', {}),
            id=data.get('id', f"ind_{random.randint(0, 1000000)}")
        )
        return individual
    
    def get_hash(self) -> str:
        """Get unique hash for individual"""
        # Hash based on pipeline structure
        return self.pipeline.get_hash()
    
    def get_size(self) -> int:
        """Get pipeline size (number of nodes)"""
        return self.pipeline.get_size()
    
    def get_depth(self) -> int:
        """Get pipeline depth"""
        return self.pipeline.get_depth()
    
    def get_complexity(self) -> float:
        """Get complexity measure"""
        # Complexity = depth * log(size)
        size = self.get_size()
        depth = self.get_depth()
        
        if size <= 1:
            return 0.0
        
        return depth * (1.0 + np.log(size))
    
    def get_operation_counts(self) -> Dict[str, int]:
        """Count occurrences of each operation"""
        operations = self.pipeline.get_operations()
        counts = {}
        
        for op in operations:
            counts[op] = counts.get(op, 0) + 1
        
        return counts
    
    def is_valid(self, grammar) -> bool:
        """
        Check if individual is valid according to grammar
        
        Args:
            grammar: Grammar instance
            
        Returns:
            True if valid
        """
        return grammar.validate_individual(self.pipeline)
    
    def copy(self) -> 'Individual':
        """Create deep copy of individual"""
        import copy
        return copy.deepcopy(self)
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Update metadata"""
        self.metadata[key] = value
    
    def add_parent(self, parent_id: str) -> None:
        """Add parent ID to metadata"""
        if 'parent_ids' not in self.metadata:
            self.metadata['parent_ids'] = []
        
        if parent_id not in self.metadata['parent_ids']:
            self.metadata['parent_ids'].append(parent_id)
    
    def increment_mutation_count(self) -> None:
        """Increment mutation count"""
        self.metadata['mutation_count'] = self.metadata.get('mutation_count', 0) + 1
    
    def increment_crossover_count(self) -> None:
        """Increment crossover count"""
        self.metadata['crossover_count'] = self.metadata.get('crossover_count', 0) + 1


class IndividualFactory:
    """Factory for creating individuals"""
    
    def __init__(self, grammar):
        """
        Initialize factory
        
        Args:
            grammar: Grammar instance
        """
        self.grammar = grammar
    
    def create_random(self, max_depth: int = 6) -> Individual:
        """
        Create random individual
        
        Args:
            max_depth: Maximum pipeline depth
            
        Returns:
            Random individual
        """
        pipeline = self.grammar.generate_random_individual(max_depth)
        return Individual(pipeline=pipeline)
    
    def create_from_pipeline(self, pipeline_dict: Dict[str, Any]) -> Individual:
        """
        Create individual from pipeline dictionary
        
        Args:
            pipeline_dict: Pipeline dictionary
            
        Returns:
            Individual
        """
        pipeline = self.grammar.individual_from_dict(pipeline_dict)
        return Individual(pipeline=pipeline)
    
    def create_mutated(self, parent: Individual, mutation_rate: float = 0.3) -> Individual:
        """
        Create mutated individual
        
        Args:
            parent: Parent individual
            mutation_rate: Mutation rate
            
        Returns:
            Mutated individual
        """
        # Deep copy parent pipeline
        import copy
        mutated_pipeline = copy.deepcopy(parent.pipeline)
        
        # Apply mutation
        mutated_pipeline = self.grammar.mutate(mutated_pipeline, mutation_rate)
        
        # Create new individual
        child = Individual(pipeline=mutated_pipeline)
        child.add_parent(parent.id)
        child.increment_mutation_count()
        child.metadata['generation'] = parent.metadata.get('generation', 0) + 1
        
        return child
    
    def create_crossover(self, parent1: Individual, parent2: Individual,
                        crossover_rate: float = 0.9) -> Tuple[Individual, Individual]:
        """
        Create crossover children
        
        Args:
            parent1: First parent
            parent2: Second parent
            crossover_rate: Crossover rate
            
        Returns:
            Tuple of (child1, child2)
        """
        # Deep copy parent pipelines
        import copy
        child1_pipeline = copy.deepcopy(parent1.pipeline)
        child2_pipeline = copy.deepcopy(parent2.pipeline)
        
        # Apply crossover
        child1_pipeline, child2_pipeline = self.grammar.crossover(
            child1_pipeline, child2_pipeline, crossover_rate
        )
        
        # Create children
        child1 = Individual(pipeline=child1_pipeline)
        child2 = Individual(pipeline=child2_pipeline)
        
        # Update metadata
        for child in [child1, child2]:
            child.add_parent(parent1.id)
            child.add_parent(parent2.id)
            child.increment_crossover_count()
            child.metadata['generation'] = max(
                parent1.metadata.get('generation', 0),
                parent2.metadata.get('generation', 0)
            ) + 1
        
        return child1, child2
