"""
Grammar definition module for GP
Execution Order: 21
"""

import random
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
import json
import logging

from config.constants import GRAMMAR_RULES, TERMINAL_PARAMS

logger = logging.getLogger(__name__)


@dataclass
class ProductionRule:
    """Production rule in grammar"""
    lhs: str  # Left-hand side (non-terminal)
    rhs: List[str]  # Right-hand side (sequence of symbols)
    probability: float = 1.0


@dataclass
class TerminalSymbol:
    """Terminal symbol with parameters"""
    name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    valid_parameters: Dict[str, List[Any]] = field(default_factory=dict)


class Grammar:
    """
    Context-free grammar G = (N, Σ, P, S)
    
    N: Non-terminals
    Σ: Terminals
    P: Production rules
    S: Start symbol
    """
    
    def __init__(self, custom_rules: Optional[Dict[str, Any]] = None):
        """
        Initialize grammar
        
        Args:
            custom_rules: Custom grammar rules (optional)
        """
        # Use custom rules or defaults
        if custom_rules:
            self.non_terminals = custom_rules.get('non_terminals', GRAMMAR_RULES)
            self.terminals = custom_rules.get('terminals', TERMINAL_PARAMS)
        else:
            self.non_terminals = GRAMMAR_RULES
            self.terminals = TERMINAL_PARAMS
        
        # Build production rules
        self.productions = self._build_production_rules()
        
        # Start symbol
        self.start_symbol = 'S'
        
        # Valid sequences for each non-terminal
        self.valid_sequences = self._build_valid_sequences()
        
        # Depth limits for each symbol
        self.max_depths = {
            'S': 6,
            'Preprocess': 2,
            'Segment': 3,
            'Postprocess': 2
        }
        
        logger.info(f"Grammar initialized with {len(self.non_terminals)} non-terminals, "
                   f"{len(self.terminals)} terminals")
    
    def _build_production_rules(self) -> Dict[str, List[ProductionRule]]:
        """Build production rules from grammar definition"""
        productions = {}
        
        for lhs, rhs_list in self.non_terminals.items():
            productions[lhs] = []
            for rhs_str in rhs_list:
                # Split RHS into symbols
                rhs_symbols = rhs_str.split()
                
                # Create production rule
                rule = ProductionRule(
                    lhs=lhs,
                    rhs=rhs_symbols,
                    probability=1.0 / len(rhs_list)  # Equal probability
                )
                
                productions[lhs].append(rule)
        
        return productions
    
    def _build_valid_sequences(self) -> Dict[str, List[Tuple[str, ...]]]:
        """Build valid sequences for each non-terminal"""
        valid_sequences = {}
        
        for lhs, rules in self.productions.items():
            valid_sequences[lhs] = []
            for rule in rules:
                valid_sequences[lhs].append(tuple(rule.rhs))
        
        return valid_sequences
    
    def generate_random_individual(self, max_depth: int = 6) -> 'PipelineNode':
        """
        Generate random individual using grammar
        
        Args:
            max_depth: Maximum tree depth
            
        Returns:
            PipelineNode representing the individual
        """
        from .individual import PipelineNode
        
        return self._expand_symbol(
            symbol=self.start_symbol,
            depth=0,
            max_depth=max_depth
        )
    
    def _expand_symbol(self, symbol: str, depth: int, max_depth: int) -> 'PipelineNode':
        """
        Recursively expand a symbol
        
        Args:
            symbol: Symbol to expand
            depth: Current depth
            max_depth: Maximum depth
            
        Returns:
            PipelineNode
        """
        from .individual import PipelineNode
        
        # Check if terminal symbol
        if symbol in self.terminals:
            # Terminal node
            params = self._generate_terminal_parameters(symbol)
            return PipelineNode(
                operation=symbol,
                parameters=params,
                children=[],
                depth=depth
            )
        
        # Non-terminal: check depth limit
        if depth >= max_depth or symbol not in self.productions:
            # Force terminal expansion or use default
            return self._expand_to_terminal(symbol, depth)
        
        # Select production rule
        rules = self.productions[symbol]
        rule = random.choice(rules)
        
        # Create node
        node = PipelineNode(
            operation=symbol,
            parameters={},
            children=[],
            depth=depth
        )
        
        # Expand RHS symbols
        for child_symbol in rule.rhs:
            child_node = self._expand_symbol(
                symbol=child_symbol,
                depth=depth + 1,
                max_depth=max_depth
            )
            node.children.append(child_node)
        
        return node
    
    def _expand_to_terminal(self, symbol: str, depth: int) -> 'PipelineNode':
        """Expand non-terminal to terminal when depth limit reached"""
        from .individual import PipelineNode
        
        # Try to find a related terminal
        related_terminals = []
        
        if symbol == 'Preprocess':
            related_terminals = ['PCA', 'Denoise', 'Normalize', 'None']
        elif symbol == 'Segment':
            related_terminals = ['KMeans', 'SS_PSO', 'Gradient', 'Watershed']
        elif symbol == 'Postprocess':
            related_terminals = ['Morphology', 'MRF', 'None']
        else:
            related_terminals = ['None']
        
        # Select random terminal
        terminal = random.choice(related_terminals)
        params = self._generate_terminal_parameters(terminal)
        
        return PipelineNode(
            operation=terminal,
            parameters=params,
            children=[],
            depth=depth
        )
    
    def _generate_terminal_parameters(self, terminal: str) -> Dict[str, Any]:
        """Generate random parameters for terminal"""
        if terminal not in self.terminals:
            return {}
        
        params = {}
        param_defs = self.terminals[terminal]
        
        for param_name, possible_values in param_defs.items():
            if param_name == 'parameters':
                # Special handling for parameter tuples
                if possible_values and isinstance(possible_values, list):
                    params['parameters'] = random.choice(possible_values)
            elif possible_values:
                if isinstance(possible_values, list):
                    params[param_name] = random.choice(possible_values)
                else:
                    params[param_name] = possible_values
        
        return params
    
    def validate_individual(self, individual: 'PipelineNode') -> bool:
        """
        Validate if individual conforms to grammar
        
        Args:
            individual: PipelineNode to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            self._validate_node(individual)
            return True
        except ValueError as e:
            logger.debug(f"Validation failed: {e}")
            return False
    
    def _validate_node(self, node: 'PipelineNode') -> None:
        """Recursively validate node"""
        # Check if operation exists
        if node.operation not in self.terminals and node.operation not in self.non_terminals:
            raise ValueError(f"Unknown operation: {node.operation}")
        
        # Terminal node: should have no children
        if node.operation in self.terminals:
            if node.children:
                raise ValueError(f"Terminal {node.operation} should have no children")
            
            # Validate parameters
            self._validate_parameters(node.operation, node.parameters)
            
        # Non-terminal node: validate children
        else:
            if node.operation not in self.valid_sequences:
                raise ValueError(f"Non-terminal {node.operation} has no valid sequences")
            
            # Check if children sequence is valid
            child_ops = tuple(child.operation for child in node.children)
            if child_ops not in self.valid_sequences[node.operation]:
                raise ValueError(
                    f"Invalid children sequence for {node.operation}: {child_ops}. "
                    f"Valid sequences: {self.valid_sequences[node.operation]}"
                )
            
            # Validate each child
            for child in node.children:
                self._validate_node(child)
    
    def _validate_parameters(self, operation: str, parameters: Dict[str, Any]) -> None:
        """Validate parameters for operation"""
        if operation not in self.terminals:
            return
        
        param_defs = self.terminals[operation]
        
        for param_name, value in parameters.items():
            if param_name not in param_defs:
                raise ValueError(f"Unknown parameter {param_name} for {operation}")
            
            # Check if value is valid
            possible_values = param_defs[param_name]
            if param_name != 'parameters' and possible_values:
                if value not in possible_values:
                    raise ValueError(
                        f"Invalid value {value} for parameter {param_name} of {operation}. "
                        f"Valid values: {possible_values}"
                    )
    
    def mutate(self, individual: 'PipelineNode', mutation_rate: float = 0.3) -> 'PipelineNode':
        """
        Mutate individual while respecting grammar
        
        Args:
            individual: Individual to mutate
            mutation_rate: Probability of mutation
            
        Returns:
            Mutated individual
        """
        if random.random() < mutation_rate:
            return self._mutate_node(individual, mutation_rate)
        return individual
    
    def _mutate_node(self, node: 'PipelineNode', mutation_rate: float) -> 'PipelineNode':
        """Recursively mutate node"""
        from .individual import PipelineNode
        
        # Terminal node: mutate parameters
        if node.operation in self.terminals:
            if random.random() < mutation_rate:
                node.parameters = self._mutate_parameters(node.operation, node.parameters)
        
        # Non-terminal node: possible structural mutation
        else:
            # Structural mutation: replace subtree
            if random.random() < mutation_rate * 0.5 and node.depth < self.max_depths.get(node.operation, 3):
                # Generate new subtree
                new_subtree = self._expand_symbol(
                    symbol=node.operation,
                    depth=node.depth,
                    max_depth=self.max_depths.get(node.operation, 3)
                )
                
                # Replace this node (but keep parent reference)
                node.operation = new_subtree.operation
                node.parameters = new_subtree.parameters
                node.children = new_subtree.children
                node.depth = new_subtree.depth
            
            # Mutate children
            for child in node.children:
                self._mutate_node(child, mutation_rate)
        
        return node
    
    def _mutate_parameters(self, operation: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate parameters of terminal"""
        if operation not in self.terminals:
            return parameters
        
        param_defs = self.terminals[operation]
        mutated_params = parameters.copy()
        
        for param_name in param_defs.keys():
            if param_name == 'parameters':
                continue
            
            if random.random() < 0.3:  # 30% chance to mutate each parameter
                possible_values = param_defs[param_name]
                if possible_values and isinstance(possible_values, list):
                    # Select new value different from current
                    current_value = mutated_params.get(param_name)
                    new_values = [v for v in possible_values if v != current_value]
                    if new_values:
                        mutated_params[param_name] = random.choice(new_values)
        
        return mutated_params
    
    def crossover(self, parent1: 'PipelineNode', parent2: 'PipelineNode',
                 crossover_rate: float = 0.9) -> Tuple['PipelineNode', 'PipelineNode']:
        """
        Crossover two individuals while respecting grammar
        
        Args:
            parent1: First parent
            parent2: Second parent
            crossover_rate: Probability of crossover
            
        Returns:
            Tuple of (child1, child2)
        """
        if random.random() > crossover_rate:
            return parent1, parent2
        
        # Find compatible crossover points
        points1 = self._find_crossover_points(parent1)
        points2 = self._find_crossover_points(parent2)
        
        if not points1 or not points2:
            return parent1, parent2
        
        # Select random crossover points
        node1, idx1 = random.choice(points1)
        node2, idx2 = random.choice(points2)
        
        # Check if crossover is valid (same operation type)
        if node1.operation == node2.operation:
            # Swap subtrees
            temp = node1.children[idx1]
            node1.children[idx1] = node2.children[idx2]
            node2.children[idx2] = temp
        
        return parent1, parent2
    
    def _find_crossover_points(self, node: 'PipelineNode') -> List[Tuple['PipelineNode', int]]:
        """
        Find all valid crossover points in tree
        
        Returns:
            List of (parent_node, child_index) pairs
        """
        points = []
        
        if node.children:
            for i, child in enumerate(node.children):
                points.append((node, i))
                points.extend(self._find_crossover_points(child))
        
        return points
    
    def individual_to_dict(self, individual: 'PipelineNode') -> Dict[str, Any]:
        """Convert individual to dictionary"""
        return {
            'operation': individual.operation,
            'parameters': individual.parameters,
            'children': [self.individual_to_dict(child) for child in individual.children],
            'depth': individual.depth
        }
    
    def individual_from_dict(self, data: Dict[str, Any]) -> 'PipelineNode':
        """Create individual from dictionary"""
        from .individual import PipelineNode
        
        node = PipelineNode(
            operation=data['operation'],
            parameters=data.get('parameters', {}),
            depth=data.get('depth', 0)
        )
        
        for child_data in data.get('children', []):
            child_node = self.individual_from_dict(child_data)
            node.children.append(child_node)
        
        return node
    
    def get_grammar_summary(self) -> Dict[str, Any]:
        """Get grammar summary"""
        return {
            'non_terminals': list(self.non_terminals.keys()),
            'terminals': list(self.terminals.keys()),
            'start_symbol': self.start_symbol,
            'max_depths': self.max_depths,
            'production_counts': {k: len(v) for k, v in self.productions.items()}
        }
