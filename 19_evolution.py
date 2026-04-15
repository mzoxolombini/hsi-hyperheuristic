"""
Evolutionary operations module for GP
Execution Order: 24
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
import random
import logging
from tqdm import tqdm
import time

from .individual import Individual, IndividualFactory
from .evaluation import MultiObjectiveEvaluator
from .pareto_front import ParetoFront
from .grammar import Grammar

logger = logging.getLogger(__name__)


class GeneticProgramming:
    """
    Main Genetic Programming engine
    
    Implements:
    1. Population initialization
    2. Fitness evaluation
    3. Selection operators
    4. Crossover and mutation
    5. Pareto-based evolution
    """
    
    def __init__(self, config: Dict[str, Any], grammar: Optional[Grammar] = None,
                 results_dir: str = "./results"):
        """
        Initialize GP engine
        
        Args:
            config: GP configuration
            grammar: Grammar instance (optional)
            results_dir: Results directory
        """
        self.config = config
        self.results_dir = results_dir
        
        # Initialize components
        self.grammar = grammar or Grammar()
        self.factory = IndividualFactory(self.grammar)
        self.evaluator = MultiObjectiveEvaluator(config)
        self.pareto_front = ParetoFront()
        
        # Population
        self.population: List[Individual] = []
        self.generation = 0
        
        # History tracking
        self.history = {
            'generations': [],
            'best_fitness': [],
            'population_stats': [],
            'pareto_sizes': []
        }
        
        # Fitness cache for efficiency
        self.fitness_cache: Dict[str, Dict[str, float]] = {}
        
        logger.info("Genetic Programming engine initialized")
    
    def initialize_population(self) -> None:
        """Initialize population using ramped half-and-half"""
        population_size = self.config.get('population_size', 100)
        
        logger.info(f"Initializing population of size {population_size}")
        
        # Ramped half-and-half: 50% full, 50% grow
        half_size = population_size // 2
        
        # Full method (deep trees)
        full_trees = []
        for i in range(half_size):
            max_depth = random.randint(3, self.config.get('max_depth', 6))
            individual = self.factory.create_random(max_depth)
            full_trees.append(individual)
        
        # Grow method (varied depth)
        grow_trees = []
        for i in range(half_size):
            max_depth = random.randint(2, self.config.get('max_depth', 6))
            individual = self.factory.create_random(max_depth)
            grow_trees.append(individual)
        
        # Combine
        self.population = full_trees + grow_trees
        
        # Ensure unique individuals
        self._ensure_population_diversity()
        
        logger.info(f"Population initialized with {len(self.population)} individuals")
    
    def _ensure_population_diversity(self, max_attempts: int = 100) -> None:
        """Ensure population has unique individuals"""
        seen_hashes = set()
        unique_population = []
        
        for individual in self.population:
            individual_hash = individual.get_hash()
            
            if individual_hash not in seen_hashes:
                seen_hashes.add(individual_hash)
                unique_population.append(individual)
            else:
                # Generate new unique individual
                for _ in range(max_attempts):
                    max_depth = random.randint(2, self.config.get('max_depth', 6))
                    new_individual = self.factory.create_random(max_depth)
                    new_hash = new_individual.get_hash()
                    
                    if new_hash not in seen_hashes:
                        seen_hashes.add(new_hash)
                        unique_population.append(new_individual)
                        break
        
        self.population = unique_population
    
    def evaluate_population(self, data: np.ndarray, ground_truth: np.ndarray,
                          meta_features_list: Optional[List[Dict[str, float]]] = None) -> None:
        """
        Evaluate population fitness
        
        Args:
            data: Input data [H, W, B]
            ground_truth: Ground truth labels [H, W]
            meta_features_list: List of meta-features for each patch
        """
        logger.info(f"Evaluating population of {len(self.population)} individuals")
        
        total_valid = 0
        total_invalid = 0
        
        for idx, individual in enumerate(tqdm(self.population, desc="Evaluating")):
            # Check cache
            individual_hash = individual.get_hash()
            if individual_hash in self.fitness_cache:
                individual.fitness = self.fitness_cache[individual_hash]
                total_valid += 1
                continue
            
            try:
                # Evaluate individual
                fitness = self.evaluator.evaluate(
                    individual=individual,
                    data=data,
                    ground_truth=ground_truth,
                    meta_features=meta_features_list[idx] if meta_features_list else None
                )
                
                # Store fitness
                individual.fitness = fitness
                self.fitness_cache[individual_hash] = fitness
                
                if fitness.get('valid', False):
                    total_valid += 1
                else:
                    total_invalid += 1
                    
            except Exception as e:
                logger.warning(f"Evaluation failed for individual {idx}: {e}")
                individual.fitness = {'valid': False, 'accuracy': 0.0, 'efficiency': 0.0, 'complexity': 0.0}
                total_invalid += 1
        
        logger.info(f"Evaluation completed: {total_valid} valid, {total_invalid} invalid")
    
    def evolve(self, data: np.ndarray, ground_truth: np.ndarray,
               n_generations: int = 50,
               meta_features_list: Optional[List[Dict[str, float]]] = None) -> Optional[Individual]:
        """
        Main evolution loop
        
        Args:
            data: Input data
            ground_truth: Ground truth labels
            n_generations: Number of generations
            meta_features_list: List of meta-features
            
        Returns:
            Best individual from Pareto front
        """
        logger.info(f"Starting evolution for {n_generations} generations")
        
        # Initialize population
        self.initialize_population()
        
        # Evolution loop
        for gen in range(n_generations):
            self.generation = gen + 1
            logger.info(f"\n{'='*60}")
            logger.info(f"Generation {self.generation}/{n_generations}")
            logger.info(f"{'='*60}")
            
            # Evaluate population
            self.evaluate_population(data, ground_truth, meta_features_list)
            
            # Update Pareto front
            self._update_pareto_front()
            
            # Log generation statistics
            self._log_generation_stats()
            
            # Check termination criteria
            if self._check_termination_criteria():
                logger.info("Termination criteria met. Stopping evolution.")
                break
            
            # Selection and breeding (except last generation)
            if gen < n_generations - 1:
                self._selection_and_breeding()
            
            # Save checkpoint
            if (gen + 1) % 10 == 0:
                self._save_checkpoint(gen + 1)
        
        # Return best individual
        best_individual = self._get_best_individual()
        
        if best_individual:
            logger.info(f"\nEvolution completed. Best individual: {best_individual}")
            logger.info(f"Best fitness: {best_individual.fitness}")
        else:
            logger.warning("Evolution completed but no valid individuals found")
        
        return best_individual
    
    def _update_pareto_front(self) -> None:
        """Update Pareto front with current population"""
        valid_individuals = [ind for ind in self.population if ind.fitness.get('valid', False)]
        
        if not valid_individuals:
            logger.warning("No valid individuals in population")
            return
        
        # Update Pareto front
        self.pareto_front.update(valid_individuals)
        
        logger.info(f"Pareto front size: {len(self.pareto_front.front)}")
    
    def _log_generation_stats(self) -> None:
        """Log generation statistics"""
        valid_individuals = [ind for ind in self.population if ind.fitness.get('valid', False)]
        
        if not valid_individuals:
            logger.warning("No valid individuals to compute statistics")
            return
        
        # Compute statistics
        accuracies = [ind.fitness.get('accuracy', 0.0) for ind in valid_individuals]
        efficiencies = [ind.fitness.get('efficiency', 0.0) for ind in valid_individuals]
        complexities = [1.0 / ind.fitness.get('complexity', 1.0) for ind in valid_individuals]
        
        stats = {
            'generation': self.generation,
            'population_size': len(self.population),
            'valid_count': len(valid_individuals),
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'accuracy_max': np.max(accuracies),
            'efficiency_mean': np.mean(efficiencies),
            'complexity_mean': np.mean(complexities),
            'pareto_size': len(self.pareto_front.front)
        }
        
        self.history['generations'].append(stats)
        
        logger.info(f"  Valid individuals: {stats['valid_count']}/{stats['population_size']}")
        logger.info(f"  Accuracy: {stats['accuracy_mean']:.4f} ± {stats['accuracy_std']:.4f}")
        logger.info(f"  Max accuracy: {stats['accuracy_max']:.4f}")
        logger.info(f"  Efficiency: {stats['efficiency_mean']:.4f}")
        logger.info(f"  Complexity: {stats['complexity_mean']:.4f}")
    
    def _check_termination_criteria(self) -> bool:
        """Check termination criteria"""
        # Check max generations (handled by evolve loop)
        
        # Check convergence (no improvement in last N generations)
        if len(self.history['generations']) >= 10:
            recent_generations = self.history['generations'][-10:]
            recent_accuracies = [g['accuracy_max'] for g in recent_generations]
            
            # Check if improvement is below threshold
            improvement = max(recent_accuracies) - min(recent_accuracies)
            if improvement < 0.01:  # 1% improvement threshold
                logger.info(f"Convergence detected: improvement = {improvement:.4f}")
                return True
        
        # Check if Pareto front has stabilized
        if len(self.history['pareto_sizes']) >= 5:
            recent_sizes = self.history['pareto_sizes'][-5:]
            if len(set(recent_sizes)) == 1:  # Same size for last 5 generations
                logger.info("Pareto front stabilized")
                return True
        
        return False
    
    def _selection_and_breeding(self) -> None:
        """Perform selection and breeding"""
        logger.info("Performing selection and breeding...")
        
        # Parameters
        population_size = self.config.get('population_size', 100)
        elitism = self.config.get('elitism', 5)
        crossover_rate = self.config.get('crossover_rate', 0.9)
        mutation_rate = self.config.get('mutation_rate', 0.3)
        tournament_size = self.config.get('tournament_size', 7)
        
        new_population = []
        
        # Elitism: keep best individuals from Pareto front
        elite_individuals = self._select_elites(elitism)
        new_population.extend(elite_individuals)
        
        # Generate rest of population through selection and breeding
        while len(new_population) < population_size:
            # Tournament selection for parents
            parent1 = self._tournament_selection(tournament_size)
            parent2 = self._tournament_selection(tournament_size)
            
            # Crossover
            if random.random() < crossover_rate:
                child1, child2 = self.factory.create_crossover(parent1, parent2, crossover_rate)
            else:
                child1, child2 = parent1.copy(), parent2.copy()
            
            # Mutation
            if random.random() < mutation_rate:
                child1 = self.factory.create_mutated(child1, mutation_rate)
            if random.random() < mutation_rate:
                child2 = self.factory.create_mutated(child2, mutation_rate)
            
            # Add to new population (may exceed, will trim)
            new_population.extend([child1, child2])
        
        # Trim to population size
        self.population = new_population[:population_size]
        
        # Update metadata
        for individual in self.population:
            individual.metadata['generation'] = self.generation
    
    def _select_elites(self, n_elites: int) -> List[Individual]:
        """Select elite individuals from Pareto front"""
        if not self.pareto_front.front:
            return []
        
        # Select diverse elites from Pareto front
        elites = []
        
        # First, get individuals with highest accuracy
        sorted_by_accuracy = sorted(
            self.pareto_front.front,
            key=lambda ind: ind.fitness.get('accuracy', 0.0),
            reverse=True
        )
        
        # Add top by accuracy
        elites.extend(sorted_by_accuracy[:min(n_elites // 2, len(sorted_by_accuracy))])
        
        # Add diverse individuals (different pipeline structures)
        remaining_slots = n_elites - len(elites)
        if remaining_slots > 0:
            # Get unique pipeline structures
            seen_hashes = set(ind.get_hash() for ind in elites)
            diverse_individuals = []
            
            for ind in self.pareto_front.front:
                if ind.get_hash() not in seen_hashes:
                    diverse_individuals.append(ind)
                    seen_hashes.add(ind.get_hash())
                
                if len(diverse_individuals) >= remaining_slots:
                    break
            
            elites.extend(diverse_individuals)
        
        return elites
    
    def _tournament_selection(self, tournament_size: int) -> Individual:
        """Tournament selection"""
        # Randomly select tournament participants
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        
        # Filter valid individuals
        valid_tournament = [ind for ind in tournament if ind.fitness.get('valid', False)]
        
        if not valid_tournament:
            # Fallback: random selection
            return random.choice(self.population)
        
        # Select winner based on weighted fitness
        def weighted_fitness(ind: Individual) -> float:
            weights = self.config.get('weights', [0.6, 0.3, 0.1])
            objectives = self.config.get('objectives', ['accuracy', 'efficiency', 'complexity'])
            
            total = 0.0
            for obj, weight in zip(objectives, weights):
                total += weight * ind.fitness.get(obj, 0.0)
            
            return total
        
        return max(valid_tournament, key=weighted_fitness)
    
    def _get_best_individual(self) -> Optional[Individual]:
        """Get best individual from Pareto front"""
        if not self.pareto_front.front:
            # Fallback: best from population
            valid_individuals = [ind for ind in self.population if ind.fitness.get('valid', False)]
            if not valid_individuals:
                return None
            
            return max(valid_individuals, key=lambda ind: ind.fitness.get('accuracy', 0.0))
        
        # Select from Pareto front based on weighted fitness
        weights = self.config.get('weights', [0.6, 0.3, 0.1])
        objectives = self.config.get('objectives', ['accuracy', 'efficiency', 'complexity'])
        
        def weighted_fitness(ind: Individual) -> float:
            total = 0.0
            for obj, weight in zip(objectives, weights):
                total += weight * ind.fitness.get(obj, 0.0)
            return total
        
        return max(self.pareto_front.front, key=weighted_fitness)
    
    def _save_checkpoint(self, generation: int) -> None:
        """Save checkpoint"""
        import os
        import json
        
        checkpoint_dir = os.path.join(self.results_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_data = {
            'generation': generation,
            'population': [ind.to_dict() for ind in self.population],
            'pareto_front': [ind.to_dict() for ind in self.pareto_front.front],
            'history': self.history,
            'config': self.config
        }
        
        checkpoint_file = os.path.join(checkpoint_dir, f'generation_{generation}.json')
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_file}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint"""
        import json
        
        with open(checkpoint_path, 'r') as f:
            checkpoint_data = json.load(f)
        
        # Load population
        self.population = []
        for ind_data in checkpoint_data['population']:
            individual = Individual.from_dict(ind_data)
            self.population.append(individual)
        
        # Load Pareto front
        self.pareto_front.front = []
        for ind_data in checkpoint_data['pareto_front']:
            individual = Individual.from_dict(ind_data)
            self.pareto_front.front.append(individual)
        
        # Load history
        self.history = checkpoint_data['history']
        
        # Update generation
        self.generation = checkpoint_data['generation']
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        logger.info(f"Generation: {self.generation}, Population size: {len(self.population)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get GP statistics"""
        valid_individuals = [ind for ind in self.population if ind.fitness.get('valid', False)]
        
        stats = {
            'generation': self.generation,
            'population_size': len(self.population),
            'valid_count': len(valid_individuals),
            'pareto_front_size': len(self.pareto_front.front),
            'fitness_cache_size': len(self.fitness_cache),
            'history_length': len(self.history['generations'])
        }
        
        if valid_individuals:
            accuracies = [ind.fitness.get('accuracy', 0.0) for ind in valid_individuals]
            stats.update({
                'accuracy_mean': np.mean(accuracies),
                'accuracy_std': np.std(accuracies),
                'accuracy_max': np.max(accuracies),
                'accuracy_min': np.min(accuracies)
            })
        
        return stats
