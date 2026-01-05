"""
Main hyper-heuristic framework module
Execution Order: 29
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import time
import json

from config.config_loader import Config
from data.dataset_loader import DatasetManager
from llhs.base import LLHRegistry
from gp.evolution import GeneticProgramming
from policy.network import PolicyNetwork
from policy.trainer import PolicyTrainer
from utils.reproducibility import ReproducibilityManager
from utils.metrics import SegmentationMetrics
from framework.segmenter import AdaptiveSegmenter
from framework.evaluator import FrameworkEvaluator

logger = logging.getLogger(__name__)


class HyperHeuristicFramework:
    """
    Main hyper-heuristic framework
    
    Integrates:
    1. Grammar-guided genetic programming
    2. Policy network for LLH selection
    3. Adaptive segmentation engine
    4. Statistical evaluation
    """
    
    def __init__(self, config: Config):
        """
        Initialize framework
        
        Args:
            config: Framework configuration
        """
        self.config = config
        self.repro_manager = ReproducibilityManager(
            config.framework.__dict__,
            results_dir=config.paths.results_dir
        )
        
        # Initialize components
        self._initialize_components()
        
        # State
        self.is_trained = False
        self.best_pipeline = None
        self.training_history = {}
        
        logger.info("Hyper-Heuristic Framework initialized")
    
    def _initialize_components(self) -> None:
        """Initialize framework components"""
        # Dataset manager
        self.dataset_manager = DatasetManager(self.config.__dict__)
        
        # LLH registry (auto-registered via decorators)
        self.llh_registry = LLHRegistry
        
        # Genetic Programming
        self.gp = GeneticProgramming(
            config=self.config.gp.__dict__,
            grammar=None,  # Will be initialized in GP
            results_dir=self.config.paths.results_dir
        )
        
        # Policy Network
        self.policy_network = PolicyNetwork(
            input_dim=12,  # Meta-feature dimension
            hidden_dims=self.config.policy.hidden_dims,
            output_dim=len(self.llh_registry.list_available()),
            dropout=self.config.policy.dropout
        )
        
        # Policy Trainer
        self.policy_trainer = PolicyTrainer(
            network=self.policy_network,
            config=self.config.policy.__dict__
        )
        
        # Adaptive Segmenter
        self.segmenter = AdaptiveSegmenter(
            llh_registry=self.llh_registry,
            policy_network=self.policy_network,
            config=self.config.__dict__
        )
        
        # Evaluator
        self.evaluator = FrameworkEvaluator(
            config=self.config.evaluation.__dict__,
            results_dir=self.config.paths.results_dir
        )
    
    def train(self, dataset_name: str, mode: str = "full") -> Dict[str, Any]:
        """
        Train the framework
        
        Args:
            dataset_name: Name of dataset to train on
            mode: Training mode ("full", "gp_only", "policy_only")
            
        Returns:
            Training statistics
        """
        logger.info(f"Starting training on {dataset_name} in {mode} mode")
        training_start = time.time()
        
        # Load dataset
        train_dataset = self.dataset_manager.load_dataset(dataset_name, mode="train")
        val_dataset = self.dataset_manager.load_dataset(dataset_name, mode="val")
        
        training_stats = {}
        
        # Step 1: GP Evolution (if needed)
        if mode in ["full", "gp_only"]:
            logger.info("Step 1: Grammar-guided GP evolution")
            gp_start = time.time()
            
            self.best_pipeline = self.gp.evolve(
                dataset=train_dataset,
                validation_dataset=val_dataset,
                n_generations=self.config.gp.generations
            )
            
            gp_time = time.time() - gp_start
            training_stats["gp_time"] = gp_time
            training_stats["best_pipeline"] = str(self.best_pipeline)
            
            logger.info(f"GP evolution completed in {gp_time:.2f}s")
        
        # Step 2: Policy Network Training (if needed)
        if mode in ["full", "policy_only"]:
            logger.info("Step 2: Policy network training")
            policy_start = time.time()
            
            policy_stats = self.policy_trainer.train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                n_epochs=self.config.policy.epochs
            )
            
            policy_time = time.time() - policy_start
            training_stats["policy_time"] = policy_time
            training_stats["policy_stats"] = policy_stats
            
            logger.info(f"Policy training completed in {policy_time:.2f}s")
        
        # Update state
        self.is_trained = True
        
        # Calculate total training time
        total_time = time.time() - training_start
        training_stats["total_time"] = total_time
        
        # Measure energy usage
        energy_info = self.repro_manager.get_energy_usage(total_time)
        training_stats["energy_usage"] = energy_info
        
        # Save training state
        self._save_training_state(dataset_name)
        
        logger.info(f"Training completed in {total_time:.2f}s")
        logger.info(f"Energy used: {energy_info.get('energy_kj', 0):.2f} kJ")
        
        return training_stats
    
    def segment(self, image: np.ndarray, mode: str = "adaptive", 
               **kwargs) -> Dict[str, Any]:
        """
        Segment a hyperspectral image
        
        Args:
            image: Input image [H, W, B]
            mode: Segmentation mode ("adaptive", "evolved", "baseline")
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with segmentation results
        """
        if not self.is_trained and mode == "adaptive":
            raise ValueError("Framework not trained. Use mode='baseline' or train first.")
        
        segmentation_start = time.time()
        
        # Perform segmentation
        if mode == "adaptive":
            result = self.segmenter.segment_adaptive(image, **kwargs)
        elif mode == "evolved" and self.best_pipeline:
            result = self.segmenter.segment_with_pipeline(image, self.best_pipeline)
        else:
            result = self.segmenter.segment_baseline(image)
        
        segmentation_time = time.time() - segmentation_start
        
        # Create result dictionary
        segmentation_result = {
            "segmentation": result,
            "execution_time": segmentation_time,
            "mode": mode,
            "image_shape": image.shape
        }
        
        # Add energy measurement if configured
        if self.config.framework.measure_energy:
            energy_info = self.repro_manager.get_energy_usage(segmentation_time)
            segmentation_result["energy_usage"] = energy_info
        
        # Log segmentation
        self.repro_manager.log_execution_step(
            step=f"segmentation_{mode}",
            details={
                "image_shape": image.shape,
                "execution_time": segmentation_time,
                "mode": mode
            }
        )
        
        return segmentation_result
    
    def evaluate(self, dataset_name: str, mode: str = "full",
                methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate framework performance
        
        Args:
            dataset_name: Dataset to evaluate on
            mode: Evaluation mode ("full", "quick")
            methods: List of methods to evaluate
            
        Returns:
            Evaluation results
        """
        logger.info(f"Evaluating on {dataset_name}")
        
        # Load test dataset
        test_dataset = self.dataset_manager.load_dataset(dataset_name, mode="test")
        
        # Default methods to evaluate
        if methods is None:
            methods = ["adaptive", "evolved", "baseline"]
        
        evaluation_results = {}
        
        for method in methods:
            logger.info(f"Evaluating {method} method...")
            
            method_results = self.evaluator.evaluate_method(
                method=method,
                dataset=test_dataset,
                segmenter=self.segmenter if method != "baseline" else None,
                best_pipeline=self.best_pipeline if method == "evolved" else None,
                n_runs=self.config.framework.n_runs if mode == "full" else 1
            )
            
            evaluation_results[method] = method_results
            
            logger.info(f"{method}: mIoU = {method_results.get('mean_mIoU', 0):.4f}")
        
        # Statistical comparison
        if "adaptive" in evaluation_results and "baseline" in evaluation_results:
            comparison = self.evaluator.compare_methods(
                method1_results=evaluation_results["adaptive"],
                method2_results=evaluation_results["baseline"],
                method1_name="adaptive",
                method2_name="baseline"
            )
            
            evaluation_results["statistical_comparison"] = comparison
            
            logger.info(f"Statistical comparison: p-value = {comparison.get('p_value', 1):.6f}")
            if comparison.get('significant', False):
                logger.info("Adaptive method is significantly better than baseline")
        
        # Save evaluation results
        self.repro_manager.save_with_provenance(
            evaluation_results,
            f"evaluation_results_{dataset_name}.json",
            metadata={
                "dataset": dataset_name,
                "mode": mode,
                "methods": methods
            }
        )
        
        return evaluation_results
    
    def run_full_validation(self, dataset_name: str, n_runs: int = 10,
                          output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run full validation pipeline
        
        Args:
            dataset_name: Dataset name
            n_runs: Number of runs for statistical significance
            output_dir: Output directory
            
        Returns:
            Validation results
        """
        if output_dir:
            self.config.paths.results_dir = output_dir
            self.repro_manager.results_dir = Path(output_dir)
        
        logger.info(f"Starting full validation on {dataset_name}")
        logger.info(f"Number of runs: {n_runs}")
        
        full_results = {
            "dataset": dataset_name,
            "n_runs": n_runs,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "config": self.config.framework.__dict__
        }
        
        # Phase 1: Training
        logger.info("Phase 1: Training")
        training_results = self.train(dataset_name, mode="full")
        full_results["training"] = training_results
        
        # Phase 2: Evaluation
        logger.info("Phase 2: Evaluation")
        evaluation_results = self.evaluate(dataset_name, mode="full")
        full_results["evaluation"] = evaluation_results
        
        # Phase 3: Cross-validation (if multiple runs)
        if n_runs > 1:
            logger.info("Phase 3: Cross-validation")
            cv_results = self._cross_validation(dataset_name, n_runs=n_runs)
            full_results["cross_validation"] = cv_results
        
        # Phase 4: Baseline comparison
        logger.info("Phase 4: Baseline comparison")
        baseline_results = self._compare_with_baselines(dataset_name)
        full_results["baseline_comparison"] = baseline_results
        
        # Phase 5: Generate report
        logger.info("Phase 5: Generating report")
        report_path = self._generate_report(full_results, output_dir)
        full_results["report_path"] = report_path
        
        # Save final results
        final_results_path = self.repro_manager.save_with_provenance(
            full_results,
            f"full_validation_results_{dataset_name}.json"
        )
        
        logger.info(f"Full validation completed. Results saved to {final_results_path}")
        
        return full_results
    
    def _cross_validation(self, dataset_name: str, n_runs: int = 10) -> Dict[str, Any]:
        """Run cross-validation"""
        cv_results = {
            "runs": [],
            "mean_scores": {},
            "std_scores": {}
        }
        
        for run in range(n_runs):
            logger.info(f"Cross-validation run {run + 1}/{n_runs}")
            
            # Set different seed for each run
            self.config.framework.random_seed = 42 + run
            self.repro_manager._set_global_seeds()
            
            # Train and evaluate
            self.train(dataset_name, mode="full")
            eval_results = self.evaluate(dataset_name, mode="quick")
            
            cv_results["runs"].append({
                "run": run,
                "seed": self.config.framework.random_seed,
                "evaluation": eval_results
            })
        
        # Calculate statistics
        all_mious = []
        for run in cv_results["runs"]:
            if "adaptive" in run["evaluation"]:
                mIoU = run["evaluation"]["adaptive"].get("mean_mIoU", 0)
                all_mious.append(mIoU)
        
        if all_mious:
            cv_results["mean_scores"]["mIoU"] = np.mean(all_mious)
            cv_results["std_scores"]["mIoU"] = np.std(all_mious)
            cv_results["ci_95_mIoU"] = stats.t.interval(
                0.95, len(all_mious)-1, 
                loc=np.mean(all_mious), 
                scale=stats.sem(all_mious)
            )
        
        return cv_results
    
    def _compare_with_baselines(self, dataset_name: str) -> Dict[str, Any]:
        """Compare with baseline methods"""
        from experiments.baseline_comparison import run_baseline_comparison
        
        baseline_results = run_baseline_comparison(
            dataset_name=dataset_name,
            config=self.config.__dict__,
            results_dir=self.config.paths.results_dir
        )
        
        return baseline_results
    
    def _generate_report(self, results: Dict[str, Any], 
                        output_dir: Optional[str] = None) -> str:
        """Generate PDF report"""
        from utils.visualization import generate_pdf_report
        
        if output_dir is None:
            output_dir = self.config.paths.results_dir
        
        report_path = Path(output_dir) / "validation_report.pdf"
        
        generate_pdf_report(
            results=results,
            output_path=report_path,
            config=self.config.__dict__
        )
        
        return str(report_path)
    
    def _save_training_state(self, dataset_name: str) -> None:
        """Save framework training state"""
        state = {
            "config": self.config.__dict__,
            "is_trained": self.is_trained,
            "best_pipeline": self.best_pipeline.to_dict() if self.best_pipeline else None,
            "policy_state": self.policy_network.state_dict() if self.is_trained else None,
            "training_history": self.training_history,
            "dataset": dataset_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        checkpoint_path = self.repro_manager.create_checkpoint(
            f"framework_state_{dataset_name}",
            state
        )
        
        logger.info(f"Framework state saved to {checkpoint_path}")
    
    def load(self, checkpoint_path: str) -> None:
        """
        Load framework from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        state = self.repro_manager.load_checkpoint(checkpoint_path)
        
        # Update configuration
        config_dict = state["config"]
        self.config = Config.from_dict(config_dict)
        
        # Update state
        self.is_trained = state["is_trained"]
        
        # Load best pipeline
        if state["best_pipeline"]:
            from gp.grammar import Grammar
            grammar = Grammar()
            self.best_pipeline = grammar.pipeline_from_dict(state["best_pipeline"])
        
        # Load policy network
        if state["policy_state"]:
            self.policy_network.load_state_dict(state["policy_state"])
        
        # Load training history
        self.training_history = state.get("training_history", {})
        
        logger.info(f"Framework loaded from {checkpoint_path}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get framework statistics"""
        return {
            "is_trained": self.is_trained,
            "num_llhs": len(self.llh_registry.list_available()),
            "best_pipeline": str(self.best_pipeline) if self.best_pipeline else None,
            "training_history_keys": list(self.training_history.keys()),
            "config_summary": {
                "dataset": self.training_history.get("dataset"),
                "training_time": self.training_history.get("total_time"),
                "energy_used": self.training_history.get("energy_usage", {}).get("energy_kj")
            }
        }
