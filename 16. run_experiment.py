"""
Experiment runner module
Execution Order: 40
"""

import yaml
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Any
import time
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config_loader import ConfigLoader
from framework.hyperheuristic import HyperHeuristicFramework
from utils.reproducibility import ReproducibilityManager
from utils.visualization import ExperimentVisualizer

logger = logging.getLogger(__name__)


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """
    Load experiment configuration
    
    Args:
        config_path: Path to experiment config file
        
    Returns:
        Experiment configuration
    """
    config_path = Path(config_path)
    
    if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    return config


def run_experiment(experiment_config_path: str) -> Dict[str, Any]:
    """
    Run a complete experiment
    
    Args:
        experiment_config_path: Path to experiment configuration
        
    Returns:
        Experiment results
    """
    logger.info(f"Starting experiment from {experiment_config_path}")
    
    # Load experiment configuration
    exp_config = load_experiment_config(experiment_config_path)
    
    # Extract experiment parameters
    experiment_name = exp_config.get("name", "unnamed_experiment")
    datasets = exp_config.get("datasets", ["Indian_Pines"])
    n_runs = exp_config.get("n_runs", 10)
    output_dir = Path(exp_config.get("output_dir", f"./results/{experiment_name}"))
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save experiment configuration
    with open(output_dir / "experiment_config.json", 'w') as f:
        json.dump(exp_config, f, indent=2)
    
    # Initialize results structure
    experiment_results = {
        "name": experiment_name,
        "config": exp_config,
        "datasets": {},
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Run experiment for each dataset
    for dataset_name in datasets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running experiment on dataset: {dataset_name}")
        logger.info(f"{'='*60}")
        
        dataset_results = run_dataset_experiment(
            dataset_name=dataset_name,
            exp_config=exp_config,
            output_dir=output_dir / dataset_name
        )
        
        experiment_results["datasets"][dataset_name] = dataset_results
        
        logger.info(f"Completed experiment for {dataset_name}")
    
    # Generate experiment summary
    experiment_results["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    experiment_results["summary"] = generate_experiment_summary(experiment_results)
    
    # Save final results
    results_file = output_dir / "experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump(experiment_results, f, indent=2)
    
    # Generate visualizations
    generate_experiment_visualizations(experiment_results, output_dir)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment '{experiment_name}' completed")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*60}")
    
    return experiment_results


def run_dataset_experiment(dataset_name: str, exp_config: Dict[str, Any],
                          output_dir: Path) -> Dict[str, Any]:
    """
    Run experiment for a single dataset
    
    Args:
        dataset_name: Dataset name
        exp_config: Experiment configuration
        output_dir: Output directory
        
    Returns:
        Dataset experiment results
    """
    # Create dataset output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load framework configuration
    framework_config = ConfigLoader.load_config()
    
    # Update configuration with experiment parameters
    update_config_with_experiment(framework_config, exp_config)
    
    # Initialize framework
    framework = HyperHeuristicFramework(framework_config)
    
    # Run different experiment types
    experiment_type = exp_config.get("type", "full_validation")
    
    if experiment_type == "full_validation":
        results = framework.run_full_validation(
            dataset_name=dataset_name,
            n_runs=exp_config.get("n_runs", 10),
            output_dir=str(output_dir)
        )
    
    elif experiment_type == "ablation_study":
        results = run_ablation_study(
            framework=framework,
            dataset_name=dataset_name,
            ablation_config=exp_config.get("ablation", {}),
            output_dir=output_dir
        )
    
    elif experiment_type == "hyperparameter_search":
        results = run_hyperparameter_search(
            framework=framework,
            dataset_name=dataset_name,
            search_config=exp_config.get("search", {}),
            output_dir=output_dir
        )
    
    elif experiment_type == "scalability_test":
        results = run_scalability_test(
            framework=framework,
            dataset_name=dataset_name,
            scalability_config=exp_config.get("scalability", {}),
            output_dir=output_dir
        )
    
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    return results


def update_config_with_experiment(framework_config: Any, exp_config: Dict[str, Any]) -> None:
    """Update framework configuration with experiment parameters"""
    # Update GP parameters
    if "gp" in exp_config:
        for key, value in exp_config["gp"].items():
            if hasattr(framework_config.gp, key):
                setattr(framework_config.gp, key, value)
    
    # Update PSO parameters
    if "pso" in exp_config:
        for key, value in exp_config["pso"].items():
            if hasattr(framework_config.pso, key):
                setattr(framework_config.pso, key, value)
    
    # Update policy parameters
    if "policy" in exp_config:
        for key, value in exp_config["policy"].items():
            if hasattr(framework_config.policy, key):
                setattr(framework_config.policy, key, value)
    
    # Update framework parameters
    if "framework" in exp_config:
        for key, value in exp_config["framework"].items():
            if hasattr(framework_config.framework, key):
                setattr(framework_config.framework, key, value)


def run_ablation_study(framework: HyperHeuristicFramework, dataset_name: str,
                      ablation_config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Run ablation study
    
    Args:
        framework: Hyper-heuristic framework
        dataset_name: Dataset name
        ablation_config: Ablation study configuration
        output_dir: Output directory
        
    Returns:
        Ablation study results
    """
    logger.info("Running ablation study")
    
    ablation_results = {
        "dataset": dataset_name,
        "ablation_config": ablation_config,
        "components": {}
    }
    
    # Get components to ablate
    components = ablation_config.get("components", ["gp", "policy", "meta_features"])
    
    for component in components:
        logger.info(f"Ablating component: {component}")
        
        # Create modified framework for ablation
        modified_framework = create_ablated_framework(framework, component)
        
        # Train and evaluate
        training_stats = modified_framework.train(dataset_name, mode="full")
        eval_results = modified_framework.evaluate(dataset_name)
        
        ablation_results["components"][component] = {
            "training": training_stats,
            "evaluation": eval_results,
            "ablated": True
        }
    
    # Baseline (no ablation)
    logger.info("Running baseline (no ablation)")
    baseline_training = framework.train(dataset_name, mode="full")
    baseline_eval = framework.evaluate(dataset_name)
    
    ablation_results["baseline"] = {
        "training": baseline_training,
        "evaluation": baseline_eval,
        "ablated": False
    }
    
    # Save results
    results_file = output_dir / "ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(ablation_results, f, indent=2)
    
    return ablation_results


def create_ablated_framework(framework: HyperHeuristicFramework, 
                           component: str) -> HyperHeuristicFramework:
    """
    Create framework with specific component ablated
    
    Args:
        framework: Original framework
        component: Component to ablate
        
    Returns:
        Modified framework
    """
    # Create copy of configuration
    import copy
    modified_config = copy.deepcopy(framework.config)
    
    # Modify configuration based on component
    if component == "gp":
        # Disable GP evolution
        modified_config.gp.generations = 0
        modified_config.gp.population_size = 0
    
    elif component == "policy":
        # Use random policy instead of learned policy
        # This would require modifying the framework initialization
        pass
    
    elif component == "meta_features":
        # Disable meta-feature conditioning
        modified_config.policy.hidden_dims = [1]  # Minimal network
    
    elif component == "spatial":
        # Disable spatial regularization
        modified_config.pso.spatial_weight = 0.0
    
    elif component == "spectral":
        # Use only spatial features
        # This would require modifying the data preprocessing
        pass
    
    # Create new framework with modified config
    modified_framework = HyperHeuristicFramework(modified_config)
    
    return modified_framework


def run_hyperparameter_search(framework: HyperHeuristicFramework, dataset_name: str,
                            search_config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Run hyperparameter search
    
    Args:
        framework: Hyper-heuristic framework
        dataset_name: Dataset name
        search_config: Search configuration
        output_dir: Output directory
        
    Returns:
        Hyperparameter search results
    """
    logger.info("Running hyperparameter search")
    
    search_results = {
        "dataset": dataset_name,
        "search_config": search_config,
        "trials": []
    }
    
    # Get search space
    search_space = search_config.get("search_space", {})
    n_trials = search_config.get("n_trials", 50)
    search_method = search_config.get("method", "random")
    
    # Import optimization library
    try:
        import optuna
    except ImportError:
        logger.error("Optuna not installed. Please install with: pip install optuna")
        return search_results
    
    # Define objective function
    def objective(trial):
        # Suggest hyperparameters
        suggested_params = {}
        
        for param, config in search_space.items():
            if config["type"] == "categorical":
                value = trial.suggest_categorical(param, config["values"])
            elif config["type"] == "float":
                value = trial.suggest_float(param, config["low"], config["high"], log=config.get("log", False))
            elif config["type"] == "int":
                value = trial.suggest_int(param, config["low"], config["high"])
            else:
                continue
            
            suggested_params[param] = value
        
        # Update framework with suggested parameters
        modified_framework = create_framework_with_params(framework, suggested_params)
        
        # Train and evaluate
        try:
            modified_framework.train(dataset_name, mode="quick")
            eval_results = modified_framework.evaluate(dataset_name, mode="quick")
            
            # Get objective value (negative mIoU for minimization)
            mIoU = eval_results.get("adaptive", {}).get("mean_mIoU", 0)
            return -mIoU  # Negative for minimization
            
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float('inf')  # Return worst possible value
    
    # Create study
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler() if search_method == "tpe" else optuna.samplers.RandomSampler()
    )
    
    # Run optimization
    study.optimize(objective, n_trials=n_trials)
    
    # Collect results
    for trial in study.trials:
        search_results["trials"].append({
            "number": trial.number,
            "params": trial.params,
            "value": trial.value,
            "datetime_start": trial.datetime_start.isoformat() if trial.datetime_start else None,
            "datetime_complete": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            "state": trial.state.name
        })
    
    # Best trial
    search_results["best_trial"] = {
        "number": study.best_trial.number,
        "params": study.best_trial.params,
        "value": study.best_trial.value
    }
    
    # Save results
    results_file = output_dir / "hyperparameter_search.json"
    with open(results_file, 'w') as f:
        json.dump(search_results, f, indent=2)
    
    # Visualize optimization history
    try:
        import optuna.visualization as vis
        
        # Optimization history plot
        fig = vis.plot_optimization_history(study)
        fig.write_html(str(output_dir / "optimization_history.html"))
        
        # Parameter importances
        fig = vis.plot_param_importances(study)
        fig.write_html(str(output_dir / "param_importances.html"))
        
    except Exception as e:
        logger.warning(f"Could not create visualizations: {e}")
    
    return search_results


def create_framework_with_params(framework: HyperHeuristicFramework,
                               params: Dict[str, Any]) -> HyperHeuristicFramework:
    """
    Create framework with specific parameters
    
    Args:
        framework: Original framework
        params: Parameters to update
        
    Returns:
        Modified framework
    """
    import copy
    
    # Create deep copy of configuration
    modified_config = copy.deepcopy(framework.config)
    
    # Update configuration with parameters
    for param_path, value in params.items():
        # Parse parameter path (e.g., "gp.population_size")
        parts = param_path.split('.')
        
        if len(parts) == 2:
            section, param = parts
            if hasattr(modified_config, section):
                section_obj = getattr(modified_config, section)
                if hasattr(section_obj, param):
                    setattr(section_obj, param, value)
    
    # Create new framework
    modified_framework = HyperHeuristicFramework(modified_config)
    
    return modified_framework


def run_scalability_test(framework: HyperHeuristicFramework, dataset_name: str,
                        scalability_config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Run scalability test
    
    Args:
        framework: Hyper-heuristic framework
        dataset_name: Dataset name
        scalability_config: Scalability test configuration
        output_dir: Output directory
        
    Returns:
        Scalability test results
    """
    logger.info("Running scalability test")
    
    scalability_results = {
        "dataset": dataset_name,
        "scalability_config": scalability_config,
        "tests": []
    }
    
    # Get test configurations
    image_sizes = scalability_config.get("image_sizes", [64, 128, 256, 512])
    n_clusters = scalability_config.get("n_clusters", [8, 16, 32, 64])
    
    # Load dataset
    from data.dataset_loader import DatasetManager
    dataset_manager = DatasetManager(framework.config.__dict__)
    dataset = dataset_manager.load_dataset(dataset_name, mode="test")
    
    # Get original image
    original_image, _ = dataset.get_full_image()
    
    for size in image_sizes:
        logger.info(f"Testing image size: {size}x{size}")
        
        # Resize image (or crop)
        if size < min(original_image.shape[0], original_image.shape[1]):
            test_image = original_image[:size, :size, :]
        else:
            # Pad image
            import cv2
            test_image = cv2.resize(original_image, (size, size))
        
        for n_cluster in n_clusters:
            logger.info(f"  Testing with {n_cluster} clusters")
            
            # Test different segmentation methods
            methods = ["adaptive", "evolved", "baseline"]
            
            for method in methods:
                test_start = time.time()
                
                try:
                    # Perform segmentation
                    if method == "adaptive":
                        result = framework.segmenter.segment_adaptive(
                            test_image, 
                            n_clusters=n_cluster
                        )
                    elif method == "evolved" and framework.best_pipeline:
                        result = framework.segmenter.segment_with_pipeline(
                            test_image,
                            framework.best_pipeline
                        )
                    else:
                        result = framework.segmenter.segment_baseline(
                            test_image,
                            n_clusters=n_cluster
                        )
                    
                    test_time = time.time() - test_start
                    
                    # Measure memory usage (approximate)
                    import psutil
                    process = psutil.Process()
                    memory_mb = process.memory_info().rss / (1024 * 1024)
                    
                    scalability_results["tests"].append({
                        "image_size": size,
                        "n_clusters": n_cluster,
                        "method": method,
                        "execution_time": test_time,
                        "memory_mb": memory_mb,
                        "success": True
                    })
                    
                except Exception as e:
                    scalability_results["tests"].append({
                        "image_size": size,
                        "n_clusters": n_cluster,
                        "method": method,
                        "error": str(e),
                        "success": False
                    })
    
    # Save results
    results_file = output_dir / "scalability_results.json"
    with open(results_file, 'w') as f:
        json.dump(scalability_results, f, indent=2)
    
    return scalability_results


def generate_experiment_summary(experiment_results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate experiment summary"""
    summary = {
        "total_datasets": len(experiment_results["datasets"]),
        "dataset_results": {}
    }
    
    for dataset_name, dataset_results in experiment_results["datasets"].items():
        # Extract key metrics
        if "evaluation" in dataset_results:
            eval_results = dataset_results["evaluation"]
            
            dataset_summary = {}
            for method, results in eval_results.items():
                if method != "statistical_comparison":
                    mIoU = results.get("mean_mIoU", 0)
                    exec_time = results.get("mean_execution_time", 0)
                    dataset_summary[method] = {
                        "mIoU": mIoU,
                        "execution_time": exec_time
                    }
            
            summary["dataset_results"][dataset_name] = dataset_summary
    
    return summary


def generate_experiment_visualizations(experiment_results: Dict[str, Any],
                                      output_dir: Path) -> None:
    """Generate experiment visualizations"""
    visualizer = ExperimentVisualizer()
    
    # Create visualizations directory
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # 1. Performance comparison across datasets
    visualizer.plot_performance_comparison(
        experiment_results,
        output_path=viz_dir / "performance_comparison.png"
    )
    
    # 2. Training time analysis
    visualizer.plot_training_time_analysis(
        experiment_results,
        output_path=viz_dir / "training_time_analysis.png"
    )
    
    # 3. Statistical significance visualization
    visualizer.plot_statistical_significance(
        experiment_results,
        output_path=viz_dir / "statistical_significance.png"
    )
    
    # 4. Generate HTML report
    visualizer.generate_html_report(
        experiment_results,
        output_path=viz_dir / "experiment_report.html"
    )
    
    logger.info(f"Visualizations saved to {viz_dir}")
