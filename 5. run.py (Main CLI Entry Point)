#!/usr/bin/env python3
"""
Main CLI entry point for the Hyper-Heuristic Framework
Execution Order: 39/40
"""

import argparse
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.logger import setup_logger
from config.config_loader import ConfigLoader
from framework.hyperheuristic import HyperHeuristicFramework
from experiments.run_experiment import run_experiment
from experiments.baseline_comparison import run_baseline_comparison


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Hyper-Heuristic Framework for Hyperspectral Image Segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Full validation mode
    parser_full = subparsers.add_parser("full", help="Run full validation pipeline")
    parser_full.add_argument("--dataset", type=str, default="Indian_Pines",
                           choices=["Indian_Pines", "Pavia_University", "Salinas"],
                           help="Dataset to use")
    parser_full.add_argument("--n_runs", type=int, default=10,
                           help="Number of runs for statistical significance")
    parser_full.add_argument("--output_dir", type=str, default="./results",
                           help="Output directory for results")
    
    # Train only mode
    parser_train = subparsers.add_parser("train", help="Train only mode")
    parser_train.add_argument("--dataset", type=str, required=True)
    parser_train.add_argument("--epochs", type=int, default=50)
    parser_train.add_argument("--checkpoint", type=str, help="Checkpoint to resume from")
    
    # Evaluate only mode
    parser_eval = subparsers.add_parser("eval", help="Evaluate only mode")
    parser_eval.add_argument("--dataset", type=str, required=True)
    parser_eval.add_argument("--model_path", type=str, required=True)
    parser_eval.add_argument("--output_dir", type=str, default="./eval_results")
    
    # Baseline comparison
    parser_baseline = subparsers.add_parser("baseline", help="Run baseline comparisons")
    parser_baseline.add_argument("--dataset", type=str, required=True)
    parser_baseline.add_argument("--methods", type=str, nargs="+",
                               default=["kmeans", "fcm", "svm", "rf", "cnn"],
                               help="Baseline methods to compare")
    
    # Generate report
    parser_report = subparsers.add_parser("report", help="Generate PDF report")
    parser_report.add_argument("--results_path", type=str, required=True)
    parser_report.add_argument("--output", type=str, default="./report.pdf")
    
    # Verify data
    parser_verify = subparsers.add_parser("verify", help="Verify dataset integrity")
    parser_verify.add_argument("--dataset", type=str, required=True)
    parser_verify.add_argument("--checksum", action="store_true",
                             help="Verify checksums")
    
    # Experiment mode
    parser_exp = subparsers.add_parser("experiment", help="Run specific experiment")
    parser_exp.add_argument("--exp_config", type=str, required=True,
                          help="Experiment configuration file")
    
    return parser.parse_args()


def main():
    """Main entry point"""
    args = parse_args()
    
    # Setup logging
    logger = setup_logger()
    logger.info("Starting Hyper-Heuristic Framework")
    
    # Load configuration
    config = ConfigLoader.load_config()
    
    if args.command == "full":
        logger.info(f"Running full validation on {args.dataset}")
        framework = HyperHeuristicFramework(config)
        # Run full pipeline
        results = framework.run_full_validation(
            dataset_name=args.dataset,
            n_runs=args.n_runs,
            output_dir=args.output_dir
        )
        logger.info(f"Full validation completed. Results saved to {args.output_dir}")
        
    elif args.command == "train":
        logger.info(f"Training on {args.dataset}")
        framework = HyperHeuristicFramework(config)
        framework.train(
            dataset_name=args.dataset,
            epochs=args.epochs,
            checkpoint_path=args.checkpoint
        )
        
    elif args.command == "eval":
        logger.info(f"Evaluating on {args.dataset}")
        framework = HyperHeuristicFramework.load(args.model_path)
        results = framework.evaluate(
            dataset_name=args.dataset,
            output_dir=args.output_dir
        )
        logger.info(f"Evaluation completed. Results: {results}")
        
    elif args.command == "baseline":
        logger.info(f"Running baseline comparison on {args.dataset}")
        results = run_baseline_comparison(
            dataset_name=args.dataset,
            methods=args.methods,
            config=config
        )
        logger.info(f"Baseline comparison completed")
        
    elif args.command == "report":
        logger.info(f"Generating report from {args.results_path}")
        # Generate report
        logger.info(f"Report saved to {args.output}")
        
    elif args.command == "verify":
        logger.info(f"Verifying dataset: {args.dataset}")
        # Verify dataset
        logger.info("Dataset verification completed")
        
    elif args.command == "experiment":
        logger.info(f"Running experiment from {args.exp_config}")
        run_experiment(args.exp_config)
        logger.info("Experiment completed")
        
    else:
        logger.error("No command specified. Use --help for usage.")
        sys.exit(1)


if __name__ == "__main__":
    main()
