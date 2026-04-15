"""
Main CLI entry point for HSI Hyper-Heuristic Framework
"""
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='HSI Hyper-Heuristic Segmentation Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='full_validation',
        choices=['full_validation', 'train_only', 'evaluate_only',
                 'ablation_study', 'hyperparameter_search',
                 'scalability_test', 'generate_report', 'verify_data'],
        help='Execution mode'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='Indian_Pines',
        choices=['Indian_Pines', 'Pavia_University', 'Salinas', 'Houston', 'Botswana'],
        help='Dataset to use'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='4. config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--n_runs',
        type=int,
        default=10,
        help='Number of runs for statistical significance'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--results_path',
        type=str,
        default=None,
        help='Path to results file (for generate_report mode)'
    )
    parser.add_argument(
        '--experiment_config',
        type=str,
        default=None,
        help='Path to experiment configuration file'
    )

    args = parser.parse_args()

    logger.info("Starting HSI Hyper-Heuristic Framework")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Dataset: {args.dataset}")

    try:
        if args.mode == 'full_validation':
            from config.config_loader import ConfigLoader
            from framework.hyperheuristic import HyperHeuristicFramework

            framework_config = ConfigLoader.load_config(args.config)
            framework = HyperHeuristicFramework(framework_config)
            framework.run_full_validation(
                dataset_name=args.dataset,
                n_runs=args.n_runs,
                output_dir=args.output_dir
            )
            logger.info(f"Full validation completed. Results saved to {args.output_dir}")

        elif args.mode in ['ablation_study', 'hyperparameter_search',
                           'scalability_test']:
            if args.experiment_config is None:
                logger.error("--experiment_config is required for this mode")
                sys.exit(1)
            from run_experiment import run_experiment
            run_experiment(args.experiment_config)
            logger.info("Experiment completed.")

        elif args.mode == 'verify_data':
            from utils.reproducibility import ReproducibilityManager
            import json
            with open(args.config) as f:
                cfg = json.load(f)
            repro = ReproducibilityManager(cfg.get('framework', {}), args.output_dir)
            data_dir = Path(cfg.get('paths', {}).get('data_dir', './data'))
            for mat_file in data_dir.rglob('*.mat'):
                result = repro.verify_dataset(str(mat_file))
                status = '✓' if result['valid_mat_file'] else '✗'
                logger.info(f"{status} {mat_file}: checksum={result.get('checksum', 'N/A')}")

        elif args.mode == 'generate_report':
            if args.results_path is None:
                logger.error("--results_path is required for generate_report mode")
                sys.exit(1)
            import json
            from utils.visualization import generate_pdf_report
            with open(args.results_path) as f:
                results = json.load(f)
            report_path = Path(args.output_dir) / 'validation_report.pdf'
            generate_pdf_report(results=results, output_path=report_path, config={})
            logger.info(f"Report generated at {report_path}")

        else:
            logger.error(f"Mode '{args.mode}' is not yet implemented as a standalone command.")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
