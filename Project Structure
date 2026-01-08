hsi-hyperheuristic/
├── README.md
├── requirements.txt
├── setup.py
├── Dockerfile
├── config.json
├── run.py                    # Main CLI entry point
├── data/                     # Datasets (auto-downloaded)
├── checkpoints/              # Model checkpoints (e.g., gradient_v6.2.pt)
├── src/
│   ├── __init__.py
│   ├── config/               # Configuration management
│   │   ├── __init__.py
│   │   ├── config_loader.py
│   │   ├── constants.py
│   │   └── hyperparameters.py
│   ├── data/                 # Data handling
│   │   ├── __init__.py
│   │   ├── dataset_loader.py
│   │   ├── preprocessing.py
│   │   ├── augmentations.py
│   │   └── meta_features.py
│   ├── llhs/                 # Low-Level Heuristics
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── sspso.py
│   │   ├── gradient.py          # HolisticGradientOperator
│   │   ├── clustering.py
│   │   ├── watershed.py
│   │   ├── mrf.py
│   │   ├── cnn_refine.py
│   │   └── crf.py
│   ├── gp/                   # Genetic Programming
│   │   ├── __init__.py
│   │   ├── grammar.py
│   │   ├── individual.py
│   │   ├── evolution.py
│   │   ├── evaluation.py
│   │   └── pareto_front.py
│   ├── policy/               # Policy Network
│   │   ├── __init__.py
│   │   ├── network.py
│   │   ├── trainer.py
│   │   └── selector.py
│   ├── transfer/             # Transfer Learning Components (NEW)
│   │   ├── __init__.py
│   │   ├── adapter.py        # GradientTransferAdapter
│   │   └── task_losses.py    # Task-specific loss functions
│   ├── deployment/           # Edge Deployment Orchestration (NEW)
│   │   ├── __init__.py
│   │   ├── orchestrator.py   # EdgeOrchestrator for resource management
│   │   └── modality_router.py
│   ├── framework/            # Main Hyper-Heuristic Framework
│   │   ├── __init__.py
│   │   ├── hyperheuristic.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   └── segmenter.py
│   ├── utils/                # Utilities
│   │   ├── __init__.py
│   │   ├── reproducibility.py
│   │   ├── stats.py
│   │   ├── visualization.py
│   │   ├── metrics.py
│   │   ├── logger.py
│   │   └── profiler.py
│   └── experiments/          # Experiment Scripts
│       ├── __init__.py
│       ├── run_experiment.py
│       └── baseline_comparison.py
├── scripts/                  # Shell and CLI Scripts
│   ├── download_datasets.sh
│   ├── run_docker.sh
│   ├── run_experiments.sh
│   └── adapt_to_domain.py    # CLI for Transfer Learning (NEW)
├── tests/                    # Unit Tests
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_llhs.py
│   ├── test_transfer.py      # Transfer Adapter Tests (NEW)
│   └── test_framework.py
├── notebooks/                # Jupyter Notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_results_analysis.ipynb
│   └── 03_transfer_learning.ipynb  # Demo Notebook (NEW)
└── results/                  # Output Directory (Auto-Created)
