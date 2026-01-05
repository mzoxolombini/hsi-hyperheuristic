"""
Configuration loader module
Execution Order: 5
"""

import json
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    name: str
    url: str
    gt_url: str
    height: int
    width: int
    bands: int
    classes: int
    bad_bands: List[List[int]] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetConfig':
        return cls(**data)


@dataclass
class PreprocessingConfig:
    """Preprocessing configuration"""
    patch_size: int
    stride: int
    train_patches: int
    test_patches: int
    augmentation_prob: float
    denoising_method: str
    dimensionality_reduction: str
    variance_threshold: float
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PreprocessingConfig':
        return cls(**data)


@dataclass
class GPConfig:
    """Genetic Programming configuration"""
    population_size: int
    generations: int
    crossover_rate: float
    mutation_rate: float
    elitism: int
    tournament_size: int
    max_depth: int
    objectives: List[str]
    weights: List[float]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GPConfig':
        return cls(**data)


@dataclass
class PSOConfig:
    """PSO configuration"""
    population: int
    iterations: int
    inertia_base: float
    inertia_min: float
    cognitive: float
    social: float
    spatial_weight: float
    variants: List[str]
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PSOConfig':
        return cls(**data)


@dataclass
class PolicyConfig:
    """Policy network configuration"""
    hidden_dims: List[int]
    dropout: float
    learning_rate: float
    epochs: int
    batch_size: int
    weight_decay: float
    entropy_coef: float
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PolicyConfig':
        return cls(**data)


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    metrics: List[str]
    confidence_level: float
    n_bootstrap: int
    test_size: float
    val_size: float
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationConfig':
        return cls(**data)


@dataclass
class HardwareConfig:
    """Hardware configuration"""
    gpu_ids: List[int]
    num_workers: int
    pin_memory: bool
    mixed_precision: bool
    cudnn_benchmark: bool
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HardwareConfig':
        return cls(**data)


@dataclass
class FrameworkConfig:
    """Main framework configuration"""
    name: str
    version: str
    mode: str
    random_seed: int
    n_runs: int
    measure_energy: bool
    enable_profiling: bool
    log_level: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FrameworkConfig':
        return cls(**data)


@dataclass
class PathsConfig:
    """Paths configuration"""
    data_dir: str
    results_dir: str
    cache_dir: str
    logs_dir: str
    checkpoints_dir: str
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PathsConfig':
        return cls(**data)


@dataclass
class Config:
    """Complete configuration"""
    framework: FrameworkConfig
    paths: PathsConfig
    datasets: Dict[str, DatasetConfig]
    preprocessing: PreprocessingConfig
    gp: GPConfig
    pso: PSOConfig
    policy: PolicyConfig
    evaluation: EvaluationConfig
    hardware: HardwareConfig
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        return cls(
            framework=FrameworkConfig.from_dict(data["framework"]),
            paths=PathsConfig.from_dict(data["paths"]),
            datasets={
                name: DatasetConfig.from_dict(dataset_data)
                for name, dataset_data in data["datasets"].items()
            },
            preprocessing=PreprocessingConfig.from_dict(data["preprocessing"]),
            gp=GPConfig.from_dict(data["gp"]),
            pso=PSOConfig.from_dict(data["pso"]),
            policy=PolicyConfig.from_dict(data["policy"]),
            evaluation=EvaluationConfig.from_dict(data["evaluation"]),
            hardware=HardwareConfig.from_dict(data["hardware"])
        )


class ConfigLoader:
    """Configuration loader"""
    
    @staticmethod
    def load_config(config_path: Optional[str] = None) -> Config:
        """
        Load configuration from file
        
        Args:
            config_path: Path to config file. If None, uses default.
            
        Returns:
            Config object
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.json"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load based on file extension
        if config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                data = json.load(f)
        elif config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        logger.info(f"Loaded configuration from {config_path}")
        return Config.from_dict(data)
    
    @staticmethod
    def save_config(config: Config, output_path: str) -> None:
        """
        Save configuration to file
        
        Args:
            config: Config object
            output_path: Output file path
        """
        output_path = Path(output_path)
        
        # Convert to dict
        data = {
            "framework": config.framework.__dict__,
            "paths": config.paths.__dict__,
            "datasets": {
                name: dataset.__dict__
                for name, dataset in config.datasets.items()
            },
            "preprocessing": config.preprocessing.__dict__,
            "gp": config.gp.__dict__,
            "pso": config.pso.__dict__,
            "policy": config.policy.__dict__,
            "evaluation": config.evaluation.__dict__,
            "hardware": config.hardware.__dict__
        }
        
        # Save based on file extension
        if output_path.suffix == '.json':
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif output_path.suffix in ['.yaml', '.yml']:
            with open(output_path, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")
        
        logger.info(f"Saved configuration to {output_path}")
