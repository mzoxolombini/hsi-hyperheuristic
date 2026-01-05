"""
Reproducibility manager module
Execution Order: 33
"""

import os
import random
import numpy as np
import torch
import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import multiprocessing as mp
import sys

logger = logging.getLogger(__name__)


class ReproducibilityManager:
    """
    Ensures exact reproducibility of experiments
    
    Features:
    1. Global seed setting for all libraries
    2. Hardware configuration logging
    3. Data checksums
    4. Execution logging
    5. Energy measurement (if available)
    """
    
    def __init__(self, config: Dict[str, Any], results_dir: str = "./results"):
        """
        Initialize reproducibility manager
        
        Args:
            config: Framework configuration
            results_dir: Results directory
        """
        self.config = config
        self.results_dir = Path(results_dir)
        self.cache_dir = Path(config.get('cache_dir', './cache'))
        self.checksums: Dict[str, str] = {}
        self.execution_log: List[Dict[str, Any]] = []
        
        # Create directories
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set global seeds
        self._set_global_seeds()
        
        # Create execution log
        self.start_time = datetime.now()
        self.execution_id = self._generate_execution_id()
        
        # Log hardware configuration
        self.hardware_config = self._log_hardware_config()
        
        logger.info(f"Reproducibility manager initialized. Execution ID: {self.execution_id}")
    
    def _set_global_seeds(self) -> None:
        """Set seeds for complete determinism"""
        seed = self.config.get('random_seed', 42)
        
        # Python random
        random.seed(seed)
        
        # NumPy
        np.random.seed(seed)
        
        # PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # CuDNN deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Environment variable
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        logger.info(f"Global seeds set to: {seed}")
    
    def _generate_execution_id(self) -> str:
        """Generate unique execution ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        seed = self.config.get('random_seed', 42)
        return f"exec_{timestamp}_seed{seed}"
    
    def _log_hardware_config(self) -> Dict[str, Any]:
        """Log computational environment"""
        hardware_info = {
            "timestamp": self.start_time.isoformat(),
            "execution_id": self.execution_id,
            "python_version": sys.version,
            "platform": sys.platform,
            "cpu_count": mp.cpu_count(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        # GPU information
        if torch.cuda.is_available():
            hardware_info["cuda_version"] = torch.version.cuda
            hardware_info["gpu_count"] = torch.cuda.device_count()
            hardware_info["gpus"] = []
            
            for i in range(torch.cuda.device_count()):
                gpu_info = {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory / (1024**3),  # GB
                }
                hardware_info["gpus"].append(gpu_info)
        
        # Try to get more detailed GPU info from nvidia-smi
        try:
            nvidia_smi = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version", 
                 "--format=csv,noheader"],
                text=True
            ).strip().split('\n')
            
            for i, line in enumerate(nvidia_smi):
                if i < len(hardware_info.get("gpus", [])):
                    name, mem_total, mem_free, driver = line.split(', ')
                    hardware_info["gpus"][i]["nvidia_name"] = name.strip()
                    hardware_info["gpus"][i]["nvidia_memory_total"] = mem_total.strip()
                    hardware_info["gpus"][i]["nvidia_memory_free"] = mem_free.strip()
                    hardware_info["gpus"][i]["driver_version"] = driver.strip()
        except:
            pass
        
        # Save hardware config
        hardware_file = self.results_dir / "hardware_config.json"
        with open(hardware_file, 'w') as f:
            json.dump(hardware_info, f, indent=2)
        
        logger.info(f"Hardware configuration logged to {hardware_file}")
        return hardware_info
    
    def compute_checksum(self, data: Any, name: str) -> str:
        """
        Compute MD5 checksum for data validation
        
        Args:
            data: Data to checksum
            name: Identifier for the data
            
        Returns:
            MD5 checksum
        """
        if isinstance(data, np.ndarray):
            # For numpy arrays
            checksum = hashlib.md5(data.tobytes()).hexdigest()
        elif isinstance(data, (str, bytes)):
            # For strings/bytes
            if isinstance(data, str):
                data = data.encode('utf-8')
            checksum = hashlib.md5(data).hexdigest()
        elif isinstance(data, dict):
            # For dictionaries
            json_str = json.dumps(data, sort_keys=True).encode('utf-8')
            checksum = hashlib.md5(json_str).hexdigest()
        else:
            # For other types, convert to string
            checksum = hashlib.md5(str(data).encode('utf-8')).hexdigest()
        
        self.checksums[name] = checksum
        
        # Log checksum
        self.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": "compute_checksum",
            "name": name,
            "checksum": checksum
        })
        
        return checksum
    
    def save_with_provenance(self, data: Any, filename: str, 
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Save data with checksum and metadata
        
        Args:
            data: Data to save
            filename: Output filename
            metadata: Additional metadata
            
        Returns:
            Path to saved file
        """
        filepath = self.results_dir / filename
        
        # Save data based on type
        if isinstance(data, np.ndarray):
            np.save(str(filepath), data)
            filepath = Path(str(filepath) + '.npy')
        elif isinstance(data, dict):
            with open(str(filepath) + '.json', 'w') as f:
                json.dump(data, f, indent=2)
            filepath = Path(str(filepath) + '.json')
        elif isinstance(data, str):
            with open(str(filepath) + '.txt', 'w') as f:
                f.write(data)
            filepath = Path(str(filepath) + '.txt')
        else:
            # Use pickle for other types
            import pickle
            with open(str(filepath) + '.pkl', 'wb') as f:
                pickle.dump(data, f)
            filepath = Path(str(filepath) + '.pkl')
        
        # Compute checksum
        checksum = self.compute_checksum(data, filename)
        
        # Create provenance metadata
        provenance = {
            "filename": filename,
            "filepath": str(filepath),
            "checksum": checksum,
            "timestamp": datetime.now().isoformat(),
            "execution_id": self.execution_id,
            "hardware_config": self.hardware_config.get("execution_id"),
        }
        
        if metadata:
            provenance["user_metadata"] = metadata
        
        # Save provenance
        provenance_file = self.results_dir / f"{filename}_provenance.json"
        with open(provenance_file, 'w') as f:
            json.dump(provenance, f, indent=2)
        
        # Update execution log
        self.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": "save_with_provenance",
            "filename": filename,
            "filepath": str(filepath),
            "checksum": checksum,
            "provenance_file": str(provenance_file)
        })
        
        logger.info(f"Saved {filename} with provenance to {filepath}")
        return str(filepath)
    
    def get_energy_usage(self, duration: float, 
                        device_id: int = 0) -> Dict[str, float]:
        """
        Measure energy consumption during execution
        
        Args:
            duration: Execution duration in seconds
            device_id: GPU device ID
            
        Returns:
            Dictionary with energy measurements
        """
        if not self.config.get('measure_energy', False):
            return {
                "energy_kj": 0.0,
                "power_w": 0.0,
                "duration_s": duration,
                "measured": False
            }
        
        try:
            # Query GPU power using nvidia-smi
            smi_output = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits",
                f"--id={device_id}"
            ], text=True).strip()
            
            # Parse power draw
            power_w = float(smi_output) if smi_output else 0.0
            
            # Calculate energy (assuming constant power during execution)
            energy_j = power_w * duration
            energy_kj = energy_j / 1000.0
            
            result = {
                "power_w": power_w,
                "energy_j": energy_j,
                "energy_kj": energy_kj,
                "duration_s": duration,
                "device_id": device_id,
                "measured": True,
                "timestamp": datetime.now().isoformat()
            }
            
            # Log energy measurement
            self.execution_log.append({
                "timestamp": datetime.now().isoformat(),
                "action": "energy_measurement",
                "duration_s": duration,
                "energy_kj": energy_kj,
                "power_w": power_w
            })
            
            return result
            
        except Exception as e:
            logger.warning(f"Failed to measure energy: {e}")
            return {
                "energy_kj": 0.0,
                "power_w": 0.0,
                "duration_s": duration,
                "measured": False,
                "error": str(e)
            }
    
    def log_execution_step(self, step: str, details: Dict[str, Any]) -> None:
        """
        Log an execution step
        
        Args:
            step: Step name/description
            details: Step details
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "details": details
        }
        
        self.execution_log.append(log_entry)
        logger.info(f"Execution step: {step}")
    
    def save_execution_log(self) -> str:
        """
        Save execution log to file
        
        Returns:
            Path to saved log file
        """
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        # Create summary
        summary = {
            "execution_id": self.execution_id,
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_s": duration,
            "total_steps": len(self.execution_log),
            "checksums": self.checksums,
            "hardware_config_id": self.hardware_config.get("execution_id")
        }
        
        # Combine summary and log
        full_log = {
            "summary": summary,
            "log": self.execution_log
        }
        
        # Save log
        log_file = self.results_dir / "execution_log.json"
        with open(log_file, 'w') as f:
            json.dump(full_log, f, indent=2)
        
        logger.info(f"Execution log saved to {log_file}")
        return str(log_file)
    
    def create_checkpoint(self, checkpoint_name: str, 
                         state: Dict[str, Any]) -> str:
        """
        Create a reproducible checkpoint
        
        Args:
            checkpoint_name: Checkpoint name
            state: State dictionary
            
        Returns:
            Path to checkpoint file
        """
        checkpoint_dir = self.results_dir / "checkpoints" / self.execution_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_file = checkpoint_dir / f"{checkpoint_name}.ckpt"
        
        # Add metadata
        state['_checkpoint_metadata'] = {
            "name": checkpoint_name,
            "execution_id": self.execution_id,
            "timestamp": datetime.now().isoformat(),
            "step": len(self.execution_log)
        }
        
        # Save checkpoint
        import pickle
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(state, f)
        
        # Compute checksum
        with open(checkpoint_file, 'rb') as f:
            checksum = hashlib.md5(f.read()).hexdigest()
        
        # Save checksum
        checksum_file = checkpoint_dir / f"{checkpoint_name}.md5"
        with open(checksum_file, 'w') as f:
            f.write(checksum)
        
        # Log checkpoint creation
        self.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": "create_checkpoint",
            "checkpoint_name": checkpoint_name,
            "checkpoint_file": str(checkpoint_file),
            "checksum": checksum,
            "step": len(self.execution_log)
        })
        
        logger.info(f"Checkpoint '{checkpoint_name}' created at {checkpoint_file}")
        return str(checkpoint_file)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load a checkpoint with validation
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Loaded state
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Verify checksum if available
        checksum_file = checkpoint_path.parent / f"{checkpoint_path.stem}.md5"
        if checksum_file.exists():
            with open(checksum_file, 'r') as f:
                expected_checksum = f.read().strip()
            
            with open(checkpoint_path, 'rb') as f:
                actual_checksum = hashlib.md5(f.read()).hexdigest()
            
            if expected_checksum != actual_checksum:
                raise ValueError(f"Checkpoint checksum mismatch for {checkpoint_path}")
        
        # Load checkpoint
        import pickle
        with open(checkpoint_path, 'rb') as f:
            state = pickle.load(f)
        
        # Log checkpoint loading
        self.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": "load_checkpoint",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_metadata": state.get('_checkpoint_metadata', {})
        })
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
        return state
    
    def verify_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """
        Verify dataset integrity
        
        Args:
            dataset_path: Path to dataset
            
        Returns:
            Verification results
        """
        import scipy.io
        
        dataset_path = Path(dataset_path)
        results = {
            "dataset": str(dataset_path),
            "exists": dataset_path.exists(),
            "valid_mat_file": False,
            "variables": [],
            "checksum": None
        }
        
        if not dataset_path.exists():
            return results
        
        try:
            # Try to load as MATLAB file
            data = scipy.io.loadmat(dataset_path)
            results["valid_mat_file"] = True
            results["variables"] = list(data.keys())
            
            # Compute checksum
            with open(dataset_path, 'rb') as f:
                results["checksum"] = hashlib.md5(f.read()).hexdigest()
            
        except Exception as e:
            results["error"] = str(e)
        
        # Log verification
        self.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": "verify_dataset",
            "dataset": str(dataset_path),
            "results": results
        })
        
        return results
