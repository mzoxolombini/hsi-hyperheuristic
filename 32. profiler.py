"""
Performance profiling module
Execution Order: 38
"""

import time
import psutil
import torch
from typing import Dict, Any, Optional
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Performance profiling and monitoring"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.metrics = {}
        
    @contextmanager
    def profile(self, name: str):
        """Context manager for profiling code blocks"""
        if not self.enabled:
            yield
            return
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            duration = end_time - start_time
            memory_used = (end_memory - start_memory) / (1024 * 1024)  # MB
            
            self.metrics[name] = {
                'duration': duration,
                'memory_used_mb': memory_used
            }
            
            logger.debug(f"Profiled {name}: {duration:.3f}s, {memory_used:.1f}MB")
    
    def get_metrics(self) -> Dict[str, Any]:
        return self.metrics.copy()
    
    def reset(self):
        self.metrics.clear()
