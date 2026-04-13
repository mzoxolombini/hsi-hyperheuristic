"""
Multi-Objective Fitness Evaluation (Section 5.3.4)
Implements Pareto-based fitness with three competing objectives:
1. Maximize mIoU (Accuracy)
2. Minimize FLOPs (Computational Cost)
3. Minimize Tree Depth (Model Complexity)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from .metrics import compute_iou


class MultiObjectiveFitness:
    """
    Pareto-based fitness evaluation for grammar-guided evolution.
    """
    
    def __init__(self, flop_counter=None):
        self.flop_counter = flop_counter or FlopCountAnalysis
    
    def compute_flops(
        self, 
        pipeline: torch.nn.Module, 
        input_tensor: torch.Tensor
    ) -> int:
        """
        Estimate FLOPs for a given pipeline on a sample input.
        
        Args:
            pipeline: Compiled segmentation pipeline
            input_tensor: Sample input (1, C, H, W)
        Returns:
            flops: Total floating-point operations
        """
        try:
            flops = self.flop_counter(pipeline, input_tensor)
            total_flops = flops.total()
        except Exception as e:
            # Penalize pipelines that can't be analyzed
            print(f"FLOP analysis failed: {e}")
            total_flops = 10 ** 12
        return total_flops
    
    def compute_parameters(self, pipeline: torch.nn.Module) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in pipeline.parameters() if p.requires_grad)
    
    def evaluate_pipeline(
        self,
        individual: Any,
        val_loader: torch.utils.data.DataLoader,
        device: str = 'cuda'
    ) -> Dict[str, float]:
        """
        Evaluate a pipeline on all three objectives.
        
        Args:
            individual: Grammar individual (derivation tree)
            val_loader: Validation data loader
            device: Computation device
        Returns:
            fitness_dict: Dictionary with mIoU, FLOPs, tree_depth
        """
        # Compile the pipeline
        pipeline = individual.compile(device)
        pipeline.eval()
        
        # 1. Accuracy Objective: Mean IoU
        ious = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                # Handle different output formats
                output = pipeline(images)
                if isinstance(output, (list, tuple)):
                    output = output[0]
                
                # Compute IoU for each class
                pred = output.argmax(dim=1) if output.shape[1] > 1 else (output > 0.5).long()
                
                batch_iou = compute_iou(pred, masks, num_classes=individual.n_classes)
                ious.append(batch_iou)
        
        avg_miou = np.mean(ious)
        
        # 2. Computational Cost Objective: FLOPs
        sample_input = next(iter(val_loader))['image'][:1].to(device)
        flops = self.compute_flops(pipeline, sample_input)
        
        # 3. Model Complexity Objective: Tree Depth
        tree_depth = individual.tree_depth()
        
        return {
            'mIoU': avg_miou,
            'FLOPs': flops,
            'tree_depth': tree_depth,
            'parameters': self.compute_parameters(pipeline)
        }
    
    def pareto_dominates(
        self, 
        a: Dict[str, float], 
        b: Dict[str, float]
    ) -> bool:
        """
        Check if solution a Pareto-dominates solution b.
        
        a dominates b if:
        - a.mIoU >= b.mIoU AND a.FLOPs <= b.FLOPs AND a.tree_depth <= b.tree_depth
        - AND at least one inequality is strict
        """
        # For maximization (mIoU) and minimization (FLOPs, depth)
        mIoU_condition = a['mIoU'] >= b['mIoU']
        flops_condition = a['FLOPs'] <= b['FLOPs']
        depth_condition = a['tree_depth'] <= b['tree_depth']
        
        strict_condition = (
            a['mIoU'] > b['mIoU'] or 
            a['FLOPs'] < b['FLOPs'] or 
            a['tree_depth'] < b['tree_depth']
        )
        
        return mIoU_condition and flops_condition and depth_condition and strict_condition
    
    def compute_pareto_frontier(
        self, 
        solutions: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        """
        Compute Pareto-optimal frontier from a list of solutions.
        
        Args:
            solutions: List of fitness dictionaries
        Returns:
            pareto_front: Non-dominated solutions
        """
        pareto_front = []
        
        for i, sol_i in enumerate(solutions):
            is_dominated = False
            for j, sol_j in enumerate(solutions):
                if i != j and self.pareto_dominates(sol_j, sol_i):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(sol_i)
        
        # Sort by mIoU descending
        pareto_front.sort(key=lambda x: x['mIoU'], reverse=True)
        return pareto_front
    
    def compute_crowding_distance(
        self, 
        front: List[Dict[str, float]]
    ) -> List[float]:
        """
        Compute crowding distance for diversity preservation (NSGA-II).
        
        Args:
            front: Pareto front solutions
        Returns:
            distances: Crowding distance for each solution
        """
        n = len(front)
        if n <= 2:
            return [float('inf')] * n
        
        distances = [0.0] * n
        
        # For each objective
        objectives = ['mIoU', 'FLOPs', 'tree_depth']
        for obj in objectives:
            # Sort by objective value
            sorted_indices = sorted(range(n), key=lambda i: front[i][obj])
            
            # Set boundary distances to infinity
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Normalize objective range
            obj_values = [front[i][obj] for i in sorted_indices]
            obj_range = max(obj_values) - min(obj_values)
            
            if obj_range > 0:
                for j in range(1, n - 1):
                    idx = sorted_indices[j]
                    prev_idx = sorted_indices[j - 1]
                    next_idx = sorted_indices[j + 1]
                    
                    distance = (front[next_idx][obj] - front[prev_idx][obj]) / obj_range
                    distances[idx] += distance
        
        return distances