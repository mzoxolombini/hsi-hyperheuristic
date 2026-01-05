"""
Metrics calculation module
Execution Order: 36
"""

import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

logger = logging.getLogger(__name__)


class SegmentationMetrics:
    """Segmentation performance metrics calculator"""
    
    @staticmethod
    def calculate_miou(pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate mean Intersection over Union"""
        if pred.shape != gt.shape:
            return 0.0
        
        classes = np.unique(gt)
        ious = []
        
        for c in classes:
            pred_c = (pred == c)
            gt_c = (gt == c)
            
            intersection = np.logical_and(pred_c, gt_c).sum()
            union = np.logical_or(pred_c, gt_c).sum()
            
            if union > 0:
                ious.append(intersection / union)
        
        return np.mean(ious) if ious else 0.0
    
    @staticmethod
    def calculate_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate overall accuracy"""
        if pred.shape != gt.shape:
            return 0.0
        
        correct = np.sum(pred == gt)
        total = pred.size
        
        return correct / total if total > 0 else 0.0
    
    @staticmethod
    def calculate_precision_recall_f1(pred: np.ndarray, gt: np.ndarray) -> Dict[str, Any]:
        """Calculate precision, recall, F1-score per class"""
        if pred.shape != gt.shape:
            return {}
        
        classes = np.unique(gt)
        results = {}
        
        for c in classes:
            pred_c = (pred == c)
            gt_c = (gt == c)
            
            tp = np.logical_and(pred_c, gt_c).sum()
            fp = np.logical_and(pred_c, np.logical_not(gt_c)).sum()
            fn = np.logical_and(np.logical_not(pred_c), gt_c).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            results[f'class_{c}'] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': int(tp),
                'fp': int(fp),
                'fn': int(fn)
            }
        
        # Calculate macro averages
        precisions = [v['precision'] for v in results.values()]
        recalls = [v['recall'] for v in results.values()]
        f1s = [v['f1'] for v in results.values()]
        
        results['macro_avg'] = {
            'precision': np.mean(precisions),
            'recall': np.mean(recalls),
            'f1': np.mean(f1s)
        }
        
        return results
    
    @staticmethod
    def calculate_kappa(pred: np.ndarray, gt: np.ndarray) -> float:
        """Calculate Cohen's Kappa coefficient"""
        if pred.shape != gt.shape:
            return 0.0
        
        try:
            return float(cohen_kappa_score(gt.flatten(), pred.flatten()))
        except:
            return 0.0
    
    @staticmethod
    def calculate_all_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, Any]:
        """Calculate all segmentation metrics"""
        return {
            'mIoU': SegmentationMetrics.calculate_miou(pred, gt),
            'OverallAccuracy': SegmentationMetrics.calculate_accuracy(pred, gt),
            'Kappa': SegmentationMetrics.calculate_kappa(pred, gt),
            'ClassMetrics': SegmentationMetrics.calculate_precision_recall_f1(pred, gt),
            'ConfusionMatrix': SegmentationMetrics.calculate_confusion_matrix(pred, gt)
        }
    
    @staticmethod
    def calculate_confusion_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """Calculate confusion matrix"""
        if pred.shape != gt.shape:
            return np.array([])
        
        classes = np.unique(np.concatenate([pred.flatten(), gt.flatten()]))
        return confusion_matrix(gt.flatten(), pred.flatten(), labels=classes)
