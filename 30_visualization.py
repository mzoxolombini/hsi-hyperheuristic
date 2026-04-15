"""
Visualization module
Execution Order: 35
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

logger = logging.getLogger(__name__)


class VisualizationEngine:
    """
    Visualization engine for hyperspectral data and results
    
    Implements:
    1. Hyperspectral data visualization
    2. Segmentation result visualization
    3. Performance metric plots
    4. Interactive visualizations
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize visualization engine
        
        Args:
            config: Visualization configuration
        """
        self.config = config or {}
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Color maps
        self.cmaps = {
            'segmentation': 'tab20',
            'gradient': 'viridis',
            'uncertainty': 'plasma',
            'probability': 'RdYlBu_r'
        }
        
        logger.info("Visualization engine initialized")
    
    def plot_hyperspectral_data(self, data: np.ndarray, 
                               save_path: Optional[str] = None) -> None:
        """
        Plot hyperspectral data
        
        Args:
            data: Hyperspectral image [H, W, B]
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot RGB composite (if available)
        if data.shape[2] >= 3:
            rgb = data[:, :, [29, 19, 9]]  # Common RGB bands for HSI
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
            axes[0, 0].imshow(rgb)
            axes[0, 0].set_title('RGB Composite (Bands 30,20,10)', fontsize=12)
            axes[0, 0].axis('off')
        
        # Plot mean band
        mean_band = np.mean(data, axis=2)
        im1 = axes[0, 1].imshow(mean_band, cmap='gray')
        axes[0, 1].set_title('Mean Band', fontsize=12)
        axes[0, 1].axis('off')
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
        
        # Plot standard deviation
        std_band = np.std(data, axis=2)
        im2 = axes[0, 2].imshow(std_band, cmap='hot')
        axes[0, 2].set_title('Standard Deviation', fontsize=12)
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # Plot spectral profile at random points
        h, w, b = data.shape
        points = []
        for _ in range(5):
            i, j = np.random.randint(0, h), np.random.randint(0, w)
            points.append((i, j))
            spectrum = data[i, j, :]
            axes[1, 0].plot(spectrum, label=f'({i},{j})', alpha=0.7)
        
        axes[1, 0].set_title('Random Spectral Profiles', fontsize=12)
        axes[1, 0].set_xlabel('Band Index')
        axes[1, 0].set_ylabel('Reflectance')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot band correlation matrix
        if b <= 50:  # Limit for visualization
            sample_data = data.reshape(-1, b)
            if sample_data.shape[0] > 1000:
                indices = np.random.choice(sample_data.shape[0], 1000, replace=False)
                sample_data = sample_data[indices]
            
            corr_matrix = np.corrcoef(sample_data.T)
            im3 = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 1].set_title('Band Correlation Matrix', fontsize=12)
            axes[1, 1].set_xlabel('Band Index')
            axes[1, 1].set_ylabel('Band Index')
            plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
        
        # Plot histogram of mean band
        axes[1, 2].hist(mean_band.flatten(), bins=50, alpha=0.7, edgecolor='black')
        axes[1, 2].set_title('Histogram of Mean Band', fontsize=12)
        axes[1, 2].set_xlabel('Intensity')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Hyperspectral Data Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Hyperspectral data plot saved to {save_path}")
        
        plt.show()
    
    def plot_segmentation_results(self, image: np.ndarray, segmentation: np.ndarray,
                                 ground_truth: Optional[np.ndarray] = None,
                                 save_path: Optional[str] = None) -> None:
        """
        Plot segmentation results
        
        Args:
            image: Original image [H, W, B]
            segmentation: Segmentation map [H, W]
            ground_truth: Ground truth labels (optional)
            save_path: Path to save plot (optional)
        """
        n_rows = 2 if ground_truth is None else 3
        fig, axes = plt.subplots(1, n_rows, figsize=(5*n_rows, 5))
        
        if n_rows == 2:
            axes = [axes[0], axes[1]]
        
        # Plot original image (mean band)
        mean_band = np.mean(image, axis=2)
        axes[0].imshow(mean_band, cmap='gray')
        axes[0].set_title('Original Image (Mean Band)', fontsize=14)
        axes[0].axis('off')
        
        # Plot segmentation
        im = axes[1].imshow(segmentation, cmap=self.cmaps['segmentation'])
        axes[1].set_title('Segmentation Result', fontsize=14)
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Plot ground truth if available
        if ground_truth is not None:
            im = axes[2].imshow(ground_truth, cmap=self.cmaps['segmentation'])
            axes[2].set_title('Ground Truth', fontsize=14)
            axes[2].axis('off')
            plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        plt.suptitle('Segmentation Results', fontsize=16, y=1.05)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Segmentation results plot saved to {save_path}")
        
        plt.show()
    
    def plot_segmentation_comparison(self, image: np.ndarray,
                                    segmentations: Dict[str, np.ndarray],
                                    ground_truth: Optional[np.ndarray] = None,
                                    save_path: Optional[str] = None) -> None:
        """
        Plot multiple segmentation results for comparison
        
        Args:
            image: Original image
            segmentations: Dictionary of {method_name: segmentation_map}
            ground_truth: Ground truth labels (optional)
            save_path: Path to save plot (optional)
        """
        n_methods = len(segmentations)
        n_rows = 2 if ground_truth is None else 3
        n_cols = max(2, n_methods)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        
        # Flatten axes for easier indexing
        if n_rows > 1 and n_cols > 1:
            axes = axes.flatten()
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes[:, 0]
        
        # Plot original image
        mean_band = np.mean(image, axis=2)
        axes[0].imshow(mean_band, cmap='gray')
        axes[0].set_title('Original Image', fontsize=12)
        axes[0].axis('off')
        
        # Plot each segmentation
        for idx, (method_name, segmentation) in enumerate(segmentations.items()):
            ax_idx = idx + 1
            if ax_idx < len(axes):
                im = axes[ax_idx].imshow(segmentation, cmap=self.cmaps['segmentation'])
                axes[ax_idx].set_title(f'{method_name}', fontsize=12)
                axes[ax_idx].axis('off')
                plt.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04)
        
        # Plot ground truth if available
        if ground_truth is not None:
            ax_idx = n_methods + 1
            if ax_idx < len(axes):
                im = axes[ax_idx].imshow(ground_truth, cmap=self.cmaps['segmentation'])
                axes[ax_idx].set_title('Ground Truth', fontsize=12)
                axes[ax_idx].axis('off')
                plt.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04)
        
        # Hide empty subplots
        for idx in range(len(segmentations) + 2, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Segmentation Method Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Segmentation comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_performance_metrics(self, metrics_df: pd.DataFrame,
                                save_path: Optional[str] = None) -> None:
        """
        Plot performance metrics comparison
        
        Args:
            metrics_df: DataFrame with metrics for each method
            save_path: Path to save plot (optional)
        """
        if metrics_df.empty:
            logger.warning("No metrics data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Bar plot of mIoU
        methods = metrics_df.index.tolist()
        miou_scores = metrics_df.get('mIoU', [0] * len(methods))
        
        bars = axes[0, 0].bar(methods, miou_scores, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Mean IoU Comparison', fontsize=14)
        axes[0, 0].set_ylabel('mIoU')
        axes[0, 0].set_ylim(0, 1.0)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Bar plot of Overall Accuracy
        if 'OverallAccuracy' in metrics_df.columns:
            oa_scores = metrics_df['OverallAccuracy']
            bars = axes[0, 1].bar(methods, oa_scores, color='lightgreen', edgecolor='black')
            axes[0, 1].set_title('Overall Accuracy Comparison', fontsize=14)
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].set_ylim(0, 1.0)
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                              f'{height:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Execution time comparison
        if 'ExecutionTime' in metrics_df.columns:
            time_scores = metrics_df['ExecutionTime']
            bars = axes[1, 0].bar(methods, time_scores, color='salmon', edgecolor='black')
            axes[1, 0].set_title('Execution Time Comparison', fontsize=14)
            axes[1, 0].set_ylabel('Time (seconds)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                              f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
        
        # Memory usage comparison
        if 'MemoryUsage' in metrics_df.columns:
            memory_scores = metrics_df['MemoryUsage']
            bars = axes[1, 1].bar(methods, memory_scores, color='gold', edgecolor='black')
            axes[1, 1].set_title('Memory Usage Comparison', fontsize=14)
            axes[1, 1].set_ylabel('Memory (MB)')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                              f'{height:.1f}MB', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('Performance Metrics Comparison', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Performance metrics plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray,
                             class_names: List[str],
                             save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrix
        
        Args:
            confusion_matrix: Confusion matrix [n_classes, n_classes]
            class_names: List of class names
            save_path: Path to save plot (optional)
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = confusion_matrix.astype('float') / \
                       (confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-10)
        
        # Create heatmap
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        
        # Add colorbar
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel('Normalized Count', rotation=-90, va="bottom")
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_yticklabels(class_names)
        
        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                # Raw count
                count = confusion_matrix[i, j]
                # Percentage
                percentage = cm_normalized[i, j] * 100
                
                text = f'{count}\n({percentage:.1f}%)'
                color = "white" if cm_normalized[i, j] > thresh else "black"
                ax.text(j, i, text,
                       ha="center", va="center",
                       color=color, fontsize=9)
        
        ax.set_title('Confusion Matrix', fontsize=16, pad=20)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {save_path}")
        
        plt.show()
    
    def plot_pareto_front(self, pareto_points: List[Tuple[float, float, float]],
                         save_path: Optional[str] = None) -> None:
        """
        Plot 3D Pareto front
        
        Args:
            pareto_points: List of (accuracy, efficiency, complexity) tuples
            save_path: Path to save plot (optional)
        """
        if not pareto_points:
            logger.warning("No Pareto points to plot")
            return
        
        # Convert to numpy array
        points = np.array(pareto_points)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract coordinates
        x = points[:, 0]  # Accuracy
        y = points[:, 1]  # Efficiency
        z = points[:, 2]  # Complexity
        
        # Create scatter plot
        sc = ax.scatter(x, y, z, c=z, cmap='viridis', s=100, alpha=0.7, edgecolors='black')
        
        # Add colorbar
        cbar = fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Complexity', rotation=270, labelpad=15)
        
        # Set labels
        ax.set_xlabel('Accuracy (mIoU)', labelpad=10)
        ax.set_ylabel('Efficiency (1/time)', labelpad=10)
        ax.set_zlabel('Complexity', labelpad=10)
        
        ax.set_title('Pareto Front: Accuracy vs Efficiency vs Complexity', fontsize=14)
        
        # Set limits
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, max(y) * 1.1)
        ax.set_zlim(0, max(z) * 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pareto front plot saved to {save_path}")
        
        plt.show()
    
    def plot_training_history(self, history: Dict[str, List[float]],
                            save_path: Optional[str] = None) -> None:
        """
        Plot training history
        
        Args:
            history: Dictionary with training metrics
            save_path: Path to save plot (optional)
        """
        if not history:
            logger.warning("No training history to plot")
            return
        
        n_metrics = len(history)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        epochs = range(1, len(next(iter(history.values()))) + 1)
        
        for idx, (metric_name, values) in enumerate(history.items()):
            if idx < len(axes):
                axes[idx].plot(epochs, values, 'b-', linewidth=2)
                axes[idx].set_title(f'{metric_name}', fontsize=14)
                axes[idx].set_xlabel('Epoch')
                axes[idx].set_ylabel(metric_name)
                axes[idx].grid(True, alpha=0.3)
                
                # Mark best value
                if 'loss' in metric_name.lower():
                    best_idx = np.argmin(values)
                    best_value = values[best_idx]
                    axes[idx].plot(best_idx + 1, best_value, 'ro', markersize=8)
                    axes[idx].annotate(f'Best: {best_value:.4f}',
                                     xy=(best_idx + 1, best_value),
                                     xytext=(10, 10),
                                     textcoords='offset points',
                                     fontsize=10)
                else:
                    best_idx = np.argmax(values)
                    best_value = values[best_idx]
                    axes[idx].plot(best_idx + 1, best_value, 'ro', markersize=8)
                    axes[idx].annotate(f'Best: {best_value:.4f}',
                                     xy=(best_idx + 1, best_value),
                                     xytext=(10, 10),
                                     textcoords='offset points',
                                     fontsize=10)
        
        plt.suptitle('Training History', fontsize=16, y=1.05)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def create_interactive_dashboard(self, data: Dict[str, Any],
                                   save_path: Optional[str] = None) -> None:
        """
        Create interactive dashboard with Plotly
        
        Args:
            data: Dictionary with visualization data
            save_path: Path to save HTML dashboard (optional)
        """
        try:
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=('Segmentation Results', 'Performance Metrics',
                              'Confusion Matrix', 'Training History',
                              'Feature Importance', 'Uncertainty Map'),
                specs=[[{'type': 'image'}, {'type': 'bar'}, {'type': 'heatmap'}],
                      [{'type': 'scatter'}, {'type': 'bar'}, {'type': 'heatmap'}]]
            )
            
            # Add segmentation image
            if 'segmentation' in data:
                seg = data['segmentation']
                fig.add_trace(
                    go.Heatmap(z=seg, colorscale='Viridis', showscale=False),
                    row=1, col=1
                )
            
            # Add performance metrics
            if 'metrics' in data:
                metrics = data['metrics']
                methods = list(metrics.keys())
                scores = list(metrics.values())
                
                fig.add_trace(
                    go.Bar(x=methods, y=scores, marker_color='skyblue'),
                    row=1, col=2
                )
            
            # Add confusion matrix
            if 'confusion_matrix' in data:
                cm = data['confusion_matrix']
                fig.add_trace(
                    go.Heatmap(z=cm, colorscale='Blues'),
                    row=1, col=3
                )
            
            # Add training history
            if 'training_history' in data:
                history = data['training_history']
                epochs = list(range(1, len(history.get('loss', [])) + 1))
                
                if 'loss' in history:
                    fig.add_trace(
                        go.Scatter(x=epochs, y=history['loss'],
                                 mode='lines', name='Loss',
                                 line=dict(color='red')),
                        row=2, col=1
                    )
                
                if 'accuracy' in history:
                    fig.add_trace(
                        go.Scatter(x=epochs, y=history['accuracy'],
                                 mode='lines', name='Accuracy',
                                 line=dict(color='blue')),
                        row=2, col=1
                    )
            
            # Update layout
            fig.update_layout(
                height=800,
                showlegend=True,
                title_text="Hyper-Heuristic Framework Dashboard",
                title_font_size=20
            )
            
            if save_path:
                fig.write_html(save_path)
                logger.info(f"Interactive dashboard saved to {save_path}")
            
            fig.show()
            
        except ImportError:
            logger.warning("Plotly not installed. Skipping interactive dashboard.")
    
    def save_all_plots(self, results: Dict[str, Any], output_dir: str) -> None:
        """
        Save all plots to directory
        
        Args:
            results: Dictionary with all results
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save segmentation results
        if 'segmentation' in results and 'image' in results:
            seg_path = output_path / 'segmentation_results.png'
            self.plot_segmentation_results(
                results['image'],
                results['segmentation'],
                results.get('ground_truth'),
                str(seg_path)
            )
        
        # Save performance metrics
        if 'metrics' in results:
            metrics_path = output_path / 'performance_metrics.png'
            # Convert metrics to DataFrame
            metrics_df = pd.DataFrame.from_dict(results['metrics'], orient='index')
            self.plot_performance_metrics(metrics_df, str(metrics_path))
        
        # Save confusion matrix
        if 'confusion_matrix' in results and 'class_names' in results:
            cm_path = output_path / 'confusion_matrix.png'
            self.plot_confusion_matrix(
                results['confusion_matrix'],
                results['class_names'],
                str(cm_path)
            )
        
        # Save training history
        if 'training_history' in results:
            history_path = output_path / 'training_history.png'
            self.plot_training_history(results['training_history'], str(history_path))
        
        # Save Pareto front if available
        if 'pareto_front' in results:
            pareto_path = output_path / 'pareto_front.png'
            self.plot_pareto_front(results['pareto_front'], str(pareto_path))
        
        logger.info(f"All plots saved to {output_dir}")


def generate_pdf_report(results: Dict[str, Any], output_path: str,
                       config: Dict[str, Any] = None) -> None:
    """
    Generate PDF report with all results
    
    Args:
        results: Dictionary with all results
        output_path: Output PDF path
        config: Configuration
    """
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        import matplotlib.pyplot as plt
        import tempfile
        import os
        
        # Create PDF document
        doc = SimpleDocTemplate(output_path, pagesize=A4,
                              rightMargin=72, leftMargin=72,
                              topMargin=72, bottomMargin=72)
        
        story = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center
        )
        
        story.append(Paragraph("Hyper-Heuristic Framework Report", title_style))
        story.append(Spacer(1, 0.25*inch))
        
        # Executive Summary
        story.append(Paragraph("Executive Summary", styles['Heading2']))
        summary_text = """
        This report presents the results of the hyper-heuristic framework for 
        hyperspectral image segmentation. The framework automatically discovers 
        optimal segmentation pipelines using grammar-guided genetic programming 
        and adapts low-level heuristics based on spectral-spatial meta-features.
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 0.25*inch))
        
        # Results Summary
        story.append(Paragraph("Results Summary", styles['Heading2']))
        
        # Create summary table
        summary_data = [
            ['Metric', 'Value'],
            ['Dataset', results.get('dataset', 'Unknown')],
            ['Best mIoU', f"{results.get('best_miou', 0):.4f}"],
            ['Training Time', f"{results.get('training_time', 0):.2f} seconds"],
            ['Energy Usage', f"{results.get('energy_usage', {}).get('energy_kj', 0):.2f} kJ"],
            ['Framework Version', results.get('version', '1.0.0')]
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 0.5*inch))
        
        # Save plots to temporary files and add to PDF
        viz = VisualizationEngine(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate and save plots
            tmp_plots = {}
            
            # Segmentation plot
            if 'segmentation' in results and 'image' in results:
                seg_path = os.path.join(tmpdir, 'segmentation.png')
                viz.plot_segmentation_results(
                    results['image'],
                    results['segmentation'],
                    results.get('ground_truth'),
                    seg_path
                )
                tmp_plots['segmentation'] = seg_path
            
            # Metrics plot
            if 'metrics' in results:
                metrics_path = os.path.join(tmpdir, 'metrics.png')
                metrics_df = pd.DataFrame.from_dict(results['metrics'], orient='index')
                viz.plot_performance_metrics(metrics_df, metrics_path)
                tmp_plots['metrics'] = metrics_path
            
            # Add plots to PDF
            for plot_name, plot_path in tmp_plots.items():
                if os.path.exists(plot_path):
                    story.append(Paragraph(plot_name.replace('_', ' ').title(), styles['Heading3']))
                    story.append(Spacer(1, 0.1*inch))
                    
                    img = Image(plot_path, width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 0.5*inch))
        
        # Statistical Significance
        if 'statistical_tests' in results:
            story.append(Paragraph("Statistical Significance", styles['Heading2']))
            
            for test_name, test_results in results['statistical_tests'].items():
                test_text = f"{test_name}: p-value = {test_results.get('p_value', 0):.6f}, "
                test_text += f"Significant = {test_results.get('significant', False)}"
                story.append(Paragraph(test_text, styles['Normal']))
            
            story.append(Spacer(1, 0.25*inch))
        
        # Conclusions
        story.append(Paragraph("Conclusions", styles['Heading2']))
        conclusions = """
        The hyper-heuristic framework successfully demonstrated adaptive 
        segmentation of hyperspectral images. The evolved pipelines show 
        competitive performance compared to baseline methods while maintaining 
        computational efficiency. The framework's ability to automatically 
        discover and adapt segmentation strategies makes it suitable for 
        various hyperspectral imaging applications.
        """
        story.append(Paragraph(conclusions, styles['Normal']))
        
        # Build PDF
        doc.build(story)
        logger.info(f"PDF report generated: {output_path}")
        
    except ImportError as e:
        logger.error(f"Cannot generate PDF report: {e}. Install reportlab.")
    except Exception as e:
        logger.error(f"Error generating PDF report: {e}")
