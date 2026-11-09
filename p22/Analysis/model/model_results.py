# Main writer: Cahit Kok
# Reviewer: Bartlomiej Kaczor
# Contributors: Elena Gostiukhina

import json
import os
from datetime import datetime
from typing import Dict, List


class ModelResults:
    """Standard container for model results and metadata."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.timestamp = datetime.now().isoformat()
        self.metrics = {}
        self.feature_importance = []
        self.hyperparameters = {}
        self.training_info = {}
        
    def set_metrics(self, metrics_dict: Dict):
        """Stores evaluation metrics (ROC-AUC, Accuracy, etc.)"""
        self.metrics = metrics_dict
        
    def set_feature_importance(self, features: List[Dict]):
        """Stores top N feature importances.
        Format: [{'name': 'feature1', 'importance': 0.25}, ...]
        """
        self.feature_importance = features
        
    def set_hyperparameters(self, params: Dict):
        """Stores model hyperparameters used."""
        self.hyperparameters = params
        
    def set_training_info(self, training_samples: int, test_samples: int, 
                         training_time_sec: float, cv_fold_count: int):
        """Stores training metadata."""
        self.training_info = {
            'training_samples': training_samples,
            'test_samples': test_samples,
            'training_time_seconds': training_time_sec,
            'cv_fold_count': cv_fold_count
        }
        
    def to_dict(self) -> Dict:
        """Converts results to dictionary for JSON serialization."""
        return {
            'model_name': self.model_name,
            'timestamp': self.timestamp,
            'metrics': self.metrics,
            'feature_importance': self.feature_importance,
            'hyperparameters': self.hyperparameters,
            'training_info': self.training_info
        }
    
    def save_json(self, filepath: str):
        """Saves results to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✓ Results saved to {filepath}")
        
    @staticmethod
    def load_json(filepath: str) -> 'ModelResults':
        """Loads results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        results = ModelResults(data['model_name'])
        results.metrics = data['metrics']
        results.feature_importance = data['feature_importance']
        results.hyperparameters = data['hyperparameters']
        results.training_info = data['training_info']
        results.timestamp = data['timestamp']
        return results


class ModelComparison:
    """Compares results across multiple models."""
    
    def __init__(self):
        self.results = {}
        
    def add_model(self, model_name: str, results: ModelResults):
        """Adds model results for comparison."""
        self.results[model_name] = results
        
    def compare_metrics(self, metric_name: str = 'roc_auc'):
        """Compares a specific metric across models."""
        comparison = {}
        for model_name, results in self.results.items():
            if metric_name in results.metrics:
                comparison[model_name] = results.metrics[metric_name]
        return comparison
    
    def compare_all_metrics(self) -> Dict[str, Dict]:
        """Get all metrics across all models for comparison."""
        all_model_names = set()
        for results in self.results.values():
            all_model_names.update(results.metrics.keys())
        
        comparison = {}
        for metric in all_model_names:
            comparison[metric] = self.compare_metrics(metric)
        return comparison
    
    def get_top_features(self, model_name: str, top_n: int = 10) -> List[Dict]:
        """Gets top N features for a specific model."""
        if model_name not in self.results:
            return []
        return self.results[model_name].feature_importance[:top_n]
    
    def print_summary(self):
        """Prints summary comparison of all models."""
        print("\n" + "="*80)
        print("MODEL COMPARISON SUMMARY")
        print("="*80)
        
        if not self.results:
            print("No results to compare.")
            return
        
        # Get all metrics
        all_metrics = self.compare_all_metrics()
        
        # Print header
        model_names = list(self.results.keys())
        header = f"{'Metric':<20}"
        for model_name in model_names:
            header += f" | {model_name:<15}"
        print(header)
        print("-" * len(header))
        
        # Print each metric
        for metric, values in sorted(all_metrics.items()):
            row = f"{metric:<20}"
            for model_name in model_names:
                val = values.get(model_name, "N/A")
                if isinstance(val, float):
                    row += f" | {val:<15.4f}"
                else:
                    row += f" | {str(val):<15}"
            print(row)
        
        print("="*80 + "\n")
    
    def save_comparison(self, filepath: str):
        """Saves comparison to JSON file."""
        comparison_data = {
            'timestamp': datetime.now().isoformat(),
            'models': {
                model_name: results.to_dict() 
                for model_name, results in self.results.items()
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        print(f" Comparison saved to {filepath}")


# ============================================================================
# STANDARD METRIC DEFINITIONS
# ============================================================================

def get_metric_definitions() -> Dict[str, str]:
    """Gets descriptions of all evaluation metrics."""
    return {
        'roc_auc': 'Area Under the Receiver Operating Characteristic Curve (0-1, higher is better)',
        'pr_auc': 'Area Under the Precision-Recall Curve',
        'accuracy': 'Overall correctness of the model',
        'precision': 'Positive Predictive Value: % of predicted AKI that are correct',
        'recall': 'Sensitivity - % of actual AKI detected',
        'f1': 'Harmonic mean of precision and recall',
        'specificity': 'True Negative Rate - % of non-AKI correctly identified',
        'sensitivity': 'True Positive Rate - same as recall'
    }
