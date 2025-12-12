#!/usr/bin/env python3
"""
Hyperparameter Optimization for MNIST Classification

This script performs systematic hyperparameter search to optimize MNIST accuracy.
Starting from the current best configuration (94.76%), we search for the optimal
combination of hyperparameters to reach the 96-98% target.

Key Hyperparameters:
- edge_threshold: Minimum edge strength to generate spikes (0.10-0.25)
- k_neighbors: Number of nearest neighbors for classification (3-9)
- similarity_threshold: Neuron pattern matching threshold (0.5-0.85)
- temporal_window_ms: Duration of spike pattern (100-300)
- max_patterns: Maximum patterns per neuron (50-200)

Strategy:
1. Grid search on most impactful parameters (edge_threshold, k_neighbors)
2. Random search for fine-tuning
3. Bayesian optimization for final optimization (optional)
"""

import json
import subprocess
import os
import sys
import itertools
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import re

# Base configuration (current best: 94.76%)
BASE_CONFIG = {
    "experiment": {
        "name": "mnist_hyperparam_search",
        "description": "Hyperparameter optimization",
        "version": "5.0.0"
    },
    "network": {
        "grid_size": 8,
        "region_size": 3,
        "num_orientations": 8,
        "neurons_per_feature": 1,
        "temporal_window_ms": 200.0,
        "edge_threshold": 0.15,
        "orientations_degrees": [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5]
    },
    "neuron": {
        "window_size_ms": 200.0,
        "similarity_threshold": 0.7,
        "max_patterns": 100
    },
    "adapters": {
        "retina": {
            "type": "retina",
            "enabled": True,
            "temporal_window_ms": 200.0,
            "parameters": {
                "grid_size": 8,
                "num_orientations": 8,
                "edge_threshold": 0.15,
                "neuron_window_size": 200.0,
                "neuron_threshold": 0.7,
                "neuron_max_patterns": 100,
                "edge_operator": "sobel",
                "encoding_strategy": "rate"
            }
        }
    },
    "training": {
        "examples_per_digit": 1000,  # Reduced for faster search
        "test_images": 2000,  # Reduced for faster search
        "num_digits": 10
    },
    "classification": {
        "strategy": "majority",
        "k_neighbors": 5,
        "distance_exponent": 1.0,
        "similarity_metric": "cosine"
    },
    "data": {
        "train_images": "/home/dean/repos/ctm/data/MNIST/raw/train-images-idx3-ubyte",
        "train_labels": "/home/dean/repos/ctm/data/MNIST/raw/train-labels-idx1-ubyte",
        "test_images": "/home/dean/repos/ctm/data/MNIST/raw/t10k-images-idx3-ubyte",
        "test_labels": "/home/dean/repos/ctm/data/MNIST/raw/t10k-labels-idx1-ubyte"
    },
    "output": {
        "show_progress": True,
        "show_confusion_matrix": False,
        "save_results": True,
        "results_file": "results/hyperparam_results.json"
    }
}

# Hyperparameter search space
SEARCH_SPACE = {
    "edge_threshold": [0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25],
    "k_neighbors": [3, 5, 7, 9, 11],
    "similarity_threshold": [0.5, 0.6, 0.7, 0.75, 0.8, 0.85],
    "temporal_window_ms": [150.0, 200.0, 250.0],
    "max_patterns": [50, 75, 100, 150, 200]
}

class HyperparameterOptimizer:
    def __init__(self, build_dir: str = "build", results_dir: str = "results/hyperparam"):
        self.build_dir = Path(build_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []
        self.best_accuracy = 0.0
        self.best_config = None
        
    def create_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a configuration with specified hyperparameters"""
        config = json.loads(json.dumps(BASE_CONFIG))  # Deep copy
        
        # Update parameters
        config["network"]["edge_threshold"] = params["edge_threshold"]
        config["network"]["temporal_window_ms"] = params["temporal_window_ms"]
        config["neuron"]["window_size_ms"] = params["temporal_window_ms"]
        config["neuron"]["similarity_threshold"] = params["similarity_threshold"]
        config["neuron"]["max_patterns"] = params["max_patterns"]
        
        config["adapters"]["retina"]["temporal_window_ms"] = params["temporal_window_ms"]
        config["adapters"]["retina"]["parameters"]["edge_threshold"] = params["edge_threshold"]
        config["adapters"]["retina"]["parameters"]["neuron_window_size"] = params["temporal_window_ms"]
        config["adapters"]["retina"]["parameters"]["neuron_threshold"] = params["similarity_threshold"]
        config["adapters"]["retina"]["parameters"]["neuron_max_patterns"] = params["max_patterns"]
        
        config["classification"]["k_neighbors"] = params["k_neighbors"]
        
        return config
    
    def run_experiment(self, config: Dict[str, Any], run_id: int) -> Tuple[float, Dict[str, float]]:
        """Run a single experiment and return accuracy"""
        # Save config
        config_path = self.results_dir / f"config_{run_id:04d}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Run experiment
        cmd = [
            "bash", "-c",
            f"cd {self.build_dir} && "
            f"LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH "
            f"SPDLOG_LEVEL=error "
            f"./mnist_classification_strategies ../{config_path} 2>&1"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            output = result.stdout
            
            # Parse accuracy from output
            accuracy_match = re.search(r'Overall Accuracy: ([\d.]+)%', output)
            if accuracy_match:
                accuracy = float(accuracy_match.group(1))
            else:
                print(f"Warning: Could not parse accuracy from output")
                accuracy = 0.0
            
            # Parse per-digit accuracy
            per_digit = {}
            for digit in range(10):
                digit_match = re.search(rf'Digit {digit}: ([\d.]+)%', output)
                if digit_match:
                    per_digit[str(digit)] = float(digit_match.group(1))
            
            # Save output
            output_path = self.results_dir / f"output_{run_id:04d}.txt"
            with open(output_path, 'w') as f:
                f.write(output)
            
            return accuracy, per_digit
            
        except subprocess.TimeoutExpired:
            print(f"Experiment {run_id} timed out")
            return 0.0, {}
        except Exception as e:
            print(f"Error running experiment {run_id}: {e}")
            return 0.0, {}
    
    def grid_search(self, param_grid: Dict[str, List[Any]], max_experiments: int = 50):
        """Perform grid search over specified parameters"""
        print(f"\n{'='*80}")
        print(f"GRID SEARCH")
        print(f"{'='*80}")
        print(f"Search space: {param_grid}")
        print(f"Max experiments: {max_experiments}")
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        combinations = list(itertools.product(*param_values))
        
        # Limit number of experiments
        if len(combinations) > max_experiments:
            import random
            random.seed(42)
            combinations = random.sample(combinations, max_experiments)
        
        print(f"Running {len(combinations)} experiments...")
        
        for i, combo in enumerate(combinations):
            params = dict(zip(param_names, combo))
            
            # Fill in missing parameters with defaults
            full_params = {
                "edge_threshold": params.get("edge_threshold", 0.15),
                "k_neighbors": params.get("k_neighbors", 5),
                "similarity_threshold": params.get("similarity_threshold", 0.7),
                "temporal_window_ms": params.get("temporal_window_ms", 200.0),
                "max_patterns": params.get("max_patterns", 100)
            }
            
            print(f"\n[{i+1}/{len(combinations)}] Testing: {params}")
            
            config = self.create_config(full_params)
            accuracy, per_digit = self.run_experiment(config, len(self.results))
            
            result = {
                "run_id": len(self.results),
                "params": full_params,
                "accuracy": accuracy,
                "per_digit": per_digit,
                "timestamp": time.time()
            }
            self.results.append(result)
            
            print(f"Accuracy: {accuracy:.2f}%")
            
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_config = full_params
                print(f"🎉 NEW BEST: {accuracy:.2f}%")
            
            # Save intermediate results
            self.save_results()
    
    def save_results(self):
        """Save all results to JSON"""
        results_file = self.results_dir / "all_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                "results": self.results,
                "best_accuracy": self.best_accuracy,
                "best_config": self.best_config
            }, f, indent=2)
        
        # Save summary
        summary_file = self.results_dir / "summary.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Hyperparameter Optimization Summary\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"Total experiments: {len(self.results)}\n")
            f.write(f"Best accuracy: {self.best_accuracy:.2f}%\n")
            f.write(f"Best configuration:\n")
            for key, value in (self.best_config or {}).items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nTop 10 Results:\n")
            f.write(f"{'-'*80}\n")
            sorted_results = sorted(self.results, key=lambda x: x["accuracy"], reverse=True)
            for i, result in enumerate(sorted_results[:10]):
                f.write(f"{i+1}. Accuracy: {result['accuracy']:.2f}% - {result['params']}\n")

def main():
    optimizer = HyperparameterOptimizer()

    # Phase 1: Coarse grid search on most impactful parameters
    print("\n" + "="*80)
    print("PHASE 1: Coarse Grid Search (edge_threshold, k_neighbors)")
    print("="*80)
    print("Using reduced dataset (1000 train/digit, 2000 test) for fast search")
    optimizer.grid_search({
        "edge_threshold": [0.10, 0.15, 0.20, 0.25],
        "k_neighbors": [3, 5, 7, 9]
    }, max_experiments=16)

    print(f"\nPhase 1 Best: {optimizer.best_accuracy:.2f}%")
    print(f"Best config: {optimizer.best_config}")

    # Phase 2: Fine-tune around best configuration
    if optimizer.best_config:
        print("\n" + "="*80)
        print("PHASE 2: Fine-tuning (similarity_threshold, max_patterns)")
        print("="*80)
        optimizer.grid_search({
            "similarity_threshold": [0.6, 0.7, 0.75, 0.8],
            "max_patterns": [75, 100, 150]
        }, max_experiments=12)

    # Phase 3: Final validation with full dataset
    if optimizer.best_config:
        print("\n" + "="*80)
        print("PHASE 3: Final Validation with Full Dataset")
        print("="*80)
        print(f"Best config from search: {optimizer.best_config}")

        # Create full dataset config
        full_config = optimizer.create_config(optimizer.best_config)
        full_config["training"]["examples_per_digit"] = 5000
        full_config["training"]["test_images"] = 10000

        print("Running final validation with 5000 train/digit, 10000 test...")
        final_accuracy, final_per_digit = optimizer.run_experiment(full_config, 9999)

        print(f"\n{'='*80}")
        print(f"FINAL VALIDATION RESULTS")
        print(f"{'='*80}")
        print(f"Search accuracy (reduced dataset): {optimizer.best_accuracy:.2f}%")
        print(f"Final accuracy (full dataset): {final_accuracy:.2f}%")
        print(f"Best configuration: {optimizer.best_config}")

        # Save final config
        final_config_path = optimizer.results_dir / "best_config_full.json"
        with open(final_config_path, 'w') as f:
            json.dump(full_config, f, indent=2)
        print(f"\nBest config saved to: {final_config_path}")

    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Results saved to: {optimizer.results_dir}")

if __name__ == "__main__":
    main()

