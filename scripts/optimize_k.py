#!/usr/bin/env python3
"""
Quick optimization of k_neighbors parameter only.
Tests k values from 1 to 15 to find the optimal value.
"""

import json
import subprocess
import re
from pathlib import Path

BASE_CONFIG = {
    "experiment": {
        "name": "mnist_k_optimization",
        "description": "Optimize k_neighbors parameter",
        "version": "5.1.0"
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
        "examples_per_digit": 5000,
        "test_images": 10000,
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
        "results_file": "results/k_optimization_results.json"
    }
}

def run_experiment(k_value):
    """Run experiment with specified k value"""
    config = json.loads(json.dumps(BASE_CONFIG))
    config["classification"]["k_neighbors"] = k_value
    
    # Save config
    results_dir = Path("results/k_optimization")
    results_dir.mkdir(parents=True, exist_ok=True)
    config_path = results_dir / f"config_k{k_value}.json"
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Run experiment
    cmd = [
        "bash", "-c",
        f"cd build && "
        f"LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH "
        f"SPDLOG_LEVEL=error "
        f"./mnist_classification_strategies ../{config_path} 2>&1"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout
        
        # Parse accuracy
        accuracy_match = re.search(r'Overall Accuracy: ([\d.]+)%', output)
        if accuracy_match:
            accuracy = float(accuracy_match.group(1))
        else:
            accuracy = 0.0
        
        # Save output
        output_path = results_dir / f"output_k{k_value}.txt"
        with open(output_path, 'w') as f:
            f.write(output)
        
        return accuracy
        
    except Exception as e:
        print(f"Error running k={k_value}: {e}")
        return 0.0

def main():
    print("="*80)
    print("K-NEIGHBORS OPTIMIZATION")
    print("="*80)

    # Test strategic k values: 1, 3, 5, 7, 9, 11
    k_values = [1, 3, 5, 7, 9, 11]
    print(f"Testing k values: {k_values} with full dataset (5000 train/digit, 10000 test)")
    print()

    results = []
    best_k = None
    best_accuracy = 0.0

    for k in k_values:
        print(f"Testing k={k}...", end=" ", flush=True)
        accuracy = run_experiment(k)
        results.append((k, accuracy))
        print(f"Accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
            print(f"  🎉 NEW BEST!")
    
    print()
    print("="*80)
    print("RESULTS")
    print("="*80)
    print(f"Best k: {best_k}")
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print()
    print("All results:")
    for k, acc in results:
        marker = " ← BEST" if k == best_k else ""
        print(f"  k={k:2d}: {acc:.2f}%{marker}")
    
    # Save summary
    summary_path = Path("results/k_optimization/summary.txt")
    with open(summary_path, 'w') as f:
        f.write("K-Neighbors Optimization Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Best k: {best_k}\n")
        f.write(f"Best accuracy: {best_accuracy:.2f}%\n\n")
        f.write("All results:\n")
        for k, acc in results:
            marker = " ← BEST" if k == best_k else ""
            f.write(f"  k={k:2d}: {acc:.2f}%{marker}\n")
    
    print(f"\nResults saved to: results/k_optimization/")

if __name__ == "__main__":
    main()

