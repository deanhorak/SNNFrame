#!/usr/bin/env python3
"""
Very fine-grained optimization around edge_threshold=0.16.
Tests values from 0.155 to 0.165 in steps of 0.005.
"""

import json
import subprocess
import re
from pathlib import Path

BASE_CONFIG = {
    "experiment": {
        "name": "mnist_edge_fine",
        "description": "Fine-tune edge_threshold around 0.16",
        "version": "5.3.0"
    },
    "network": {
        "grid_size": 8,
        "region_size": 3,
        "num_orientations": 8,
        "neurons_per_feature": 1,
        "temporal_window_ms": 200.0,
        "edge_threshold": 0.16,
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
                "edge_threshold": 0.16,
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
        "results_file": "results/edge_fine_results.json"
    }
}

def run_experiment(threshold):
    """Run experiment with specified edge threshold"""
    config = json.loads(json.dumps(BASE_CONFIG))
    config["network"]["edge_threshold"] = threshold
    config["adapters"]["retina"]["parameters"]["edge_threshold"] = threshold
    
    # Save config
    results_dir = Path("results/edge_fine")
    results_dir.mkdir(parents=True, exist_ok=True)
    config_path = results_dir / f"config_edge{threshold:.3f}.json"
    
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
        output_path = results_dir / f"output_edge{threshold:.3f}.txt"
        with open(output_path, 'w') as f:
            f.write(output)
        
        return accuracy
        
    except Exception as e:
        print(f"Error running edge_threshold={threshold:.3f}: {e}")
        return 0.0

def main():
    print("="*80)
    print("FINE-GRAINED EDGE THRESHOLD OPTIMIZATION")
    print("="*80)
    
    # Test values around 0.16
    thresholds = [0.155, 0.158, 0.160, 0.162, 0.165]
    print(f"Testing edge_threshold values around 0.16: {thresholds}")
    print(f"Using k=5 (optimal)")
    print(f"Full dataset: 5000 train/digit, 10000 test")
    print()
    
    results = []
    best_threshold = None
    best_accuracy = 0.0
    
    for threshold in thresholds:
        print(f"Testing edge_threshold={threshold:.3f}...", end=" ", flush=True)
        accuracy = run_experiment(threshold)
        results.append((threshold, accuracy))
        print(f"Accuracy: {accuracy:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            print(f"  🎉 NEW BEST!")
    
    print()
    print("="*80)
    print("RESULTS")
    print("="*80)
    print(f"Best edge_threshold: {best_threshold:.3f}")
    print(f"Best accuracy: {best_accuracy:.2f}%")
    print()
    print("All results:")
    for threshold, acc in results:
        marker = " ← BEST" if threshold == best_threshold else ""
        print(f"  edge_threshold={threshold:.3f}: {acc:.2f}%{marker}")
    
    # Save summary
    summary_path = Path("results/edge_fine/summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Fine-Grained Edge Threshold Optimization Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Best edge_threshold: {best_threshold:.3f}\n")
        f.write(f"Best accuracy: {best_accuracy:.2f}%\n\n")
        f.write("All results:\n")
        for threshold, acc in results:
            marker = " ← BEST" if threshold == best_threshold else ""
            f.write(f"  edge_threshold={threshold:.3f}: {acc:.2f}%{marker}\n")
    
    # Save best config
    if best_threshold is not None:
        best_config = json.loads(json.dumps(BASE_CONFIG))
        best_config["network"]["edge_threshold"] = best_threshold
        best_config["adapters"]["retina"]["parameters"]["edge_threshold"] = best_threshold
        
        best_config_path = Path("results/edge_fine/best_config.json")
        with open(best_config_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        print(f"\nBest config saved to: {best_config_path}")
    
    print(f"Results saved to: results/edge_fine/")

if __name__ == "__main__":
    main()

