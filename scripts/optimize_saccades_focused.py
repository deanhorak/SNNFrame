#!/usr/bin/env python3
"""
Focused Optuna optimization based on insights from first 50 trials.
Goal: Push beyond 72% accuracy by focusing on the most impactful parameters.
"""

import optuna
import subprocess
import json
import re
import os
import sys
from pathlib import Path

# Configuration
BASE_CONFIG = "configs/emnist_letters_saccades_best.json"
TEMP_CONFIG = "configs/emnist_letters_saccades_opt2.json"
EXECUTABLE = "build/emnist_letters_v1"
BASELINE_ACCURACY = 71.91  # Best from previous optimization
NUM_TRIALS = 30
TIMEOUT_SECONDS = 7200  # 2 hours per trial

def run_experiment(config_path):
    """Run experiment and extract accuracy."""
    try:
        # Run with timeout
        result = subprocess.run(
            [EXECUTABLE, config_path],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=os.getcwd(),
            env={**os.environ, 'LD_LIBRARY_PATH': '/usr/lib/x86_64-linux-gnu'}
        )
        
        # Extract accuracy from output
        output = result.stdout + result.stderr
        
        # Look for "Overall accuracy: XX.XX%"
        match = re.search(r'Overall accuracy:\s+([\d.]+)%', output)
        if match:
            accuracy = float(match.group(1))
            print(f"  → Accuracy: {accuracy:.2f}%")
            return accuracy
        else:
            print("  → ERROR: Could not parse accuracy from output")
            return 0.0
            
    except subprocess.TimeoutExpired:
        print(f"  → TIMEOUT after {TIMEOUT_SECONDS}s")
        return 0.0
    except Exception as e:
        print(f"  → ERROR: {e}")
        return 0.0

def objective(trial):
    """Optuna objective function - FOCUSED on most impactful parameters."""
    
    # Load base configuration (best from previous optimization)
    with open(BASE_CONFIG, 'r') as f:
        config = json.load(f)
    
    # Based on correlation analysis, focus on the MOST IMPACTFUL parameters:
    # 1. neurons_per_class: -0.759 correlation (STRONG negative) - FIXED at 15
    # 2. gabor_threshold: +0.463 correlation (STRONG positive) - OPTIMIZE
    # 3. recurrent_weight: -0.407 correlation (STRONG negative) - OPTIMIZE
    # 4. similarity_threshold: +0.387 correlation (STRONG positive) - OPTIMIZE
    # 5. layer23_neurons: +0.302 correlation (STRONG positive) - OPTIMIZE
    
    # FIXED parameters (best values from analysis)
    config['architecture']['output']['neurons_per_class'] = 15  # ALL top 10 trials used 15
    config['neuron']['max_patterns'] = 500  # 8/10 top trials used 500
    config['neuron']['window_size_ms'] = 500  # 6/10 top trials used 500
    
    # OPTIMIZE: Gabor threshold (STRONG positive correlation)
    # Top 10 mean: 0.18, Bottom 10 mean: 0.13
    # Focus on higher values
    config['architecture']['gabor']['threshold'] = trial.suggest_float('gabor_threshold', 0.15, 0.25, step=0.01)
    
    # OPTIMIZE: Recurrent weight (STRONG negative correlation)
    # Top 10 mean: 0.40, Bottom 10 mean: 0.49
    # Focus on lower values
    config['architecture']['connectivity']['recurrent_weight'] = trial.suggest_float('recurrent_weight', 0.3, 0.5, step=0.05)
    
    # OPTIMIZE: Similarity threshold (STRONG positive correlation)
    # Top 10 mean: 0.93, Bottom 10 mean: 0.91
    # Focus on higher values
    config['neuron']['similarity_threshold'] = trial.suggest_float('similarity_threshold', 0.90, 0.95, step=0.01)
    
    # OPTIMIZE: Layer 2/3 neurons (STRONG positive correlation)
    # Top 10 mean: 454, Bottom 10 mean: 410
    # 9/10 top trials used 448, so focus around that
    config['architecture']['layers']['layer23_neurons'] = trial.suggest_int('layer23_neurons', 384, 512, step=64)
    
    # OPTIMIZE: Layer 5 neurons (weak correlation, but still optimize)
    # Top 10 had variety: 80, 96, 112
    config['architecture']['layers']['layer5_neurons'] = trial.suggest_int('layer5_neurons', 80, 128, step=16)
    
    # OPTIMIZE: Fixation duration (moderate positive correlation)
    # Top 10 mean: 104, Bottom 10 mean: 101
    # 6/10 top trials used 100ms
    config['saccades']['fixation_duration_ms'] = trial.suggest_int('fixation_duration_ms', 100, 120, step=5)
    
    # OPTIMIZE: Recurrent connectivity (moderate positive correlation)
    config['architecture']['connectivity']['recurrent_connectivity'] = trial.suggest_float('recurrent_connectivity', 0.15, 0.25, step=0.05)
    
    # OPTIMIZE: Lateral connectivity (weak correlation, but include)
    config['architecture']['connectivity']['lateral_connectivity'] = trial.suggest_float('lateral_connectivity', 0.15, 0.25, step=0.05)
    
    # Save temporary config
    with open(TEMP_CONFIG, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Print trial info
    print(f"\n{'='*80}")
    print(f"Trial {trial.number + 1}/{NUM_TRIALS}")
    print(f"{'='*80}")
    print("Parameters:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
    print("\nFixed parameters:")
    print(f"  neurons_per_class: 15")
    print(f"  max_patterns: 500")
    print(f"  window_size_ms: 500")
    print()
    
    # Run experiment
    accuracy = run_experiment(TEMP_CONFIG)
    
    # Report improvement over baseline
    improvement = accuracy - BASELINE_ACCURACY
    print(f"\nResult: {accuracy:.2f}% (baseline: {BASELINE_ACCURACY:.2f}%, improvement: {improvement:+.2f}%)")
    
    return accuracy

def main():
    """Run focused optimization study."""
    
    # Check if executable exists
    if not os.path.exists(EXECUTABLE):
        print(f"ERROR: Executable not found: {EXECUTABLE}")
        print("Please compile first: cd build && make emnist_letters_v1")
        sys.exit(1)
    
    # Check if base config exists
    if not os.path.exists(BASE_CONFIG):
        print(f"ERROR: Base config not found: {BASE_CONFIG}")
        sys.exit(1)
    
    print("="*80)
    print("EMNIST Letters Saccades FOCUSED Optimization")
    print("="*80)
    print(f"Baseline accuracy: {BASELINE_ACCURACY:.2f}% (best from previous optimization)")
    print(f"Number of trials: {NUM_TRIALS}")
    print(f"Timeout per trial: {TIMEOUT_SECONDS}s ({TIMEOUT_SECONDS/60:.0f} minutes)")
    print(f"Base config: {BASE_CONFIG}")
    print(f"Temp config: {TEMP_CONFIG}")
    print(f"Executable: {EXECUTABLE}")
    print()
    print("Strategy: Focus on parameters with strongest correlations")
    print("  - neurons_per_class: FIXED at 15 (all top 10 trials)")
    print("  - gabor_threshold: OPTIMIZE (0.15-0.25, +0.463 correlation)")
    print("  - recurrent_weight: OPTIMIZE (0.3-0.5, -0.407 correlation)")
    print("  - similarity_threshold: OPTIMIZE (0.90-0.95, +0.387 correlation)")
    print("  - layer23_neurons: OPTIMIZE (384-512, +0.302 correlation)")
    print("="*80)
    print()
    
    # Create study
    study = optuna.create_study(
        study_name="emnist_saccades_focused_optimization",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=43),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0)
    )
    
    # Run optimization
    study.optimize(objective, n_trials=NUM_TRIALS, show_progress_bar=True)
    
    # Print results
    print("\n" + "="*80)
    print("FOCUSED OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nBest accuracy: {study.best_value:.2f}%")
    print(f"Previous best: {BASELINE_ACCURACY:.2f}%")
    print(f"Improvement: {study.best_value - BASELINE_ACCURACY:+.2f}%")
    print(f"\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best config if improved
    if study.best_value > BASELINE_ACCURACY:
        best_config_path = "configs/emnist_letters_saccades_best_v2.json"
        with open(BASE_CONFIG, 'r') as f:
            best_config = json.load(f)
        
        # Apply best parameters
        best_config['architecture']['gabor']['threshold'] = study.best_params['gabor_threshold']
        best_config['architecture']['connectivity']['recurrent_weight'] = study.best_params['recurrent_weight']
        best_config['neuron']['similarity_threshold'] = study.best_params['similarity_threshold']
        best_config['architecture']['layers']['layer23_neurons'] = study.best_params['layer23_neurons']
        best_config['architecture']['layers']['layer5_neurons'] = study.best_params['layer5_neurons']
        best_config['saccades']['fixation_duration_ms'] = study.best_params['fixation_duration_ms']
        best_config['architecture']['connectivity']['recurrent_connectivity'] = study.best_params['recurrent_connectivity']
        best_config['architecture']['connectivity']['lateral_connectivity'] = study.best_params['lateral_connectivity']
        
        with open(best_config_path, 'w') as f:
            json.dump(best_config, f, indent=2)
        
        print(f"\n✅ NEW BEST configuration saved to: {best_config_path}")
    else:
        print(f"\n❌ No improvement over previous best ({BASELINE_ACCURACY:.2f}%)")
    
    # Save study
    study_path = "optimization_study_saccades_focused.pkl"
    import pickle
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    print(f"\nStudy saved to: {study_path}")

if __name__ == "__main__":
    main()

