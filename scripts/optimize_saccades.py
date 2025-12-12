#!/usr/bin/env python3
"""
Optuna optimization for EMNIST letters with saccades.
Goal: Improve beyond 69.26% baseline accuracy.
"""

import optuna
import subprocess
import json
import re
import os
import sys
from pathlib import Path

# Configuration
BASE_CONFIG = "configs/emnist_letters_saccades_position.json"
TEMP_CONFIG = "configs/emnist_letters_saccades_opt.json"
EXECUTABLE = "build/emnist_letters_v1"
BASELINE_ACCURACY = 69.26
NUM_TRIALS = 50
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
            print("  Last 50 lines of output:")
            print('\n'.join(output.split('\n')[-50:]))
            return 0.0
            
    except subprocess.TimeoutExpired:
        print(f"  → TIMEOUT after {TIMEOUT_SECONDS}s")
        return 0.0
    except Exception as e:
        print(f"  → ERROR: {e}")
        return 0.0

def objective(trial):
    """Optuna objective function."""
    
    # Load base configuration
    with open(BASE_CONFIG, 'r') as f:
        config = json.load(f)
    
    # Disable position encoding (we know it doesn't help)
    config['position_encoding']['enabled'] = False
    
    # Sample hyperparameters
    # Based on previous optimization, these are the most impactful parameters
    
    # Temporal window (critical for saccades)
    config['neuron']['window_size_ms'] = trial.suggest_int('window_size_ms', 400, 600, step=50)
    
    # Similarity threshold
    config['neuron']['similarity_threshold'] = trial.suggest_float('similarity_threshold', 0.85, 0.95, step=0.01)
    
    # Max patterns per neuron
    config['neuron']['max_patterns'] = trial.suggest_int('max_patterns', 300, 700, step=100)
    
    # Layer 2/3 neurons (integration layer)
    config['architecture']['layers']['layer23_neurons'] = trial.suggest_int('layer23_neurons', 256, 512, step=64)
    
    # Layer 5 neurons (output layer)
    config['architecture']['layers']['layer5_neurons'] = trial.suggest_int('layer5_neurons', 64, 128, step=16)
    
    # Output neurons per class
    config['architecture']['output']['neurons_per_class'] = trial.suggest_int('neurons_per_class', 15, 30, step=5)
    
    # Lateral connectivity
    config['architecture']['connectivity']['lateral_connectivity'] = trial.suggest_float('lateral_connectivity', 0.1, 0.3, step=0.05)
    
    # Recurrent connectivity (within L2/3)
    config['architecture']['connectivity']['recurrent_connectivity'] = trial.suggest_float('recurrent_connectivity', 0.1, 0.25, step=0.05)
    
    # Recurrent weight
    config['architecture']['connectivity']['recurrent_weight'] = trial.suggest_float('recurrent_weight', 0.3, 0.6, step=0.1)
    
    # Gabor threshold
    config['architecture']['gabor']['threshold'] = trial.suggest_float('gabor_threshold', 0.05, 0.2, step=0.05)
    
    # Fixation duration
    config['saccades']['fixation_duration_ms'] = trial.suggest_int('fixation_duration_ms', 80, 120, step=10)
    
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
    print()
    
    # Run experiment
    accuracy = run_experiment(TEMP_CONFIG)
    
    # Report improvement over baseline
    improvement = accuracy - BASELINE_ACCURACY
    print(f"\nResult: {accuracy:.2f}% (baseline: {BASELINE_ACCURACY:.2f}%, improvement: {improvement:+.2f}%)")
    
    return accuracy

def main():
    """Run optimization study."""
    
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
    print("EMNIST Letters Saccades Optimization")
    print("="*80)
    print(f"Baseline accuracy: {BASELINE_ACCURACY:.2f}%")
    print(f"Number of trials: {NUM_TRIALS}")
    print(f"Timeout per trial: {TIMEOUT_SECONDS}s ({TIMEOUT_SECONDS/60:.0f} minutes)")
    print(f"Base config: {BASE_CONFIG}")
    print(f"Temp config: {TEMP_CONFIG}")
    print(f"Executable: {EXECUTABLE}")
    print("="*80)
    print()
    
    # Create study
    study = optuna.create_study(
        study_name="emnist_saccades_optimization",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    )
    
    # Run optimization
    study.optimize(objective, n_trials=NUM_TRIALS, show_progress_bar=True)
    
    # Print results
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print(f"\nBest accuracy: {study.best_value:.2f}%")
    print(f"Improvement over baseline: {study.best_value - BASELINE_ACCURACY:+.2f}%")
    print(f"\nBest parameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Save best config
    best_config_path = "configs/emnist_letters_saccades_best.json"
    with open(BASE_CONFIG, 'r') as f:
        best_config = json.load(f)
    
    # Apply best parameters
    best_config['position_encoding']['enabled'] = False
    best_config['neuron']['window_size_ms'] = study.best_params['window_size_ms']
    best_config['neuron']['similarity_threshold'] = study.best_params['similarity_threshold']
    best_config['neuron']['max_patterns'] = study.best_params['max_patterns']
    best_config['architecture']['layers']['layer23_neurons'] = study.best_params['layer23_neurons']
    best_config['architecture']['layers']['layer5_neurons'] = study.best_params['layer5_neurons']
    best_config['architecture']['output']['neurons_per_class'] = study.best_params['neurons_per_class']
    best_config['architecture']['connectivity']['lateral_connectivity'] = study.best_params['lateral_connectivity']
    best_config['architecture']['connectivity']['recurrent_connectivity'] = study.best_params['recurrent_connectivity']
    best_config['architecture']['connectivity']['recurrent_weight'] = study.best_params['recurrent_weight']
    best_config['architecture']['gabor']['threshold'] = study.best_params['gabor_threshold']
    best_config['saccades']['fixation_duration_ms'] = study.best_params['fixation_duration_ms']
    
    with open(best_config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"\nBest configuration saved to: {best_config_path}")
    
    # Print top 10 trials
    print("\n" + "="*80)
    print("TOP 10 TRIALS")
    print("="*80)
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values('value', ascending=False).head(10)
    print(trials_df[['number', 'value', 'params_window_size_ms', 'params_similarity_threshold', 
                     'params_max_patterns', 'params_layer23_neurons', 'params_neurons_per_class']])
    
    # Save study
    study_path = "optimization_study_saccades.pkl"
    import pickle
    with open(study_path, 'wb') as f:
        pickle.dump(study, f)
    print(f"\nStudy saved to: {study_path}")

if __name__ == "__main__":
    main()

