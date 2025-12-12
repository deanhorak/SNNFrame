#!/usr/bin/env python3
"""
Analyze the Optuna optimization study to understand parameter correlations and importance.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Load the study
study_path = "optimization_study_saccades.pkl"

print("="*80)
print("OPTIMIZATION STUDY ANALYSIS")
print("="*80)
print()

# Load study
with open(study_path, 'rb') as f:
    study = pickle.load(f)

print(f"Study name: {study.study_name}")
print(f"Direction: {study.direction.name}")
print(f"Number of trials: {len(study.trials)}")
print(f"Best value: {study.best_value:.2f}%")
print()

# Convert to DataFrame
df = study.trials_dataframe()

# Basic statistics
print("="*80)
print("ACCURACY STATISTICS")
print("="*80)
print(f"Mean accuracy: {df['value'].mean():.2f}%")
print(f"Std deviation: {df['value'].std():.2f}%")
print(f"Min accuracy: {df['value'].min():.2f}%")
print(f"Max accuracy: {df['value'].max():.2f}%")
print(f"Median accuracy: {df['value'].median():.2f}%")
print()

# Quartiles
print("Accuracy quartiles:")
print(f"  25th percentile: {df['value'].quantile(0.25):.2f}%")
print(f"  50th percentile: {df['value'].quantile(0.50):.2f}%")
print(f"  75th percentile: {df['value'].quantile(0.75):.2f}%")
print()

# Top 10 trials
print("="*80)
print("TOP 10 TRIALS")
print("="*80)
top10 = df.nlargest(10, 'value')
print(top10[['number', 'value', 'params_window_size_ms', 'params_similarity_threshold', 
             'params_max_patterns', 'params_layer23_neurons', 'params_layer5_neurons',
             'params_neurons_per_class', 'params_gabor_threshold']].to_string(index=False))
print()

# Bottom 10 trials
print("="*80)
print("BOTTOM 10 TRIALS")
print("="*80)
bottom10 = df.nsmallest(10, 'value')
print(bottom10[['number', 'value', 'params_window_size_ms', 'params_similarity_threshold', 
                'params_max_patterns', 'params_layer23_neurons', 'params_layer5_neurons',
                'params_neurons_per_class', 'params_gabor_threshold']].to_string(index=False))
print()

# Parameter value distributions in top 10 vs bottom 10
print("="*80)
print("PARAMETER DISTRIBUTIONS: TOP 10 vs BOTTOM 10")
print("="*80)

param_cols = [col for col in df.columns if col.startswith('params_')]

for param_col in param_cols:
    param_name = param_col.replace('params_', '')
    top_mean = top10[param_col].mean()
    bottom_mean = bottom10[param_col].mean()
    all_mean = df[param_col].mean()
    
    print(f"\n{param_name}:")
    print(f"  Top 10 mean:    {top_mean:.2f}")
    print(f"  Bottom 10 mean: {bottom_mean:.2f}")
    print(f"  All trials mean: {all_mean:.2f}")
    print(f"  Difference (top-bottom): {top_mean - bottom_mean:+.2f}")

print()

# Correlation analysis
print("="*80)
print("PARAMETER CORRELATIONS WITH ACCURACY")
print("="*80)

correlations = []
for param_col in param_cols:
    corr = df[param_col].corr(df['value'])
    param_name = param_col.replace('params_', '')
    correlations.append((param_name, corr))

# Sort by absolute correlation
correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print("\nParameters ranked by correlation strength:")
for param_name, corr in correlations:
    direction = "↑" if corr > 0 else "↓"
    strength = "STRONG" if abs(corr) > 0.3 else "MODERATE" if abs(corr) > 0.15 else "WEAK"
    print(f"  {param_name:30s} {corr:+.3f} {direction} {strength}")

print()

# Most common values in top 10
print("="*80)
print("MOST COMMON VALUES IN TOP 10 TRIALS")
print("="*80)

for param_col in param_cols:
    param_name = param_col.replace('params_', '')
    value_counts = top10[param_col].value_counts()
    
    if len(value_counts) > 0:
        most_common_value = value_counts.index[0]
        count = value_counts.iloc[0]
        print(f"\n{param_name}:")
        print(f"  Most common: {most_common_value} (appears in {count}/10 top trials)")
        
        if len(value_counts) > 1:
            print(f"  Distribution:")
            for val, cnt in value_counts.items():
                print(f"    {val}: {cnt} trials")

print()

# Accuracy progression over time
print("="*80)
print("ACCURACY PROGRESSION")
print("="*80)

# Calculate running best
df_sorted = df.sort_values('number')
df_sorted['running_best'] = df_sorted['value'].cummax()

print("\nMilestones (when new best was found):")
prev_best = 0
for idx, row in df_sorted.iterrows():
    if row['running_best'] > prev_best:
        print(f"  Trial {int(row['number']):2d}: {row['value']:.2f}% (improvement: +{row['value'] - prev_best:.2f}%)")
        prev_best = row['running_best']

print()

# Improvement over baseline
baseline = 69.26
print(f"Baseline accuracy: {baseline:.2f}%")
print(f"Best accuracy: {study.best_value:.2f}%")
print(f"Absolute improvement: +{study.best_value - baseline:.2f}%")
print(f"Relative improvement: +{((study.best_value - baseline) / baseline * 100):.2f}%")
print()

# Trials that beat baseline
beat_baseline = df[df['value'] > baseline]
print(f"Trials that beat baseline ({baseline:.2f}%): {len(beat_baseline)}/{len(df)} ({len(beat_baseline)/len(df)*100:.1f}%)")
print()

# Save detailed analysis
output_file = "optimization_analysis.txt"
with open(output_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("DETAILED OPTIMIZATION ANALYSIS\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Study: {study.study_name}\n")
    f.write(f"Trials: {len(study.trials)}\n")
    f.write(f"Best accuracy: {study.best_value:.2f}%\n")
    f.write(f"Baseline: {baseline:.2f}%\n")
    f.write(f"Improvement: +{study.best_value - baseline:.2f}%\n\n")
    
    f.write("="*80 + "\n")
    f.write("BEST PARAMETERS\n")
    f.write("="*80 + "\n\n")
    for key, value in study.best_params.items():
        f.write(f"{key}: {value}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("ALL TRIALS (sorted by accuracy)\n")
    f.write("="*80 + "\n\n")
    
    df_sorted_by_acc = df.sort_values('value', ascending=False)
    f.write(df_sorted_by_acc.to_string())

print(f"Detailed analysis saved to: {output_file}")
print()

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)

