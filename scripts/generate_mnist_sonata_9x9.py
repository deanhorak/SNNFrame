#!/usr/bin/env python3
"""
Generate SONATA network file for MNIST experiment with 9×9 grid.

This script creates an HDF5 file in SONATA format containing:
- 648 neurons (9×9 grid × 8 orientations × 1 neuron per feature)
- Node properties: window_size_ms, similarity_threshold, max_patterns,
                   region_row, region_col, orientation, neuron_index
- Metadata: grid_size, num_orientations, neurons_per_feature
"""

import h5py
import numpy as np
import os

# Network parameters
GRID_SIZE = 9
NUM_ORIENTATIONS = 8
NEURONS_PER_FEATURE = 1
REGION_SIZE = 3  # ~3.1×3.1 pixels per region (28/9)

# Neuron parameters
WINDOW_SIZE_MS = 200.0
SIMILARITY_THRESHOLD = 0.7
MAX_PATTERNS = 100

# Calculate total neurons
NUM_REGIONS = GRID_SIZE * GRID_SIZE
NUM_NEURONS = NUM_REGIONS * NUM_ORIENTATIONS * NEURONS_PER_FEATURE

print(f"Generating SONATA network file for 9×9 MNIST experiment")
print(f"  Grid: {GRID_SIZE}×{GRID_SIZE} = {NUM_REGIONS} regions")
print(f"  Region size: ~{28/GRID_SIZE:.1f}×{28/GRID_SIZE:.1f} pixels")
print(f"  Orientations: {NUM_ORIENTATIONS}")
print(f"  Neurons per feature: {NEURONS_PER_FEATURE}")
print(f"  Total neurons: {NUM_NEURONS}")

# Create output directory if it doesn't exist
os.makedirs('configs', exist_ok=True)

# Create HDF5 file
output_file = 'configs/mnist_network_9x9.h5'
print(f"\nCreating {output_file}...")

with h5py.File(output_file, 'w') as f:
    # Create nodes group
    nodes_group = f.create_group('nodes')
    population_group = nodes_group.create_group('layer1_neurons')
    
    # Create node_type_id (all neurons are the same type)
    node_type_id = np.zeros(NUM_NEURONS, dtype=np.int32)
    population_group.create_dataset('node_type_id', data=node_type_id)
    
    # Create node_group_id (group by region)
    node_group_id = np.zeros(NUM_NEURONS, dtype=np.int32)
    for i in range(NUM_NEURONS):
        region_id = i // (NUM_ORIENTATIONS * NEURONS_PER_FEATURE)
        node_group_id[i] = region_id
    population_group.create_dataset('node_group_id', data=node_group_id)
    
    # Create node group 0 with neuron parameters
    group_0 = population_group.create_group('0')
    
    # Window size (ms)
    window_sizes = np.full(NUM_NEURONS, WINDOW_SIZE_MS, dtype=np.float64)
    group_0.create_dataset('window_size_ms', data=window_sizes)
    
    # Similarity threshold
    thresholds = np.full(NUM_NEURONS, SIMILARITY_THRESHOLD, dtype=np.float64)
    group_0.create_dataset('similarity_threshold', data=thresholds)
    
    # Max patterns
    max_patterns = np.full(NUM_NEURONS, MAX_PATTERNS, dtype=np.int32)
    group_0.create_dataset('max_patterns', data=max_patterns)
    
    # Spatial organization: region_row, region_col, orientation, neuron_index
    region_rows = np.zeros(NUM_NEURONS, dtype=np.int32)
    region_cols = np.zeros(NUM_NEURONS, dtype=np.int32)
    orientations = np.zeros(NUM_NEURONS, dtype=np.int32)
    neuron_indices = np.zeros(NUM_NEURONS, dtype=np.int32)
    
    neuron_id = 0
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            for orient in range(NUM_ORIENTATIONS):
                for neuron_idx in range(NEURONS_PER_FEATURE):
                    region_rows[neuron_id] = row
                    region_cols[neuron_id] = col
                    orientations[neuron_id] = orient
                    neuron_indices[neuron_id] = neuron_idx
                    neuron_id += 1
    
    group_0.create_dataset('region_row', data=region_rows)
    group_0.create_dataset('region_col', data=region_cols)
    group_0.create_dataset('orientation', data=orientations)
    group_0.create_dataset('neuron_index', data=neuron_indices)
    
    print(f"✓ Created {NUM_NEURONS} neurons")
    print(f"  - Region rows: 0-{GRID_SIZE-1}")
    print(f"  - Region cols: 0-{GRID_SIZE-1}")
    print(f"  - Orientations: 0-{NUM_ORIENTATIONS-1}")
    print(f"  - Neurons per feature: {NEURONS_PER_FEATURE}")

print(f"\n✓ Successfully created {output_file}")
print(f"\nFile structure:")
print(f"  /nodes/layer1_neurons/")
print(f"    - node_type_id: {NUM_NEURONS} nodes")
print(f"    - node_group_id: region assignments")
print(f"    - 0/window_size_ms: {WINDOW_SIZE_MS} ms")
print(f"    - 0/similarity_threshold: {SIMILARITY_THRESHOLD}")
print(f"    - 0/max_patterns: {MAX_PATTERNS}")
print(f"    - 0/region_row: spatial row index (0-{GRID_SIZE-1})")
print(f"    - 0/region_col: spatial column index (0-{GRID_SIZE-1})")
print(f"    - 0/orientation: orientation index (0-{NUM_ORIENTATIONS-1})")
print(f"    - 0/neuron_index: neuron index within feature (0-{NEURONS_PER_FEATURE-1})")

