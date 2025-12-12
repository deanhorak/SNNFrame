#!/usr/bin/env python3
"""
Generate SONATA network file for MNIST experiment with 5×5 grid.

This script creates an HDF5 file in SONATA format containing:
- 200 neurons (5×5 grid × 8 orientations × 1 neuron per feature)
- Node properties: window_size_ms, similarity_threshold, max_patterns,
                   region_row, region_col, orientation, neuron_index
- Metadata: grid_size, num_orientations, neurons_per_feature
"""

import h5py
import numpy as np
import os

# Network parameters
GRID_SIZE = 5
NUM_ORIENTATIONS = 8
NEURONS_PER_FEATURE = 1
REGION_SIZE = 5  # 5×5 pixels per region

# Neuron parameters
WINDOW_SIZE_MS = 200.0
SIMILARITY_THRESHOLD = 0.7
MAX_PATTERNS = 100

# Calculate total neurons
NUM_REGIONS = GRID_SIZE * GRID_SIZE
NUM_NEURONS = NUM_REGIONS * NUM_ORIENTATIONS * NEURONS_PER_FEATURE

print(f"Generating SONATA network file for 5×5 MNIST experiment")
print(f"  Grid: {GRID_SIZE}×{GRID_SIZE} = {NUM_REGIONS} regions")
print(f"  Region size: {REGION_SIZE}×{REGION_SIZE} pixels")
print(f"  Orientations: {NUM_ORIENTATIONS}")
print(f"  Neurons per feature: {NEURONS_PER_FEATURE}")
print(f"  Total neurons: {NUM_NEURONS}")

# Create output directory if it doesn't exist
os.makedirs('configs', exist_ok=True)

# Create HDF5 file
output_file = 'configs/mnist_network_5x5.h5'
print(f"\nCreating {output_file}...")

with h5py.File(output_file, 'w') as f:
    # Create nodes group
    nodes_group = f.create_group('nodes')
    population_group = nodes_group.create_group('layer1_neurons')
    
    # Create node_type_id (all neurons are the same type)
    node_type_id = np.zeros(NUM_NEURONS, dtype=np.int32)
    population_group.create_dataset('node_type_id', data=node_type_id)
    
    # Create 0 group for dynamic attributes
    attr_group = population_group.create_group('0')
    
    # Initialize arrays for neuron properties
    window_sizes = np.full(NUM_NEURONS, WINDOW_SIZE_MS, dtype=np.float64)
    thresholds = np.full(NUM_NEURONS, SIMILARITY_THRESHOLD, dtype=np.float64)
    max_patterns = np.full(NUM_NEURONS, MAX_PATTERNS, dtype=np.uint64)
    region_rows = np.zeros(NUM_NEURONS, dtype=np.int32)
    region_cols = np.zeros(NUM_NEURONS, dtype=np.int32)
    orientations = np.zeros(NUM_NEURONS, dtype=np.int32)
    neuron_indices = np.zeros(NUM_NEURONS, dtype=np.int32)
    
    # Fill in neuron properties
    neuron_id = 0
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            for orientation in range(NUM_ORIENTATIONS):
                for neuron_idx in range(NEURONS_PER_FEATURE):
                    region_rows[neuron_id] = row
                    region_cols[neuron_id] = col
                    orientations[neuron_id] = orientation
                    neuron_indices[neuron_id] = neuron_idx
                    neuron_id += 1
    
    # Create datasets for neuron properties
    attr_group.create_dataset('window_size_ms', data=window_sizes)
    attr_group.create_dataset('similarity_threshold', data=thresholds)
    attr_group.create_dataset('max_patterns', data=max_patterns)
    attr_group.create_dataset('region_row', data=region_rows)
    attr_group.create_dataset('region_col', data=region_cols)
    attr_group.create_dataset('orientation', data=orientations)
    attr_group.create_dataset('neuron_index', data=neuron_indices)
    
    # Add metadata as attributes
    population_group.attrs['grid_size'] = GRID_SIZE
    population_group.attrs['region_size'] = REGION_SIZE
    population_group.attrs['num_orientations'] = NUM_ORIENTATIONS
    population_group.attrs['neurons_per_feature'] = NEURONS_PER_FEATURE
    population_group.attrs['total_neurons'] = NUM_NEURONS
    
    print(f"✓ Created node population 'layer1_neurons' with {NUM_NEURONS} neurons")
    print(f"  Properties: window_size_ms, similarity_threshold, max_patterns,")
    print(f"              region_row, region_col, orientation, neuron_index")
    print(f"  Metadata: grid_size={GRID_SIZE}, region_size={REGION_SIZE},")
    print(f"            num_orientations={NUM_ORIENTATIONS}, neurons_per_feature={NEURONS_PER_FEATURE}")

print(f"\n✓ Successfully created {output_file}")
print(f"\nTo use this network:")
print(f"  cd build")
print(f"  ./mnist_optimized ../configs/mnist_config_5x5.json")

