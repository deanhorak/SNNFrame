#!/usr/bin/env python3
"""
Generate SONATA network file for MNIST experiment

This script creates a SONATA-format HDF5 file describing the neural network
structure for the MNIST digit recognition experiment.

Network structure:
- 392 neurons (49 regions × 8 orientations × 1 neuron per feature)
- Each neuron has properties: window_size_ms, similarity_threshold, max_patterns
- Spatial organization: region_row, region_col, orientation

SONATA format specification:
https://github.com/AllenInstitute/sonata/blob/master/docs/SONATA_DEVELOPER_GUIDE.md
"""

import h5py
import numpy as np
import json
import sys
from pathlib import Path

def create_mnist_network(output_file, config_file="configs/mnist_config.json"):
    """
    Create SONATA network file for MNIST experiment
    
    Args:
        output_file: Path to output HDF5 file
        config_file: Path to JSON configuration file
    """
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    grid_size = config['network']['grid_size']
    num_orientations = config['network']['num_orientations']
    neurons_per_feature = config['network']['neurons_per_feature']
    window_size_ms = config['neuron']['window_size_ms']
    similarity_threshold = config['neuron']['similarity_threshold']
    max_patterns = config['neuron']['max_patterns']
    
    num_regions = grid_size * grid_size
    total_neurons = num_regions * num_orientations * neurons_per_feature
    
    print(f"Generating SONATA network file: {output_file}")
    print(f"  Grid size: {grid_size}×{grid_size} = {num_regions} regions")
    print(f"  Orientations: {num_orientations}")
    print(f"  Neurons per feature: {neurons_per_feature}")
    print(f"  Total neurons: {total_neurons}")
    
    # Create HDF5 file
    with h5py.File(output_file, 'w') as f:
        # Create nodes group
        nodes = f.create_group('nodes')
        
        # Create population
        population_name = config['sonata']['node_population']
        pop = nodes.create_group(population_name)
        
        # Create node_type_id dataset (all neurons are the same type)
        node_type_id = np.zeros(total_neurons, dtype=np.int32)
        pop.create_dataset('node_type_id', data=node_type_id)
        
        # Create node_group_id dataset (group by region)
        node_group_id = np.zeros(total_neurons, dtype=np.int32)
        node_group_index = np.zeros(total_neurons, dtype=np.int64)
        
        # Create attribute datasets
        window_sizes = np.full(total_neurons, window_size_ms, dtype=np.float64)
        thresholds = np.full(total_neurons, similarity_threshold, dtype=np.float64)
        max_patterns_arr = np.full(total_neurons, max_patterns, dtype=np.uint64)
        region_rows = np.zeros(total_neurons, dtype=np.int32)
        region_cols = np.zeros(total_neurons, dtype=np.int32)
        orientations = np.zeros(total_neurons, dtype=np.int32)
        neuron_indices = np.zeros(total_neurons, dtype=np.int32)
        
        # Fill in neuron properties
        neuron_id = 0
        for region_row in range(grid_size):
            for region_col in range(grid_size):
                region_id = region_row * grid_size + region_col
                for orientation in range(num_orientations):
                    for neuron_idx in range(neurons_per_feature):
                        node_group_id[neuron_id] = region_id
                        node_group_index[neuron_id] = neuron_id
                        region_rows[neuron_id] = region_row
                        region_cols[neuron_id] = region_col
                        orientations[neuron_id] = orientation
                        neuron_indices[neuron_id] = neuron_idx
                        neuron_id += 1
        
        # Create datasets
        pop.create_dataset('node_group_id', data=node_group_id)
        pop.create_dataset('node_group_index', data=node_group_index)
        
        # Create 0 group for attributes
        group_0 = pop.create_group('0')
        group_0.create_dataset('window_size_ms', data=window_sizes)
        group_0.create_dataset('similarity_threshold', data=thresholds)
        group_0.create_dataset('max_patterns', data=max_patterns_arr)
        group_0.create_dataset('region_row', data=region_rows)
        group_0.create_dataset('region_col', data=region_cols)
        group_0.create_dataset('orientation', data=orientations)
        group_0.create_dataset('neuron_index', data=neuron_indices)
        
        # Add metadata
        pop.attrs['population_name'] = population_name
        pop.attrs['total_neurons'] = total_neurons
        pop.attrs['grid_size'] = grid_size
        pop.attrs['num_orientations'] = num_orientations
        pop.attrs['neurons_per_feature'] = neurons_per_feature
        
        print(f"✓ Created {total_neurons} neurons in population '{population_name}'")
        
        # Create node_types group
        node_types = nodes.create_group('node_types')
        type_0 = node_types.create_group('0')
        type_0.attrs['model_type'] = 'snnfw_neuron'
        type_0.attrs['model_template'] = 'pattern_learning_neuron'
        
        print(f"✓ Successfully created SONATA network file: {output_file}")
        print(f"\nFile structure:")
        print(f"  /nodes/{population_name}/")
        print(f"    - node_type_id: {total_neurons} nodes")
        print(f"    - node_group_id: region assignments")
        print(f"    - 0/window_size_ms: {window_size_ms} ms")
        print(f"    - 0/similarity_threshold: {similarity_threshold}")
        print(f"    - 0/max_patterns: {max_patterns}")
        print(f"    - 0/region_row: spatial row index")
        print(f"    - 0/region_col: spatial column index")
        print(f"    - 0/orientation: orientation index (0-{num_orientations-1})")
        print(f"    - 0/neuron_index: neuron index within feature")

def verify_sonata_file(sonata_file):
    """Verify the SONATA file structure"""
    print(f"\nVerifying SONATA file: {sonata_file}")
    
    with h5py.File(sonata_file, 'r') as f:
        print("\nFile structure:")
        
        def print_structure(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}{name}: {obj.shape} {obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{indent}{name}/")
                if obj.attrs:
                    for key, value in obj.attrs.items():
                        print(f"{indent}  @{key}: {value}")
        
        f.visititems(print_structure)

if __name__ == "__main__":
    # Create output directory
    output_dir = Path("configs")
    output_dir.mkdir(exist_ok=True)
    
    # Generate SONATA file
    output_file = output_dir / "mnist_network.h5"
    
    try:
        create_mnist_network(str(output_file))
        verify_sonata_file(str(output_file))
        print("\n✓ SONATA network file generation complete!")
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)

