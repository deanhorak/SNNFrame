# SNNFrame - Spiking Neural Network Framework

A modern, production-ready C++ framework for building and simulating spiking neural networks (SNNs) with hierarchical organization, real-time visualization, and comprehensive analysis tools.

## Features

### Core Framework
- **Hierarchical Neural Structure**: 7-level organization (Brain → Hemisphere → Lobe → Region → Nucleus → Column → Layer → Cluster → Neuron)
- **Biologically-Inspired Architecture**: Canonical cortical microcircuit with 6-layer organization
- **Spike-Based Learning**: STDP (Spike-Timing-Dependent Plasticity) with retrograde signaling
- **Temporal Pattern Matching**: Neurons learn and recognize spike patterns within configurable temporal windows
- **Persistent Storage**: RocksDB-backed datastore with LRU caching for efficient memory management

### Advanced Features
- **Multi-Column Networks**: Support for orientation-selective and feature-selective columns
- **Saccade-Based Attention**: Sequential spatial attention mechanism for improved feature learning
- **Real-Time Visualization**: OpenGL-based 3D network visualization with activity heatmaps
- **Activity Recording & Playback**: Record spike activity for later analysis and playback
- **Comprehensive Monitoring**: ActivityMonitor with hierarchical activity tracking
- **Flexible Encoding**: Multiple encoding strategies (rate, temporal, population)
- **Edge Detection**: Gabor, Sobel, and Difference-of-Gaussians operators

### Performance
- **Multi-Threaded Spike Processing**: Configurable thread pool for parallel spike delivery
- **Real-Time Synchronization**: Optional 1:1 real-time mapping (1ms simulation = 1ms wall-clock)
- **Efficient Datastore**: LRU cache with automatic dirty-tracking and flush-on-eviction
- **Optimized Compilation**: Release builds with -O3 optimization

## Quick Start

### Installation

#### Prerequisites
- C++17 compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.16+
- RocksDB development libraries
- OpenGL 4.5+ (for visualization)

#### Ubuntu/Debian
```bash
# Install system dependencies
sudo apt-get install build-essential cmake librocksdb-dev libglfw3-dev libglew-dev

# Clone and build
git clone https://github.com/yourusername/SNNFrame.git
cd SNNFrame
mkdir build && cd build
cmake ..
make -j$(nproc)
```

#### macOS
```bash
# Install with Homebrew
brew install cmake rocksdb glfw3 glew

# Clone and build
git clone https://github.com/yourusername/SNNFrame.git
cd SNNFrame
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Running the Best Experiment

The framework includes the EMNIST letters classification experiment that achieved **71.93% accuracy**:

```bash
# Training (headless, ~60 minutes)
./emnist_letters_training

# Training with visualization
./emnist_letters_visualized

# Training with recording for playback
./emnist_letters_training --record emnist_session.snnr

# Playback recorded session
./emnist_letters_visualized --playback emnist_session.snnr
```

## Architecture Overview

### Network Structure
```
Brain (1)
├── Hemisphere (2)
│   ├── Lobe (4)
│   │   ├── Region (8)
│   │   │   ├── Nucleus (16)
│   │   │   │   ├── Column (24)
│   │   │   │   │   ├── Layer (6)
│   │   │   │   │   │   ├── Cluster (multiple)
│   │   │   │   │   │   │   └── Neuron (multiple)
```

### Cortical Microcircuit (6-Layer Model)
- **Layer 1**: Apical dendrites, modulatory inputs
- **Layer 2/3**: Superficial pyramidal neurons, lateral connections
- **Layer 4**: Granular input layer (thalamic/sensory input)
- **Layer 5**: Deep pyramidal neurons, output layer
- **Layer 6**: Corticothalamic feedback neurons

### Connectivity
- **Feedforward**: Input → L4 → L2/3 → L5 → Output
- **Feedback**: L6 → L4 (corticothalamic)
- **Lateral**: Within-layer connections for competition and cooperation
- **Recurrent**: L5 → L2/3 for temporal integration

## Configuration

The framework uses JSON configuration files. See `configs/emnist_letters_saccades_best_v2.json` for the best-performing configuration:

```json
{
  "neuron": {
    "window_size_ms": 500,
    "similarity_threshold": 0.93,
    "max_patterns": 500
  },
  "spike_processor": {
    "num_threads": 24,
    "time_slice_ms": 1.0,
    "real_time_sync": false
  },
  "architecture": {
    "columns": {
      "num_orientations": 8,
      "num_frequencies": 2
    },
    "layers": {
      "layer1_neurons": 32,
      "layer23_neurons": 448,
      "layer4_size": 7,
      "layer5_neurons": 80,
      "layer6_neurons": 32
    }
  }
}
```

## API Documentation

### Creating a Network

```cpp
#include "snnfw/NetworkBuilder.h"
#include "snnfw/Datastore.h"

using namespace snnfw;

// Initialize datastore
Datastore datastore("./my_network_db");

// Create network builder
NetworkBuilder builder(datastore);

// Build hierarchical structure
auto brain = builder.createBrain();
auto hemisphere = builder.createHemisphere(brain);
auto lobe = builder.createLobe(hemisphere);
auto region = builder.createRegion(lobe);
auto nucleus = builder.createNucleus(region);
auto column = builder.createColumn(nucleus);
auto layer = builder.createLayer(column);
auto cluster = builder.createCluster(layer);

// Create neurons
for (int i = 0; i < 100; ++i) {
    auto neuron = builder.createNeuron(cluster);
}
```

### Training with Spike Patterns

```cpp
// Create spike processor
SpikeProcessor processor(10000, 20);  // 10000 time slices, 20 threads
processor.start();

// Inject spikes
for (auto& neuron : neurons) {
    neuron->injectSpike(100.0);  // Spike at 100ms
}

// Process spikes
processor.update(deltaTime);

// Check neuron state
if (neuron->hasFired()) {
    std::cout << "Neuron fired!" << std::endl;
}
```

### Monitoring Activity

```cpp
#include "snnfw/ActivityMonitor.h"

ActivityMonitor monitor(1000);  // 1000ms history

// Build hierarchical cache for cluster-level monitoring
monitor.buildHierarchicalCache(brain->getId());

// Get activity snapshot
auto snapshot = monitor.getActivitySnapshot();
for (const auto& [clusterId, spikeCount] : snapshot.clusterActivity) {
    std::cout << "Cluster " << clusterId << ": " << spikeCount << " spikes" << std::endl;
}
```

### Visualization

```cpp
#include "snnfw/VisualizationManager.h"

VisualizationManager vizManager(1920, 1080);
vizManager.initialize();

// Render loop
while (!vizManager.shouldClose()) {
    vizManager.update(deltaTime);
    vizManager.render();
}
```

## Performance Characteristics

### EMNIST Letters (71.93% accuracy)
- **Network Size**: 16,542 neurons, 4,865,329 synapses
- **Training Time**: ~60 minutes (2,600 images)
- **Testing Time**: ~22 minutes (20,800 images)
- **Memory**: ~4.9M synapses, ~16.5K neurons
- **Accuracy**: 71.93% (14,958/20,800 correct)

### Scaling
- Tested up to 24 cortical columns
- Supports brain-scale networks with proper hierarchical organization
- Linear scaling with thread count (up to 24 threads tested)

## Testing

Run the comprehensive test suite:

```bash
cd build
ctest --output-on-failure
```

Key test categories:
- Unit tests for neural components
- Integration tests for network construction
- Performance regression tests
- Stress tests for large networks

## Documentation

- `docs/QUICK_REFERENCE.md` - Quick API reference
- `docs/CONFIGURATION_GUIDE.md` - Configuration system guide
- `docs/VISUALIZATION_QUICK_START.md` - Visualization setup
- `docs/RECORDING_PLAYBACK_USAGE.md` - Recording and playback
- `docs/HIERARCHICAL_ACTIVITY_USAGE.md` - Activity monitoring

## Contributing

Contributions are welcome! Please ensure:
1. Code follows the existing style
2. All tests pass
3. New features include tests
4. Documentation is updated

## License

[Your License Here]

## Citation

If you use SNNFrame in your research, please cite:

```bibtex
@software{snnframe2025,
  title={SNNFrame: A Spiking Neural Network Framework},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/SNNFrame}
}
```

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review example code in `examples/` and `experiments/`

---

**Status**: Production-ready  
**Latest Version**: 1.0.0  
**Last Updated**: 2025-12-12
