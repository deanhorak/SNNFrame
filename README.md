# SNNFrame - Spiking Neural Network Framework

A modern, production-ready C++ framework for building and simulating spiking neural networks (SNNs) with hierarchical organization, real-time visualization, and comprehensive analysis tools.

## Features

### Core Framework
- **Hierarchical Neural Structure**: 7-level organization (Brain → Hemisphere → Lobe → Region → Nucleus → Column → Layer → Cluster → Neuron)
- **Biologically-Inspired Architecture**: Canonical cortical microcircuit with 6-layer organization
- **Spike-Based Learning**: STDP (Spike-Timing-Dependent Plasticity) with retrograde signaling and runtime enable/disable for inference
- **Temporal Pattern Matching**: Neurons learn and recognize spike patterns within configurable temporal windows
- **Persistent Storage**: RocksDB-backed datastore with LRU caching for efficient memory management

### Declarative Network Loading
- **Multi-Format Support**: Define networks in configuration files instead of C++ code
- **Native JSON** (`.snnf.json`): Full-featured SNNFrame format with column templates, path-based connectivity
- **SONATA** (`circuit_config.json`): Blue Brain Project / Allen Institute HDF5 format for large-scale models
- **NeuroML** (`.nml`, `.neuroml`): XML-based community standard with SNNFrame property extensions
- **HOC** (`.hoc`): NEURON simulator scripting language — extracts structural information from HOC scripts
- **Auto-Detection**: Format is automatically detected from file extension
- **NetworkIR**: Common intermediate representation enables cross-format interoperability
- **Custom Parsers**: Extensible parser interface for adding new formats

### Experiment Framework
- **ExperimentRunner**: High-level API for loading models and running training/testing
- **SpikeEncoder**: Converts input data to spike times with schedule horizon management
- **CompetitionManager**: Winner-take-all competition with configurable thresholds
- **KNNClassifier**: k-NN and centroid-based classification from activation patterns
- **SupervisedTeacher**: Supervised learning signal for output layer training
- **TrainingPipeline**: Multi-pass training with convergence detection

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
- **Optimized Spike Delivery**: ~15% performance improvement through thread pool optimization and smart batching
- **Efficient Neuron Operations**: O(1) spike insertion with cached max spike time tracking
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
git clone https://github.com/deanhorak/SNNFrame.git
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
git clone https://github.com/deanhorak/SNNFrame.git
cd SNNFrame
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Running Example Experiments

The framework includes a high-performance EMNIST letters classification experiment achieving ~90% accuracy:

#### Hardcoded C++ Experiment
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

#### Declarative SONATA Experiment
```bash
# Run experiment from SONATA configuration
./experiments/emnist_sonata_training \
  --config ../configs/emnist_v1_sonata/circuit_config.json \
  --train-images ../data/EMNIST/emnist-letters-train-images-idx3-ubyte \
  --train-labels ../data/EMNIST/emnist-letters-train-labels-idx1-ubyte \
  --test-images ../data/EMNIST/emnist-letters-test-images-idx3-ubyte \
  --test-labels ../data/EMNIST/emnist-letters-test-labels-idx1-ubyte \
  --datastore ./sonata_experiment_db \
  --max-passes 5 \
  --test-limit 1000
```

#### Declarative Retina Experiments
```bash
# Unilateral Retina static reference
./scripts/run_emnist_retina_unilateral.sh

# Bilateral Retina static reference (corpus-callosum fusion)
./scripts/run_emnist_retina_bilateral.sh

# Bilateral Retina continuous-learning reference
./scripts/run_emnist_retina_bilateral_continuous.sh
```

Static reference configs:
- `configs/emnist_retina_experimental.sonata.json`
- `configs/emnist_retina_bilateral_experimental.sonata.json`

Continuous-learning reference config:
- `configs/emnist_retina_bilateral_continuous.sonata.json`

The bilateral configs use two transformed hemisphere views and corpus-callosum-style weighted fusion over the hemisphere classifications. The continuous config adds reward-driven online adaptation with delayed replay and context-gated plasticity.

Reference full-run benchmarks on EMNIST letters (`3200/class`, `5200` test, `seed 42`):

Static inference benchmarks:
- Unilateral Retina: `86.17%`
- Bilateral Retina: `87.29%`

Continuous-learning benchmark:
- Bilateral Retina continuous: initial `87.56%`, post-correction `88.44%`

Static and continuous results are separate benchmark categories. The continuous result includes online reward-driven adaptation during evaluation and is not directly comparable to the static held-out metric.

Benchmark summary:

| Mode | Config | Metric | Result |
| --- | --- | --- | --- |
| Unilateral Retina | `configs/emnist_retina_experimental.sonata.json` | Static accuracy | `86.17%` |
| Bilateral Retina | `configs/emnist_retina_bilateral_experimental.sonata.json` | Static accuracy | `87.29%` |
| Bilateral Retina Continuous | `configs/emnist_retina_bilateral_continuous.sonata.json` | Initial / post-correction accuracy | `87.56% -> 88.44%` |

Reference commands:
```bash
./scripts/run_emnist_retina_unilateral.sh
./scripts/run_emnist_retina_bilateral.sh
./scripts/run_emnist_retina_bilateral_continuous.sh
```

Expected logs:
- `build/emnist_retina_unilateral_experimental.log`
- `build/emnist_retina_bilateral_experimental.log`
- `build/emnist_retina_bilateral_continuous.log`

**Performance**: The framework achieves approximately **90% accuracy** on EMNIST letters classification (26 classes) using:
- Cosine similarity-based pattern matching
- STDP frozen during testing to prevent weight drift
- Multi-column architecture with 8 orientations and 2 frequencies
- 6-layer canonical cortical microcircuit
- Saccade-based attention mechanism
- Tile-based receptive fields for spatial locality

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

## Declarative Network Loading

Instead of building networks in C++, define them in configuration files and load them at runtime. Four formats are supported:

### Loading a Network

```cpp
#include "snnfw/declarative/DeclarativeLoader.h"

using namespace snnfw;
using namespace snnfw::declarative;

NeuralObjectFactory factory;
Datastore datastore("./my_network_db");
DeclarativeLoader loader(factory, datastore);

// Load from any supported format — auto-detected from extension
auto network = loader.loadNetwork("configs/emnist_v1_network.snnf.json");

// network.brain           — constructed Brain hierarchy
// network.spikeProcessor  — ready SpikeProcessor
// network.propagator      — ready NetworkPropagator
// network.inputNeurons    — input layer neurons
// network.outputPopulations — output neurons grouped by class
// network.columns         — per-column neuron groups
```

### Supported Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| **Native JSON** | `.snnf.json` | Full-featured SNNFrame format with column templates, path-based connectivity, Gabor/saccade config |
| **SONATA** | `circuit_config.json`, `.sonata.json` | HDF5-based format from Blue Brain Project / Allen Institute |
| **NeuroML** | `.nml`, `.neuroml` | XML-based community standard with `snnfw:` property extensions |
| **HOC** | `.hoc` | NEURON simulator scripting language — structural extraction |

### Native JSON Example (`.snnf.json`)

```json
{
  "snnframe_version": "1.0",
  "neuron_params": {
    "cortical_default": {
      "window_size_ms": 500.0,
      "similarity_threshold": 0.93,
      "max_reference_patterns": 500
    }
  },
  "brain": {
    "name": "MyBrain",
    "hemispheres": [{
      "name": "Left",
      "lobes": [{
        "name": "Occipital",
        "regions": [{
          "name": "V1",
          "nuclei": [{
            "name": "FeatureColumns",
            "column_template": {
              "orientations": [0, 45, 90, 135],
              "frequencies": [3.0, 8.0],
              "layers": [
                { "name": "L4", "populations": [
                  { "name": "L4_stellate", "count": 49, "neuron_params": "cortical_default" }
                ]}
              ]
            }
          }]
        }]
      }]
    }]
  },
  "projections": [
    { "name": "Input_to_L4", "source": "InputGrid", "target": "V1/*/L4",
      "pattern": "random_sparse", "probability": 0.3, "weight": 0.1 }
  ]
}
```

### NeuroML Example (`.nml`)

```xml
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="example">
  <cell id="cortical_cell">
    <property tag="snnfw:window_size_ms" value="200.0"/>
    <property tag="snnfw:similarity_threshold" value="0.93"/>
  </cell>
  <network id="MyNetwork">
    <population id="L4" component="cortical_cell" size="49"/>
    <projection id="L4_to_L5" presynapticPopulation="L4" postsynapticPopulation="L5" synapse="exc"/>
  </network>
</neuroml>
```

### HOC Example (`.hoc`)

```hoc
begintemplate CorticalL4
    proc init() {
        window_size_ms = 200
        similarity_threshold = 0.93
    }
    create soma, axon, dendrite
endtemplate CorticalL4

for i = 0, 48 {
    l4_cells.append(new CorticalL4())
}

for i = 0, 48 {
    nc = new NetCon(l4_cells.o(i).soma(0.5), l5_cells.o(i).syn, 0, 1.5, 0.5)
}
```

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
    }
  }
}
```

## API Documentation

### Creating a Network (Programmatic)

```cpp
#include "snnfw/NetworkBuilder.h"
#include "snnfw/Datastore.h"

using namespace snnfw;

Datastore datastore("./my_network_db");
NetworkBuilder builder(datastore);

auto brain = builder.createBrain();
auto hemisphere = builder.createHemisphere(brain);
auto lobe = builder.createLobe(hemisphere);
auto region = builder.createRegion(lobe);
auto nucleus = builder.createNucleus(region);
auto column = builder.createColumn(nucleus);
auto layer = builder.createLayer(column);
auto cluster = builder.createCluster(layer);

for (int i = 0; i < 100; ++i) {
    auto neuron = builder.createNeuron(cluster);
}
```

### Creating a Network (Declarative)

```cpp
#include "snnfw/declarative/DeclarativeLoader.h"

using namespace snnfw;
using namespace snnfw::declarative;

NeuralObjectFactory factory;
Datastore datastore("./my_network_db");
DeclarativeLoader loader(factory, datastore);

// Load from any supported format
auto network = loader.loadNetwork("configs/my_network.snnf.json");
// Or: loader.loadNetwork("configs/model.nml");
// Or: loader.loadNetwork("configs/circuit_config.json");
// Or: loader.loadNetwork("configs/model.hoc");
```

## Performance Characteristics

### EMNIST Letters Classification
- **Accuracy**: ~90% on 26-letter classification task
- **Network Size**: Multi-column architecture with 8 orientations × 2 frequencies
- **Training**: ~60 minutes on full training set (5,200 images, 200 per letter)
- **Testing**: ~22 minutes on full test set (20,800 images)
- **Key Features**:
  - Cosine similarity-based pattern matching
  - STDP frozen during testing to prevent weight drift
  - Saccade-based attention mechanism for sequential processing
  - 6-layer canonical cortical microcircuit

### Scaling
- Tested up to 24 cortical columns
- Supports brain-scale networks with proper hierarchical organization
- Linear scaling with thread count (up to 24 threads tested)
- Efficient memory management with RocksDB-backed persistent storage
- Multi-threaded spike processing for real-time performance

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

- `docs/DEVELOPER_MANUAL.md` - Comprehensive developer manual
- `docs/DEVELOPER_GUIDE_PATTERNS.md` - Common patterns and recipes
- `docs/DEVELOPER_GUIDE_ADVANCED.md` - Advanced topics and custom parsers
- `docs/API_REFERENCE.md` - API class reference
- `docs/DOCUMENTATION_INDEX.md` - Full documentation index
- `docs/QUICK_REFERENCE.md` - Quick API reference
- `docs/CONFIGURATION_GUIDE.md` - Configuration system guide
- `docs/FORMAT_REFERENCE.md` - Declarative format specifications

## Contributing

Contributions are welcome! Please ensure:
1. Code follows the existing style
2. All tests pass
3. New features include tests
4. Documentation is updated

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use SNNFrame in your research, please cite:

```bibtex
@software{snnframe2025,
  title={SNNFrame: A Spiking Neural Network Framework},
  author={Dean Horak},
  year={2025},
  url={https://github.com/deanhorak/SNNFrame}
}
```

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review example code in `examples/` and `experiments/`

---

**Status**: Production-ready
**Latest Version**: 1.1.0
**Last Updated**: 2026-02-06
