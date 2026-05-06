# SNNFrame Quick Start Guide

Get up and running with SNNFrame in 5 minutes!

## Installation (2 minutes)

```bash
# Clone the repository
git clone https://github.com/deanhorak/SNNFrame.git
cd SNNFrame

# Install dependencies (Ubuntu/Debian)
sudo apt-get install librocksdb-dev libglfw3-dev libglew-dev

# Build
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Run the Best Experiment (3 minutes)

The framework includes a pre-configured EMNIST letters classification experiment that achieves **~90% accuracy** on 26-letter classification:

```bash
cd build

# Option 1: Headless training (fastest, ~60 minutes)
./emnist_letters_training

# Option 2: With real-time visualization
./emnist_letters_visualized

# Option 3: Record activity for later playback
./emnist_letters_training --record my_session.snnr
```

**Key Features of the 90% Accuracy Model**:
- Cosine similarity-based pattern matching
- STDP frozen during testing to prevent weight drift
- Multi-column architecture (8 orientations × 2 frequencies)
- 6-layer canonical cortical microcircuit
- Saccade-based attention mechanism

## Your First Network (Declarative)

The fastest way to create a network is with a configuration file:

**1. Create a network config** (`my_network.snnf.json`):

```json
{
  "snnframe_version": "1.0",
  "neuron_params": {
    "default": {
      "window_size_ms": 200.0,
      "similarity_threshold": 0.93,
      "max_reference_patterns": 500,
      "similarity_metric": "cosine"
    }
  },
  "input_layer": { "name": "Input", "rows": 14, "cols": 14, "latency_ms": 15.0 },
  "output_layer": { "name": "Output", "num_classes": 10, "neurons_per_class": 3 },
  "brain": {
    "name": "MyBrain",
    "hemispheres": [{
      "name": "Left",
      "lobes": [{ "name": "Visual", "regions": [{ "name": "V1",
        "nuclei": [{ "name": "Columns",
          "columns": [{ "name": "Col1",
            "layers": [{ "name": "L4", "populations": [
              { "name": "L4_cells", "count": 49, "neuron_params": "default" }
            ]}]
          }]
        }]
      }]}]
    }]
  },
  "projections": [
    { "name": "Input_to_L4", "source": "Input", "target": "V1/*/L4",
      "pattern": "random_sparse", "probability": 0.3, "weight": 0.1 }
  ]
}
```

**2. Load and run** (`my_first_network.cpp`):

```cpp
#include <iostream>
#include "snnfw/declarative/DeclarativeLoader.h"

using namespace snnfw;
using namespace snnfw::declarative;

int main() {
    NeuralObjectFactory factory;
    Datastore datastore("./my_network_db");
    DeclarativeLoader loader(factory, datastore);

    auto network = loader.loadNetwork("my_network.snnf.json");
    // network.brain, network.spikeProcessor, network.propagator are ready
    std::cout << "Network loaded successfully!" << std::endl;
    return 0;
}
```

**3. Compile and run:**

```bash
cd build
g++ -std=c++17 -I../include ../my_first_network.cpp -o my_first_network \
    -L. -lsnnfw -lrocksdb -lpthread
./my_first_network
```

### Supported Formats

You can also define networks in other standard formats:

| Format | Extension | Example |
|--------|-----------|---------|
| Native JSON | `.snnf.json` | `configs/emnist_v1_network.snnf.json` |
| SONATA | `circuit_config.json` | `configs/example_sonata/circuit_config.json` |
| NeuroML | `.nml` | `configs/example_network.nml` |
| HOC | `.hoc` | `configs/example_network.hoc` |

All formats are auto-detected from the file extension:

```cpp
// Any of these work — format auto-detected
auto net1 = loader.loadNetwork("model.snnf.json");
auto net2 = loader.loadNetwork("circuit_config.json");
auto net3 = loader.loadNetwork("model.nml");
auto net4 = loader.loadNetwork("model.hoc");
```

## Your First Network (Programmatic)

For full control, build the network in C++ code:

```cpp
#include <iostream>
#include "snnfw/NetworkBuilder.h"
#include "snnfw/Datastore.h"
#include "snnfw/SpikeProcessor.h"

using namespace snnfw;

int main() {
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

    std::vector<std::shared_ptr<Neuron>> neurons;
    for (int i = 0; i < 100; ++i) {
        neurons.push_back(builder.createNeuron(cluster));
    }

    SpikeProcessor processor(1000, 4);
    processor.start();
    neurons[0]->injectSpike(50.0);
    for (int t = 0; t < 100; ++t) {
        processor.update(1.0);
    }

    std::cout << "Network created with " << neurons.size() << " neurons" << std::endl;
    return 0;
}
```

## Key Concepts

### Hierarchical Organization
SNNFrame uses a 7-level hierarchy:
```
Brain → Hemisphere → Lobe → Region → Nucleus → Column → Layer → Cluster → Neuron
```

### Spike-Based Learning
Neurons learn spike patterns within a temporal window (default: 500ms):
- Pattern matching based on spike timing
- STDP (Spike-Timing-Dependent Plasticity) for weight updates
- Configurable similarity threshold (default: 0.93)

### Multi-Column Architecture
Multiple columns with different feature selectivity:
- Orientation columns (8 orientations)
- Frequency columns (2 frequencies)
- Center-surround columns (3 scales)
- Specialized detectors

### Activity Monitoring
Track network activity at multiple levels:
```cpp
ActivityMonitor monitor(1000);  // 1000ms history
monitor.buildHierarchicalCache(brain->getId());
auto snapshot = monitor.getActivitySnapshot();
```

## Configuration

Edit `configs/emnist_letters_saccades_best_v2.json` to customize:

```json
{
  "neuron": {
    "window_size_ms": 500,        // Temporal window for pattern matching
    "similarity_threshold": 0.93,  // Pattern recognition threshold
    "max_patterns": 500            // Max patterns per neuron
  },
  "spike_processor": {
    "num_threads": 24,             // Thread count
    "time_slice_ms": 1.0,          // Time slice duration
    "real_time_sync": false        // 1:1 real-time mapping
  },
  "architecture": {
    "columns": {
      "num_orientations": 8,       // Orientation selectivity
      "num_frequencies": 2         // Frequency selectivity
    }
  }
}
```

## Visualization Controls

When running with visualization (`emnist_letters_visualized`):

- **Mouse**: Rotate camera
- **WASD**: Pan camera
- **Q/E**: Zoom in/out
- **Space**: Pause/Resume training
- **ESC**: Exit

## Recording and Playback

Record a training session:
```bash
./emnist_letters_training --record my_session.snnr
```

Playback later:
```bash
./emnist_letters_visualized --playback my_session.snnr
```

## Performance Tips

1. **Use Release Build**: `cmake -DCMAKE_BUILD_TYPE=Release ..`
2. **Increase Threads**: Set `num_threads` to your CPU core count
3. **Disable Visualization**: Use `emnist_letters_training` instead of `emnist_letters_visualized`
4. **Disable Recording**: Set `recording.enabled: false` in config
5. **Reduce Network Size**: Decrease `num_orientations` or `layer23_neurons`

## Troubleshooting

### "RocksDB not found"
```bash
sudo apt-get install librocksdb-dev
```

### "GLFW not found"
```bash
sudo apt-get install libglfw3-dev
```

### "Compilation errors"
Ensure C++17 support:
```bash
g++ --version  # Should be 7.0+
```

### "Low accuracy"
Check that you're using the correct configuration:
```bash
./emnist_letters_training  # Uses default config
```

## Next Steps

1. **Try the declarative loader**: Define a network in JSON/NeuroML/HOC and load it
2. **Read the developer manual**: `docs/DEVELOPER_MANUAL.md`
3. **Explore format options**: `docs/FORMAT_REFERENCE.md`
4. **Study the best experiment**: `experiments/emnist_letters_training.cpp`
5. **Read the API reference**: `docs/API_REFERENCE.md`
6. **Customize for your task**: Modify network architecture and parameters

## Common Tasks

### Create a Custom Network
See `examples/neural_network_example.cpp`

### Add Custom Encoding
See `examples/custom_adapter.cpp`

### Visualize Network Structure
See `examples/network_visualization_demo.cpp`

### Monitor Activity
See `examples/activity_demo.cpp`

## Resources

- **Documentation**: `docs/` directory
- **Examples**: `examples/` directory
- **Experiments**: `experiments/` directory
- **Tests**: `tests/` directory

## Getting Help

- Check `docs/CONFIGURATION_GUIDE.md` for configuration options
- Review `examples/` for code samples
- Read `INSTALLATION.md` for troubleshooting
- Open an issue on GitHub

---

**Ready to build amazing SNNs!** 🧠⚡

For detailed information, see `README.md` and the documentation in `docs/`.
