# SNNFrame Developer Manual

A comprehensive guide for developing programs using the SNNFrame spiking neural network framework.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Core Concepts](#core-concepts)
3. [Architecture Overview](#architecture-overview)
4. [Building Networks](#building-networks)
5. [Declarative Network Loading](#declarative-network-loading)
6. [Spike Processing](#spike-processing)
7. [Learning Mechanisms](#learning-mechanisms)
8. [Data Management](#data-management)
9. [Visualization](#visualization)
10. [Configuration](#configuration)
11. [Advanced Topics](#advanced-topics)
12. [Best Practices](#best-practices)
13. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

- C++17 compatible compiler (GCC 9+, Clang 10+, MSVC 2017+)
- CMake 3.16+
- RocksDB development libraries
- GLFW3 and GLEW (for visualization)

### Basic Program Structure

```cpp
#include "snnfw/Datastore.h"
#include "snnfw/NetworkBuilder.h"
#include "snnfw/SpikeProcessor.h"
#include "snnfw/NetworkPropagator.h"

using namespace snnfw;

int main() {
    // 1. Initialize datastore
    Datastore datastore("./my_network_db");
    
    // 2. Create network structure
    NetworkBuilder builder(datastore);
    auto brain = builder.createBrain();
    
    // 3. Create spike processor
    SpikeProcessor processor(10000, 4);  // 10s buffer, 4 threads
    processor.start();
    
    // 4. Create network propagator
    auto propagator = std::make_shared<NetworkPropagator>();
    
    // 5. Run simulation
    // ... inject spikes, process events ...
    
    processor.stop();
    return 0;
}
```

---

## Core Concepts

### Neurons

Neurons in SNNFrame learn **temporal spike patterns** rather than weights:

```cpp
// Create a neuron
auto neuron = std::make_shared<Neuron>(
    500.0,    // window size (ms)
    0.93,     // similarity threshold (0.0-1.0)
    500,      // max patterns per neuron
    neuronId  // unique ID
);

// Insert spikes
neuron->insertSpike(10.0);  // spike at 10ms
neuron->insertSpike(15.0);  // spike at 15ms

// Learn the current pattern
neuron->learnCurrentPattern();

// Check if neuron should fire
if (neuron->checkShouldFire()) {
    // Fire the neuron
}
```

### Synaptic Connections

Connections are made through Axons, Dendrites, and Synapses:

```cpp
// Create axon (output from neuron)
auto axon = std::make_shared<Axon>(sourceNeuronId, axonId);

// Create dendrite (input to neuron)
auto dendrite = std::make_shared<Dendrite>(targetNeuronId, dendriteId);

// Create synapse (connection)
auto synapse = std::make_shared<Synapse>(
    axonId,      // source
    dendriteId,  // target
    0.8,         // weight
    1.0,         // delay (ms)
    synapseId    // unique ID
);

// Register synapse with dendrite
dendrite->addSynapse(synapseId);
```

### Spike Timing

Spikes are scheduled with precise timing:

```cpp
// Create action potential
auto ap = std::make_shared<ActionPotential>(
    dendriteId,  // target
    10.0,        // delivery time (ms)
    0.8          // weight
);

// Schedule for delivery
processor.scheduleSpike(ap);
```

---

## Architecture Overview

### Hierarchical Organization

SNNFrame uses a 7-level biological hierarchy:

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

Each level provides:
- **Spatial organization** for network structure
- **Activity monitoring** at each level
- **Hierarchical caching** for efficient queries
- **Modular connectivity** patterns

### 6-Layer Cortical Microcircuit

The framework implements a canonical cortical microcircuit:

- **Layer 1**: Modulatory (feedback from higher areas)
- **Layer 2/3**: Superficial pyramidal (local processing)
- **Layer 4**: Granular (main input layer)
- **Layer 5**: Deep pyramidal (output to other areas)
- **Layer 6**: Corticothalamic (feedback to thalamus)

---

## Building Networks

### Using NetworkBuilder

The fluent API makes network construction intuitive:

```cpp
NetworkBuilder builder(datastore);

auto brain = builder.createBrain("MainBrain");
auto hemisphere = builder.createHemisphere(brain, "LeftHemisphere");
auto lobe = builder.createLobe(hemisphere, "OccipitalLobe");
auto region = builder.createRegion(lobe, "V1");
auto nucleus = builder.createNucleus(region, "Layer4C");
auto column = builder.createColumn(nucleus, "Orientation0");
auto layer = builder.createLayer(column, "Layer2/3");
auto cluster = builder.createCluster(layer);

// Create neurons in the cluster
std::vector<std::shared_ptr<Neuron>> neurons;
for (int i = 0; i < 100; ++i) {
    neurons.push_back(builder.createNeuron(cluster));
}
```

### Manual Network Construction

For more control, build networks manually:

```cpp
// Create objects directly
auto brain = std::make_shared<Brain>(brainId, "MyBrain");
auto hemisphere = std::make_shared<Hemisphere>(hemId, "Left");

// Add to hierarchy
brain->addHemisphere(hemId);
datastore.put(brain);
datastore.put(hemisphere);
```

---

## Declarative Network Loading

Instead of constructing networks programmatically in C++, you can define network structure
in configuration files and load them at runtime. The declarative loader supports four formats.

### Overview

The declarative loading pipeline works in three stages:

```
Description File → Parser → NetworkIR → NetworkConstructor → Running Network
```

1. **Parser**: Reads a format-specific file and produces a `NetworkIR`
2. **NetworkIR**: A common intermediate representation shared by all formats
3. **NetworkConstructor**: Builds the actual SNNFrame hierarchy from the IR

### Using DeclarativeLoader

```cpp
#include "snnfw/declarative/DeclarativeLoader.h"

using namespace snnfw;
using namespace snnfw::declarative;

NeuralObjectFactory factory;
Datastore datastore("./my_network_db");
DeclarativeLoader loader(factory, datastore);

// Load from any supported format — auto-detected from extension
auto network = loader.loadNetwork("configs/emnist_v1_network.snnf.json");

// The returned ConstructedNetwork contains:
// network.brain           — fully constructed Brain hierarchy
// network.spikeProcessor  — SpikeProcessor ready to start()
// network.propagator      — NetworkPropagator with all synapses registered
// network.inputNeurons    — input grid neurons (for spike injection)
// network.outputPopulations — output neurons grouped by class
// network.columns         — per-column neuron groups
```

### Parse Without Constructing

To inspect or modify the IR before construction:

```cpp
// Parse only — returns NetworkIR without building the network
auto ir = loader.parseOnly("configs/emnist_v1_network.snnf.json");

// Inspect the IR
std::cout << "Brain name: " << ir.brain.name << std::endl;
std::cout << "Projections: " << ir.projections.size() << std::endl;

// Validate
auto errors = ir.validate();
for (const auto& err : errors) {
    std::cerr << "Validation error: " << err << std::endl;
}
```

### Supported Formats

#### Native JSON (`.snnf.json`)

The most expressive format, designed specifically for SNNFrame:

- **Column templates**: Generate multiple columns from a single template with parameterized orientations and frequencies
- **Named neuron parameter sets**: Define parameter profiles once, reference by name
- **Path-based connectivity**: Use glob patterns like `V1/*/L4` for cross-column projections
- **Synapse groups**: Tag projections for differential STDP treatment
- **Full config**: Gabor filters, saccade regions, simulation parameters

```json
{
  "snnframe_version": "1.0",
  "neuron_params": {
    "cortical": { "window_size_ms": 500.0, "similarity_threshold": 0.93, "max_reference_patterns": 500 }
  },
  "brain": {
    "name": "MyBrain",
    "hemispheres": [{ "name": "Left", "lobes": [{ "name": "Visual", "regions": [{ "name": "V1",
      "nuclei": [{ "name": "Columns",
        "column_template": {
          "orientations": [0, 45, 90, 135],
          "frequencies": [3.0, 8.0],
          "layers": [
            { "name": "L4", "populations": [
              { "name": "cells", "count": 49, "neuron_params": "cortical" }
            ]}
          ]
        }
      }]
    }]}]}]
  },
  "projections": [
    { "name": "L4_to_L5", "source": "V1/*/L4", "target": "V1/*/L5",
      "pattern": "random_sparse", "probability": 0.25, "weight": 0.1,
      "scope": "intra_column", "synapse_group": "L4ToL5" }
  ]
}
```

**Connectivity scope**: `"global"` connects across all columns; `"intra_column"` connects only within the same column.

**Connectivity patterns**: `random_sparse`, `all_to_all`, `one_to_one`, `many_to_one`, `distance_dependent`, `topographic`, `small_world`.

#### SONATA (`circuit_config.json`, `.sonata.json`)

The [SONATA](https://github.com/AllenInstitute/sonata) format from the Blue Brain Project / Allen Institute uses HDF5 files for node and edge data:

```json
{
  "manifest": { "$BASE_DIR": ".", "$NETWORK_DIR": "$BASE_DIR/networks" },
  "networks": {
    "nodes": [{ "nodes_file": "$NETWORK_DIR/nodes.h5" }],
    "edges": [{ "edges_file": "$NETWORK_DIR/edges.h5" }]
  },
  "snnframe": {
    "neuron_params": { "cortical": { "window_size_ms": 200.0, "similarity_threshold": 0.93 } },
    "input_layer": { "rows": 14, "cols": 14 },
    "output_layer": { "num_classes": 26, "neurons_per_class": 3 }
  }
}
```

The parser reads node populations from HDF5 `nodes.h5` files and edge populations from `edges.h5` files using libsonata, then builds the SNNFrame hierarchy. SNNFrame-specific extensions go in the `"snnframe"` section.

#### NeuroML (`.nml`, `.neuroml`)

[NeuroML v2](https://docs.neuroml.org/) is an XML community standard. SNNFrame extends it with `snnfw:` property tags:

```xml
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="example">
  <cell id="cortical_cell">
    <property tag="snnfw:window_size_ms" value="200.0"/>
    <property tag="snnfw:similarity_threshold" value="0.93"/>
    <property tag="snnfw:max_reference_patterns" value="500"/>
    <property tag="snnfw:similarity_metric" value="cosine"/>
  </cell>

  <expOneSynapse id="exc_syn" tauDecay="5ms" gbase="0.8nS" erev="0mV">
    <property tag="snnfw:weight" value="0.5"/>
    <property tag="snnfw:delay" value="1.5"/>
  </expOneSynapse>

  <network id="MyNetwork">
    <population id="L4" component="cortical_cell" size="49">
      <property tag="snnfw:layer" value="L4"/>
    </population>
    <projection id="L4_to_L5" presynapticPopulation="L4"
                postsynapticPopulation="L5" synapse="exc_syn">
      <connection id="0" preCellId="../L4[0]" postCellId="../L5[0]"/>
    </projection>
  </network>
</neuroml>
```

#### HOC (`.hoc`)

The [NEURON simulator](https://www.neuron.yale.edu/) HOC scripting language is procedural, not declarative. The parser extracts structural information:

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

The parser extracts:
- **Cell templates** from `begintemplate`/`endtemplate` blocks → neuron parameter sets
- **Instantiation counts** from `for` loops with `new TemplateName()` → population sizes
- **Connections** from `new NetCon(source, target, threshold, delay, weight)` → projections

---

## Spike Processing

### SpikeProcessor

Manages spike delivery with multi-threaded processing:

```cpp
// Create processor
SpikeProcessor processor(
    10000,  // time slices (10 seconds at 1ms per slice)
    24      // delivery threads
);

processor.start();

// Schedule spikes
auto ap = std::make_shared<ActionPotential>(dendriteId, 10.0, 0.8);
processor.scheduleSpike(ap);

// Process for 100ms
for (int t = 0; t < 100; ++t) {
    processor.update(1.0);  // 1ms per step
}

processor.stop();
```

### NetworkPropagator

Manages neuron firing and spike propagation:

```cpp
auto propagator = std::make_shared<NetworkPropagator>();

// Register neurons
propagator->registerNeuron(neuron);

// Register synapses
propagator->registerSynapse(synapse);

// Fire a neuron
propagator->fireNeuron(neuronId, currentTime);
```

---

## Learning Mechanisms

### STDP (Spike-Timing-Dependent Plasticity)

STDP adjusts synaptic weights based on spike timing:

```cpp
// Enable STDP during training
processor.setSTDPEnabled(true);

// ... training loop ...

// Disable STDP during testing
processor.setSTDPEnabled(false);

// ... testing loop ...
```

### Pattern Learning

Neurons learn spike patterns automatically:

```cpp
// Inject spikes
neuron->insertSpike(10.0);
neuron->insertSpike(15.0);
neuron->insertSpike(20.0);

// Learn the pattern
neuron->learnCurrentPattern();

// Check learned patterns
size_t patternCount = neuron->getLearnedPatternCount();
const auto& patterns = neuron->getLearnedPatterns();
```

---

## Data Management

### Datastore

High-performance persistent storage with LRU caching:

```cpp
// Create datastore
Datastore datastore("./neural_db", 1000000);  // 1M object cache

// Store objects
auto neuron = std::make_shared<Neuron>(500.0, 0.93, 500);
datastore.put(neuron);

// Retrieve objects
auto retrieved = datastore.getNeuron(neuronId);

// Mark as modified
datastore.markDirty(neuronId);

// Flush to disk
datastore.flushAll();

// Get statistics
auto stats = datastore.getStatistics();
std::cout << "Cache hits: " << stats.cacheHits << std::endl;
```

### Serialization

All neural objects support JSON serialization:

```cpp
// Serialize to JSON
std::string json = neuron->toJson();

// Deserialize from JSON
neuron->fromJson(json);
```

---

## Visualization

### ActivityMonitor

Track network activity at all hierarchical levels:

```cpp
ActivityMonitor monitor(1000);  // 1000ms history

// Build hierarchical cache
monitor.buildHierarchicalCache(brain->getId());

// Get activity snapshot
auto snapshot = monitor.getActivitySnapshot();

// Query specific levels
auto clusterActivity = monitor.getClusterActivity(clusterId);
auto layerActivity = monitor.getLayerActivity(layerId);
```

### VisualizationManager

Real-time 3D visualization:

```cpp
VisualizationManager visualizer(datastore, networkAdapter);

// Configure visualization
visualizer.setShowSpikes(true);
visualizer.setShowActivity(true);
visualizer.setDecayRate(2.0f);

// Render each frame
visualizer.update(deltaTime);
visualizer.render();
```

---

## Configuration

### ConfigLoader

Load hyperparameters from JSON:

```cpp
ConfigLoader config;
config.load("config.json");

// Access values
double windowSize = config.get<double>("/neuron/window_size_ms", 500.0);
double threshold = config.get<double>("/neuron/similarity_threshold", 0.93);
int maxPatterns = config.get<int>("/neuron/max_patterns", 500);

// Check existence
if (config.has("/training/examples_per_class")) {
    // ...
}

// Save modified config
config.save("output_config.json");
```

### Configuration File Format

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
  "training": {
    "examples_per_letter": 800,
    "test_images": 20800
  }
}
```

---

## Advanced Topics

### Custom Similarity Metrics

Change how neurons match patterns:

```cpp
// Available metrics
enum class SimilarityMetric {
    COSINE,      // Default, best for most tasks
    HISTOGRAM,   // Spike count overlap
    EUCLIDEAN,   // Euclidean distance
    CORRELATION, // Pearson correlation
    WAVEFORM     // Temporal shape matching
};

// Set metric
neuron->setSimilarityMetric(SimilarityMetric::COSINE);
```

### Lateral Connectivity

Create within-layer connections:

```cpp
// Connect neurons in same layer
for (size_t i = 0; i < neurons.size(); ++i) {
    for (size_t j = i + 1; j < neurons.size(); ++j) {
        // Create synapse from neuron i to neuron j
        auto synapse = std::make_shared<Synapse>(
            neurons[i]->getAxonId(),
            neurons[j]->getDendriteIds()[0],
            0.5,  // weight
            1.0,  // delay
            synapseId++
        );
    }
}
```

### Recurrent Connections

Create feedback loops:

```cpp
// L5 → L2/3 recurrent connection
auto synapse = std::make_shared<Synapse>(
    layer5Neuron->getAxonId(),
    layer23Neuron->getDendriteIds()[0],
    0.3,   // weight
    2.0,   // delay (ms)
    synapseId
);
```

---

## Best Practices

### 1. Memory Management

- Use `std::shared_ptr` for all neural objects
- Let Datastore manage object lifecycle
- Call `flushAll()` periodically during long runs
- Monitor cache statistics

### 2. Thread Safety

- SpikeProcessor is thread-safe
- Datastore is thread-safe
- Protect custom data structures with mutexes
- Use `markDirty()` after modifications

### 3. Performance Optimization

- Use Release build: `cmake -DCMAKE_BUILD_TYPE=Release`
- Increase thread count for larger networks
- Disable visualization during training
- Use appropriate window sizes (not too large)

### 4. Network Design

- Start with small networks and scale up
- Use multiple columns for feature selectivity
- Implement lateral inhibition for competition
- Use recurrent connections for temporal integration

### 5. Debugging

- Enable logging: `Logger::getInstance().initialize()`
- Use NetworkInspector to examine structure
- Monitor activity with ActivityMonitor
- Check spike statistics

---

## Troubleshooting

### Low Accuracy

**Problem**: Network achieves poor classification accuracy

**Solutions**:
1. Increase training examples per class
2. Adjust similarity threshold (try 0.85-0.95)
3. Increase window size for richer patterns
4. Use more neurons per feature
5. Verify input encoding is correct

### Memory Issues

**Problem**: Out of memory errors

**Solutions**:
1. Reduce cache size: `Datastore(path, 100000)`
2. Flush more frequently: `datastore.flushAll()`
3. Reduce network size
4. Disable recording during training

### Slow Performance

**Problem**: Simulation runs slowly

**Solutions**:
1. Use Release build
2. Increase thread count
3. Disable visualization
4. Reduce network size
5. Profile with `perf` or `gprof`

### Spikes Not Propagating

**Problem**: Spikes scheduled but not delivered

**Solutions**:
1. Verify dendrites are registered with processor
2. Check spike timing is within buffer range
3. Ensure processor is running
4. Verify synapse weights are non-zero

---

## Example Programs

See the `examples/` directory for complete working programs:

- `neural_network_example.cpp` - Basic network setup
- `hierarchical_structure_example.cpp` - Building hierarchies
- `activity_demo.cpp` - Monitoring activity
- `network_visualization_demo.cpp` - Real-time visualization
- `experiment_config_example.cpp` - Configuration management

---

## API Reference

For detailed API documentation, see:
- `docs/API_REFERENCE.md` - Complete class reference (includes declarative loader API)
- `docs/ARCHITECTURE.md` - Architecture details
- `docs/STDP_GUIDE.md` - STDP learning guide
- `docs/FORMAT_REFERENCE.md` - Declarative format specifications
- `include/snnfw/declarative/*.h` - Declarative loader header files
- `include/snnfw/*.h` - Core framework header files

---

## Performance Metrics

### EMNIST Letters Classification

- **Accuracy**: ~90% on 26-letter classification
- **Training**: ~60 minutes (5,200 images, 200 per letter)
- **Testing**: ~22 minutes (20,800 images)
- **Network**: 8 orientations × 2 frequencies = 16 columns
- **Neurons**: 392 neurons (49 regions × 8 orientations)

---

## Contributing

To contribute improvements:

1. Follow the coding standards in `CONTRIBUTING.md`
2. Write tests for new features
3. Update documentation
4. Submit a pull request

---

## License

SNNFrame is released under the MIT License. See `LICENSE` for details.

---

**Last Updated**: 2026-02-06
**Version**: 1.1.0

