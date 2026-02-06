# SNNFrame Documentation Index

Complete guide to all SNNFrame documentation and resources.

## Quick Navigation

### For New Users
1. **[QUICK_START.md](../QUICK_START.md)** - Get running in 5 minutes
2. **[INSTALLATION.md](../INSTALLATION.md)** - Detailed installation guide
3. **[README.md](../README.md)** - Project overview and features

### For Developers
1. **[DEVELOPER_MANUAL.md](DEVELOPER_MANUAL.md)** - Complete developer guide (includes declarative network loading)
2. **[DEVELOPER_GUIDE_PATTERNS.md](DEVELOPER_GUIDE_PATTERNS.md)** - Common patterns and recipes (includes declarative loading patterns)
3. **[DEVELOPER_GUIDE_ADVANCED.md](DEVELOPER_GUIDE_ADVANCED.md)** - Advanced techniques (includes custom parser development)

### For Declarative Network Loading
1. **[FORMAT_REFERENCE.md](FORMAT_REFERENCE.md)** - Complete format specifications for all four supported formats
2. **[DEVELOPER_MANUAL.md § Declarative Network Loading](DEVELOPER_MANUAL.md#declarative-network-loading)** - Usage guide
3. **[DEVELOPER_GUIDE_ADVANCED.md § Custom Format Parsers](DEVELOPER_GUIDE_ADVANCED.md#custom-format-parsers)** - Writing new parsers
4. **[API_REFERENCE.md § Declarative Loader](API_REFERENCE.md#declarativeloader)** - API class reference

### For Reference
1. **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API documentation (core + declarative)
2. **[ARCHITECTURE.md](ARCHITECTURE.md)** - Architecture and design
3. **[STDP_GUIDE.md](STDP_GUIDE.md)** - STDP learning guide
4. **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions

---

## Documentation Overview

### DEVELOPER_MANUAL.md

**Purpose**: Comprehensive guide for developing with SNNFrame

**Contents**:
- Getting started with basic program structure
- Core concepts (neurons, synapses, spike timing)
- Architecture overview (7-level hierarchy, 6-layer cortical microcircuit)
- Building networks with NetworkBuilder
- **Declarative network loading** (all four formats, pipeline overview, API usage)
- Spike processing and delivery
- Learning mechanisms (STDP, pattern learning)
- Data management with Datastore
- Visualization with ActivityMonitor
- Configuration management with ConfigLoader
- Advanced topics and best practices
- Troubleshooting guide

**Best For**: Developers new to SNNFrame who want a complete overview

---

### DEVELOPER_GUIDE_PATTERNS.md

**Purpose**: Common patterns and recipes for SNNFrame development

**Contents**:
- Creating networks (simple and multi-layer)
- **Declarative loading patterns** (load/run, validate, column templates, path-based connectivity, format conversion)
- Spike injection and stimulus encoding
- Pattern learning and matching
- Classification techniques (k-NN, voting, confidence)
- Monitoring and debugging
- Data encoding (Gabor, temporal, population coding)
- Multi-column architectures
- Training vs testing phases
- Performance optimization tips

**Best For**: Developers looking for code examples and recipes

---

### DEVELOPER_GUIDE_ADVANCED.md

**Purpose**: Advanced techniques and patterns for expert developers

**Contents**:
- Custom similarity metrics
- Connectivity patterns (lateral inhibition, WTA, recurrent)
- STDP customization and freezing
- Homeostatic plasticity
- Attention mechanisms
- Saccade-based processing
- Recording and playback
- Performance profiling
- Custom adapters and classifiers
- Distributed networks
- **Custom format parsers** (implementing FormatParser, registering parsers, building NetworkIR)
- **Extending NetworkIR** (custom properties, custom construction logic)
- Advanced debugging techniques

**Best For**: Experienced developers implementing advanced features

---

## Core Documentation

### API_REFERENCE.md
Complete reference for all SNNFrame classes and methods.

**Key Classes**:
- NetworkBuilder - Fluent API for network construction
- Neuron - Pattern-learning neurons
- SpikeProcessor - Spike delivery management
- NetworkPropagator - Neuron firing and propagation
- Datastore - Persistent storage with LRU caching
- ActivityMonitor - Network activity tracking
- ConfigLoader - Configuration management
- DeclarativeLoader - Load networks from configuration files
- NetworkIR - Intermediate representation for all formats
- NetworkConstructor - Build networks from IR
- FormatParser - Interface for custom format parsers

### ARCHITECTURE.md
Detailed architecture documentation.

**Topics**:
- 7-level hierarchical organization
- 6-layer cortical microcircuit
- Connectivity patterns
- Core components
- Performance characteristics

### STDP_GUIDE.md
Comprehensive guide to STDP learning.

**Topics**:
- STDP mathematical formulation
- Implementation details
- Configuration parameters
- Training vs inference modes
- Weight update mechanisms

### TROUBLESHOOTING.md
Solutions to common problems.

**Topics**:
- Build issues
- Runtime errors
- Performance problems
- Accuracy issues
- Memory management

---

## Example Programs

Located in `examples/` directory:

- `neural_network_example.cpp` - Basic network setup
- `hierarchical_structure_example.cpp` - Building hierarchies
- `activity_demo.cpp` - Monitoring activity
- `network_visualization_demo.cpp` - Real-time visualization
- `experiment_config_example.cpp` - Configuration management
- `datastore_example.cpp` - Persistent storage
- `custom_adapter.cpp` - Custom data encoding
- `threading_example.cpp` - Multi-threaded processing

---

## Experiment Programs

Located in `experiments/` directory:

- `emnist_letters_training.cpp` - Headless EMNIST training
- `emnist_letters_visualized.cpp` - EMNIST with visualization
- `retina_mnist.cpp` - Retina-based MNIST processing

---

## Performance Metrics

### EMNIST Letters Classification
- **Accuracy**: ~90% on 26-letter classification
- **Training**: ~60 minutes (5,200 images, 200 per letter)
- **Testing**: ~22 minutes (20,800 images)
- **Network**: 8 orientations × 2 frequencies = 16 columns
- **Neurons**: 392 neurons (49 regions × 8 orientations)

### Key Features
- Cosine similarity-based pattern matching
- STDP frozen during testing to prevent weight drift
- Multi-column architecture with saccade-based attention
- 6-layer canonical cortical microcircuit

---

## Learning Path

### Beginner
1. Read QUICK_START.md
2. Run example programs
3. Study DEVELOPER_MANUAL.md sections 1-5
4. Try simple network examples

### Intermediate
1. Study DEVELOPER_MANUAL.md sections 6-10
2. Review DEVELOPER_GUIDE_PATTERNS.md
3. Implement custom networks
4. Experiment with different architectures

### Advanced
1. Study DEVELOPER_GUIDE_ADVANCED.md
2. Implement custom similarity metrics
3. Optimize performance
4. Contribute improvements

---

## Key Concepts

### Hierarchical Organization
```
Brain → Hemisphere → Lobe → Region → Nucleus → Column → Layer → Cluster → Neuron
```

### Pattern Learning
Neurons learn temporal spike patterns rather than weights.

### STDP Learning
Spike-Timing-Dependent Plasticity adjusts weights based on spike timing.

### Multi-Column Architecture
Multiple columns with different feature selectivity (orientations, frequencies).

### Saccade-Based Processing
Sequential processing through multiple fixation points.

---

## File Organization

```
SNNFrame/
├── docs/
│   ├── DEVELOPER_MANUAL.md           ← Start here
│   ├── DEVELOPER_GUIDE_PATTERNS.md   ← Common patterns
│   ├── DEVELOPER_GUIDE_ADVANCED.md   ← Advanced topics
│   ├── API_REFERENCE.md              ← API docs
│   ├── ARCHITECTURE.md               ← Architecture
│   ├── STDP_GUIDE.md                 ← STDP learning
│   ├── TROUBLESHOOTING.md            ← Problem solving
│   └── DOCUMENTATION_INDEX.md        ← This file
├── examples/                          ← Working examples
├── experiments/                       ← Full experiments
├── include/snnfw/                    ← Header files
├── src/                              ← Implementation
├── tests/                            ← Unit tests
├── configs/                          ← Configuration files
└── README.md                         ← Project overview
```

---

## Getting Help

1. **Check TROUBLESHOOTING.md** - Solutions to common problems
2. **Review examples/** - Working code samples
3. **Read API_REFERENCE.md** - Detailed API documentation
4. **Study DEVELOPER_GUIDE_PATTERNS.md** - Common patterns
5. **Open an issue on GitHub** - Report bugs or request features

---

## Contributing

See CONTRIBUTING.md for guidelines on:
- Code style and standards
- Testing requirements
- Documentation updates
- Pull request process

---

## License

SNNFrame is released under the MIT License. See LICENSE for details.

---

## Version Information

- **Current Version**: 1.0.0
- **Last Updated**: 2026-01-10
- **Status**: Production-ready
- **Accuracy**: ~90% on EMNIST letters classification

---

## Quick Reference

### Load a Network from Config (Declarative)
```cpp
DeclarativeLoader loader(factory, datastore);
auto network = loader.loadNetwork("my_network.snnf.json");
// network.brain, network.spikeProcessor, network.propagator are ready
```

### Create a Network (Programmatic)
```cpp
NetworkBuilder builder(datastore);
auto brain = builder.createBrain();
auto hemisphere = builder.createHemisphere(brain);
// ... continue building hierarchy
```

### Inject Spikes
```cpp
neuron->insertSpike(10.0);
neuron->insertSpike(15.0);
neuron->learnCurrentPattern();
```

### Process Spikes
```cpp
SpikeProcessor processor(10000, 24);
processor.start();
processor.update(1.0);  // 1ms
processor.stop();
```

### Monitor Activity
```cpp
ActivityMonitor monitor(1000);
monitor.buildHierarchicalCache(brain->getId());
auto snapshot = monitor.getActivitySnapshot();
```

---

**For detailed information, start with [DEVELOPER_MANUAL.md](DEVELOPER_MANUAL.md)**

