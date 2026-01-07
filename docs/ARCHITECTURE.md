# SNNFrame Architecture

## Hierarchical Organization

SNNFrame implements a 7-level hierarchical organization inspired by neuroscience:

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

## Cortical Microcircuit (6-Layer Model)

The framework implements a canonical cortical microcircuit with 6 layers:

### Layer 1: Apical Dendrites
- Receives modulatory inputs
- Integrates feedback from higher areas
- Influences dendritic computation

### Layer 2/3: Superficial Pyramidal Neurons
- Local processing and lateral connections
- Integrates input from Layer 4
- Projects to Layer 5

### Layer 4: Granular Input Layer
- Primary sensory/thalamic input
- Feedforward processing
- Strongest input layer

### Layer 5: Deep Pyramidal Neurons
- Output layer
- Projects to subcortical structures
- Receives feedback from Layer 6

### Layer 6: Corticothalamic Feedback
- Feedback to thalamus
- Modulates input processing
- Closes feedback loop

## Connectivity Patterns

### Feedforward
```
Input → L4 → L2/3 → L5 → Output
```

### Feedback
```
L6 → Thalamus → L4
```

### Lateral
```
Within-layer connections for competition and cooperation
```

### Recurrent
```
L5 → L2/3 for temporal integration
```

## Core Components

### SpikeProcessor
- **Role**: Manages spike delivery and timing
- **Features**:
  - Multi-threaded spike delivery
  - Time-slice based processing
  - Real-time synchronization option
  - STDP application with enable/disable control

### NetworkPropagator
- **Role**: Manages neuron firing and spike propagation
- **Features**:
  - Neuron registration and management
  - Spike propagation to synapses
  - STDP learning with retrograde signaling
  - Runtime STDP enable/disable for inference

### Neuron
- **Role**: Implements spiking neuron model
- **Features**:
  - Temporal pattern matching
  - Configurable learning window
  - Similarity-based pattern recognition
  - Spike generation and propagation

### Synapse
- **Role**: Implements synaptic connection
- **Features**:
  - Learnable weight
  - Transmission delay
  - STDP-based weight updates
  - Weight bounds [0, 2]

### ActivityMonitor
- **Role**: Tracks neural activity
- **Features**:
  - Hierarchical activity tracking
  - Real-time activity snapshots
  - Historical activity queries
  - Efficient caching

## Learning Mechanisms

### STDP (Spike-Timing-Dependent Plasticity)

Classic STDP learning rule:
- **LTP** (Long-Term Potentiation): Pre-spike before post-spike → strengthen
- **LTD** (Long-Term Depression): Post-spike before pre-spike → weaken
- **Magnitude**: Exponentially decreases with time difference

### Retrograde Signaling

- Post-synaptic neuron fires
- Retrograde signal travels back to synapse
- STDP applied based on pre-synaptic spike timing
- Enables causal learning

### STDP Enable/Disable

- **Training Mode**: STDP enabled (default)
- **Inference Mode**: STDP disabled to prevent weight drift
- **Runtime Control**: `setStdpEnabled(bool)` on both NetworkPropagator and SpikeProcessor
- **Thread-Safe**: Uses atomic flags for safe concurrent access

## Data Storage

### RocksDB Integration
- Persistent storage of network structure
- LRU caching for memory efficiency
- Automatic dirty-tracking
- Flush-on-eviction strategy

### Datastore
- Manages all persistent objects
- Provides transaction support
- Efficient key-value storage
- Hierarchical object relationships

## Visualization Pipeline

### Components
1. **VisualizationManager**: Main visualization controller
2. **NetworkGraphRenderer**: Renders network structure
3. **ActivityVisualizer**: Visualizes neural activity
4. **SpikeRenderer**: Renders individual spikes
5. **RasterPlotRenderer**: Raster plot visualization
6. **ActivityHistogram**: Activity distribution

### Features
- Real-time 3D rendering with OpenGL
- ImGui-based UI
- Interactive camera control
- Activity heatmaps
- Spike trails

## Recording and Playback

### Recording
- Binary format for efficiency
- Streaming or memory mode
- Metadata tracking
- Spike timestamps and IDs

### Playback
- Replay recorded sessions
- Speed control
- Looping support
- Visualization integration

## Performance Characteristics

### Multi-Threading
- Configurable thread pool (default: 20 threads)
- Parallel spike delivery
- Lock-free data structures where possible
- Minimal synchronization overhead

### Optimization
- Release builds with -O3 and LTO
- SIMD-friendly data layouts
- Cache-aware algorithms
- Efficient memory management

### Scaling
- Linear scaling with thread count (up to 24 threads tested)
- Supports brain-scale networks
- Hierarchical organization enables modularity

## Extension Points

### Custom Encoders
Implement `Encoder` interface for custom input encoding

### Custom Classifiers
Implement `Classifier` interface for custom classification

### Custom Learning Strategies
Implement `LearningStrategy` interface for pattern learning

### Custom Adapters
Implement `Adapter` interface for custom data sources

## See Also

- [API Reference](API_REFERENCE.md)
- [Configuration Guide](CONFIGURATION.md)
- [Examples](../examples/)

