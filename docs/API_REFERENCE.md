# SNNFrame API Reference

## Core Classes

### NetworkBuilder

Fluent API for constructing hierarchical neural networks.

```cpp
NetworkBuilder builder(datastore);
auto brain = builder.createBrain()
    .addHemispheres(2)
    .addLobes(4)
    .addRegions(8)
    .addNuclei(16)
    .addColumns(24)
    .addLayers(6)
    .addClusters(100, 50)  // 100 clusters with 50 neurons each
    .build();
```

**Key Methods:**
- `createBrain()` - Create root brain object
- `addHemispheres(count)` - Add hemispheres
- `addLobes(count)` - Add lobes
- `addRegions(count)` - Add regions
- `addNuclei(count)` - Add nuclei
- `addColumns(count)` - Add columns
- `addLayers(count)` - Add layers
- `addClusters(count, neuronsPerCluster)` - Add clusters with neurons
- `build()` - Finalize and return brain

### SpikeProcessor

Manages spike delivery and STDP learning with multi-threaded processing.

```cpp
auto processor = std::make_shared<SpikeProcessor>(10000, 20);  // 10000 time slices, 20 threads
processor.start();
processor.setSTDPParameters(0.05, 0.05, 20.0, 20.0);  // A+, A-, τ+, τ-
processor.setStdpEnabled(true);   // Enable STDP learning
processor.setStdpEnabled(false);  // Disable STDP (inference mode)
processor.stop();
```

**Key Methods:**
- `start()` - Start spike processing
- `stop()` - Stop spike processing
- `setSTDPParameters(aPlus, aMinus, tauPlus, tauMinus)` - Configure STDP
- `setStdpEnabled(bool)` - Enable/disable STDP learning
- `isStdpEnabled()` - Query STDP state
- `update(deltaTime)` - Process spikes for time interval

### NetworkPropagator

Manages neuron firing, spike propagation, and STDP application.

```cpp
auto propagator = std::make_shared<NetworkPropagator>(spikeProcessor);
propagator->setSTDPParameters(0.05, 0.05, 20.0, 20.0);
propagator->setStdpEnabled(true);   // Training mode
propagator->setStdpEnabled(false);  // Inference mode
propagator->fireNeuron(neuronId, firingTime);
```

**Key Methods:**
- `registerNeuron(neuron)` - Register neuron for propagation
- `registerSynapse(synapse)` - Register synapse for STDP
- `fireNeuron(neuronId, time)` - Fire a neuron
- `setSTDPParameters(...)` - Configure STDP
- `setStdpEnabled(bool)` - Enable/disable STDP
- `isStdpEnabled()` - Query STDP state
- `applySTDP(synapseId, timeDifference)` - Apply STDP to synapse

### Neuron

Represents a spiking neuron with pattern learning.

```cpp
auto neuron = factory.createNeuron(500.0, 0.93, 500);  // window, threshold, max patterns
neuron->injectSpike(100.0);  // Inject spike at 100ms
if (neuron->hasFired()) {
    std::cout << "Neuron fired!" << std::endl;
}
```

**Key Methods:**
- `injectSpike(time)` - Inject spike at specified time
- `hasFired()` - Check if neuron has fired
- `getWeight()` - Get neuron weight
- `setWeight(weight)` - Set neuron weight
- `getId()` - Get neuron ID

### Synapse

Represents a synaptic connection with learnable weight.

```cpp
auto synapse = factory.createSynapse(axonId, dendriteId, weight, delay);
double w = synapse->getWeight();
synapse->setWeight(w + 0.01);  // Strengthen synapse
```

**Key Methods:**
- `getWeight()` - Get synaptic weight
- `setWeight(weight)` - Set synaptic weight
- `getDelay()` - Get transmission delay
- `getId()` - Get synapse ID

### ActivityMonitor

Tracks neural activity across the hierarchy.

```cpp
ActivityMonitor monitor(1000);  // 1000ms history
monitor.buildHierarchicalCache(brain->getId());
auto snapshot = monitor.getActivitySnapshot();
for (const auto& [clusterId, count] : snapshot.clusterActivity) {
    std::cout << "Cluster " << clusterId << ": " << count << " spikes" << std::endl;
}
```

**Key Methods:**
- `recordSpike(neuronId, time)` - Record spike
- `buildHierarchicalCache(brainId)` - Build activity cache
- `getActivitySnapshot()` - Get current activity
- `getHistoricalActivity(startTime, endTime)` - Get activity range

### VisualizationManager

Manages real-time 3D visualization.

```cpp
VisualizationManager vizManager(1920, 1080);
vizManager.initialize();
while (!vizManager.shouldClose()) {
    vizManager.update(deltaTime);
    vizManager.render();
}
```

**Key Methods:**
- `initialize()` - Initialize OpenGL/GLFW
- `update(deltaTime)` - Update visualization
- `render()` - Render frame
- `shouldClose()` - Check if window should close

## Encoding Strategies

### RateEncoder

Encodes input as spike rate.

```cpp
RateEncoder encoder(maxRate);
auto spikes = encoder.encode(inputValue, startTime, duration);
```

### TemporalEncoder

Encodes input as spike timing.

```cpp
TemporalEncoder encoder(maxTime);
auto spikes = encoder.encode(inputValue, startTime);
```

### PopulationEncoder

Encodes input across population of neurons.

```cpp
PopulationEncoder encoder(populationSize);
auto spikes = encoder.encode(inputValue, startTime, duration);
```

## Classification Strategies

### MajorityVoting

Classifies based on majority vote.

```cpp
MajorityVoting classifier;
int label = classifier.classify(neuronOutputs);
```

### WeightedDistance

Classifies based on weighted distance.

```cpp
WeightedDistance classifier;
int label = classifier.classify(neuronOutputs, weights);
```

## Declarative Network Loading

### DeclarativeLoader

Main entry-point for loading networks from configuration files.

```cpp
#include "snnfw/declarative/DeclarativeLoader.h"

NeuralObjectFactory factory;
Datastore datastore("./db");
DeclarativeLoader loader(factory, datastore);

// Load network — format auto-detected from extension
auto network = loader.loadNetwork("configs/my_network.snnf.json");

// Parse only (returns NetworkIR without constructing)
auto ir = loader.parseOnly("configs/my_network.snnf.json");

// Register a custom parser
loader.registerParser(std::make_unique<MyCustomParser>());
```

**Key Methods:**
- `loadNetwork(filepath)` - Parse file and construct the full network
- `parseOnly(filepath)` - Parse file into NetworkIR without construction
- `registerParser(parser)` - Register a custom format parser

**Returned `ConstructedNetwork` contains:**
- `brain` - Fully constructed Brain hierarchy
- `spikeProcessor` - SpikeProcessor ready to start()
- `propagator` - NetworkPropagator with all synapses registered
- `inputNeurons` - Input grid neurons for spike injection
- `outputPopulations` - Output neurons grouped by class
- `columns` - Per-column neuron groups

### NetworkIR

Common intermediate representation shared by all format parsers.

```cpp
#include "snnfw/declarative/NetworkIR.h"

NetworkIR ir;
ir.brain.name = "MyBrain";
// ... populate hierarchy, projections, etc.

// Validate the IR
auto errors = ir.validate();
for (const auto& err : errors) {
    std::cerr << err << std::endl;
}
```

**Key Structures:**
- `NeuronParamsIR` - Neuron parameters (window_size_ms, similarity_threshold, max_reference_patterns, similarity_metric)
- `PopulationIR` - Group of neurons (name, count, neuron_params_ref, grid_layout)
- `LayerIR` - Layer containing populations
- `ColumnIR` - Column containing layers
- `ColumnTemplateIR` - Template for generating multiple columns (orientations, frequencies)
- `NucleusIR` - Nucleus containing columns or column templates
- `RegionIR`, `LobeIR`, `HemisphereIR`, `BrainIR` - Hierarchy levels
- `ProjectionIR` - Connection between populations (source, target, pattern, weight, delay, scope)
- `InputLayerIR` - Input layer definition (rows, cols, latency_ms)
- `OutputLayerIR` - Output layer definition (num_classes, neurons_per_class)
- `GaborConfigIR` - Gabor filter parameters
- `SaccadeConfigIR` - Saccade region definitions
- `SimulationConfigIR` - Spike processor and STDP parameters

### NetworkConstructor

Builds the actual SNNFrame hierarchy from a NetworkIR.

```cpp
#include "snnfw/declarative/NetworkConstructor.h"

NetworkConstructor constructor(factory, datastore);
auto network = constructor.construct(ir);
```

**Key Methods:**
- `construct(ir)` - Four-phase construction: build hierarchy → create neurons → create connectivity → initialize runtime

### FormatParser

Abstract interface for implementing custom format parsers.

```cpp
#include "snnfw/declarative/FormatParser.h"

class MyParser : public FormatParser {
public:
    NetworkIR parse(const std::string& filepath) override;
    bool canParse(const std::string& filepath) const override;
    std::string formatName() const override;
};
```

### Built-in Parsers

| Parser | Class | Extensions | Description |
|--------|-------|------------|-------------|
| Native JSON | `NativeJSONParser` | `.snnf.json` | Full SNNFrame format with column templates |
| SONATA | `SONATAParser` | `circuit_config.json`, `.sonata.json` | HDF5-based Blue Brain / Allen Institute format |
| NeuroML | `NeuroMLParser` | `.nml`, `.neuroml` | XML community standard with `snnfw:` extensions |
| HOC | `HOCParser` | `.hoc` | NEURON simulator scripting language |

## See Also

- [Architecture Guide](ARCHITECTURE.md)
- [Configuration Guide](CONFIGURATION.md)
- [Developer Manual - Declarative Loading](DEVELOPER_MANUAL.md#declarative-network-loading)
- [Format Reference](FORMAT_REFERENCE.md)
- [Examples](../examples/)

