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

## See Also

- [Architecture Guide](ARCHITECTURE.md)
- [Configuration Guide](CONFIGURATION.md)
- [Examples](../examples/)

