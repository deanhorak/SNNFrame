# SNNFrame Developer Guide: Common Patterns

This guide covers common patterns and recipes for developing with SNNFrame.

## Table of Contents

1. [Creating Networks](#creating-networks)
2. [Spike Injection](#spike-injection)
3. [Pattern Learning](#pattern-learning)
4. [Classification](#classification)
5. [Monitoring and Debugging](#monitoring-and-debugging)
6. [Data Encoding](#data-encoding)
7. [Multi-Column Architectures](#multi-column-architectures)
8. [Training vs Testing](#training-vs-testing)

---

## Creating Networks

### Simple Feed-Forward Network

```cpp
#include "snnfw/NetworkBuilder.h"
#include "snnfw/Datastore.h"

Datastore datastore("./network_db");
NetworkBuilder builder(datastore);

// Create hierarchy
auto brain = builder.createBrain("SimpleBrain");
auto hemisphere = builder.createHemisphere(brain, "Left");
auto lobe = builder.createLobe(hemisphere, "Sensory");
auto region = builder.createRegion(lobe, "Input");
auto nucleus = builder.createNucleus(region, "Layer4");
auto column = builder.createColumn(nucleus, "Col1");
auto layer = builder.createLayer(column, "Layer2/3");
auto cluster = builder.createCluster(layer);

// Create neurons
std::vector<std::shared_ptr<Neuron>> neurons;
for (int i = 0; i < 100; ++i) {
    neurons.push_back(builder.createNeuron(cluster));
}
```

### Multi-Layer Network

```cpp
// Create multiple layers
std::vector<std::shared_ptr<Layer>> layers;
for (int l = 0; l < 6; ++l) {
    auto layer = builder.createLayer(column, "Layer" + std::to_string(l));
    layers.push_back(layer);
}

// Create neurons in each layer
std::vector<std::vector<std::shared_ptr<Neuron>>> layerNeurons(6);
for (int l = 0; l < 6; ++l) {
    for (int n = 0; n < 50; ++n) {
        auto neuron = builder.createNeuron(layers[l]);
        layerNeurons[l].push_back(neuron);
    }
}

// Connect layers
for (int l = 0; l < 5; ++l) {
    for (auto& srcNeuron : layerNeurons[l]) {
        for (auto& dstNeuron : layerNeurons[l + 1]) {
            // Create synapse from layer l to layer l+1
            auto synapse = std::make_shared<Synapse>(
                srcNeuron->getAxonId(),
                dstNeuron->getDendriteIds()[0],
                0.5,  // weight
                1.0,  // delay
                synapseId++
            );
            processor.registerSynapse(synapse);
        }
    }
}
```

---

## Spike Injection

### Inject Spikes into Neurons

```cpp
// Direct spike injection
neuron->insertSpike(10.0);  // Spike at 10ms
neuron->insertSpike(15.0);  // Spike at 15ms
neuron->insertSpike(20.0);  // Spike at 20ms

// Learn the pattern
neuron->learnCurrentPattern();
```

### Scheduled Spike Delivery

```cpp
// Create action potential
auto ap = std::make_shared<ActionPotential>(
    dendriteId,  // target dendrite
    10.0,        // delivery time (ms)
    0.8          // weight
);

// Schedule for delivery
processor.scheduleSpike(ap);

// Process spikes
processor.update(1.0);  // Advance 1ms
```

### Stimulus Encoding

```cpp
// Encode stimulus as spike times
std::vector<double> spikeTimes;
for (int i = 0; i < stimulusSize; ++i) {
    if (stimulus[i] > threshold) {
        // Encode as spike time proportional to intensity
        double spikeTime = (stimulus[i] / 255.0) * windowSize;
        spikeTimes.push_back(spikeTime);
    }
}

// Inject into neuron
for (double t : spikeTimes) {
    neuron->insertSpike(t);
}
```

---

## Pattern Learning

### Automatic Pattern Learning

```cpp
// Neurons learn patterns automatically
for (int example = 0; example < numExamples; ++example) {
    // Inject spikes for this example
    for (auto& neuron : neurons) {
        neuron->clearSpikes();
        
        // Encode input
        for (int i = 0; i < inputSize; ++i) {
            if (input[example][i] > threshold) {
                neuron->insertSpike(i * timeStep);
            }
        }
        
        // Learn the pattern
        neuron->learnCurrentPattern();
    }
}
```

### Examining Learned Patterns

```cpp
// Get number of patterns
size_t patternCount = neuron->getLearnedPatternCount();
std::cout << "Learned " << patternCount << " patterns\n";

// Get all patterns
const auto& patterns = neuron->getLearnedPatterns();
for (size_t i = 0; i < patterns.size(); ++i) {
    std::cout << "Pattern " << i << ": " << patterns[i].toString() << "\n";
}

// Get best similarity
double bestSim = neuron->getBestSimilarity();
std::cout << "Best similarity: " << bestSim << "\n";
```

### Pattern Matching

```cpp
// Check if current pattern matches learned patterns
if (neuron->checkShouldFire()) {
    std::cout << "Pattern matched!\n";
    
    // Get activation level
    double activation = neuron->getActivation();
    std::cout << "Activation: " << activation << "\n";
}
```

---

## Classification

### k-NN Classification

```cpp
// Collect activation vectors from all neurons
std::vector<std::vector<double>> activations;
for (auto& neuron : neurons) {
    std::vector<double> activation;
    activation.push_back(neuron->getBestSimilarity());
    activations.push_back(activation);
}

// Classify using k-NN
int predictedClass = classifyKNN(activations, k);
```

### Voting-Based Classification

```cpp
// Count votes from output neurons
std::vector<int> votes(numClasses, 0);

for (int classId = 0; classId < numClasses; ++classId) {
    for (auto& neuron : outputNeurons[classId]) {
        if (neuron->checkShouldFire()) {
            votes[classId]++;
        }
    }
}

// Find class with most votes
int predictedClass = std::max_element(
    votes.begin(), votes.end()
) - votes.begin();
```

### Confidence Scoring

```cpp
// Calculate confidence based on activation
double maxActivation = 0.0;
int predictedClass = -1;

for (int classId = 0; classId < numClasses; ++classId) {
    double classActivation = 0.0;
    for (auto& neuron : outputNeurons[classId]) {
        classActivation += neuron->getActivation();
    }
    
    if (classActivation > maxActivation) {
        maxActivation = classActivation;
        predictedClass = classId;
    }
}

double confidence = maxActivation / totalActivation;
```

---

## Monitoring and Debugging

### Activity Monitoring

```cpp
ActivityMonitor monitor(1000);  // 1000ms history

// Build cache
monitor.buildHierarchicalCache(brain->getId());

// Get activity at different levels
auto brainActivity = monitor.getBrainActivity(brain->getId());
auto layerActivity = monitor.getLayerActivity(layer->getId());
auto clusterActivity = monitor.getClusterActivity(cluster->getId());

// Print statistics
std::cout << "Brain spikes: " << brainActivity.spikeCount << "\n";
std::cout << "Layer spikes: " << layerActivity.spikeCount << "\n";
```

### Network Inspection

```cpp
NetworkInspector inspector(datastore);

// Get network statistics
auto stats = inspector.getNetworkStatistics(brain->getId());
std::cout << "Total neurons: " << stats.neuronCount << "\n";
std::cout << "Total synapses: " << stats.synapseCount << "\n";

// Validate network
bool valid = inspector.validateNetwork(brain->getId());
if (!valid) {
    auto errors = inspector.getValidationErrors();
    for (const auto& error : errors) {
        std::cerr << "Error: " << error << "\n";
    }
}
```

### Logging

```cpp
#include "snnfw/Logger.h"

// Initialize logger
Logger::getInstance().initialize("app.log", spdlog::level::info);

// Log messages
SNNFW_INFO("Training started");
SNNFW_DEBUG("Processing example {}", exampleId);
SNNFW_WARN("Low accuracy: {}", accuracy);
SNNFW_ERROR("Failed to load data");
```

---

## Data Encoding

### Image to Spikes (Gabor Filters)

```cpp
// Apply Gabor filter to image
std::vector<double> gaborResponse = applyGaborFilter(image, orientation, frequency);

// Encode as spike times
std::vector<double> spikeTimes;
for (int i = 0; i < gaborResponse.size(); ++i) {
    if (gaborResponse[i] > threshold) {
        // Spike time proportional to response magnitude
        double spikeTime = (gaborResponse[i] / maxResponse) * windowSize;
        spikeTimes.push_back(spikeTime);
    }
}

// Inject into neuron
for (double t : spikeTimes) {
    neuron->insertSpike(t);
}
```

### Temporal Encoding

```cpp
// Encode value as spike latency
double value = input[i];  // 0.0 to 1.0
double spikeTime = (1.0 - value) * windowSize;  // Earlier = higher value
neuron->insertSpike(spikeTime);
```

### Population Coding

```cpp
// Use multiple neurons to encode single value
double value = input[i];  // 0.0 to 1.0
for (int n = 0; n < populationSize; ++n) {
    double preferredValue = (double)n / populationSize;
    double distance = std::abs(value - preferredValue);
    
    if (distance < tuningWidth) {
        // Neuron fires if value is near preferred value
        double spikeTime = distance * windowSize;
        neurons[n]->insertSpike(spikeTime);
    }
}
```

---

## Multi-Column Architectures

### Orientation Columns

```cpp
// Create columns for different orientations
int numOrientations = 8;
std::vector<std::shared_ptr<Column>> orientationColumns;

for (int o = 0; o < numOrientations; ++o) {
    auto column = builder.createColumn(nucleus, "Orientation" + std::to_string(o));
    orientationColumns.push_back(column);
    
    // Create neurons in this column
    for (int n = 0; n < neuronsPerColumn; ++n) {
        auto neuron = builder.createNeuron(column);
        // Configure for this orientation
        neuron->setSimilarityMetric(SimilarityMetric::COSINE);
    }
}
```

### Frequency Columns

```cpp
// Create columns for different frequencies
int numFrequencies = 2;
std::vector<std::shared_ptr<Column>> frequencyColumns;

for (int f = 0; f < numFrequencies; ++f) {
    auto column = builder.createColumn(nucleus, "Frequency" + std::to_string(f));
    frequencyColumns.push_back(column);
}
```

### Spatial Tiling

```cpp
// Create grid of columns
int gridSize = 7;  // 7x7 grid
std::vector<std::vector<std::shared_ptr<Column>>> columnGrid(gridSize);

for (int x = 0; x < gridSize; ++x) {
    columnGrid[x].resize(gridSize);
    for (int y = 0; y < gridSize; ++y) {
        auto column = builder.createColumn(
            nucleus, 
            "Col_" + std::to_string(x) + "_" + std::to_string(y)
        );
        columnGrid[x][y] = column;
    }
}
```

---

## Training vs Testing

### Training Phase

```cpp
// Enable STDP learning
processor.setSTDPEnabled(true);

// Training loop
for (int epoch = 0; epoch < numEpochs; ++epoch) {
    for (int example = 0; example < trainingSet.size(); ++example) {
        // Inject spikes
        injectSpikes(neurons, trainingSet[example]);
        
        // Process
        processor.update(windowSize);
        
        // Learn patterns
        for (auto& neuron : neurons) {
            neuron->learnCurrentPattern();
        }
    }
}
```

### Testing Phase

```cpp
// Disable STDP to prevent weight drift
processor.setSTDPEnabled(false);

// Testing loop
int correct = 0;
for (int example = 0; example < testSet.size(); ++example) {
    // Inject spikes
    injectSpikes(neurons, testSet[example]);
    
    // Process
    processor.update(windowSize);
    
    // Classify
    int predicted = classify(neurons);
    if (predicted == testSet[example].label) {
        correct++;
    }
}

double accuracy = (double)correct / testSet.size();
std::cout << "Accuracy: " << accuracy * 100 << "%\n";
```

### Checkpoint Saving

```cpp
// Save network state
datastore.flushAll();

// Save configuration
ConfigLoader config;
config.save("checkpoint_config.json");

// Save statistics
std::ofstream statsFile("checkpoint_stats.txt");
statsFile << "Epoch: " << epoch << "\n";
statsFile << "Accuracy: " << accuracy << "\n";
statsFile.close();
```

---

## Performance Tips

### Memory Optimization

```cpp
// Reduce cache size for memory-constrained systems
Datastore datastore("./db", 100000);  // 100K objects instead of 1M

// Flush frequently
if (example % 100 == 0) {
    datastore.flushAll();
}

// Clear spikes periodically
for (auto& neuron : neurons) {
    neuron->clearSpikes();
}
```

### Parallelization

```cpp
// Use multiple threads for spike delivery
SpikeProcessor processor(10000, 24);  // 24 delivery threads

// Process in parallel
#pragma omp parallel for
for (int i = 0; i < neurons.size(); ++i) {
    neurons[i]->learnCurrentPattern();
}
```

### Profiling

```cpp
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();

// ... code to profile ...

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
std::cout << "Time: " << duration.count() << "ms\n";
```

---

**Last Updated**: 2026-01-10
**Version**: 1.0.0

