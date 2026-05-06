# SNNFrame Developer Guide: Advanced Topics

Advanced techniques and patterns for expert SNNFrame developers.

## Table of Contents

1. [Custom Similarity Metrics](#custom-similarity-metrics)
2. [Connectivity Patterns](#connectivity-patterns)
3. [STDP Customization](#stdp-customization)
4. [Homeostatic Plasticity](#homeostatic-plasticity)
5. [Attention Mechanisms](#attention-mechanisms)
6. [Saccade-Based Processing](#saccade-based-processing)
7. [Recording and Playback](#recording-and-playback)
8. [Performance Profiling](#performance-profiling)
9. [Custom Adapters](#custom-adapters)
10. [Distributed Networks](#distributed-networks)
11. [Custom Format Parsers](#custom-format-parsers)
12. [Extending NetworkIR](#extending-networkir)

---

## Custom Similarity Metrics

### Available Metrics

SNNFrame supports multiple similarity metrics for pattern matching:

```cpp
enum class SimilarityMetric {
    COSINE,      // Cosine similarity (default)
    HISTOGRAM,   // Histogram intersection
    EUCLIDEAN,   // Euclidean distance
    CORRELATION, // Pearson correlation
    WAVEFORM     // Temporal shape matching
};

// Set metric for a neuron
neuron->setSimilarityMetric(SimilarityMetric::COSINE);
```

### Cosine Similarity (Default)

Best for most tasks. Measures angle between spike patterns:

```cpp
// Automatically used for pattern matching
// High value = similar patterns
// Low value = different patterns
double similarity = neuron->getBestSimilarity();
```

### Histogram Similarity

Counts spike overlap without considering timing:

```cpp
neuron->setSimilarityMetric(SimilarityMetric::HISTOGRAM);

// Good for tasks where spike count matters more than timing
// Useful for rate-coded representations
```

### Waveform Similarity

Matches temporal shape of spike patterns:

```cpp
neuron->setSimilarityMetric(SimilarityMetric::WAVEFORM);

// Excellent for temporal pattern recognition
// Sensitive to spike timing relationships
```

### Implementing Custom Metrics

To add custom metrics, extend the Neuron class:

```cpp
class CustomNeuron : public Neuron {
private:
    double computeCustomSimilarity(
        const BinaryPattern& a, 
        const BinaryPattern& b) const {
        // Your custom similarity computation
        // Return value between 0.0 and 1.0
        return customMetric;
    }
};
```

---

## Connectivity Patterns

### Lateral Inhibition

Create competition within layers:

```cpp
// Connect each neuron to neighbors
for (size_t i = 0; i < neurons.size(); ++i) {
    for (size_t j = i + 1; j < neurons.size(); ++j) {
        // Inhibitory synapse (negative weight)
        auto synapse = std::make_shared<Synapse>(
            neurons[i]->getAxonId(),
            neurons[j]->getDendriteIds()[0],
            -0.5,  // Negative weight = inhibition
            1.0,
            synapseId++
        );
        processor.registerSynapse(synapse);
    }
}
```

### Winner-Take-All

Implement competition for sparse coding:

```cpp
// Apply inhibition based on activation
double maxActivation = 0.0;
int winnerIdx = -1;

for (size_t i = 0; i < neurons.size(); ++i) {
    double activation = neurons[i]->getActivation();
    if (activation > maxActivation) {
        maxActivation = activation;
        winnerIdx = i;
    }
}

// Inhibit all but winner
for (size_t i = 0; i < neurons.size(); ++i) {
    if (i != winnerIdx) {
        neurons[i]->applyInhibition(maxActivation);
    }
}
```

### Recurrent Connections

Create feedback loops for temporal integration:

```cpp
// L5 → L2/3 recurrent connection
for (auto& l5Neuron : layer5Neurons) {
    for (auto& l23Neuron : layer23Neurons) {
        auto synapse = std::make_shared<Synapse>(
            l5Neuron->getAxonId(),
            l23Neuron->getDendriteIds()[0],
            0.3,   // Recurrent weight
            2.0,   // Recurrent delay (ms)
            synapseId++
        );
        processor.registerSynapse(synapse);
    }
}
```

### Sparse Connectivity

Reduce connections for efficiency:

```cpp
// Connect only nearby neurons
double connectionProbability = 0.1;  // 10% connectivity
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

for (auto& srcNeuron : sourceNeurons) {
    for (auto& dstNeuron : targetNeurons) {
        if (dis(gen) < connectionProbability) {
            auto synapse = std::make_shared<Synapse>(
                srcNeuron->getAxonId(),
                dstNeuron->getDendriteIds()[0],
                0.5,
                1.0,
                synapseId++
            );
            processor.registerSynapse(synapse);
        }
    }
}
```

---

## STDP Customization

### Enable/Disable STDP

Control learning during training and testing:

```cpp
// Training phase - enable STDP
processor.setSTDPEnabled(true);

// ... training loop ...

// Testing phase - disable STDP
processor.setSTDPEnabled(false);

// ... testing loop ...
```

### STDP Parameters

Customize learning rates and time windows:

```cpp
// STDP uses spike timing differences
// Positive timing (post before pre) = potentiation
// Negative timing (pre before post) = depression

// Typical parameters:
double learningRate = 0.01;      // How much to change weights
double potentiationWindow = 20.0; // ms for potentiation
double depressionWindow = 20.0;   // ms for depression
```

### Freeze STDP During Testing

Prevent weight drift during inference:

```cpp
// Before testing
processor.setSTDPEnabled(false);

// Test loop
for (auto& example : testSet) {
    injectSpikes(neurons, example);
    processor.update(windowSize);
    
    // Weights remain unchanged
    int prediction = classify(neurons);
}

// Re-enable for next training phase
processor.setSTDPEnabled(true);
```

---

## Homeostatic Plasticity

### Firing Rate Regulation

Maintain stable firing rates:

```cpp
// Set target firing rate
double targetRate = 10.0;  // Hz
neuron->setTargetFiringRate(targetRate);

// Update firing rate periodically
for (int t = 0; t < simulationTime; ++t) {
    processor.update(1.0);
    
    if (t % 100 == 0) {  // Every 100ms
        neuron->updateFiringRate(t);
        neuron->applyHomeostaticPlasticity();
    }
}
```

### Intrinsic Excitability

Adjust neuron responsiveness:

```cpp
// Get current firing rate
double firingRate = neuron->getFiringRate();
double targetRate = neuron->getTargetFiringRate();

// If firing too slowly, increase excitability
// If firing too fast, decrease excitability
neuron->applyHomeostaticPlasticity();
```

---

## Attention Mechanisms

### Spatial Attention

Focus on specific regions:

```cpp
// Define attention region
struct AttentionRegion {
    int x_start, x_end;
    int y_start, y_end;
};

AttentionRegion focus = {5, 15, 5, 15};

// Boost neurons in focus region
for (auto& neuron : neurons) {
    if (isInRegion(neuron, focus)) {
        neuron->applyInhibition(-0.5);  // Negative = boost
    } else {
        neuron->applyInhibition(0.5);   // Positive = suppress
    }
}
```

### Feature-Based Attention

Attend to specific features:

```cpp
// Boost neurons selective for target feature
for (auto& neuron : neurons) {
    if (neuron->getPreferredFeature() == targetFeature) {
        neuron->applyInhibition(-0.3);  // Boost
    }
}
```

---

## Saccade-Based Processing

### Sequential Fixations

Process image through multiple fixations:

```cpp
struct Fixation {
    int x, y;
    int duration_ms;
};

std::vector<Fixation> fixations = {
    {7, 7, 100},    // Center
    {0, 0, 100},    // Top-left
    {14, 14, 100},  // Bottom-right
    {7, 14, 100}    // Bottom-center
};

// Process each fixation
for (const auto& fixation : fixations) {
    // Extract region around fixation
    auto region = extractRegion(image, fixation.x, fixation.y);
    
    // Encode and process
    injectSpikes(neurons, region);
    processor.update(fixation.duration_ms);
    
    // Accumulate evidence
    accumulateActivation(neurons);
}

// Classify based on accumulated evidence
int prediction = classifyAccumulated();
```

### Attention Saccades

Dynamically choose fixation points:

```cpp
// Start with center fixation
Fixation current = {7, 7, 100};

for (int saccade = 0; saccade < maxSaccades; ++saccade) {
    // Process current fixation
    injectSpikes(neurons, current);
    processor.update(current.duration_ms);
    
    // Compute attention map
    auto attentionMap = computeAttentionMap(neurons);
    
    // Find next fixation point
    current = findMaxAttention(attentionMap);
}
```

---

## Recording and Playback

### Record Network Activity

```cpp
#include "snnfw/RecordingManager.h"

RecordingManager recorder;
recorder.startRecording("session.snnr");

// Run simulation
for (int t = 0; t < simulationTime; ++t) {
    processor.update(1.0);
    
    // Record activity
    recorder.recordActivity(neurons, t);
}

recorder.stopRecording();
```

### Playback Recorded Session

```cpp
RecordingManager player;
player.loadRecording("session.snnr");

// Playback
while (player.hasMoreFrames()) {
    auto frame = player.getNextFrame();
    visualizer.renderFrame(frame);
}
```

---

## Performance Profiling

### Timing Analysis

```cpp
#include <chrono>

struct TimingStats {
    double spikeProcessing = 0.0;
    double patternLearning = 0.0;
    double classification = 0.0;
};

TimingStats stats;

// Profile spike processing
auto start = std::chrono::high_resolution_clock::now();
processor.update(1.0);
auto end = std::chrono::high_resolution_clock::now();
stats.spikeProcessing += std::chrono::duration<double>(end - start).count();

// Profile pattern learning
start = std::chrono::high_resolution_clock::now();
for (auto& neuron : neurons) {
    neuron->learnCurrentPattern();
}
end = std::chrono::high_resolution_clock::now();
stats.patternLearning += std::chrono::duration<double>(end - start).count();

// Print results
std::cout << "Spike processing: " << stats.spikeProcessing << "s\n";
std::cout << "Pattern learning: " << stats.patternLearning << "s\n";
```

### Memory Profiling

```cpp
// Get datastore statistics
auto stats = datastore.getStatistics();
std::cout << "Cache hits: " << stats.cacheHits << "\n";
std::cout << "Cache misses: " << stats.cacheMisses << "\n";
std::cout << "Hit rate: " << (double)stats.cacheHits / 
    (stats.cacheHits + stats.cacheMisses) * 100 << "%\n";

// Monitor memory usage
std::cout << "Objects in cache: " << stats.objectsInCache << "\n";
std::cout << "Objects on disk: " << stats.objectsOnDisk << "\n";
```

---

## Custom Adapters

### Data Encoding Adapter

```cpp
class CustomAdapter {
public:
    virtual std::vector<double> encode(const InputData& input) = 0;
    virtual InputData decode(const std::vector<double>& spikes) = 0;
};

class GaborAdapter : public CustomAdapter {
public:
    std::vector<double> encode(const InputData& input) override {
        // Apply Gabor filters
        std::vector<double> result;
        for (int orientation = 0; orientation < 8; ++orientation) {
            auto response = applyGabor(input, orientation);
            result.push_back(response);
        }
        return result;
    }
};
```

### Custom Classifier

```cpp
class CustomClassifier {
public:
    virtual int classify(const std::vector<std::shared_ptr<Neuron>>& neurons) = 0;
};

class EnsembleClassifier : public CustomClassifier {
private:
    std::vector<std::shared_ptr<CustomClassifier>> classifiers;
    
public:
    int classify(const std::vector<std::shared_ptr<Neuron>>& neurons) override {
        std::vector<int> votes(numClasses, 0);
        
        for (auto& classifier : classifiers) {
            votes[classifier->classify(neurons)]++;
        }
        
        return std::max_element(votes.begin(), votes.end()) - votes.begin();
    }
};
```

---

## Distributed Networks

### Multi-Process Communication

```cpp
#include <zmq.hpp>

// Process A: Send spikes
zmq::context_t context(1);
zmq::socket_t socket(context, zmq::socket_type::push);
socket.bind("tcp://127.0.0.1:5555");

// Send spike data
std::string spikeData = serializeSpikes(neurons);
socket.send(zmq::buffer(spikeData), zmq::send_flags::none);

// Process B: Receive spikes
zmq::socket_t receiver(context, zmq::socket_type::pull);
receiver.connect("tcp://127.0.0.1:5555");

zmq::message_t message;
receiver.recv(message, zmq::recv_flags::none);
auto spikes = deserializeSpikes(message.to_string());
```

### Network Synchronization

```cpp
// Synchronize multiple networks
class NetworkCluster {
private:
    std::vector<std::shared_ptr<SpikeProcessor>> processors;
    
public:
    void synchronizeStep(double timeStep) {
        // All processors advance by same amount
        for (auto& processor : processors) {
            processor->update(timeStep);
        }
        
        // Exchange spikes between networks
        exchangeSpikes();
    }
    
private:
    void exchangeSpikes() {
        // Collect spikes from all networks
        // Deliver to appropriate targets
    }
};
```

---

## Debugging Advanced Issues

### Spike Propagation Debugging

```cpp
// Enable detailed logging
Logger::getInstance().setLevel(spdlog::level::debug);

// Trace spike delivery
processor.setDebugMode(true);

// Check dendrite registration
auto registeredDendrites = processor.getRegisteredDendrites();
std::cout << "Registered dendrites: " << registeredDendrites.size() << "\n";

// Verify synapse connectivity
for (auto& synapse : synapses) {
    bool valid = processor.validateSynapse(synapse);
    if (!valid) {
        std::cerr << "Invalid synapse: " << synapse->getId() << "\n";
    }
}
```

### Pattern Learning Debugging

```cpp
// Examine pattern learning process
neuron->printSpikes();           // Show current spikes
neuron->printReferencePatterns(); // Show learned patterns

// Check similarity computation
double similarity = neuron->getBestSimilarity();
std::cout << "Best similarity: " << similarity << "\n";

// Verify pattern capacity
size_t learned = neuron->getLearnedPatternCount();
size_t max = neuron->getMaxReferencePatterns();
std::cout << "Patterns: " << learned << "/" << max << "\n";
```

---

## Custom Format Parsers

You can add support for new network description formats by implementing the `FormatParser` interface.

### FormatParser Interface

```cpp
#include "snnfw/declarative/FormatParser.h"

class MyCustomParser : public FormatParser {
public:
    // Parse a file and return the intermediate representation
    NetworkIR parse(const std::string& filepath) override {
        NetworkIR ir;
        // Read your format and populate the IR fields:
        // ir.brain, ir.projections, ir.neuron_params, etc.
        return ir;
    }

    // Return true if this parser can handle the given file
    bool canParse(const std::string& filepath) const override {
        return filepath.ends_with(".myformat");
    }

    // Return a human-readable name for the format
    std::string formatName() const override {
        return "MyCustomFormat";
    }
};
```

### Registering a Custom Parser

```cpp
DeclarativeLoader loader(factory, datastore);

// Register your custom parser
loader.registerParser(std::make_unique<MyCustomParser>());

// Now loadNetwork() will try your parser for matching files
auto network = loader.loadNetwork("model.myformat");
```

### Building a NetworkIR

The key structures you need to populate:

```cpp
NetworkIR ir;

// 1. Neuron parameter sets (referenced by name)
ir.neuron_params["excitatory"] = {500.0, 0.93, 500, "cosine"};

// 2. Brain hierarchy
ir.brain.name = "MyBrain";
HemisphereIR hem; hem.name = "Left";
LobeIR lobe; lobe.name = "Visual";
RegionIR region; region.name = "V1";
NucleusIR nucleus; nucleus.name = "Columns";
ColumnIR col; col.name = "Col1";
LayerIR layer; layer.name = "L4";
PopulationIR pop; pop.name = "cells"; pop.count = 49;
pop.neuron_params_ref = "excitatory";
layer.populations.push_back(pop);
col.layers.push_back(layer);
nucleus.columns.push_back(col);
region.nuclei.push_back(nucleus);
lobe.regions.push_back(region);
hem.lobes.push_back(lobe);
ir.brain.hemispheres.push_back(hem);

// 3. Projections
ProjectionIR proj;
proj.name = "L4_to_L5";
proj.source = "V1/*/L4";
proj.target = "V1/*/L5";
proj.pattern = "random_sparse";
proj.probability = 0.25;
proj.weight = 0.1;
ir.projections.push_back(proj);

// 4. Validate before use
auto errors = ir.validate();
```

### NetworkIR Validation

The `validate()` method checks for:
- Complete hierarchy: brain → hemisphere → lobe → region → nucleus → column → layer
- Valid neuron parameter references (every `neuron_params_ref` exists in `neuron_params`)
- Valid input/output layer definitions
- Non-empty population counts

---

## Extending NetworkIR

### Adding Custom Properties

If your format has properties not covered by the standard IR, use the `properties` map in each IR struct:

```cpp
// Add custom properties during parsing
PopulationIR pop;
pop.name = "L4_cells";
pop.count = 49;
pop.properties["compartment_model"] = "HH";
pop.properties["ion_channels"] = "Na,K,Ca";
```

### Custom Construction Logic

Override `NetworkConstructor` behavior by subclassing:

```cpp
class MyConstructor : public NetworkConstructor {
public:
    using NetworkConstructor::NetworkConstructor;

    ConstructedNetwork construct(const NetworkIR& ir) override {
        auto network = NetworkConstructor::construct(ir);
        // Add custom post-construction logic
        configureIonChannels(network, ir);
        return network;
    }

private:
    void configureIonChannels(ConstructedNetwork& net, const NetworkIR& ir) {
        // Custom logic for your format extensions
    }
};
```

---

**Last Updated**: 2026-02-06
**Version**: 1.1.0

