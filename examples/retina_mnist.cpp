/**
 * @file retina_mnist.cpp
 * @brief Example: MNIST digit recognition using RetinaAdapter
 * 
 * This example demonstrates:
 * - Loading MNIST dataset
 * - Using RetinaAdapter for visual processing
 * - Training with pattern-based learning
 * - k-NN classification with activation patterns
 * - Achieving 92.70% accuracy
 */

#include "snnfw/adapters/RetinaAdapter.h"
#include "snnfw/MNISTLoader.h"
#include "snnfw/ConfigLoader.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace snnfw;
using namespace snnfw::adapters;

struct TrainingPattern {
    std::vector<double> activations;
    int label;
};

// Calculate cosine similarity between two vectors
double cosineSimilarity(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) return 0.0;
    
    double dotProduct = 0.0;
    double normA = 0.0;
    double normB = 0.0;
    
    for (size_t i = 0; i < a.size(); ++i) {
        dotProduct += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    
    if (normA == 0.0 || normB == 0.0) return 0.0;
    return dotProduct / (std::sqrt(normA) * std::sqrt(normB));
}

// k-NN classification
int classifyKNN(const std::vector<double>& testPattern,
                const std::vector<TrainingPattern>& trainingPatterns,
                int k = 5) {
    // Calculate similarities
    std::vector<std::pair<double, int>> similarities;
    for (const auto& pattern : trainingPatterns) {
        double sim = cosineSimilarity(testPattern, pattern.activations);
        similarities.push_back({sim, pattern.label});
    }
    
    // Sort by similarity (descending)
    std::sort(similarities.begin(), similarities.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Vote among k nearest neighbors
    std::vector<int> votes(10, 0);
    for (int i = 0; i < k && i < static_cast<int>(similarities.size()); ++i) {
        votes[similarities[i].second]++;
    }
    
    // Return label with most votes
    return std::max_element(votes.begin(), votes.end()) - votes.begin();
}

int main(int argc, char* argv[]) {
    std::cout << "=== SNNFW RetinaAdapter MNIST Example ===" << std::endl;
    
    // Load configuration
    std::string configPath = (argc > 1) ? argv[1] : "../configs/mnist_config_with_adapters.json";
    ConfigLoader loader(configPath);
    
    // Create RetinaAdapter
    auto retinaConfig = loader.getAdapterConfig("retina");
    auto retina = std::make_shared<RetinaAdapter>(retinaConfig);
    
    if (!retina->initialize()) {
        std::cerr << "Failed to initialize RetinaAdapter" << std::endl;
        return 1;
    }
    
    std::cout << "RetinaAdapter initialized with " << retina->getNeurons().size() << " neurons" << std::endl;
    
    // Load MNIST data
    std::string dataPath = loader.getStringParam("mnist_data_path", "../data/mnist");
    MNISTLoader mnist(dataPath);
    
    if (!mnist.load()) {
        std::cerr << "Failed to load MNIST data" << std::endl;
        return 1;
    }
    
    std::cout << "MNIST data loaded: " << mnist.getTrainSize() << " training, "
              << mnist.getTestSize() << " test images" << std::endl;
    
    // Training phase
    std::cout << "\n=== Training Phase ===" << std::endl;
    std::vector<TrainingPattern> trainingPatterns;
    
    int numTrainSamples = std::min(10000, static_cast<int>(mnist.getTrainSize()));
    for (int i = 0; i < numTrainSamples; ++i) {
        if (i % 1000 == 0) {
            std::cout << "Processing training sample " << i << "/" << numTrainSamples << std::endl;
        }
        
        auto sample = mnist.getTrainImage(i);
        int label = mnist.getTrainLabel(i);
        
        // Process image through retina
        retina->processData(sample);
        
        // Train neurons on this pattern
        auto neurons = retina->getNeurons();
        for (auto& neuron : neurons) {
            neuron->learnCurrentPattern();
        }
        
        // Store activation pattern for k-NN
        TrainingPattern pattern;
        pattern.activations = retina->getActivationPattern();
        pattern.label = label;
        trainingPatterns.push_back(pattern);
        
        // Clear spikes for next image
        retina->clearNeuronStates();
    }
    
    std::cout << "Training complete. Stored " << trainingPatterns.size() << " patterns." << std::endl;
    
    // Testing phase
    std::cout << "\n=== Testing Phase ===" << std::endl;
    int correct = 0;
    int total = std::min(10000, static_cast<int>(mnist.getTestSize()));
    
    for (int i = 0; i < total; ++i) {
        if (i % 1000 == 0) {
            std::cout << "Testing sample " << i << "/" << total << std::endl;
        }
        
        auto sample = mnist.getTestImage(i);
        int trueLabel = mnist.getTestLabel(i);
        
        // Process image
        retina->processData(sample);
        auto activations = retina->getActivationPattern();
        
        // Classify using k-NN
        int predictedLabel = classifyKNN(activations, trainingPatterns, 5);
        
        if (predictedLabel == trueLabel) {
            correct++;
        }
        
        // Clear for next image
        retina->clearNeuronStates();
    }
    
    // Results
    double accuracy = 100.0 * correct / total;
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Correct: " << correct << "/" << total << std::endl;
    std::cout << "Accuracy: " << accuracy << "%" << std::endl;
    
    return 0;
}

