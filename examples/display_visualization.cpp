/**
 * @file display_visualization.cpp
 * @brief Example: Visualizing neural activity with DisplayAdapter
 * 
 * This example demonstrates:
 * - Creating a simple spiking neural network
 * - Using DisplayAdapter to visualize activity
 * - Different display modes (raster, heatmap, vector)
 * - Real-time activity monitoring
 */

#include "snnfw/adapters/DisplayAdapter.h"
#include "snnfw/Neuron.h"
#include <iostream>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <random>

using namespace snnfw;
using namespace snnfw::adapters;

int main() {
    std::cout << "=== SNNFW DisplayAdapter Example ===" << std::endl;
    
    // Create DisplayAdapter configuration
    BaseAdapter::Config displayConfig;
    displayConfig.name = "display";
    displayConfig.type = "display";
    displayConfig.temporalWindow = 100.0;
    displayConfig.intParams["display_width"] = 80;
    displayConfig.intParams["display_height"] = 24;
    displayConfig.doubleParams["update_interval"] = 50.0;
    displayConfig.stringParams["mode"] = "heatmap";
    
    auto display = std::make_shared<DisplayAdapter>(displayConfig);
    
    if (!display->initialize()) {
        std::cerr << "Failed to initialize DisplayAdapter" << std::endl;
        return 1;
    }
    
    std::cout << "DisplayAdapter initialized" << std::endl;
    
    // Create a population of neurons
    const int numNeurons = 100;
    std::vector<std::shared_ptr<Neuron>> neurons;
    
    for (int i = 0; i < numNeurons; ++i) {
        auto neuron = std::make_shared<Neuron>(
            100.0,  // windowSizeMs
            0.7,    // similarityThreshold
            20,     // maxReferencePatterns
            i       // neuronId
        );
        neurons.push_back(neuron);
    }
    
    // Connect neurons to display
    for (auto& neuron : neurons) {
        display->addNeuron(neuron);
    }
    
    std::cout << "Created " << numNeurons << " neurons" << std::endl;
    std::cout << "\nSimulating neural activity...\n" << std::endl;
    
    // Random number generator for spike generation
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> spikeProb(0.0, 1.0);
    std::uniform_real_distribution<> spikeTime(0.0, 10.0);
    
    // Simulation loop
    double currentTime = 0.0;
    const double timeStep = 10.0; // ms
    const int numSteps = 20;
    
    for (int step = 0; step < numSteps; ++step) {
        currentTime += timeStep;
        
        // Generate random spikes with spatial pattern
        // Create a "wave" pattern across neurons
        double wavePosition = (step % 10) * 10.0;
        
        for (int i = 0; i < numNeurons; ++i) {
            // Higher probability near wave position
            double distance = std::abs(i - wavePosition);
            double probability = std::exp(-distance / 20.0) * 0.5;
            
            if (spikeProb(gen) < probability) {
                double t = currentTime + spikeTime(gen);
                neurons[i]->insertSpike(t);
            }
        }
        
        // Update display
        display->update(currentTime);
        
        // Get and print visualization
        std::string visualization = display->getDisplayBuffer();
        
        // Clear screen (ANSI escape code)
        std::cout << "\033[2J\033[H";
        
        std::cout << "=== Neural Activity Visualization ===" << std::endl;
        std::cout << "Time: " << currentTime << " ms" << std::endl;
        std::cout << "Step: " << (step + 1) << "/" << numSteps << std::endl;
        std::cout << "\n" << visualization << std::endl;
        
        // Sleep to make visualization visible
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    
    std::cout << "\n=== Demonstration of Different Display Modes ===" << std::endl;
    
    // Generate some activity
    for (int i = 0; i < numNeurons; ++i) {
        if (i % 3 == 0) {
            neurons[i]->insertSpike(currentTime + 5.0);
        }
        if (i % 5 == 0) {
            neurons[i]->insertSpike(currentTime + 15.0);
        }
    }
    
    currentTime += 20.0;
    
    // Show different display modes
    std::vector<std::string> modes = {"raster", "heatmap", "vector", "ascii"};
    
    for (const auto& mode : modes) {
        display->setDisplayMode(
            mode == "raster" ? DisplayAdapter::DisplayMode::RASTER :
            mode == "heatmap" ? DisplayAdapter::DisplayMode::HEATMAP :
            mode == "vector" ? DisplayAdapter::DisplayMode::VECTOR :
            DisplayAdapter::DisplayMode::ASCII
        );
        
        display->update(currentTime);
        
        std::cout << "\n--- Mode: " << mode << " ---" << std::endl;
        std::cout << display->getDisplayBuffer() << std::endl;
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }
    
    std::cout << "\n=== Simulation Complete ===" << std::endl;
    
    return 0;
}

