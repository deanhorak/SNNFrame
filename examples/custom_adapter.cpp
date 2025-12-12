/**
 * @file custom_adapter.cpp
 * @brief Example: Creating a custom sensor adapter
 * 
 * This example demonstrates:
 * - Implementing a custom SensoryAdapter
 * - Feature extraction from custom data
 * - Spike encoding strategies
 * - Integration with SNNFW
 */

#include "snnfw/adapters/SensoryAdapter.h"
#include "snnfw/Neuron.h"
#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <random>

using namespace snnfw;
using namespace snnfw::adapters;

/**
 * @brief Custom Temperature Sensor Adapter
 * 
 * Simulates a grid of temperature sensors and encodes
 * temperature readings as spike trains.
 */
class TemperatureSensorAdapter : public SensoryAdapter {
public:
    TemperatureSensorAdapter(const Config& config) 
        : SensoryAdapter(config) {
        // Get custom parameters
        gridWidth_ = config.getIntParam("grid_width", 5);
        gridHeight_ = config.getIntParam("grid_height", 5);
        minTemp_ = config.getDoubleParam("min_temp", 0.0);
        maxTemp_ = config.getDoubleParam("max_temp", 100.0);
        
        numSensors_ = gridWidth_ * gridHeight_;
        
        std::cout << "TemperatureSensorAdapter: " << gridWidth_ << "x" << gridHeight_ 
                  << " grid (" << numSensors_ << " sensors)" << std::endl;
        std::cout << "Temperature range: " << minTemp_ << "°C to " << maxTemp_ << "°C" << std::endl;
    }
    
    bool initialize() override {
        // Create one neuron per sensor
        createNeurons();
        
        std::cout << "Created " << neurons_.size() << " neurons" << std::endl;
        return true;
    }
    
    /**
     * @brief Extract temperature features from raw sensor data
     * 
     * Input data format: vector of temperature readings (one per sensor)
     */
    FeatureVector extractFeatures(const std::vector<uint8_t>& data) override {
        FeatureVector features;
        features.timestamp = getCurrentTime();
        
        // Convert raw bytes to temperature values and normalize
        for (size_t i = 0; i < data.size() && i < numSensors_; ++i) {
            // Convert byte to temperature
            double temp = minTemp_ + (data[i] / 255.0) * (maxTemp_ - minTemp_);
            
            // Normalize to [0, 1]
            double normalized = (temp - minTemp_) / (maxTemp_ - minTemp_);
            
            features.features.push_back(normalized);
            features.labels.push_back("sensor_" + std::to_string(i));
        }
        
        return features;
    }
    
    /**
     * @brief Encode temperature features as spike trains
     * 
     * Uses rate coding: higher temperature → earlier spike
     */
    SpikePattern encodeFeatures(const FeatureVector& features) override {
        SpikePattern pattern;
        pattern.timestamp = features.timestamp;
        pattern.duration = config_.temporalWindow;
        
        // Create spike times for each sensor
        pattern.spikeTimes.resize(features.features.size());
        
        for (size_t i = 0; i < features.features.size(); ++i) {
            // Rate coding: higher value → earlier spike
            double spikeTime = featureToSpikeTime(features.features[i], pattern.duration);
            
            if (spikeTime >= 0.0) {
                pattern.spikeTimes[i].push_back(spikeTime);
                
                // Insert spike into corresponding neuron
                if (i < neurons_.size()) {
                    neurons_[i]->insertSpike(spikeTime);
                }
            }
        }
        
        return pattern;
    }
    
    int getNumSensors() const override {
        return numSensors_;
    }
    
    /**
     * @brief Get spatial position of a sensor
     */
    std::pair<int, int> getSensorPosition(int sensorId) const {
        int x = sensorId % gridWidth_;
        int y = sensorId / gridWidth_;
        return {x, y};
    }
    
private:
    int gridWidth_;
    int gridHeight_;
    int numSensors_;
    double minTemp_;
    double maxTemp_;
};

/**
 * @brief Simulate temperature sensor readings
 */
std::vector<uint8_t> simulateTemperatureData(int numSensors, double time) {
    std::vector<uint8_t> data(numSensors);
    
    // Create a moving "hot spot" pattern
    double hotSpotX = 2.5 + 1.5 * std::sin(time * 0.1);
    double hotSpotY = 2.5 + 1.5 * std::cos(time * 0.1);
    
    for (int i = 0; i < numSensors; ++i) {
        int x = i % 5;
        int y = i / 5;
        
        // Distance from hot spot
        double dx = x - hotSpotX;
        double dy = y - hotSpotY;
        double distance = std::sqrt(dx * dx + dy * dy);
        
        // Temperature decreases with distance from hot spot
        double temp = 255.0 * std::exp(-distance / 2.0);
        
        data[i] = static_cast<uint8_t>(std::min(255.0, std::max(0.0, temp)));
    }
    
    return data;
}

/**
 * @brief Visualize temperature grid
 */
void visualizeTemperature(const std::vector<uint8_t>& data, int width, int height) {
    const char* gradient = " .:-=+*#%@";
    int gradientLen = 10;
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int idx = y * width + x;
            int level = (data[idx] * gradientLen) / 256;
            std::cout << gradient[level] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    std::cout << "=== SNNFW Custom Adapter Example ===" << std::endl;
    std::cout << "Temperature Sensor Grid Simulation\n" << std::endl;
    
    // Create adapter configuration
    BaseAdapter::Config config;
    config.name = "temperature";
    config.type = "temperature";
    config.temporalWindow = 100.0;
    config.intParams["grid_width"] = 5;
    config.intParams["grid_height"] = 5;
    config.doubleParams["min_temp"] = 0.0;
    config.doubleParams["max_temp"] = 100.0;
    
    // Create custom adapter
    auto tempSensor = std::make_shared<TemperatureSensorAdapter>(config);
    
    if (!tempSensor->initialize()) {
        std::cerr << "Failed to initialize adapter" << std::endl;
        return 1;
    }
    
    // Simulation loop
    const int numSteps = 10;
    double currentTime = 0.0;
    const double timeStep = 100.0; // ms
    
    for (int step = 0; step < numSteps; ++step) {
        std::cout << "\n=== Time: " << currentTime << " ms ===" << std::endl;
        
        // Simulate sensor readings
        auto sensorData = simulateTemperatureData(25, currentTime / 100.0);
        
        // Visualize temperature grid
        std::cout << "Temperature Grid:" << std::endl;
        visualizeTemperature(sensorData, 5, 5);
        
        // Process through adapter
        tempSensor->processData(sensorData);
        
        // Get activation pattern
        auto activations = tempSensor->getActivationPattern();
        
        // Display activation statistics
        double totalActivation = 0.0;
        double maxActivation = 0.0;
        int maxNeuron = 0;
        
        for (size_t i = 0; i < activations.size(); ++i) {
            totalActivation += activations[i];
            if (activations[i] > maxActivation) {
                maxActivation = activations[i];
                maxNeuron = i;
            }
        }
        
        double avgActivation = totalActivation / activations.size();
        auto [maxX, maxY] = tempSensor->getSensorPosition(maxNeuron);
        
        std::cout << "\nNeural Activity:" << std::endl;
        std::cout << "  Average activation: " << avgActivation << std::endl;
        std::cout << "  Max activation: " << maxActivation 
                  << " at sensor (" << maxX << ", " << maxY << ")" << std::endl;
        
        // Clear for next iteration
        tempSensor->clearNeuronStates();
        
        currentTime += timeStep;
    }
    
    std::cout << "\n=== Simulation Complete ===" << std::endl;
    std::cout << "\nThis example demonstrated:" << std::endl;
    std::cout << "  ✓ Creating a custom SensoryAdapter" << std::endl;
    std::cout << "  ✓ Feature extraction from sensor data" << std::endl;
    std::cout << "  ✓ Spike encoding with rate coding" << std::endl;
    std::cout << "  ✓ Neural activation patterns" << std::endl;
    
    return 0;
}

