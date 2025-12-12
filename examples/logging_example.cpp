#include "snnfw/Logger.h"
#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Neuron.h"
#include "snnfw/Cluster.h"
#include <spdlog/spdlog.h>
#include <memory>

int main() {
    // Initialize logger with DEBUG level to see all messages
    snnfw::Logger::getInstance().initialize("snnfw_debug.log", spdlog::level::debug);

    SNNFW_INFO("=== Logging Example Started ===");
    SNNFW_INFO("Demonstrating different log levels:");

    // Demonstrate all log levels
    SNNFW_TRACE("This is a TRACE message - very detailed");
    SNNFW_DEBUG("This is a DEBUG message - detailed information");
    SNNFW_INFO("This is an INFO message - general information");
    SNNFW_WARN("This is a WARN message - warning");
    SNNFW_ERROR("This is an ERROR message - error occurred");
    SNNFW_CRITICAL("This is a CRITICAL message - critical error");

    SNNFW_INFO("");
    SNNFW_INFO("=== Creating Neural Objects ===");

    // Create factory
    snnfw::NeuralObjectFactory factory;

    // Create neurons and demonstrate logging
    auto neuron1 = factory.createNeuron(50.0, 0.95, 20);
    SNNFW_INFO("Created neuron with ID: {}", neuron1->getId());

    // Insert spikes
    SNNFW_DEBUG("Inserting spikes into neuron {}", neuron1->getId());
    neuron1->insertSpike(10.0);
    neuron1->insertSpike(20.0);
    neuron1->insertSpike(30.0);

    // Learn pattern
    SNNFW_DEBUG("Learning pattern for neuron {}", neuron1->getId());
    neuron1->learnCurrentPattern();

    // Create a cluster
    auto cluster = factory.createCluster();
    SNNFW_INFO("Created cluster with ID: {}", cluster->getId());

    // Add neurons to cluster by ID
    cluster->addNeuron(neuron1->getId());
    auto neuron2 = factory.createNeuron(50.0, 0.95, 20);
    cluster->addNeuron(neuron2->getId());
    SNNFW_INFO("Added {} neuron IDs to cluster", cluster->size());

    // Print cluster info
    cluster->printInfo();
    
    SNNFW_INFO("");
    SNNFW_INFO("=== Changing Log Level to WARN ===");
    snnfw::Logger::getInstance().setLevel(spdlog::level::warn);
    
    SNNFW_DEBUG("This DEBUG message will NOT be shown");
    SNNFW_INFO("This INFO message will NOT be shown");
    SNNFW_WARN("This WARN message WILL be shown");
    SNNFW_ERROR("This ERROR message WILL be shown");
    
    SNNFW_WARN("");
    SNNFW_WARN("=== Changing Log Level back to INFO ===");
    snnfw::Logger::getInstance().setLevel(spdlog::level::info);
    
    SNNFW_INFO("Log level restored to INFO");
    SNNFW_DEBUG("This DEBUG message will NOT be shown");
    SNNFW_INFO("This INFO message WILL be shown");
    
    SNNFW_INFO("");
    SNNFW_INFO("=== Logging Example Finished ===");
    SNNFW_INFO("Check snnfw_debug.log for the complete log");
    
    return 0;
}

