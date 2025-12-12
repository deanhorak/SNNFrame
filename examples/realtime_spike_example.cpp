/**
 * @file realtime_spike_example.cpp
 * @brief Demonstrates real-time spike processing with configurable threading
 *
 * This example shows:
 * - Real-time synchronization (1ms per timeslice)
 * - Configurable number of delivery threads (default: 20)
 * - Timing statistics and drift monitoring
 * - Spike scheduling and delivery
 */

#include "snnfw/SpikeProcessor.h"
#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Neuron.h"
#include "snnfw/Axon.h"
#include "snnfw/Dendrite.h"
#include "snnfw/Synapse.h"
#include "snnfw/ActionPotential.h"
#include "snnfw/Logger.h"
#include <iostream>
#include <thread>
#include <chrono>
#include <iomanip>

using namespace snnfw;

int main() {
    // Initialize logger
    Logger::getInstance().setLevel(spdlog::level::info);
    
    std::cout << "=== Real-Time Spike Processing Demo ===" << std::endl;
    std::cout << std::endl;
    
    // Create factory
    NeuralObjectFactory factory;
    
    // Configuration
    const size_t numNeurons = 100;
    const size_t numDeliveryThreads = 20;  // Configurable, default is 20
    const size_t timeSliceCount = 10000;   // 10 seconds of buffering
    
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Neurons: " << numNeurons << std::endl;
    std::cout << "  Delivery threads: " << numDeliveryThreads << std::endl;
    std::cout << "  Time slice buffer: " << timeSliceCount << " ms" << std::endl;
    std::cout << "  Real-time sync: ENABLED (1ms per timeslice)" << std::endl;
    std::cout << std::endl;
    
    // Create SpikeProcessor with configurable threading
    SpikeProcessor processor(timeSliceCount, numDeliveryThreads);
    
    // Create a network of neurons
    std::vector<std::shared_ptr<Neuron>> neurons;
    std::vector<std::shared_ptr<Dendrite>> dendrites;
    std::vector<std::shared_ptr<Axon>> axons;
    
    std::cout << "Creating neural network..." << std::endl;
    
    for (size_t i = 0; i < numNeurons; ++i) {
        // Create neuron
        auto neuron = factory.createNeuron(50.0, 0.95, 20);
        neurons.push_back(neuron);
        
        // Create axon
        auto axon = factory.createAxon(neuron->getId());
        neuron->setAxonId(axon->getId());
        axons.push_back(axon);
        
        // Create dendrite
        auto dendrite = factory.createDendrite(neuron->getId());
        neuron->addDendrite(dendrite->getId());
        dendrites.push_back(dendrite);
        
        // Register dendrite with processor
        processor.registerDendrite(dendrite);
    }
    
    std::cout << "Created " << neurons.size() << " neurons with axons and dendrites" << std::endl;
    std::cout << std::endl;
    
    // Start the processor
    std::cout << "Starting SpikeProcessor..." << std::endl;
    processor.start();
    
    // Wait a moment for processor to initialize
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    std::cout << "Processor started. Scheduling spikes..." << std::endl;
    std::cout << std::endl;
    
    // Schedule spikes at various times
    auto wallClockStart = std::chrono::steady_clock::now();
    
    // Schedule 1000 spikes spread over 5 seconds
    const size_t numSpikes = 1000;
    const double timeSpan = 5000.0; // 5 seconds
    
    for (size_t i = 0; i < numSpikes; ++i) {
        // Random time within the span
        double scheduledTime = (static_cast<double>(i) / numSpikes) * timeSpan;
        
        // Random target dendrite
        size_t targetIdx = i % numNeurons;
        
        // Create action potential
        auto spike = std::make_shared<ActionPotential>(
            axons[i % numNeurons]->getId(),
            dendrites[targetIdx]->getId(),
            scheduledTime,
            1.0  // weight
        );
        
        processor.scheduleSpike(spike);
    }
    
    std::cout << "Scheduled " << numSpikes << " spikes over " << timeSpan << " ms" << std::endl;
    std::cout << std::endl;
    
    // Monitor progress in real-time
    std::cout << "Monitoring real-time execution:" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::setw(10) << "Sim Time" 
              << std::setw(12) << "Wall Time"
              << std::setw(15) << "Pending"
              << std::setw(15) << "Avg Loop"
              << std::setw(15) << "Max Loop"
              << std::setw(13) << "Drift" << std::endl;
    std::cout << std::setw(10) << "(ms)" 
              << std::setw(12) << "(ms)"
              << std::setw(15) << "Spikes"
              << std::setw(15) << "(μs)"
              << std::setw(15) << "(μs)"
              << std::setw(13) << "(ms)" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    // Monitor for 6 seconds (to see all spikes delivered)
    for (int i = 0; i < 60; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        auto now = std::chrono::steady_clock::now();
        auto wallElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            now - wallClockStart);
        
        double simTime = processor.getCurrentTime();
        size_t pending = processor.getPendingSpikeCount();
        
        double avgLoop, maxLoop, drift;
        processor.getTimingStats(avgLoop, maxLoop, drift);
        
        // Print every 500ms
        if (i % 5 == 0) {
            std::cout << std::fixed << std::setprecision(1)
                      << std::setw(10) << simTime
                      << std::setw(12) << wallElapsed.count()
                      << std::setw(15) << pending
                      << std::setw(15) << avgLoop
                      << std::setw(15) << maxLoop
                      << std::setw(13) << drift
                      << std::endl;
        }
    }
    
    std::cout << std::string(80, '-') << std::endl;
    std::cout << std::endl;
    
    // Get final statistics
    double avgLoop, maxLoop, drift;
    processor.getTimingStats(avgLoop, maxLoop, drift);
    
    std::cout << "Final Statistics:" << std::endl;
    std::cout << "  Simulation time: " << processor.getCurrentTime() << " ms" << std::endl;
    std::cout << "  Average loop time: " << avgLoop << " μs" << std::endl;
    std::cout << "  Maximum loop time: " << maxLoop << " μs" << std::endl;
    std::cout << "  Final drift: " << drift << " ms" << std::endl;
    std::cout << "  Pending spikes: " << processor.getPendingSpikeCount() << std::endl;
    std::cout << std::endl;
    
    // Demonstrate non-real-time mode
    std::cout << "Switching to non-real-time mode (fast as possible)..." << std::endl;
    processor.stop();
    
    processor.setRealTimeSync(false);
    processor.start();
    
    // Schedule more spikes
    for (size_t i = 0; i < 500; ++i) {
        double scheduledTime = processor.getCurrentTime() + 100.0 + (i * 2.0);
        size_t targetIdx = i % numNeurons;
        
        auto spike = std::make_shared<ActionPotential>(
            axons[i % numNeurons]->getId(),
            dendrites[targetIdx]->getId(),
            scheduledTime,
            1.0
        );
        
        processor.scheduleSpike(spike);
    }
    
    std::cout << "Scheduled 500 more spikes. Running for 1 second of wall-clock time..." << std::endl;
    
    double startSimTime = processor.getCurrentTime();
    auto fastStart = std::chrono::steady_clock::now();
    
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    auto fastEnd = std::chrono::steady_clock::now();
    double endSimTime = processor.getCurrentTime();
    
    auto fastElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        fastEnd - fastStart);
    
    double simElapsed = endSimTime - startSimTime;
    double speedup = simElapsed / fastElapsed.count();
    
    std::cout << std::endl;
    std::cout << "Non-real-time performance:" << std::endl;
    std::cout << "  Wall-clock time: " << fastElapsed.count() << " ms" << std::endl;
    std::cout << "  Simulation time: " << simElapsed << " ms" << std::endl;
    std::cout << "  Speedup: " << speedup << "x real-time" << std::endl;
    std::cout << std::endl;
    
    // Stop processor
    std::cout << "Stopping SpikeProcessor..." << std::endl;
    processor.stop();
    
    std::cout << std::endl;
    std::cout << "=== Demo Complete ===" << std::endl;
    
    return 0;
}

