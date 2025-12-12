#include "snnfw/Logger.h"
#include "snnfw/Neuron.h"
#include "snnfw/Axon.h"
#include "snnfw/Dendrite.h"
#include "snnfw/Synapse.h"
#include "snnfw/ActionPotential.h"
#include "snnfw/SpikeProcessor.h"
#include <spdlog/spdlog.h>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>

using namespace snnfw;

/**
 * @brief Example demonstrating the complete neural network infrastructure
 *
 * This example creates a simple network with:
 * - 3 neurons (source neurons)
 * - 1 target neuron
 * - Axons for each source neuron
 * - Dendrites for the target neuron
 * - Synapses connecting axons to dendrites
 * - SpikeProcessor managing spike delivery
 */

int main() {
    // Initialize logger
    Logger::getInstance().initialize("neural_network.log", spdlog::level::info);
    
    SNNFW_INFO("=== Neural Network Infrastructure Example ===\n");
    
    // ========================================================================
    // Step 1: Create Neurons
    // ========================================================================
    SNNFW_INFO("Step 1: Creating neurons...");
    
    auto neuron1 = std::make_shared<Neuron>(50.0, 0.95, 20, 1);
    auto neuron2 = std::make_shared<Neuron>(50.0, 0.95, 20, 2);
    auto neuron3 = std::make_shared<Neuron>(50.0, 0.95, 20, 3);
    auto targetNeuron = std::make_shared<Neuron>(50.0, 0.95, 20, 100);
    
    SNNFW_INFO("Created 3 source neurons (IDs: 1, 2, 3) and 1 target neuron (ID: 100)\n");
    
    // ========================================================================
    // Step 2: Create Axons (one per source neuron)
    // ========================================================================
    SNNFW_INFO("Step 2: Creating axons...");
    
    auto axon1 = std::make_shared<Axon>(1, 1001);  // Axon for neuron 1
    auto axon2 = std::make_shared<Axon>(2, 1002);  // Axon for neuron 2
    auto axon3 = std::make_shared<Axon>(3, 1003);  // Axon for neuron 3
    
    SNNFW_INFO("Created 3 axons (IDs: 1001, 1002, 1003)\n");
    
    // ========================================================================
    // Step 3: Create Dendrites (for target neuron)
    // ========================================================================
    SNNFW_INFO("Step 3: Creating dendrites...");
    
    auto dendrite1 = std::make_shared<Dendrite>(100, 2001);  // Dendrite 1 for target neuron
    auto dendrite2 = std::make_shared<Dendrite>(100, 2002);  // Dendrite 2 for target neuron
    auto dendrite3 = std::make_shared<Dendrite>(100, 2003);  // Dendrite 3 for target neuron
    
    SNNFW_INFO("Created 3 dendrites (IDs: 2001, 2002, 2003) for target neuron\n");
    
    // ========================================================================
    // Step 4: Create Synapses (connecting axons to dendrites)
    // ========================================================================
    SNNFW_INFO("Step 4: Creating synapses...");
    
    // Synapse 1: Axon1 -> Dendrite1 (weight: 0.8, delay: 1.0ms)
    auto synapse1 = std::make_shared<Synapse>(1001, 2001, 0.8, 1.0, 3001);
    
    // Synapse 2: Axon2 -> Dendrite2 (weight: 0.6, delay: 1.5ms)
    auto synapse2 = std::make_shared<Synapse>(1002, 2002, 0.6, 1.5, 3002);
    
    // Synapse 3: Axon3 -> Dendrite3 (weight: 0.9, delay: 2.0ms)
    auto synapse3 = std::make_shared<Synapse>(1003, 2003, 0.9, 2.0, 3003);
    
    SNNFW_INFO("Created 3 synapses:");
    SNNFW_INFO("  Synapse 3001: Axon 1001 -> Dendrite 2001 (weight: 0.8, delay: 1.0ms)");
    SNNFW_INFO("  Synapse 3002: Axon 1002 -> Dendrite 2002 (weight: 0.6, delay: 1.5ms)");
    SNNFW_INFO("  Synapse 3003: Axon 1003 -> Dendrite 2003 (weight: 0.9, delay: 2.0ms)\n");
    
    // Register synapses with dendrites
    dendrite1->addSynapse(3001);
    dendrite2->addSynapse(3002);
    dendrite3->addSynapse(3003);
    
    // ========================================================================
    // Step 5: Create and configure SpikeProcessor
    // ========================================================================
    SNNFW_INFO("Step 5: Creating SpikeProcessor...");
    
    SpikeProcessor processor(10000, 4);  // 10 seconds buffer, 4 delivery threads
    
    // Register dendrites with the processor
    processor.registerDendrite(dendrite1);
    processor.registerDendrite(dendrite2);
    processor.registerDendrite(dendrite3);
    
    SNNFW_INFO("SpikeProcessor created with 10000 time slices and 4 delivery threads");
    SNNFW_INFO("Registered 3 dendrites with the processor\n");
    
    // ========================================================================
    // Step 6: Start the SpikeProcessor
    // ========================================================================
    SNNFW_INFO("Step 6: Starting SpikeProcessor...");
    processor.start();
    SNNFW_INFO("SpikeProcessor started\n");
    
    // ========================================================================
    // Step 7: Schedule Action Potentials
    // ========================================================================
    SNNFW_INFO("Step 7: Scheduling action potentials...");
    
    // Simulate neuron 1 firing at time 10ms
    auto ap1 = std::make_shared<ActionPotential>(
        3001,                           // Synapse ID
        2001,                           // Dendrite ID
        10.0 + synapse1->getDelay(),   // Scheduled time (firing time + delay)
        synapse1->getWeight()          // Amplitude (modulated by weight)
    );
    processor.scheduleSpike(ap1);
    SNNFW_INFO("Scheduled spike from neuron 1 (arrives at {:.1f}ms)", 10.0 + synapse1->getDelay());
    
    // Simulate neuron 2 firing at time 15ms
    auto ap2 = std::make_shared<ActionPotential>(
        3002,
        2002,
        15.0 + synapse2->getDelay(),
        synapse2->getWeight()
    );
    processor.scheduleSpike(ap2);
    SNNFW_INFO("Scheduled spike from neuron 2 (arrives at {:.1f}ms)", 15.0 + synapse2->getDelay());
    
    // Simulate neuron 3 firing at time 20ms
    auto ap3 = std::make_shared<ActionPotential>(
        3003,
        2003,
        20.0 + synapse3->getDelay(),
        synapse3->getWeight()
    );
    processor.scheduleSpike(ap3);
    SNNFW_INFO("Scheduled spike from neuron 3 (arrives at {:.1f}ms)", 20.0 + synapse3->getDelay());
    
    SNNFW_INFO("Total pending spikes: {}\n", processor.getPendingSpikeCount());
    
    // ========================================================================
    // Step 8: Schedule multiple spikes to demonstrate parallel delivery
    // ========================================================================
    SNNFW_INFO("Step 8: Scheduling burst of spikes...");
    
    for (int i = 0; i < 10; ++i) {
        double firingTime = 50.0 + i * 2.0;  // Spikes every 2ms starting at 50ms
        
        auto ap = std::make_shared<ActionPotential>(
            3001,
            2001,
            firingTime + synapse1->getDelay(),
            synapse1->getWeight()
        );
        processor.scheduleSpike(ap);
    }
    
    SNNFW_INFO("Scheduled burst of 10 spikes");
    SNNFW_INFO("Total pending spikes: {}\n", processor.getPendingSpikeCount());
    
    // ========================================================================
    // Step 9: Let the simulation run
    // ========================================================================
    SNNFW_INFO("Step 9: Running simulation...");
    SNNFW_INFO("Current simulation time: {:.1f}ms", processor.getCurrentTime());
    
    // Run for 200ms
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    SNNFW_INFO("Simulation time after 200ms: {:.1f}ms", processor.getCurrentTime());
    SNNFW_INFO("Remaining pending spikes: {}\n", processor.getPendingSpikeCount());
    
    // ========================================================================
    // Step 10: Demonstrate synaptic plasticity
    // ========================================================================
    SNNFW_INFO("Step 10: Demonstrating synaptic plasticity...");
    
    SNNFW_INFO("Original synapse 1 weight: {:.2f}", synapse1->getWeight());
    
    // Strengthen the synapse (simulating learning)
    synapse1->modifyWeight(0.1);
    SNNFW_INFO("After strengthening: {:.2f}", synapse1->getWeight());
    
    // Weaken the synapse
    synapse1->modifyWeight(-0.05);
    SNNFW_INFO("After weakening: {:.2f}\n", synapse1->getWeight());
    
    // ========================================================================
    // Step 11: Stop the processor
    // ========================================================================
    SNNFW_INFO("Step 11: Stopping SpikeProcessor...");
    processor.stop();
    SNNFW_INFO("SpikeProcessor stopped");
    SNNFW_INFO("Final simulation time: {:.1f}ms\n", processor.getCurrentTime());
    
    // ========================================================================
    // Summary
    // ========================================================================
    SNNFW_INFO("=== Summary ===");
    SNNFW_INFO("Network structure:");
    SNNFW_INFO("  - 3 source neurons with axons");
    SNNFW_INFO("  - 1 target neuron with 3 dendrites");
    SNNFW_INFO("  - 3 synapses connecting the network");
    SNNFW_INFO("  - SpikeProcessor managing spike delivery");
    SNNFW_INFO("");
    SNNFW_INFO("Biological accuracy:");
    SNNFW_INFO("  - Axons transmit signals from neurons");
    SNNFW_INFO("  - Dendrites receive signals at neurons");
    SNNFW_INFO("  - Synapses connect axons to dendrites");
    SNNFW_INFO("  - Action potentials propagate with realistic delays");
    SNNFW_INFO("  - Synaptic weights modulate signal strength");
    SNNFW_INFO("  - Parallel spike delivery simulates biological concurrency");
    SNNFW_INFO("");
    SNNFW_INFO("=== Example Complete ===");
    
    return 0;
}

