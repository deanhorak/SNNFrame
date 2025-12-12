/**
 * @file datastore_example.cpp
 * @brief Comprehensive example demonstrating the Datastore persistence system
 * 
 * This example shows:
 * 1. Creating a neural network with neurons, axons, dendrites, and synapses
 * 2. Persisting the network to disk using the Datastore
 * 3. Loading the network from disk in a new session
 * 4. Monitoring cache statistics
 * 5. Batch operations and flushing
 */

#include "snnfw/Datastore.h"
#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Neuron.h"
#include "snnfw/Axon.h"
#include "snnfw/Dendrite.h"
#include "snnfw/Synapse.h"
#include "snnfw/Cluster.h"
#include "snnfw/Logger.h"
#include <iostream>
#include <memory>
#include <vector>
#include <filesystem>

using namespace snnfw;

// Helper function to register all factories with the datastore
void registerFactories(Datastore& datastore) {
    // Register Neuron factory
    datastore.registerFactory("Neuron", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto neuron = std::make_shared<Neuron>(0, 0, 0);
        if (neuron->fromJson(json)) {
            return neuron;
        }
        return nullptr;
    });
    
    // Register Axon factory
    datastore.registerFactory("Axon", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto axon = std::make_shared<Axon>(0);
        if (axon->fromJson(json)) {
            return axon;
        }
        return nullptr;
    });
    
    // Register Dendrite factory
    datastore.registerFactory("Dendrite", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto dendrite = std::make_shared<Dendrite>(0);
        if (dendrite->fromJson(json)) {
            return dendrite;
        }
        return nullptr;
    });
    
    // Register Synapse factory
    datastore.registerFactory("Synapse", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto synapse = std::make_shared<Synapse>(0, 0, 0.0, 0.0);
        if (synapse->fromJson(json)) {
            return synapse;
        }
        return nullptr;
    });
    
    // Register Cluster factory
    datastore.registerFactory("Cluster", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto cluster = std::make_shared<Cluster>();
        if (cluster->fromJson(json)) {
            return cluster;
        }
        return nullptr;
    });
}

// Example 1: Creating and persisting a simple neural network
void example1_CreateAndPersist() {
    std::cout << "\n=== Example 1: Creating and Persisting a Neural Network ===\n" << std::endl;
    
    const std::string dbPath = "./example_network_db";
    
    // Clean up any existing database
    std::filesystem::remove_all(dbPath);
    
    // Create datastore with 1000 object cache
    Datastore datastore(dbPath, 1000);
    registerFactories(datastore);
    
    // Create factory for neural objects
    NeuralObjectFactory factory;
    
    // Create a simple network: 3 neurons connected in a chain
    std::cout << "Creating neural network..." << std::endl;
    
    // Create neurons
    auto neuron1 = factory.createNeuron(50.0, 0.95, 20);
    auto neuron2 = factory.createNeuron(50.0, 0.95, 20);
    auto neuron3 = factory.createNeuron(50.0, 0.95, 20);
    
    std::cout << "  Created neurons: " << neuron1->getId() << ", " 
              << neuron2->getId() << ", " << neuron3->getId() << std::endl;
    
    // Create axons for each neuron
    auto axon1 = factory.createAxon(neuron1->getId());
    auto axon2 = factory.createAxon(neuron2->getId());
    neuron1->setAxonId(axon1->getId());
    neuron2->setAxonId(axon2->getId());
    
    // Create dendrites for receiving neurons
    auto dendrite2 = factory.createDendrite(neuron2->getId());
    auto dendrite3 = factory.createDendrite(neuron3->getId());
    neuron2->addDendrite(dendrite2->getId());
    neuron3->addDendrite(dendrite3->getId());
    
    // Create synapses to connect them
    auto synapse1 = factory.createSynapse(axon1->getId(), dendrite2->getId(), 0.8, 1.5);
    auto synapse2 = factory.createSynapse(axon2->getId(), dendrite3->getId(), 0.7, 2.0);
    
    // Register synapses with axons and dendrites
    axon1->addSynapse(synapse1->getId());
    dendrite2->addSynapse(synapse1->getId());
    axon2->addSynapse(synapse2->getId());
    dendrite3->addSynapse(synapse2->getId());
    
    std::cout << "  Created connections via synapses: " << synapse1->getId() 
              << ", " << synapse2->getId() << std::endl;
    
    // Create a cluster containing all neurons
    auto cluster = factory.createCluster();
    cluster->addNeuron(neuron1->getId());
    cluster->addNeuron(neuron2->getId());
    cluster->addNeuron(neuron3->getId());
    
    std::cout << "  Created cluster: " << cluster->getId() << " with " 
              << cluster->size() << " neurons" << std::endl;
    
    // Store all objects in the datastore
    std::cout << "\nStoring objects in datastore..." << std::endl;
    datastore.put(neuron1);
    datastore.put(neuron2);
    datastore.put(neuron3);
    datastore.put(axon1);
    datastore.put(axon2);
    datastore.put(dendrite2);
    datastore.put(dendrite3);
    datastore.put(synapse1);
    datastore.put(synapse2);
    datastore.put(cluster);
    
    // Mark all as dirty to ensure they're written to disk
    datastore.markDirty(neuron1->getId());
    datastore.markDirty(neuron2->getId());
    datastore.markDirty(neuron3->getId());
    datastore.markDirty(axon1->getId());
    datastore.markDirty(axon2->getId());
    datastore.markDirty(dendrite2->getId());
    datastore.markDirty(dendrite3->getId());
    datastore.markDirty(synapse1->getId());
    datastore.markDirty(synapse2->getId());
    datastore.markDirty(cluster->getId());
    
    // Flush all to disk
    size_t flushed = datastore.flushAll();
    std::cout << "  Flushed " << flushed << " objects to disk" << std::endl;
    
    // Display cache statistics
    uint64_t hits, misses;
    datastore.getCacheStats(hits, misses);
    std::cout << "\nCache Statistics:" << std::endl;
    std::cout << "  Hits: " << hits << std::endl;
    std::cout << "  Misses: " << misses << std::endl;
    std::cout << "  Hit Rate: " << (hits + misses > 0 ? 100.0 * hits / (hits + misses) : 0.0) << "%" << std::endl;
    std::cout << "  Cache Size: " << datastore.getCacheSize() << std::endl;
    
    std::cout << "\n✓ Network created and persisted successfully!" << std::endl;
}

// Example 2: Loading a persisted network
void example2_LoadNetwork() {
    std::cout << "\n=== Example 2: Loading a Persisted Neural Network ===\n" << std::endl;
    
    const std::string dbPath = "./example_network_db";
    
    // Create new datastore instance (simulating a new session)
    Datastore datastore(dbPath, 1000);
    registerFactories(datastore);
    
    std::cout << "Loading network from disk..." << std::endl;
    
    // We know the first neuron ID from the factory (starts at 100000000000000)
    uint64_t neuron1Id = 100000000000000ULL;
    uint64_t neuron2Id = 100000000000001ULL;
    uint64_t neuron3Id = 100000000000002ULL;
    
    // Load neurons
    auto neuron1 = datastore.getNeuron(neuron1Id);
    auto neuron2 = datastore.getNeuron(neuron2Id);
    auto neuron3 = datastore.getNeuron(neuron3Id);
    
    if (neuron1 && neuron2 && neuron3) {
        std::cout << "  ✓ Loaded 3 neurons successfully" << std::endl;
        std::cout << "    Neuron 1 ID: " << neuron1->getId() << std::endl;
        std::cout << "    Neuron 2 ID: " << neuron2->getId() << std::endl;
        std::cout << "    Neuron 3 ID: " << neuron3->getId() << std::endl;
        
        // Load axons
        auto axon1 = datastore.getAxon(neuron1->getAxonId());
        auto axon2 = datastore.getAxon(neuron2->getAxonId());
        
        if (axon1 && axon2) {
            std::cout << "  ✓ Loaded axons successfully" << std::endl;
            std::cout << "    Axon 1 has " << axon1->getSynapseCount() << " synapse(s)" << std::endl;
            std::cout << "    Axon 2 has " << axon2->getSynapseCount() << " synapse(s)" << std::endl;
        }
        
        // Load cluster (ID starts at 500000000000000)
        uint64_t clusterId = 500000000000000ULL;
        auto cluster = datastore.getCluster(clusterId);
        
        if (cluster) {
            std::cout << "  ✓ Loaded cluster successfully" << std::endl;
            std::cout << "    Cluster contains " << cluster->size() << " neurons" << std::endl;
        }
    } else {
        std::cout << "  ✗ Failed to load neurons" << std::endl;
    }
    
    // Display cache statistics
    uint64_t hits, misses;
    datastore.getCacheStats(hits, misses);
    std::cout << "\nCache Statistics:" << std::endl;
    std::cout << "  Hits: " << hits << std::endl;
    std::cout << "  Misses: " << misses << std::endl;
    std::cout << "  Hit Rate: " << (hits + misses > 0 ? 100.0 * hits / (hits + misses) : 0.0) << "%" << std::endl;
    std::cout << "  Cache Size: " << datastore.getCacheSize() << std::endl;
    
    std::cout << "\n✓ Network loaded successfully!" << std::endl;
}

// Example 3: Modifying and re-persisting
void example3_ModifyAndPersist() {
    std::cout << "\n=== Example 3: Modifying and Re-Persisting ===\n" << std::endl;
    
    const std::string dbPath = "./example_network_db";
    
    Datastore datastore(dbPath, 1000);
    registerFactories(datastore);
    
    // Load a neuron
    uint64_t neuron1Id = 100000000000000ULL;
    auto neuron1 = datastore.getNeuron(neuron1Id);
    
    if (neuron1) {
        std::cout << "Loaded neuron: " << neuron1->getId() << std::endl;
        
        // Modify the neuron by adding spikes
        std::cout << "Adding spike events..." << std::endl;
        neuron1->insertSpike(10.0);
        neuron1->insertSpike(20.0);
        neuron1->insertSpike(30.0);
        
        // Mark as dirty and flush
        datastore.markDirty(neuron1->getId());
        bool flushed = datastore.flush(neuron1->getId());
        
        std::cout << "  ✓ Modified neuron and flushed to disk: " << (flushed ? "success" : "failed") << std::endl;
    }
    
    std::cout << "\n✓ Modification complete!" << std::endl;
}

int main() {
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   SNNFW Datastore Example - Persistent Neural Networks    ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
    
    try {
        // Run examples
        example1_CreateAndPersist();
        example2_LoadNetwork();
        example3_ModifyAndPersist();
        
        std::cout << "\n╔════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║              All Examples Completed Successfully!         ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

