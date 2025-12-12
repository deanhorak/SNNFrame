#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Logger.h"
#include <iostream>
#include <iomanip>

using namespace snnfw;

/**
 * @brief Example demonstrating the NeuralObjectFactory
 *
 * This example shows how to use the NeuralObjectFactory to create
 * neural network components with automatically assigned IDs from
 * type-specific ranges.
 */
int main() {
    // Initialize logger
    Logger::getInstance().initialize("factory_example.log", spdlog::level::info);
    
    std::cout << "=== NeuralObjectFactory Example ===" << std::endl;
    std::cout << std::endl;
    
    // Create a factory instance
    NeuralObjectFactory factory;
    
    std::cout << "ID Range Assignments (100 trillion IDs per type):" << std::endl;
    std::cout << "  Neurons:   " << NeuralObjectFactory::NEURON_ID_START
              << " - " << NeuralObjectFactory::NEURON_ID_END << std::endl;
    std::cout << "  Axons:     " << NeuralObjectFactory::AXON_ID_START
              << " - " << NeuralObjectFactory::AXON_ID_END << std::endl;
    std::cout << "  Dendrites: " << NeuralObjectFactory::DENDRITE_ID_START
              << " - " << NeuralObjectFactory::DENDRITE_ID_END << std::endl;
    std::cout << "  Synapses:  " << NeuralObjectFactory::SYNAPSE_ID_START
              << " - " << NeuralObjectFactory::SYNAPSE_ID_END << std::endl;
    std::cout << "  Clusters:  " << NeuralObjectFactory::CLUSTER_ID_START
              << " - " << NeuralObjectFactory::CLUSTER_ID_END << std::endl;
    std::cout << std::endl;
    std::cout << "Biological Context:" << std::endl;
    std::cout << "  Human brain: ~86 billion neurons, ~100 trillion synapses" << std::endl;
    std::cout << "  These ranges support networks at or beyond biological scale" << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // Example 1: Create individual components
    // ========================================================================
    
    std::cout << "--- Example 1: Creating Individual Components ---" << std::endl;
    
    // Create a neuron
    auto neuron1 = factory.createNeuron(50.0, 0.95, 20);
    std::cout << "Created Neuron with ID: " << neuron1->getId() 
              << " (Type: " << NeuralObjectFactory::getObjectTypeName(neuron1->getId()) << ")"
              << std::endl;
    
    // Create an axon
    auto axon1 = factory.createAxon(neuron1->getId());
    std::cout << "Created Axon with ID: " << axon1->getId() 
              << " (Type: " << NeuralObjectFactory::getObjectTypeName(axon1->getId()) << ")"
              << " for Neuron " << neuron1->getId() << std::endl;
    
    // Create a dendrite
    auto dendrite1 = factory.createDendrite(neuron1->getId());
    std::cout << "Created Dendrite with ID: " << dendrite1->getId() 
              << " (Type: " << NeuralObjectFactory::getObjectTypeName(dendrite1->getId()) << ")"
              << " for Neuron " << neuron1->getId() << std::endl;
    
    // Create a synapse
    auto synapse1 = factory.createSynapse(axon1->getId(), dendrite1->getId(), 1.0, 1.5);
    std::cout << "Created Synapse with ID: " << synapse1->getId() 
              << " (Type: " << NeuralObjectFactory::getObjectTypeName(synapse1->getId()) << ")"
              << " connecting Axon " << axon1->getId() 
              << " to Dendrite " << dendrite1->getId() << std::endl;
    
    // Create a cluster
    auto cluster1 = factory.createCluster();
    std::cout << "Created Cluster with ID: " << cluster1->getId() 
              << " (Type: " << NeuralObjectFactory::getObjectTypeName(cluster1->getId()) << ")"
              << std::endl;
    
    std::cout << std::endl;
    
    // ========================================================================
    // Example 2: Create a simple neural connection
    // ========================================================================
    
    std::cout << "--- Example 2: Creating a Neural Connection ---" << std::endl;
    
    // Create two neurons
    auto presynapticNeuron = factory.createNeuron(50.0, 0.95, 20);
    auto postsynapticNeuron = factory.createNeuron(50.0, 0.95, 20);
    
    std::cout << "Created presynaptic neuron: " << presynapticNeuron->getId() << std::endl;
    std::cout << "Created postsynaptic neuron: " << postsynapticNeuron->getId() << std::endl;
    
    // Create axon for presynaptic neuron
    auto axon = factory.createAxon(presynapticNeuron->getId());
    std::cout << "Created axon: " << axon->getId() 
              << " (source: " << axon->getSourceNeuronId() << ")" << std::endl;
    
    // Create dendrite for postsynaptic neuron
    auto dendrite = factory.createDendrite(postsynapticNeuron->getId());
    std::cout << "Created dendrite: " << dendrite->getId() 
              << " (target: " << dendrite->getTargetNeuronId() << ")" << std::endl;
    
    // Create synapse connecting them
    auto synapse = factory.createSynapse(axon->getId(), dendrite->getId(), 1.5, 2.0);
    std::cout << "Created synapse: " << synapse->getId() 
              << " (weight: " << synapse->getWeight() 
              << ", delay: " << synapse->getDelay() << "ms)" << std::endl;
    
    std::cout << "Connection established: Neuron " << presynapticNeuron->getId() 
              << " -> Axon " << axon->getId() 
              << " -> Synapse " << synapse->getId() 
              << " -> Dendrite " << dendrite->getId() 
              << " -> Neuron " << postsynapticNeuron->getId() << std::endl;
    
    std::cout << std::endl;
    
    // ========================================================================
    // Example 3: Create a small network
    // ========================================================================
    
    std::cout << "--- Example 3: Creating a Small Network (3 neurons, fully connected) ---" << std::endl;
    
    // Create 3 neurons
    std::vector<std::shared_ptr<Neuron>> neurons;
    for (int i = 0; i < 3; ++i) {
        neurons.push_back(factory.createNeuron(50.0, 0.95, 20));
        std::cout << "Created neuron " << i << ": " << neurons[i]->getId() << std::endl;
    }
    
    // Create axons for each neuron
    std::vector<std::shared_ptr<Axon>> axons;
    for (size_t i = 0; i < neurons.size(); ++i) {
        axons.push_back(factory.createAxon(neurons[i]->getId()));
        std::cout << "Created axon for neuron " << i << ": " << axons[i]->getId() << std::endl;
    }
    
    // Create dendrites for each neuron
    std::vector<std::shared_ptr<Dendrite>> dendrites;
    for (size_t i = 0; i < neurons.size(); ++i) {
        dendrites.push_back(factory.createDendrite(neurons[i]->getId()));
        std::cout << "Created dendrite for neuron " << i << ": " << dendrites[i]->getId() << std::endl;
    }
    
    // Create synapses (fully connected)
    std::cout << "Creating synapses (fully connected):" << std::endl;
    int synapseCount = 0;
    for (size_t i = 0; i < axons.size(); ++i) {
        for (size_t j = 0; j < dendrites.size(); ++j) {
            auto syn = factory.createSynapse(axons[i]->getId(), dendrites[j]->getId());
            std::cout << "  Synapse " << synapseCount++ << ": " << syn->getId() 
                      << " (Neuron " << i << " -> Neuron " << j << ")" << std::endl;
        }
    }
    
    std::cout << std::endl;
    
    // ========================================================================
    // Example 4: Factory statistics
    // ========================================================================
    
    std::cout << "--- Example 4: Factory Statistics ---" << std::endl;
    
    std::cout << "Objects created:" << std::endl;
    std::cout << "  Neurons:   " << factory.getObjectCount(NeuralObjectFactory::ObjectType::NEURON) << std::endl;
    std::cout << "  Axons:     " << factory.getObjectCount(NeuralObjectFactory::ObjectType::AXON) << std::endl;
    std::cout << "  Dendrites: " << factory.getObjectCount(NeuralObjectFactory::ObjectType::DENDRITE) << std::endl;
    std::cout << "  Synapses:  " << factory.getObjectCount(NeuralObjectFactory::ObjectType::SYNAPSE) << std::endl;
    std::cout << "  Clusters:  " << factory.getObjectCount(NeuralObjectFactory::ObjectType::CLUSTER) << std::endl;
    
    std::cout << std::endl;
    std::cout << "Current ID counters:" << std::endl;
    std::cout << "  Neurons:   " << factory.getCurrentId(NeuralObjectFactory::ObjectType::NEURON) << std::endl;
    std::cout << "  Axons:     " << factory.getCurrentId(NeuralObjectFactory::ObjectType::AXON) << std::endl;
    std::cout << "  Dendrites: " << factory.getCurrentId(NeuralObjectFactory::ObjectType::DENDRITE) << std::endl;
    std::cout << "  Synapses:  " << factory.getCurrentId(NeuralObjectFactory::ObjectType::SYNAPSE) << std::endl;
    std::cout << "  Clusters:  " << factory.getCurrentId(NeuralObjectFactory::ObjectType::CLUSTER) << std::endl;
    
    std::cout << std::endl;
    
    // ========================================================================
    // Example 5: Type identification from ID
    // ========================================================================
    
    std::cout << "--- Example 5: Type Identification from ID ---" << std::endl;
    
    uint64_t testIds[] = {
        neuron1->getId(),
        axon1->getId(),
        dendrite1->getId(),
        synapse1->getId(),
        cluster1->getId()
    };
    
    for (uint64_t id : testIds) {
        auto type = NeuralObjectFactory::getObjectType(id);
        std::cout << "ID " << id << " is a " 
                  << NeuralObjectFactory::getObjectTypeName(type) << std::endl;
    }
    
    std::cout << std::endl;
    std::cout << "=== Example Complete ===" << std::endl;
    
    return 0;
}

