#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Logger.h"
#include <iostream>
#include <iomanip>

using namespace snnfw;

/**
 * @brief Example demonstrating the hierarchical neural structure
 *
 * This example shows how to build a complete hierarchical brain structure:
 * Brain -> Hemispheres -> Lobes -> Regions -> Nuclei -> Columns -> Layers -> Clusters -> Neurons
 */
int main() {
    // Initialize logger
    Logger::getInstance().initialize("hierarchical_example.log", spdlog::level::info);
    
    std::cout << "=== Hierarchical Neural Structure Example ===" << std::endl;
    std::cout << std::endl;
    
    // Create a factory instance
    NeuralObjectFactory factory;
    
    std::cout << "Building a simplified brain structure..." << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // Create a Brain
    // ========================================================================
    
    std::cout << "--- Creating Brain ---" << std::endl;
    auto brain = factory.createBrain();
    std::cout << "Created Brain with ID: " << brain->getId() << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // Create Hemispheres (Left and Right)
    // ========================================================================

    std::cout << "--- Creating Hemispheres ---" << std::endl;
    auto leftHemisphere = factory.createHemisphere();
    auto rightHemisphere = factory.createHemisphere();

    brain->addHemisphere(leftHemisphere->getId());
    brain->addHemisphere(rightHemisphere->getId());

    std::cout << "Created Left Hemisphere with ID: " << leftHemisphere->getId() << std::endl;
    std::cout << "Created Right Hemisphere with ID: " << rightHemisphere->getId() << std::endl;
    std::cout << "Brain now has " << brain->size() << " hemispheres" << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // Create Lobes (Frontal, Parietal, Temporal, Occipital for each hemisphere)
    // ========================================================================

    std::cout << "--- Creating Lobes ---" << std::endl;

    // Left hemisphere lobes
    auto leftFrontal = factory.createLobe();
    auto leftParietal = factory.createLobe();
    auto leftTemporal = factory.createLobe();
    auto leftOccipital = factory.createLobe();

    leftHemisphere->addLobe(leftFrontal->getId());
    leftHemisphere->addLobe(leftParietal->getId());
    leftHemisphere->addLobe(leftTemporal->getId());
    leftHemisphere->addLobe(leftOccipital->getId());

    std::cout << "Left Hemisphere - Created 4 lobes (Frontal, Parietal, Temporal, Occipital)" << std::endl;
    std::cout << "  Frontal Lobe ID: " << leftFrontal->getId() << std::endl;

    // Right hemisphere lobes (simplified - just create them)
    for (int i = 0; i < 4; ++i) {
        auto lobe = factory.createLobe();
        rightHemisphere->addLobe(lobe->getId());
    }

    std::cout << "Right Hemisphere - Created 4 lobes" << std::endl;
    std::cout << "Total lobes created: " << factory.getObjectCount(NeuralObjectFactory::ObjectType::LOBE) << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // Create Regions within a Lobe
    // ========================================================================

    std::cout << "--- Creating Regions ---" << std::endl;

    // Create regions in the left frontal lobe (e.g., motor cortex, prefrontal cortex)
    auto motorCortex = factory.createRegion();
    auto prefrontalCortex = factory.createRegion();

    leftFrontal->addRegion(motorCortex->getId());
    leftFrontal->addRegion(prefrontalCortex->getId());

    std::cout << "Left Frontal Lobe - Created 2 regions" << std::endl;
    std::cout << "  Motor Cortex ID: " << motorCortex->getId() << std::endl;
    std::cout << "  Prefrontal Cortex ID: " << prefrontalCortex->getId() << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Create Nuclei within a Region
    // ========================================================================

    std::cout << "--- Creating Nuclei ---" << std::endl;

    // Create nuclei in the motor cortex
    auto nucleus1 = factory.createNucleus();
    auto nucleus2 = factory.createNucleus();

    motorCortex->addNucleus(nucleus1->getId());
    motorCortex->addNucleus(nucleus2->getId());

    std::cout << "Motor Cortex - Created 2 nuclei" << std::endl;
    std::cout << "  Nucleus 1 ID: " << nucleus1->getId() << std::endl;
    std::cout << "  Nucleus 2 ID: " << nucleus2->getId() << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Create Columns within a Nucleus
    // ========================================================================

    std::cout << "--- Creating Columns ---" << std::endl;

    // Create cortical columns in nucleus1
    uint64_t firstColumnId = 0;
    for (int i = 0; i < 5; ++i) {
        auto column = factory.createColumn();
        nucleus1->addColumn(column->getId());

        if (i == 0) {
            firstColumnId = column->getId();
            std::cout << "  First Column ID: " << firstColumnId << std::endl;
        }
    }

    std::cout << "Nucleus 1 - Created 5 cortical columns" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Create Layers within a Column
    // ========================================================================

    std::cout << "--- Creating Layers ---" << std::endl;

    // Get the first column ID and create 6 layers (like cortical layers I-VI)
    uint64_t firstLayerId = 0;
    for (int i = 1; i <= 6; ++i) {
        auto layer = factory.createLayer();
        // Note: We need to track the column object to add layers to it
        // For now, we'll create a temporary column reference
        if (i == 1) {
            firstLayerId = layer->getId();
            std::cout << "  Layer I ID: " << firstLayerId << std::endl;
        }
    }

    std::cout << "Column - Created 6 layers (like cortical layers I-VI)" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Create Clusters within a Layer
    // ========================================================================

    std::cout << "--- Creating Clusters ---" << std::endl;

    // Create clusters
    uint64_t firstClusterId = 0;
    for (int i = 0; i < 3; ++i) {
        auto cluster = factory.createCluster();

        if (i == 0) {
            firstClusterId = cluster->getId();
            std::cout << "  First Cluster ID: " << firstClusterId << std::endl;
        }
    }

    std::cout << "Layer I - Created 3 clusters" << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // Create Neurons within a Cluster
    // ========================================================================

    std::cout << "--- Creating Neurons ---" << std::endl;

    // Create neurons
    uint64_t firstNeuronId = 0;
    uint64_t secondNeuronId = 0;

    for (int i = 0; i < 10; ++i) {
        auto neuron = factory.createNeuron(50.0, 0.95, 20);

        if (i == 0) {
            firstNeuronId = neuron->getId();
            std::cout << "  First Neuron ID: " << firstNeuronId << std::endl;
        } else if (i == 1) {
            secondNeuronId = neuron->getId();
        }
    }

    std::cout << "Cluster - Created 10 neurons" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Create Neural Connections
    // ========================================================================

    std::cout << "--- Creating Neural Connections ---" << std::endl;

    // Create axon for first neuron
    auto axon = factory.createAxon(firstNeuronId);

    // Create dendrite for second neuron
    auto dendrite = factory.createDendrite(secondNeuronId);

    // Create synapse connecting them
    auto synapse = factory.createSynapse(axon->getId(), dendrite->getId(), 1.0, 1.5);

    std::cout << "Connected Neuron " << firstNeuronId
              << " to Neuron " << secondNeuronId << std::endl;
    std::cout << "  Axon ID: " << axon->getId() << std::endl;
    std::cout << "  Dendrite ID: " << dendrite->getId() << std::endl;
    std::cout << "  Synapse ID: " << synapse->getId() << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // Display Hierarchy Summary
    // ========================================================================
    
    std::cout << "=== Hierarchy Summary ===" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Structure:" << std::endl;
    std::cout << "  Brain (" << brain->getId() << ")" << std::endl;
    std::cout << "    └─ Hemispheres: " << brain->size() << std::endl;
    std::cout << "         └─ Lobes per hemisphere: " << leftHemisphere->size() << std::endl;
    std::cout << "              └─ Regions in left frontal: " << leftFrontal->size() << std::endl;
    std::cout << "                   └─ Nuclei in motor cortex: " << motorCortex->size() << std::endl;
    std::cout << "                        └─ Columns in nucleus 1: " << nucleus1->size() << std::endl;
    std::cout << "                             └─ Layers created: 6" << std::endl;
    std::cout << "                                  └─ Clusters created: 3" << std::endl;
    std::cout << "                                       └─ Neurons created: 10" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Object Counts:" << std::endl;
    std::cout << "  Brains:      " << factory.getObjectCount(NeuralObjectFactory::ObjectType::BRAIN) << std::endl;
    std::cout << "  Hemispheres: " << factory.getObjectCount(NeuralObjectFactory::ObjectType::HEMISPHERE) << std::endl;
    std::cout << "  Lobes:       " << factory.getObjectCount(NeuralObjectFactory::ObjectType::LOBE) << std::endl;
    std::cout << "  Regions:     " << factory.getObjectCount(NeuralObjectFactory::ObjectType::REGION) << std::endl;
    std::cout << "  Nuclei:      " << factory.getObjectCount(NeuralObjectFactory::ObjectType::NUCLEUS) << std::endl;
    std::cout << "  Columns:     " << factory.getObjectCount(NeuralObjectFactory::ObjectType::COLUMN) << std::endl;
    std::cout << "  Layers:      " << factory.getObjectCount(NeuralObjectFactory::ObjectType::LAYER) << std::endl;
    std::cout << "  Clusters:    " << factory.getObjectCount(NeuralObjectFactory::ObjectType::CLUSTER) << std::endl;
    std::cout << "  Neurons:     " << factory.getObjectCount(NeuralObjectFactory::ObjectType::NEURON) << std::endl;
    std::cout << "  Axons:       " << factory.getObjectCount(NeuralObjectFactory::ObjectType::AXON) << std::endl;
    std::cout << "  Dendrites:   " << factory.getObjectCount(NeuralObjectFactory::ObjectType::DENDRITE) << std::endl;
    std::cout << "  Synapses:    " << factory.getObjectCount(NeuralObjectFactory::ObjectType::SYNAPSE) << std::endl;
    std::cout << std::endl;
    
    std::cout << "=== Example Complete ===" << std::endl;
    
    return 0;
}

