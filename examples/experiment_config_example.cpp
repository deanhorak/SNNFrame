/**
 * @file experiment_config_example.cpp
 * @brief Example demonstrating experiment-based datastore management with named hierarchical structures
 * 
 * This example shows:
 * 1. Using ExperimentConfig to manage experiment-specific datastores
 * 2. Creating named hierarchical neural structures (Brain, Hemisphere, Lobe, etc.)
 * 3. Persisting and loading experiments
 * 4. Managing multiple experiments
 */

#include "snnfw/ExperimentConfig.h"
#include "snnfw/Datastore.h"
#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Brain.h"
#include "snnfw/Hemisphere.h"
#include "snnfw/Lobe.h"
#include "snnfw/Region.h"
#include "snnfw/Nucleus.h"
#include "snnfw/Logger.h"
#include <iostream>
#include <memory>

using namespace snnfw;

// Helper function to register all factories with the datastore
void registerFactories(Datastore& datastore) {
    datastore.registerFactory("Brain", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto brain = std::make_shared<Brain>();
        return brain->fromJson(json) ? brain : nullptr;
    });
    
    datastore.registerFactory("Hemisphere", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto hemisphere = std::make_shared<Hemisphere>();
        return hemisphere->fromJson(json) ? hemisphere : nullptr;
    });
    
    datastore.registerFactory("Lobe", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto lobe = std::make_shared<Lobe>();
        return lobe->fromJson(json) ? lobe : nullptr;
    });
    
    datastore.registerFactory("Region", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto region = std::make_shared<Region>();
        return region->fromJson(json) ? region : nullptr;
    });
    
    datastore.registerFactory("Nucleus", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto nucleus = std::make_shared<Nucleus>();
        return nucleus->fromJson(json) ? nucleus : nullptr;
    });
}

// Example 1: Creating a named brain structure for an experiment
void example1_CreateNamedBrainStructure() {
    std::cout << "\n=== Example 1: Creating Named Brain Structure ===\n" << std::endl;
    
    // Create experiment configuration
    ExperimentConfig config("visual_cortex_study");
    config.createDirectories();
    
    std::cout << "Experiment: " << config.getName() << std::endl;
    std::cout << "Datastore path: " << config.getDatastorePath() << std::endl;
    
    // Create datastore for this experiment
    Datastore datastore(config.getDatastorePath(), 10000);
    registerFactories(datastore);
    
    // Create factory
    NeuralObjectFactory factory;
    
    // Create a brain with hemispheres
    auto brain = factory.createBrain();
    brain->setName("Human Visual System Model");
    
    // Create left hemisphere
    auto leftHemisphere = factory.createHemisphere();
    leftHemisphere->setName("Left Hemisphere");
    brain->addHemisphere(leftHemisphere->getId());
    
    // Create right hemisphere
    auto rightHemisphere = factory.createHemisphere();
    rightHemisphere->setName("Right Hemisphere");
    brain->addHemisphere(rightHemisphere->getId());
    
    // Create occipital lobe in left hemisphere
    auto occipitalLobe = factory.createLobe();
    occipitalLobe->setName("Occipital Lobe");
    leftHemisphere->addLobe(occipitalLobe->getId());
    
    // Create visual cortex regions
    auto v1Region = factory.createRegion();
    v1Region->setName("Primary Visual Cortex (V1)");
    occipitalLobe->addRegion(v1Region->getId());
    
    auto v2Region = factory.createRegion();
    v2Region->setName("Secondary Visual Cortex (V2)");
    occipitalLobe->addRegion(v2Region->getId());
    
    // Create nuclei in V1
    auto v1Nucleus1 = factory.createNucleus();
    v1Nucleus1->setName("V1 Layer 4C");
    v1Region->addNucleus(v1Nucleus1->getId());
    
    auto v1Nucleus2 = factory.createNucleus();
    v1Nucleus2->setName("V1 Layer 2/3");
    v1Region->addNucleus(v1Nucleus2->getId());
    
    std::cout << "\nCreated brain structure:" << std::endl;
    std::cout << "  Brain: " << brain->getName() << " (ID: " << brain->getId() << ")" << std::endl;
    std::cout << "    Left Hemisphere: " << leftHemisphere->getName() << " (ID: " << leftHemisphere->getId() << ")" << std::endl;
    std::cout << "      Occipital Lobe: " << occipitalLobe->getName() << " (ID: " << occipitalLobe->getId() << ")" << std::endl;
    std::cout << "        V1 Region: " << v1Region->getName() << " (ID: " << v1Region->getId() << ")" << std::endl;
    std::cout << "          Nucleus: " << v1Nucleus1->getName() << " (ID: " << v1Nucleus1->getId() << ")" << std::endl;
    std::cout << "          Nucleus: " << v1Nucleus2->getName() << " (ID: " << v1Nucleus2->getId() << ")" << std::endl;
    std::cout << "        V2 Region: " << v2Region->getName() << " (ID: " << v2Region->getId() << ")" << std::endl;
    std::cout << "    Right Hemisphere: " << rightHemisphere->getName() << " (ID: " << rightHemisphere->getId() << ")" << std::endl;
    
    // Store all objects
    datastore.put(brain);
    datastore.put(leftHemisphere);
    datastore.put(rightHemisphere);
    datastore.put(occipitalLobe);
    datastore.put(v1Region);
    datastore.put(v2Region);
    datastore.put(v1Nucleus1);
    datastore.put(v1Nucleus2);
    
    // Mark all as dirty and flush
    datastore.markDirty(brain->getId());
    datastore.markDirty(leftHemisphere->getId());
    datastore.markDirty(rightHemisphere->getId());
    datastore.markDirty(occipitalLobe->getId());
    datastore.markDirty(v1Region->getId());
    datastore.markDirty(v2Region->getId());
    datastore.markDirty(v1Nucleus1->getId());
    datastore.markDirty(v1Nucleus2->getId());
    
    size_t flushed = datastore.flushAll();
    std::cout << "\n✓ Flushed " << flushed << " objects to experiment datastore" << std::endl;
}

// Example 2: Loading an experiment
void example2_LoadExperiment() {
    std::cout << "\n=== Example 2: Loading Experiment ===\n" << std::endl;
    
    // Load the same experiment
    ExperimentConfig config("visual_cortex_study");
    
    if (!config.exists()) {
        std::cout << "✗ Experiment does not exist!" << std::endl;
        return;
    }
    
    std::cout << "Loading experiment: " << config.getName() << std::endl;
    std::cout << "Experiment size: " << config.getExperimentSize() << " bytes" << std::endl;
    
    // Create datastore
    Datastore datastore(config.getDatastorePath(), 10000);
    registerFactories(datastore);
    
    // Load the brain (we know the first brain ID)
    uint64_t brainId = 1200000000000000ULL;
    auto brain = std::dynamic_pointer_cast<Brain>(datastore.get(brainId));
    
    if (brain) {
        std::cout << "\n✓ Loaded brain: " << brain->getName() << std::endl;
        std::cout << "  Number of hemispheres: " << brain->getHemisphereIds().size() << std::endl;
        
        // Load hemispheres
        for (uint64_t hemisphereId : brain->getHemisphereIds()) {
            auto hemisphere = std::dynamic_pointer_cast<Hemisphere>(datastore.get(hemisphereId));
            if (hemisphere) {
                std::cout << "    Hemisphere: " << hemisphere->getName() << std::endl;
                std::cout << "      Number of lobes: " << hemisphere->getLobeIds().size() << std::endl;
                
                // Load lobes
                for (uint64_t lobeId : hemisphere->getLobeIds()) {
                    auto lobe = std::dynamic_pointer_cast<Lobe>(datastore.get(lobeId));
                    if (lobe) {
                        std::cout << "        Lobe: " << lobe->getName() << std::endl;
                        std::cout << "          Number of regions: " << lobe->getRegionIds().size() << std::endl;
                        
                        // Load regions
                        for (uint64_t regionId : lobe->getRegionIds()) {
                            auto region = std::dynamic_pointer_cast<Region>(datastore.get(regionId));
                            if (region) {
                                std::cout << "            Region: " << region->getName() << std::endl;
                                std::cout << "              Number of nuclei: " << region->getNucleusIds().size() << std::endl;
                                
                                // Load nuclei
                                for (uint64_t nucleusId : region->getNucleusIds()) {
                                    auto nucleus = std::dynamic_pointer_cast<Nucleus>(datastore.get(nucleusId));
                                    if (nucleus) {
                                        std::cout << "                Nucleus: " << nucleus->getName() << std::endl;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } else {
        std::cout << "✗ Failed to load brain" << std::endl;
    }
}

// Example 3: Managing multiple experiments
void example3_MultipleExperiments() {
    std::cout << "\n=== Example 3: Managing Multiple Experiments ===\n" << std::endl;
    
    // Create multiple experiments
    std::vector<std::string> experimentNames = {
        "motor_cortex_study",
        "auditory_processing",
        "language_networks"
    };
    
    for (const auto& name : experimentNames) {
        ExperimentConfig config(name);
        config.createDirectories();
        
        std::cout << "Created experiment: " << name << std::endl;
        std::cout << "  Path: " << config.getExperimentDirectory() << std::endl;
        std::cout << "  Datastore: " << config.getDatastorePath() << std::endl;
        
        // You could create different brain structures for each experiment here
    }
    
    std::cout << "\n✓ Multiple experiments created successfully!" << std::endl;
    std::cout << "  Each experiment has its own isolated datastore" << std::endl;
}

// Example 4: Experiment cleanup
void example4_CleanupExperiment() {
    std::cout << "\n=== Example 4: Experiment Cleanup ===\n" << std::endl;
    
    // Clean up the test experiments
    std::vector<std::string> experimentsToClean = {
        "motor_cortex_study",
        "auditory_processing",
        "language_networks"
    };
    
    for (const auto& name : experimentsToClean) {
        ExperimentConfig config(name);
        if (config.exists()) {
            size_t size = config.getExperimentSize();
            config.deleteExperiment();
            std::cout << "Deleted experiment: " << name << " (" << size << " bytes)" << std::endl;
        }
    }
    
    std::cout << "\n✓ Cleanup complete!" << std::endl;
}

int main() {
    std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   SNNFW Experiment Configuration Example                  ║" << std::endl;
    std::cout << "║   Named Hierarchical Structures & Experiment Management   ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
    
    try {
        example1_CreateNamedBrainStructure();
        example2_LoadExperiment();
        example3_MultipleExperiments();
        example4_CleanupExperiment();
        
        std::cout << "\n╔════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║              All Examples Completed Successfully!         ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

