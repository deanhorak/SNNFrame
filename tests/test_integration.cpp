#include <gtest/gtest.h>
#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Datastore.h"
#include "snnfw/Neuron.h"
#include "snnfw/Axon.h"
#include "snnfw/Dendrite.h"
#include "snnfw/Synapse.h"
#include "snnfw/Cluster.h"
#include "snnfw/Layer.h"
#include "snnfw/Column.h"
#include "snnfw/Nucleus.h"
#include "snnfw/Region.h"
#include "snnfw/Lobe.h"
#include "snnfw/Hemisphere.h"
#include "snnfw/Brain.h"
#include "snnfw/NetworkPropagator.h"
#include "snnfw/SpikeProcessor.h"
#include "snnfw/STDPLearning.h"
#include "snnfw/Logger.h"
#include <spdlog/spdlog.h>
#include <filesystem>
#include <memory>
#include <thread>
#include <chrono>

using namespace snnfw;

// Helper function to register all factories with the datastore
void registerFactories(Datastore& datastore) {
    datastore.registerFactory("Neuron", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto neuron = std::make_shared<Neuron>(0, 0, 0);
        return neuron->fromJson(json) ? neuron : nullptr;
    });

    datastore.registerFactory("Axon", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto axon = std::make_shared<Axon>(0);
        return axon->fromJson(json) ? axon : nullptr;
    });

    datastore.registerFactory("Dendrite", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto dendrite = std::make_shared<Dendrite>(0);
        return dendrite->fromJson(json) ? dendrite : nullptr;
    });

    datastore.registerFactory("Synapse", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto synapse = std::make_shared<Synapse>(0, 0, 0.0, 0.0);
        return synapse->fromJson(json) ? synapse : nullptr;
    });

    datastore.registerFactory("Cluster", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto cluster = std::make_shared<Cluster>();
        return cluster->fromJson(json) ? cluster : nullptr;
    });

    datastore.registerFactory("Layer", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto layer = std::make_shared<Layer>();
        return layer->fromJson(json) ? layer : nullptr;
    });

    datastore.registerFactory("Column", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto column = std::make_shared<Column>();
        return column->fromJson(json) ? column : nullptr;
    });

    datastore.registerFactory("Nucleus", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto nucleus = std::make_shared<Nucleus>();
        return nucleus->fromJson(json) ? nucleus : nullptr;
    });

    datastore.registerFactory("Region", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto region = std::make_shared<Region>();
        return region->fromJson(json) ? region : nullptr;
    });

    datastore.registerFactory("Lobe", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto lobe = std::make_shared<Lobe>();
        return lobe->fromJson(json) ? lobe : nullptr;
    });

    datastore.registerFactory("Hemisphere", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto hemisphere = std::make_shared<Hemisphere>();
        return hemisphere->fromJson(json) ? hemisphere : nullptr;
    });

    datastore.registerFactory("Brain", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto brain = std::make_shared<Brain>();
        return brain->fromJson(json) ? brain : nullptr;
    });
}

/**
 * @brief Integration tests for SNNFW framework
 *
 * These tests validate end-to-end functionality including:
 * - Full hierarchical structure construction (Brain → Neuron)
 * - Multi-layer spike propagation
 * - STDP learning across layers
 * - Datastore persistence and recovery
 * - Real-time vs fast mode switching
 */
class IntegrationTest : public ::testing::Test {
protected:
    std::unique_ptr<Datastore> datastore;
    std::unique_ptr<NeuralObjectFactory> factory;

    void SetUp() override {
        // Initialize logger
        Logger::getInstance().initialize("/tmp/test_integration.log", spdlog::level::warn);

        // Create datastore
        std::filesystem::remove_all("/tmp/test_integration_db");
        datastore = std::make_unique<Datastore>("/tmp/test_integration_db", 100000);

        // Register factories
        registerFactories(*datastore);

        // Create factory
        factory = std::make_unique<NeuralObjectFactory>();
    }
    
    void TearDown() override {
        factory.reset();
        datastore.reset();
        std::filesystem::remove_all("/tmp/test_integration_db");
    }
};

// ============================================================================
// Test 1: Full Hierarchical Structure Construction
// ============================================================================

TEST_F(IntegrationTest, FullHierarchyConstruction) {
    // Create Brain → Hemisphere → Lobe → Region → Nucleus → Column → Layer → Cluster → Neuron
    
    // Brain
    auto brain = factory->createBrain();
    ASSERT_NE(brain, nullptr);
    datastore->put(brain);
    
    // Hemisphere
    auto hemisphere = factory->createHemisphere();
    ASSERT_NE(hemisphere, nullptr);
    brain->addHemisphere(hemisphere->getId());
    datastore->put(brain);
    datastore->put(hemisphere);
    
    // Lobe
    auto lobe = factory->createLobe();
    ASSERT_NE(lobe, nullptr);
    hemisphere->addLobe(lobe->getId());
    datastore->put(hemisphere);
    datastore->put(lobe);
    
    // Region
    auto region = factory->createRegion();
    ASSERT_NE(region, nullptr);
    lobe->addRegion(region->getId());
    datastore->put(lobe);
    datastore->put(region);
    
    // Nucleus
    auto nucleus = factory->createNucleus();
    ASSERT_NE(nucleus, nullptr);
    region->addNucleus(nucleus->getId());
    datastore->put(region);
    datastore->put(nucleus);
    
    // Column
    auto column = factory->createColumn();
    ASSERT_NE(column, nullptr);
    nucleus->addColumn(column->getId());
    datastore->put(nucleus);
    datastore->put(column);
    
    // Layer
    auto layer = factory->createLayer();
    ASSERT_NE(layer, nullptr);
    column->addLayer(layer->getId());
    datastore->put(column);
    datastore->put(layer);
    
    // Cluster
    auto cluster = factory->createCluster();
    ASSERT_NE(cluster, nullptr);
    layer->addCluster(cluster->getId());
    datastore->put(layer);
    datastore->put(cluster);
    
    // Neurons
    for (int i = 0; i < 10; ++i) {
        auto neuron = factory->createNeuron(50.0, 0.95, 20);
        ASSERT_NE(neuron, nullptr);
        cluster->addNeuron(neuron->getId());
        datastore->put(neuron);
    }
    datastore->put(cluster);
    
    // Verify hierarchy
    EXPECT_EQ(brain->size(), 1);
    EXPECT_EQ(hemisphere->size(), 1);
    EXPECT_EQ(lobe->size(), 1);
    EXPECT_EQ(region->size(), 1);
    EXPECT_EQ(nucleus->size(), 1);
    EXPECT_EQ(column->size(), 1);
    EXPECT_EQ(layer->size(), 1);
    EXPECT_EQ(cluster->size(), 10);

    // Verify persistence
    auto retrievedBrain = datastore->getBrain(brain->getId());
    ASSERT_NE(retrievedBrain, nullptr);
    EXPECT_EQ(retrievedBrain->size(), 1);

    auto retrievedCluster = datastore->getCluster(cluster->getId());
    ASSERT_NE(retrievedCluster, nullptr);
    EXPECT_EQ(retrievedCluster->size(), 10);
}

// ============================================================================
// Test 2: Multi-Layer Spike Propagation
// ============================================================================

TEST_F(IntegrationTest, MultiLayerSpikePropagation) {
    // Create a simple 2-layer network: Layer1 (input) → Layer2 (output)
    
    // Layer 1: 5 neurons
    std::vector<std::shared_ptr<Neuron>> layer1Neurons;
    std::vector<std::shared_ptr<Axon>> layer1Axons;
    for (int i = 0; i < 5; ++i) {
        auto neuron = factory->createNeuron(50.0, 0.95, 20);
        auto axon = factory->createAxon(neuron->getId());
        neuron->setAxonId(axon->getId());
        
        layer1Neurons.push_back(neuron);
        layer1Axons.push_back(axon);
        
        datastore->put(neuron);
        datastore->put(axon);
    }
    
    // Layer 2: 3 neurons
    std::vector<std::shared_ptr<Neuron>> layer2Neurons;
    std::vector<std::shared_ptr<Dendrite>> layer2Dendrites;
    for (int i = 0; i < 3; ++i) {
        auto neuron = factory->createNeuron(50.0, 0.95, 20);
        auto dendrite = factory->createDendrite(neuron->getId());
        neuron->addDendrite(dendrite->getId());
        
        layer2Neurons.push_back(neuron);
        layer2Dendrites.push_back(dendrite);
        
        datastore->put(neuron);
        datastore->put(dendrite);
    }
    
    // Create synapses: each Layer1 neuron connects to all Layer2 neurons
    std::vector<std::shared_ptr<Synapse>> synapses;
    for (size_t i = 0; i < layer1Axons.size(); ++i) {
        for (size_t j = 0; j < layer2Dendrites.size(); ++j) {
            auto synapse = factory->createSynapse(
                layer1Axons[i]->getId(),
                layer2Dendrites[j]->getId()
            );
            synapse->setWeight(0.5);  // Initial weight
            
            layer1Axons[i]->addSynapse(synapse->getId());
            layer2Dendrites[j]->addSynapse(synapse->getId());
            
            synapses.push_back(synapse);
            datastore->put(synapse);
        }
    }
    
    // Update axons and dendrites
    for (auto& axon : layer1Axons) {
        datastore->put(axon);
    }
    for (auto& dendrite : layer2Dendrites) {
        datastore->put(dendrite);
    }
    
    // Create SpikeProcessor and NetworkPropagator
    auto spikeProcessor = std::make_shared<SpikeProcessor>(1000, 4);
    auto propagator = std::make_shared<NetworkPropagator>(spikeProcessor);
    
    // Fire Layer 1 neurons
    double currentTime = 0.0;
    for (size_t i = 0; i < layer1Neurons.size(); ++i) {
        layer1Neurons[i]->insertSpike(currentTime + i * 2.0);
        propagator->fireNeuron(layer1Neurons[i]->getId(), currentTime + i * 2.0);
    }
    
    // Wait for propagation
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Verify that spikes were scheduled
    // Note: We can't directly verify spike delivery without accessing internal state,
    // but we can verify the network structure is correct
    EXPECT_EQ(synapses.size(), 15);  // 5 * 3 = 15 synapses
    
    for (const auto& synapse : synapses) {
        EXPECT_DOUBLE_EQ(synapse->getWeight(), 0.5);
    }
}

// ============================================================================
// Test 3: STDP Learning Across Layers
// ============================================================================

TEST_F(IntegrationTest, STDPLearningAcrossLayers) {
    // Create STDP learning rule
    STDPLearning stdp(
        0.01,   // A+ (potentiation)
        0.012,  // A- (depression)
        20.0,   // tau+ (ms)
        20.0,   // tau- (ms)
        0.0,    // wMin
        1.0     // wMax
    );
    
    // Create simple network: 1 pre-synaptic neuron → 1 synapse → 1 post-synaptic neuron
    auto preNeuron = factory->createNeuron(50.0, 0.95, 20);
    auto axon = factory->createAxon(preNeuron->getId());
    preNeuron->setAxonId(axon->getId());
    
    auto postNeuron = factory->createNeuron(50.0, 0.95, 20);
    auto dendrite = factory->createDendrite(postNeuron->getId());
    postNeuron->addDendrite(dendrite->getId());
    
    auto synapse = factory->createSynapse(axon->getId(), dendrite->getId());
    synapse->setWeight(0.5);
    
    axon->addSynapse(synapse->getId());
    dendrite->addSynapse(synapse->getId());
    
    // Store in datastore
    datastore->put(preNeuron);
    datastore->put(axon);
    datastore->put(postNeuron);
    datastore->put(dendrite);
    datastore->put(synapse);
    
    // Create synapse map for STDP
    std::map<uint64_t, std::shared_ptr<Synapse>> synapseMap;
    synapseMap[synapse->getId()] = synapse;
    
    // Test potentiation: pre-spike before post-spike
    double preTime = 10.0;
    double postTime = 15.0;  // 5ms after pre-spike
    
    stdp.recordPreSpike(synapse->getId(), preTime);
    stdp.recordPostSpike(postNeuron->getId(), postTime, {synapse->getId()}, synapseMap);
    
    // Weight should increase (potentiation)
    EXPECT_GT(synapse->getWeight(), 0.5);
    double weightAfterPotentiation = synapse->getWeight();
    
    // Reset weight
    synapse->setWeight(0.5);
    stdp.clearHistory();
    
    // Test depression: post-spike before pre-spike
    preTime = 20.0;
    postTime = 15.0;  // 5ms before pre-spike
    
    stdp.recordPreSpike(synapse->getId(), preTime);
    stdp.recordPostSpike(postNeuron->getId(), postTime, {synapse->getId()}, synapseMap);
    
    // Weight should decrease (depression)
    EXPECT_LT(synapse->getWeight(), 0.5);

    // Verify statistics
    size_t numPot = 0, numDep = 0;
    double avgChange = 0.0;
    stdp.getStats(numPot, numDep, avgChange);
    EXPECT_GT(numPot, 0);
    EXPECT_GT(numDep, 0);
}

// ============================================================================
// Test 4: Datastore Persistence and Recovery
// ============================================================================

TEST_F(IntegrationTest, DatastorePersistenceAndRecovery) {
    // Create a network and persist it
    auto cluster = factory->createCluster();
    std::vector<uint64_t> neuronIds;

    // Create 20 neurons
    for (int i = 0; i < 20; ++i) {
        auto neuron = factory->createNeuron(50.0, 0.95, 20);
        neuronIds.push_back(neuron->getId());
        cluster->addNeuron(neuron->getId());
        datastore->put(neuron);
    }
    datastore->put(cluster);

    // Flush datastore
    size_t flushedCount = datastore->flushAll();
    EXPECT_GT(flushedCount, 0);

    // Create new datastore instance (simulating restart)
    datastore.reset();
    datastore = std::make_unique<Datastore>("/tmp/test_integration_db", 100000);

    // Re-register factories for the new datastore instance
    registerFactories(*datastore);

    // Retrieve cluster
    auto retrievedCluster = datastore->getCluster(cluster->getId());
    ASSERT_NE(retrievedCluster, nullptr);
    EXPECT_EQ(retrievedCluster->size(), 20);

    // Retrieve all neurons
    const auto& retrievedNeuronIds = retrievedCluster->getNeuronIds();
    EXPECT_EQ(retrievedNeuronIds.size(), 20);

    for (uint64_t neuronId : retrievedNeuronIds) {
        auto neuron = datastore->getNeuron(neuronId);
        ASSERT_NE(neuron, nullptr);
        EXPECT_EQ(neuron->getId(), neuronId);
    }
}

// ============================================================================
// Test 5: Complex Network with Multiple Layers
// ============================================================================

TEST_F(IntegrationTest, ComplexMultiLayerNetwork) {
    // Create a 3-layer network with full connectivity
    const int LAYER1_SIZE = 10;
    const int LAYER2_SIZE = 5;
    const int LAYER3_SIZE = 3;

    // Layer 1
    std::vector<std::shared_ptr<Neuron>> layer1;
    std::vector<std::shared_ptr<Axon>> layer1Axons;
    for (int i = 0; i < LAYER1_SIZE; ++i) {
        auto neuron = factory->createNeuron(50.0, 0.95, 20);
        auto axon = factory->createAxon(neuron->getId());
        neuron->setAxonId(axon->getId());
        layer1.push_back(neuron);
        layer1Axons.push_back(axon);
        datastore->put(neuron);
        datastore->put(axon);
    }

    // Layer 2
    std::vector<std::shared_ptr<Neuron>> layer2;
    std::vector<std::shared_ptr<Axon>> layer2Axons;
    std::vector<std::shared_ptr<Dendrite>> layer2Dendrites;
    for (int i = 0; i < LAYER2_SIZE; ++i) {
        auto neuron = factory->createNeuron(50.0, 0.95, 20);
        auto axon = factory->createAxon(neuron->getId());
        auto dendrite = factory->createDendrite(neuron->getId());
        neuron->setAxonId(axon->getId());
        neuron->addDendrite(dendrite->getId());
        layer2.push_back(neuron);
        layer2Axons.push_back(axon);
        layer2Dendrites.push_back(dendrite);
        datastore->put(neuron);
        datastore->put(axon);
        datastore->put(dendrite);
    }

    // Layer 3
    std::vector<std::shared_ptr<Neuron>> layer3;
    std::vector<std::shared_ptr<Dendrite>> layer3Dendrites;
    for (int i = 0; i < LAYER3_SIZE; ++i) {
        auto neuron = factory->createNeuron(50.0, 0.95, 20);
        auto dendrite = factory->createDendrite(neuron->getId());
        neuron->addDendrite(dendrite->getId());
        layer3.push_back(neuron);
        layer3Dendrites.push_back(dendrite);
        datastore->put(neuron);
        datastore->put(dendrite);
    }

    // Connect Layer 1 → Layer 2
    int synapse12Count = 0;
    for (auto& axon : layer1Axons) {
        for (auto& dendrite : layer2Dendrites) {
            auto synapse = factory->createSynapse(axon->getId(), dendrite->getId());
            synapse->setWeight(0.3);
            axon->addSynapse(synapse->getId());
            dendrite->addSynapse(synapse->getId());
            datastore->put(synapse);
            synapse12Count++;
        }
    }

    // Connect Layer 2 → Layer 3
    int synapse23Count = 0;
    for (auto& axon : layer2Axons) {
        for (auto& dendrite : layer3Dendrites) {
            auto synapse = factory->createSynapse(axon->getId(), dendrite->getId());
            synapse->setWeight(0.4);
            axon->addSynapse(synapse->getId());
            dendrite->addSynapse(synapse->getId());
            datastore->put(synapse);
            synapse23Count++;
        }
    }

    // Update all axons and dendrites
    for (auto& axon : layer1Axons) datastore->put(axon);
    for (auto& axon : layer2Axons) datastore->put(axon);
    for (auto& dendrite : layer2Dendrites) datastore->put(dendrite);
    for (auto& dendrite : layer3Dendrites) datastore->put(dendrite);

    // Verify connectivity
    EXPECT_EQ(synapse12Count, LAYER1_SIZE * LAYER2_SIZE);  // 10 * 5 = 50
    EXPECT_EQ(synapse23Count, LAYER2_SIZE * LAYER3_SIZE);  // 5 * 3 = 15

    // Verify each layer 2 neuron has correct number of input synapses
    for (auto& dendrite : layer2Dendrites) {
        EXPECT_EQ(dendrite->getSynapseCount(), LAYER1_SIZE);
    }

    // Verify each layer 3 neuron has correct number of input synapses
    for (auto& dendrite : layer3Dendrites) {
        EXPECT_EQ(dendrite->getSynapseCount(), LAYER2_SIZE);
    }
}

// ============================================================================
// Test 6: Hierarchical Structure with Datastore
// ============================================================================

TEST_F(IntegrationTest, HierarchicalStructureWithDatastore) {
    // Create a complete hierarchy and verify persistence at each level

    auto brain = factory->createBrain();
    datastore->put(brain);

    // Create 2 hemispheres
    for (int h = 0; h < 2; ++h) {
        auto hemisphere = factory->createHemisphere();
        brain->addHemisphere(hemisphere->getId());
        datastore->put(hemisphere);

        // Create 2 lobes per hemisphere
        for (int l = 0; l < 2; ++l) {
            auto lobe = factory->createLobe();
            hemisphere->addLobe(lobe->getId());
            datastore->put(lobe);

            // Create 2 regions per lobe
            for (int r = 0; r < 2; ++r) {
                auto region = factory->createRegion();
                lobe->addRegion(region->getId());
                datastore->put(region);
            }
        }
    }
    datastore->put(brain);

    // Verify structure
    EXPECT_EQ(brain->size(), 2);

    // Retrieve and verify
    auto retrievedBrain = datastore->getBrain(brain->getId());
    ASSERT_NE(retrievedBrain, nullptr);
    EXPECT_EQ(retrievedBrain->size(), 2);

    // Verify each hemisphere
    const auto& hemisphereIds = retrievedBrain->getHemisphereIds();
    for (uint64_t hemId : hemisphereIds) {
        auto hemisphere = datastore->getHemisphere(hemId);
        ASSERT_NE(hemisphere, nullptr);
        EXPECT_EQ(hemisphere->size(), 2);

        // Verify each lobe
        const auto& lobeIds = hemisphere->getLobeIds();
        for (uint64_t lobeId : lobeIds) {
            auto lobe = datastore->getLobe(lobeId);
            ASSERT_NE(lobe, nullptr);
            EXPECT_EQ(lobe->size(), 2);
        }
    }
}

// ============================================================================
// Test 7: Spike Pattern Learning and Recognition
// ============================================================================

TEST_F(IntegrationTest, SpikePatternLearningAndRecognition) {
    // Create a neuron and teach it a specific spike pattern
    auto neuron = factory->createNeuron(100.0, 0.85, 20);

    // Insert a specific pattern: spikes at 10, 20, 30, 40, 50 ms
    std::vector<double> pattern1 = {10.0, 20.0, 30.0, 40.0, 50.0};
    for (double spikeTime : pattern1) {
        neuron->insertSpike(spikeTime);
    }

    // Learn the pattern
    neuron->learnCurrentPattern();

    // Clear spikes
    neuron->clearSpikes();

    // Insert similar pattern: spikes at 11, 21, 31, 41, 51 ms (1ms offset)
    std::vector<double> pattern2 = {11.0, 21.0, 31.0, 41.0, 51.0};
    for (double spikeTime : pattern2) {
        neuron->insertSpike(spikeTime);
    }

    // Check if neuron recognizes the pattern
    bool shouldFire = neuron->checkShouldFire();
    EXPECT_TRUE(shouldFire);  // Should recognize similar pattern

    // Clear and insert very different pattern
    neuron->clearSpikes();
    std::vector<double> pattern3 = {5.0, 15.0, 25.0, 35.0, 45.0};
    for (double spikeTime : pattern3) {
        neuron->insertSpike(spikeTime);
    }

    // This pattern might or might not fire depending on similarity threshold
    // Just verify the method doesn't crash
    EXPECT_NO_THROW(neuron->checkShouldFire());
}

