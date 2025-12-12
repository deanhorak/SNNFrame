#include <gtest/gtest.h>
#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Logger.h"
#include <thread>
#include <vector>
#include <set>

using namespace snnfw;

class NeuralObjectFactoryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize logger with warning level to reduce test output
        Logger::getInstance().initialize("test_factory.log", spdlog::level::warn);
    }
};

// ============================================================================
// Basic Creation Tests
// ============================================================================

TEST_F(NeuralObjectFactoryTest, CreateNeuron) {
    NeuralObjectFactory factory;
    
    auto neuron = factory.createNeuron(50.0, 0.95, 20);
    
    ASSERT_NE(neuron, nullptr);
    EXPECT_GE(neuron->getId(), NeuralObjectFactory::NEURON_ID_START);
    EXPECT_LE(neuron->getId(), NeuralObjectFactory::NEURON_ID_END);
}

TEST_F(NeuralObjectFactoryTest, CreateAxon) {
    NeuralObjectFactory factory;
    
    auto neuron = factory.createNeuron(50.0, 0.95, 20);
    auto axon = factory.createAxon(neuron->getId());
    
    ASSERT_NE(axon, nullptr);
    EXPECT_GE(axon->getId(), NeuralObjectFactory::AXON_ID_START);
    EXPECT_LE(axon->getId(), NeuralObjectFactory::AXON_ID_END);
    EXPECT_EQ(axon->getSourceNeuronId(), neuron->getId());
}

TEST_F(NeuralObjectFactoryTest, CreateDendrite) {
    NeuralObjectFactory factory;
    
    auto neuron = factory.createNeuron(50.0, 0.95, 20);
    auto dendrite = factory.createDendrite(neuron->getId());
    
    ASSERT_NE(dendrite, nullptr);
    EXPECT_GE(dendrite->getId(), NeuralObjectFactory::DENDRITE_ID_START);
    EXPECT_LE(dendrite->getId(), NeuralObjectFactory::DENDRITE_ID_END);
    EXPECT_EQ(dendrite->getTargetNeuronId(), neuron->getId());
}

TEST_F(NeuralObjectFactoryTest, CreateSynapse) {
    NeuralObjectFactory factory;
    
    auto neuron1 = factory.createNeuron(50.0, 0.95, 20);
    auto neuron2 = factory.createNeuron(50.0, 0.95, 20);
    auto axon = factory.createAxon(neuron1->getId());
    auto dendrite = factory.createDendrite(neuron2->getId());
    
    auto synapse = factory.createSynapse(axon->getId(), dendrite->getId(), 1.5, 2.0);
    
    ASSERT_NE(synapse, nullptr);
    EXPECT_GE(synapse->getId(), NeuralObjectFactory::SYNAPSE_ID_START);
    EXPECT_LE(synapse->getId(), NeuralObjectFactory::SYNAPSE_ID_END);
    EXPECT_EQ(synapse->getAxonId(), axon->getId());
    EXPECT_EQ(synapse->getDendriteId(), dendrite->getId());
    EXPECT_DOUBLE_EQ(synapse->getWeight(), 1.5);
    EXPECT_DOUBLE_EQ(synapse->getDelay(), 2.0);
}

TEST_F(NeuralObjectFactoryTest, CreateCluster) {
    NeuralObjectFactory factory;
    
    auto cluster = factory.createCluster();
    
    ASSERT_NE(cluster, nullptr);
    EXPECT_GE(cluster->getId(), NeuralObjectFactory::CLUSTER_ID_START);
    EXPECT_LE(cluster->getId(), NeuralObjectFactory::CLUSTER_ID_END);
}

// ============================================================================
// ID Uniqueness Tests
// ============================================================================

TEST_F(NeuralObjectFactoryTest, NeuronIdsAreUnique) {
    NeuralObjectFactory factory;
    
    auto neuron1 = factory.createNeuron(50.0, 0.95, 20);
    auto neuron2 = factory.createNeuron(50.0, 0.95, 20);
    auto neuron3 = factory.createNeuron(50.0, 0.95, 20);
    
    EXPECT_NE(neuron1->getId(), neuron2->getId());
    EXPECT_NE(neuron2->getId(), neuron3->getId());
    EXPECT_NE(neuron1->getId(), neuron3->getId());
}

TEST_F(NeuralObjectFactoryTest, AxonIdsAreUnique) {
    NeuralObjectFactory factory;
    
    auto neuron = factory.createNeuron(50.0, 0.95, 20);
    auto axon1 = factory.createAxon(neuron->getId());
    auto axon2 = factory.createAxon(neuron->getId());
    auto axon3 = factory.createAxon(neuron->getId());
    
    EXPECT_NE(axon1->getId(), axon2->getId());
    EXPECT_NE(axon2->getId(), axon3->getId());
    EXPECT_NE(axon1->getId(), axon3->getId());
}

TEST_F(NeuralObjectFactoryTest, AllObjectTypesHaveUniqueIds) {
    NeuralObjectFactory factory;
    
    auto neuron = factory.createNeuron(50.0, 0.95, 20);
    auto axon = factory.createAxon(neuron->getId());
    auto dendrite = factory.createDendrite(neuron->getId());
    auto synapse = factory.createSynapse(axon->getId(), dendrite->getId());
    auto cluster = factory.createCluster();
    
    std::set<uint64_t> ids;
    ids.insert(neuron->getId());
    ids.insert(axon->getId());
    ids.insert(dendrite->getId());
    ids.insert(synapse->getId());
    ids.insert(cluster->getId());
    
    // All IDs should be unique
    EXPECT_EQ(ids.size(), 5);
}

// ============================================================================
// ID Range Tests
// ============================================================================

TEST_F(NeuralObjectFactoryTest, NeuronIdsInCorrectRange) {
    NeuralObjectFactory factory;
    
    for (int i = 0; i < 10; ++i) {
        auto neuron = factory.createNeuron(50.0, 0.95, 20);
        EXPECT_GE(neuron->getId(), NeuralObjectFactory::NEURON_ID_START);
        EXPECT_LE(neuron->getId(), NeuralObjectFactory::NEURON_ID_END);
    }
}

TEST_F(NeuralObjectFactoryTest, AxonIdsInCorrectRange) {
    NeuralObjectFactory factory;
    auto neuron = factory.createNeuron(50.0, 0.95, 20);
    
    for (int i = 0; i < 10; ++i) {
        auto axon = factory.createAxon(neuron->getId());
        EXPECT_GE(axon->getId(), NeuralObjectFactory::AXON_ID_START);
        EXPECT_LE(axon->getId(), NeuralObjectFactory::AXON_ID_END);
    }
}

TEST_F(NeuralObjectFactoryTest, DendriteIdsInCorrectRange) {
    NeuralObjectFactory factory;
    auto neuron = factory.createNeuron(50.0, 0.95, 20);
    
    for (int i = 0; i < 10; ++i) {
        auto dendrite = factory.createDendrite(neuron->getId());
        EXPECT_GE(dendrite->getId(), NeuralObjectFactory::DENDRITE_ID_START);
        EXPECT_LE(dendrite->getId(), NeuralObjectFactory::DENDRITE_ID_END);
    }
}

TEST_F(NeuralObjectFactoryTest, SynapseIdsInCorrectRange) {
    NeuralObjectFactory factory;
    auto neuron = factory.createNeuron(50.0, 0.95, 20);
    auto axon = factory.createAxon(neuron->getId());
    auto dendrite = factory.createDendrite(neuron->getId());
    
    for (int i = 0; i < 10; ++i) {
        auto synapse = factory.createSynapse(axon->getId(), dendrite->getId());
        EXPECT_GE(synapse->getId(), NeuralObjectFactory::SYNAPSE_ID_START);
        EXPECT_LE(synapse->getId(), NeuralObjectFactory::SYNAPSE_ID_END);
    }
}

TEST_F(NeuralObjectFactoryTest, ClusterIdsInCorrectRange) {
    NeuralObjectFactory factory;
    
    for (int i = 0; i < 10; ++i) {
        auto cluster = factory.createCluster();
        EXPECT_GE(cluster->getId(), NeuralObjectFactory::CLUSTER_ID_START);
        EXPECT_LE(cluster->getId(), NeuralObjectFactory::CLUSTER_ID_END);
    }
}

// ============================================================================
// Object Type Identification Tests
// ============================================================================

TEST_F(NeuralObjectFactoryTest, GetObjectTypeFromId) {
    NeuralObjectFactory factory;
    
    auto neuron = factory.createNeuron(50.0, 0.95, 20);
    auto axon = factory.createAxon(neuron->getId());
    auto dendrite = factory.createDendrite(neuron->getId());
    auto synapse = factory.createSynapse(axon->getId(), dendrite->getId());
    auto cluster = factory.createCluster();
    
    EXPECT_EQ(NeuralObjectFactory::getObjectType(neuron->getId()), 
              NeuralObjectFactory::ObjectType::NEURON);
    EXPECT_EQ(NeuralObjectFactory::getObjectType(axon->getId()), 
              NeuralObjectFactory::ObjectType::AXON);
    EXPECT_EQ(NeuralObjectFactory::getObjectType(dendrite->getId()), 
              NeuralObjectFactory::ObjectType::DENDRITE);
    EXPECT_EQ(NeuralObjectFactory::getObjectType(synapse->getId()), 
              NeuralObjectFactory::ObjectType::SYNAPSE);
    EXPECT_EQ(NeuralObjectFactory::getObjectType(cluster->getId()), 
              NeuralObjectFactory::ObjectType::CLUSTER);
}

TEST_F(NeuralObjectFactoryTest, GetObjectTypeName) {
    EXPECT_STREQ(NeuralObjectFactory::getObjectTypeName(
        NeuralObjectFactory::ObjectType::NEURON), "Neuron");
    EXPECT_STREQ(NeuralObjectFactory::getObjectTypeName(
        NeuralObjectFactory::ObjectType::AXON), "Axon");
    EXPECT_STREQ(NeuralObjectFactory::getObjectTypeName(
        NeuralObjectFactory::ObjectType::DENDRITE), "Dendrite");
    EXPECT_STREQ(NeuralObjectFactory::getObjectTypeName(
        NeuralObjectFactory::ObjectType::SYNAPSE), "Synapse");
    EXPECT_STREQ(NeuralObjectFactory::getObjectTypeName(
        NeuralObjectFactory::ObjectType::CLUSTER), "Cluster");
    EXPECT_STREQ(NeuralObjectFactory::getObjectTypeName(
        NeuralObjectFactory::ObjectType::UNKNOWN), "Unknown");
}

TEST_F(NeuralObjectFactoryTest, GetObjectTypeNameFromId) {
    NeuralObjectFactory factory;
    
    auto neuron = factory.createNeuron(50.0, 0.95, 20);
    auto axon = factory.createAxon(neuron->getId());
    
    EXPECT_STREQ(NeuralObjectFactory::getObjectTypeName(neuron->getId()), "Neuron");
    EXPECT_STREQ(NeuralObjectFactory::getObjectTypeName(axon->getId()), "Axon");
}

TEST_F(NeuralObjectFactoryTest, UnknownIdReturnsUnknown) {
    uint64_t invalidId = 999;
    EXPECT_EQ(NeuralObjectFactory::getObjectType(invalidId), 
              NeuralObjectFactory::ObjectType::UNKNOWN);
    EXPECT_STREQ(NeuralObjectFactory::getObjectTypeName(invalidId), "Unknown");
}

// ============================================================================
// Counter and Statistics Tests
// ============================================================================

TEST_F(NeuralObjectFactoryTest, GetCurrentId) {
    NeuralObjectFactory factory;
    
    uint64_t initialNeuronId = factory.getCurrentId(NeuralObjectFactory::ObjectType::NEURON);
    EXPECT_EQ(initialNeuronId, NeuralObjectFactory::NEURON_ID_START);
    
    factory.createNeuron(50.0, 0.95, 20);
    
    uint64_t afterNeuronId = factory.getCurrentId(NeuralObjectFactory::ObjectType::NEURON);
    EXPECT_EQ(afterNeuronId, NeuralObjectFactory::NEURON_ID_START + 1);
}

TEST_F(NeuralObjectFactoryTest, GetObjectCount) {
    NeuralObjectFactory factory;
    
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::NEURON), 0);
    
    factory.createNeuron(50.0, 0.95, 20);
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::NEURON), 1);
    
    factory.createNeuron(50.0, 0.95, 20);
    factory.createNeuron(50.0, 0.95, 20);
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::NEURON), 3);
}

// ============================================================================
// Reset Tests
// ============================================================================

TEST_F(NeuralObjectFactoryTest, ResetCounters) {
    NeuralObjectFactory factory;

    // Create some objects
    factory.createNeuron(50.0, 0.95, 20);
    factory.createNeuron(50.0, 0.95, 20);
    auto neuron = factory.createNeuron(50.0, 0.95, 20);

    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::NEURON), 3);
    EXPECT_EQ(neuron->getId(), NeuralObjectFactory::NEURON_ID_START + 2);

    // Reset
    factory.reset();

    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::NEURON), 0);
    EXPECT_EQ(factory.getCurrentId(NeuralObjectFactory::ObjectType::NEURON),
              NeuralObjectFactory::NEURON_ID_START);

    // Create new object after reset
    auto newNeuron = factory.createNeuron(50.0, 0.95, 20);
    EXPECT_EQ(newNeuron->getId(), NeuralObjectFactory::NEURON_ID_START);
}

TEST_F(NeuralObjectFactoryTest, ResetAllTypes) {
    NeuralObjectFactory factory;

    // Create objects of all types
    auto neuron = factory.createNeuron(50.0, 0.95, 20);
    auto axon = factory.createAxon(neuron->getId());
    auto dendrite = factory.createDendrite(neuron->getId());
    auto synapse = factory.createSynapse(axon->getId(), dendrite->getId());
    auto cluster = factory.createCluster();

    // Reset
    factory.reset();

    // Verify all counters reset
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::NEURON), 0);
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::AXON), 0);
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::DENDRITE), 0);
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::SYNAPSE), 0);
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::CLUSTER), 0);
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(NeuralObjectFactoryTest, ThreadSafeNeuronCreation) {
    NeuralObjectFactory factory;
    const int numThreads = 10;
    const int neuronsPerThread = 100;

    std::vector<std::thread> threads;
    std::vector<std::vector<uint64_t>> threadIds(numThreads);

    // Create neurons from multiple threads
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&factory, &threadIds, t, neuronsPerThread]() {
            for (int i = 0; i < neuronsPerThread; ++i) {
                auto neuron = factory.createNeuron(50.0, 0.95, 20);
                threadIds[t].push_back(neuron->getId());
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Collect all IDs
    std::set<uint64_t> allIds;
    for (const auto& ids : threadIds) {
        for (uint64_t id : ids) {
            allIds.insert(id);
        }
    }

    // All IDs should be unique
    EXPECT_EQ(allIds.size(), numThreads * neuronsPerThread);
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::NEURON),
              numThreads * neuronsPerThread);
}

TEST_F(NeuralObjectFactoryTest, ThreadSafeMixedCreation) {
    NeuralObjectFactory factory;
    const int numThreads = 4;
    const int objectsPerThread = 50;

    std::vector<std::thread> threads;

    // Create mixed objects from multiple threads
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([&factory, objectsPerThread]() {
            for (int i = 0; i < objectsPerThread; ++i) {
                auto neuron = factory.createNeuron(50.0, 0.95, 20);
                auto axon = factory.createAxon(neuron->getId());
                auto dendrite = factory.createDendrite(neuron->getId());
                auto synapse = factory.createSynapse(axon->getId(), dendrite->getId());
                auto cluster = factory.createCluster();
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    // Verify counts
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::NEURON),
              numThreads * objectsPerThread);
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::AXON),
              numThreads * objectsPerThread);
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::DENDRITE),
              numThreads * objectsPerThread);
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::SYNAPSE),
              numThreads * objectsPerThread);
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::CLUSTER),
              numThreads * objectsPerThread);
}

// ============================================================================
// Sequential ID Tests
// ============================================================================

TEST_F(NeuralObjectFactoryTest, NeuronIdsAreSequential) {
    NeuralObjectFactory factory;

    auto neuron1 = factory.createNeuron(50.0, 0.95, 20);
    auto neuron2 = factory.createNeuron(50.0, 0.95, 20);
    auto neuron3 = factory.createNeuron(50.0, 0.95, 20);

    EXPECT_EQ(neuron1->getId(), NeuralObjectFactory::NEURON_ID_START);
    EXPECT_EQ(neuron2->getId(), NeuralObjectFactory::NEURON_ID_START + 1);
    EXPECT_EQ(neuron3->getId(), NeuralObjectFactory::NEURON_ID_START + 2);
}

TEST_F(NeuralObjectFactoryTest, AllTypesHaveSequentialIds) {
    NeuralObjectFactory factory;

    auto neuron = factory.createNeuron(50.0, 0.95, 20);

    auto axon1 = factory.createAxon(neuron->getId());
    auto axon2 = factory.createAxon(neuron->getId());
    EXPECT_EQ(axon2->getId(), axon1->getId() + 1);

    auto dendrite1 = factory.createDendrite(neuron->getId());
    auto dendrite2 = factory.createDendrite(neuron->getId());
    EXPECT_EQ(dendrite2->getId(), dendrite1->getId() + 1);

    auto synapse1 = factory.createSynapse(axon1->getId(), dendrite1->getId());
    auto synapse2 = factory.createSynapse(axon2->getId(), dendrite2->getId());
    EXPECT_EQ(synapse2->getId(), synapse1->getId() + 1);

    auto cluster1 = factory.createCluster();
    auto cluster2 = factory.createCluster();
    EXPECT_EQ(cluster2->getId(), cluster1->getId() + 1);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(NeuralObjectFactoryTest, CreateCompleteNeuralConnection) {
    NeuralObjectFactory factory;

    // Create two neurons
    auto presynapticNeuron = factory.createNeuron(50.0, 0.95, 20);
    auto postsynapticNeuron = factory.createNeuron(50.0, 0.95, 20);

    // Create axon for presynaptic neuron
    auto axon = factory.createAxon(presynapticNeuron->getId());

    // Create dendrite for postsynaptic neuron
    auto dendrite = factory.createDendrite(postsynapticNeuron->getId());

    // Create synapse connecting them
    auto synapse = factory.createSynapse(axon->getId(), dendrite->getId(), 1.0, 1.5);

    // Verify the connection
    EXPECT_EQ(axon->getSourceNeuronId(), presynapticNeuron->getId());
    EXPECT_EQ(dendrite->getTargetNeuronId(), postsynapticNeuron->getId());
    EXPECT_EQ(synapse->getAxonId(), axon->getId());
    EXPECT_EQ(synapse->getDendriteId(), dendrite->getId());

    // Verify all objects have correct types
    EXPECT_EQ(NeuralObjectFactory::getObjectType(presynapticNeuron->getId()),
              NeuralObjectFactory::ObjectType::NEURON);
    EXPECT_EQ(NeuralObjectFactory::getObjectType(axon->getId()),
              NeuralObjectFactory::ObjectType::AXON);
    EXPECT_EQ(NeuralObjectFactory::getObjectType(dendrite->getId()),
              NeuralObjectFactory::ObjectType::DENDRITE);
    EXPECT_EQ(NeuralObjectFactory::getObjectType(synapse->getId()),
              NeuralObjectFactory::ObjectType::SYNAPSE);
}

TEST_F(NeuralObjectFactoryTest, CreateNeuralNetwork) {
    NeuralObjectFactory factory;

    // Create a small network: 3 neurons, fully connected
    std::vector<std::shared_ptr<Neuron>> neurons;
    for (int i = 0; i < 3; ++i) {
        neurons.push_back(factory.createNeuron(50.0, 0.95, 20));
    }

    // Create axons for each neuron
    std::vector<std::shared_ptr<Axon>> axons;
    for (const auto& neuron : neurons) {
        axons.push_back(factory.createAxon(neuron->getId()));
    }

    // Create dendrites for each neuron
    std::vector<std::shared_ptr<Dendrite>> dendrites;
    for (const auto& neuron : neurons) {
        dendrites.push_back(factory.createDendrite(neuron->getId()));
    }

    // Create synapses (fully connected)
    std::vector<std::shared_ptr<Synapse>> synapses;
    for (const auto& axon : axons) {
        for (const auto& dendrite : dendrites) {
            synapses.push_back(factory.createSynapse(axon->getId(), dendrite->getId()));
        }
    }

    // Verify counts
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::NEURON), 3);
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::AXON), 3);
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::DENDRITE), 3);
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::SYNAPSE), 9); // 3x3
}


