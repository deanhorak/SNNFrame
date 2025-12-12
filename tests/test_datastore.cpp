#include <gtest/gtest.h>
#include "snnfw/Datastore.h"
#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Neuron.h"
#include "snnfw/Axon.h"
#include "snnfw/Dendrite.h"
#include "snnfw/Synapse.h"
#include "snnfw/Cluster.h"
#include <filesystem>
#include <thread>
#include <chrono>

using namespace snnfw;

// Test fixture for Datastore tests
class DatastoreTest : public ::testing::Test {
protected:
    std::string dbPath;

    void SetUp() override {
        // Create a unique temporary database path for each test
        dbPath = "./test_db_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count());
    }

    void TearDown() override {
        // Clean up the test database
        try {
            std::filesystem::remove_all(dbPath);
        } catch (...) {
            // Ignore cleanup errors
        }
    }

    // Helper function to register all factories
    void registerFactories(Datastore& datastore) {
        // Register Neuron factory
        datastore.registerFactory("Neuron", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
            auto neuron = std::make_shared<Neuron>(0, 0, 0);  // Dummy values
            if (neuron->fromJson(json)) {
                return neuron;
            }
            return nullptr;
        });

        // Register Axon factory
        datastore.registerFactory("Axon", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
            auto axon = std::make_shared<Axon>(0);  // Dummy value
            if (axon->fromJson(json)) {
                return axon;
            }
            return nullptr;
        });

        // Register Dendrite factory
        datastore.registerFactory("Dendrite", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
            auto dendrite = std::make_shared<Dendrite>(0);  // Dummy value
            if (dendrite->fromJson(json)) {
                return dendrite;
            }
            return nullptr;
        });

        // Register Synapse factory
        datastore.registerFactory("Synapse", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
            auto synapse = std::make_shared<Synapse>(0, 0, 0.0, 0.0);  // Dummy values
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
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(DatastoreTest, ConstructorCreatesDatabase) {
    EXPECT_NO_THROW({
        Datastore datastore(dbPath, 1000);
    });
    
    // Verify database directory was created
    EXPECT_TRUE(std::filesystem::exists(dbPath));
}

TEST_F(DatastoreTest, PutAndGetNeuron) {
    Datastore datastore(dbPath, 1000);
    NeuralObjectFactory factory;
    
    auto neuron = factory.createNeuron(50.0, 0.95, 20);
    uint64_t neuronId = neuron->getId();
    
    // Store neuron
    EXPECT_TRUE(datastore.put(neuron));
    
    // Retrieve neuron
    auto retrieved = datastore.getNeuron(neuronId);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->getId(), neuronId);
}

TEST_F(DatastoreTest, PutAndGetAxon) {
    Datastore datastore(dbPath, 1000);
    NeuralObjectFactory factory;
    
    auto axon = factory.createAxon(100000000000001ULL);
    uint64_t axonId = axon->getId();
    
    EXPECT_TRUE(datastore.put(axon));
    
    auto retrieved = datastore.getAxon(axonId);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->getId(), axonId);
    EXPECT_EQ(retrieved->getSourceNeuronId(), 100000000000001ULL);
}

TEST_F(DatastoreTest, PutAndGetDendrite) {
    Datastore datastore(dbPath, 1000);
    NeuralObjectFactory factory;
    
    auto dendrite = factory.createDendrite(100000000000002ULL);
    uint64_t dendriteId = dendrite->getId();
    
    EXPECT_TRUE(datastore.put(dendrite));
    
    auto retrieved = datastore.getDendrite(dendriteId);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->getId(), dendriteId);
    EXPECT_EQ(retrieved->getTargetNeuronId(), 100000000000002ULL);
}

TEST_F(DatastoreTest, PutAndGetSynapse) {
    Datastore datastore(dbPath, 1000);
    NeuralObjectFactory factory;
    
    auto synapse = factory.createSynapse(200000000000001ULL, 300000000000001ULL, 0.8, 1.5);
    uint64_t synapseId = synapse->getId();
    
    EXPECT_TRUE(datastore.put(synapse));
    
    auto retrieved = datastore.getSynapse(synapseId);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->getId(), synapseId);
    EXPECT_EQ(retrieved->getAxonId(), 200000000000001ULL);
    EXPECT_EQ(retrieved->getDendriteId(), 300000000000001ULL);
    EXPECT_DOUBLE_EQ(retrieved->getWeight(), 0.8);
    EXPECT_DOUBLE_EQ(retrieved->getDelay(), 1.5);
}

TEST_F(DatastoreTest, PutAndGetCluster) {
    Datastore datastore(dbPath, 1000);
    NeuralObjectFactory factory;
    
    auto cluster = factory.createCluster();
    cluster->addNeuron(100000000000001ULL);
    cluster->addNeuron(100000000000002ULL);
    uint64_t clusterId = cluster->getId();
    
    EXPECT_TRUE(datastore.put(cluster));
    
    auto retrieved = datastore.getCluster(clusterId);
    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(retrieved->getId(), clusterId);
    EXPECT_EQ(retrieved->size(), 2);
}

// ============================================================================
// Cache Tests
// ============================================================================

TEST_F(DatastoreTest, CacheHitAfterPut) {
    Datastore datastore(dbPath, 1000);
    NeuralObjectFactory factory;
    
    auto neuron = factory.createNeuron(50.0, 0.95, 20);
    uint64_t neuronId = neuron->getId();
    
    datastore.put(neuron);
    
    // First retrieval should be a cache hit
    uint64_t hits1, misses1;
    datastore.getCacheStats(hits1, misses1);
    
    auto retrieved = datastore.getNeuron(neuronId);
    
    uint64_t hits2, misses2;
    datastore.getCacheStats(hits2, misses2);
    
    EXPECT_EQ(hits2, hits1 + 1);
    EXPECT_EQ(misses2, misses1);
}

TEST_F(DatastoreTest, CacheMissFromDisk) {
    uint64_t neuronId;

    // Create datastore, add neuron, let it flush to disk
    {
        Datastore datastore(dbPath, 1000);
        registerFactories(datastore);
        NeuralObjectFactory factory;

        auto neuron = factory.createNeuron(50.0, 0.95, 20);
        neuronId = neuron->getId();
        datastore.put(neuron);
        datastore.flush(neuronId);
    }

    // Create new datastore (empty cache)
    Datastore datastore(dbPath, 1000);
    registerFactories(datastore);

    uint64_t hits1, misses1;
    datastore.getCacheStats(hits1, misses1);

    // This should be a cache miss
    auto retrieved = datastore.getNeuron(neuronId);

    uint64_t hits2, misses2;
    datastore.getCacheStats(hits2, misses2);

    ASSERT_NE(retrieved, nullptr);
    EXPECT_EQ(hits2, hits1);
    EXPECT_EQ(misses2, misses1 + 1);
}

TEST_F(DatastoreTest, CacheSizeTracking) {
    Datastore datastore(dbPath, 100);
    NeuralObjectFactory factory;
    
    EXPECT_EQ(datastore.getCacheSize(), 0);
    
    for (int i = 0; i < 10; i++) {
        auto neuron = factory.createNeuron(50.0, 0.95, 20);
        datastore.put(neuron);
    }
    
    EXPECT_EQ(datastore.getCacheSize(), 10);
}

// ============================================================================
// Dirty Tracking and Persistence Tests
// ============================================================================

TEST_F(DatastoreTest, MarkDirtyAndFlush) {
    Datastore datastore(dbPath, 1000);
    NeuralObjectFactory factory;
    
    auto neuron = factory.createNeuron(50.0, 0.95, 20);
    uint64_t neuronId = neuron->getId();
    
    datastore.put(neuron);
    
    // Modify neuron
    neuron->insertSpike(10.0);
    datastore.markDirty(neuronId);
    
    // Flush to disk
    EXPECT_TRUE(datastore.flush(neuronId));
}

TEST_F(DatastoreTest, FlushAllDirtyObjects) {
    Datastore datastore(dbPath, 1000);
    NeuralObjectFactory factory;
    
    // Create and modify multiple neurons
    for (int i = 0; i < 5; i++) {
        auto neuron = factory.createNeuron(50.0, 0.95, 20);
        datastore.put(neuron);
        neuron->insertSpike(10.0 * i);
        datastore.markDirty(neuron->getId());
    }
    
    // Flush all
    size_t flushed = datastore.flushAll();
    EXPECT_EQ(flushed, 5);
}

TEST_F(DatastoreTest, PersistenceAcrossInstances) {
    uint64_t neuronId;

    // Create and store neuron
    {
        Datastore datastore(dbPath, 1000);
        registerFactories(datastore);
        NeuralObjectFactory factory;

        auto neuron = factory.createNeuron(50.0, 0.95, 20);
        neuron->insertSpike(10.0);
        neuron->insertSpike(20.0);
        neuronId = neuron->getId();

        datastore.put(neuron);
        datastore.markDirty(neuronId);
        // Datastore destructor will flush
    }

    // Retrieve from new datastore instance
    {
        Datastore datastore(dbPath, 1000);
        registerFactories(datastore);
        auto retrieved = datastore.getNeuron(neuronId);

        ASSERT_NE(retrieved, nullptr);
        EXPECT_EQ(retrieved->getId(), neuronId);
        // Note: Spike data should be persisted via JSON
    }
}

// ============================================================================
// LRU Eviction Tests
// ============================================================================

TEST_F(DatastoreTest, LRUEvictionWhenCacheFull) {
    Datastore datastore(dbPath, 10);  // Small cache
    NeuralObjectFactory factory;
    
    // Fill cache
    for (int i = 0; i < 10; i++) {
        auto neuron = factory.createNeuron(50.0, 0.95, 20);
        datastore.put(neuron);
    }
    
    EXPECT_EQ(datastore.getCacheSize(), 10);
    
    // Add one more - should trigger eviction
    auto neuron = factory.createNeuron(50.0, 0.95, 20);
    datastore.put(neuron);
    
    EXPECT_EQ(datastore.getCacheSize(), 10);  // Still at max
}

// ============================================================================
// Remove Tests
// ============================================================================

TEST_F(DatastoreTest, RemoveObject) {
    Datastore datastore(dbPath, 1000);
    NeuralObjectFactory factory;
    
    auto neuron = factory.createNeuron(50.0, 0.95, 20);
    uint64_t neuronId = neuron->getId();
    
    datastore.put(neuron);
    EXPECT_TRUE(datastore.remove(neuronId));
    
    // Should not be retrievable
    auto retrieved = datastore.getNeuron(neuronId);
    EXPECT_EQ(retrieved, nullptr);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

TEST_F(DatastoreTest, GetNonExistentObject) {
    Datastore datastore(dbPath, 1000);
    
    auto neuron = datastore.getNeuron(999999999999999ULL);
    EXPECT_EQ(neuron, nullptr);
}

TEST_F(DatastoreTest, PutNullObject) {
    Datastore datastore(dbPath, 1000);
    
    EXPECT_FALSE(datastore.put(nullptr));
}

// Run all tests
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

