#include <gtest/gtest.h>
#include "snnfw/ConnectivityPattern.h"
#include "snnfw/ConnectivityBuilder.h"
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
#include "snnfw/Logger.h"
#include <filesystem>
#include <spdlog/spdlog.h>

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

class ConnectivityTest : public ::testing::Test {
protected:
    std::unique_ptr<Datastore> datastore_;
    std::unique_ptr<NeuralObjectFactory> factory_;
    std::filesystem::path testDbPath_;
    
    void SetUp() override {
        // Set log level to error to reduce test output
        spdlog::set_level(spdlog::level::err);
        
        // Create temporary directory for test database
        testDbPath_ = std::filesystem::temp_directory_path() / "snnfw_test_connectivity";
        std::filesystem::remove_all(testDbPath_);
        std::filesystem::create_directories(testDbPath_);
        
        // Create datastore and factory
        datastore_ = std::make_unique<Datastore>(testDbPath_.string(), 10000);
        registerFactories(*datastore_);
        factory_ = std::make_unique<NeuralObjectFactory>();
    }
    
    void TearDown() override {
        datastore_.reset();
        std::filesystem::remove_all(testDbPath_);
    }
    
    // Helper function to create neurons
    std::vector<std::shared_ptr<Neuron>> createNeurons(size_t count) {
        std::vector<std::shared_ptr<Neuron>> neurons;
        for (size_t i = 0; i < count; ++i) {
            auto neuron = factory_->createNeuron(50.0, 0.95, 20);
            datastore_->put(neuron);
            neurons.push_back(neuron);
        }
        return neurons;
    }
};

// ============================================================================
// RandomSparsePattern Tests
// ============================================================================

TEST_F(ConnectivityTest, RandomSparsePattern_BasicConnectivity) {
    auto sourceNeurons = createNeurons(10);
    auto targetNeurons = createNeurons(10);
    
    std::vector<uint64_t> sourceIds, targetIds;
    for (const auto& n : sourceNeurons) sourceIds.push_back(n->getId());
    for (const auto& n : targetNeurons) targetIds.push_back(n->getId());
    
    RandomSparsePattern pattern(0.5, 1.0, 1.5);  // 50% connectivity
    pattern.setSeed(42);  // For reproducibility
    
    auto connections = pattern.generateConnections(sourceIds, targetIds);
    
    // With 10x10 neurons and 50% probability, expect around 50 connections
    // Allow some variance due to randomness
    EXPECT_GT(connections.size(), 30);
    EXPECT_LT(connections.size(), 70);
    
    // Verify connection properties
    for (const auto& conn : connections) {
        EXPECT_DOUBLE_EQ(conn.weight, 1.0);
        EXPECT_DOUBLE_EQ(conn.delay, 1.5);
    }
}

TEST_F(ConnectivityTest, RandomSparsePattern_ZeroProbability) {
    auto sourceNeurons = createNeurons(5);
    auto targetNeurons = createNeurons(5);
    
    std::vector<uint64_t> sourceIds, targetIds;
    for (const auto& n : sourceNeurons) sourceIds.push_back(n->getId());
    for (const auto& n : targetNeurons) targetIds.push_back(n->getId());
    
    RandomSparsePattern pattern(0.0, 1.0, 1.0);  // 0% connectivity
    auto connections = pattern.generateConnections(sourceIds, targetIds);
    
    EXPECT_EQ(connections.size(), 0);
}

TEST_F(ConnectivityTest, RandomSparsePattern_FullProbability) {
    auto sourceNeurons = createNeurons(5);
    auto targetNeurons = createNeurons(5);
    
    std::vector<uint64_t> sourceIds, targetIds;
    for (const auto& n : sourceNeurons) sourceIds.push_back(n->getId());
    for (const auto& n : targetNeurons) targetIds.push_back(n->getId());
    
    RandomSparsePattern pattern(1.0, 1.0, 1.0);  // 100% connectivity
    auto connections = pattern.generateConnections(sourceIds, targetIds);
    
    EXPECT_EQ(connections.size(), 25);  // 5x5 = 25
}

// ============================================================================
// AllToAllPattern Tests
// ============================================================================

TEST_F(ConnectivityTest, AllToAllPattern_BasicConnectivity) {
    auto sourceNeurons = createNeurons(3);
    auto targetNeurons = createNeurons(4);
    
    std::vector<uint64_t> sourceIds, targetIds;
    for (const auto& n : sourceNeurons) sourceIds.push_back(n->getId());
    for (const auto& n : targetNeurons) targetIds.push_back(n->getId());
    
    AllToAllPattern pattern(0.8, 2.0);
    auto connections = pattern.generateConnections(sourceIds, targetIds);
    
    EXPECT_EQ(connections.size(), 12);  // 3x4 = 12
    
    // Verify all connections have correct weight and delay
    for (const auto& conn : connections) {
        EXPECT_DOUBLE_EQ(conn.weight, 0.8);
        EXPECT_DOUBLE_EQ(conn.delay, 2.0);
    }
}

// ============================================================================
// OneToOnePattern Tests
// ============================================================================

TEST_F(ConnectivityTest, OneToOnePattern_EqualSizes) {
    auto sourceNeurons = createNeurons(5);
    auto targetNeurons = createNeurons(5);
    
    std::vector<uint64_t> sourceIds, targetIds;
    for (const auto& n : sourceNeurons) sourceIds.push_back(n->getId());
    for (const auto& n : targetNeurons) targetIds.push_back(n->getId());
    
    OneToOnePattern pattern(1.2, 1.0);
    auto connections = pattern.generateConnections(sourceIds, targetIds);
    
    EXPECT_EQ(connections.size(), 5);
    
    // Verify connections are one-to-one
    for (size_t i = 0; i < connections.size(); ++i) {
        EXPECT_EQ(connections[i].sourceNeuronId, sourceIds[i]);
        EXPECT_EQ(connections[i].targetNeuronId, targetIds[i]);
        EXPECT_DOUBLE_EQ(connections[i].weight, 1.2);
        EXPECT_DOUBLE_EQ(connections[i].delay, 1.0);
    }
}

TEST_F(ConnectivityTest, OneToOnePattern_UnequalSizes) {
    auto sourceNeurons = createNeurons(3);
    auto targetNeurons = createNeurons(5);
    
    std::vector<uint64_t> sourceIds, targetIds;
    for (const auto& n : sourceNeurons) sourceIds.push_back(n->getId());
    for (const auto& n : targetNeurons) targetIds.push_back(n->getId());
    
    OneToOnePattern pattern(1.0, 1.0);
    
    // Should throw exception for unequal sizes
    EXPECT_THROW(
        pattern.generateConnections(sourceIds, targetIds),
        std::invalid_argument
    );
}

// ============================================================================
// ManyToOnePattern Tests
// ============================================================================

TEST_F(ConnectivityTest, ManyToOnePattern_BasicConnectivity) {
    auto sourceNeurons = createNeurons(4);
    auto targetNeurons = createNeurons(2);
    
    std::vector<uint64_t> sourceIds, targetIds;
    for (const auto& n : sourceNeurons) sourceIds.push_back(n->getId());
    for (const auto& n : targetNeurons) targetIds.push_back(n->getId());
    
    ManyToOnePattern pattern(0.9, 1.5);
    auto connections = pattern.generateConnections(sourceIds, targetIds);
    
    EXPECT_EQ(connections.size(), 8);  // 4 sources * 2 targets = 8
    
    // Verify each target receives connections from all sources
    std::map<uint64_t, int> targetConnectionCount;
    for (const auto& conn : connections) {
        targetConnectionCount[conn.targetNeuronId]++;
        EXPECT_DOUBLE_EQ(conn.weight, 0.9);
        EXPECT_DOUBLE_EQ(conn.delay, 1.5);
    }
    
    for (const auto& targetId : targetIds) {
        EXPECT_EQ(targetConnectionCount[targetId], 4);  // Each target gets 4 connections
    }
}

// ============================================================================
// DistanceDependentPattern Tests
// ============================================================================

TEST_F(ConnectivityTest, DistanceDependentPattern_WithPositions) {
    auto sourceNeurons = createNeurons(3);
    auto targetNeurons = createNeurons(3);
    
    std::vector<uint64_t> sourceIds, targetIds;
    for (const auto& n : sourceNeurons) sourceIds.push_back(n->getId());
    for (const auto& n : targetNeurons) targetIds.push_back(n->getId());
    
    // Set up positions
    std::map<uint64_t, SpatialPosition> positions;
    positions[sourceIds[0]] = SpatialPosition(0.0, 0.0, 0.0);
    positions[sourceIds[1]] = SpatialPosition(1.0, 0.0, 0.0);
    positions[sourceIds[2]] = SpatialPosition(2.0, 0.0, 0.0);
    positions[targetIds[0]] = SpatialPosition(0.0, 0.0, 0.0);
    positions[targetIds[1]] = SpatialPosition(1.0, 0.0, 0.0);
    positions[targetIds[2]] = SpatialPosition(2.0, 0.0, 0.0);
    
    DistanceDependentPattern pattern(1.0, 1.0, 1.0);  // sigma=1.0
    pattern.setPositions(positions);
    pattern.setSeed(42);
    
    auto connections = pattern.generateConnections(sourceIds, targetIds);
    
    // Nearby neurons should have higher connection probability
    EXPECT_GT(connections.size(), 0);
}

TEST_F(ConnectivityTest, DistanceDependentPattern_NoPositions) {
    auto sourceNeurons = createNeurons(2);
    auto targetNeurons = createNeurons(2);
    
    std::vector<uint64_t> sourceIds, targetIds;
    for (const auto& n : sourceNeurons) sourceIds.push_back(n->getId());
    for (const auto& n : targetNeurons) targetIds.push_back(n->getId());
    
    DistanceDependentPattern pattern(1.0, 1.0, 1.0);
    
    // Should throw exception when positions not set
    EXPECT_THROW(
        pattern.generateConnections(sourceIds, targetIds),
        std::runtime_error
    );
}

// ============================================================================
// TopographicPattern Tests
// ============================================================================

TEST_F(ConnectivityTest, TopographicPattern_BasicConnectivity) {
    auto sourceNeurons = createNeurons(5);
    auto targetNeurons = createNeurons(5);
    
    std::vector<uint64_t> sourceIds, targetIds;
    for (const auto& n : sourceNeurons) sourceIds.push_back(n->getId());
    for (const auto& n : targetNeurons) targetIds.push_back(n->getId());
    
    TopographicPattern pattern(0.2, 1.0, 1.0);  // sigma=0.2 for tight topographic mapping
    pattern.setSeed(42);
    
    auto connections = pattern.generateConnections(sourceIds, targetIds);
    
    // Should create some connections
    EXPECT_GT(connections.size(), 0);
}

// ============================================================================
// SmallWorldPattern Tests
// ============================================================================

TEST_F(ConnectivityTest, SmallWorldPattern_WithPositions) {
    auto sourceNeurons = createNeurons(4);
    auto targetNeurons = createNeurons(4);
    
    std::vector<uint64_t> sourceIds, targetIds;
    for (const auto& n : sourceNeurons) sourceIds.push_back(n->getId());
    for (const auto& n : targetNeurons) targetIds.push_back(n->getId());
    
    // Set up positions in a line
    std::map<uint64_t, SpatialPosition> positions;
    for (size_t i = 0; i < sourceIds.size(); ++i) {
        positions[sourceIds[i]] = SpatialPosition(static_cast<double>(i), 0.0, 0.0);
    }
    for (size_t i = 0; i < targetIds.size(); ++i) {
        positions[targetIds[i]] = SpatialPosition(static_cast<double>(i), 0.0, 0.0);
    }
    
    SmallWorldPattern pattern(0.8, 0.1, 1.5, 1.0, 1.0);  // High local, low long-range
    pattern.setPositions(positions);
    pattern.setSeed(42);
    
    auto connections = pattern.generateConnections(sourceIds, targetIds);
    
    // Should create connections
    EXPECT_GT(connections.size(), 0);
}

TEST_F(ConnectivityTest, SmallWorldPattern_NoPositions) {
    auto sourceNeurons = createNeurons(2);
    auto targetNeurons = createNeurons(2);

    std::vector<uint64_t> sourceIds, targetIds;
    for (const auto& n : sourceNeurons) sourceIds.push_back(n->getId());
    for (const auto& n : targetNeurons) targetIds.push_back(n->getId());

    SmallWorldPattern pattern(0.5, 0.1, 1.0, 1.0, 1.0);

    // Should throw exception when positions not set
    EXPECT_THROW(
        pattern.generateConnections(sourceIds, targetIds),
        std::runtime_error
    );
}

// ============================================================================
// ConnectivityBuilder Tests
// ============================================================================

TEST_F(ConnectivityTest, ConnectivityBuilder_AllToAll) {
    auto sourceNeurons = createNeurons(3);
    auto targetNeurons = createNeurons(2);

    AllToAllPattern pattern(1.0, 1.5);
    ConnectivityBuilder builder(*factory_, *datastore_);

    auto stats = builder.connect(sourceNeurons, targetNeurons, pattern);

    EXPECT_EQ(stats.connectionsRequested, 6);  // 3x2 = 6
    EXPECT_EQ(stats.synapsesCreated, 6);
    EXPECT_EQ(stats.failedConnections, 0);

    // Verify synapses were created
    auto synapses = builder.getCreatedSynapses();
    EXPECT_EQ(synapses.size(), 6);

    for (const auto& synapse : synapses) {
        EXPECT_DOUBLE_EQ(synapse->getWeight(), 1.0);
        EXPECT_DOUBLE_EQ(synapse->getDelay(), 1.5);
    }
}

TEST_F(ConnectivityTest, ConnectivityBuilder_OneToOne) {
    auto sourceNeurons = createNeurons(4);
    auto targetNeurons = createNeurons(4);

    OneToOnePattern pattern(0.8, 2.0);
    ConnectivityBuilder builder(*factory_, *datastore_);

    auto stats = builder.connect(sourceNeurons, targetNeurons, pattern);

    EXPECT_EQ(stats.connectionsRequested, 4);
    EXPECT_EQ(stats.synapsesCreated, 4);
    EXPECT_EQ(stats.failedConnections, 0);

    // Verify axons and dendrites were created
    EXPECT_GT(builder.getCreatedAxons().size(), 0);
    EXPECT_EQ(builder.getCreatedDendrites().size(), 4);
}

TEST_F(ConnectivityTest, ConnectivityBuilder_RandomSparse) {
    auto sourceNeurons = createNeurons(10);
    auto targetNeurons = createNeurons(10);

    RandomSparsePattern pattern(0.3, 1.0, 1.0);  // 30% connectivity
    pattern.setSeed(42);

    ConnectivityBuilder builder(*factory_, *datastore_);
    auto stats = builder.connect(sourceNeurons, targetNeurons, pattern);

    // Should create approximately 30 connections (30% of 100)
    EXPECT_GT(stats.synapsesCreated, 15);
    EXPECT_LT(stats.synapsesCreated, 45);
    EXPECT_EQ(stats.failedConnections, 0);
}

TEST_F(ConnectivityTest, ConnectivityBuilder_AutoPersist) {
    auto sourceNeurons = createNeurons(2);
    auto targetNeurons = createNeurons(2);

    AllToAllPattern pattern(1.0, 1.0);
    ConnectivityBuilder builder(*factory_, *datastore_, true);  // Auto-persist enabled

    auto stats = builder.connect(sourceNeurons, targetNeurons, pattern);

    EXPECT_EQ(stats.synapsesCreated, 4);

    // Verify synapses are in datastore
    auto synapses = builder.getCreatedSynapses();
    for (const auto& synapse : synapses) {
        auto retrieved = datastore_->getSynapse(synapse->getId());
        ASSERT_NE(retrieved, nullptr);
        EXPECT_EQ(retrieved->getId(), synapse->getId());
    }
}

TEST_F(ConnectivityTest, ConnectivityBuilder_IDBasedConnect) {
    auto sourceNeurons = createNeurons(3);
    auto targetNeurons = createNeurons(3);

    std::vector<uint64_t> sourceIds, targetIds;
    for (const auto& n : sourceNeurons) sourceIds.push_back(n->getId());
    for (const auto& n : targetNeurons) targetIds.push_back(n->getId());

    OneToOnePattern pattern(1.0, 1.0);
    ConnectivityBuilder builder(*factory_, *datastore_);

    auto stats = builder.connect(sourceIds, targetIds, pattern);

    EXPECT_EQ(stats.connectionsRequested, 3);
    EXPECT_EQ(stats.synapsesCreated, 3);
    EXPECT_EQ(stats.failedConnections, 0);
}

TEST_F(ConnectivityTest, ConnectivityBuilder_ClearCreatedObjects) {
    auto sourceNeurons = createNeurons(2);
    auto targetNeurons = createNeurons(2);

    AllToAllPattern pattern(1.0, 1.0);
    ConnectivityBuilder builder(*factory_, *datastore_);

    builder.connect(sourceNeurons, targetNeurons, pattern);

    EXPECT_GT(builder.getCreatedSynapses().size(), 0);

    builder.clearCreatedObjects();

    EXPECT_EQ(builder.getCreatedSynapses().size(), 0);
    EXPECT_EQ(builder.getCreatedAxons().size(), 0);
    EXPECT_EQ(builder.getCreatedDendrites().size(), 0);
}

// ============================================================================
// SpatialPosition Tests
// ============================================================================

TEST_F(ConnectivityTest, SpatialPosition_Distance) {
    SpatialPosition p1(0.0, 0.0, 0.0);
    SpatialPosition p2(3.0, 4.0, 0.0);

    double distance = p1.distanceTo(p2);
    EXPECT_DOUBLE_EQ(distance, 5.0);  // 3-4-5 triangle
}

TEST_F(ConnectivityTest, SpatialPosition_Distance3D) {
    SpatialPosition p1(0.0, 0.0, 0.0);
    SpatialPosition p2(1.0, 1.0, 1.0);

    double distance = p1.distanceTo(p2);
    EXPECT_NEAR(distance, std::sqrt(3.0), 1e-10);
}


