#include <gtest/gtest.h>
#include "snnfw/Cluster.h"
#include "snnfw/Neuron.h"
#include <memory>

class ClusterTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Test neuron IDs
        neuronId1 = 100000000000001;
        neuronId2 = 100000000000002;
        neuronId3 = 100000000000003;
    }

    void TearDown() override {
        // Clean up
    }

    uint64_t neuronId1;
    uint64_t neuronId2;
    uint64_t neuronId3;
};

// Test: Constructor with ID
TEST_F(ClusterTest, ConstructorWithId) {
    snnfw::Cluster cluster(42);
    EXPECT_EQ(cluster.getId(), 42);
    EXPECT_EQ(cluster.size(), 0);
}

// Test: Default constructor
TEST_F(ClusterTest, DefaultConstructor) {
    snnfw::Cluster cluster;
    EXPECT_EQ(cluster.getId(), 0);
    EXPECT_EQ(cluster.size(), 0);
}

// Test: Add single neuron by ID
TEST_F(ClusterTest, AddSingleNeuron) {
    snnfw::Cluster cluster(100);
    cluster.addNeuron(neuronId1);

    EXPECT_EQ(cluster.size(), 1);
    EXPECT_EQ(cluster.getNeuronId(0), neuronId1);
}

// Test: Add multiple neurons by ID
TEST_F(ClusterTest, AddMultipleNeurons) {
    snnfw::Cluster cluster(100);
    cluster.addNeuron(neuronId1);
    cluster.addNeuron(neuronId2);
    cluster.addNeuron(neuronId3);

    EXPECT_EQ(cluster.size(), 3);
    EXPECT_EQ(cluster.getNeuronId(0), neuronId1);
    EXPECT_EQ(cluster.getNeuronId(1), neuronId2);
    EXPECT_EQ(cluster.getNeuronId(2), neuronId3);
}

// Test: Add zero ID (should be added - no validation)
TEST_F(ClusterTest, AddZeroId) {
    snnfw::Cluster cluster(100);
    cluster.addNeuron(0);

    EXPECT_EQ(cluster.size(), 1);
    EXPECT_EQ(cluster.getNeuronId(0), 0);
}

// Test: Get neuron ID by valid index
TEST_F(ClusterTest, GetNeuronIdValidIndex) {
    snnfw::Cluster cluster(100);
    cluster.addNeuron(neuronId1);
    cluster.addNeuron(neuronId2);

    uint64_t retrieved = cluster.getNeuronId(1);
    EXPECT_EQ(retrieved, neuronId2);
}

// Test: Get neuron ID by invalid index
TEST_F(ClusterTest, GetNeuronIdInvalidIndex) {
    snnfw::Cluster cluster(100);
    cluster.addNeuron(neuronId1);

    uint64_t retrieved = cluster.getNeuronId(5);
    EXPECT_EQ(retrieved, 0);
}

// Test: Remove neuron by ID
TEST_F(ClusterTest, RemoveNeuronById) {
    snnfw::Cluster cluster(100);
    cluster.addNeuron(neuronId1);
    cluster.addNeuron(neuronId2);
    cluster.addNeuron(neuronId3);

    EXPECT_EQ(cluster.size(), 3);

    bool removed = cluster.removeNeuron(neuronId2);
    EXPECT_TRUE(removed);
    EXPECT_EQ(cluster.size(), 2);

    // Check that the right neuron was removed
    EXPECT_EQ(cluster.getNeuronId(0), neuronId1);
    EXPECT_EQ(cluster.getNeuronId(1), neuronId3);
}

// Test: Remove neuron by invalid ID
TEST_F(ClusterTest, RemoveNeuronInvalidId) {
    snnfw::Cluster cluster(100);
    cluster.addNeuron(neuronId1);

    bool removed = cluster.removeNeuron(999999);
    EXPECT_FALSE(removed);
    EXPECT_EQ(cluster.size(), 1);
}

// Test: Clear all neurons
TEST_F(ClusterTest, ClearAllNeurons) {
    snnfw::Cluster cluster(100);
    cluster.addNeuron(neuronId1);
    cluster.addNeuron(neuronId2);
    cluster.addNeuron(neuronId3);

    EXPECT_EQ(cluster.size(), 3);

    cluster.clear();
    EXPECT_EQ(cluster.size(), 0);
}

// Test: Get neuron IDs vector
TEST_F(ClusterTest, GetNeuronIdsVector) {
    snnfw::Cluster cluster(100);
    cluster.addNeuron(neuronId1);
    cluster.addNeuron(neuronId2);

    const auto& neuronIds = cluster.getNeuronIds();
    EXPECT_EQ(neuronIds.size(), 2);
    EXPECT_EQ(neuronIds[0], neuronId1);
    EXPECT_EQ(neuronIds[1], neuronId2);
}

// Test: Neuron IDs are preserved
TEST_F(ClusterTest, NeuronIdsPreserved) {
    snnfw::Cluster cluster(100);
    cluster.addNeuron(neuronId1);
    cluster.addNeuron(neuronId2);
    cluster.addNeuron(neuronId3);

    EXPECT_EQ(cluster.getNeuronId(0), neuronId1);
    EXPECT_EQ(cluster.getNeuronId(1), neuronId2);
    EXPECT_EQ(cluster.getNeuronId(2), neuronId3);
}

// Test: Cluster inherits from NeuralObject
TEST_F(ClusterTest, InheritsFromNeuralObject) {
    snnfw::Cluster cluster(999);

    // Should be able to use NeuralObject interface
    uint64_t id = cluster.getId();
    EXPECT_EQ(id, 999);
}

// Test: Print info doesn't crash
TEST_F(ClusterTest, PrintInfoDoesntCrash) {
    snnfw::Cluster cluster(100);
    cluster.addNeuron(neuronId1);
    cluster.addNeuron(neuronId2);

    // Just verify it doesn't crash
    cluster.printInfo();
    SUCCEED();
}

// Test: Empty cluster print info
TEST_F(ClusterTest, EmptyClusterPrintInfo) {
    snnfw::Cluster cluster(100);

    // Should work even with no neurons
    cluster.printInfo();
    SUCCEED();
}

// Test: Multiple clusters with different IDs
TEST_F(ClusterTest, MultipleClustersWithDifferentIds) {
    snnfw::Cluster cluster1(100);
    snnfw::Cluster cluster2(200);
    snnfw::Cluster cluster3(300);

    EXPECT_EQ(cluster1.getId(), 100);
    EXPECT_EQ(cluster2.getId(), 200);
    EXPECT_EQ(cluster3.getId(), 300);
}

// Test: Add duplicate neuron IDs
TEST_F(ClusterTest, AddDuplicateNeuronIds) {
    snnfw::Cluster cluster(100);

    cluster.addNeuron(neuronId1);
    cluster.addNeuron(neuronId1);  // Add same ID again

    // Both should be added (no duplicate checking)
    EXPECT_EQ(cluster.size(), 2);
    EXPECT_EQ(cluster.getNeuronId(0), neuronId1);
    EXPECT_EQ(cluster.getNeuronId(1), neuronId1);
}

// Test: Remove first occurrence of duplicate ID
TEST_F(ClusterTest, RemoveDuplicateId) {
    snnfw::Cluster cluster(100);

    cluster.addNeuron(neuronId1);
    cluster.addNeuron(neuronId2);
    cluster.addNeuron(neuronId1);  // Duplicate

    EXPECT_EQ(cluster.size(), 3);

    // Remove should remove first occurrence
    bool removed = cluster.removeNeuron(neuronId1);
    EXPECT_TRUE(removed);
    EXPECT_EQ(cluster.size(), 2);

    // Second occurrence should still be there
    EXPECT_EQ(cluster.getNeuronId(0), neuronId2);
    EXPECT_EQ(cluster.getNeuronId(1), neuronId1);
}

