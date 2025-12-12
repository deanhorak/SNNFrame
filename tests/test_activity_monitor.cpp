#include <gtest/gtest.h>
#include "snnfw/ActivityMonitor.h"
#include "snnfw/Datastore.h"
#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Neuron.h"
#include "snnfw/Axon.h"
#include "snnfw/Dendrite.h"
#include "snnfw/Synapse.h"
#include "snnfw/ActionPotential.h"
#include "snnfw/Cluster.h"
#include "snnfw/Layer.h"
#include "snnfw/Column.h"
#include "snnfw/Nucleus.h"
#include "snnfw/Region.h"
#include "snnfw/Lobe.h"
#include "snnfw/Hemisphere.h"
#include "snnfw/Brain.h"
#include <memory>
#include <thread>
#include <chrono>

using namespace snnfw;

class ActivityMonitorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary datastore
        datastore = std::make_unique<Datastore>("test_activity_monitor_db");
        factory = std::make_unique<NeuralObjectFactory>();

        // Create a simple network for testing
        // Source neuron -> Axon -> Synapse -> Dendrite -> Target neuron
        sourceNeuron = factory->createNeuron(50.0, 0.95, 20);
        targetNeuron = factory->createNeuron(50.0, 0.95, 20);

        axon = factory->createAxon(sourceNeuron->getId());
        dendrite = factory->createDendrite(targetNeuron->getId());
        synapse = factory->createSynapse(axon->getId(), dendrite->getId(), 0.8, 1.5);

        // Link them together
        sourceNeuron->setAxonId(axon->getId());
        axon->addSynapse(synapse->getId());
        targetNeuron->addDendrite(dendrite->getId());

        // Store in datastore
        datastore->put(sourceNeuron);
        datastore->put(targetNeuron);
        datastore->put(axon);
        datastore->put(dendrite);
        datastore->put(synapse);
    }
    
    void TearDown() override {
        factory.reset();
        datastore.reset();
        
        // Clean up test database
        std::system("rm -rf test_activity_monitor_db");
    }
    
    std::unique_ptr<Datastore> datastore;
    std::unique_ptr<NeuralObjectFactory> factory;
    std::shared_ptr<Neuron> sourceNeuron;
    std::shared_ptr<Neuron> targetNeuron;
    std::shared_ptr<Axon> axon;
    std::shared_ptr<Dendrite> dendrite;
    std::shared_ptr<Synapse> synapse;
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(ActivityMonitorTest, ConstructorAndDefaults) {
    ActivityMonitor monitor(*datastore);
    
    EXPECT_FALSE(monitor.isMonitoring());
    EXPECT_DOUBLE_EQ(monitor.getHistoryDuration(), 1000.0);  // Default 1 second
    EXPECT_DOUBLE_EQ(monitor.getSnapshotInterval(), 10.0);   // Default 10ms
    EXPECT_EQ(monitor.getTotalSpikeCount(), 0);
}

TEST_F(ActivityMonitorTest, StartAndStopMonitoring) {
    ActivityMonitor monitor(*datastore);
    
    EXPECT_FALSE(monitor.isMonitoring());
    
    monitor.startMonitoring();
    EXPECT_TRUE(monitor.isMonitoring());
    
    monitor.stopMonitoring();
    EXPECT_FALSE(monitor.isMonitoring());
}

TEST_F(ActivityMonitorTest, StartTwice) {
    ActivityMonitor monitor(*datastore);
    
    monitor.startMonitoring();
    EXPECT_TRUE(monitor.isMonitoring());
    
    // Starting again should not crash
    monitor.startMonitoring();
    EXPECT_TRUE(monitor.isMonitoring());
    
    monitor.stopMonitoring();
}

TEST_F(ActivityMonitorTest, StopWithoutStart) {
    ActivityMonitor monitor(*datastore);
    
    // Stopping without starting should not crash
    EXPECT_NO_THROW(monitor.stopMonitoring());
    EXPECT_FALSE(monitor.isMonitoring());
}

TEST_F(ActivityMonitorTest, SetHistoryDuration) {
    ActivityMonitor monitor(*datastore);
    
    monitor.setHistoryDuration(500.0);
    EXPECT_DOUBLE_EQ(monitor.getHistoryDuration(), 500.0);
    
    monitor.setHistoryDuration(2000.0);
    EXPECT_DOUBLE_EQ(monitor.getHistoryDuration(), 2000.0);
}

TEST_F(ActivityMonitorTest, SetSnapshotInterval) {
    ActivityMonitor monitor(*datastore);
    
    monitor.setSnapshotInterval(5.0);
    EXPECT_DOUBLE_EQ(monitor.getSnapshotInterval(), 5.0);
    
    monitor.setSnapshotInterval(20.0);
    EXPECT_DOUBLE_EQ(monitor.getSnapshotInterval(), 20.0);
}

// ============================================================================
// Spike Recording Tests
// ============================================================================

TEST_F(ActivityMonitorTest, RecordSingleSpike) {
    ActivityMonitor monitor(*datastore);
    monitor.startMonitoring();
    
    auto ap = std::make_shared<ActionPotential>(
        synapse->getId(),
        dendrite->getId(),
        10.0,  // scheduled time
        0.8    // amplitude
    );
    
    monitor.recordSpike(ap, 10.0);
    
    EXPECT_EQ(monitor.getTotalSpikeCount(), 1);
}

TEST_F(ActivityMonitorTest, RecordMultipleSpikes) {
    ActivityMonitor monitor(*datastore);
    monitor.startMonitoring();
    
    for (int i = 0; i < 10; ++i) {
        auto ap = std::make_shared<ActionPotential>(
            synapse->getId(),
            dendrite->getId(),
            i * 1.0,
            0.8
        );
        monitor.recordSpike(ap, i * 1.0);
    }
    
    EXPECT_EQ(monitor.getTotalSpikeCount(), 10);
}

TEST_F(ActivityMonitorTest, RecordSpikeWhenNotMonitoring) {
    ActivityMonitor monitor(*datastore);
    // Not started
    
    auto ap = std::make_shared<ActionPotential>(
        synapse->getId(),
        dendrite->getId(),
        10.0,
        0.8
    );
    
    monitor.recordSpike(ap, 10.0);
    
    // Should not record
    EXPECT_EQ(monitor.getTotalSpikeCount(), 0);
}

TEST_F(ActivityMonitorTest, RecordNullSpike) {
    ActivityMonitor monitor(*datastore);
    monitor.startMonitoring();
    
    monitor.recordSpike(nullptr, 10.0);
    
    // Should not crash, should not record
    EXPECT_EQ(monitor.getTotalSpikeCount(), 0);
}

TEST_F(ActivityMonitorTest, RecordNeuronFiring) {
    ActivityMonitor monitor(*datastore);
    monitor.startMonitoring();
    
    // Should not crash
    EXPECT_NO_THROW(monitor.recordNeuronFiring(sourceNeuron->getId(), 10.0));
}

// ============================================================================
// Activity Snapshot Tests
// ============================================================================

TEST_F(ActivityMonitorTest, GetActivitySnapshotEmpty) {
    ActivityMonitor monitor(*datastore);
    monitor.startMonitoring();
    
    auto snapshot = monitor.getActivitySnapshot(0.0, 100.0);
    
    EXPECT_DOUBLE_EQ(snapshot.startTime, 0.0);
    EXPECT_DOUBLE_EQ(snapshot.endTime, 100.0);
    EXPECT_EQ(snapshot.totalSpikes, 0);
    EXPECT_DOUBLE_EQ(snapshot.averageSpikeRate, 0.0);
}

TEST_F(ActivityMonitorTest, GetActivitySnapshotWithSpikes) {
    ActivityMonitor monitor(*datastore);
    monitor.startMonitoring();
    
    // Record 5 spikes at different times
    for (int i = 0; i < 5; ++i) {
        auto ap = std::make_shared<ActionPotential>(
            synapse->getId(),
            dendrite->getId(),
            i * 10.0,
            0.8
        );
        monitor.recordSpike(ap, i * 10.0);
    }
    
    auto snapshot = monitor.getActivitySnapshot(0.0, 50.0);
    
    EXPECT_EQ(snapshot.totalSpikes, 5);
    EXPECT_GT(snapshot.averageSpikeRate, 0.0);
    EXPECT_EQ(snapshot.neuronSpikeCount.size(), 1);  // Only source neuron
}

TEST_F(ActivityMonitorTest, GetActivitySnapshotTimeWindow) {
    ActivityMonitor monitor(*datastore);
    monitor.startMonitoring();
    
    // Record spikes at times 0, 10, 20, 30, 40
    for (int i = 0; i < 5; ++i) {
        auto ap = std::make_shared<ActionPotential>(
            synapse->getId(),
            dendrite->getId(),
            i * 10.0,
            0.8
        );
        monitor.recordSpike(ap, i * 10.0);
    }
    
    // Get snapshot for window [10, 30)
    auto snapshot = monitor.getActivitySnapshot(10.0, 30.0);
    
    // Should include spikes at 10 and 20, but not 0 or 30
    EXPECT_EQ(snapshot.totalSpikes, 2);
}

TEST_F(ActivityMonitorTest, GetLatestSnapshotEmpty) {
    ActivityMonitor monitor(*datastore);
    monitor.startMonitoring();
    
    auto snapshot = monitor.getLatestSnapshot();
    
    // Should return empty snapshot
    EXPECT_EQ(snapshot.totalSpikes, 0);
}

// ============================================================================
// Spike Events Tests
// ============================================================================

TEST_F(ActivityMonitorTest, GetSpikeEventsEmpty) {
    ActivityMonitor monitor(*datastore);
    monitor.startMonitoring();
    
    auto events = monitor.getSpikeEvents(0.0, 100.0);
    
    EXPECT_TRUE(events.empty());
}

TEST_F(ActivityMonitorTest, GetSpikeEventsWithData) {
    ActivityMonitor monitor(*datastore);
    monitor.startMonitoring();
    
    // Record 3 spikes
    for (int i = 0; i < 3; ++i) {
        auto ap = std::make_shared<ActionPotential>(
            synapse->getId(),
            dendrite->getId(),
            i * 10.0,
            0.8
        );
        monitor.recordSpike(ap, i * 10.0);
    }
    
    auto events = monitor.getSpikeEvents(0.0, 30.0);
    
    EXPECT_EQ(events.size(), 3);
    EXPECT_EQ(events[0].synapseId, synapse->getId());
    EXPECT_EQ(events[0].sourceNeuronId, sourceNeuron->getId());
    EXPECT_EQ(events[0].targetNeuronId, targetNeuron->getId());
}

TEST_F(ActivityMonitorTest, GetSpikeEventsTimeWindow) {
    ActivityMonitor monitor(*datastore);
    monitor.startMonitoring();
    
    // Record spikes at times 0, 10, 20, 30, 40
    for (int i = 0; i < 5; ++i) {
        auto ap = std::make_shared<ActionPotential>(
            synapse->getId(),
            dendrite->getId(),
            i * 10.0,
            0.8
        );
        monitor.recordSpike(ap, i * 10.0);
    }
    
    // Get events for window [15, 35)
    auto events = monitor.getSpikeEvents(15.0, 35.0);
    
    // Should include spikes at 20 and 30
    EXPECT_EQ(events.size(), 2);
    EXPECT_DOUBLE_EQ(events[0].timestamp, 20.0);
    EXPECT_DOUBLE_EQ(events[1].timestamp, 30.0);
}

// ============================================================================
// Flow Statistics Tests
// ============================================================================

TEST_F(ActivityMonitorTest, GetFlowStatisticsEmpty) {
    ActivityMonitor monitor(*datastore);
    monitor.startMonitoring();
    
    auto stats = monitor.getFlowStatistics(0.0, 100.0);
    
    EXPECT_TRUE(stats.clusterToClusterFlow.empty());
    EXPECT_TRUE(stats.layerToLayerFlow.empty());
    EXPECT_TRUE(stats.columnToColumnFlow.empty());
}

// ============================================================================
// Callback Tests
// ============================================================================

TEST_F(ActivityMonitorTest, RegisterAndUnregisterCallback) {
    ActivityMonitor monitor(*datastore);
    
    bool callbackCalled = false;
    auto callback = [&callbackCalled](const SpikeEvent& event) {
        callbackCalled = true;
    };
    
    uint64_t id = monitor.registerSpikeCallback(callback);
    EXPECT_GT(id, 0);
    
    monitor.startMonitoring();
    
    auto ap = std::make_shared<ActionPotential>(
        synapse->getId(),
        dendrite->getId(),
        10.0,
        0.8
    );
    monitor.recordSpike(ap, 10.0);
    
    EXPECT_TRUE(callbackCalled);
    
    // Unregister
    monitor.unregisterSpikeCallback(id);
}

TEST_F(ActivityMonitorTest, MultipleCallbacks) {
    ActivityMonitor monitor(*datastore);
    
    int callback1Count = 0;
    int callback2Count = 0;
    
    auto callback1 = [&callback1Count](const SpikeEvent& event) {
        callback1Count++;
    };
    
    auto callback2 = [&callback2Count](const SpikeEvent& event) {
        callback2Count++;
    };
    
    uint64_t id1 = monitor.registerSpikeCallback(callback1);
    uint64_t id2 = monitor.registerSpikeCallback(callback2);
    
    monitor.startMonitoring();
    
    auto ap = std::make_shared<ActionPotential>(
        synapse->getId(),
        dendrite->getId(),
        10.0,
        0.8
    );
    monitor.recordSpike(ap, 10.0);
    
    EXPECT_EQ(callback1Count, 1);
    EXPECT_EQ(callback2Count, 1);
}

// ============================================================================
// History Management Tests
// ============================================================================

TEST_F(ActivityMonitorTest, ClearHistory) {
    ActivityMonitor monitor(*datastore);
    monitor.startMonitoring();
    
    // Record some spikes
    for (int i = 0; i < 5; ++i) {
        auto ap = std::make_shared<ActionPotential>(
            synapse->getId(),
            dendrite->getId(),
            i * 10.0,
            0.8
        );
        monitor.recordSpike(ap, i * 10.0);
    }
    
    EXPECT_EQ(monitor.getTotalSpikeCount(), 5);
    
    monitor.clearHistory();
    
    EXPECT_EQ(monitor.getTotalSpikeCount(), 0);
}

TEST_F(ActivityMonitorTest, GetCurrentSpikeRate) {
    ActivityMonitor monitor(*datastore);
    monitor.startMonitoring();
    
    // Initially zero
    EXPECT_DOUBLE_EQ(monitor.getCurrentSpikeRate(100.0), 0.0);
    
    // Record some spikes
    for (int i = 0; i < 10; ++i) {
        auto ap = std::make_shared<ActionPotential>(
            synapse->getId(),
            dendrite->getId(),
            i * 1.0,
            0.8
        );
        monitor.recordSpike(ap, i * 1.0);
    }
    
    // Should have non-zero rate
    double rate = monitor.getCurrentSpikeRate(10.0);
    EXPECT_GT(rate, 0.0);
}

// ============================================================================
// Hierarchical Context Cache Tests
// ============================================================================

TEST_F(ActivityMonitorTest, BuildHierarchicalCache) {
    // Create a hierarchical structure: Brain -> Hemisphere -> Lobe -> Region -> Nucleus -> Column -> Layer -> Cluster -> Neurons
    auto brain = factory->createBrain();
    auto hemisphere = factory->createHemisphere();
    auto lobe = factory->createLobe();
    auto region = factory->createRegion();
    auto nucleus = factory->createNucleus();
    auto column = factory->createColumn();
    auto layer = factory->createLayer();
    auto cluster = factory->createCluster();

    // Build hierarchy
    brain->addHemisphere(hemisphere->getId());
    hemisphere->addLobe(lobe->getId());
    lobe->addRegion(region->getId());
    region->addNucleus(nucleus->getId());
    nucleus->addColumn(column->getId());
    column->addLayer(layer->getId());
    layer->addCluster(cluster->getId());

    // Add neurons to cluster
    auto neuron1 = factory->createNeuron(50.0, 0.95, 20);
    auto neuron2 = factory->createNeuron(50.0, 0.95, 20);
    auto neuron3 = factory->createNeuron(50.0, 0.95, 20);

    cluster->addNeuron(neuron1->getId());
    cluster->addNeuron(neuron2->getId());
    cluster->addNeuron(neuron3->getId());

    // Store everything
    datastore->put(brain);
    datastore->put(hemisphere);
    datastore->put(lobe);
    datastore->put(region);
    datastore->put(nucleus);
    datastore->put(column);
    datastore->put(layer);
    datastore->put(cluster);
    datastore->put(neuron1);
    datastore->put(neuron2);
    datastore->put(neuron3);

    // Build hierarchical cache
    ActivityMonitor monitor(*datastore);
    size_t neuronCount = monitor.buildHierarchicalCache(brain->getId());

    // Should have indexed 3 neurons
    EXPECT_EQ(neuronCount, 3);

    // Now test that context resolution works
    monitor.startMonitoring();

    // Create a spike event
    auto axon1 = factory->createAxon(neuron1->getId());
    auto dendrite2 = factory->createDendrite(neuron2->getId());
    auto synapse12 = factory->createSynapse(axon1->getId(), dendrite2->getId(), 0.8, 1.5);

    neuron1->setAxonId(axon1->getId());
    neuron2->addDendrite(dendrite2->getId());

    datastore->put(axon1);
    datastore->put(dendrite2);
    datastore->put(synapse12);

    // Record a spike
    auto actionPotential = std::make_shared<ActionPotential>(
        synapse12->getId(), dendrite2->getId(), 1.0, 0.0);

    monitor.recordSpike(actionPotential, 10.0);

    // Get activity snapshot
    auto snapshot = monitor.getActivitySnapshot(0.0, 20.0);

    // Should have recorded the spike
    EXPECT_EQ(snapshot.totalSpikes, 1);

    // Should have cluster-level aggregation (only counts once since source and target are in same cluster)
    EXPECT_EQ(snapshot.clusterSpikeCount.size(), 1);
    EXPECT_EQ(snapshot.clusterSpikeCount[cluster->getId()], 1);

    // Should have layer-level aggregation (only counts once since source and target are in same layer)
    EXPECT_EQ(snapshot.layerSpikeCount.size(), 1);
    EXPECT_EQ(snapshot.layerSpikeCount[layer->getId()], 1);

    // Should have column-level aggregation (only counts once since source and target are in same column)
    EXPECT_EQ(snapshot.columnSpikeCount.size(), 1);
    EXPECT_EQ(snapshot.columnSpikeCount[column->getId()], 1);

    // Propagation type should be INTRA_CLUSTER (same cluster)
    EXPECT_EQ(snapshot.propagationTypeCount[SpikeEvent::PropagationType::INTRA_CLUSTER], 1);
}

TEST_F(ActivityMonitorTest, HierarchicalCacheMultipleClusters) {
    // Create two clusters in different layers
    auto brain = factory->createBrain();
    auto hemisphere = factory->createHemisphere();
    auto lobe = factory->createLobe();
    auto region = factory->createRegion();
    auto nucleus = factory->createNucleus();
    auto column = factory->createColumn();

    auto layer1 = factory->createLayer();
    auto layer2 = factory->createLayer();

    auto cluster1 = factory->createCluster();
    auto cluster2 = factory->createCluster();

    // Build hierarchy
    brain->addHemisphere(hemisphere->getId());
    hemisphere->addLobe(lobe->getId());
    lobe->addRegion(region->getId());
    region->addNucleus(nucleus->getId());
    nucleus->addColumn(column->getId());
    column->addLayer(layer1->getId());
    column->addLayer(layer2->getId());
    layer1->addCluster(cluster1->getId());
    layer2->addCluster(cluster2->getId());

    // Add neurons
    auto neuron1 = factory->createNeuron(50.0, 0.95, 20);
    auto neuron2 = factory->createNeuron(50.0, 0.95, 20);

    cluster1->addNeuron(neuron1->getId());
    cluster2->addNeuron(neuron2->getId());

    // Store everything
    datastore->put(brain);
    datastore->put(hemisphere);
    datastore->put(lobe);
    datastore->put(region);
    datastore->put(nucleus);
    datastore->put(column);
    datastore->put(layer1);
    datastore->put(layer2);
    datastore->put(cluster1);
    datastore->put(cluster2);
    datastore->put(neuron1);
    datastore->put(neuron2);

    // Build cache
    ActivityMonitor monitor(*datastore);
    size_t neuronCount = monitor.buildHierarchicalCache(brain->getId());
    EXPECT_EQ(neuronCount, 2);

    monitor.startMonitoring();

    // Create inter-layer connection
    auto axon1 = factory->createAxon(neuron1->getId());
    auto dendrite2 = factory->createDendrite(neuron2->getId());
    auto synapse12 = factory->createSynapse(axon1->getId(), dendrite2->getId(), 0.8, 1.5);

    neuron1->setAxonId(axon1->getId());
    neuron2->addDendrite(dendrite2->getId());

    datastore->put(axon1);
    datastore->put(dendrite2);
    datastore->put(synapse12);

    // Record spike
    auto actionPotential = std::make_shared<ActionPotential>(
        synapse12->getId(), dendrite2->getId(), 1.0, 0.0);

    monitor.recordSpike(actionPotential, 10.0);

    // Get snapshot
    auto snapshot = monitor.getActivitySnapshot(0.0, 20.0);

    // Should detect INTRA_COLUMN propagation (same column, different layers)
    EXPECT_EQ(snapshot.propagationTypeCount[SpikeEvent::PropagationType::INTRA_COLUMN], 1);

    // Should have 2 different clusters
    EXPECT_EQ(snapshot.clusterSpikeCount.size(), 2);

    // Should have 2 different layers
    EXPECT_EQ(snapshot.layerSpikeCount.size(), 2);

    // Should have 1 column
    EXPECT_EQ(snapshot.columnSpikeCount.size(), 1);
}

// Main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

