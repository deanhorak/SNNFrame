#include <gtest/gtest.h>
#include "snnfw/NetworkInspector.h"
#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Datastore.h"
#include "snnfw/Brain.h"
#include "snnfw/Hemisphere.h"
#include "snnfw/Lobe.h"
#include "snnfw/Region.h"
#include "snnfw/Nucleus.h"
#include "snnfw/Column.h"
#include "snnfw/Layer.h"
#include "snnfw/Cluster.h"
#include "snnfw/Neuron.h"
#include "snnfw/Axon.h"
#include "snnfw/Dendrite.h"
#include "snnfw/Synapse.h"
#include "snnfw/Logger.h"
#include <filesystem>
#include <memory>

using namespace snnfw;

class NetworkInspectorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize logger to suppress output during tests
        Logger::getInstance().getLogger()->set_level(spdlog::level::warn);
        
        // Create unique database path for this test
        dbPath = "./test_inspector_db_" + std::to_string(testCounter++);
        
        // Clean up any existing database
        if (std::filesystem::exists(dbPath)) {
            std::filesystem::remove_all(dbPath);
        }
        
        datastore = std::make_unique<Datastore>(dbPath, 1000);
        factory = std::make_unique<NeuralObjectFactory>();
        inspector = std::make_unique<NetworkInspector>();
    }
    
    void TearDown() override {
        // Clean up
        inspector.reset();
        datastore.reset();
        factory.reset();
        
        // Remove test database
        if (std::filesystem::exists(dbPath)) {
            std::filesystem::remove_all(dbPath);
        }
    }
    
    // Helper to create a simple network: Brain -> Hemisphere -> Lobe -> Region -> Nucleus -> Column -> Layer -> Cluster -> Neurons
    void createSimpleHierarchy() {
        // Create brain
        brain = factory->createBrain();
        brain->setName("TestBrain");
        datastore->put(brain);

        // Create hemisphere
        hemisphere = factory->createHemisphere();
        hemisphere->setName("LeftHemisphere");
        datastore->put(hemisphere);
        brain->addHemisphere(hemisphere->getId());
        datastore->markDirty(brain->getId());

        // Create lobe
        lobe = factory->createLobe();
        lobe->setName("OccipitalLobe");
        datastore->put(lobe);
        hemisphere->addLobe(lobe->getId());
        datastore->markDirty(hemisphere->getId());

        // Create region
        region = factory->createRegion();
        region->setName("V1");
        datastore->put(region);
        lobe->addRegion(region->getId());
        datastore->markDirty(lobe->getId());

        // Create nucleus
        nucleus = factory->createNucleus();
        nucleus->setName("V1Nucleus");
        datastore->put(nucleus);
        region->addNucleus(nucleus->getId());
        datastore->markDirty(region->getId());

        // Create column
        column = factory->createColumn();
        datastore->put(column);
        nucleus->addColumn(column->getId());
        datastore->markDirty(nucleus->getId());

        // Create layer
        layer = factory->createLayer();
        datastore->put(layer);
        column->addLayer(layer->getId());
        datastore->markDirty(column->getId());
        
        // Create cluster
        cluster = factory->createCluster();
        datastore->put(cluster);
        layer->addCluster(cluster->getId());
        datastore->markDirty(layer->getId());
        
        // Create neurons
        for (int i = 0; i < 5; ++i) {
            auto neuron = factory->createNeuron(100.0, 0.7, 10);
            datastore->put(neuron);
            cluster->addNeuron(neuron->getId());
            neurons.push_back(neuron);
        }
        datastore->markDirty(cluster->getId());
    }
    
    // Helper to create connections between neurons
    void createConnections() {
        if (neurons.size() < 2) return;
        
        // Connect neuron 0 -> neuron 1
        auto axon0 = factory->createAxon(neurons[0]->getId());
        datastore->put(axon0);
        neurons[0]->setAxonId(axon0->getId());
        datastore->markDirty(neurons[0]->getId());
        
        auto dendrite1 = factory->createDendrite(neurons[1]->getId());
        datastore->put(dendrite1);
        neurons[1]->addDendrite(dendrite1->getId());
        datastore->markDirty(neurons[1]->getId());

        auto synapse = factory->createSynapse(axon0->getId(), dendrite1->getId(), 0.5, 1.0);
        datastore->put(synapse);
        axon0->addSynapse(synapse->getId());
        dendrite1->addSynapse(synapse->getId());
        datastore->markDirty(axon0->getId());
        datastore->markDirty(dendrite1->getId());
        
        // Connect neuron 1 -> neuron 2
        auto axon1 = factory->createAxon(neurons[1]->getId());
        datastore->put(axon1);
        neurons[1]->setAxonId(axon1->getId());
        datastore->markDirty(neurons[1]->getId());
        
        auto dendrite2 = factory->createDendrite(neurons[2]->getId());
        datastore->put(dendrite2);
        neurons[2]->addDendrite(dendrite2->getId());
        datastore->markDirty(neurons[2]->getId());

        auto synapse2 = factory->createSynapse(axon1->getId(), dendrite2->getId(), 0.8, 1.0);
        datastore->put(synapse2);
        axon1->addSynapse(synapse2->getId());
        dendrite2->addSynapse(synapse2->getId());
        datastore->markDirty(axon1->getId());
        datastore->markDirty(dendrite2->getId());
    }
    
    std::string dbPath;
    std::unique_ptr<Datastore> datastore;
    std::unique_ptr<NeuralObjectFactory> factory;
    std::unique_ptr<NetworkInspector> inspector;
    
    std::shared_ptr<Brain> brain;
    std::shared_ptr<Hemisphere> hemisphere;
    std::shared_ptr<Lobe> lobe;
    std::shared_ptr<Region> region;
    std::shared_ptr<Nucleus> nucleus;
    std::shared_ptr<Column> column;
    std::shared_ptr<Layer> layer;
    std::shared_ptr<Cluster> cluster;
    std::vector<std::shared_ptr<Neuron>> neurons;
    
    static int testCounter;
};

int NetworkInspectorTest::testCounter = 0;

// ============================================================================
// Hierarchy Inspection Tests
// ============================================================================

TEST_F(NetworkInspectorTest, InspectBrainBasic) {
    createSimpleHierarchy();
    
    auto stats = inspector->inspectBrain(brain->getId(), *datastore);
    
    EXPECT_EQ(stats.typeName, "Brain");
    EXPECT_EQ(stats.id, brain->getId());
    EXPECT_EQ(stats.name, "TestBrain");
    EXPECT_EQ(stats.childCount, 1);  // 1 hemisphere
    EXPECT_EQ(stats.totalNeurons, 5);  // 5 neurons
    EXPECT_EQ(stats.totalClusters, 1);  // 1 cluster
}

TEST_F(NetworkInspectorTest, InspectHemisphere) {
    createSimpleHierarchy();
    
    auto stats = inspector->inspectHierarchy(hemisphere->getId(), "Hemisphere", *datastore);
    
    EXPECT_EQ(stats.typeName, "Hemisphere");
    EXPECT_EQ(stats.id, hemisphere->getId());
    EXPECT_EQ(stats.name, "LeftHemisphere");
    EXPECT_EQ(stats.totalNeurons, 5);
}

TEST_F(NetworkInspectorTest, InspectCluster) {
    createSimpleHierarchy();
    
    auto stats = inspector->inspectHierarchy(cluster->getId(), "Cluster", *datastore);
    
    EXPECT_EQ(stats.typeName, "Cluster");
    EXPECT_EQ(stats.id, cluster->getId());
    EXPECT_EQ(stats.childCount, 5);  // 5 neurons
    EXPECT_EQ(stats.totalNeurons, 5);
    EXPECT_EQ(stats.totalClusters, 1);
}

TEST_F(NetworkInspectorTest, InspectNonexistentBrain) {
    auto stats = inspector->inspectBrain(999999999999999ULL, *datastore);
    
    // Should return empty stats
    EXPECT_EQ(stats.totalNeurons, 0);
    EXPECT_EQ(stats.childCount, 0);
}

// ============================================================================
// Connectivity Analysis Tests
// ============================================================================

TEST_F(NetworkInspectorTest, AnalyzeConnectivityNoConnections) {
    createSimpleHierarchy();
    
    auto stats = inspector->analyzeConnectivity(neurons[0]->getId(), *datastore);
    
    EXPECT_EQ(stats.neuronId, neurons[0]->getId());
    EXPECT_EQ(stats.inDegree, 0);
    EXPECT_EQ(stats.outDegree, 0);
}

TEST_F(NetworkInspectorTest, AnalyzeConnectivityWithConnections) {
    createSimpleHierarchy();
    createConnections();
    
    // Neuron 0 has 1 outgoing connection
    auto stats0 = inspector->analyzeConnectivity(neurons[0]->getId(), *datastore);
    EXPECT_EQ(stats0.outDegree, 1);
    EXPECT_EQ(stats0.inDegree, 0);
    EXPECT_EQ(stats0.postsynapticNeurons.size(), 1);
    EXPECT_EQ(stats0.postsynapticNeurons[0], neurons[1]->getId());
    
    // Neuron 1 has 1 incoming and 1 outgoing connection
    auto stats1 = inspector->analyzeConnectivity(neurons[1]->getId(), *datastore);
    EXPECT_EQ(stats1.inDegree, 1);
    EXPECT_EQ(stats1.outDegree, 1);
    EXPECT_EQ(stats1.presynapticNeurons.size(), 1);
    EXPECT_EQ(stats1.presynapticNeurons[0], neurons[0]->getId());
    EXPECT_EQ(stats1.postsynapticNeurons.size(), 1);
    EXPECT_EQ(stats1.postsynapticNeurons[0], neurons[2]->getId());
    
    // Neuron 2 has 1 incoming connection
    auto stats2 = inspector->analyzeConnectivity(neurons[2]->getId(), *datastore);
    EXPECT_EQ(stats2.inDegree, 1);
    EXPECT_EQ(stats2.outDegree, 0);
    EXPECT_EQ(stats2.presynapticNeurons.size(), 1);
    EXPECT_EQ(stats2.presynapticNeurons[0], neurons[1]->getId());
}

TEST_F(NetworkInspectorTest, AnalyzeClusterConnectivity) {
    createSimpleHierarchy();
    createConnections();
    
    auto results = inspector->analyzeClusterConnectivity(cluster->getId(), *datastore);
    
    EXPECT_EQ(results.size(), 5);  // 5 neurons in cluster
}

// ============================================================================
// Neuron State Inspection Tests
// ============================================================================

TEST_F(NetworkInspectorTest, InspectNeuronBasic) {
    createSimpleHierarchy();
    
    auto stats = inspector->inspectNeuron(neurons[0]->getId(), *datastore);
    
    EXPECT_EQ(stats.neuronId, neurons[0]->getId());
    EXPECT_EQ(stats.learnedPatternCount, 0);  // No patterns learned yet
    EXPECT_EQ(stats.currentSpikeCount, 0);  // No spikes yet
    EXPECT_EQ(stats.windowSizeMs, 100.0);
    EXPECT_EQ(stats.similarityThreshold, 0.7);
    EXPECT_EQ(stats.maxReferencePatterns, 10);
}

TEST_F(NetworkInspectorTest, InspectNeuronWithSpikes) {
    createSimpleHierarchy();
    
    // Add some spikes
    neurons[0]->insertSpike(10.0);
    neurons[0]->insertSpike(20.0);
    neurons[0]->insertSpike(30.0);
    datastore->markDirty(neurons[0]->getId());
    
    auto stats = inspector->inspectNeuron(neurons[0]->getId(), *datastore);
    
    EXPECT_EQ(stats.currentSpikeCount, 3);
}

TEST_F(NetworkInspectorTest, InspectNeuronWithLearnedPatterns) {
    createSimpleHierarchy();
    
    // Add spikes and learn pattern
    neurons[0]->insertSpike(10.0);
    neurons[0]->insertSpike(20.0);
    neurons[0]->learnCurrentPattern();
    datastore->markDirty(neurons[0]->getId());
    
    auto stats = inspector->inspectNeuron(neurons[0]->getId(), *datastore);
    
    EXPECT_EQ(stats.learnedPatternCount, 1);
}

// ============================================================================
// Report Generation Tests
// ============================================================================

TEST_F(NetworkInspectorTest, GenerateTextReport) {
    createSimpleHierarchy();
    
    inspector->inspectBrain(brain->getId(), *datastore);
    std::string report = inspector->generateReport(ReportFormat::TEXT);
    
    EXPECT_FALSE(report.empty());
    EXPECT_NE(report.find("Network Inspection Report"), std::string::npos);
    EXPECT_NE(report.find("Hierarchy Statistics"), std::string::npos);
}

TEST_F(NetworkInspectorTest, GenerateJsonReport) {
    createSimpleHierarchy();
    
    inspector->inspectBrain(brain->getId(), *datastore);
    std::string report = inspector->generateReport(ReportFormat::JSON);
    
    EXPECT_FALSE(report.empty());
    EXPECT_NE(report.find("hierarchy"), std::string::npos);
}

TEST_F(NetworkInspectorTest, GenerateMarkdownReport) {
    createSimpleHierarchy();
    
    inspector->inspectBrain(brain->getId(), *datastore);
    std::string report = inspector->generateReport(ReportFormat::MARKDOWN);
    
    EXPECT_FALSE(report.empty());
    EXPECT_NE(report.find("# Network Inspection Report"), std::string::npos);
    EXPECT_NE(report.find("## Hierarchy Statistics"), std::string::npos);
}

TEST_F(NetworkInspectorTest, GenerateCsvReport) {
    createSimpleHierarchy();
    
    inspector->inspectBrain(brain->getId(), *datastore);
    std::string report = inspector->generateReport(ReportFormat::CSV);
    
    EXPECT_FALSE(report.empty());
    EXPECT_NE(report.find("Type,ID,Name"), std::string::npos);
}

// ============================================================================
// Cache Management Tests
// ============================================================================

TEST_F(NetworkInspectorTest, ClearCache) {
    createSimpleHierarchy();
    
    inspector->inspectBrain(brain->getId(), *datastore);
    EXPECT_FALSE(inspector->getLastHierarchyStats().empty());
    
    inspector->clearCache();
    EXPECT_TRUE(inspector->getLastHierarchyStats().empty());
}

TEST_F(NetworkInspectorTest, CacheAccumulation) {
    createSimpleHierarchy();

    inspector->inspectBrain(brain->getId(), *datastore);
    size_t count1 = inspector->getLastHierarchyStats().size();

    inspector->inspectHierarchy(cluster->getId(), "Cluster", *datastore);
    size_t count2 = inspector->getLastHierarchyStats().size();
    
    EXPECT_GT(count2, count1);  // Cache should accumulate
}

