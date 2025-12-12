#include <gtest/gtest.h>
#include "snnfw/NetworkBuilder.h"
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
#include "snnfw/Logger.h"
#include <spdlog/spdlog.h>
#include <filesystem>
#include <memory>

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
 * @brief Test fixture for NetworkBuilder tests
 */
class NetworkBuilderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Set log level to error to reduce test output
        spdlog::set_level(spdlog::level::err);

        // Create temporary directory for test database
        testDbPath_ = std::filesystem::temp_directory_path() / "snnfw_test_network_builder";
        std::filesystem::remove_all(testDbPath_);
        std::filesystem::create_directories(testDbPath_);

        // Create datastore and factory
        datastore_ = std::make_unique<Datastore>(testDbPath_.string(), 10000);
        registerFactories(*datastore_);
        factory_ = std::make_unique<NeuralObjectFactory>();
    }

    void TearDown() override {
        factory_.reset();
        datastore_.reset();
        std::filesystem::remove_all(testDbPath_);
    }

    std::filesystem::path testDbPath_;
    std::unique_ptr<Datastore> datastore_;
    std::unique_ptr<NeuralObjectFactory> factory_;
};

/**
 * @brief Test basic brain creation
 */
TEST_F(NetworkBuilderTest, BasicBrainCreation) {
    NetworkBuilder builder(*factory_, *datastore_, false);

    auto brain = builder.createBrain("TestBrain").build();

    ASSERT_NE(brain, nullptr);
    EXPECT_EQ(brain->getName(), "TestBrain");
    EXPECT_NE(brain->getId(), 0);
}

/**
 * @brief Test hierarchical structure creation
 */
TEST_F(NetworkBuilderTest, HierarchicalStructure) {
    NetworkBuilder builder(*factory_, *datastore_, false);

    auto brain = builder.createBrain("TestBrain")
        .addHemisphere("Left")
            .addLobe("Occipital")
                .addRegion("V1")
                    .addNucleus("LGN")
        .build();

    ASSERT_NE(brain, nullptr);
    EXPECT_EQ(brain->size(), 1);  // 1 hemisphere

    auto hemisphere = datastore_->getHemisphere(brain->getHemisphereId(0));
    ASSERT_NE(hemisphere, nullptr);
    EXPECT_EQ(hemisphere->getName(), "Left");
    EXPECT_EQ(hemisphere->size(), 1);  // 1 lobe

    auto lobe = datastore_->getLobe(hemisphere->getLobeId(0));
    ASSERT_NE(lobe, nullptr);
    EXPECT_EQ(lobe->getName(), "Occipital");
    EXPECT_EQ(lobe->size(), 1);  // 1 region

    auto region = datastore_->getRegion(lobe->getRegionId(0));
    ASSERT_NE(region, nullptr);
    EXPECT_EQ(region->getName(), "V1");
    EXPECT_EQ(region->size(), 1);  // 1 nucleus

    auto nucleus = datastore_->getNucleus(region->getNucleusId(0));
    ASSERT_NE(nucleus, nullptr);
    EXPECT_EQ(nucleus->getName(), "LGN");
}

/**
 * @brief Test column and layer creation
 */
TEST_F(NetworkBuilderTest, ColumnsAndLayers) {
    NetworkBuilder builder(*factory_, *datastore_, false);

    auto brain = builder.createBrain("TestBrain")
        .addHemisphere("Left")
            .addLobe("Occipital")
                .addRegion("V1")
                    .addNucleus("LGN")
                        .addColumn()
                            .addLayer()
        .build();

    ASSERT_NE(brain, nullptr);

    // Navigate to nucleus
    auto hemisphere = datastore_->getHemisphere(brain->getHemisphereId(0));
    auto lobe = datastore_->getLobe(hemisphere->getLobeId(0));
    auto region = datastore_->getRegion(lobe->getRegionId(0));
    auto nucleus = datastore_->getNucleus(region->getNucleusId(0));

    ASSERT_NE(nucleus, nullptr);
    EXPECT_EQ(nucleus->size(), 1);  // 1 column

    auto column = datastore_->getColumn(nucleus->getColumnId(0));
    ASSERT_NE(column, nullptr);
    EXPECT_EQ(column->size(), 1);  // 1 layer

    auto layer = datastore_->getLayer(column->getLayerId(0));
    ASSERT_NE(layer, nullptr);
}

/**
 * @brief Test bulk column creation
 */
TEST_F(NetworkBuilderTest, BulkColumns) {
    NetworkBuilder builder(*factory_, *datastore_, false);

    auto brain = builder.createBrain("TestBrain")
        .addHemisphere("Left")
            .addLobe("Occipital")
                .addRegion("V1")
                    .addNucleus("LGN")
                        .addColumns(5, "Column")
        .build();

    ASSERT_NE(brain, nullptr);

    // Navigate to nucleus
    auto hemisphere = datastore_->getHemisphere(brain->getHemisphereId(0));
    auto lobe = datastore_->getLobe(hemisphere->getLobeId(0));
    auto region = datastore_->getRegion(lobe->getRegionId(0));
    auto nucleus = datastore_->getNucleus(region->getNucleusId(0));

    ASSERT_NE(nucleus, nullptr);
    EXPECT_EQ(nucleus->size(), 5);  // 5 columns
}

/**
 * @brief Test bulk layer creation
 */
TEST_F(NetworkBuilderTest, BulkLayers) {
    NetworkBuilder builder(*factory_, *datastore_, false);

    auto brain = builder.createBrain("TestBrain")
        .addHemisphere("Left")
            .addLobe("Occipital")
                .addRegion("V1")
                    .addNucleus("LGN")
                        .addColumn()
                            .addLayers(6, "Layer")
        .build();

    ASSERT_NE(brain, nullptr);

    // Navigate to column
    auto hemisphere = datastore_->getHemisphere(brain->getHemisphereId(0));
    auto lobe = datastore_->getLobe(hemisphere->getLobeId(0));
    auto region = datastore_->getRegion(lobe->getRegionId(0));
    auto nucleus = datastore_->getNucleus(region->getNucleusId(0));
    auto column = datastore_->getColumn(nucleus->getColumnId(0));

    ASSERT_NE(column, nullptr);
    EXPECT_EQ(column->size(), 6);  // 6 layers
}

/**
 * @brief Test cluster creation with neurons
 */
TEST_F(NetworkBuilderTest, ClusterWithNeurons) {
    NetworkBuilder builder(*factory_, *datastore_, false);

    auto brain = builder.createBrain("TestBrain")
        .addHemisphere("Left")
            .addLobe("Occipital")
                .addRegion("V1")
                    .addNucleus("LGN")
                        .addColumn()
                            .addLayer()
                                .addCluster(10)  // 10 neurons
        .build();

    ASSERT_NE(brain, nullptr);

    auto neurons = builder.getNeurons();
    EXPECT_EQ(neurons.size(), 10);

    auto clusters = builder.getClusters();
    EXPECT_EQ(clusters.size(), 1);
    EXPECT_EQ(clusters[0]->size(), 10);
}

/**
 * @brief Test bulk cluster creation
 */
TEST_F(NetworkBuilderTest, BulkClusters) {
    NetworkBuilder builder(*factory_, *datastore_, false);

    auto brain = builder.createBrain("TestBrain")
        .addHemisphere("Left")
            .addLobe("Occipital")
                .addRegion("V1")
                    .addNucleus("LGN")
                        .addColumn()
                            .addLayer()
                                .addClusters(3, 5)  // 3 clusters, 5 neurons each
        .build();

    ASSERT_NE(brain, nullptr);

    auto neurons = builder.getNeurons();
    EXPECT_EQ(neurons.size(), 15);  // 3 * 5 = 15

    auto clusters = builder.getClusters();
    EXPECT_EQ(clusters.size(), 3);
    for (const auto& cluster : clusters) {
        EXPECT_EQ(cluster->size(), 5);
    }
}

/**
 * @brief Test navigation with up()
 */
TEST_F(NetworkBuilderTest, NavigationUp) {
    NetworkBuilder builder(*factory_, *datastore_, false);

    auto brain = builder.createBrain("TestBrain")
        .addHemisphere("Left")
            .addLobe("Occipital")
                .addRegion("V1")
                    .addNucleus("LGN")
                        .addColumn()
                            .addLayer()
                                .addCluster(5)
                            .up()  // Back to column
                            .addLayer()
                                .addCluster(5)
        .build();

    ASSERT_NE(brain, nullptr);

    auto neurons = builder.getNeurons();
    EXPECT_EQ(neurons.size(), 10);  // 2 clusters * 5 neurons

    auto clusters = builder.getClusters();
    EXPECT_EQ(clusters.size(), 2);

    auto layers = builder.getLayers();
    EXPECT_EQ(layers.size(), 2);
}

/**
 * @brief Test navigation with toRoot()
 */
TEST_F(NetworkBuilderTest, NavigationToRoot) {
    NetworkBuilder builder(*factory_, *datastore_, false);

    auto brain = builder.createBrain("TestBrain")
        .addHemisphere("Left")
            .addLobe("Occipital")
                .addRegion("V1")
                    .addNucleus("LGN")
                        .addColumn()
                            .addLayer()
                                .addCluster(5)
                        .toRoot()  // Back to brain
        .addHemisphere("Right")
            .addLobe("Occipital")
                .addRegion("V1")
                    .addNucleus("LGN")
        .build();

    ASSERT_NE(brain, nullptr);
    EXPECT_EQ(brain->size(), 2);  // 2 hemispheres
}

/**
 * @brief Test auto-persist functionality
 */
TEST_F(NetworkBuilderTest, AutoPersist) {
    NetworkBuilder builder(*factory_, *datastore_, false);
    builder.setAutoPersist(true);

    auto brain = builder.createBrain("TestBrain")
        .addHemisphere("Left")
            .addLobe("Occipital")
                .addRegion("V1")
                    .addNucleus("LGN")
                        .addColumn()
                            .addLayer()
                                .addCluster(10)
        .build();

    ASSERT_NE(brain, nullptr);

    // Verify all objects are persisted in datastore
    auto brainFromDb = datastore_->getBrain(brain->getId());
    ASSERT_NE(brainFromDb, nullptr);
    EXPECT_EQ(brainFromDb->getName(), "TestBrain");

    auto hemisphere = datastore_->getHemisphere(brain->getHemisphereId(0));
    ASSERT_NE(hemisphere, nullptr);

    auto neurons = builder.getNeurons();
    for (const auto& neuron : neurons) {
        auto neuronFromDb = datastore_->getNeuron(neuron->getId());
        ASSERT_NE(neuronFromDb, nullptr);
    }
}

/**
 * @brief Test neuron parameter configuration
 */
TEST_F(NetworkBuilderTest, NeuronParameters) {
    NetworkBuilder builder(*factory_, *datastore_, false);
    builder.setNeuronParams(20.0, 0.8, 100);  // windowSize, threshold, maxPatterns

    auto brain = builder.createBrain("TestBrain")
        .addHemisphere("Left")
            .addLobe("Occipital")
                .addRegion("V1")
                    .addNucleus("LGN")
                        .addColumn()
                            .addLayer()
                                .addCluster(5)
        .build();

    ASSERT_NE(brain, nullptr);

    auto neurons = builder.getNeurons();
    EXPECT_EQ(neurons.size(), 5);

    // Verify neuron parameters
    for (const auto& neuron : neurons) {
        EXPECT_DOUBLE_EQ(neuron->getWindowSize(), 20.0);
        EXPECT_DOUBLE_EQ(neuron->getSimilarityThreshold(), 0.8);
        EXPECT_EQ(neuron->getMaxReferencePatterns(), 100);
    }
}

/**
 * @brief Test error handling - build without brain
 */
TEST_F(NetworkBuilderTest, ErrorNoBrain) {
    NetworkBuilder builder(*factory_, *datastore_, false);

    EXPECT_THROW(builder.build(), std::runtime_error);
}

/**
 * @brief Test error handling - add hemisphere without brain
 */
TEST_F(NetworkBuilderTest, ErrorNoContext) {
    NetworkBuilder builder(*factory_, *datastore_, false);

    EXPECT_THROW(builder.addHemisphere("Left"), std::runtime_error);
}

/**
 * @brief Test complex hierarchical structure
 */
TEST_F(NetworkBuilderTest, ComplexHierarchy) {
    NetworkBuilder builder(*factory_, *datastore_, false);

    auto brain = builder.createBrain("HumanBrain")
        .addHemisphere("Left")
            .addLobe("Occipital")
                .addRegion("V1")
                    .addNucleus("LGN")
                        .addColumns(12, "Orientation")
                            .addLayers(6, "Cortical")
                                .addClusters(10, 100)  // 10 clusters, 100 neurons each
        .build();

    ASSERT_NE(brain, nullptr);

    auto neurons = builder.getNeurons();
    EXPECT_EQ(neurons.size(), 1000);  // 10 * 100

    auto clusters = builder.getClusters();
    EXPECT_EQ(clusters.size(), 10);

    auto layers = builder.getLayers();
    EXPECT_EQ(layers.size(), 6);

    auto columns = builder.getColumns();
    EXPECT_EQ(columns.size(), 12);
}

/**
 * @brief Test multiple hemispheres
 */
TEST_F(NetworkBuilderTest, MultipleHemispheres) {
    NetworkBuilder builder(*factory_, *datastore_, false);

    auto brain = builder.createBrain("TestBrain")
        .addHemisphere("Left")
            .addLobe("Occipital")
                .addRegion("V1")
                    .addNucleus("LGN")
                        .addColumn()
                            .addLayer()
                                .addCluster(10)
        .toRoot()
        .addHemisphere("Right")
            .addLobe("Occipital")
                .addRegion("V1")
                    .addNucleus("LGN")
                        .addColumn()
                            .addLayer()
                                .addCluster(10)
        .build();

    ASSERT_NE(brain, nullptr);
    EXPECT_EQ(brain->size(), 2);  // 2 hemispheres

    auto neurons = builder.getNeurons();
    EXPECT_EQ(neurons.size(), 20);  // 2 * 10
}

/**
 * @brief Test getBrain() method
 */
TEST_F(NetworkBuilderTest, GetBrain) {
    NetworkBuilder builder(*factory_, *datastore_, false);

    builder.createBrain("TestBrain")
        .addHemisphere("Left");

    auto brain = builder.getBrain();
    ASSERT_NE(brain, nullptr);
    EXPECT_EQ(brain->getName(), "TestBrain");
}

/**
 * @brief Main function for running tests
 */
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

