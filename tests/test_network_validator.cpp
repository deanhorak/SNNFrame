#include <gtest/gtest.h>
#include "snnfw/NetworkValidator.h"
#include "snnfw/Datastore.h"
#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Neuron.h"
#include "snnfw/Synapse.h"
#include "snnfw/Axon.h"
#include "snnfw/Dendrite.h"
#include "snnfw/Cluster.h"
#include "snnfw/Layer.h"
#include "snnfw/Column.h"
#include "snnfw/Nucleus.h"
#include "snnfw/Region.h"
#include "snnfw/Lobe.h"
#include "snnfw/Hemisphere.h"
#include "snnfw/Brain.h"
#include <memory>
#include <filesystem>

using namespace snnfw;

class NetworkValidatorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary directory for the test datastore
        testDbPath = std::filesystem::temp_directory_path() / "test_validator_db";
        std::filesystem::remove_all(testDbPath);
        
        datastore = std::make_unique<Datastore>(testDbPath.string());
        factory = std::make_unique<NeuralObjectFactory>();
        validator = std::make_unique<NetworkValidator>();
    }
    
    void TearDown() override {
        validator.reset();
        factory.reset();
        datastore.reset();
        std::filesystem::remove_all(testDbPath);
    }
    
    std::filesystem::path testDbPath;
    std::unique_ptr<Datastore> datastore;
    std::unique_ptr<NeuralObjectFactory> factory;
    std::unique_ptr<NetworkValidator> validator;
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(NetworkValidatorTest, ConstructorAndDefaults) {
    NetworkValidator v;
    
    const auto& config = v.getConfig();
    EXPECT_TRUE(config.checkIdExistence);
    EXPECT_TRUE(config.checkConnectivity);
    EXPECT_TRUE(config.checkHierarchy);
    EXPECT_TRUE(config.checkResourceLimits);
    EXPECT_FALSE(config.checkCycles);  // Expensive, disabled by default
}

TEST_F(NetworkValidatorTest, CustomConfiguration) {
    ValidationConfig config;
    config.checkCycles = true;
    config.maxSynapsesPerNeuron = 50000;
    config.verbose = true;
    
    NetworkValidator v(config);
    
    const auto& retrievedConfig = v.getConfig();
    EXPECT_TRUE(retrievedConfig.checkCycles);
    EXPECT_EQ(retrievedConfig.maxSynapsesPerNeuron, 50000);
    EXPECT_TRUE(retrievedConfig.verbose);
}

TEST_F(NetworkValidatorTest, SetConfiguration) {
    NetworkValidator v;
    
    ValidationConfig config;
    config.checkConnectivity = false;
    v.setConfig(config);
    
    const auto& retrievedConfig = v.getConfig();
    EXPECT_FALSE(retrievedConfig.checkConnectivity);
}

// ============================================================================
// ID Existence Tests
// ============================================================================

TEST_F(NetworkValidatorTest, CheckIdExistsNonExistent) {
    EXPECT_FALSE(validator->checkIdExists(12345, *datastore));
}

TEST_F(NetworkValidatorTest, CheckIdExistsZero) {
    EXPECT_FALSE(validator->checkIdExists(0, *datastore));
}

TEST_F(NetworkValidatorTest, CheckIdExistsValid) {
    auto neuron = factory->createNeuron(100, 0.85, 10);
    datastore->put(neuron);
    
    EXPECT_TRUE(validator->checkIdExists(neuron->getId(), *datastore));
}

// ============================================================================
// Neuron Validation Tests
// ============================================================================

TEST_F(NetworkValidatorTest, ValidateNeuronNonExistent) {
    auto result = validator->validateNeuron(12345, *datastore);
    
    EXPECT_FALSE(result.isValid);
    EXPECT_EQ(result.criticalCount, 1);
    EXPECT_FALSE(result.errors.empty());
    EXPECT_EQ(result.errors[0].type, ValidationErrorType::MISSING_ID);
}

TEST_F(NetworkValidatorTest, ValidateNeuronValid) {
    auto neuron = factory->createNeuron(100, 0.85, 10);
    datastore->put(neuron);
    
    auto result = validator->validateNeuron(neuron->getId(), *datastore);
    
    EXPECT_TRUE(result.isValid);
    EXPECT_EQ(result.criticalCount, 0);
    EXPECT_EQ(result.errorCount, 0);
}

TEST_F(NetworkValidatorTest, ValidateNeuronWithAxon) {
    auto neuron = factory->createNeuron(100, 0.85, 10);
    auto axon = factory->createAxon(neuron->getId());
    neuron->setAxonId(axon->getId());
    
    datastore->put(neuron);
    datastore->put(axon);
    
    auto result = validator->validateNeuron(neuron->getId(), *datastore);
    
    EXPECT_TRUE(result.isValid);
}

TEST_F(NetworkValidatorTest, ValidateNeuronWithMissingAxon) {
    auto neuron = factory->createNeuron(100, 0.85, 10);
    neuron->setAxonId(999999);  // Non-existent axon
    
    datastore->put(neuron);
    
    auto result = validator->validateNeuron(neuron->getId(), *datastore);
    
    EXPECT_FALSE(result.isValid);
    EXPECT_GT(result.errorCount, 0);
    
    bool foundDanglingRef = false;
    for (const auto& error : result.errors) {
        if (error.type == ValidationErrorType::DANGLING_REFERENCE) {
            foundDanglingRef = true;
            break;
        }
    }
    EXPECT_TRUE(foundDanglingRef);
}

TEST_F(NetworkValidatorTest, ValidateNeuronWithDendrites) {
    auto neuron = factory->createNeuron(100, 0.85, 10);
    auto dendrite1 = factory->createDendrite(neuron->getId());
    auto dendrite2 = factory->createDendrite(neuron->getId());

    neuron->addDendrite(dendrite1->getId());
    neuron->addDendrite(dendrite2->getId());

    datastore->put(neuron);
    datastore->put(dendrite1);
    datastore->put(dendrite2);

    auto result = validator->validateNeuron(neuron->getId(), *datastore);

    EXPECT_TRUE(result.isValid);
}

TEST_F(NetworkValidatorTest, ValidateNeuronWithMissingDendrite) {
    auto neuron = factory->createNeuron(100, 0.85, 10);
    neuron->addDendrite(999999);  // Non-existent dendrite

    datastore->put(neuron);

    auto result = validator->validateNeuron(neuron->getId(), *datastore);

    EXPECT_FALSE(result.isValid);
    EXPECT_GT(result.errorCount, 0);
}

// ============================================================================
// Synapse Validation Tests
// ============================================================================

TEST_F(NetworkValidatorTest, ValidateSynapseNonExistent) {
    auto result = validator->validateSynapse(12345, *datastore);
    
    EXPECT_FALSE(result.isValid);
    EXPECT_EQ(result.criticalCount, 1);
}

TEST_F(NetworkValidatorTest, ValidateSynapseValid) {
    auto neuron1 = factory->createNeuron(100, 0.85, 10);
    auto neuron2 = factory->createNeuron(100, 0.85, 10);
    auto axon = factory->createAxon(neuron1->getId());
    auto dendrite = factory->createDendrite(neuron2->getId());
    auto synapse = factory->createSynapse(axon->getId(), dendrite->getId(), 1.0, 1.0);
    
    datastore->put(neuron1);
    datastore->put(neuron2);
    datastore->put(axon);
    datastore->put(dendrite);
    datastore->put(synapse);
    
    auto result = validator->validateSynapse(synapse->getId(), *datastore);
    
    EXPECT_TRUE(result.isValid);
}

TEST_F(NetworkValidatorTest, ValidateSynapseWithMissingAxon) {
    auto neuron = factory->createNeuron(100, 0.85, 10);
    auto dendrite = factory->createDendrite(neuron->getId());
    auto synapse = factory->createSynapse(999999, dendrite->getId(), 1.0, 1.0);
    
    datastore->put(neuron);
    datastore->put(dendrite);
    datastore->put(synapse);
    
    auto result = validator->validateSynapse(synapse->getId(), *datastore);
    
    EXPECT_FALSE(result.isValid);
    EXPECT_GT(result.errorCount, 0);
}

TEST_F(NetworkValidatorTest, ValidateSynapseWithMissingDendrite) {
    auto neuron = factory->createNeuron(100, 0.85, 10);
    auto axon = factory->createAxon(neuron->getId());
    auto synapse = factory->createSynapse(axon->getId(), 999999, 1.0, 1.0);
    
    datastore->put(neuron);
    datastore->put(axon);
    datastore->put(synapse);
    
    auto result = validator->validateSynapse(synapse->getId(), *datastore);
    
    EXPECT_FALSE(result.isValid);
    EXPECT_GT(result.errorCount, 0);
}

// ============================================================================
// Hierarchy Validation Tests
// ============================================================================

TEST_F(NetworkValidatorTest, ValidateHierarchyNonExistent) {
    auto result = validator->validateHierarchy(12345, *datastore);
    
    EXPECT_FALSE(result.isValid);
    EXPECT_EQ(result.criticalCount, 1);
}

TEST_F(NetworkValidatorTest, ValidateHierarchyCluster) {
    auto cluster = factory->createCluster();
    datastore->put(cluster);

    auto result = validator->validateHierarchy(cluster->getId(), *datastore);

    EXPECT_TRUE(result.isValid);
}

TEST_F(NetworkValidatorTest, ValidateHierarchyClusterWithNeurons) {
    auto cluster = factory->createCluster();
    auto neuron1 = factory->createNeuron(100, 0.85, 10);
    auto neuron2 = factory->createNeuron(100, 0.85, 10);

    cluster->addNeuron(neuron1->getId());
    cluster->addNeuron(neuron2->getId());

    datastore->put(cluster);
    datastore->put(neuron1);
    datastore->put(neuron2);

    auto result = validator->validateHierarchy(cluster->getId(), *datastore);

    EXPECT_TRUE(result.isValid);
}

TEST_F(NetworkValidatorTest, ValidateHierarchyClusterWithMissingNeuron) {
    auto cluster = factory->createCluster();
    cluster->addNeuron(999999);  // Non-existent neuron

    datastore->put(cluster);

    auto result = validator->validateHierarchy(cluster->getId(), *datastore);

    EXPECT_FALSE(result.isValid);
    EXPECT_GT(result.errorCount, 0);
}

TEST_F(NetworkValidatorTest, ValidateHierarchyLayerWithClusters) {
    auto layer = factory->createLayer();
    auto cluster1 = factory->createCluster();
    auto cluster2 = factory->createCluster();

    layer->addCluster(cluster1->getId());
    layer->addCluster(cluster2->getId());

    datastore->put(layer);
    datastore->put(cluster1);
    datastore->put(cluster2);

    auto result = validator->validateHierarchy(layer->getId(), *datastore);

    EXPECT_TRUE(result.isValid);
}

TEST_F(NetworkValidatorTest, ValidateHierarchyColumnWithLayers) {
    auto column = factory->createColumn();
    auto layer1 = factory->createLayer();
    auto layer2 = factory->createLayer();

    column->addLayer(layer1->getId());
    column->addLayer(layer2->getId());

    datastore->put(column);
    datastore->put(layer1);
    datastore->put(layer2);

    auto result = validator->validateHierarchy(column->getId(), *datastore);

    EXPECT_TRUE(result.isValid);
}

// ============================================================================
// Network Validation Tests
// ============================================================================

TEST_F(NetworkValidatorTest, ValidateNetworkNonExistentRoot) {
    auto result = validator->validateNetwork(12345, *datastore);
    
    EXPECT_FALSE(result.isValid);
    EXPECT_EQ(result.criticalCount, 1);
}

TEST_F(NetworkValidatorTest, ValidateNetworkSimple) {
    auto brain = factory->createBrain();
    datastore->put(brain);

    auto result = validator->validateNetwork(brain->getId(), *datastore);

    EXPECT_TRUE(result.isValid);
}

TEST_F(NetworkValidatorTest, ValidateNetworkWithHierarchy) {
    auto brain = factory->createBrain();
    auto hemisphere = factory->createHemisphere();
    auto lobe = factory->createLobe();

    brain->addHemisphere(hemisphere->getId());
    hemisphere->addLobe(lobe->getId());

    datastore->put(brain);
    datastore->put(hemisphere);
    datastore->put(lobe);

    auto result = validator->validateNetwork(brain->getId(), *datastore);

    EXPECT_TRUE(result.isValid);
}

// ============================================================================
// Validation Result Tests
// ============================================================================

TEST_F(NetworkValidatorTest, ValidationResultSummary) {
    ValidationResult result;
    result.addError(ValidationError(ValidationSeverity::CRITICAL, ValidationErrorType::MISSING_ID, "Test critical"));
    result.addError(ValidationError(ValidationSeverity::ERROR, ValidationErrorType::DANGLING_REFERENCE, "Test error"));
    result.addError(ValidationError(ValidationSeverity::WARNING, ValidationErrorType::RESOURCE_LIMIT_EXCEEDED, "Test warning"));
    result.addError(ValidationError(ValidationSeverity::INFO, ValidationErrorType::UNKNOWN_ERROR, "Test info"));
    
    EXPECT_FALSE(result.isValid);
    EXPECT_EQ(result.criticalCount, 1);
    EXPECT_EQ(result.errorCount, 1);
    EXPECT_EQ(result.warningCount, 1);
    EXPECT_EQ(result.infoCount, 1);
    
    std::string summary = result.getSummary();
    EXPECT_NE(summary.find("FAILED"), std::string::npos);
    EXPECT_NE(summary.find("1 critical"), std::string::npos);
}

TEST_F(NetworkValidatorTest, ValidationResultDetailedReport) {
    ValidationResult result;
    result.addError(ValidationError(ValidationSeverity::ERROR, ValidationErrorType::MISSING_ID, "Test error", 12345, "Neuron"));
    
    std::string report = result.getDetailedReport();
    EXPECT_NE(report.find("ERROR"), std::string::npos);
    EXPECT_NE(report.find("Neuron"), std::string::npos);
    EXPECT_NE(report.find("12345"), std::string::npos);
    EXPECT_NE(report.find("Test error"), std::string::npos);
}

// Main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

