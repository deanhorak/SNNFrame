/**
 * @file test_model_importer.cpp
 * @brief Unit tests for ModelImporter (CSV, SWC, NEST, NeuroML, HDF5)
 */

#include <gtest/gtest.h>
#include "snnfw/ModelImporter.h"
#include "snnfw/Datastore.h"
#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Neuron.h"
#include <fstream>
#include <filesystem>
#include <hdf5.h>

using namespace snnfw;

class ModelImporterTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Clean up test directory
        if (std::filesystem::exists("test_import_db")) {
            std::filesystem::remove_all("test_import_db");
        }
        
        datastore = std::make_unique<Datastore>("test_import_db", 1000);
        importer = std::make_unique<ModelImporter>(*datastore);
        factory = std::make_unique<NeuralObjectFactory>();
    }
    
    void TearDown() override {
        importer.reset();
        datastore.reset();
        
        // Clean up test files
        std::filesystem::remove("test_positions.csv");
        std::filesystem::remove("test_morphology.swc");
        std::filesystem::remove("test_nest.json");
        std::filesystem::remove("test_neuroml.nml");
        std::filesystem::remove("test_positions.h5");
        
        if (std::filesystem::exists("test_import_db")) {
            std::filesystem::remove_all("test_import_db");
        }
    }
    
    void createTestNeurons(const std::vector<uint64_t>& ids) {
        for (uint64_t id : ids) {
            auto neuron = factory->createNeuron(100.0, 0.8, 10);
            datastore->put(neuron);
        }
    }
    
    std::unique_ptr<Datastore> datastore;
    std::unique_ptr<ModelImporter> importer;
    std::unique_ptr<NeuralObjectFactory> factory;
};

// Test Position3D structure
TEST_F(ModelImporterTest, Position3DBasics) {
    Position3D p1(1.0f, 2.0f, 3.0f);
    Position3D p2(4.0f, 5.0f, 6.0f);
    
    EXPECT_FLOAT_EQ(p1.x, 1.0f);
    EXPECT_FLOAT_EQ(p1.y, 2.0f);
    EXPECT_FLOAT_EQ(p1.z, 3.0f);
    
    // Test distance
    float dist = p1.distanceTo(p2);
    EXPECT_NEAR(dist, 5.196f, 0.01f);  // sqrt(9 + 9 + 9)
    
    // Test operators
    Position3D p3 = p1 + p2;
    EXPECT_FLOAT_EQ(p3.x, 5.0f);
    EXPECT_FLOAT_EQ(p3.y, 7.0f);
    EXPECT_FLOAT_EQ(p3.z, 9.0f);
    
    Position3D p4 = p2 - p1;
    EXPECT_FLOAT_EQ(p4.x, 3.0f);
    EXPECT_FLOAT_EQ(p4.y, 3.0f);
    EXPECT_FLOAT_EQ(p4.z, 3.0f);
}

// Test CSV import
TEST_F(ModelImporterTest, ImportFromCSV) {
    // Create test CSV file
    std::ofstream csv("test_positions.csv");
    csv << "neuron_id,x,y,z\n";
    csv << "100000000000000,10.0,20.0,30.0\n";
    csv << "100000000000001,15.0,25.0,35.0\n";
    csv << "100000000000002,20.0,30.0,40.0\n";
    csv.close();
    
    // Create neurons
    createTestNeurons({100000000000000ULL, 100000000000001ULL, 100000000000002ULL});
    
    // Import
    ImportConfig config;
    auto result = importer->importFromCSV("test_positions.csv", config);
    
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.positionsSet, 3);
    EXPECT_EQ(result.neuronsImported, 3);
    
    // Verify positions
    Position3D pos;
    EXPECT_TRUE(importer->getNeuronPosition(100000000000000ULL, pos));
    EXPECT_FLOAT_EQ(pos.x, 10.0f);
    EXPECT_FLOAT_EQ(pos.y, 20.0f);
    EXPECT_FLOAT_EQ(pos.z, 30.0f);
}

// Test CSV import with transformation
TEST_F(ModelImporterTest, ImportFromCSVWithTransformation) {
    std::ofstream csv("test_positions.csv");
    csv << "100000000000000,100.0,200.0,50.0\n";
    csv.close();
    
    createTestNeurons({100000000000000ULL});
    
    ImportConfig config;
    config.positionScale = 0.001f;  // Scale down
    config.offsetX = 10.0f;
    config.offsetY = 20.0f;
    config.offsetZ = 5.0f;
    
    auto result = importer->importFromCSV("test_positions.csv", config);
    
    EXPECT_TRUE(result.success);
    
    Position3D pos;
    EXPECT_TRUE(importer->getNeuronPosition(100000000000000ULL, pos));
    EXPECT_FLOAT_EQ(pos.x, 10.1f);   // (100 * 0.001) + 10
    EXPECT_FLOAT_EQ(pos.y, 20.2f);   // (200 * 0.001) + 20
    EXPECT_FLOAT_EQ(pos.z, 5.05f);   // (50 * 0.001) + 5
}

// Test SWC import
TEST_F(ModelImporterTest, ImportFromSWC) {
    // Use neuron IDs that match the base neuron ID range
    uint64_t baseId = 100000000000000ULL;

    std::ofstream swc("test_morphology.swc");
    swc << "# NEURON morphology file\n";
    swc << baseId << " 1 0.0 0.0 0.0 1.0 -1\n";
    swc << (baseId + 1) << " 3 10.0 5.0 2.0 0.5 " << baseId << "\n";
    swc << (baseId + 2) << " 3 20.0 10.0 4.0 0.5 " << (baseId + 1) << "\n";
    swc.close();

    createTestNeurons({baseId, baseId + 1, baseId + 2});

    ImportConfig config;
    auto result = importer->importFromSWC("test_morphology.swc", config);

    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.positionsSet, 3);

    Position3D pos;
    EXPECT_TRUE(importer->getNeuronPosition(baseId + 1, pos));
    EXPECT_FLOAT_EQ(pos.x, 10.0f);
    EXPECT_FLOAT_EQ(pos.y, 5.0f);
    EXPECT_FLOAT_EQ(pos.z, 2.0f);
}

// Test NEST JSON import
TEST_F(ModelImporterTest, ImportFromNEST) {
    std::ofstream json("test_nest.json");
    json << R"({
        "neurons": [
            {"id": 100000000000000, "position": [10.0, 20.0, 30.0]},
            {"id": 100000000000001, "position": [15.0, 25.0, 35.0]}
        ]
    })";
    json.close();
    
    createTestNeurons({100000000000000ULL, 100000000000001ULL});
    
    ImportConfig config;
    auto result = importer->importFromNEST("test_nest.json", config);
    
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.positionsSet, 2);
    
    Position3D pos;
    EXPECT_TRUE(importer->getNeuronPosition(100000000000001ULL, pos));
    EXPECT_FLOAT_EQ(pos.x, 15.0f);
    EXPECT_FLOAT_EQ(pos.y, 25.0f);
    EXPECT_FLOAT_EQ(pos.z, 35.0f);
}

// Test NeuroML import
TEST_F(ModelImporterTest, ImportFromNeuroML) {
    std::ofstream nml("test_neuroml.nml");
    nml << R"(<?xml version="1.0" encoding="UTF-8"?>
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" id="TestNetwork">
    <network id="Net1">
        <population id="Pop0" component="cell" size="3">
            <instance id="100000000000000">
                <location x="10.0" y="20.0" z="30.0"/>
            </instance>
            <instance id="100000000000001">
                <location x="15.0" y="25.0" z="35.0"/>
            </instance>
            <instance id="100000000000002">
                <location x="20.0" y="30.0" z="40.0"/>
            </instance>
        </population>
    </network>
</neuroml>
)";
    nml.close();
    
    createTestNeurons({100000000000000ULL, 100000000000001ULL, 100000000000002ULL});
    
    ImportConfig config;
    auto result = importer->importFromNeuroML("test_neuroml.nml", config);
    
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.positionsSet, 3);
    EXPECT_EQ(result.neuronsImported, 3);
    
    Position3D pos;
    EXPECT_TRUE(importer->getNeuronPosition(100000000000000ULL, pos));
    EXPECT_FLOAT_EQ(pos.x, 10.0f);
    EXPECT_FLOAT_EQ(pos.y, 20.0f);
    EXPECT_FLOAT_EQ(pos.z, 30.0f);
    
    EXPECT_TRUE(importer->getNeuronPosition(100000000000002ULL, pos));
    EXPECT_FLOAT_EQ(pos.x, 20.0f);
    EXPECT_FLOAT_EQ(pos.y, 30.0f);
    EXPECT_FLOAT_EQ(pos.z, 40.0f);
}

// Test HDF5 import (generic format)
TEST_F(ModelImporterTest, ImportFromHDF5Generic) {
    // Create HDF5 file
    hid_t file_id = H5Fcreate("test_positions.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    
    // Create positions dataset
    hsize_t dims[2] = {3, 3};
    hid_t dataspace_id = H5Screate_simple(2, dims, nullptr);
    hid_t dataset_id = H5Dcreate(file_id, "/positions", H5T_NATIVE_FLOAT,
                                 dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    float positions[3][3] = {
        {10.0f, 20.0f, 30.0f},
        {15.0f, 25.0f, 35.0f},
        {20.0f, 30.0f, 40.0f}
    };
    
    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, positions);
    
    // Create neuron IDs dataset
    hid_t id_dataspace = H5Screate_simple(1, &dims[0], nullptr);
    hid_t id_dataset = H5Dcreate(file_id, "/neuron_ids", H5T_NATIVE_UINT64,
                                 id_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    uint64_t neuron_ids[3] = {100000000000000ULL, 100000000000001ULL, 100000000000002ULL};
    H5Dwrite(id_dataset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, neuron_ids);
    
    H5Dclose(id_dataset);
    H5Sclose(id_dataspace);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);
    
    // Create neurons
    createTestNeurons({100000000000000ULL, 100000000000001ULL, 100000000000002ULL});
    
    // Import
    ImportConfig config;
    auto result = importer->importFromHDF5("test_positions.h5", config);
    
    EXPECT_TRUE(result.success);
    EXPECT_EQ(result.positionsSet, 3);
    EXPECT_EQ(result.neuronsImported, 3);
    
    Position3D pos;
    EXPECT_TRUE(importer->getNeuronPosition(100000000000001ULL, pos));
    EXPECT_FLOAT_EQ(pos.x, 15.0f);
    EXPECT_FLOAT_EQ(pos.y, 25.0f);
    EXPECT_FLOAT_EQ(pos.z, 35.0f);
}

// Test export to CSV
TEST_F(ModelImporterTest, ExportToCSV) {
    // Create neurons with positions
    createTestNeurons({100000000000000ULL, 100000000000001ULL});
    
    importer->setNeuronPosition(100000000000000ULL, 10.0f, 20.0f, 30.0f);
    importer->setNeuronPosition(100000000000001ULL, 15.0f, 25.0f, 35.0f);
    
    // Export
    std::vector<uint64_t> ids = {100000000000000ULL, 100000000000001ULL};
    EXPECT_TRUE(importer->exportToCSV("test_positions.csv", ids));
    
    // Verify file contents
    std::ifstream csv("test_positions.csv");
    std::string line;
    
    std::getline(csv, line);  // Header
    EXPECT_EQ(line, "neuron_id,x,y,z");
    
    std::getline(csv, line);
    EXPECT_TRUE(line.find("100000000000000") != std::string::npos);
    EXPECT_TRUE(line.find("10") != std::string::npos);
    
    csv.close();
}

// Test error handling - file not found
TEST_F(ModelImporterTest, ErrorHandlingFileNotFound) {
    ImportConfig config;
    auto result = importer->importFromCSV("nonexistent.csv", config);
    
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.errorMessage.empty());
}

// Test error handling - invalid NeuroML
TEST_F(ModelImporterTest, ErrorHandlingInvalidNeuroML) {
    std::ofstream nml("test_neuroml.nml");
    nml << "This is not valid XML";
    nml.close();
    
    ImportConfig config;
    auto result = importer->importFromNeuroML("test_neuroml.nml", config);
    
    EXPECT_FALSE(result.success);
    EXPECT_FALSE(result.errorMessage.empty());
}

// Test direct position manipulation
TEST_F(ModelImporterTest, DirectPositionManipulation) {
    createTestNeurons({100000000000000ULL});
    
    // Set position
    EXPECT_TRUE(importer->setNeuronPosition(100000000000000ULL, 10.0f, 20.0f, 30.0f));
    
    // Get position
    Position3D pos;
    EXPECT_TRUE(importer->getNeuronPosition(100000000000000ULL, pos));
    EXPECT_FLOAT_EQ(pos.x, 10.0f);
    EXPECT_FLOAT_EQ(pos.y, 20.0f);
    EXPECT_FLOAT_EQ(pos.z, 30.0f);
    
    // Try to set position for non-existent neuron
    EXPECT_FALSE(importer->setNeuronPosition(999999ULL, 0.0f, 0.0f, 0.0f));
    
    // Try to get position for non-existent neuron
    EXPECT_FALSE(importer->getNeuronPosition(999999ULL, pos));
}

