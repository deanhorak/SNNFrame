/**
 * @file neuroml_hdf5_import_example.cpp
 * @brief Example demonstrating NeuroML and HDF5 position import
 * 
 * This example shows:
 * 1. Creating sample NeuroML and HDF5 files
 * 2. Importing positions from NeuroML XML format
 * 3. Importing positions from HDF5 format (SONATA, NWB, generic)
 * 4. Verifying imported positions
 */

#include "snnfw/Datastore.h"
#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Neuron.h"
#include "snnfw/ModelImporter.h"
#include "snnfw/Logger.h"
#include <iostream>
#include <fstream>
#include <hdf5.h>

using namespace snnfw;

// Helper function to create a sample NeuroML file
void createSampleNeuroMLFile(const std::string& filename) {
    std::ofstream file(filename);
    file << R"(<?xml version="1.0" encoding="UTF-8"?>
<neuroml xmlns="http://www.neuroml.org/schema/neuroml2" 
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://www.neuroml.org/schema/neuroml2 https://raw.github.com/NeuroML/NeuroML2/development/Schemas/NeuroML2/NeuroML_v2.3.xsd"
         id="SampleNetwork">
    
    <izhikevich2007Cell id="izh2007RS0" C="100pF" v0="-60mV" k="0.7nS_per_mV" 
                        vr="-60mV" vt="-40mV" vpeak="35mV" 
                        a="0.03per_ms" b="-2nS" c="-50.0mV" d="100pA"/>
    
    <network id="Network1">
        <population id="Pop0" component="izh2007RS0" size="5">
            <instance id="100000000000000">
                <location x="0.0" y="0.0" z="0.0"/>
            </instance>
            <instance id="100000000000001">
                <location x="10.0" y="5.0" z="2.0"/>
            </instance>
            <instance id="100000000000002">
                <location x="20.0" y="10.0" z="4.0"/>
            </instance>
            <instance id="100000000000003">
                <location x="30.0" y="15.0" z="6.0"/>
            </instance>
            <instance id="100000000000004">
                <location x="40.0" y="20.0" z="8.0"/>
            </instance>
        </population>
    </network>
</neuroml>
)";
    file.close();
    std::cout << "  ✓ Created sample NeuroML file: " << filename << std::endl;
}

// Helper function to create a sample HDF5 file
void createSampleHDF5File(const std::string& filename) {
    // Create HDF5 file
    hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    
    // Create positions dataset (5 neurons x 3 coordinates)
    hsize_t dims[2] = {5, 3};
    hid_t dataspace_id = H5Screate_simple(2, dims, nullptr);
    hid_t dataset_id = H5Dcreate(file_id, "/positions", H5T_NATIVE_FLOAT, 
                                 dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    // Sample position data
    float positions[5][3] = {
        {100.0f, 200.0f, 50.0f},
        {110.0f, 205.0f, 52.0f},
        {120.0f, 210.0f, 54.0f},
        {130.0f, 215.0f, 56.0f},
        {140.0f, 220.0f, 58.0f}
    };
    
    H5Dwrite(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, positions);
    
    // Create neuron IDs dataset
    hid_t id_dataspace = H5Screate_simple(1, &dims[0], nullptr);
    hid_t id_dataset = H5Dcreate(file_id, "/neuron_ids", H5T_NATIVE_UINT64,
                                 id_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    uint64_t neuron_ids[5] = {
        100000000000005ULL,
        100000000000006ULL,
        100000000000007ULL,
        100000000000008ULL,
        100000000000009ULL
    };
    
    H5Dwrite(id_dataset, H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, neuron_ids);
    
    // Close resources
    H5Dclose(id_dataset);
    H5Sclose(id_dataspace);
    H5Dclose(dataset_id);
    H5Sclose(dataspace_id);
    H5Fclose(file_id);
    
    std::cout << "  ✓ Created sample HDF5 file: " << filename << std::endl;
}

// Helper function to create neurons in datastore
void createNeurons(Datastore& datastore, const std::vector<uint64_t>& neuronIds) {
    NeuralObjectFactory factory;
    
    for (uint64_t id : neuronIds) {
        auto neuron = factory.createNeuron(100.0, 0.8, 10);
        // Override the auto-generated ID with our specific ID
        auto neuronPtr = std::const_pointer_cast<Neuron>(neuron);
        // Note: We can't directly set the ID, so we'll create neurons and then
        // the importer will set positions on existing neurons by ID
        datastore.put(neuron);
    }
}

int main() {
    // Initialize logging
    Logger::getInstance().initialize("neuroml_hdf5_import.log", spdlog::level::info);
    
    std::cout << "=== SNNFW NeuroML and HDF5 Import Example ===" << std::endl << std::endl;
    
    // Initialize datastore
    Datastore datastore("./import_example_db", 10000);
    NeuralObjectFactory factory;
    
    // Part 1: Create sample files
    std::cout << "Part 1: Creating sample import files..." << std::endl;
    createSampleNeuroMLFile("sample_network.nml");
    createSampleHDF5File("sample_positions.h5");
    std::cout << std::endl;
    
    // Part 2: Create neurons for NeuroML import
    std::cout << "Part 2: Creating neurons for NeuroML import..." << std::endl;
    std::vector<uint64_t> neuromlNeuronIds = {
        100000000000000ULL,
        100000000000001ULL,
        100000000000002ULL,
        100000000000003ULL,
        100000000000004ULL
    };
    
    for (uint64_t id : neuromlNeuronIds) {
        auto neuron = factory.createNeuron(100.0, 0.8, 10);
        datastore.put(neuron);
    }
    std::cout << "  ✓ Created " << neuromlNeuronIds.size() << " neurons" << std::endl << std::endl;
    
    // Part 3: Import from NeuroML
    std::cout << "Part 3: Importing positions from NeuroML..." << std::endl;
    ModelImporter importer(datastore);
    
    ImportConfig neuromlConfig;
    neuromlConfig.positionScale = 1.0f;
    
    auto neuromlResult = importer.importFromNeuroML("sample_network.nml", neuromlConfig);
    
    if (neuromlResult.success) {
        std::cout << "  ✓ Successfully imported from NeuroML" << std::endl;
        std::cout << "    - Positions set: " << neuromlResult.positionsSet << std::endl;
        std::cout << "    - Neurons imported: " << neuromlResult.neuronsImported << std::endl;
    } else {
        std::cout << "  ✗ Failed to import from NeuroML: " << neuromlResult.errorMessage << std::endl;
    }
    std::cout << std::endl;
    
    // Part 4: Verify NeuroML positions
    std::cout << "Part 4: Verifying NeuroML positions..." << std::endl;
    int neuromlVerified = 0;
    for (uint64_t id : neuromlNeuronIds) {
        Position3D pos;
        if (importer.getNeuronPosition(id, pos)) {
            std::cout << "  ✓ Neuron " << id << ": (" 
                     << pos.x << ", " << pos.y << ", " << pos.z << ")" << std::endl;
            neuromlVerified++;
        }
    }
    std::cout << "  ✓ Verified " << neuromlVerified << "/" << neuromlNeuronIds.size() 
             << " NeuroML positions" << std::endl << std::endl;
    
    // Part 5: Create neurons for HDF5 import
    std::cout << "Part 5: Creating neurons for HDF5 import..." << std::endl;
    std::vector<uint64_t> hdf5NeuronIds = {
        100000000000005ULL,
        100000000000006ULL,
        100000000000007ULL,
        100000000000008ULL,
        100000000000009ULL
    };
    
    for (uint64_t id : hdf5NeuronIds) {
        auto neuron = factory.createNeuron(100.0, 0.8, 10);
        datastore.put(neuron);
    }
    std::cout << "  ✓ Created " << hdf5NeuronIds.size() << " neurons" << std::endl << std::endl;
    
    // Part 6: Import from HDF5
    std::cout << "Part 6: Importing positions from HDF5..." << std::endl;
    
    ImportConfig hdf5Config;
    hdf5Config.positionScale = 0.001f;  // Convert μm to mm
    hdf5Config.offsetX = 50.0f;
    hdf5Config.offsetY = 100.0f;
    hdf5Config.offsetZ = 25.0f;
    
    auto hdf5Result = importer.importFromHDF5("sample_positions.h5", hdf5Config);
    
    if (hdf5Result.success) {
        std::cout << "  ✓ Successfully imported from HDF5" << std::endl;
        std::cout << "    - Positions set: " << hdf5Result.positionsSet << std::endl;
        std::cout << "    - Neurons imported: " << hdf5Result.neuronsImported << std::endl;
    } else {
        std::cout << "  ✗ Failed to import from HDF5: " << hdf5Result.errorMessage << std::endl;
    }
    std::cout << std::endl;
    
    // Part 7: Verify HDF5 positions
    std::cout << "Part 7: Verifying HDF5 positions (with transformation)..." << std::endl;
    int hdf5Verified = 0;
    for (uint64_t id : hdf5NeuronIds) {
        Position3D pos;
        if (importer.getNeuronPosition(id, pos)) {
            std::cout << "  ✓ Neuron " << id << ": (" 
                     << pos.x << ", " << pos.y << ", " << pos.z << ")" << std::endl;
            hdf5Verified++;
        }
    }
    std::cout << "  ✓ Verified " << hdf5Verified << "/" << hdf5NeuronIds.size() 
             << " HDF5 positions" << std::endl << std::endl;
    
    // Summary
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "✓ NeuroML import: " << neuromlResult.positionsSet << " positions" << std::endl;
    std::cout << "✓ HDF5 import: " << hdf5Result.positionsSet << " positions" << std::endl;
    std::cout << "✓ Total positions imported: " 
             << (neuromlResult.positionsSet + hdf5Result.positionsSet) << std::endl;
    std::cout << std::endl;
    
    std::cout << "Next steps:" << std::endl;
    std::cout << "  - Use these positions in NetworkDataAdapter for visualization" << std::endl;
    std::cout << "  - Import real NeuroML models from NeuroML-DB" << std::endl;
    std::cout << "  - Import SONATA/NWB datasets from neuroscience repositories" << std::endl;
    
    return 0;
}

