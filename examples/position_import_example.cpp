/**
 * @file position_import_example.cpp
 * @brief Example demonstrating 3D position storage and import functionality
 * 
 * This example shows how to:
 * 1. Create neurons with 3D positions
 * 2. Import positions from CSV files
 * 3. Export positions to CSV files
 * 4. Use stored positions in visualization
 */

#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Datastore.h"
#include "snnfw/ModelImporter.h"
#include "snnfw/Neuron.h"
#include "snnfw/Logger.h"
#include <iostream>
#include <fstream>
#include <filesystem>

using namespace snnfw;

int main() {
    // Initialize logging
    Logger::getInstance().initialize("position_import.log", spdlog::level::info);
    
    std::cout << "=== SNNFW Position Import/Export Example ===" << std::endl;
    std::cout << std::endl;
    
    // Create datastore
    std::string dbPath = "./position_example_db";
    if (std::filesystem::exists(dbPath)) {
        std::filesystem::remove_all(dbPath);
    }
    
    Datastore datastore(dbPath, 10000);
    NeuralObjectFactory factory;
    
    // ========================================================================
    // Part 1: Create neurons and set positions programmatically
    // ========================================================================
    std::cout << "Part 1: Creating neurons with positions..." << std::endl;
    
    std::vector<uint64_t> neuronIds;
    
    // Create 10 neurons in a line
    for (int i = 0; i < 10; ++i) {
        auto neuron = factory.createNeuron(100.0, 0.8, 10);
        uint64_t neuronId = neuron->getId();
        
        // Set position (neurons arranged in a line along X axis)
        neuron->setPosition(i * 10.0f, 0.0f, 0.0f);
        
        datastore.put(neuron);
        neuronIds.push_back(neuronId);
        
        std::cout << "  Created neuron " << neuronId 
                  << " at position (" << i * 10.0f << ", 0, 0)" << std::endl;
    }
    
    std::cout << "  ✓ Created " << neuronIds.size() << " neurons with positions" << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // Part 2: Export positions to CSV
    // ========================================================================
    std::cout << "Part 2: Exporting positions to CSV..." << std::endl;
    
    ModelImporter importer(datastore);
    std::string exportFile = "neuron_positions.csv";
    
    if (importer.exportToCSV(exportFile, neuronIds)) {
        std::cout << "  ✓ Exported positions to " << exportFile << std::endl;
        
        // Show file contents
        std::ifstream file(exportFile);
        std::string line;
        std::cout << "  File contents:" << std::endl;
        while (std::getline(file, line)) {
            std::cout << "    " << line << std::endl;
        }
    } else {
        std::cout << "  ✗ Failed to export positions" << std::endl;
    }
    std::cout << std::endl;
    
    // ========================================================================
    // Part 3: Clear positions and re-import from CSV
    // ========================================================================
    std::cout << "Part 3: Clearing positions and re-importing..." << std::endl;
    
    // Clear positions
    for (uint64_t neuronId : neuronIds) {
        auto neuron = datastore.getNeuron(neuronId);
        if (neuron) {
            neuron->clearPosition();
            datastore.put(neuron);
        }
    }
    std::cout << "  ✓ Cleared all positions" << std::endl;
    
    // Verify positions are cleared
    int clearedCount = 0;
    for (uint64_t neuronId : neuronIds) {
        auto neuron = datastore.getNeuron(neuronId);
        if (neuron && !neuron->hasPosition()) {
            clearedCount++;
        }
    }
    std::cout << "  ✓ Verified " << clearedCount << " neurons have no position" << std::endl;
    
    // Re-import from CSV
    ImportConfig config;
    config.importPositions = true;
    
    auto result = importer.importFromCSV(exportFile, config);
    
    if (result.success) {
        std::cout << "  ✓ Imported " << result.positionsSet << " positions from CSV" << std::endl;
    } else {
        std::cout << "  ✗ Import failed: " << result.errorMessage << std::endl;
    }
    std::cout << std::endl;
    
    // ========================================================================
    // Part 4: Verify imported positions
    // ========================================================================
    std::cout << "Part 4: Verifying imported positions..." << std::endl;
    
    int verifiedCount = 0;
    for (size_t i = 0; i < neuronIds.size(); ++i) {
        auto neuron = datastore.getNeuron(neuronIds[i]);
        if (neuron && neuron->hasPosition()) {
            const Position3D& pos = neuron->getPosition();
            float expectedX = i * 10.0f;
            
            if (std::abs(pos.x - expectedX) < 0.01f && 
                std::abs(pos.y - 0.0f) < 0.01f && 
                std::abs(pos.z - 0.0f) < 0.01f) {
                verifiedCount++;
                std::cout << "  ✓ Neuron " << neuronIds[i] 
                          << " position verified: (" << pos.x << ", " << pos.y << ", " << pos.z << ")" 
                          << std::endl;
            } else {
                std::cout << "  ✗ Neuron " << neuronIds[i] 
                          << " position mismatch: expected (" << expectedX << ", 0, 0), got (" 
                          << pos.x << ", " << pos.y << ", " << pos.z << ")" << std::endl;
            }
        } else {
            std::cout << "  ✗ Neuron " << neuronIds[i] << " has no position" << std::endl;
        }
    }
    
    std::cout << "  ✓ Verified " << verifiedCount << "/" << neuronIds.size() << " positions" << std::endl;
    std::cout << std::endl;
    
    // ========================================================================
    // Part 5: Demonstrate position scaling and offset
    // ========================================================================
    std::cout << "Part 5: Demonstrating position transformation..." << std::endl;
    
    // Create a new CSV with different positions
    std::string transformFile = "transform_positions.csv";
    {
        std::ofstream file(transformFile);
        file << "neuron_id,x,y,z\n";
        for (size_t i = 0; i < 5; ++i) {
            uint64_t neuronId = neuronIds[i];
            file << neuronId << "," << i << "," << i << "," << i << "\n";
        }
    }
    
    // Import with scaling and offset
    ImportConfig transformConfig;
    transformConfig.positionScale = 2.0f;  // Scale by 2x
    transformConfig.offsetX = 100.0f;      // Offset by 100 in X
    transformConfig.offsetY = 50.0f;       // Offset by 50 in Y
    transformConfig.offsetZ = 25.0f;       // Offset by 25 in Z
    
    auto transformResult = importer.importFromCSV(transformFile, transformConfig);
    
    if (transformResult.success) {
        std::cout << "  ✓ Imported with transformation" << std::endl;
        
        for (size_t i = 0; i < 5; ++i) {
            auto neuron = datastore.getNeuron(neuronIds[i]);
            if (neuron && neuron->hasPosition()) {
                const Position3D& pos = neuron->getPosition();
                std::cout << "    Neuron " << neuronIds[i] 
                          << ": original (" << i << "," << i << "," << i << ")"
                          << " → transformed (" << pos.x << "," << pos.y << "," << pos.z << ")" 
                          << std::endl;
            }
        }
    }
    std::cout << std::endl;
    
    // ========================================================================
    // Summary
    // ========================================================================
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "✓ Demonstrated position storage in NeuralObject base class" << std::endl;
    std::cout << "✓ Demonstrated CSV export/import functionality" << std::endl;
    std::cout << "✓ Demonstrated position transformation (scale + offset)" << std::endl;
    std::cout << "✓ All positions are persisted to datastore" << std::endl;
    std::cout << std::endl;
    std::cout << "Next steps:" << std::endl;
    std::cout << "  - Import positions from NEURON (.swc) files" << std::endl;
    std::cout << "  - Import positions from NEST (.json) files" << std::endl;
    std::cout << "  - Use stored positions in visualization (NetworkDataAdapter)" << std::endl;
    std::cout << "  - Combine imported positions with LayoutEngine for hybrid layouts" << std::endl;
    
    // Cleanup
    std::filesystem::remove(exportFile);
    std::filesystem::remove(transformFile);
    
    return 0;
}

