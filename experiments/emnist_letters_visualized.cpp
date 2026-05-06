/**
 * @file emnist_letters_visualized.cpp
 * @brief EMNIST Letters classification with real-time 3D visualization
 *
 * This experiment combines the EMNIST letters V1 architecture with
 * comprehensive real-time visualization to see what's happening under the hood.
 *
 * Features:
 * - Real-time 3D network visualization
 * - Spike propagation animation
 * - Activity heatmaps
 * - Raster plots
 * - Pattern detection
 * - Training statistics
 * - Interactive controls
 */

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <thread>
#include <filesystem>
#include <map>
#include <unordered_map>

// Core SNNFW
#include "snnfw/NeuralObjectFactory.h"
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
#include "snnfw/Synapse.h"
#include "snnfw/Dendrite.h"
#include "snnfw/NetworkPropagator.h"
#include "snnfw/SpikeProcessor.h"
#include "snnfw/ConfigLoader.h"
#include "snnfw/EMNISTLoader.h"
#include "snnfw/Datastore.h"
#include "snnfw/NetworkInspector.h"
#include "snnfw/ActivityMonitor.h"

// Visualization
#include "snnfw/VisualizationManager.h"
#include "snnfw/SimulationConfig.h"
#include "snnfw/RecordingManager.h"
#include "snnfw/PlaybackControls.h"
#include "snnfw/Camera.h"
#include "snnfw/ShaderManager.h"
#include "snnfw/NetworkDataAdapter.h"
#include "snnfw/LayoutEngine.h"
#include "snnfw/NetworkGraphRenderer.h"
#include "snnfw/ActivityVisualizer.h"
#include "snnfw/SpikeRenderer.h"
#include "snnfw/RecordingManager.h"
#include "snnfw/RasterPlotRenderer.h"
#include "snnfw/InteractionManager.h"
#include "snnfw/PatternDetector.h"
#include "snnfw/ActivityHistogram.h"

#include <imgui.h>
#include <glm/glm.hpp>
#include <GL/gl.h>
#include <GLFW/glfw3.h>

using namespace snnfw;

// Global variable for scroll input (updated by GLFW callback)
static double g_scrollYOffset = 0.0;

// GLFW scroll callback
static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    // Let ImGui handle scroll first
    ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureMouse) {
        // If ImGui doesn't want the scroll, use it for camera zoom
        g_scrollYOffset += yoffset;
    }
}

// Configuration for visualization with full 6-layer architecture
struct VisualizationConfig {
    // Data paths
    std::string trainImagesPath = "/home/dean/repos/snnfw/data/EMNIST/emnist-letters-train-images-idx3-ubyte";
    std::string trainLabelsPath = "/home/dean/repos/snnfw/data/EMNIST/emnist-letters-train-labels-idx1-ubyte";
    std::string testImagesPath = "/home/dean/repos/snnfw/data/EMNIST/emnist-letters-test-images-idx3-ubyte";
    std::string testLabelsPath = "/home/dean/repos/snnfw/data/EMNIST/emnist-letters-test-labels-idx1-ubyte";

    // Network parameters (matching 71% training config)
    double neuronWindow = 500.0;  // Larger window for richer patterns
    double neuronThreshold = 0.93;  // Higher threshold for better discrimination
    int neuronMaxPatterns = 500;  // More patterns per neuron

    // Training parameters
    int trainingExamplesPerLetter = 200;  // Match training experiment
    int maxTrainingImages = 5200;  // 200 per letter × 26 letters

    // Spike processor
    int numThreads = 24;  // More threads for larger network

    // Multi-column 6-layer architecture (matching 71% config)
    int numOrientations = 8;  // 8 orientations (0°, 22.5°, 45°, 67.5°, 90°, 112.5°, 135°, 157.5°)
    int numFrequencies = 2;   // 2 frequencies (low, high)
    int numColumns = 16;      // 8 orientations × 2 frequencies = 16 columns

    // Layer sizes (matching 71% config)
    int layer1Neurons = 32;   // Modulatory
    int layer23Neurons = 448; // Superficial pyramidal (much larger)
    int layer4Size = 7;       // 7×7 grid = 49 neurons
    int layer5Neurons = 80;   // Deep pyramidal output
    int layer6Neurons = 32;   // Corticothalamic feedback

    int neuronsPerOutputClass = 15;  // More output neurons for better discrimination

    // Gabor filter parameters (matching 71% config)
    double freqLow = 8.0;
    double freqHigh = 3.0;
    double gaborThreshold = 0.24;

    // Visualization
    int windowWidth = 1920;
    int windowHeight = 1080;
    bool enableVisualization = true;
};

// Structure to hold a single cortical column with 6 layers
struct CorticalColumn {
    std::shared_ptr<Column> column;

    // Layers
    std::shared_ptr<Layer> layer1;   // Apical dendrites, modulatory
    std::shared_ptr<Layer> layer23;  // Superficial pyramidal
    std::shared_ptr<Layer> layer4;   // Granular input
    std::shared_ptr<Layer> layer5;   // Deep pyramidal output
    std::shared_ptr<Layer> layer6;   // Corticothalamic feedback

    // Neurons in each layer
    std::vector<std::shared_ptr<Neuron>> layer1Neurons;
    std::vector<std::shared_ptr<Neuron>> layer23Neurons;
    std::vector<std::shared_ptr<Neuron>> layer4Neurons;
    std::vector<std::shared_ptr<Neuron>> layer5Neurons;
    std::vector<std::shared_ptr<Neuron>> layer6Neurons;

    double orientation;  // Preferred orientation for this column (0-180 degrees)
    double spatialFrequency;  // Spatial frequency (lambda)
    std::string featureType;  // Feature type identifier
    std::vector<std::vector<double>> gaborKernel;  // Gabor filter for this orientation and frequency
};

/**
 * @brief Create a Gabor filter kernel
 * @param orientation Orientation in degrees (0-180)
 * @param lambda Wavelength (spatial frequency)
 * @return 2D Gabor kernel (11×11)
 */
std::vector<std::vector<double>> createGaborKernel(double orientation, double lambda) {
    const int kernelSize = 11;
    const int halfSize = kernelSize / 2;
    const double sigma = lambda * 0.56;  // Bandwidth
    const double gamma = 0.5;  // Aspect ratio

    double theta = orientation * M_PI / 180.0;  // Convert to radians

    std::vector<std::vector<double>> kernel(kernelSize, std::vector<double>(kernelSize));

    for (int y = -halfSize; y <= halfSize; ++y) {
        for (int x = -halfSize; x <= halfSize; ++x) {
            // Rotate coordinates
            double xTheta = x * std::cos(theta) + y * std::sin(theta);
            double yTheta = -x * std::sin(theta) + y * std::cos(theta);

            // Gabor function
            double gaussian = std::exp(-(xTheta * xTheta + gamma * gamma * yTheta * yTheta) / (2 * sigma * sigma));
            double sinusoid = std::cos(2 * M_PI * xTheta / lambda);

            kernel[y + halfSize][x + halfSize] = gaussian * sinusoid;
        }
    }

    return kernel;
}

/**
 * @brief Apply Gabor filter to image
 * @param img EMNIST image
 * @param gaborKernel Gabor filter kernel
 * @return Response values for each position (flattened 6×6 grid)
 */
std::vector<double> applyGaborToImage(const EMNISTLoader::Image& img,
                                      const std::vector<std::vector<double>>& gaborKernel) {
    const int imgWidth = 28;
    const int imgHeight = 28;
    const int kernelSize = gaborKernel.size();
    const int halfKernel = kernelSize / 2;

    // Sample at 6×6 grid positions (matching Layer 4 grid)
    const int gridSize = 6;
    std::vector<double> responses;

    for (int gy = 0; gy < gridSize; ++gy) {
        for (int gx = 0; gx < gridSize; ++gx) {
            // Map grid position to image coordinates
            int centerX = (gx * imgWidth) / gridSize + imgWidth / (2 * gridSize);
            int centerY = (gy * imgHeight) / gridSize + imgHeight / (2 * gridSize);

            // Convolve with Gabor kernel
            double response = 0.0;
            for (int ky = 0; ky < kernelSize; ++ky) {
                for (int kx = 0; kx < kernelSize; ++kx) {
                    int imgX = centerX + (kx - halfKernel);
                    int imgY = centerY + (ky - halfKernel);

                    if (imgX >= 0 && imgX < imgWidth && imgY >= 0 && imgY < imgHeight) {
                        double pixel = img.pixels[imgY * imgWidth + imgX] / 255.0;
                        response += pixel * gaborKernel[ky][kx];
                    }
                }
            }

            responses.push_back(std::abs(response));  // Take absolute value
        }
    }

    return responses;
}

/**
 * @brief Copy spike pattern from source neurons to target neurons
 */
void copyLayerSpikePattern(const std::vector<std::shared_ptr<Neuron>>& sourceNeurons,
                          const std::vector<std::shared_ptr<Neuron>>& targetNeurons) {
    for (auto& targetNeuron : targetNeurons) {
        targetNeuron->clearSpikes();

        for (const auto& sourceNeuron : sourceNeurons) {
            const auto& spikes = sourceNeuron->getSpikes();
            for (double spikeTime : spikes) {
                targetNeuron->insertSpike(spikeTime);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "=== EMNIST Letters with Real-Time Visualization ===" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Parse Command-Line Arguments
    // ========================================================================
    snnfw::SimulationConfig simConfig;
    simConfig.enableVisualization = false;  // Default: no visualization (will be set based on mode)
    simConfig.enableRecording = false;      // Default: no recording
    simConfig.realTimeSync = true;          // Default: real-time sync for visualization

    VisualizationConfig config;
    std::string configFile;
    bool startPaused = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--config" && i + 1 < argc) {
            configFile = argv[++i];
        }
        else if (arg == "--record" && i + 1 < argc) {
            simConfig.enableRecording = true;
            simConfig.recordingFilename = argv[++i];
            std::cout << "Recording enabled: " << simConfig.recordingFilename << std::endl;
        }
        else if (arg == "--no-viz") {
            simConfig.enableVisualization = false;
            std::cout << "Visualization disabled" << std::endl;
        }
        else if (arg == "--no-realtime") {
            simConfig.realTimeSync = false;
            std::cout << "Real-time sync disabled (fast mode)" << std::endl;
        }
        else if (arg == "--playback" && i + 1 < argc) {
            simConfig.playbackMode = true;
            simConfig.playbackFilename = argv[++i];
            simConfig.enableVisualization = false;  // Playback uses visualization window, not live viz
            std::cout << "Playback mode: " << simConfig.playbackFilename << std::endl;
        }
        else if (arg == "--start-paused") {
            startPaused = true;
        }
        else if (i == 1) {
            config.trainImagesPath = arg;
        }
        else if (i == 2) {
            config.trainLabelsPath = arg;
        }
    }

    // If not in playback mode and not explicitly disabled, enable visualization
    if (!simConfig.playbackMode && !simConfig.enableVisualization) {
        simConfig.enableVisualization = true;
    }

    // Load configuration from file if provided
    if (!configFile.empty()) {
        try {
            ConfigLoader cfgLoader(configFile);

            // Load recording configuration (only if not in playback mode)
            if (!simConfig.playbackMode) {
                simConfig.enableRecording = cfgLoader.get<bool>("/recording/enabled", false);
                simConfig.recordingFilename = cfgLoader.get<std::string>("/recording/filename", "emnist_training.snnr");
                simConfig.recordingPath = cfgLoader.get<std::string>("/recording/path", "");
                simConfig.autoSaveRecording = cfgLoader.get<bool>("/recording/auto_save", true);
            }

            // Load network structure configuration
            simConfig.networkStructurePath = cfgLoader.get<std::string>("/network_structure/path", "");

            // Load spike processor configuration
            simConfig.realTimeSync = cfgLoader.get<bool>("/spike_processor/real_time_sync", true);

            // Load neuron configuration
            config.neuronWindow = cfgLoader.get<double>("/neuron/window_size_ms", 500.0);
            config.neuronThreshold = cfgLoader.get<double>("/neuron/similarity_threshold", 0.93);
            config.neuronMaxPatterns = cfgLoader.get<int>("/neuron/max_patterns", 500);

            std::cout << "Configuration loaded from: " << configFile << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to load config file: " << e.what() << std::endl;
            std::cerr << "Using default configuration" << std::endl;
        }
    }

    // Validate configuration
    if (!simConfig.validate()) {
        std::cerr << "Invalid configuration!" << std::endl;
        return 1;
    }
    
    // ========================================================================
    // Initialize Datastore and Core Components
    // ========================================================================
    std::cout << "Initializing datastore..." << std::endl;
    // Use the trained network database instead of creating a new one
    Datastore datastore("./emnist_training_db", 1000000);

    NeuralObjectFactory factory;
    NetworkInspector inspector;
    ActivityMonitor activityMonitor(datastore, simConfig);

    // Set up recording/playback manager if needed
    snnfw::RecordingManager* recordingManager = nullptr;
    if (simConfig.enableRecording || simConfig.playbackMode) {
        recordingManager = new snnfw::RecordingManager();
        if (simConfig.enableRecording) {
            activityMonitor.setRecordingManager(recordingManager);
            std::cout << "Recording manager initialized" << std::endl;
        }
    }
    
    // ========================================================================
    // Load EMNIST Data
    // ========================================================================
    std::cout << "Loading EMNIST Letters dataset..." << std::endl;
    EMNISTLoader trainLoader(EMNISTLoader::Variant::LETTERS);
    if (!trainLoader.load(config.trainImagesPath, config.trainLabelsPath,
                         config.maxTrainingImages, true)) {
        std::cerr << "Failed to load training data!" << std::endl;
        std::cerr << "Expected paths:" << std::endl;
        std::cerr << "  Images: " << config.trainImagesPath << std::endl;
        std::cerr << "  Labels: " << config.trainLabelsPath << std::endl;
        return 1;
    }
    std::cout << "Loaded " << trainLoader.size() << " training images" << std::endl;

    // Load test dataset
    EMNISTLoader testLoader(EMNISTLoader::Variant::LETTERS);
    if (!testLoader.load(config.testImagesPath, config.testLabelsPath,
                        0, true)) {  // 0 = load all test images
        std::cerr << "Failed to load test data!" << std::endl;
        std::cerr << "Expected paths:" << std::endl;
        std::cerr << "  Images: " << config.testImagesPath << std::endl;
        std::cerr << "  Labels: " << config.testLabelsPath << std::endl;
        return 1;
    }
    std::cout << "Loaded " << testLoader.size() << " test images" << std::endl;
    
    // ========================================================================
    // Build Multi-Column 6-Layer Hierarchical V1 Architecture
    // ========================================================================
    std::cout << "\nBuilding 6-layer hierarchical V1 architecture..." << std::endl;
    std::cout << "  " << config.numColumns << " cortical columns (" << config.numOrientations
              << " orientations × " << config.numFrequencies << " frequencies)" << std::endl;

    // Create hierarchical structure
    auto brain = factory.createBrain();
    brain->setName("Visual Processing Network");

    auto hemisphere = factory.createHemisphere();
    hemisphere->setName("Left Hemisphere");
    brain->addHemisphere(hemisphere->getId());

    auto occipitalLobe = factory.createLobe();
    occipitalLobe->setName("Occipital Lobe");
    hemisphere->addLobe(occipitalLobe->getId());

    auto v1Region = factory.createRegion();
    v1Region->setName("Primary Visual Cortex (V1)");
    occipitalLobe->addRegion(v1Region->getId());

    auto v1Nucleus = factory.createNucleus();
    v1Nucleus->setName("V1 Multi-Column Nucleus");
    v1Region->addNucleus(v1Nucleus->getId());

    datastore.put(brain);
    datastore.put(hemisphere);
    datastore.put(occipitalLobe);
    datastore.put(v1Region);
    datastore.put(v1Nucleus);

    // Create cortical columns
    std::vector<CorticalColumn> corticalColumns;
    std::vector<uint64_t> allNeuronIds;

    const double ORIENTATION_STEP = 180.0 / config.numOrientations;
    const std::vector<double> SPATIAL_FREQUENCIES = {config.freqLow, config.freqHigh};
    const std::vector<std::string> FREQ_NAMES = {"low_freq", "high_freq"};

    std::random_device rd;
    std::mt19937 gen(rd());

    int colIdx = 0;
    for (int oriIdx = 0; oriIdx < config.numOrientations; ++oriIdx) {
        double orientation = oriIdx * ORIENTATION_STEP;

        for (int freqIdx = 0; freqIdx < config.numFrequencies; ++freqIdx) {
            CorticalColumn col;
            col.orientation = orientation;
            col.spatialFrequency = SPATIAL_FREQUENCIES[freqIdx];
            col.featureType = "orientation_" + FREQ_NAMES[freqIdx];
            col.gaborKernel = createGaborKernel(col.orientation, col.spatialFrequency);

            // Create column
            col.column = factory.createColumn();
            v1Nucleus->addColumn(col.column->getId());

            // Spatial position for this column
            float xPos = (colIdx - config.numColumns / 2.0f) * 30.0f;
            col.column->setPosition(xPos, 0.0f, 0.0f);

            std::cout << "Column " << colIdx << " (" << orientation << "°, "
                      << col.featureType << "): ";

            // Create Layer 1 (Apical dendrites, modulatory) - Z=0
            col.layer1 = factory.createLayer();
            col.layer1->setPosition(xPos, 0.0f, 0.0f);
            col.column->addLayer(col.layer1->getId());
            auto layer1Cluster = factory.createCluster();
            col.layer1->addCluster(layer1Cluster->getId());

            for (int i = 0; i < config.layer1Neurons; ++i) {
                auto neuron = factory.createNeuron(config.neuronWindow, config.neuronThreshold, config.neuronMaxPatterns);
                neuron->setPosition(xPos + (std::rand() % 100 - 50) / 50.0f,
                                   (std::rand() % 100 - 50) / 50.0f,
                                   (std::rand() % 100 - 50) / 50.0f);
                auto axon = factory.createAxon(neuron->getId());
                auto dendrite = factory.createDendrite(neuron->getId());
                neuron->setAxonId(axon->getId());
                neuron->addDendrite(dendrite->getId());

                col.layer1Neurons.push_back(neuron);
                layer1Cluster->addNeuron(neuron->getId());
                allNeuronIds.push_back(neuron->getId());

                datastore.put(neuron);
                datastore.put(axon);
                datastore.put(dendrite);
            }
            datastore.put(layer1Cluster);
            datastore.put(col.layer1);

            // Create Layer 2/3 (Superficial pyramidal) - Z=10
            col.layer23 = factory.createLayer();
            col.layer23->setPosition(xPos, 0.0f, 10.0f);
            col.column->addLayer(col.layer23->getId());
            auto layer23Cluster = factory.createCluster();
            col.layer23->addCluster(layer23Cluster->getId());

            for (int i = 0; i < config.layer23Neurons; ++i) {
                auto neuron = factory.createNeuron(config.neuronWindow, config.neuronThreshold, config.neuronMaxPatterns);
                neuron->setPosition(xPos + (std::rand() % 100 - 50) / 50.0f,
                                   (std::rand() % 100 - 50) / 50.0f,
                                   10.0f + (std::rand() % 100 - 50) / 50.0f);
                auto axon = factory.createAxon(neuron->getId());
                auto dendrite = factory.createDendrite(neuron->getId());
                neuron->setAxonId(axon->getId());
                neuron->addDendrite(dendrite->getId());

                col.layer23Neurons.push_back(neuron);
                layer23Cluster->addNeuron(neuron->getId());
                allNeuronIds.push_back(neuron->getId());

                datastore.put(neuron);
                datastore.put(axon);
                datastore.put(dendrite);
            }
            datastore.put(layer23Cluster);
            datastore.put(col.layer23);

            std::cout << "L1=" << col.layer1Neurons.size() << ", L2/3=" << col.layer23Neurons.size();

            corticalColumns.push_back(col);
            colIdx++;
        }
    }

    // Continue creating remaining layers for each column
    std::cout << "\nCreating remaining layers (L4, L5, L6) for all columns..." << std::endl;
    for (size_t i = 0; i < corticalColumns.size(); ++i) {
        auto& col = corticalColumns[i];
        float xPos = (static_cast<int>(i) - config.numColumns / 2.0f) * 30.0f;

        // Create Layer 4 (Granular input layer) - Z=20
        col.layer4 = factory.createLayer();
        col.layer4->setPosition(xPos, 0.0f, 20.0f);
        col.column->addLayer(col.layer4->getId());
        auto layer4Cluster = factory.createCluster();
        col.layer4->addCluster(layer4Cluster->getId());

        const int LAYER4_NEURONS = config.layer4Size * config.layer4Size;
        for (int j = 0; j < LAYER4_NEURONS; ++j) {
            auto neuron = factory.createNeuron(config.neuronWindow, config.neuronThreshold, config.neuronMaxPatterns);
            neuron->setPosition(xPos + (std::rand() % 100 - 50) / 50.0f,
                               (std::rand() % 100 - 50) / 50.0f,
                               20.0f + (std::rand() % 100 - 50) / 50.0f);
            auto axon = factory.createAxon(neuron->getId());
            auto dendrite = factory.createDendrite(neuron->getId());
            neuron->setAxonId(axon->getId());
            neuron->addDendrite(dendrite->getId());

            col.layer4Neurons.push_back(neuron);
            layer4Cluster->addNeuron(neuron->getId());
            allNeuronIds.push_back(neuron->getId());

            datastore.put(neuron);
            datastore.put(axon);
            datastore.put(dendrite);
        }
        datastore.put(layer4Cluster);
        datastore.put(col.layer4);

        // Create Layer 5 (Deep pyramidal output) - Z=30
        col.layer5 = factory.createLayer();
        col.layer5->setPosition(xPos, 0.0f, 30.0f);
        col.column->addLayer(col.layer5->getId());
        auto layer5Cluster = factory.createCluster();
        col.layer5->addCluster(layer5Cluster->getId());

        for (int j = 0; j < config.layer5Neurons; ++j) {
            auto neuron = factory.createNeuron(config.neuronWindow, config.neuronThreshold, config.neuronMaxPatterns);
            neuron->setPosition(xPos + (std::rand() % 100 - 50) / 50.0f,
                               (std::rand() % 100 - 50) / 50.0f,
                               30.0f + (std::rand() % 100 - 50) / 50.0f);
            auto axon = factory.createAxon(neuron->getId());
            auto dendrite = factory.createDendrite(neuron->getId());
            neuron->setAxonId(axon->getId());
            neuron->addDendrite(dendrite->getId());

            col.layer5Neurons.push_back(neuron);
            layer5Cluster->addNeuron(neuron->getId());
            allNeuronIds.push_back(neuron->getId());

            datastore.put(neuron);
            datastore.put(axon);
            datastore.put(dendrite);
        }
        datastore.put(layer5Cluster);
        datastore.put(col.layer5);

        // Create Layer 6 (Corticothalamic feedback) - Z=40
        col.layer6 = factory.createLayer();
        col.layer6->setPosition(xPos, 0.0f, 40.0f);
        col.column->addLayer(col.layer6->getId());
        auto layer6Cluster = factory.createCluster();
        col.layer6->addCluster(layer6Cluster->getId());

        for (int j = 0; j < config.layer6Neurons; ++j) {
            auto neuron = factory.createNeuron(config.neuronWindow, config.neuronThreshold, config.neuronMaxPatterns);
            neuron->setPosition(xPos + (std::rand() % 100 - 50) / 50.0f,
                               (std::rand() % 100 - 50) / 50.0f,
                               40.0f + (std::rand() % 100 - 50) / 50.0f);
            auto axon = factory.createAxon(neuron->getId());
            auto dendrite = factory.createDendrite(neuron->getId());
            neuron->setAxonId(axon->getId());
            neuron->addDendrite(dendrite->getId());

            col.layer6Neurons.push_back(neuron);
            layer6Cluster->addNeuron(neuron->getId());
            allNeuronIds.push_back(neuron->getId());

            datastore.put(neuron);
            datastore.put(axon);
            datastore.put(dendrite);
        }
        datastore.put(layer6Cluster);
        datastore.put(col.layer6);

        datastore.put(col.column);

        std::cout << "  Column " << i << ": L4=" << col.layer4Neurons.size()
                  << ", L5=" << col.layer5Neurons.size()
                  << ", L6=" << col.layer6Neurons.size() << std::endl;
    }

    std::cout << "\n✓ Created " << corticalColumns.size() << " cortical columns with 6 layers each" << std::endl;
    std::cout << "  Total neurons per column: " << (config.layer1Neurons + config.layer23Neurons +
                                                     config.layer4Size * config.layer4Size +
                                                     config.layer5Neurons + config.layer6Neurons) << std::endl;

    // Create output clusters for letter classification (spatially separated from cortical columns)
    const int NUM_LETTERS = 26;
    std::vector<std::vector<std::shared_ptr<Neuron>>> outputPopulations;
    std::vector<std::shared_ptr<Cluster>> outputClusters;

    // Create output column and layer
    auto outputColumn = factory.createColumn();
    v1Nucleus->addColumn(outputColumn->getId());

    auto outputLayer = factory.createLayer();
    outputLayer->setPosition(0.0f, 0.0f, 50.0f);
    outputColumn->addLayer(outputLayer->getId());

    datastore.put(outputColumn);

    for (int letter = 0; letter < NUM_LETTERS; ++letter) {
        auto cluster = factory.createCluster();

        // Arrange output clusters in a grid
        int row = letter / 6;  // 6 letters per row
        int col = letter % 6;
        float xPos = (col - 2.5f) * 25.0f;  // Spread horizontally
        float yPos = (row - 2.0f) * 25.0f;  // Spread vertically
        cluster->setPosition(xPos, yPos, 50.0f);  // Output clusters at Z=50

        std::vector<std::shared_ptr<Neuron>> population;
        for (int n = 0; n < config.neuronsPerOutputClass; ++n) {
            auto neuron = factory.createNeuron(config.neuronWindow,
                                              config.neuronThreshold,
                                              config.neuronMaxPatterns);

            // Set neuron position within cluster
            float localX = xPos + (std::rand() % 100 - 50) / 30.0f;
            float localY = yPos + (std::rand() % 100 - 50) / 30.0f;
            float localZ = 50.0f + (std::rand() % 100 - 50) / 30.0f;
            neuron->setPosition(localX, localY, localZ);

            auto axon = factory.createAxon(neuron->getId());
            auto dendrite = factory.createDendrite(neuron->getId());

            neuron->setAxonId(axon->getId());
            neuron->addDendrite(dendrite->getId());

            cluster->addNeuron(neuron->getId());
            population.push_back(neuron);
            allNeuronIds.push_back(neuron->getId());

            datastore.put(neuron);
            datastore.put(axon);
            datastore.put(dendrite);
        }

        outputLayer->addCluster(cluster->getId());
        outputClusters.push_back(cluster);
        outputPopulations.push_back(population);

        datastore.put(cluster);
    }

    datastore.put(outputLayer);
    std::cout << "\n✓ Created output layer with " << NUM_LETTERS << " letter populations" << std::endl;

    // ========================================================================
    // Create Inter-Layer Connectivity (Canonical Cortical Microcircuit)
    // ========================================================================
    std::cout << "\nCreating inter-layer connections..." << std::endl;
    std::uniform_real_distribution<> dis(0.0, 1.0);

    int totalSynapses = 0;
    std::vector<std::shared_ptr<Synapse>> allSynapses;  // Track all synapses for registration

    // For each cortical column, create feedforward and feedback connections
    for (auto& col : corticalColumns) {
        // Layer 4 → Layer 2/3 (feedforward, 40% connectivity)
        for (auto& l4Neuron : col.layer4Neurons) {
            auto axon = datastore.getAxon(l4Neuron->getAxonId());
            if (!axon) continue;

            for (auto& l23Neuron : col.layer23Neurons) {
                if (dis(gen) < 0.4) {
                    auto dendriteIds = l23Neuron->getDendriteIds();
                    if (!dendriteIds.empty()) {
                        auto dendrite = datastore.getDendrite(dendriteIds[0]);
                        if (dendrite) {
                            auto synapse = factory.createSynapse(axon->getId(), dendrite->getId(), 0.8, 1.0);
                            axon->addSynapse(synapse->getId());
                            allSynapses.push_back(synapse);
                            datastore.put(synapse);
                            totalSynapses++;
                        }
                    }
                }
            }
            datastore.put(axon);
        }

        // Layer 2/3 → Layer 5 (feedforward, 30% connectivity)
        for (auto& l23Neuron : col.layer23Neurons) {
            auto axon = datastore.getAxon(l23Neuron->getAxonId());
            if (!axon) continue;

            for (auto& l5Neuron : col.layer5Neurons) {
                if (dis(gen) < 0.3) {
                    auto dendriteIds = l5Neuron->getDendriteIds();
                    if (!dendriteIds.empty()) {
                        auto dendrite = datastore.getDendrite(dendriteIds[0]);
                        if (dendrite) {
                            auto synapse = factory.createSynapse(axon->getId(), dendrite->getId(), 0.7, 1.0);
                            axon->addSynapse(synapse->getId());
                            allSynapses.push_back(synapse);
                            datastore.put(synapse);
                            totalSynapses++;
                        }
                    }
                }
            }
            datastore.put(axon);
        }

        // Layer 5 → Layer 6 (feedforward, 25% connectivity)
        for (auto& l5Neuron : col.layer5Neurons) {
            auto axon = datastore.getAxon(l5Neuron->getAxonId());
            if (!axon) continue;

            for (auto& l6Neuron : col.layer6Neurons) {
                if (dis(gen) < 0.25) {
                    auto dendriteIds = l6Neuron->getDendriteIds();
                    if (!dendriteIds.empty()) {
                        auto dendrite = datastore.getDendrite(dendriteIds[0]);
                        if (dendrite) {
                            auto synapse = factory.createSynapse(axon->getId(), dendrite->getId(), 0.6, 1.0);
                            axon->addSynapse(synapse->getId());
                            allSynapses.push_back(synapse);
                            datastore.put(synapse);
                            totalSynapses++;
                        }
                    }
                }
            }
            datastore.put(axon);
        }

        // Layer 6 → Layer 4 (feedback, 15% connectivity, weaker weights)
        for (auto& l6Neuron : col.layer6Neurons) {
            auto axon = datastore.getAxon(l6Neuron->getAxonId());
            if (!axon) continue;

            for (auto& l4Neuron : col.layer4Neurons) {
                if (dis(gen) < 0.15) {
                    auto dendriteIds = l4Neuron->getDendriteIds();
                    if (!dendriteIds.empty()) {
                        auto dendrite = datastore.getDendrite(dendriteIds[0]);
                        if (dendrite) {
                            auto synapse = factory.createSynapse(axon->getId(), dendrite->getId(), 0.4, 1.5);
                            axon->addSynapse(synapse->getId());
                            allSynapses.push_back(synapse);
                            datastore.put(synapse);
                            totalSynapses++;
                        }
                    }
                }
            }
            datastore.put(axon);
        }
    }

    std::cout << "  ✓ Created " << totalSynapses << " inter-layer synapses" << std::endl;

    // Layer 5 → Output layer connections (50% connectivity)
    std::cout << "Creating Layer 5 → Output connections..." << std::endl;
    int outputSynapses = 0;
    for (auto& col : corticalColumns) {
        for (auto& l5Neuron : col.layer5Neurons) {
            auto axon = datastore.getAxon(l5Neuron->getAxonId());
            if (!axon) continue;

            for (int letter = 0; letter < NUM_LETTERS; ++letter) {
                for (auto& outputNeuron : outputPopulations[letter]) {
                    if (dis(gen) < 0.5) {
                        auto dendriteIds = outputNeuron->getDendriteIds();
                        if (!dendriteIds.empty()) {
                            auto dendrite = datastore.getDendrite(dendriteIds[0]);
                            if (dendrite) {
                                auto synapse = factory.createSynapse(axon->getId(), dendrite->getId(), 0.5, 1.0);
                                axon->addSynapse(synapse->getId());
                                allSynapses.push_back(synapse);
                                datastore.put(synapse);
                                outputSynapses++;
                            }
                        }
                    }
                }
            }
            datastore.put(axon);
        }
    }

    std::cout << "  ✓ Created " << outputSynapses << " Layer 5 → Output synapses" << std::endl;
    std::cout << "  Total synapses: " << (totalSynapses + outputSynapses) << std::endl;

    // ========================================================================
    // Initialize Spike Processor
    // ========================================================================
    std::cout << "\nInitializing spike processor..." << std::endl;
    auto spikeProcessor = std::make_shared<SpikeProcessor>(10000, config.numThreads);

    // Set real-time sync based on configuration
    if (simConfig.realTimeSync) {
        std::cout << "Real-time synchronization enabled (1ms = 1ms)" << std::endl;
    } else {
        std::cout << "Real-time synchronization disabled (fast mode)" << std::endl;
    }

    // Connect SpikeProcessor to ActivityMonitor for spike recording
    if (simConfig.enableRecording) {
        spikeProcessor->setActivityMonitor(&activityMonitor);
        activityMonitor.startMonitoring();
        std::cout << "Spike recording connected and monitoring started" << std::endl;
    }

    auto networkPropagator = std::make_shared<NetworkPropagator>(spikeProcessor);
    spikeProcessor->start();

    // ========================================================================
    // Register all neurons, axons, and dendrites with NetworkPropagator
    // ========================================================================
    std::cout << "Registering neurons with NetworkPropagator..." << std::endl;

    // Register cortical column neurons
    for (auto& col : corticalColumns) {
        // Layer 1
        for (auto& neuron : col.layer1Neurons) {
            networkPropagator->registerNeuron(neuron);
            neuron->setNetworkPropagator(networkPropagator);

            auto axon = datastore.getAxon(neuron->getAxonId());
            if (axon) {
                networkPropagator->registerAxon(axon);
            }

            for (auto dendriteId : neuron->getDendriteIds()) {
                auto dendrite = datastore.getDendrite(dendriteId);
                if (dendrite) {
                    networkPropagator->registerDendrite(dendrite);
                    dendrite->setNetworkPropagator(networkPropagator);
                    spikeProcessor->registerDendrite(dendrite);
                }
            }
        }

        // Layer 2/3
        for (auto& neuron : col.layer23Neurons) {
            networkPropagator->registerNeuron(neuron);
            neuron->setNetworkPropagator(networkPropagator);

            auto axon = datastore.getAxon(neuron->getAxonId());
            if (axon) {
                networkPropagator->registerAxon(axon);
            }

            for (auto dendriteId : neuron->getDendriteIds()) {
                auto dendrite = datastore.getDendrite(dendriteId);
                if (dendrite) {
                    networkPropagator->registerDendrite(dendrite);
                    dendrite->setNetworkPropagator(networkPropagator);
                    spikeProcessor->registerDendrite(dendrite);
                }
            }
        }

        // Layer 4
        for (auto& neuron : col.layer4Neurons) {
            networkPropagator->registerNeuron(neuron);
            neuron->setNetworkPropagator(networkPropagator);

            auto axon = datastore.getAxon(neuron->getAxonId());
            if (axon) {
                networkPropagator->registerAxon(axon);
            }

            for (auto dendriteId : neuron->getDendriteIds()) {
                auto dendrite = datastore.getDendrite(dendriteId);
                if (dendrite) {
                    networkPropagator->registerDendrite(dendrite);
                    dendrite->setNetworkPropagator(networkPropagator);
                    spikeProcessor->registerDendrite(dendrite);
                }
            }
        }

        // Layer 5
        for (auto& neuron : col.layer5Neurons) {
            networkPropagator->registerNeuron(neuron);
            neuron->setNetworkPropagator(networkPropagator);

            auto axon = datastore.getAxon(neuron->getAxonId());
            if (axon) {
                networkPropagator->registerAxon(axon);
            }

            for (auto dendriteId : neuron->getDendriteIds()) {
                auto dendrite = datastore.getDendrite(dendriteId);
                if (dendrite) {
                    networkPropagator->registerDendrite(dendrite);
                    dendrite->setNetworkPropagator(networkPropagator);
                    spikeProcessor->registerDendrite(dendrite);
                }
            }
        }

        // Layer 6
        for (auto& neuron : col.layer6Neurons) {
            networkPropagator->registerNeuron(neuron);
            neuron->setNetworkPropagator(networkPropagator);

            auto axon = datastore.getAxon(neuron->getAxonId());
            if (axon) {
                networkPropagator->registerAxon(axon);
            }

            for (auto dendriteId : neuron->getDendriteIds()) {
                auto dendrite = datastore.getDendrite(dendriteId);
                if (dendrite) {
                    networkPropagator->registerDendrite(dendrite);
                    dendrite->setNetworkPropagator(networkPropagator);
                    spikeProcessor->registerDendrite(dendrite);
                }
            }
        }
    }

    // Register output neurons
    for (auto& population : outputPopulations) {
        for (auto& neuron : population) {
            networkPropagator->registerNeuron(neuron);
            neuron->setNetworkPropagator(networkPropagator);

            auto axon = datastore.getAxon(neuron->getAxonId());
            if (axon) {
                networkPropagator->registerAxon(axon);
            }

            for (auto dendriteId : neuron->getDendriteIds()) {
                auto dendrite = datastore.getDendrite(dendriteId);
                if (dendrite) {
                    networkPropagator->registerDendrite(dendrite);
                    dendrite->setNetworkPropagator(networkPropagator);
                    spikeProcessor->registerDendrite(dendrite);
                }
            }
        }
    }

    // Register all synapses
    std::cout << "Registering " << allSynapses.size() << " synapses with NetworkPropagator..." << std::endl;
    for (auto& synapse : allSynapses) {
        networkPropagator->registerSynapse(synapse);
    }

    std::cout << "  ✓ Registered all neurons, axons, dendrites, and synapses" << std::endl;

    // Build hierarchical cache for activity monitoring
    std::cout << "Building hierarchical cache for activity monitoring..." << std::endl;
    activityMonitor.buildHierarchicalCache(brain->getId());
    std::cout << "  ✓ Hierarchical cache built" << std::endl;

    // ========================================================================
    // Initialize Visualization
    // ========================================================================
    if (simConfig.enableVisualization || simConfig.playbackMode) {
        std::cout << "\nInitializing visualization..." << std::endl;
    }

    VisualizationManager vizManager(simConfig);
    vizManager.setTargetFPS(60);
    vizManager.enableVSync(true);
    vizManager.setBackgroundColor(0.05f, 0.05f, 0.1f);

    // Set up playback controls if in playback mode
    snnfw::PlaybackControls* playbackControls = nullptr;
    if (simConfig.playbackMode && recordingManager) {
        playbackControls = new snnfw::PlaybackControls(recordingManager);
        playbackControls->setPosition(10.0f, 10.0f);
        std::cout << "Playback controls initialized" << std::endl;
    }

    if (vizManager.isVisualizationEnabled()) {
        std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    }

    // Create visualization components
    NetworkDataAdapter adapter(datastore, inspector, &activityMonitor);
    ShaderManager shaderManager;
    NetworkGraphRenderer networkRenderer(shaderManager);
    SpikeRenderer spikeRenderer(shaderManager);
    ActivityVisualizer activityVisualizer(activityMonitor, adapter);
    RasterPlotRenderer rasterPlotRenderer(shaderManager);
    InteractionManager interactionManager(adapter);
    PatternDetector patternDetector;
    ActivityHistogram activityHistogram(shaderManager);

    if (simConfig.playbackMode && recordingManager) {
        if (!recordingManager->loadRecording(simConfig.playbackFilename, true)) {
            std::cerr << "Failed to load recording: " << simConfig.playbackFilename << std::endl;
            return 1;
        }
        recordingManager->setPlaybackCallback([&activityVisualizer](uint64_t source,
                                                                    uint64_t target,
                                                                    uint64_t synapse,
                                                                    uint64_t timestamp) {
            activityVisualizer.recordSpike(source, target, synapse, timestamp);
        });
        recordingManager->setSpeed(simConfig.playbackSpeed);
        recordingManager->setLooping(simConfig.playbackLooping);
        if (startPaused) {
            recordingManager->stop();
            activityVisualizer.clear();
            rasterPlotRenderer.clearSpikes();
        } else {
            recordingManager->play();
        }
    }

    // Flush all objects to disk to ensure they're available for extraction
    std::cout << "Flushing datastore..." << std::endl;
    size_t flushed = datastore.flushAll();
    std::cout << "Flushed " << flushed << " objects to disk" << std::endl;

    // Extract network structure from all layers
    std::cout << "Extracting network structure from all 6 layers + output..." << std::endl;

    // Collect all cluster IDs from all layers
    std::vector<uint64_t> allClusterIds;
    std::vector<uint64_t> layer1ClusterIds, layer23ClusterIds, layer4ClusterIds;
    std::vector<uint64_t> layer5ClusterIds, layer6ClusterIds, outputClusterIds;
    std::unordered_map<uint64_t, uint64_t> neuronToCluster;

    for (const auto& col : corticalColumns) {
        // Get cluster IDs from each layer
        auto l1Clusters = col.layer1->getClusterIds();
        auto l23Clusters = col.layer23->getClusterIds();
        auto l4Clusters = col.layer4->getClusterIds();
        auto l5Clusters = col.layer5->getClusterIds();
        auto l6Clusters = col.layer6->getClusterIds();

        layer1ClusterIds.insert(layer1ClusterIds.end(), l1Clusters.begin(), l1Clusters.end());
        layer23ClusterIds.insert(layer23ClusterIds.end(), l23Clusters.begin(), l23Clusters.end());
        layer4ClusterIds.insert(layer4ClusterIds.end(), l4Clusters.begin(), l4Clusters.end());
        layer5ClusterIds.insert(layer5ClusterIds.end(), l5Clusters.begin(), l5Clusters.end());
        layer6ClusterIds.insert(layer6ClusterIds.end(), l6Clusters.begin(), l6Clusters.end());
    }

    for (const auto& cluster : outputClusters) {
        outputClusterIds.push_back(cluster->getId());
    }

    // Combine all cluster IDs for extraction
    allClusterIds.insert(allClusterIds.end(), layer1ClusterIds.begin(), layer1ClusterIds.end());
    allClusterIds.insert(allClusterIds.end(), layer23ClusterIds.begin(), layer23ClusterIds.end());
    allClusterIds.insert(allClusterIds.end(), layer4ClusterIds.begin(), layer4ClusterIds.end());
    allClusterIds.insert(allClusterIds.end(), layer5ClusterIds.begin(), layer5ClusterIds.end());
    allClusterIds.insert(allClusterIds.end(), layer6ClusterIds.begin(), layer6ClusterIds.end());
    allClusterIds.insert(allClusterIds.end(), outputClusterIds.begin(), outputClusterIds.end());

    std::cout << "Extracting " << allClusterIds.size() << " clusters..." << std::endl;

    if (!adapter.extractMultipleClusters(allClusterIds)) {
        std::cerr << "Failed to extract clusters!" << std::endl;
        return 1;
    }

    // Set layer colors (distinct color for each layer)
    // Layer 1: Light Blue (modulatory)
    // Layer 2/3: Green (superficial pyramidal)
    // Layer 4: Yellow (granular input)
    // Layer 5: Orange (deep pyramidal output)
    // Layer 6: Red (corticothalamic feedback)
    // Output: Magenta (classification)

    uint64_t layer1Id = corticalColumns[0].layer1->getId();
    uint64_t layer23Id = corticalColumns[0].layer23->getId();
    uint64_t layer4Id = corticalColumns[0].layer4->getId();
    uint64_t layer5Id = corticalColumns[0].layer5->getId();
    uint64_t layer6Id = corticalColumns[0].layer6->getId();
    uint64_t outputLayerId = outputLayer->getId();

    adapter.setLayerColor(layer1Id, 0.5f, 0.8f, 1.0f);    // Light Blue
    adapter.setLayerColor(layer23Id, 0.2f, 1.0f, 0.3f);   // Green
    adapter.setLayerColor(layer4Id, 1.0f, 1.0f, 0.2f);    // Yellow
    adapter.setLayerColor(layer5Id, 1.0f, 0.6f, 0.2f);    // Orange
    adapter.setLayerColor(layer6Id, 1.0f, 0.2f, 0.2f);    // Red
    adapter.setLayerColor(outputLayerId, 1.0f, 0.2f, 0.8f); // Magenta

    adapter.setClusterLayerIds(layer1ClusterIds, layer1Id);
    adapter.setClusterLayerIds(layer23ClusterIds, layer23Id);
    adapter.setClusterLayerIds(layer4ClusterIds, layer4Id);
    adapter.setClusterLayerIds(layer5ClusterIds, layer5Id);
    adapter.setClusterLayerIds(layer6ClusterIds, layer6Id);
    adapter.setClusterLayerIds(outputClusterIds, outputLayerId);

    std::cout << "Extracted " << adapter.getNeurons().size() << " neurons" << std::endl;
    std::cout << "Extracted " << adapter.getSynapses().size() << " synapses" << std::endl;

    activityVisualizer.rebuildFromAdapter();
    for (const auto& neuron : adapter.getNeurons()) {
        neuronToCluster[neuron.id] = neuron.clusterId;
    }

    // Use stored positions (neurons already have explicit 3D positions set above)
    std::cout << "Using pre-set 3D positions for spatial organization!" << std::endl;

    // Initialize renderers
    if (!networkRenderer.initialize() || !spikeRenderer.initialize() ||
        !rasterPlotRenderer.initialize()) {
        std::cerr << "Failed to initialize renderers!" << std::endl;
        return 1;
    }
    activityHistogram.initialize();

    // Setup raster plot
    rasterPlotRenderer.setNeuronMapping(allNeuronIds);

    const glm::vec3 initialCamPos(0.0f, 120.0f, 260.0f);
    const glm::vec3 initialCamTarget(0.0f, 20.0f, 0.0f);
    Camera camera;
    camera.setPosition(initialCamPos);
    camera.lookAt(initialCamTarget);

    // Set custom scroll callback for camera zoom
    GLFWwindow* window = vizManager.getWindow();
    glfwSetScrollCallback(window, scrollCallback);

    // Configure visualization
    ActivityConfig activityConfig;
    activityConfig.showPropagation = true;
    activityConfig.showHeatmap = true;
    activityConfig.decayRate = 2.0f;
    activityConfig.particleLifetime = 500;
    activityConfig.propagationSpeed = 15.0f;
    activityVisualizer.setConfig(activityConfig);

    RenderConfig renderConfig;
    renderConfig.mode = RenderMode::NEURONS_ONLY;  // Don't show synapses
    renderConfig.enableLighting = true;
    renderConfig.neuronBaseRadius = 0.8f;  // Smaller neurons
    renderConfig.renderSynapses = false;  // Disable synapse rendering

    SpikeRenderConfig spikeConfig;
    spikeConfig.showTrails = true;
    spikeConfig.glowIntensity = 2.5f;

    RasterPlotConfig rasterConfig;
    rasterConfig.timeWindowMs = 3000.0f;
    rasterConfig.spikeMarkerSize = 3.0f;
    rasterConfig.colorByNeuronType = true;
    rasterConfig.showGrid = true;
    rasterPlotRenderer.setConfig(rasterConfig);

    std::cout << "\n=== Visualization Ready ===" << std::endl;
    std::cout << "Controls:" << std::endl;
    std::cout << "  - Mouse: Rotate camera" << std::endl;
    std::cout << "  - WASD: Pan camera" << std::endl;
    std::cout << "  - Q/E: Zoom in/out" << std::endl;
    std::cout << "  - Space: Pause/Resume training" << std::endl;
    std::cout << "  - ESC: Exit" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Training State
    // ========================================================================
    // Simulation time - updated from SpikeProcessor's getCurrentTime()
    double simulationTime = 0.0;

    // Training progress
    std::vector<int> trainCount(NUM_LETTERS, 0);
    size_t currentImageIdx = 0;
    bool trainingPaused = startPaused;
    bool trainingComplete = false;
    bool testingPhase = false;
    bool testingComplete = false;
    size_t currentTestIdx = 0;

    // Statistics
    int totalPatternsLearned = 0;
    int imagesProcessed = 0;
    double avgProcessingTime = 0.0;

    // Testing statistics
    int testCorrect = 0;
    int testTotal = 0;
    double testAccuracy = 0.0;

    // Mouse interaction
    int screenWidth = config.windowWidth;
    int screenHeight = config.windowHeight;
    double mouseX = 0.0, mouseY = 0.0;
    double lastMouseX = 0.0, lastMouseY = 0.0;
    bool leftMouseDown = false;

    // ========================================================================
    // EMNIST Image Display Texture
    // ========================================================================
    GLuint emnistTexture = 0;
    glGenTextures(1, &emnistTexture);
    glBindTexture(GL_TEXTURE_2D, emnistTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // Create initial blank texture (28x28 grayscale)
    std::vector<uint8_t> blankImage(28 * 28, 0);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, 28, 28, 0, GL_RED, GL_UNSIGNED_BYTE, blankImage.data());
    glBindTexture(GL_TEXTURE_2D, 0);

    // Current image being displayed
    int currentDisplayLabel = -1;
    std::vector<uint8_t> currentDisplayImage;
    if (simConfig.playbackMode && trainLoader.size() > 0) {
        const auto& emnistImg = trainLoader.getImage(0);
        currentDisplayLabel = emnistImg.label - 1;
        currentDisplayImage = emnistImg.pixels;
        glBindTexture(GL_TEXTURE_2D, emnistTexture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 28, 28, GL_RED, GL_UNSIGNED_BYTE,
                       emnistImg.pixels.data());
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    // ========================================================================
    // Visualization Update Callback
    // ========================================================================
    vizManager.setUpdateCallback([&](double deltaTime) {
        // Get current simulation time
        if (simConfig.playbackMode && recordingManager) {
            simulationTime = recordingManager->getPlaybackState().currentTime;
        } else {
            simulationTime = spikeProcessor->getCurrentTime();
        }

        // Get window size and mouse position
        glfwGetFramebufferSize(vizManager.getWindow(), &screenWidth, &screenHeight);
        glfwGetCursorPos(vizManager.getWindow(), &mouseX, &mouseY);

        // ====================================================================
        // Camera Controls - Mouse Orbit
        // ====================================================================
        GLFWwindow* window = vizManager.getWindow();

        // Check if ImGui wants to capture mouse (don't rotate camera if hovering over UI)
        ImGuiIO& io = ImGui::GetIO();
        bool mouseOverImGui = io.WantCaptureMouse;

        if (!mouseOverImGui && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            if (leftMouseDown) {
                // Orbit camera based on mouse drag
                float deltaX = static_cast<float>(mouseX - lastMouseX);
                float deltaY = static_cast<float>(mouseY - lastMouseY);
                camera.orbit(deltaX * 0.01f, deltaY * 0.01f);
            }
            leftMouseDown = true;
        } else {
            leftMouseDown = false;
        }

        lastMouseX = mouseX;
        lastMouseY = mouseY;

        // ====================================================================
        // Camera Controls - Keyboard
        // ====================================================================
        float panSpeed = 2.0f * deltaTime;
        float zoomSpeed = 50.0f * deltaTime;

        if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) {
            camera.pan(0.0f, panSpeed);
        }
        if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) {
            camera.pan(0.0f, -panSpeed);
        }
        if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS) {
            camera.pan(-panSpeed, 0.0f);
        }
        if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS) {
            camera.pan(panSpeed, 0.0f);
        }
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) {
            camera.zoom(zoomSpeed);  // Q = zoom in (positive delta moves closer)
        }
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) {
            camera.zoom(-zoomSpeed);  // E = zoom out (negative delta moves farther)
        }
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            vizManager.close();
        }

        // ====================================================================
        // Camera Controls - Mouse Wheel Zoom
        // ====================================================================
        if (g_scrollYOffset != 0.0) {
            // Scroll up = zoom in, scroll down = zoom out
            // Each scroll "tick" zooms by 10 units
            camera.zoom(static_cast<float>(g_scrollYOffset) * 10.0f);
            g_scrollYOffset = 0.0;  // Reset scroll offset
        }

        // Process training (one image per frame if not paused)
        if (!simConfig.playbackMode && !trainingPaused && !trainingComplete && currentImageIdx < trainLoader.size()) {
            auto imageStart = std::chrono::high_resolution_clock::now();

            const auto& emnistImg = trainLoader.getImage(currentImageIdx);
            int label = emnistImg.label - 1;  // Convert 1-26 to 0-25

            // Update display texture with current image
            currentDisplayLabel = label;
            currentDisplayImage = emnistImg.pixels;
            glBindTexture(GL_TEXTURE_2D, emnistTexture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 28, 28, GL_RED, GL_UNSIGNED_BYTE,
                           emnistImg.pixels.data());
            glBindTexture(GL_TEXTURE_2D, 0);

            // Check if we need more training for this letter
            if (trainCount[label] < config.trainingExamplesPerLetter) {
                // Clear all neuron spikes
                for (auto& neuron : allNeuronIds) {
                    auto n = datastore.getNeuron(neuron);
                    if (n) n->clearSpikes();
                }

                // Apply Gabor filters and fire Layer 4 neurons
                // Get current time from SpikeProcessor right before firing to ensure times are in the future
                // Add 200ms buffer to account for visualization overhead, processing time, and retrograde spike scheduling
                // (temporal signatures can have offsets up to 100ms, so we need buffer > 100ms + processing time)
                double baseTime = spikeProcessor->getCurrentTime() + 200.0;
                for (size_t colIdx = 0; colIdx < corticalColumns.size(); ++colIdx) {
                    auto& col = corticalColumns[colIdx];

                    // Apply Gabor filter for this column's orientation and frequency
                    auto gaborResponse = applyGaborToImage(emnistImg, col.gaborKernel);

                    // Fire Layer 4 neurons based on Gabor response
                    for (size_t neuronIdx = 0; neuronIdx < col.layer4Neurons.size() && neuronIdx < gaborResponse.size(); ++neuronIdx) {
                        if (gaborResponse[neuronIdx] > config.gaborThreshold) {
                            double firingTime = baseTime + (1.0 - gaborResponse[neuronIdx]) * 10.0;
                            col.layer4Neurons[neuronIdx]->fireSignature(firingTime);
                            networkPropagator->fireNeuron(col.layer4Neurons[neuronIdx]->getId(), firingTime);

                            // Visualize Layer 4 spike
                            activityVisualizer.recordSpike(col.layer4Neurons[neuronIdx]->getId(),
                                                          col.layer4Neurons[neuronIdx]->getId(),
                                                          0, simulationTime);
                            rasterPlotRenderer.addSpike(col.layer4Neurons[neuronIdx]->getId(), simulationTime, true);
                        }
                    }
                }

                // Allow time for propagation through layers (L4 → L2/3 → L5)
                double propagationTime = baseTime + 15.0;

                // Fire Layer 5 neurons (they receive input from Layer 2/3)
                std::vector<std::shared_ptr<Neuron>> layer5Neurons;
                for (auto& col : corticalColumns) {
                    // Fire a subset of Layer 5 neurons
                    for (size_t i = 0; i < col.layer5Neurons.size() / 2; ++i) {
                        auto& l5Neuron = col.layer5Neurons[i];
                        l5Neuron->fireSignature(propagationTime);
                        l5Neuron->fireAndAcknowledge(propagationTime);
                        networkPropagator->fireNeuron(l5Neuron->getId(), propagationTime);
                        l5Neuron->learnCurrentPattern();

                        layer5Neurons.push_back(l5Neuron);

                        // Visualize Layer 5 spike
                        activityVisualizer.recordSpike(l5Neuron->getId(), l5Neuron->getId(),
                                                      0, simulationTime + 15);
                        rasterPlotRenderer.addSpike(l5Neuron->getId(), simulationTime + 15, true);
                    }
                }

                // Copy Layer 5 spike pattern to output neurons for this letter
                copyLayerSpikePattern(layer5Neurons, outputPopulations[label]);

                // Train output neurons for this letter
                for (auto& outputNeuron : outputPopulations[label]) {
                    if (!outputNeuron->getSpikes().empty()) {
                        outputNeuron->fireAndAcknowledge(propagationTime + 5.0);
                        networkPropagator->fireNeuron(outputNeuron->getId(), propagationTime + 5.0);
                        outputNeuron->learnCurrentPattern();
                        totalPatternsLearned++;

                        // Visualize output activity
                        activityVisualizer.recordSpike(outputNeuron->getId(),
                                                      outputNeuron->getId(),
                                                      0, simulationTime + 20);
                        rasterPlotRenderer.addSpike(outputNeuron->getId(),
                                                   simulationTime + 20, true);
                    }
                }

                trainCount[label]++;
                imagesProcessed++;

                auto imageEnd = std::chrono::high_resolution_clock::now();
                double processingTime = std::chrono::duration<double, std::milli>(
                    imageEnd - imageStart).count();
                avgProcessingTime = (avgProcessingTime * (imagesProcessed - 1) + processingTime)
                                  / imagesProcessed;
            }

            currentImageIdx++;

            // Check if training is complete
            bool allLettersTrained = true;
            for (int count : trainCount) {
                if (count < config.trainingExamplesPerLetter) {
                    allLettersTrained = false;
                    break;
                }
            }
            if (allLettersTrained || currentImageIdx >= trainLoader.size()) {
                trainingComplete = true;
                testingPhase = true;
                std::cout << "\n=== Training Complete ===" << std::endl;
                std::cout << "Total patterns learned: " << totalPatternsLearned << std::endl;
                std::cout << "\n=== Starting Testing Phase ===" << std::endl;
            }
        }

        // Process testing (one image per frame if training complete and not paused)
        if (!simConfig.playbackMode && !trainingPaused && trainingComplete && testingPhase && !testingComplete
            && currentTestIdx < testLoader.size()) {
            auto imageStart = std::chrono::high_resolution_clock::now();

            const auto& emnistImg = testLoader.getImage(currentTestIdx);
            int label = emnistImg.label - 1;  // Convert 1-26 to 0-25

            // Update display texture with current image
            currentDisplayLabel = label;
            currentDisplayImage = emnistImg.pixels;
            glBindTexture(GL_TEXTURE_2D, emnistTexture);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 28, 28, GL_RED, GL_UNSIGNED_BYTE,
                           emnistImg.pixels.data());
            glBindTexture(GL_TEXTURE_2D, 0);

            // Clear all neuron spikes
            for (auto& neuron : allNeuronIds) {
                auto n = datastore.getNeuron(neuron);
                if (n) n->clearSpikes();
            }

            // Apply Gabor filters and fire Layer 4 neurons (same as training)
            // Get current time from SpikeProcessor right before firing to ensure times are in the future
            // Add 200ms buffer to account for visualization overhead, processing time, and retrograde spike scheduling
            // (temporal signatures can have offsets up to 100ms, so we need buffer > 100ms + processing time)
            double baseTime = spikeProcessor->getCurrentTime() + 200.0;
            for (size_t colIdx = 0; colIdx < corticalColumns.size(); ++colIdx) {
                auto& col = corticalColumns[colIdx];

                // Apply Gabor filter for this column's orientation and frequency
                auto gaborResponse = applyGaborToImage(emnistImg, col.gaborKernel);

                // Fire Layer 4 neurons based on Gabor response
                for (size_t neuronIdx = 0; neuronIdx < col.layer4Neurons.size() && neuronIdx < gaborResponse.size(); ++neuronIdx) {
                    if (gaborResponse[neuronIdx] > config.gaborThreshold) {
                        double firingTime = baseTime + (1.0 - gaborResponse[neuronIdx]) * 10.0;
                        col.layer4Neurons[neuronIdx]->fireSignature(firingTime);
                        networkPropagator->fireNeuron(col.layer4Neurons[neuronIdx]->getId(), firingTime);

                        // Visualize Layer 4 spike
                        activityVisualizer.recordSpike(col.layer4Neurons[neuronIdx]->getId(),
                                                      col.layer4Neurons[neuronIdx]->getId(),
                                                      0, simulationTime);
                        rasterPlotRenderer.addSpike(col.layer4Neurons[neuronIdx]->getId(), simulationTime, true);
                    }
                }
            }

            // Allow time for propagation through layers (L4 → L2/3 → L5)
            double propagationTime = baseTime + 15.0;

            // Fire Layer 5 neurons (they receive input from Layer 2/3)
            std::vector<std::shared_ptr<Neuron>> layer5Neurons;
            for (auto& col : corticalColumns) {
                // Fire a subset of Layer 5 neurons
                for (size_t i = 0; i < col.layer5Neurons.size() / 2; ++i) {
                    auto& l5Neuron = col.layer5Neurons[i];
                    l5Neuron->fireSignature(propagationTime);
                    l5Neuron->fireAndAcknowledge(propagationTime);
                    networkPropagator->fireNeuron(l5Neuron->getId(), propagationTime);

                    layer5Neurons.push_back(l5Neuron);

                    // Visualize Layer 5 spike
                    activityVisualizer.recordSpike(l5Neuron->getId(), l5Neuron->getId(),
                                                  0, simulationTime + 15);
                    rasterPlotRenderer.addSpike(l5Neuron->getId(), simulationTime + 15, true);
                }
            }

            // Copy Layer 5 spike pattern to ALL output neurons for pattern matching
            for (int i = 0; i < NUM_LETTERS; ++i) {
                copyLayerSpikePattern(layer5Neurons, outputPopulations[i]);
            }

            // Check which output cluster has highest pattern similarity
            int predictedLabel = -1;
            double maxSimilarity = -1.0;
            for (int i = 0; i < NUM_LETTERS; ++i) {
                double clusterSimilarity = 0.0;
                int neuronCount = 0;

                for (auto& outputNeuron : outputPopulations[i]) {
                    double similarity = outputNeuron->getBestSimilarity();
                    clusterSimilarity += similarity;
                    neuronCount++;

                    // Visualize output neuron activity based on similarity
                    if (similarity > 0.5) {
                        activityVisualizer.recordSpike(outputNeuron->getId(),
                                                      outputNeuron->getId(),
                                                      0, simulationTime + 10);
                        rasterPlotRenderer.addSpike(outputNeuron->getId(),
                                                   simulationTime + 10, true);
                    }
                }

                // Average similarity across all neurons in this cluster
                if (neuronCount > 0) {
                    clusterSimilarity /= neuronCount;
                }

                if (clusterSimilarity > maxSimilarity) {
                    maxSimilarity = clusterSimilarity;
                    predictedLabel = i;
                }
            }

            // Check if prediction is correct
            testTotal++;
            if (predictedLabel == label) {
                testCorrect++;
            }
            testAccuracy = (testTotal > 0) ? (100.0 * testCorrect / testTotal) : 0.0;

            currentTestIdx++;

            if (currentTestIdx >= testLoader.size()) {
                testingComplete = true;
                std::cout << "\n=== Testing Complete ===" << std::endl;
                std::cout << "Test Accuracy: " << testAccuracy << "% ("
                         << testCorrect << "/" << testTotal << ")" << std::endl;
            }
        }

        // Update visualization components
        activityVisualizer.update(simulationTime);
        if (recordingManager) {
            recordingManager->update(static_cast<uint64_t>(deltaTime * 1000.0));
        }
        patternDetector.update(simulationTime);
        activityHistogram.update(activityVisualizer, adapter, simulationTime);

        // Apply activity heatmap
        if (activityConfig.showHeatmap) {
            spikeRenderer.applyActivityHeatmap(
                activityVisualizer.getNeuronActivity(), adapter);
        }

        // Update hover state
        PickResult hoverResult = interactionManager.pickNeuron(
            static_cast<float>(mouseX), static_cast<float>(mouseY),
            screenWidth, screenHeight, camera);

        if (hoverResult.hit) {
            interactionManager.setHoveredNeuron(hoverResult.neuronId);
        } else {
            interactionManager.clearHover();
        }
    });

    // ========================================================================
    // Visualization Render Callback
    // ========================================================================
    vizManager.setRenderCallback([&](double deltaTime) {
        // Render 3D network
        networkRenderer.render(adapter, camera, renderConfig);

        // Render clusters with activity-based coloring
        if (renderConfig.renderClusters) {
            if (simConfig.playbackMode) {
                std::map<uint64_t, size_t> playbackClusterCounts;
                for (const auto& activity : activityVisualizer.getNeuronActivity()) {
                    if (activity.activityLevel <= 0.0f) {
                        continue;
                    }
                    auto it = neuronToCluster.find(activity.neuronId);
                    if (it == neuronToCluster.end()) {
                        continue;
                    }
                    size_t scaled = static_cast<size_t>(activity.activityLevel * 100.0f);
                    playbackClusterCounts[it->second] += std::max<size_t>(1, scaled);
                }
                networkRenderer.renderClusters(adapter, camera, renderConfig, playbackClusterCounts);
            } else {
                // Get cluster activity from activity monitor
                auto snapshot = activityMonitor.getLatestSnapshot();
                networkRenderer.renderClusters(adapter, camera, renderConfig, snapshot.clusterSpikeCount);
            }
        }

        // Render selection highlighting
        if (interactionManager.getSelectionCount() > 0) {
            networkRenderer.renderSelectedNeurons(
                interactionManager.getSelectedNeurons(),
                adapter, camera, interactionManager.getHighlightColor());
        }

        // Render hover highlighting
        if (interactionManager.getHoveredNeuron() != 0) {
            std::unordered_set<uint64_t> hovered = {interactionManager.getHoveredNeuron()};
            networkRenderer.renderSelectedNeurons(
                hovered, adapter, camera, glm::vec4(0.5f, 0.8f, 1.0f, 1.0f));
        }

        // Render spike particles and trails
        spikeRenderer.renderSpikeParticles(
            activityVisualizer.getSpikeParticles(), camera, spikeConfig);
        spikeRenderer.renderSpikeTrails(
            activityVisualizer.getSpikeParticles(), adapter, camera, spikeConfig);

        // ====================================================================
        // ImGui UI - Training/Testing Status
        // ====================================================================
        ImGui::Begin("EMNIST Training/Testing Status");

        ImGui::Text("Simulation Time: %lu ms", static_cast<unsigned long>(simulationTime));
        ImGui::Separator();

        if (!testingPhase) {
            // Training progress
            ImGui::Text("Training Progress");
            ImGui::Text("Images Processed: %d / %zu", imagesProcessed, trainLoader.size());
            ImGui::Text("Patterns Learned: %d", totalPatternsLearned);
            ImGui::Text("Avg Processing: %.2f ms/image", avgProcessingTime);

            ImGui::ProgressBar(static_cast<float>(currentImageIdx) / trainLoader.size(),
                              ImVec2(-1, 0), "");

            if (trainingComplete) {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "TRAINING COMPLETE!");
            } else if (trainingPaused) {
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "PAUSED");
            } else {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "TRAINING...");
            }
        } else {
            // Testing progress
            ImGui::Text("Testing Progress");
            ImGui::Text("Images Tested: %zu / %zu", currentTestIdx, testLoader.size());
            ImGui::Text("Correct: %d / %d", testCorrect, testTotal);
            ImGui::Text("Accuracy: %.2f%%", testAccuracy);

            ImGui::ProgressBar(static_cast<float>(currentTestIdx) / testLoader.size(),
                              ImVec2(-1, 0), "");

            if (testingComplete) {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "TESTING COMPLETE!");
            } else if (trainingPaused) {
                ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "PAUSED");
            } else {
                ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "TESTING...");
            }
        }

        if (ImGui::Button(trainingPaused ? "Resume" : "Pause")) {
            trainingPaused = !trainingPaused;
        }

        ImGui::Separator();
        ImGui::Text("Per-Letter Progress");

        for (int i = 0; i < NUM_LETTERS; ++i) {
            char letter = 'A' + i;
            float progress = static_cast<float>(trainCount[i]) / config.trainingExamplesPerLetter;

            ImGui::Text("%c:", letter);
            ImGui::SameLine();
            ImGui::ProgressBar(progress, ImVec2(200, 0),
                             (std::to_string(trainCount[i]) + "/" +
                              std::to_string(config.trainingExamplesPerLetter)).c_str());

            if ((i + 1) % 2 == 0) {
                // New line every 2 letters
            } else {
                ImGui::SameLine();
            }
        }

        ImGui::End();

        // ====================================================================
        // ImGui UI - Current EMNIST Image Display
        // ====================================================================
        ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(200, 280), ImGuiCond_FirstUseEver);
        ImGui::Begin("Current Image", nullptr, ImGuiWindowFlags_NoResize);

        if (currentDisplayLabel >= 0 && !currentDisplayImage.empty()) {
            // Display the current letter and phase
            char currentLetter = 'A' + currentDisplayLabel;
            ImGui::SetWindowFontScale(2.0f);
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Letter: %c", currentLetter);
            ImGui::SetWindowFontScale(1.0f);

            // Show phase (Training or Testing)
            if (testingPhase) {
                ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Phase: TESTING");
            } else {
                ImGui::TextColored(ImVec4(0.0f, 0.8f, 1.0f, 1.0f), "Phase: TRAINING");
            }

            ImGui::Separator();

            // Display the EMNIST image (scaled up for visibility)
            ImGui::Text("Input Image:");
            ImTextureID texId = (ImTextureID)(intptr_t)emnistTexture;
            ImGui::Image(texId, ImVec2(168, 168), ImVec2(1, 0), ImVec2(0, 1),
                        ImVec4(1, 1, 1, 1), ImVec4(0.5f, 0.5f, 0.5f, 1.0f));
        } else {
            ImGui::Text("No image loaded");
        }

        ImGui::End();

        // ====================================================================
        // ImGui UI - Network Legend
        // ====================================================================
        ImGui::SetNextWindowPos(ImVec2(screenWidth - 310, 10), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(300, 500), ImGuiCond_FirstUseEver);
        ImGui::Begin("Network Legend");

        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "6-Layer Cortical Architecture");
        ImGui::Separator();

        // Cortical Columns
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "CORTICAL COLUMNS");
        ImGui::Text("%d columns with 6 layers each", config.numColumns);
        ImGui::Indent();
        ImGui::Text("Layer 1 (Z=0): %d neurons/col", config.layer1Neurons);
        ImGui::Text("Layer 2/3 (Z=10): %d neurons/col", config.layer23Neurons);
        ImGui::Text("Layer 4 (Z=20): %d neurons/col", config.layer4Size * config.layer4Size);
        ImGui::Text("Layer 5 (Z=30): %d neurons/col", config.layer5Neurons);
        ImGui::Text("Layer 6 (Z=40): %d neurons/col", config.layer6Neurons);
        ImGui::Unindent();

        ImGui::Separator();

        // Output Layer
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "OUTPUT LAYER (Z=50)");
        ImGui::Text("26 Letter Classification Clusters");
        ImGui::Text("Arranged in 5 rows x 6 columns grid");
        ImGui::Indent();

        // Show letters in a grid layout
        for (int row = 0; row < 5; ++row) {
            std::string rowText = "Row " + std::to_string(row) + ": ";
            for (int col = 0; col < 6; ++col) {
                int letter = row * 6 + col;
                if (letter < NUM_LETTERS) {
                    rowText += std::string(1, 'A' + letter) + " ";
                }
            }
            ImGui::Text("%s", rowText.c_str());
        }
        ImGui::Unindent();

        ImGui::Separator();

        // Layer Color Legend
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Layer Colors");
        ImGui::Separator();

        ImGui::TextColored(ImVec4(0.5f, 0.8f, 1.0f, 1.0f), "● Layer 1 (Light Blue)");
        ImGui::Text("  Apical dendrites, modulatory");

        ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.3f, 1.0f), "● Layer 2/3 (Green)");
        ImGui::Text("  Superficial pyramidal");

        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.2f, 1.0f), "● Layer 4 (Yellow)");
        ImGui::Text("  Granular input layer");

        ImGui::TextColored(ImVec4(1.0f, 0.6f, 0.2f, 1.0f), "● Layer 5 (Orange)");
        ImGui::Text("  Deep pyramidal output");

        ImGui::TextColored(ImVec4(1.0f, 0.2f, 0.2f, 1.0f), "● Layer 6 (Red)");
        ImGui::Text("  Corticothalamic feedback");

        ImGui::TextColored(ImVec4(1.0f, 0.2f, 0.8f, 1.0f), "● Output (Magenta)");
        ImGui::Text("  Letter classification");

        ImGui::Separator();

        // Other Visual Elements
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Other Elements");
        ImGui::Separator();

        ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), "— Synapses (Cyan)");
        ImGui::Text("  Excitatory connections");

        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.2f, 1.0f), "★ Active Spikes");
        ImGui::Text("  Yellow = high activity");

        ImGui::Separator();

        // Statistics
        ImGui::TextColored(ImVec4(1.0f, 1.0f, 0.0f, 1.0f), "Network Stats");
        ImGui::Separator();
        int neuronsPerColumn = config.layer1Neurons + config.layer23Neurons +
                               (config.layer4Size * config.layer4Size) +
                               config.layer5Neurons + config.layer6Neurons;
        ImGui::Text("Neurons/Column: %d", neuronsPerColumn);
        ImGui::Text("Cortical Neurons: %d", config.numColumns * neuronsPerColumn);
        ImGui::Text("Output Neurons: %d", NUM_LETTERS * config.neuronsPerOutputClass);
        ImGui::Text("Total Neurons: %d",
                   config.numColumns * neuronsPerColumn +
                   NUM_LETTERS * config.neuronsPerOutputClass);
        ImGui::Text("Synapses: %d", totalSynapses + outputSynapses);

        ImGui::End();

        // ====================================================================
        // ImGui UI - Network Activity
        // ====================================================================
        ImGui::Begin("Network Activity");

        ImGui::Text("Total Spikes: %u", activityVisualizer.getTotalSpikes());
        ImGui::Text("Active Neurons: %u", activityVisualizer.getActiveNeuronCount());
        ImGui::Text("Avg Activity: %.3f", activityVisualizer.getAverageActivityLevel());
        ImGui::Text("Active Particles: %zu", activityVisualizer.getSpikeParticles().size());

        ImGui::Separator();
        ImGui::Text("Visualization Settings");

        if (ImGui::Checkbox("Show Propagation", &activityConfig.showPropagation)) {
            activityVisualizer.setConfig(activityConfig);
        }
        if (ImGui::Checkbox("Show Heatmap", &activityConfig.showHeatmap)) {
            activityVisualizer.setConfig(activityConfig);
        }
        if (ImGui::Checkbox("Show Trails", &spikeConfig.showTrails)) {
            // Config applied automatically
        }

        if (ImGui::SliderFloat("Decay Rate", &activityConfig.decayRate, 0.1f, 5.0f)) {
            activityVisualizer.setConfig(activityConfig);
        }
        if (ImGui::SliderFloat("Glow Intensity", &spikeConfig.glowIntensity, 0.5f, 5.0f)) {
            // Config applied automatically
        }

        ImGui::Separator();
        ImGui::Text("Render Mode");

        static int renderMode = 0;
        const char* renderModes[] = {"Neurons + Synapses", "Neurons Only", "Synapses Only"};
        if (ImGui::Combo("Mode", &renderMode, renderModes, 3)) {
            renderConfig.mode = static_cast<RenderMode>(renderMode);
        }

        ImGui::Checkbox("Enable Lighting", &renderConfig.enableLighting);

        ImGui::End();

        // ====================================================================
        // ImGui UI - Raster Plot
        // ====================================================================
        bool showRasterPlot = true;
        if (showRasterPlot) {
            ImGui::Begin("Raster Plot", &showRasterPlot);

            ImVec2 windowSize = ImGui::GetContentRegionAvail();
            float controlHeight = 100.0f;
            float plotHeight = windowSize.y - controlHeight;

            if (plotHeight > 100.0f) {
                ImVec2 plotPos = ImGui::GetCursorScreenPos();
                rasterPlotRenderer.render(plotPos.x, plotPos.y, windowSize.x, plotHeight,
                                         simulationTime);
                ImGui::Dummy(ImVec2(windowSize.x, plotHeight));
            }

            ImGui::Separator();
            ImGui::Text("Spikes in Plot: %zu", rasterPlotRenderer.getSpikeCount());

            if (ImGui::SliderFloat("Time Window (ms)", &rasterConfig.timeWindowMs,
                                  500.0f, 10000.0f)) {
                rasterPlotRenderer.setConfig(rasterConfig);
            }

            if (ImGui::Button("Clear Raster Plot")) {
                rasterPlotRenderer.clearSpikes();
            }

            ImGui::End();
        }

        // ====================================================================
        // ImGui UI - Playback Controls (if in playback mode)
        // ====================================================================
        if (playbackControls) {
            playbackControls->render();
        }

        // ====================================================================
        // ImGui UI - Camera Controls
        // ====================================================================
        ImGui::Begin("Camera");

        glm::vec3 camPos = camera.getPosition();
        ImGui::Text("Position: (%.1f, %.1f, %.1f)", camPos.x, camPos.y, camPos.z);

        if (ImGui::Button("Reset Camera")) {
            camera.setPosition(initialCamPos);
            camera.lookAt(initialCamTarget);
        }

        if (ImGui::Button("Top View")) {
            camera.setPosition(glm::vec3(0.0f, 200.0f, 0.1f));
            camera.lookAt(glm::vec3(0.0f, 0.0f, 0.0f));
        }

        ImGui::SameLine();
        if (ImGui::Button("Side View")) {
            camera.setPosition(glm::vec3(200.0f, 0.0f, 0.0f));
            camera.lookAt(glm::vec3(0.0f, 0.0f, 0.0f));
        }

        ImGui::End();
    });

    // ========================================================================
    // Mouse Click Callback
    // ========================================================================
    vizManager.setMouseClickCallback([&](int button, int action, int mods, double x, double y) {
        if (button == 0 && action == 1) {  // Left click
            PickResult result = interactionManager.pickNeuron(
                static_cast<float>(x), static_cast<float>(y),
                screenWidth, screenHeight, camera);

            if (result.hit) {
                SelectionMode mode = SelectionMode::SINGLE;
                if (mods & 0x0002) mode = SelectionMode::ADDITIVE;
                else if (mods & 0x0004) mode = SelectionMode::SUBTRACTIVE;
                else if (mods & 0x0001) mode = SelectionMode::TOGGLE;

                interactionManager.selectNeuron(result.neuronId, mode);
            } else if (mods == 0) {
                interactionManager.clearSelection();
            }
        }
    });

    // ========================================================================
    // Run Visualization Loop
    // ========================================================================
    std::cout << "Starting visualization and training loop..." << std::endl;
    vizManager.run();

    // ========================================================================
    // Cleanup
    // ========================================================================
    std::cout << "\nShutting down..." << std::endl;

    spikeProcessor->stop();

    // Save recording if enabled
    if (recordingManager && simConfig.enableRecording) {
        std::string filename = simConfig.recordingFilename;
        if (filename.empty()) {
            filename = simConfig.generateRecordingFilename();
        }

        // Construct full recording path
        std::string recordingPath = filename;
        if (!simConfig.recordingPath.empty()) {
            std::filesystem::path fullPath(simConfig.recordingPath);

            // Try to create the directory if it's a local path (not SMB)
            if (simConfig.recordingPath.find("://") == std::string::npos) {
                try {
                    std::filesystem::create_directories(fullPath);
                    std::cout << "Recording directory created/verified: " << simConfig.recordingPath << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Could not create recording directory: " << e.what() << std::endl;
                }
            } else {
                std::cout << "Recording path is remote (SMB/network): " << simConfig.recordingPath << std::endl;
            }

            fullPath /= filename;
            recordingPath = fullPath.string();
        }

        std::cout << "Saving recording to " << recordingPath << "..." << std::endl;
        if (activityMonitor.saveRecording(recordingPath)) {
            std::cout << "Recording saved successfully!" << std::endl;

            const auto& metadata = recordingManager->getMetadata();
            std::cout << "  Duration: " << metadata.duration << " ms" << std::endl;
            std::cout << "  Spikes: " << metadata.spikeCount << std::endl;
            std::cout << "  Neurons: " << metadata.neuronCount << std::endl;
        } else {
            std::cerr << "Failed to save recording!" << std::endl;
        }
    }

    networkRenderer.cleanup();
    spikeRenderer.cleanup();
    rasterPlotRenderer.cleanup();
    activityHistogram.cleanup();

    // Cleanup EMNIST texture
    if (emnistTexture != 0) {
        glDeleteTextures(1, &emnistTexture);
    }

    std::cout << "\n=== Final Statistics ===" << std::endl;
    std::cout << "Total images processed: " << imagesProcessed << std::endl;
    std::cout << "Total patterns learned: " << totalPatternsLearned << std::endl;
    std::cout << "Total spikes generated: " << activityVisualizer.getTotalSpikes() << std::endl;
    std::cout << "Average processing time: " << avgProcessingTime << " ms/image" << std::endl;

    std::cout << "\nPer-letter training counts:" << std::endl;
    for (int i = 0; i < NUM_LETTERS; ++i) {
        char letter = 'A' + i;
        std::cout << "  " << letter << ": " << trainCount[i] << " patterns" << std::endl;
    }

    std::cout << "\n=== Experiment Complete ===" << std::endl;

    // Cleanup recording manager and playback controls
    if (recordingManager) {
        delete recordingManager;
    }
    if (playbackControls) {
        delete playbackControls;
    }

    return 0;
}
