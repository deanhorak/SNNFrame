/**
 * @file emnist_letters_training.cpp
 * @brief EMNIST Letters classification - Performance-optimized training
 *
 * This experiment focuses on training performance without visualization overhead.
 * Uses the same 6-layer V1 architecture as the visualization experiment.
 * Supports optional recording for later playback/analysis.
 *
 * Features:
 * - No visualization overhead (headless mode)
 * - Optional spike recording for playback
 * - Full-scale network (larger than visualization version)
 * - Progress reporting to console
 * - Identical training/testing logic to visualization experiment
 */

#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <climits>

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
#include "snnfw/EMNISTLoader.h"
#include "snnfw/Datastore.h"
#include "snnfw/NetworkInspector.h"
#include "snnfw/ActivityMonitor.h"
#include "snnfw/SimulationConfig.h"
#include "snnfw/RecordingManager.h"
#include "snnfw/NetworkDataAdapter.h"
#include "snnfw/LayoutEngine.h"

using namespace snnfw;

constexpr int NUM_LETTERS = 26;

// Configuration for training
struct TrainingConfig {
    // Data paths
    std::string trainImagesPath = "/home/dean/repos/snnfw/data/EMNIST/emnist-letters-train-images-idx3-ubyte";
    std::string trainLabelsPath = "/home/dean/repos/snnfw/data/EMNIST/emnist-letters-train-labels-idx1-ubyte";
    std::string testImagesPath = "/home/dean/repos/snnfw/data/EMNIST/emnist-letters-test-images-idx3-ubyte";
    std::string testLabelsPath = "/home/dean/repos/snnfw/data/EMNIST/emnist-letters-test-labels-idx1-ubyte";

    // Network parameters (matching 71% config)
    double neuronWindow = 500.0;  // Larger window for richer patterns
    double neuronThreshold = 0.93;  // Higher threshold for better discrimination
    int neuronMaxPatterns = 500;  // More patterns per neuron

    // Training parameters
    int trainingExamplesPerLetter = 200;  // Will increase to 800 for final run
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

    // Inter-layer connectivity
    double layer4ToLayer23Prob = 0.3;
    double layer4ToLayer5Prob = 0.5;   // Direct L4 → L5 connections (bypass L2/3)
    double layer23ToLayer5Prob = 0.4;
    double layer5ToLayer6Prob = 0.3;
    double layer6ToLayer4Prob = 0.2;

    // Synaptic parameters
    double initialWeight = 0.5;
    double maxWeight = 1.0;
};

// Cortical column structure
struct CorticalColumn {
    double orientation;
    double spatialFrequency;
    std::string featureType;
    std::vector<std::vector<double>> gaborKernel;

    std::shared_ptr<Column> column;
    std::shared_ptr<Layer> layer1;
    std::shared_ptr<Layer> layer23;
    std::shared_ptr<Layer> layer4;
    std::shared_ptr<Layer> layer5;
    std::shared_ptr<Layer> layer6;

    std::vector<std::shared_ptr<Neuron>> layer1Neurons;
    std::vector<std::shared_ptr<Neuron>> layer23Neurons;
    std::vector<std::shared_ptr<Neuron>> layer4Neurons;
    std::vector<std::shared_ptr<Neuron>> layer5Neurons;
    std::vector<std::shared_ptr<Neuron>> layer6Neurons;
};

/**
 * @brief Create Gabor filter kernel
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
 * @param gridSize Grid size for sampling (e.g., 8 for 8×8 grid)
 * @return Response values for each position (flattened grid)
 */
std::vector<double> applyGaborToImage(const EMNISTLoader::Image& img,
                                      const std::vector<std::vector<double>>& gaborKernel,
                                      int gridSize) {
    const int imgWidth = 28;
    const int imgHeight = 28;
    const int kernelSize = gaborKernel.size();
    const int halfKernel = kernelSize / 2;

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

            responses.push_back(std::abs(response));  // Absolute value for response magnitude
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

/**
 * @brief Print progress bar
 */
void printProgress(const std::string& label, int current, int total, double accuracy = -1.0) {
    const int barWidth = 50;
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(barWidth * progress);

    std::cout << "\r" << label << " [";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100.0) << "% "
              << "(" << current << "/" << total << ")";

    if (accuracy >= 0.0) {
        std::cout << " | Acc: " << std::fixed << std::setprecision(2) << accuracy << "%";
    }

    std::cout << std::flush;
}

int main(int argc, char* argv[]) {
    std::cout << "=== EMNIST Letters Training (Performance Mode) ===" << std::endl;
    std::cout << std::endl;

    // ========================================================================
    // Parse Command-Line Arguments
    // ========================================================================
    snnfw::SimulationConfig simConfig;
    simConfig.enableVisualization = false;  // No visualization
    simConfig.enableRecording = false;      // Default: no recording
    simConfig.realTimeSync = false;         // Disabled for fast training (run as fast as possible)

    TrainingConfig config;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--record" && i + 1 < argc) {
            simConfig.enableRecording = true;
            simConfig.recordingFilename = argv[++i];
        } else if (arg == "--examples" && i + 1 < argc) {
            config.trainingExamplesPerLetter = std::stoi(argv[++i]);
            config.maxTrainingImages = config.trainingExamplesPerLetter * NUM_LETTERS;
        } else if (arg == "--threads" && i + 1 < argc) {
            config.numThreads = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --record <filename>    Record spikes to file" << std::endl;
            std::cout << "  --examples <n>         Training examples per letter (default: 200)" << std::endl;
            std::cout << "  --threads <n>          Number of spike processor threads (default: 20)" << std::endl;
            std::cout << "  --help                 Show this help message" << std::endl;
            return 0;
        }
    }

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Training examples per letter: " << config.trainingExamplesPerLetter << std::endl;
    std::cout << "  Spike processor threads: " << config.numThreads << std::endl;
    std::cout << "  Recording: " << (simConfig.enableRecording ? "enabled" : "disabled") << std::endl;
    std::cout << "  Real-time sync: disabled (fast mode)" << std::endl;
    std::cout << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();

    // ========================================================================
    // Initialize Datastore
    // ========================================================================
    std::cout << "Initializing datastore..." << std::endl;
    Datastore datastore("./emnist_training_db", 1000000);
    NeuralObjectFactory factory;
    NetworkInspector inspector;

    // ========================================================================
    // Initialize Activity Monitor and Recording
    // ========================================================================
    ActivityMonitor activityMonitor(datastore, simConfig);
    RecordingManager* recordingManager = nullptr;
    if (simConfig.enableRecording) {
        recordingManager = new RecordingManager();
        // Use streaming mode to write spikes directly to disk (avoids memory issues with large recordings)
        activityMonitor.setRecordingManager(recordingManager, true, simConfig.recordingFilename);
        std::cout << "Recording enabled (streaming to file): " << simConfig.recordingFilename << std::endl;
    }

    // ========================================================================
    // Load EMNIST Dataset
    // ========================================================================
    std::cout << "\nLoading EMNIST Letters dataset..." << std::endl;
    EMNISTLoader trainLoader(EMNISTLoader::Variant::LETTERS);
    if (!trainLoader.load(config.trainImagesPath, config.trainLabelsPath,
                         config.maxTrainingImages, true)) {
        std::cerr << "Failed to load training data!" << std::endl;
        return 1;
    }
    std::cout << "Loaded " << trainLoader.size() << " training images" << std::endl;

    EMNISTLoader testLoader(EMNISTLoader::Variant::LETTERS);
    if (!testLoader.load(config.testImagesPath, config.testLabelsPath, 0, true)) {
        std::cerr << "Failed to load test data!" << std::endl;
        return 1;
    }
    std::cout << "Loaded " << testLoader.size() << " test images" << std::endl;

    // ========================================================================
    // Build Network Architecture
    // ========================================================================
    std::cout << "\nBuilding 6-layer hierarchical V1 architecture..." << std::endl;
    std::cout << "  " << config.numColumns << " cortical columns ("
              << config.numOrientations << " orientations × "
              << config.numFrequencies << " frequencies)" << std::endl;

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

            // Create Layer 1 (Apical dendrites, modulatory)
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

            // Create Layer 2/3 (Superficial pyramidal)
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

            corticalColumns.push_back(col);
            colIdx++;
        }
    }

    // Create remaining layers (L4, L5, L6) for all columns
    std::cout << "Creating remaining layers (L4, L5, L6) for all columns..." << std::endl;
    for (size_t i = 0; i < corticalColumns.size(); ++i) {
        auto& col = corticalColumns[i];
        float xPos = (static_cast<int>(i) - config.numColumns / 2.0f) * 30.0f;

        // Create Layer 4 (Granular input layer)
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

        // Create Layer 5 (Deep pyramidal output)
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

        // Create Layer 6 (Corticothalamic feedback)
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
    }

    std::cout << "✓ Created " << corticalColumns.size() << " cortical columns with 6 layers each" << std::endl;
    int neuronsPerColumn = config.layer1Neurons + config.layer23Neurons +
                          (config.layer4Size * config.layer4Size) +
                          config.layer5Neurons + config.layer6Neurons;
    std::cout << "  Total neurons per column: " << neuronsPerColumn << std::endl;

    // Create output layer
    std::cout << "\nCreating output layer..." << std::endl;
    auto outputLayer = factory.createLayer();
    outputLayer->setPosition(0.0f, 0.0f, 50.0f);
    datastore.put(outputLayer);

    std::vector<std::vector<std::shared_ptr<Neuron>>> outputPopulations(NUM_LETTERS);
    std::vector<std::shared_ptr<Cluster>> outputClusters;

    for (int i = 0; i < NUM_LETTERS; ++i) {
        char letter = 'A' + i;
        auto cluster = factory.createCluster();
        cluster->setPosition((i - NUM_LETTERS / 2.0f) * 5.0f, 0.0f, 50.0f);
        outputLayer->addCluster(cluster->getId());

        for (int j = 0; j < config.neuronsPerOutputClass; ++j) {
            auto neuron = factory.createNeuron(config.neuronWindow, config.neuronThreshold, config.neuronMaxPatterns);
            neuron->setPosition((i - NUM_LETTERS / 2.0f) * 5.0f + (std::rand() % 100 - 50) / 100.0f,
                               (std::rand() % 100 - 50) / 100.0f,
                               50.0f + (std::rand() % 100 - 50) / 100.0f);
            auto axon = factory.createAxon(neuron->getId());
            auto dendrite = factory.createDendrite(neuron->getId());
            neuron->setAxonId(axon->getId());
            neuron->addDendrite(dendrite->getId());

            outputPopulations[i].push_back(neuron);
            cluster->addNeuron(neuron->getId());
            allNeuronIds.push_back(neuron->getId());

            datastore.put(neuron);
            datastore.put(axon);
            datastore.put(dendrite);
        }

        outputClusters.push_back(cluster);
        datastore.put(cluster);
    }
    datastore.put(outputLayer);

    std::cout << "✓ Created output layer with " << NUM_LETTERS << " letter populations" << std::endl;

    // Create inter-layer connections
    std::cout << "\nCreating inter-layer connections..." << std::endl;
    std::uniform_real_distribution<> dis(0.0, 1.0);
    int totalSynapses = 0;
    std::vector<std::shared_ptr<Synapse>> allSynapses;  // Track all synapses for registration

    for (auto& col : corticalColumns) {
        // L4 → L2/3 connections
        for (auto& l4Neuron : col.layer4Neurons) {
            auto axon = datastore.getAxon(l4Neuron->getAxonId());
            if (!axon) continue;

            for (auto& l23Neuron : col.layer23Neurons) {
                if (dis(gen) < config.layer4ToLayer23Prob) {
                    auto synapse = factory.createSynapse(
                        axon->getId(),
                        l23Neuron->getDendriteIds()[0],
                        config.initialWeight,
                        config.maxWeight
                    );
                    axon->addSynapse(synapse->getId());
                    allSynapses.push_back(synapse);
                    datastore.put(synapse);
                    totalSynapses++;
                }
            }
            datastore.put(axon);
        }

        // L4 → L5 connections (direct, bypassing L2/3)
        for (auto& l4Neuron : col.layer4Neurons) {
            auto axon = datastore.getAxon(l4Neuron->getAxonId());
            if (!axon) continue;

            for (auto& l5Neuron : col.layer5Neurons) {
                if (dis(gen) < config.layer4ToLayer5Prob) {
                    auto synapse = factory.createSynapse(
                        axon->getId(),
                        l5Neuron->getDendriteIds()[0],
                        config.initialWeight,
                        config.maxWeight
                    );
                    axon->addSynapse(synapse->getId());
                    allSynapses.push_back(synapse);
                    datastore.put(synapse);
                    totalSynapses++;
                }
            }
            datastore.put(axon);
        }

        // L2/3 → L5 connections
        for (auto& l23Neuron : col.layer23Neurons) {
            auto axon = datastore.getAxon(l23Neuron->getAxonId());
            if (!axon) continue;

            for (auto& l5Neuron : col.layer5Neurons) {
                if (dis(gen) < config.layer23ToLayer5Prob) {
                    auto synapse = factory.createSynapse(
                        axon->getId(),
                        l5Neuron->getDendriteIds()[0],
                        config.initialWeight,
                        config.maxWeight
                    );
                    axon->addSynapse(synapse->getId());
                    allSynapses.push_back(synapse);
                    datastore.put(synapse);
                    totalSynapses++;
                }
            }
            datastore.put(axon);
        }

        // L5 → L6 connections
        for (auto& l5Neuron : col.layer5Neurons) {
            auto axon = datastore.getAxon(l5Neuron->getAxonId());
            if (!axon) continue;

            for (auto& l6Neuron : col.layer6Neurons) {
                if (dis(gen) < config.layer5ToLayer6Prob) {
                    auto synapse = factory.createSynapse(
                        axon->getId(),
                        l6Neuron->getDendriteIds()[0],
                        config.initialWeight,
                        config.maxWeight
                    );
                    axon->addSynapse(synapse->getId());
                    allSynapses.push_back(synapse);
                    datastore.put(synapse);
                    totalSynapses++;
                }
            }
            datastore.put(axon);
        }

        // L6 → L4 feedback connections
        for (auto& l6Neuron : col.layer6Neurons) {
            auto axon = datastore.getAxon(l6Neuron->getAxonId());
            if (!axon) continue;

            for (auto& l4Neuron : col.layer4Neurons) {
                if (dis(gen) < config.layer6ToLayer4Prob) {
                    auto synapse = factory.createSynapse(
                        axon->getId(),
                        l4Neuron->getDendriteIds()[0],
                        config.initialWeight,
                        config.maxWeight
                    );
                    axon->addSynapse(synapse->getId());
                    allSynapses.push_back(synapse);
                    datastore.put(synapse);
                    totalSynapses++;
                }
            }
            datastore.put(axon);
        }
    }
    std::cout << "  ✓ Created " << totalSynapses << " inter-layer synapses" << std::endl;

    // Create Layer 5 → Output connections
    std::cout << "Creating Layer 5 → Output connections..." << std::endl;
    int outputSynapses = 0;
    for (auto& col : corticalColumns) {
        for (auto& l5Neuron : col.layer5Neurons) {
            auto axon = datastore.getAxon(l5Neuron->getAxonId());
            if (!axon) continue;

            for (int i = 0; i < NUM_LETTERS; ++i) {
                for (auto& outputNeuron : outputPopulations[i]) {
                    if (dis(gen) < 0.5) {  // 50% connectivity to output
                        auto synapse = factory.createSynapse(
                            axon->getId(),
                            outputNeuron->getDendriteIds()[0],
                            config.initialWeight,
                            config.maxWeight
                        );
                        axon->addSynapse(synapse->getId());
                        allSynapses.push_back(synapse);
                        datastore.put(synapse);
                        outputSynapses++;
                    }
                }
            }
            datastore.put(axon);
        }
    }
    std::cout << "  ✓ Created " << outputSynapses << " Layer 5 → Output synapses" << std::endl;
    std::cout << "  Total synapses: " << (totalSynapses + outputSynapses) << std::endl;

    // Initialize spike processor
    std::cout << "\nInitializing spike processor..." << std::endl;
    auto spikeProcessor = std::make_shared<SpikeProcessor>(10000, config.numThreads);

    // Set real-time sync mode
    spikeProcessor->setRealTimeSync(simConfig.realTimeSync);
    if (simConfig.realTimeSync) {
        std::cout << "Real-time synchronization enabled (1ms = 1ms)" << std::endl;
    } else {
        std::cout << "Real-time synchronization disabled (fast mode)" << std::endl;
    }

    auto networkPropagator = std::make_shared<NetworkPropagator>(spikeProcessor);
    spikeProcessor->start();

    // Register all neurons with NetworkPropagator
    std::cout << "Registering neurons with NetworkPropagator..." << std::endl;
    for (uint64_t neuronId : allNeuronIds) {
        auto neuron = datastore.getNeuron(neuronId);
        if (neuron) {
            networkPropagator->registerNeuron(neuron);
            neuron->setNetworkPropagator(networkPropagator);

            auto axon = datastore.getAxon(neuron->getAxonId());
            if (axon) {
                networkPropagator->registerAxon(axon);
            }

            for (uint64_t dendriteId : neuron->getDendriteIds()) {
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

    // ========================================================================
    // Export Network Structure for Visualization
    // ========================================================================
    if (simConfig.enableRecording) {
        std::cout << "\nExporting network structure for visualization..." << std::endl;

        // Flush all objects to datastore to ensure they're available
        std::cout << "  Flushing datastore..." << std::endl;
        size_t flushed = datastore.flushAll();
        std::cout << "  ✓ Flushed " << flushed << " objects to disk" << std::endl;

        // Create NetworkDataAdapter and extract network
        NetworkDataAdapter adapter(datastore, inspector, &activityMonitor);
        std::cout << "  Extracting network from brain (ID: " << brain->getId() << ")..." << std::endl;

        if (!adapter.extractNetwork(brain->getId())) {
            std::cerr << "  WARNING: Failed to extract network structure!" << std::endl;
        } else {
            std::cout << "  ✓ Extracted " << adapter.getNeurons().size() << " neurons and "
                      << adapter.getSynapses().size() << " synapses" << std::endl;

            // Compute layout for visualization
            std::cout << "  Computing hierarchical grouped layout..." << std::endl;
            LayoutEngine layoutEngine;
            LayoutConfig layoutConfig;
            layoutConfig.algorithm = LayoutAlgorithm::HIERARCHICAL_GROUPED;
            layoutConfig.layerSpacing = 50.0f;
            layoutConfig.columnSpacing = 100.0f;
            layoutConfig.clusterSpacing = 15.0f;
            layoutConfig.neuronSpacing = 2.0f;
            layoutConfig.overrideStoredPositions = false;  // Respect manually set positions
            layoutEngine.computeLayout(adapter, layoutConfig);
            adapter.updateSynapsePositions();
            std::cout << "  ✓ Layout computed" << std::endl;

            // Export to .snnw file
            std::string networkFilename = simConfig.recordingFilename;
            // Replace .snnr extension with .snnw
            size_t dotPos = networkFilename.find_last_of('.');
            if (dotPos != std::string::npos) {
                networkFilename = networkFilename.substr(0, dotPos) + ".snnw";
            } else {
                networkFilename += ".snnw";
            }

            std::cout << "  Saving network structure to: " << networkFilename << std::endl;
            if (adapter.exportNetworkStructure(networkFilename, "EMNIST Letters V1 Network")) {
                std::cout << "  ✓ Network structure exported successfully" << std::endl;
            } else {
                std::cerr << "  WARNING: Failed to export network structure!" << std::endl;
            }
        }
    }

    // ========================================================================
    // Training Phase
    // ========================================================================
    std::cout << "\n=== Starting Training Phase ===" << std::endl;
    std::vector<int> trainCount(NUM_LETTERS, 0);
    int totalPatternsLearned = 0;
    int imagesProcessed = 0;

    auto trainingStart = std::chrono::high_resolution_clock::now();

    for (size_t imgIdx = 0; imgIdx < trainLoader.size(); ++imgIdx) {
        const auto& emnistImg = trainLoader.getImage(imgIdx);
        int label = emnistImg.label - 1;  // Convert 1-26 to 0-25

        // Check if we need more training for this letter
        if (trainCount[label] >= config.trainingExamplesPerLetter) {
            continue;
        }

        // Clear all neuron spikes
        for (auto& neuronId : allNeuronIds) {
            auto n = datastore.getNeuron(neuronId);
            if (n) n->clearSpikes();
        }

        // Apply Gabor filters and fire Layer 4 neurons
        // Get current time and add buffer for processing (200ms like visualization experiment)
        double baseTime = spikeProcessor->getCurrentTime() + 200.0;

        std::vector<std::shared_ptr<Neuron>> layer5Neurons;

        for (size_t colIdx = 0; colIdx < corticalColumns.size(); ++colIdx) {
            auto& col = corticalColumns[colIdx];

            // Apply Gabor filter for this column's orientation and frequency
            auto gaborResponse = applyGaborToImage(emnistImg, col.gaborKernel, config.layer4Size);

            // Collect active Layer 4 neurons
            std::vector<std::pair<size_t, double>> activeL4;
            for (size_t neuronIdx = 0; neuronIdx < col.layer4Neurons.size() && neuronIdx < gaborResponse.size(); ++neuronIdx) {
                if (gaborResponse[neuronIdx] > config.gaborThreshold) {
                    activeL4.push_back({neuronIdx, gaborResponse[neuronIdx]});
                }
            }

            // Fire Layer 4 neurons
            for (const auto& [neuronIdx, response] : activeL4) {
                double firingTime = baseTime + (1.0 - response) * 10.0;
                col.layer4Neurons[neuronIdx]->fireSignature(firingTime);
                networkPropagator->fireNeuron(col.layer4Neurons[neuronIdx]->getId(), firingTime);
            }

            // Fire Layer 5 neurons based on which Layer 4 neurons were active
            // KEY: different L4 patterns → different L5 neurons fire → different output patterns
            for (size_t i = 0; i < activeL4.size() && i < col.layer5Neurons.size(); ++i) {
                size_t l4Idx = activeL4[i].first;
                size_t l5Idx = l4Idx % col.layer5Neurons.size();  // Map L4 index to L5 index

                auto& l5Neuron = col.layer5Neurons[l5Idx];
                double l5FireTime = baseTime + 15.0 + (colIdx * 1.5) + (i * 0.2);
                l5Neuron->fireSignature(l5FireTime);
                l5Neuron->fireAndAcknowledge(l5FireTime);
                networkPropagator->fireNeuron(l5Neuron->getId(), l5FireTime);
                l5Neuron->learnCurrentPattern();
            }

            // Collect all Layer 5 neurons for pattern copying
            layer5Neurons.insert(layer5Neurons.end(), col.layer5Neurons.begin(), col.layer5Neurons.end());
        }

        // Copy Layer 5 spike pattern to output neurons for this letter
        copyLayerSpikePattern(layer5Neurons, outputPopulations[label]);

        // Train output neurons for this letter
        int neuronsWithSpikes = 0;
        for (auto& outputNeuron : outputPopulations[label]) {
            if (!outputNeuron->getSpikes().empty()) {
                outputNeuron->fireAndAcknowledge(baseTime + 20.0);
                networkPropagator->fireNeuron(outputNeuron->getId(), baseTime + 20.0);
                outputNeuron->learnCurrentPattern();
                totalPatternsLearned++;
                neuronsWithSpikes++;
            }
        }

        // Debug: Print info for first few images
        if (imagesProcessed < 3) {
            int totalL5Spikes = 0;
            for (auto& l5n : layer5Neurons) {
                totalL5Spikes += l5n->getSpikes().size();
            }

            std::cout << "  Image " << imagesProcessed << " (label=" << label << "): "
                      << "L5 total spikes=" << totalL5Spikes
                      << ", Output neurons with spikes=" << neuronsWithSpikes
                      << "/" << outputPopulations[label].size() << std::endl;
        }

        trainCount[label]++;
        imagesProcessed++;

        // Print progress every 100 images
        if (imagesProcessed % 100 == 0) {
            printProgress("Training", imagesProcessed, config.maxTrainingImages);
        }

        // Check if training is complete
        bool allLettersTrained = true;
        for (int count : trainCount) {
            if (count < config.trainingExamplesPerLetter) {
                allLettersTrained = false;
                break;
            }
        }
        if (allLettersTrained) {
            break;
        }
    }

    auto trainingEnd = std::chrono::high_resolution_clock::now();
    double trainingTime = std::chrono::duration<double>(trainingEnd - trainingStart).count();

    std::cout << std::endl;  // New line after progress bar
    std::cout << "\n=== Training Complete ===" << std::endl;
    std::cout << "Total patterns learned: " << totalPatternsLearned << std::endl;
    std::cout << "Training time: " << std::fixed << std::setprecision(2) << trainingTime << " seconds" << std::endl;
    std::cout << "Images/second: " << std::fixed << std::setprecision(1)
              << (imagesProcessed / trainingTime) << std::endl;

    // ========================================================================
    // Testing Phase
    // ========================================================================
    std::cout << "\n=== Starting Testing Phase ===" << std::endl;
    int testCorrect = 0;
    int testTotal = 0;

    auto testingStart = std::chrono::high_resolution_clock::now();

    for (size_t testIdx = 0; testIdx < testLoader.size(); ++testIdx) {
        const auto& emnistImg = testLoader.getImage(testIdx);
        int label = emnistImg.label - 1;  // Convert 1-26 to 0-25

        // Clear all neuron spikes
        for (auto& neuronId : allNeuronIds) {
            auto n = datastore.getNeuron(neuronId);
            if (n) n->clearSpikes();
        }

        // Apply Gabor filters and fire Layer 4 neurons (same as training)
        double baseTime = spikeProcessor->getCurrentTime() + 200.0;

        std::vector<std::shared_ptr<Neuron>> layer5Neurons;

        for (size_t colIdx = 0; colIdx < corticalColumns.size(); ++colIdx) {
            auto& col = corticalColumns[colIdx];

            // Apply Gabor filter for this column's orientation and frequency
            auto gaborResponse = applyGaborToImage(emnistImg, col.gaborKernel, config.layer4Size);

            // Collect active Layer 4 neurons
            std::vector<std::pair<size_t, double>> activeL4;
            for (size_t neuronIdx = 0; neuronIdx < col.layer4Neurons.size() && neuronIdx < gaborResponse.size(); ++neuronIdx) {
                if (gaborResponse[neuronIdx] > config.gaborThreshold) {
                    activeL4.push_back({neuronIdx, gaborResponse[neuronIdx]});
                }
            }

            // Fire Layer 4 neurons
            for (const auto& [neuronIdx, response] : activeL4) {
                double firingTime = baseTime + (1.0 - response) * 10.0;
                col.layer4Neurons[neuronIdx]->fireSignature(firingTime);
                networkPropagator->fireNeuron(col.layer4Neurons[neuronIdx]->getId(), firingTime);
            }

            // Fire Layer 5 neurons based on which Layer 4 neurons were active (same as training)
            for (size_t i = 0; i < activeL4.size() && i < col.layer5Neurons.size(); ++i) {
                size_t l4Idx = activeL4[i].first;
                size_t l5Idx = l4Idx % col.layer5Neurons.size();

                auto& l5Neuron = col.layer5Neurons[l5Idx];
                double l5FireTime = baseTime + 15.0 + (colIdx * 1.5) + (i * 0.2);
                l5Neuron->fireSignature(l5FireTime);
                l5Neuron->fireAndAcknowledge(l5FireTime);
                networkPropagator->fireNeuron(l5Neuron->getId(), l5FireTime);
            }

            // Collect all Layer 5 neurons for pattern copying
            layer5Neurons.insert(layer5Neurons.end(), col.layer5Neurons.begin(), col.layer5Neurons.end());
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

        // Debug: Print info for first few test images
        if (testIdx < 3) {
            std::cout << "  Test " << testIdx << " (label=" << label << "): "
                      << "L5 neurons=" << layer5Neurons.size()
                      << ", Predicted=" << predictedLabel
                      << ", MaxSim=" << std::fixed << std::setprecision(3) << maxSimilarity
                      << std::endl;
        }

        // Check if prediction is correct
        testTotal++;
        if (predictedLabel == label) {
            testCorrect++;
        }

        // Print progress every 1000 images
        if (testTotal % 1000 == 0) {
            double accuracy = (testTotal > 0) ? (100.0 * testCorrect / testTotal) : 0.0;
            printProgress("Testing", testTotal, testLoader.size(), accuracy);
        }
    }

    auto testingEnd = std::chrono::high_resolution_clock::now();
    double testingTime = std::chrono::duration<double>(testingEnd - testingStart).count();
    double testAccuracy = (testTotal > 0) ? (100.0 * testCorrect / testTotal) : 0.0;

    std::cout << std::endl;  // New line after progress bar
    std::cout << "\n=== Testing Complete ===" << std::endl;
    std::cout << "Test Accuracy: " << std::fixed << std::setprecision(2) << testAccuracy << "% ("
              << testCorrect << "/" << testTotal << ")" << std::endl;
    std::cout << "Testing time: " << std::fixed << std::setprecision(2) << testingTime << " seconds" << std::endl;
    std::cout << "Images/second: " << std::fixed << std::setprecision(1)
              << (testTotal / testingTime) << std::endl;

    // ========================================================================
    // Save Recording (if enabled)
    // ========================================================================
    if (recordingManager) {
        std::cout << "\nSaving recording..." << std::endl;
        // In streaming mode, saveRecording() will just finalize the file
        if (activityMonitor.saveRecording(simConfig.recordingFilename)) {
            auto metadata = recordingManager->getMetadata();
            std::cout << "Recording saved: " << simConfig.recordingFilename << std::endl;
            std::cout << "  Duration: " << metadata.duration << " ms" << std::endl;
            std::cout << "  Spikes: " << metadata.spikeCount << std::endl;
            std::cout << "  Neurons: " << metadata.neuronCount << std::endl;
        } else {
            std::cerr << "Failed to save recording!" << std::endl;
        }
    }

    // ========================================================================
    // Final Statistics
    // ========================================================================
    auto endTime = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double>(endTime - startTime).count();

    std::cout << "\n=== Final Statistics ===" << std::endl;
    std::cout << "Total runtime: " << std::fixed << std::setprecision(2) << totalTime << " seconds" << std::endl;
    std::cout << "Training images: " << imagesProcessed << std::endl;
    std::cout << "Testing images: " << testTotal << std::endl;
    std::cout << "Patterns learned: " << totalPatternsLearned << std::endl;
    std::cout << "Final accuracy: " << std::fixed << std::setprecision(2) << testAccuracy << "%" << std::endl;

    std::cout << "\nPer-letter training counts:" << std::endl;
    for (int i = 0; i < NUM_LETTERS; ++i) {
        char letter = 'A' + i;
        std::cout << "  " << letter << ": " << trainCount[i] << " patterns" << std::endl;
    }

    std::cout << "\n=== Experiment Complete ===" << std::endl;

    // Cleanup
    if (recordingManager) {
        delete recordingManager;
    }

    return 0;
}

