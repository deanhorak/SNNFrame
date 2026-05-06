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
#include <cstdint>
#include <array>
#include <set>
#include <limits>
#include <unordered_set>
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
    unsigned int seed = 0; // Seed for random number generator
    std::string trainLabelsPath = "/home/dean/repos/snnfw/data/EMNIST/emnist-letters-train-labels-idx1-ubyte";
    std::string testImagesPath = "/home/dean/repos/snnfw/data/EMNIST/emnist-letters-test-images-idx3-ubyte";
    std::string testLabelsPath = "/home/dean/repos/snnfw/data/EMNIST/emnist-letters-test-labels-idx1-ubyte";
    std::string datastorePath = "./emnist_training_db";

    // Network parameters (matching 71% config)
    double neuronWindow = 500.0;  // Larger window for richer patterns
    double neuronThreshold = 1.2;   // Higher threshold for sparser firing
    int neuronMaxPatterns = 500;  // More patterns per neuron
    double inputLatencyMs = 15.0;

    // Training parameters
    int trainingExamplesPerLetter = 200;   // More examples per letter to populate patterns
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

    int neuronsPerOutputClass = 3;   // Fewer output neurons to reduce saturation

    // Inter-layer connectivity
    double layer4ToLayer23Prob = 0.25;
    double layer4ToLayer5Prob = 0.002;   // Direct L4 → L5 drive (very sparse)
    double layer23ToLayer5Prob = 0.002;
    double layer5ToLayer6Prob = 0.3;
    double layer6ToLayer4Prob = 0.2;

    // Synaptic parameters
    double initialWeight = 0.1;
    double maxWeight = 0.5;
    // Output-layer specific connectivity
    double outputConnectProb = 0.02;    // stronger L5 → output fan-out for supervised binding
    double outputInitialWeight = 0.02;  // stronger initial weight for output synapses

    // Input/competition controls
    double pixelThreshold = 0.4;
    int maskMinActive = 6;
    int tilesPerColumn = 3;
    int l4Keep = 8;
    int l5Keep = 8;
    double l4RowDelay = 0.3;
    double l4ColDelay = 0.2;
    double interImageGapMs = 550.0;
    double l4TargetRate = 8.0;
    double l5TargetRate = 5.0;
    double outputTargetRate = 2.0;
    double stdpLtdScale = 0.3;
    double stdpLtdWindowMs = 70.0;
    double l4PostShiftMs = -6.0;
    double l4PostJitterMs = 2.0;
    bool enableL5Inhibition = true;
    double l5InhibitLoser = 1.2;
    double l5InhibitThreshold = 0.5;
    int l5MinSpikes = 1;
    bool traceStdp = true;
    bool enableOutputCompetition = true;
    int outputCompetitionKeep = 3;
    int outputCompetitionMinSpikes = 3;
    std::array<bool, NUM_LETTERS> includeLetters{};
};

std::array<bool, NUM_LETTERS> gIncludeLetters{};

// Cortical column structure
struct CorticalColumn {
    double orientation;
    double spatialFrequency;
    std::string featureType;
    std::vector<std::vector<double>> gaborKernel;
    std::vector<double> inputMask; // column-specific input weighting
    std::vector<int> inputMaskActiveIdx; // indices with non-zero weight for fast gating
    std::vector<int> tileIndices; // selected tiles for this column (input/L4 alignment)
    double thresholdScale = 1.0; // per-column threshold scaling

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

struct WeightStats {
    size_t count = 0;
    double mean = 0.0;
    double min = 0.0;
    double max = 0.0;
};

using L5CountVector = std::vector<uint16_t>;

// Store L5 spike-count patterns for each letter.
// NOTE: We cap history per pass to bound memory/compute for k-NN.
std::vector<L5CountVector> letterL5Patterns[NUM_LETTERS];

// Store L5 neuron frequency counts for each letter (for centroid-based matching)
const size_t TOTAL_L5_NEURONS = 1280;  // 16 columns * 80 L5 neurons per column
const size_t MAX_PATTERNS_PER_LETTER = 1024;  // cap per-pass history
std::array<std::array<int, TOTAL_L5_NEURONS>, NUM_LETTERS> letterL5Counts;
std::array<int, NUM_LETTERS> letterPatternCounts;

/**
 * @brief Compute cosine similarity between two spike-count vectors
 */
double cosineSimilarity(const L5CountVector& a, const L5CountVector& b) {
    if (a.empty() || b.empty() || a.size() != b.size()) return 0.0;
    double dot = 0.0;
    double normA = 0.0;
    double normB = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double av = static_cast<double>(a[i]);
        double bv = static_cast<double>(b[i]);
        dot += av * bv;
        normA += av * av;
        normB += bv * bv;
    }
    if (normA <= 0.0 || normB <= 0.0) return 0.0;
    return dot / (std::sqrt(normA) * std::sqrt(normB));
}

void applyHomeostasis(const std::vector<std::shared_ptr<Neuron>>& neurons,
                      const std::string& label) {
    size_t adjusted = 0;
    for (const auto& neuron : neurons) {
        if (!neuron) continue;
        neuron->applyHomeostaticPlasticity();
        adjusted++;
    }
    if (adjusted > 0) {
        std::cout << "  Homeostasis applied (" << label << "): neurons="
                  << adjusted << std::endl;
    }
}

/**
 * @brief Compute weighted similarity between test pattern and letter centroid
 * Uses the frequency of each L5 neuron in training patterns as weights
 */
double centroidSimilarity(const L5CountVector& testCounts, int letter) {
    if (testCounts.empty() || letterPatternCounts[letter] == 0) return 0.0;

    double dot = 0.0;
    double normTest = 0.0;
    double normCentroid = 0.0;
    double invCount = 1.0 / static_cast<double>(letterPatternCounts[letter]);
    for (size_t idx = 0; idx < testCounts.size(); ++idx) {
        double testVal = static_cast<double>(testCounts[idx]);
        double centroidVal = static_cast<double>(letterL5Counts[letter][idx]) * invCount;
        dot += testVal * centroidVal;
        normTest += testVal * testVal;
        normCentroid += centroidVal * centroidVal;
    }
    if (normTest <= 0.0 || normCentroid <= 0.0) return 0.0;
    return dot / (std::sqrt(normTest) * std::sqrt(normCentroid));
}

/**
 * @brief Find best matching letter for a given L5 pattern using centroid similarity
 */
std::pair<int, double> findBestMatchingLetterCentroid(const L5CountVector& testCounts) {
    int bestLetter = -1;
    double bestSimilarity = -1.0;

    for (int letter = 0; letter < NUM_LETTERS; ++letter) {
        if (!gIncludeLetters[letter]) continue;
        double sim = centroidSimilarity(testCounts, letter);
        if (sim > bestSimilarity) {
            bestSimilarity = sim;
            bestLetter = letter;
        }
    }

    return {bestLetter, bestSimilarity};
}

/**
 * @brief Find best matching letter for a given L5 pattern using k-NN voting
 * Returns the letter with the most votes among the k most similar patterns
 */
std::pair<int, double> findBestMatchingLetter(const L5CountVector& testCounts) {
    const int K = 5;  // Number of nearest neighbors to consider

    // Collect all similarities with their letter labels
    std::vector<std::pair<double, int>> allSimilarities;

    for (int letter = 0; letter < NUM_LETTERS; ++letter) {
        if (!gIncludeLetters[letter]) continue;
        for (const auto& trainPattern : letterL5Patterns[letter]) {
            double sim = cosineSimilarity(testCounts, trainPattern);
            allSimilarities.push_back({sim, letter});
        }
    }

    // Sort by similarity (descending)
    std::sort(allSimilarities.begin(), allSimilarities.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    // Vote among top K neighbors
    std::array<int, NUM_LETTERS> votes = {};
    double maxSim = 0.0;
    int numVotes = std::min(K, static_cast<int>(allSimilarities.size()));

    for (int i = 0; i < numVotes; ++i) {
        votes[allSimilarities[i].second]++;
        if (i == 0) maxSim = allSimilarities[i].first;
    }

    // Find letter with most votes
    int bestLetter = -1;
    int maxVotes = 0;
    for (int letter = 0; letter < NUM_LETTERS; ++letter) {
        if (!gIncludeLetters[letter]) continue;
        if (votes[letter] > maxVotes) {
            maxVotes = votes[letter];
            bestLetter = letter;
        }
    }

    return {bestLetter, maxSim};
}

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
                          const std::vector<std::shared_ptr<Neuron>>& targetNeurons,
                          double currentTime) {
    for (auto& targetNeuron : targetNeurons) {
        targetNeuron->periodicMemoryCleanup(currentTime);

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
    config.includeLetters.fill(true);
    gIncludeLetters.fill(true);

    int testLimit = 0;  // 0 means use all test images
    bool enableFullPropagation = true;   // Default: full biological propagation
    bool keepL5History = false;          // Keep patterns across passes
    bool disableOutputTeach = false;     // Diagnostic: skip supervised output teaching
    bool disableOutputSignature = false; // Diagnostic: skip output neuron signatures
    bool enableOutputVote = true;        // Prefer output spike vote during testing
    int maxPasses = 20;                  // Cap on training passes

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--record" && i + 1 < argc) {
            simConfig.enableRecording = true;
            simConfig.recordingFilename = argv[++i];
        } else if (arg == "--full-propagation") {
            enableFullPropagation = true;  // Explicit enable (no-op, kept for compatibility)
        } else if (arg == "--bypass-propagation") {
            enableFullPropagation = false; // Legacy fast path (non-biological)
        } else if (arg == "--examples" && i + 1 < argc) {
            config.trainingExamplesPerLetter = std::stoi(argv[++i]);
            config.maxTrainingImages = config.trainingExamplesPerLetter * NUM_LETTERS;
        } else if (arg == "--test-limit" && i + 1 < argc) {
            testLimit = std::stoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            config.numThreads = std::stoi(argv[++i]);
        } else if (arg == "--keep-l5-history") {
            keepL5History = true;
        } else if (arg == "--no-output-teach") {
            disableOutputTeach = true;
        } else if (arg == "--no-output-signature") {
            disableOutputSignature = true;
        } else if (arg == "--no-output-vote") {
            enableOutputVote = false;
        } else if (arg == "--max-passes" && i + 1 < argc) {
            maxPasses = std::max(1, std::stoi(argv[++i]));
        } else if (arg == "--db-path" && i + 1 < argc) {
            config.datastorePath = argv[++i];
        } else if (arg == "--pixel-threshold" && i + 1 < argc) {
            config.pixelThreshold = std::stod(argv[++i]);
        } else if (arg == "--mask-min-active" && i + 1 < argc) {
            config.maskMinActive = std::stoi(argv[++i]);
        } else if (arg == "--tiles-per-column" && i + 1 < argc) {
            config.tilesPerColumn = std::stoi(argv[++i]);
        } else if (arg == "--l4-keep" && i + 1 < argc) {
            config.l4Keep = std::stoi(argv[++i]);
        } else if (arg == "--l5-keep" && i + 1 < argc) {
            config.l5Keep = std::stoi(argv[++i]);
        } else if (arg == "--l4-row-delay" && i + 1 < argc) {
            config.l4RowDelay = std::stod(argv[++i]);
        } else if (arg == "--l4-col-delay" && i + 1 < argc) {
            config.l4ColDelay = std::stod(argv[++i]);
        } else if (arg == "--inter-image-gap" && i + 1 < argc) {
            config.interImageGapMs = std::stod(argv[++i]);
        } else if (arg == "--l4-target-rate" && i + 1 < argc) {
            config.l4TargetRate = std::stod(argv[++i]);
        } else if (arg == "--l5-target-rate" && i + 1 < argc) {
            config.l5TargetRate = std::stod(argv[++i]);
        } else if (arg == "--output-target-rate" && i + 1 < argc) {
            config.outputTargetRate = std::stod(argv[++i]);
        } else if (arg == "--stdp-ltd-scale" && i + 1 < argc) {
            config.stdpLtdScale = std::stod(argv[++i]);
        } else if (arg == "--stdp-ltd-window" && i + 1 < argc) {
            config.stdpLtdWindowMs = std::stod(argv[++i]);
        } else if (arg == "--input-latency" && i + 1 < argc) {
            config.inputLatencyMs = std::stod(argv[++i]);
        } else if (arg == "--l4-post-shift" && i + 1 < argc) {
            config.l4PostShiftMs = std::stod(argv[++i]);
        } else if (arg == "--l4-post-jitter" && i + 1 < argc) {
            config.l4PostJitterMs = std::stod(argv[++i]);
        } else if (arg == "--l5-inhibit-losers") {
            config.enableL5Inhibition = true;
        } else if (arg == "--no-l5-inhibit") {
            config.enableL5Inhibition = false;
        } else if (arg == "--l5-inhibit-amount" && i + 1 < argc) {
            config.l5InhibitLoser = std::stod(argv[++i]);
        } else if (arg == "--l5-inhibit-threshold" && i + 1 < argc) {
            config.l5InhibitThreshold = std::stod(argv[++i]);
        } else if (arg == "--l5-min-spikes" && i + 1 < argc) {
            config.l5MinSpikes = std::stoi(argv[++i]);
        } else if (arg == "--trace-stdp") {
            config.traceStdp = true;
        } else if (arg == "--no-trace-stdp") {
            config.traceStdp = false;
        } else if (arg == "--output-connect" && i + 1 < argc) {
            config.outputConnectProb = std::stod(argv[++i]);
        } else if (arg == "--output-weight" && i + 1 < argc) {
            config.outputInitialWeight = std::stod(argv[++i]);
        } else if (arg == "--output-competition") {
            config.enableOutputCompetition = true;
        } else if (arg == "--no-output-competition") {
            config.enableOutputCompetition = false;
        } else if (arg == "--output-keep" && i + 1 < argc) {
            config.outputCompetitionKeep = std::stoi(argv[++i]);
        } else if (arg == "--output-min-spikes" && i + 1 < argc) {
            config.outputCompetitionMinSpikes = std::stoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            config.seed = std::stoul(argv[++i]);
        } else if (arg == "--letters" && i + 1 < argc) {
            std::string letters = argv[++i];
            config.includeLetters.fill(false);
            for (char ch : letters) {
                if (ch >= 'a' && ch <= 'z') ch = static_cast<char>(ch - 'a' + 'A');
                if (ch < 'A' || ch > 'Z') continue;
                int idx = ch - 'A';
                if (idx >= 0 && idx < NUM_LETTERS) {
                    config.includeLetters[idx] = true;
                }
            }
            gIncludeLetters = config.includeLetters;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --record <filename>    Record spikes to file" << std::endl;
            std::cout << "  --full-propagation     Enable full network propagation (default)" << std::endl;
            std::cout << "  --bypass-propagation   Use fast non-biological bypass (no synaptic propagation)" << std::endl;
            std::cout << "  --examples <n>         Training examples per letter (default: 200)" << std::endl;
            std::cout << "  --test-limit <n>       Limit test images (default: 0 = all)" << std::endl;
            std::cout << "  --threads <n>          Number of spike processor threads (default: 20)" << std::endl;
            std::cout << "  --keep-l5-history      Do not clear L5 pattern memory between passes" << std::endl;
            std::cout << "  --no-output-teach      Skip supervised output teaching (diagnostic)" << std::endl;
            std::cout << "  --no-output-signature  Disable output neuron temporal signatures (diagnostic)" << std::endl;
            std::cout << "  --no-output-vote       Skip output spike voting during testing" << std::endl;
            std::cout << "  --max-passes <n>       Max training passes (default: 20)" << std::endl;
            std::cout << "  --db-path <dir>        Datastore directory (default: ./emnist_training_db)" << std::endl;
            std::cout << "  --pixel-threshold <v>  Input pixel threshold (default: 0.4)" << std::endl;
            std::cout << "  --mask-min-active <n>  Min active masked pixels per column (default: 6)" << std::endl;
            std::cout << "  --tiles-per-column <n> Tiles per column receptive field (default: 3)" << std::endl;
            std::cout << "  --l4-keep <n>          L4 winners per column (default: 8)" << std::endl;
            std::cout << "  --l5-keep <n>          L5 winners per column (default: 8)" << std::endl;
            std::cout << "  --l4-row-delay <v>     L4 row delay scale (default: 0.3)" << std::endl;
            std::cout << "  --l4-col-delay <v>     L4 col delay scale (default: 0.2)" << std::endl;
            std::cout << "  --inter-image-gap <v>  Gap between images in ms (default: 550)" << std::endl;
            std::cout << "  --l4-target-rate <v>   L4 target firing rate in Hz (default: 8)" << std::endl;
            std::cout << "  --l5-target-rate <v>   L5 target firing rate in Hz (default: 5)" << std::endl;
            std::cout << "  --output-target-rate <v> Output target firing rate in Hz (default: 2)" << std::endl;
            std::cout << "  --stdp-ltd-scale <v>   Scale factor for LTD updates (default: 0.3)" << std::endl;
            std::cout << "  --stdp-ltd-window <v>  LTD window in ms for trace STDP (default: 70)" << std::endl;
            std::cout << "  --input-latency <v>    Input spike latency window in ms (default: 15)" << std::endl;
            std::cout << "  --l4-post-shift <v>    L4 post-spike timing shift in ms (default: -6)" << std::endl;
            std::cout << "  --l4-post-jitter <v>   L4 post-spike timing jitter in ms (default: 2)" << std::endl;
            std::cout << "  --l5-inhibit-losers    Enable lateral inhibition for L5 (default)" << std::endl;
            std::cout << "  --no-l5-inhibit        Disable lateral inhibition for L5" << std::endl;
            std::cout << "  --l5-inhibit-amount <v> Inhibition applied to L5 losers (default: 1.2)" << std::endl;
            std::cout << "  --l5-inhibit-threshold <v> Max inhibition to allow firing (default: 0.5)" << std::endl;
            std::cout << "  --l5-min-spikes <n>    Min spikes for L5 to compete (default: 1)" << std::endl;
            std::cout << "  --trace-stdp           Enable trace-based STDP (default)" << std::endl;
            std::cout << "  --no-trace-stdp        Disable trace-based STDP" << std::endl;
            std::cout << "  --output-connect <v>   L5->output connect probability (default: 0.02)" << std::endl;
            std::cout << "  --output-weight <v>    L5->output initial weight (default: 0.02)" << std::endl;
            std::cout << "  --output-competition   Enable output winner-take-most masking (default)" << std::endl;
            std::cout << "  --no-output-competition Disable output competition masking" << std::endl;
            std::cout << "  --output-keep <n>      Output populations kept after competition (default: 3)" << std::endl;
            std::cout << "  --output-min-spikes <n> Min spikes per population to compete (default: 3)" << std::endl;
            std::cout << "  --seed <n>             Seed for random number generator (default: 0, for random)" << std::endl;
            std::cout << "  --letters <A,B,...>    Restrict training/testing to specific letters" << std::endl;
            std::cout << "  --help                 Show this help message" << std::endl;
            return 0;
        }
    }

    std::cout << "Configuration:" << std::endl;
    std::cout << "  Training examples per letter: " << config.trainingExamplesPerLetter << std::endl;
    std::cout << "  Spike processor threads: " << config.numThreads << std::endl;
    std::cout << "  Random seed: " << (config.seed == 0 ? "random" : std::to_string(config.seed)) << std::endl;
    std::cout << "  Recording: " << (simConfig.enableRecording ? "enabled" : "disabled") << std::endl;
    std::cout << "  Full propagation: " << (enableFullPropagation ? "enabled (biological)" : "bypassed (legacy fast mode)") << std::endl;
    std::cout << "  Keep L5 history across passes: " << (keepL5History ? "yes" : "no") << std::endl;
    std::cout << "  Output teaching: " << (disableOutputTeach ? "disabled (diagnostic)" : "enabled") << std::endl;
    std::cout << "  Max passes: " << maxPasses << std::endl;
    std::cout << "  Output competition: " << (config.enableOutputCompetition ? "enabled" : "disabled")
              << " (keep=" << config.outputCompetitionKeep
              << ", min-spikes=" << config.outputCompetitionMinSpikes << ")" << std::endl;
    std::cout << "  Letters: ";
    bool anyLetter = false;
    for (int i = 0; i < NUM_LETTERS; ++i) {
        if (config.includeLetters[i]) {
            std::cout << char('A' + i);
            anyLetter = true;
        }
    }
    if (!anyLetter) {
        std::cout << "(none)";
    }
    std::cout << std::endl;
    std::cout << "  Output signatures: " << (disableOutputSignature ? "disabled (diagnostic)" : "enabled") << std::endl;
    std::cout << "  Output vote during testing: " << (enableOutputVote ? "enabled" : "disabled") << std::endl;
    std::cout << "  Datastore path: " << config.datastorePath << std::endl;
    std::cout << "  Pixel threshold: " << config.pixelThreshold << std::endl;
    std::cout << "  Mask min active: " << config.maskMinActive << std::endl;
    std::cout << "  Tiles per column: " << config.tilesPerColumn << std::endl;
    std::cout << "  L4 keep: " << config.l4Keep << std::endl;
    std::cout << "  L5 keep: " << config.l5Keep << std::endl;
    std::cout << "  L4 delay (row,col): " << config.l4RowDelay << ", " << config.l4ColDelay << std::endl;
    std::cout << "  Inter-image gap: " << config.interImageGapMs << " ms" << std::endl;
    std::cout << "  Target rates (Hz) L4/L5/Output: " << config.l4TargetRate << "/"
              << config.l5TargetRate << "/" << config.outputTargetRate << std::endl;
    std::cout << "  STDP LTD scale/window: " << config.stdpLtdScale << "/"
              << config.stdpLtdWindowMs << " ms" << std::endl;
    std::cout << "  Input latency window (ms): " << config.inputLatencyMs << std::endl;
    std::cout << "  L4 post timing shift/jitter (ms): " << config.l4PostShiftMs << "/"
              << config.l4PostJitterMs << std::endl;
    std::cout << "  L5 inhibition: " << (config.enableL5Inhibition ? "enabled" : "disabled")
              << " (amount=" << config.l5InhibitLoser
              << ", threshold=" << config.l5InhibitThreshold
              << ", min-spikes=" << config.l5MinSpikes << ")" << std::endl;
    std::cout << "  Trace STDP: " << (config.traceStdp ? "enabled" : "disabled") << std::endl;
    std::cout << "  Output connect/weight: " << config.outputConnectProb << ", " << config.outputInitialWeight << std::endl;
    std::cout << "  Real-time sync: disabled (fast mode)" << std::endl;
    std::cout << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();

    // ========================================================================
    // Initialize Datastore
    // ========================================================================
    std::cout << "Initializing datastore..." << std::endl;
    Datastore datastore(config.datastorePath, 1000000);
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

    std::vector<CorticalColumn> corticalColumns;
    std::vector<uint64_t> allNeuronIds;
    std::vector<std::shared_ptr<Synapse>> allSynapses;  // Track all synapses for registration
    std::vector<uint64_t> synInputToL4Ids;
    std::vector<uint64_t> synL4ToL5Ids;
    std::vector<uint64_t> synL5ToOutputIds;
    int totalSynapses = 0;

    // Input layer (one neuron per pixel)
    const int INPUT_SIZE = 28;
    std::vector<std::shared_ptr<Neuron>> inputNeurons;
    inputNeurons.reserve(INPUT_SIZE * INPUT_SIZE);
    auto inputLayer = factory.createLayer();
    datastore.put(inputLayer);

    for (int r = 0; r < INPUT_SIZE; ++r) {
        for (int c = 0; c < INPUT_SIZE; ++c) {
            auto n = factory.createNeuron(config.neuronWindow, config.neuronThreshold, config.neuronMaxPatterns);
            auto axon = factory.createAxon(n->getId());
            auto dend = factory.createDendrite(n->getId());
            n->setAxonId(axon->getId());
            n->addDendrite(dend->getId());
            inputNeurons.push_back(n);
            allNeuronIds.push_back(n->getId());
            datastore.put(n);
            datastore.put(axon);
            datastore.put(dend);
        }
    }

    // Create cortical columns
    const double ORIENTATION_STEP = 180.0 / config.numOrientations;
    const std::vector<std::string> FREQ_NAMES = {"low_freq", "high_freq"};

    std::random_device rd;
    std::mt19937 gen(config.seed == 0 ? rd() : config.seed);
    std::uniform_real_distribution<double> l4PostJitterDist(-config.l4PostJitterMs, config.l4PostJitterMs);

    // Use a distribution for neuron positions
    std::uniform_real_distribution<> pos_dis(-1.0, 1.0);

    int colIdx = 0;
    for (int oriIdx = 0; oriIdx < config.numOrientations; ++oriIdx) {
        double orientation = oriIdx * ORIENTATION_STEP;

        for (int freqIdx = 0; freqIdx < config.numFrequencies; ++freqIdx) {
            CorticalColumn col;
            col.orientation = orientation;
            col.spatialFrequency = 1.0;
            col.featureType = "orientation_" + FREQ_NAMES[freqIdx];
            col.gaborKernel.clear();

            // Create column
            col.column = factory.createColumn();
            v1Nucleus->addColumn(col.column->getId());

            // Spatial position for this column
            float xPos = (colIdx - config.numColumns / 2.0f) * 30.0f;
            col.column->setPosition(xPos, 0.0f, 0.0f);

            // Column-specific tiled receptive field (4x4 tiles over 28x28 -> 7x7 patch per column)
            // This deterministically breaks symmetry across columns while keeping biological-like tiling.
            const int tilesPerSide = 4;
            const int tileSize = INPUT_SIZE / tilesPerSide; // 28/4 = 7
            const int tilesPerColumn = std::max(1, config.tilesPerColumn);
            col.inputMask.assign(INPUT_SIZE * INPUT_SIZE, 0.0);
            col.inputMaskActiveIdx.clear();

            // Deterministic per-column tile selection
            std::mt19937 tileGen(static_cast<uint32_t>(colIdx * 9973 + 17));
            std::vector<int> tileIndices(tilesPerSide * tilesPerSide);
            std::iota(tileIndices.begin(), tileIndices.end(), 0);
            std::shuffle(tileIndices.begin(), tileIndices.end(), tileGen);
            col.tileIndices.assign(tileIndices.begin(), tileIndices.begin() + tilesPerColumn);

            for (int tileIndex : col.tileIndices) {
                int tileRow = tileIndex / tilesPerSide;
                int tileCol = tileIndex % tilesPerSide;
                int startR = tileRow * tileSize;
                int startC = tileCol * tileSize;
                for (int r = 0; r < tileSize; ++r) {
                    for (int c = 0; c < tileSize; ++c) {
                        int rr = startR + r;
                        int cc = startC + c;
                        int idx = rr * INPUT_SIZE + cc;
                        if (idx >= 0 && idx < INPUT_SIZE * INPUT_SIZE && col.inputMask[idx] == 0.0) {
                            col.inputMask[idx] = 1.0;
                            col.inputMaskActiveIdx.push_back(idx);
                        }
                    }
                }
            }
            col.thresholdScale = 1.0;

            // Create Layer 1 (Apical dendrites, modulatory)
            col.layer1 = factory.createLayer();
            col.layer1->setPosition(xPos, 0.0f, 0.0f);
            col.column->addLayer(col.layer1->getId());
            auto layer1Cluster = factory.createCluster();
            col.layer1->addCluster(layer1Cluster->getId());

            for (int i = 0; i < config.layer1Neurons; ++i) {
                auto neuron = factory.createNeuron(config.neuronWindow, config.neuronThreshold, config.neuronMaxPatterns);
                neuron->setPosition(xPos + pos_dis(gen),
                                   pos_dis(gen),
                                   pos_dis(gen));
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
                neuron->setPosition(xPos + pos_dis(gen),
                                   pos_dis(gen),
                                   10.0f + pos_dis(gen));
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
        // Tile the input space so each column sees a distinct receptive field
        const int tilesPerSide = 4;
        const int tileSize = INPUT_SIZE / tilesPerSide; // 7 for 28x28
        for (int j = 0; j < LAYER4_NEURONS; ++j) {
            auto neuron = factory.createNeuron(config.neuronWindow, config.neuronThreshold, config.neuronMaxPatterns);
            neuron->setTargetFiringRate(config.l4TargetRate);
            neuron->setPosition(xPos + pos_dis(gen),
                               pos_dis(gen),
                               20.0f + pos_dis(gen));
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

            // Connect input pixels to this L4 neuron within the column's tile
            int tileChoice = 0;
            if (!col.tileIndices.empty()) {
                tileChoice = col.tileIndices[j % static_cast<int>(col.tileIndices.size())];
            } else {
                tileChoice = static_cast<int>(i) % (tilesPerSide * tilesPerSide);
            }
            int tileRow = tileChoice / tilesPerSide;
            int tileCol = tileChoice % tilesPerSide;
            int tileStartR = tileRow * tileSize;
            int tileStartC = tileCol * tileSize;
            int l4Row = j / config.layer4Size;
            int l4Col = j % config.layer4Size;
            int patchSize = std::max(1, tileSize / config.layer4Size); // 1 for 7->7
            int startR = tileStartR + l4Row * patchSize;
            int startC = tileStartC + l4Col * patchSize;
            for (int pr = 0; pr < patchSize; ++pr) {
                for (int pc = 0; pc < patchSize; ++pc) {
                    int idx = (startR + pr) * INPUT_SIZE + (startC + pc);
                    if (idx >= 0 && idx < static_cast<int>(inputNeurons.size())) {
                        auto inNeuron = inputNeurons[idx];
                        auto inAxon = datastore.getAxon(inNeuron->getAxonId());
                        if (!inAxon) continue;
                        auto syn = factory.createSynapse(inAxon->getId(), dendrite->getId(), 0.1, config.maxWeight);
                        inAxon->addSynapse(syn->getId());
                        synInputToL4Ids.push_back(syn->getId());
                        allSynapses.push_back(syn);
                        datastore.put(syn);
                        totalSynapses++;
                    }
                }
            }
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
            neuron->setTargetFiringRate(config.l5TargetRate);
            neuron->setPosition(xPos + pos_dis(gen),
                               pos_dis(gen),
                               30.0f + pos_dis(gen));
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
            neuron->setPosition(xPos + pos_dis(gen),
                               pos_dis(gen),
                               40.0f + pos_dis(gen));
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
    std::unordered_set<uint64_t> outputDendriteIds;

    for (int i = 0; i < NUM_LETTERS; ++i) {
        char letter = 'A' + i;
        auto cluster = factory.createCluster();
        cluster->setPosition((i - NUM_LETTERS / 2.0f) * 5.0f, 0.0f, 50.0f);
        outputLayer->addCluster(cluster->getId());

        for (int j = 0; j < config.neuronsPerOutputClass; ++j) {
            auto neuron = factory.createNeuron(config.neuronWindow, config.neuronThreshold, config.neuronMaxPatterns);
            if (disableOutputSignature) {
                neuron->disableTemporalSignature();
            }
            neuron->setTargetFiringRate(config.outputTargetRate);
            neuron->setPosition((i - NUM_LETTERS / 2.0f) * 5.0f + pos_dis(gen) * 0.5f,
                               pos_dis(gen) * 0.5f,
                               50.0f + pos_dis(gen) * 0.5f);
            auto axon = factory.createAxon(neuron->getId());
            auto dendrite = factory.createDendrite(neuron->getId());
            neuron->setAxonId(axon->getId());
            neuron->addDendrite(dendrite->getId());

            outputPopulations[i].push_back(neuron);
            cluster->addNeuron(neuron->getId());
            allNeuronIds.push_back(neuron->getId());
            if (!neuron->getDendriteIds().empty()) {
                outputDendriteIds.insert(neuron->getDendriteIds()[0]);
            }

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
    // Debug: record synapse counts for a few L4 axons to verify connectivity
    std::vector<std::pair<uint64_t, size_t>> debugL4AxonSynCounts;

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
            if (debugL4AxonSynCounts.size() < 5) {
                size_t synCount = axon->getSynapseCount();
                debugL4AxonSynCounts.emplace_back(axon->getId(), synCount);
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
                    synL4ToL5Ids.push_back(synapse->getId());
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

            int synAdded = 0;
            for (int i = 0; i < NUM_LETTERS; ++i) {
                for (auto& outputNeuron : outputPopulations[i]) {
                    if (dis(gen) < config.outputConnectProb) {
                        auto synapse = factory.createSynapse(
                            axon->getId(),
                            outputNeuron->getDendriteIds()[0],
                            config.outputInitialWeight,
                            config.maxWeight
                        );
                        axon->addSynapse(synapse->getId());
                        synL5ToOutputIds.push_back(synapse->getId());
                        allSynapses.push_back(synapse);
                        datastore.put(synapse);
                        outputSynapses++;
                        synAdded++;
                    }
                }
            }

            // Ensure at least one outgoing synapse per L5 axon to avoid empty axons,
            // but only when output connectivity is enabled.
            if (synAdded == 0 && config.outputConnectProb > 0.0 && !outputPopulations.empty()) {
                std::uniform_int_distribution<> cls_dis(0, NUM_LETTERS - 1);
                int cls = cls_dis(gen);
                if (!outputPopulations[cls].empty()) {
                    std::uniform_int_distribution<> neur_dis(0, static_cast<int>(outputPopulations[cls].size()) - 1);
                    int nidx = neur_dis(gen);
                    auto& outputNeuron = outputPopulations[cls][nidx];
                    auto synapse = factory.createSynapse(
                        axon->getId(),
                        outputNeuron->getDendriteIds()[0],
                        config.outputInitialWeight,
                        config.maxWeight
                    );
                    axon->addSynapse(synapse->getId());
                    synL5ToOutputIds.push_back(synapse->getId());
                    allSynapses.push_back(synapse);
                    datastore.put(synapse);
                    outputSynapses++;
                    synAdded++;
                }
            }

            datastore.put(axon);
        }
    }
    std::cout << "  ✓ Created " << outputSynapses << " Layer 5 → Output synapses" << std::endl;
    std::cout << "  Total synapses: " << (totalSynapses + outputSynapses) << std::endl;
    if (!debugL4AxonSynCounts.empty()) {
        std::cout << "  Debug L4 axon synapse counts (first few): ";
        for (size_t i = 0; i < debugL4AxonSynCounts.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << debugL4AxonSynCounts[i].first << "=" << debugL4AxonSynCounts[i].second;
        }
        std::cout << std::endl;
    }

    // Secondary safeguard: ensure every axon has at least one synapse.
    for (auto neuronId : allNeuronIds) {
        auto neuron = datastore.getNeuron(neuronId);
        if (!neuron) continue;
        auto axon = datastore.getAxon(neuron->getAxonId());
        if (!axon) continue;
        if (axon->getSynapseCount() == 0) {
            // If this is a layer 5 neuron, try wiring to output
            bool isLayer5 = false;
            for (auto& col : corticalColumns) {
                auto it = std::find_if(col.layer5Neurons.begin(), col.layer5Neurons.end(),
                                       [neuronId](const std::shared_ptr<Neuron>& n){ return n->getId() == neuronId; });
                if (it != col.layer5Neurons.end()) { isLayer5 = true; break; }
            }
            if (isLayer5 && config.outputConnectProb > 0.0 && !outputPopulations.empty()) {
                std::uniform_int_distribution<> cls_dis(0, NUM_LETTERS - 1);
                int cls = cls_dis(gen);
                if (!outputPopulations[cls].empty()) {
                    std::uniform_int_distribution<> neur_dis(0, static_cast<int>(outputPopulations[cls].size()) - 1);
                    int nidx = neur_dis(gen);
                    auto& outputNeuron = outputPopulations[cls][nidx];
                    auto synapse = factory.createSynapse(
                        axon->getId(),
                        outputNeuron->getDendriteIds()[0],
                        config.outputInitialWeight,
                        config.maxWeight
                    );
                    axon->addSynapse(synapse->getId());
                    synL5ToOutputIds.push_back(synapse->getId());
                    allSynapses.push_back(synapse);
                    datastore.put(synapse);
                    outputSynapses++;
                }
            }
            // Fallback: connect to own dendrite if still empty
            if (axon->getSynapseCount() == 0 && !neuron->getDendriteIds().empty()) {
                auto synapse = factory.createSynapse(
                    axon->getId(),
                    neuron->getDendriteIds()[0],
                    config.initialWeight,
                    config.maxWeight
                );
                axon->addSynapse(synapse->getId());
                allSynapses.push_back(synapse);
                datastore.put(synapse);
                outputSynapses++;
            }
            datastore.put(axon);
        }
    }

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
    networkPropagator->setTraceStdpEnabled(config.traceStdp);
    networkPropagator->setStdpLtdScale(config.stdpLtdScale);
    networkPropagator->setStdpLtdWindowMs(config.stdpLtdWindowMs);
    spikeProcessor->start();

    // Attach activity monitor / recorder so propagation gets captured
    if (simConfig.enableRecording) {
        spikeProcessor->setActivityMonitor(&activityMonitor);
        networkPropagator->setActivityMonitor(&activityMonitor);
        networkPropagator->setRecordingManager(recordingManager);
    }

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

    for (uint64_t sid : synInputToL4Ids) {
        networkPropagator->registerSynapseGroup(sid, NetworkPropagator::SynapseGroup::InputToL4);
    }
    for (uint64_t sid : synL4ToL5Ids) {
        networkPropagator->registerSynapseGroup(sid, NetworkPropagator::SynapseGroup::L4ToL5);
    }
    for (uint64_t sid : synL5ToOutputIds) {
        networkPropagator->registerSynapseGroup(sid, NetworkPropagator::SynapseGroup::L5ToOutput);
    }

    {
        std::unordered_set<uint64_t> l4NeuronIds;
        l4NeuronIds.reserve(static_cast<size_t>(config.numColumns * config.layer4Size * config.layer4Size));
        for (const auto& col : corticalColumns) {
            for (const auto& neuron : col.layer4Neurons) {
                l4NeuronIds.insert(neuron->getId());
            }
        }
        spikeProcessor->setInputToL4NeuronIds(l4NeuronIds);
    }

    auto computeWeightStats = [&](const std::vector<uint64_t>& synapseIds) -> WeightStats {
        WeightStats stats;
        if (synapseIds.empty()) {
            return stats;
        }
        double sum = 0.0;
        double minW = std::numeric_limits<double>::max();
        double maxW = std::numeric_limits<double>::lowest();
        for (uint64_t sid : synapseIds) {
            auto syn = datastore.getSynapse(sid);
            if (!syn) continue;
            double w = syn->getWeight();
            sum += w;
            minW = std::min(minW, w);
            maxW = std::max(maxW, w);
            stats.count++;
        }
        if (stats.count > 0) {
            stats.mean = sum / static_cast<double>(stats.count);
            stats.min = minW;
            stats.max = maxW;
        }
        return stats;
    };

    WeightStats baseInputToL4 = computeWeightStats(synInputToL4Ids);
    WeightStats baseL4ToL5 = computeWeightStats(synL4ToL5Ids);
    WeightStats baseL5ToOutput = computeWeightStats(synL5ToOutputIds);
    std::unordered_map<uint64_t, double> baseL5ToOutputWeights;
    baseL5ToOutputWeights.reserve(synL5ToOutputIds.size());
    for (uint64_t sid : synL5ToOutputIds) {
        auto syn = datastore.getSynapse(sid);
        if (!syn) continue;
        baseL5ToOutputWeights.emplace(sid, syn->getWeight());
    }
    std::cout << "Initial weight stats: input->L4 mean=" << baseInputToL4.mean
              << " min=" << baseInputToL4.min << " max=" << baseInputToL4.max
              << " count=" << baseInputToL4.count << std::endl;
    std::cout << "Initial weight stats: L4->L5 mean=" << baseL4ToL5.mean
              << " min=" << baseL4ToL5.min << " max=" << baseL4ToL5.max
              << " count=" << baseL4ToL5.count << std::endl;
    std::cout << "Initial weight stats: L5->Output mean=" << baseL5ToOutput.mean
              << " min=" << baseL5ToOutput.min << " max=" << baseL5ToOutput.max
              << " count=" << baseL5ToOutput.count << std::endl;

    // Debug: verify L5 axon connectivity to output dendrites for a sample neuron.
    if (!corticalColumns.empty() && !corticalColumns.front().layer5Neurons.empty()) {
        auto sampleL5 = corticalColumns.front().layer5Neurons.front();
        auto axon = datastore.getAxon(sampleL5->getAxonId());
        size_t totalSyn = 0;
        size_t outputSyn = 0;
        size_t registeredSyn = 0;
        if (axon) {
            const auto& synIds = axon->getSynapseIds();
            totalSyn = synIds.size();
            for (uint64_t sid : synIds) {
                auto syn = datastore.getSynapse(sid);
                if (!syn) continue;
                if (outputDendriteIds.find(syn->getDendriteId()) != outputDendriteIds.end()) {
                    outputSyn++;
                }
                if (networkPropagator->getSynapse(sid)) {
                    registeredSyn++;
                }
            }
        }
        std::cout << "Sample L5 axon " << sampleL5->getAxonId()
                  << ": synapses=" << totalSyn
                  << " output-targets=" << outputSyn
                  << " registered=" << registeredSyn << std::endl;
    }

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
    // Multi-Pass Training with Active Classification Testing
    // ========================================================================
    std::cout << "\n=== Starting Multi-Pass Training ===" << std::endl;

    // Training configuration
    const double ACCURACY_EPSILON = 0.001;  // Accuracy change threshold to consider "stable"
    const int STABLE_PASSES_REQUIRED = 3;  // Number of stable passes before stopping
    const double CLASSIFICATION_TIMEOUT_MS = 1000.0;  // 1 second timeout per letter
    const double CHECK_INTERVAL_MS = 10.0;  // Check classification every 10ms

    double previousAccuracy = -1.0;
    int stablePasses = 0;
    int currentPass = 0;
    std::vector<double> passAccuracies;

    std::vector<int> trainCount(NUM_LETTERS, 0);
    int totalPatternsLearned = 0;
    int imagesProcessed = 0;
    std::vector<std::set<size_t>> diagL4Patterns(NUM_LETTERS);
    bool diagL4Reported = false;
    std::vector<std::set<size_t>> diagL5Patterns(NUM_LETTERS);
    bool diagL5Reported = false;
    size_t diagReplayIdx = std::numeric_limits<size_t>::max();
    std::set<size_t> diagTrainL5Pattern;
    L5CountVector diagTrainL5Counts;
    bool diagReplayCaptured = false;

    auto trainingStart = std::chrono::high_resolution_clock::now();

    std::unordered_map<uint64_t, size_t> l4DendriteToCol;
    {
        size_t colIdx = 0;
        for (const auto& col : corticalColumns) {
            for (const auto& neuron : col.layer4Neurons) {
                const auto& dendIds = neuron->getDendriteIds();
                if (!dendIds.empty()) {
                    l4DendriteToCol.emplace(dendIds[0], colIdx);
                }
            }
            colIdx++;
        }
    }

    struct InferenceResult {
        std::set<size_t> l5Pattern;
        L5CountVector l5Counts;
        std::array<int, NUM_LETTERS> outSpikeCounts{};
        std::array<int, NUM_LETTERS> rawOutSpikeCounts{};
        size_t totalL5Spikes = 0;
    };

    double lastInferenceEndTime = spikeProcessor->getCurrentTime();
    auto waitForSimTime = [&](double targetTimeMs, double timeoutMs) {
        auto start = std::chrono::steady_clock::now();
        while (spikeProcessor->getCurrentTime() < targetTimeMs) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed > timeoutMs) {
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    };
    auto applyOutputCompetition = [&](std::array<int, NUM_LETTERS>& counts) {
        if (!config.enableOutputCompetition) return;
        if (config.outputCompetitionKeep <= 0) return;
        std::vector<std::pair<int, int>> ranked;
        ranked.reserve(NUM_LETTERS);
        for (int i = 0; i < NUM_LETTERS; ++i) {
            ranked.emplace_back(counts[i], i);
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        if (ranked.empty() || ranked.front().first < config.outputCompetitionMinSpikes) {
            return;
        }
        int keep = std::min(config.outputCompetitionKeep, static_cast<int>(ranked.size()));
        for (int i = keep; i < static_cast<int>(ranked.size()); ++i) {
            counts[ranked[i].second] = 0;
        }
    };
    auto runInference = [&](const decltype(trainLoader.getImage(0))& emnistImg) -> InferenceResult {
        InferenceResult result;
        result.l5Counts.assign(TOTAL_L5_NEURONS, 0);

        double nowTime = spikeProcessor->getCurrentTime();
        const double scheduleHorizon = spikeProcessor->getMaxScheduleAheadMs() - 2.0;
        while (lastInferenceEndTime + config.interImageGapMs > nowTime + scheduleHorizon) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            nowTime = spikeProcessor->getCurrentTime();
        }
        double baseTime = std::max(nowTime, lastInferenceEndTime) + config.interImageGapMs;
        lastInferenceEndTime = baseTime + config.neuronWindow;
        for (auto& col : corticalColumns) {
            for (auto& n : col.layer1Neurons) n->periodicMemoryCleanup(baseTime);
            for (auto& n : col.layer23Neurons) n->periodicMemoryCleanup(baseTime);
            for (auto& n : col.layer4Neurons) n->periodicMemoryCleanup(baseTime);
            for (auto& n : col.layer5Neurons) n->periodicMemoryCleanup(baseTime);
            for (auto& n : col.layer6Neurons) n->periodicMemoryCleanup(baseTime);
        }
        for (auto& pop : outputPopulations) {
            for (auto& n : pop) n->periodicMemoryCleanup(baseTime);
        }
        for (auto& n : inputNeurons) n->periodicMemoryCleanup(baseTime);
        // Ensure output spikes from prior images do not leak into this image's window.
        for (auto& pop : outputPopulations) {
            for (auto& n : pop) n->removeSpikesBefore(baseTime);
        }
        std::vector<std::shared_ptr<Neuron>> layer5Neurons;

        std::vector<bool> inputFired(inputNeurons.size(), false);
        for (size_t idx = 0; idx < inputNeurons.size() && idx < emnistImg.pixels.size(); ++idx) {
            double norm = emnistImg.pixels[idx] / 255.0;
            if (norm > config.pixelThreshold) {
                double fireT = baseTime + (1.0 - norm) * config.inputLatencyMs;
                inputNeurons[idx]->fireSignature(fireT);
                networkPropagator->fireNeuron(inputNeurons[idx]->getId(), fireT);
                inputFired[idx] = true;
            }
        }
        waitForSimTime(baseTime + config.inputLatencyMs + 5.0, 200.0);

        int colIdxSeq = 0;
        std::vector<bool> colHasL4(corticalColumns.size(), false);
        for (auto& col : corticalColumns) {
            int maskedCount = 0;
            for (int idx : col.inputMaskActiveIdx) {
                if (idx >= 0 && idx < static_cast<int>(inputFired.size()) && inputFired[idx]) {
                    maskedCount++;
                    if (maskedCount >= config.maskMinActive) break;
                }
            }
            if (maskedCount < config.maskMinActive) {
                colIdxSeq++;
                continue;
            }

            const int L4_KEEP = std::max(1, config.l4Keep);
            std::vector<std::pair<size_t, size_t>> rankedL4;
            rankedL4.reserve(col.layer4Neurons.size());
            for (size_t i = 0; i < col.layer4Neurons.size(); ++i) {
                rankedL4.emplace_back(col.layer4Neurons[i]->getSpikes().size(), i);
            }
            std::sort(rankedL4.begin(), rankedL4.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
            int winners = 0;
            for (size_t i = 0; i < rankedL4.size() && winners < L4_KEEP; ++i) {
                if (rankedL4[i].first == 0) break;
                int localIdx = static_cast<int>(rankedL4[i].second);
                int l4Row = localIdx / config.layer4Size;
                int l4Col = localIdx % config.layer4Size;
                double spatialDelay = (l4Row * config.l4RowDelay) + (l4Col * config.l4ColDelay);
                double l4Jitter = (config.l4PostJitterMs > 0.0) ? l4PostJitterDist(gen) : 0.0;
                double l4Fire = baseTime + 10.0 + config.l4PostShiftMs + (colIdxSeq * 0.5)
                                + spatialDelay + l4Jitter;
                if (l4Fire < baseTime) l4Fire = baseTime;
                auto& l4Neuron = col.layer4Neurons[localIdx];
                l4Neuron->fireSignature(l4Fire);
                networkPropagator->fireNeuron(l4Neuron->getId(), l4Fire);
                l4Neuron->fireAndAcknowledge(l4Fire);
                colHasL4[colIdxSeq] = true;
                winners++;
            }
            colIdxSeq++;
        }

        for (auto& col : corticalColumns) {
            layer5Neurons.insert(layer5Neurons.end(), col.layer5Neurons.begin(), col.layer5Neurons.end());
        }

        waitForSimTime(baseTime + 30.0, 200.0);

        const int L5_KEEP = std::max(1, config.l5Keep);
        std::vector<bool> l5WinnerGlobal(layer5Neurons.size(), false);
        colIdxSeq = 0;
        size_t l5Offset = 0;
        for (auto& col : corticalColumns) {
            for (auto& l5Neuron : col.layer5Neurons) {
                l5Neuron->resetInhibition();
            }
            if (colIdxSeq < static_cast<int>(colHasL4.size()) && !colHasL4[colIdxSeq]) {
                l5Offset += col.layer5Neurons.size();
                colIdxSeq++;
                continue;
            }
            std::vector<std::pair<size_t, size_t>> ranked;
            ranked.reserve(col.layer5Neurons.size());
            for (size_t i = 0; i < col.layer5Neurons.size(); ++i) {
                ranked.emplace_back(col.layer5Neurons[i]->getSpikes().size(), i);
            }
            std::sort(ranked.begin(), ranked.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
            int winners = 0;
            std::vector<bool> l5WinnerLocal(col.layer5Neurons.size(), false);
            for (size_t i = 0; i < ranked.size() && winners < L5_KEEP; ++i) {
                if (ranked[i].first < static_cast<size_t>(config.l5MinSpikes)) break;
                l5WinnerGlobal[l5Offset + ranked[i].second] = true;
                l5WinnerLocal[ranked[i].second] = true;
                winners++;
            }
            if (config.enableL5Inhibition) {
                for (size_t i = 0; i < col.layer5Neurons.size(); ++i) {
                    if (!l5WinnerLocal[i]) {
                        col.layer5Neurons[i]->applyInhibition(config.l5InhibitLoser);
                    }
                }
            }
            l5Offset += col.layer5Neurons.size();
            colIdxSeq++;
        }

        if (enableFullPropagation) {
            int colIdxSeq = 0;
            size_t l5Offset = 0;
            for (auto& col : corticalColumns) {
                if (colIdxSeq < static_cast<int>(colHasL4.size()) && !colHasL4[colIdxSeq]) {
                    l5Offset += col.layer5Neurons.size();
                    colIdxSeq++;
                    continue;
                }
                int localIdx = 0;
                for (auto& l5Neuron : col.layer5Neurons) {
                    if (l5Offset + localIdx < l5WinnerGlobal.size() &&
                        l5WinnerGlobal[l5Offset + localIdx] &&
                        l5Neuron->getInhibition() <= config.l5InhibitThreshold) {
                        double l5FireTime = baseTime + 15.0 + (colIdxSeq * 0.1) + (localIdx * 0.02);
                        networkPropagator->fireNeuron(l5Neuron->getId(), l5FireTime);
                        l5Neuron->fireAndAcknowledge(l5FireTime);
                    }
                    localIdx++;
                }
                l5Offset += col.layer5Neurons.size();
                colIdxSeq++;
            }
            int letterIdx = 0;
            for (auto& pop : outputPopulations) {
                int localIdx = 0;
                for (auto& outNeuron : pop) {
                    if (!outNeuron->getSpikes().empty()) {
                        double outFireTime = baseTime + 25.0 + (letterIdx * 0.1) + (localIdx * 0.01);
                        networkPropagator->fireNeuron(outNeuron->getId(), outFireTime);
                        outNeuron->fireAndAcknowledge(outFireTime);
                    }
                    localIdx++;
                }
                letterIdx++;
            }
        }

        for (size_t i = 0; i < layer5Neurons.size(); ++i) {
            size_t spikes = layer5Neurons[i]->getSpikes().size();
            if (spikes > 0 && i < l5WinnerGlobal.size() && l5WinnerGlobal[i] &&
                layer5Neurons[i]->getInhibition() <= config.l5InhibitThreshold) {
                result.l5Pattern.insert(i);
                result.totalL5Spikes += spikes;
                if (i < result.l5Counts.size()) {
                    result.l5Counts[i] = static_cast<uint16_t>(std::min<size_t>(spikes, 65535));
                }
            }
        }

        for (int i = 0; i < NUM_LETTERS; ++i) {
            for (auto& n : outputPopulations[i]) {
                result.outSpikeCounts[i] += static_cast<int>(n->getSpikes().size());
            }
        }
        result.rawOutSpikeCounts = result.outSpikeCounts;
        applyOutputCompetition(result.outSpikeCounts);

        return result;
    };

    // Multi-pass training loop
    double lastTrainingEndTime = spikeProcessor->getCurrentTime();
    while (currentPass < maxPasses) {
        currentPass++;
        std::cout << "\n--- Training Pass " << currentPass << " ---" << std::endl;
        std::cout.flush();

        // Reset per-pass counters and pattern memory (keep per-pass bounded)
        std::fill(trainCount.begin(), trainCount.end(), 0);
        int passPatterns = 0;
        int passImages = 0;
        const size_t diagPerPassImages = 10;
        size_t outputPreTeachSpikesSum = 0;
        size_t outputPostTeachSpikesSum = 0;
        size_t outputPreTeachCorrectSum = 0;
        std::array<uint64_t, NUM_LETTERS> outputPostTeachByClass{};
        std::array<double, NUM_LETTERS> outputWeightSumByClass{};
        std::array<size_t, NUM_LETTERS> outputWeightCountByClass{};
        std::array<size_t, NUM_LETTERS> l4ZeroWinnerCounts{};
        std::array<size_t, NUM_LETTERS> l4WinnerSum{};
        std::array<size_t, NUM_LETTERS> l4WinnerMin{};
        std::array<size_t, NUM_LETTERS> l4WinnerMax{};
        std::array<size_t, NUM_LETTERS> l4SampleCounts{};
        l4WinnerMin.fill(std::numeric_limits<size_t>::max());
        l4WinnerMax.fill(0);
        if (!keepL5History) {
            for (int letter = 0; letter < NUM_LETTERS; ++letter) {
                letterL5Patterns[letter].clear();
                letterL5Counts[letter].fill(0);
                letterPatternCounts[letter] = 0;
            }
        }
        networkPropagator->resetStdpUpdateStats();

        // Training: Process all training images
        std::cout << "  Processing " << trainLoader.size() << " training images..." << std::endl;
        std::cout.flush();

        for (size_t imgIdx = 0; imgIdx < trainLoader.size(); ++imgIdx) {
            const auto& emnistImg = trainLoader.getImage(imgIdx);
            int label = emnistImg.label - 1;  // Convert 1-26 to 0-25

            if (label < 0 || label >= NUM_LETTERS || !config.includeLetters[label]) {
                continue;
            }

            // Check if we need more training for this letter
            if (trainCount[label] >= config.trainingExamplesPerLetter) {
                continue;
            }

            // Debug: Print first image being processed
            if (passImages == 0) {
                std::cout << "  First image: idx=" << imgIdx << ", label=" << label << std::endl;
                std::cout.flush();
            }

            // Encode input image as spikes (do not clear spikes; let them expire naturally)
            double nowTime = spikeProcessor->getCurrentTime();
            const double scheduleHorizon = spikeProcessor->getMaxScheduleAheadMs() - 2.0;
            while (lastTrainingEndTime + config.interImageGapMs > nowTime + scheduleHorizon) {
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                nowTime = spikeProcessor->getCurrentTime();
            }
            double baseTime = std::max(nowTime, lastTrainingEndTime) + config.interImageGapMs;
            lastTrainingEndTime = baseTime + config.neuronWindow;
            for (auto& col : corticalColumns) {
                for (auto& n : col.layer1Neurons) n->periodicMemoryCleanup(baseTime);
                for (auto& n : col.layer23Neurons) n->periodicMemoryCleanup(baseTime);
                for (auto& n : col.layer4Neurons) {
                    n->periodicMemoryCleanup(baseTime);
                    n->resetIncomingSpikeCount();
                    if (!n->getDendriteIds().empty()) {
                        auto d = datastore.getDendrite(n->getDendriteIds().front());
                        if (d) {
                            d->resetReceivedSpikeCount();
                        }
                    }
                }
                for (auto& n : col.layer5Neurons) n->periodicMemoryCleanup(baseTime);
                for (auto& n : col.layer6Neurons) n->periodicMemoryCleanup(baseTime);
            }
            for (auto& pop : outputPopulations) {
                for (auto& n : pop) n->periodicMemoryCleanup(baseTime);
            }
            for (auto& n : inputNeurons) n->periodicMemoryCleanup(baseTime);
            // Ensure output spikes from prior images do not leak into this image's window.
            for (auto& pop : outputPopulations) {
                for (auto& n : pop) n->removeSpikesBefore(baseTime);
            }
            std::vector<std::shared_ptr<Neuron>> layer5Neurons;

            // Debug: Print progress for first image
            if (passImages == 0) {
                std::cout << "  Processing columns..." << std::endl;
                std::cout.flush();
            }

            bool debugDeliveries = (imagesProcessed < 10);
            if (debugDeliveries) {
                networkPropagator->resetDeliveryStats();
                networkPropagator->resetScheduleStats();
                spikeProcessor->resetDeliveryStats();
            }

            // Fire input neurons based on pixel intensity (simple rate code)
            std::vector<bool> inputFired(inputNeurons.size(), false);
            size_t inputFiredCount = 0;
            std::vector<size_t> inputSynToCol;
            if (passImages <= static_cast<int>(diagPerPassImages)) {
                inputSynToCol.assign(corticalColumns.size(), 0);
            }
            for (size_t idx = 0; idx < inputNeurons.size() && idx < emnistImg.pixels.size(); ++idx) {
                double norm = emnistImg.pixels[idx] / 255.0;
                if (norm > config.pixelThreshold) {
                    double fireT = baseTime + (1.0 - norm) * config.inputLatencyMs;
                    inputNeurons[idx]->fireSignature(fireT);
                    networkPropagator->fireNeuron(inputNeurons[idx]->getId(), fireT);
                    inputFired[idx] = true;
                    inputFiredCount++;
                    if (!inputSynToCol.empty()) {
                        auto axon = datastore.getAxon(inputNeurons[idx]->getAxonId());
                        if (axon) {
                            for (auto sid : axon->getSynapseIds()) {
                                auto syn = datastore.getSynapse(sid);
                                if (!syn) {
                                    continue;
                                }
                                auto it = l4DendriteToCol.find(syn->getDendriteId());
                                if (it != l4DendriteToCol.end()) {
                                    inputSynToCol[it->second]++;
                                }
                            }
                        }
                    }
                }
            }
            waitForSimTime(baseTime + config.inputLatencyMs + 5.0, 200.0);

            // After input delivery, fire L4 neurons that received spikes
            int colIdxSeq = 0;
            int totalScheduledFromL4 = 0;
            std::vector<bool> colHasL4(corticalColumns.size(), false);
            size_t colsMaskedPass = 0;
            size_t colsMaskedFail = 0;
            std::vector<int> maskedCounts;
            const int tilesPerSide = 4;
            const int tileSize = INPUT_SIZE / tilesPerSide;
            if (passImages <= static_cast<int>(diagPerPassImages)) {
                maskedCounts.reserve(corticalColumns.size());
            }
            size_t zeroL4Logged = 0;
            const size_t totalL4Neurons = static_cast<size_t>(config.numColumns) *
                                          static_cast<size_t>(config.layer4Size * config.layer4Size);
            std::vector<bool> l4WinnerGlobal(totalL4Neurons, false);
            size_t l4Offset = 0;
            for (auto& col : corticalColumns) {
                int tileFiredCount = 0;
                if (passImages <= static_cast<int>(diagPerPassImages)) {
                    if (!col.tileIndices.empty()) {
                        for (int tileIndex : col.tileIndices) {
                            int tileRow = tileIndex / tilesPerSide;
                            int tileCol = tileIndex % tilesPerSide;
                            int tileStartR = tileRow * tileSize;
                            int tileStartC = tileCol * tileSize;
                            for (int r = 0; r < tileSize; ++r) {
                                int rr = tileStartR + r;
                                for (int c = 0; c < tileSize; ++c) {
                                    int cc = tileStartC + c;
                                    int idx = rr * INPUT_SIZE + cc;
                                    if (idx >= 0 && idx < static_cast<int>(inputFired.size()) && inputFired[idx]) {
                                        tileFiredCount++;
                                    }
                                }
                            }
                        }
                    } else {
                        int tileRow = colIdxSeq / tilesPerSide;
                        int tileCol = colIdxSeq % tilesPerSide;
                        int tileStartR = tileRow * tileSize;
                        int tileStartC = tileCol * tileSize;
                        for (int r = 0; r < tileSize; ++r) {
                            int rr = tileStartR + r;
                            for (int c = 0; c < tileSize; ++c) {
                                int cc = tileStartC + c;
                                int idx = rr * INPUT_SIZE + cc;
                                if (idx >= 0 && idx < static_cast<int>(inputFired.size()) && inputFired[idx]) {
                                    tileFiredCount++;
                                }
                            }
                        }
                    }
                }
                // If this column has insufficient masked input activity, skip it entirely
                int maskedCount = 0;
                for (int idx : col.inputMaskActiveIdx) {
                    if (idx >= 0 && idx < static_cast<int>(inputFired.size()) && inputFired[idx]) {
                        maskedCount++;
                        if (maskedCount >= config.maskMinActive) break;
                    }
                }
                if (passImages <= static_cast<int>(diagPerPassImages)) {
                    maskedCounts.push_back(maskedCount);
                    size_t l4SpikeNonzero = 0;
                    size_t l4SpikeTotal = 0;
                    size_t l4SpikeMax = 0;
                    size_t l4DendriteInTotal = 0;
                    for (auto& l4Neuron : col.layer4Neurons) {
                        size_t spikes = l4Neuron->getSpikes().size();
                        if (spikes > 0) {
                            l4SpikeNonzero++;
                            l4SpikeTotal += spikes;
                            l4SpikeMax = std::max(l4SpikeMax, spikes);
                        }
                        if (!l4Neuron->getDendriteIds().empty()) {
                            auto dend = datastore.getDendrite(l4Neuron->getDendriteIds().front());
                            if (dend) {
                                l4DendriteInTotal += dend->getReceivedSpikeCount();
                            }
                        }
                    }
                    std::cout << "      Col " << colIdxSeq
                              << " masked=" << maskedCount
                              << " tile_fired=" << tileFiredCount
                              << " in_syn=" << (colIdxSeq < static_cast<int>(inputSynToCol.size())
                                  ? inputSynToCol[colIdxSeq]
                                  : 0)
                              << " dend_in=" << l4DendriteInTotal
                              << " l4_nonzero=" << l4SpikeNonzero
                              << " l4_spikes=" << l4SpikeTotal
                              << " l4_max=" << l4SpikeMax
                              << std::endl;
                    if (maskedCount >= config.maskMinActive && l4SpikeNonzero == 0 && zeroL4Logged < 2) {
                        auto& sample = col.layer4Neurons.front();
                        auto sampleDend = datastore.getDendrite(sample->getDendriteIds().front());
                        double bestSim = sample->getBestSimilarity();
                        std::cout << "        L4 sample stats: spikes=" << sample->getSpikes().size()
                                  << " bestSim=" << std::fixed << std::setprecision(3) << bestSim
                                  << " activation=" << sample->getActivation()
                                  << " inhibition=" << sample->getInhibition()
                                  << " firingRate=" << sample->getFiringRate()
                                  << " incoming=" << sample->getIncomingSpikeCount()
                                  << " dendrite_in=" << (sampleDend ? sampleDend->getReceivedSpikeCount() : 0)
                                  << std::endl;
                        zeroL4Logged++;
                    }
                }
                if (maskedCount < config.maskMinActive) {
                    colsMaskedFail++;
                    l4Offset += col.layer4Neurons.size();
                    colIdxSeq++;
                    continue;
                }
                colsMaskedPass++;

                const int L4_KEEP = std::max(1, config.l4Keep);
                std::vector<std::pair<size_t, size_t>> rankedL4;
                rankedL4.reserve(col.layer4Neurons.size());
                for (size_t i = 0; i < col.layer4Neurons.size(); ++i) {
                    rankedL4.emplace_back(col.layer4Neurons[i]->getSpikes().size(), i);
                }
                std::sort(rankedL4.begin(), rankedL4.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
                int winners = 0;
                for (size_t i = 0; i < rankedL4.size() && winners < L4_KEEP; ++i) {
                    if (rankedL4[i].first == 0) break;
                    int localIdx = static_cast<int>(rankedL4[i].second);
                    int l4Row = localIdx / config.layer4Size;
                    int l4Col = localIdx % config.layer4Size;
                    double spatialDelay = (l4Row * config.l4RowDelay) + (l4Col * config.l4ColDelay);
                    double l4Jitter = (config.l4PostJitterMs > 0.0) ? l4PostJitterDist(gen) : 0.0;
                    double l4Fire = baseTime + 10.0 + config.l4PostShiftMs + (colIdxSeq * 0.5)
                                    + spatialDelay + l4Jitter;
                    if (l4Fire < baseTime) l4Fire = baseTime;
                    auto& l4Neuron = col.layer4Neurons[localIdx];
                    l4Neuron->fireSignature(l4Fire);
                    int scheduled = networkPropagator->fireNeuron(l4Neuron->getId(), l4Fire);
                    totalScheduledFromL4 += std::max(0, scheduled);
                    l4Neuron->fireAndAcknowledge(l4Fire);
                    colHasL4[colIdxSeq] = true;
                    size_t globalIdx = l4Offset + static_cast<size_t>(localIdx);
                    if (globalIdx < l4WinnerGlobal.size()) {
                        l4WinnerGlobal[globalIdx] = true;
                    }
                    winners++;

                    // Deep debug: inspect the first few active L4 neurons on first image
                    if (imagesProcessed == 0 && trainCount[label] == 0 && localIdx < 3 && colIdxSeq < 2) {
                        auto ax = datastore.getAxon(l4Neuron->getAxonId());
                        size_t synCount = ax ? ax->getSynapseCount() : 0;
                        std::cout << "      L4 neuron " << l4Neuron->getId()
                                  << " synapses=" << synCount
                                  << " temporalSig=" << l4Neuron->getTemporalSignature().size()
                                  << " scheduled=" << scheduled << std::endl;
                    }
                }
                l4Offset += col.layer4Neurons.size();
                colIdxSeq++;
            }

            // Give time for L4 → L5 synaptic delivery before we inspect L5
            waitForSimTime(baseTime + 30.0, 200.0);

            // L5 competition: keep only a small set of most-active neurons per column
            const int L5_KEEP = std::max(1, config.l5Keep);
            std::vector<bool> l5WinnerGlobal(TOTAL_L5_NEURONS, false);
            colIdxSeq = 0;
            size_t l5Offset = 0;
            for (auto& col : corticalColumns) {
                for (auto& l5Neuron : col.layer5Neurons) {
                    l5Neuron->resetInhibition();
                }
                if (colIdxSeq < static_cast<int>(colHasL4.size()) && !colHasL4[colIdxSeq]) {
                    l5Offset += col.layer5Neurons.size();
                    colIdxSeq++;
                    continue;
                }
                std::vector<std::pair<size_t, size_t>> ranked;
                ranked.reserve(col.layer5Neurons.size());
                for (size_t i = 0; i < col.layer5Neurons.size(); ++i) {
                    ranked.emplace_back(col.layer5Neurons[i]->getSpikes().size(), i);
                }
                std::sort(ranked.begin(), ranked.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
                int winners = 0;
                std::vector<bool> l5WinnerLocal(col.layer5Neurons.size(), false);
                for (size_t i = 0; i < ranked.size() && winners < L5_KEEP; ++i) {
                    if (ranked[i].first < static_cast<size_t>(config.l5MinSpikes)) break;
                    size_t globalIdx = l5Offset + ranked[i].second;
                    if (globalIdx < l5WinnerGlobal.size()) {
                        l5WinnerGlobal[globalIdx] = true;
                        l5WinnerLocal[ranked[i].second] = true;
                    }
                    winners++;
                }
                if (config.enableL5Inhibition) {
                    for (size_t i = 0; i < col.layer5Neurons.size(); ++i) {
                        if (!l5WinnerLocal[i]) {
                            col.layer5Neurons[i]->applyInhibition(config.l5InhibitLoser);
                        }
                    }
                }
                l5Offset += col.layer5Neurons.size();
                colIdxSeq++;
            }

            // In full propagation mode, drive L5 neurons that received spikes so downstream STDP can occur
            if (enableFullPropagation) {
                int colIdxSeq = 0;
                size_t l5Offset = 0;
                for (auto& col : corticalColumns) {
                    // Gate: skip L5 of columns that had no L4 activity
                    if (colIdxSeq < static_cast<int>(colHasL4.size()) && !colHasL4[colIdxSeq]) {
                        l5Offset += col.layer5Neurons.size();
                        colIdxSeq++;
                        continue;
                    }
                    int localIdx = 0;
                    for (auto& l5Neuron : col.layer5Neurons) {
                        size_t globalIdx = l5Offset + localIdx;
                        if (globalIdx < l5WinnerGlobal.size() && l5WinnerGlobal[globalIdx] &&
                            l5Neuron->getInhibition() <= config.l5InhibitThreshold) {
                            double l5FireTime = baseTime + 15.0 + (colIdxSeq * 0.1) + (localIdx * 0.02);
                            // Propagate to downstream synapses and apply learning
                            networkPropagator->fireNeuron(l5Neuron->getId(), l5FireTime);
                            l5Neuron->fireAndAcknowledge(l5FireTime);
                            l5Neuron->learnCurrentPattern();
                        }
                        localIdx++;
                    }
                    l5Offset += col.layer5Neurons.size();
                    colIdxSeq++;
                }

                size_t preTeachSpikes = 0;
                size_t preTeachActive = 0;
                size_t preTeachCorrect = 0;
                size_t preTeachOlder = 0;
                size_t preTeachCurrent = 0;
                double preTeachMinTime = std::numeric_limits<double>::infinity();
                double preTeachMaxTime = -std::numeric_limits<double>::infinity();
                for (int i = 0; i < NUM_LETTERS; ++i) {
                    for (auto& outputNeuron : outputPopulations[i]) {
                        const auto& spikesVec = outputNeuron->getSpikes();
                        size_t spikes = spikesVec.size();
                        if (spikes > 0) {
                            preTeachActive++;
                            preTeachSpikes += spikes;
                            if (i == label) {
                                preTeachCorrect += spikes;
                            }
                            if (imagesProcessed < 3) {
                                for (double t : spikesVec) {
                                    preTeachMinTime = std::min(preTeachMinTime, t);
                                    preTeachMaxTime = std::max(preTeachMaxTime, t);
                                    if (t < baseTime) {
                                        preTeachOlder++;
                                    } else {
                                        preTeachCurrent++;
                                    }
                                }
                            }
                        }
                    }
                }

                if (!disableOutputTeach) {
                    // Supervised teaching signal: force the correct output population
                    // to spike and learn the current pattern so STDP can bind L5 activity
                    // to the label-specific outputs.
                    double teachTime = baseTime + 30.0;
                    for (auto& outputNeuron : outputPopulations[label]) {
                        outputNeuron->fireSignature(teachTime);
                        networkPropagator->fireNeuron(outputNeuron->getId(), teachTime);
                        outputNeuron->fireAndAcknowledge(teachTime);
                        outputNeuron->learnCurrentPattern();
                        passPatterns++;
                    }
                }

                size_t postTeachSpikes = 0;
                for (int i = 0; i < NUM_LETTERS; ++i) {
                    for (auto& outputNeuron : outputPopulations[i]) {
                        size_t spikes = outputNeuron->getSpikes().size();
                        if (spikes > 0) {
                            postTeachSpikes += spikes;
                            outputPostTeachByClass[i] += spikes;
                        }
                    }
                }
                outputPreTeachSpikesSum += preTeachSpikes;
                outputPreTeachCorrectSum += preTeachCorrect;
                outputPostTeachSpikesSum += postTeachSpikes;

                if (imagesProcessed < 3) {
                    auto deliveryStats = networkPropagator->getDeliveryStats();
                    double deliveryMean = (deliveryStats.l5ToOutput > 0)
                        ? (deliveryStats.l5ToOutputDeltaSum / static_cast<double>(deliveryStats.l5ToOutput))
                        : 0.0;
                    auto flags = std::cout.flags();
                    auto precision = std::cout.precision();
                    std::cout << "    Debug output pre-teach: total=" << preTeachSpikes
                              << " active=" << preTeachActive
                              << " correct=" << preTeachCorrect
                              << " post-teach total=" << postTeachSpikes;
                    if (preTeachSpikes > 0) {
                        std::cout << " time-range=[" << preTeachMinTime << "," << preTeachMaxTime << "]"
                                  << " older=" << preTeachOlder
                                  << " current=" << preTeachCurrent;
                    }
                    std::cout << " | L5->Output deliveries=" << deliveryStats.l5ToOutput
                              << " deltaMean=" << std::fixed << std::setprecision(3) << deliveryMean
                              << " deltaRange=[" << deliveryStats.l5ToOutputDeltaMin
                              << "," << deliveryStats.l5ToOutputDeltaMax << "]";
                    std::cout << std::endl;
                    std::cout.flags(flags);
                    std::cout.precision(precision);
                }
            }

            // Collect all Layer 5 neurons for pattern copying
            for (auto& col : corticalColumns) {
                layer5Neurons.insert(layer5Neurons.end(), col.layer5Neurons.begin(), col.layer5Neurons.end());
            }

            // Debug: log activity for first few images to verify propagation
            if (imagesProcessed < 3) {
                size_t activeL5 = 0, activeOut = 0, activeL4 = 0, activeInput = 0;
                for (auto& n : layer5Neurons) if (!n->getSpikes().empty()) activeL5++;
                for (auto& pop : outputPopulations) {
                    for (auto& n : pop) if (!n->getSpikes().empty()) activeOut++;
                }
                for (auto& n : inputNeurons) if (!n->getSpikes().empty()) activeInput++;
                for (size_t i = 0; i < l4WinnerGlobal.size(); ++i) {
                    if (l4WinnerGlobal[i]) activeL4++;
                }
                std::cout << "    Debug img " << imagesProcessed << ": active input=" << activeInput
                          << " L4=" << activeL4 << " L5=" << activeL5
                          << " output=" << activeOut
                          << " scheduledFromL4=" << totalScheduledFromL4
                          << std::endl;
                auto schedStats = networkPropagator->getScheduleStats();
                std::cout << "      Scheduled spikes: input->L4=" << schedStats.inputToL4
                          << " other=" << schedStats.other << std::endl;
                auto spStats = spikeProcessor->getDeliveryStats();
                std::cout << "      SpikeProcessor delivery: scheduled=" << spStats.scheduled
                          << " input->L4=" << spStats.scheduledInputToL4
                          << " delivered=" << spStats.delivered
                          << " input->L4=" << spStats.deliveredInputToL4
                          << " missing=" << spStats.missingDendrite
                          << " missingInput=" << spStats.missingDendriteInputToL4
                          << std::endl;
            }

            // Skip output-layer firing/learning in full propagation mode; rely on L5 centroids for classification.

            // Copy L5 pattern to output neurons and train
            if (!enableFullPropagation) {
                copyLayerSpikePattern(layer5Neurons, outputPopulations[label], baseTime);
                for (size_t idx = 0; idx < outputPopulations[label].size(); ++idx) {
                    auto& outputNeuron = outputPopulations[label][idx];
                    double outFireTime = baseTime + 20.0 + idx * 0.01;
                    if (!outputNeuron->getSpikes().empty()) {
                        outputNeuron->fireAndAcknowledge(outFireTime);
                        outputNeuron->learnCurrentPattern();
                        passPatterns++;
                    }
                }
            }

            // Collect L5 pattern (spike counts per L5 neuron)
            std::set<size_t> firedL5Global;
            L5CountVector firedL5Counts(TOTAL_L5_NEURONS, 0);
            size_t totalSpikes = 0;
            for (size_t i = 0; i < layer5Neurons.size(); ++i) {
                size_t spikes = layer5Neurons[i]->getSpikes().size();
                if (spikes > 0 && i < l5WinnerGlobal.size() && l5WinnerGlobal[i] &&
                    layer5Neurons[i]->getInhibition() <= config.l5InhibitThreshold) {
                    firedL5Global.insert(i);
                    totalSpikes += spikes;
                    if (i < firedL5Counts.size()) {
                        firedL5Counts[i] = static_cast<uint16_t>(std::min<size_t>(spikes, 65535));
                    }
                }
            }

            size_t activeL4 = 0;
            for (size_t i = 0; i < l4WinnerGlobal.size(); ++i) {
                if (l4WinnerGlobal[i]) {
                    activeL4++;
                }
            }

            l4WinnerSum[label] += activeL4;
            l4WinnerMax[label] = std::max(l4WinnerMax[label], activeL4);
            l4WinnerMin[label] = std::min(l4WinnerMin[label], activeL4);
            if (activeL4 == 0) {
                l4ZeroWinnerCounts[label]++;
            }
            l4SampleCounts[label]++;

            // Store L5 pattern for this letter (for Jaccard similarity matching)
            bool storedL5Pattern = false;
            if (!firedL5Global.empty()) {
                if (letterL5Patterns[label].size() >= MAX_PATTERNS_PER_LETTER) {
                    letterL5Patterns[label].erase(letterL5Patterns[label].begin());
                }
                letterL5Patterns[label].push_back(firedL5Counts);
                storedL5Pattern = true;
            }

            if (passImages <= static_cast<int>(diagPerPassImages)) {
                size_t activeL4 = 0;
                for (size_t i = 0; i < l4WinnerGlobal.size(); ++i) {
                    if (l4WinnerGlobal[i]) {
                        activeL4++;
                    }
                }
                std::cout << "    Pass " << currentPass << " img " << passImages
                          << " label=" << char('A' + label)
                          << " L4_winners=" << activeL4
                          << " L5_winners=" << firedL5Global.size()
                          << " L5_spikes=" << totalSpikes
                          << " L5_store=" << (storedL5Pattern ? "yes" : "no")
                          << " L5_patterns=" << letterL5Patterns[label].size()
                          << " mask_pass=" << colsMaskedPass
                          << " mask_fail=" << colsMaskedFail
                          << " input_fired=" << inputFiredCount
                          << std::endl;
                std::cout << "      Masked counts: ";
                for (size_t i = 0; i < maskedCounts.size(); ++i) {
                    if (i) std::cout << ",";
                    std::cout << maskedCounts[i];
                }
                std::cout << std::endl;
            }

            if (!diagReplayCaptured && currentPass == 1 && passImages == 0) {
                diagReplayIdx = imgIdx;
                diagTrainL5Pattern = firedL5Global;
                diagTrainL5Counts = firedL5Counts;
                diagReplayCaptured = true;
            }

            // Update centroid counts for this letter
            for (size_t idx = 0; idx < firedL5Counts.size(); ++idx) {
                if (firedL5Counts[idx] > 0) {
                    letterL5Counts[label][idx] += firedL5Counts[idx];
                }
            }
            letterPatternCounts[label]++;

            // Record L4 activity pattern for first example of each letter in pass 1
            if (currentPass == 1 && trainCount[label] == 0) {
                std::set<size_t> firedL4;
                for (size_t i = 0; i < l4WinnerGlobal.size(); ++i) {
                    if (l4WinnerGlobal[i]) firedL4.insert(i);
                }
                diagL4Patterns[label] = std::move(firedL4);
                // Once we have all letters, report overlap
                bool allHave = true;
                for (int l = 0; l < NUM_LETTERS; ++l) {
                    if (!config.includeLetters[l]) {
                        continue;
                    }
                    if (diagL4Patterns[l].empty()) { allHave = false; break; }
                }
                if (allHave && !diagL4Reported) {
                    double sumJ = 0.0;
                    int pairs = 0;
                    for (int a = 0; a < NUM_LETTERS; ++a) {
                        if (!config.includeLetters[a]) continue;
                        for (int b = a + 1; b < NUM_LETTERS; ++b) {
                            if (!config.includeLetters[b]) continue;
                            std::set<size_t> inter;
                            std::set<size_t> uni;
                            std::set_intersection(diagL4Patterns[a].begin(), diagL4Patterns[a].end(),
                                                  diagL4Patterns[b].begin(), diagL4Patterns[b].end(),
                                                  std::inserter(inter, inter.begin()));
                            std::set_union(diagL4Patterns[a].begin(), diagL4Patterns[a].end(),
                                           diagL4Patterns[b].begin(), diagL4Patterns[b].end(),
                                           std::inserter(uni, uni.begin()));
                            double jac = uni.empty() ? 0.0 : static_cast<double>(inter.size()) / static_cast<double>(uni.size());
                            sumJ += jac;
                            pairs++;
                        }
                    }
                    double avgJ = pairs > 0 ? sumJ / pairs : 0.0;
                    std::cout << "  L4 diagnostic (pass 1): avg Jaccard overlap across letters = "
                              << std::fixed << std::setprecision(3) << avgJ << std::endl;
                    diagL4Reported = true;
                }
            }

            if (currentPass == 1 && trainCount[label] == 0) {
                diagL5Patterns[label] = firedL5Global;
                bool allHave = true;
                for (int l = 0; l < NUM_LETTERS; ++l) {
                    if (!config.includeLetters[l]) {
                        continue;
                    }
                    if (diagL5Patterns[l].empty()) { allHave = false; break; }
                }
                if (allHave && !diagL5Reported) {
                    double sumJ = 0.0;
                    int pairs = 0;
                    for (int a = 0; a < NUM_LETTERS; ++a) {
                        if (!config.includeLetters[a]) continue;
                        for (int b = a + 1; b < NUM_LETTERS; ++b) {
                            if (!config.includeLetters[b]) continue;
                            std::set<size_t> inter;
                            std::set<size_t> uni;
                            std::set_intersection(diagL5Patterns[a].begin(), diagL5Patterns[a].end(),
                                                  diagL5Patterns[b].begin(), diagL5Patterns[b].end(),
                                                  std::inserter(inter, inter.begin()));
                            std::set_union(diagL5Patterns[a].begin(), diagL5Patterns[a].end(),
                                           diagL5Patterns[b].begin(), diagL5Patterns[b].end(),
                                           std::inserter(uni, uni.begin()));
                            double jac = uni.empty() ? 0.0 : static_cast<double>(inter.size()) / static_cast<double>(uni.size());
                            sumJ += jac;
                            pairs++;
                        }
                    }
                    double avgJ = pairs > 0 ? sumJ / pairs : 0.0;
                    std::cout << "  L5 diagnostic (pass 1): avg Jaccard overlap across letters = "
                              << std::fixed << std::setprecision(3) << avgJ << std::endl;
                    diagL5Reported = true;
                }
            }

            // Debug: Show L5 stats for first image of each letter
            if (trainCount[label] == 0 && currentPass == 1) {
                std::cout << "    Letter " << char('A' + label) << ": "
                          << firedL5Global.size() << " L5 neurons, "
                          << totalSpikes << " spikes";
                // Show first 20 L5 neuron indices to see the pattern
                std::cout << " L5=[";
                int count = 0;
                for (size_t idx : firedL5Global) {
                    if (count++ >= 20) { std::cout << "..."; break; }
                    std::cout << idx << " ";
                }
                std::cout << "]" << std::endl;
            }

            trainCount[label]++;
            passImages++;
            imagesProcessed++;

            // Print progress every 10 images for small runs, every 500 for large
            int progressInterval = (config.trainingExamplesPerLetter <= 10) ? 10 : 500;
            if (passImages % progressInterval == 0) {
                std::cout << "  Training: " << passImages << " images processed" << std::endl;
                auto stdpStats = networkPropagator->getStdpUpdateStats();
                std::cout << "    STDP updates: total=" << stdpStats.total
                          << " (LTP=" << stdpStats.ltp
                          << ", LTD=" << stdpStats.ltd
                          << ", reward=" << stdpStats.reward << ")" << std::endl;
                auto groupStats = networkPropagator->getStdpGroupStats();
                std::cout << "    STDP LTP/LTD by group: "
                          << "input->L4=" << groupStats.inputToL4.ltp << "/" << groupStats.inputToL4.ltd
                          << " L4->L5=" << groupStats.l4ToL5.ltp << "/" << groupStats.l4ToL5.ltd
                          << " L5->Output=" << groupStats.l5ToOutput.ltp << "/" << groupStats.l5ToOutput.ltd
                          << " other=" << groupStats.other.ltp << "/" << groupStats.other.ltd
                          << std::endl;
                auto timingStats = networkPropagator->getStdpTimingStats();
                std::cout << "    STDP timing by group (pre<post / post<pre / zero): "
                          << "input->L4=" << timingStats.inputToL4.preBeforePost << "/"
                          << timingStats.inputToL4.postBeforePre << "/"
                          << timingStats.inputToL4.nearZero
                          << " L4->L5=" << timingStats.l4ToL5.preBeforePost << "/"
                          << timingStats.l4ToL5.postBeforePre << "/"
                          << timingStats.l4ToL5.nearZero
                          << " L5->Output=" << timingStats.l5ToOutput.preBeforePost << "/"
                          << timingStats.l5ToOutput.postBeforePre << "/"
                          << timingStats.l5ToOutput.nearZero
                          << " other=" << timingStats.other.preBeforePost << "/"
                          << timingStats.other.postBeforePre << "/"
                          << timingStats.other.nearZero
                          << std::endl;
                std::cout.flush();
                std::fflush(stdout); // ensure log is flushed for crash diagnostics
            }

            // Check if training is complete for this pass
        bool allLettersTrained = true;
        for (int i = 0; i < NUM_LETTERS; ++i) {
            if (!config.includeLetters[i]) {
                continue;
            }
            if (trainCount[i] < config.trainingExamplesPerLetter) {
                allLettersTrained = false;
                break;
            }
        }
            if (allLettersTrained) break;
        }

        totalPatternsLearned += passPatterns;
        std::cout << "  Pass " << currentPass << " complete: " << passPatterns << " patterns learned" << std::endl;
        auto passStdpStats = networkPropagator->getStdpUpdateStats();
        std::cout << "  Pass " << currentPass << " STDP updates: total=" << passStdpStats.total
                  << " (LTP=" << passStdpStats.ltp
                  << ", LTD=" << passStdpStats.ltd
                  << ", reward=" << passStdpStats.reward << ")" << std::endl;
        auto passGroupStats = networkPropagator->getStdpGroupStats();
        std::cout << "  Pass " << currentPass << " STDP LTP/LTD by group: "
                  << "input->L4=" << passGroupStats.inputToL4.ltp << "/" << passGroupStats.inputToL4.ltd
                  << " L4->L5=" << passGroupStats.l4ToL5.ltp << "/" << passGroupStats.l4ToL5.ltd
                  << " L5->Output=" << passGroupStats.l5ToOutput.ltp << "/" << passGroupStats.l5ToOutput.ltd
                  << " other=" << passGroupStats.other.ltp << "/" << passGroupStats.other.ltd
                  << std::endl;
        auto passTimingStats = networkPropagator->getStdpTimingStats();
        std::cout << "  Pass " << currentPass << " STDP timing by group (pre<post / post<pre / zero): "
                  << "input->L4=" << passTimingStats.inputToL4.preBeforePost << "/"
                  << passTimingStats.inputToL4.postBeforePre << "/"
                  << passTimingStats.inputToL4.nearZero
                  << " L4->L5=" << passTimingStats.l4ToL5.preBeforePost << "/"
                  << passTimingStats.l4ToL5.postBeforePre << "/"
                  << passTimingStats.l4ToL5.nearZero
                  << " L5->Output=" << passTimingStats.l5ToOutput.preBeforePost << "/"
                  << passTimingStats.l5ToOutput.postBeforePre << "/"
                  << passTimingStats.l5ToOutput.nearZero
                  << " other=" << passTimingStats.other.preBeforePost << "/"
                  << passTimingStats.other.postBeforePre << "/"
                  << passTimingStats.other.nearZero
                  << std::endl;
        std::cout << "  L4 winner stats by class (min/mean/max, zero-count):" << std::endl;
        for (int i = 0; i < NUM_LETTERS; ++i) {
            if (!config.includeLetters[i]) {
                continue;
            }
            double mean = l4SampleCounts[i] > 0
                ? static_cast<double>(l4WinnerSum[i]) / static_cast<double>(l4SampleCounts[i])
                : 0.0;
            size_t minVal = (l4SampleCounts[i] > 0) ? l4WinnerMin[i] : 0;
            size_t maxVal = (l4SampleCounts[i] > 0) ? l4WinnerMax[i] : 0;
            std::cout << "    " << char('A' + i)
                      << ": " << minVal
                      << "/" << std::fixed << std::setprecision(1) << mean
                      << "/" << maxVal
                      << " zero=" << l4ZeroWinnerCounts[i]
                      << " samples=" << l4SampleCounts[i]
                      << std::endl;
        }
        bool missingClassPatterns = false;
        for (int i = 0; i < NUM_LETTERS; ++i) {
            if (!config.includeLetters[i]) {
                continue;
            }
            if (letterL5Patterns[i].empty()) {
                std::cerr << "  ERROR: No L5 patterns stored for class "
                          << char('A' + i) << " after pass " << currentPass << std::endl;
                missingClassPatterns = true;
            }
        }
        if (missingClassPatterns) {
            std::cerr << "  ERROR: Aborting due to missing L5 patterns in one or more classes." << std::endl;
            return 1;
        }
        std::cout << "  Output spikes pre/post teach (sum): pre=" << outputPreTeachSpikesSum
                  << " correct=" << outputPreTeachCorrectSum
                  << " post=" << outputPostTeachSpikesSum << std::endl;
        std::cout << "  Output spikes post-teach by class: ";
        for (int i = 0; i < NUM_LETTERS; ++i) {
            std::cout << char('A' + i) << "=" << outputPostTeachByClass[i] << " ";
        }
        std::cout << std::endl;
        std::vector<std::shared_ptr<Neuron>> allL4;
        allL4.reserve(static_cast<size_t>(config.numColumns * config.layer4Size * config.layer4Size));
        std::vector<std::shared_ptr<Neuron>> allL5;
        allL5.reserve(static_cast<size_t>(config.numColumns * config.layer5Neurons));
        for (auto& col : corticalColumns) {
            allL4.insert(allL4.end(), col.layer4Neurons.begin(), col.layer4Neurons.end());
            allL5.insert(allL5.end(), col.layer5Neurons.begin(), col.layer5Neurons.end());
        }
        std::vector<std::shared_ptr<Neuron>> allOutput;
        for (auto& pop : outputPopulations) {
            allOutput.insert(allOutput.end(), pop.begin(), pop.end());
        }
        applyHomeostasis(allL4, "L4");
        applyHomeostasis(allL5, "L5");
        applyHomeostasis(allOutput, "Output");
        auto curInputToL4 = computeWeightStats(synInputToL4Ids);
        auto curL4ToL5 = computeWeightStats(synL4ToL5Ids);
        auto curL5ToOutput = computeWeightStats(synL5ToOutputIds);
        for (uint64_t sid : synL5ToOutputIds) {
            auto syn = datastore.getSynapse(sid);
            if (!syn) continue;
            auto dendrite = datastore.getDendrite(syn->getDendriteId());
            if (!dendrite) continue;
            uint64_t targetNeuronId = dendrite->getTargetNeuronId();
            for (int i = 0; i < NUM_LETTERS; ++i) {
                for (auto& outputNeuron : outputPopulations[i]) {
                    if (outputNeuron->getId() == targetNeuronId) {
                        outputWeightSumByClass[i] += syn->getWeight();
                        outputWeightCountByClass[i]++;
                        break;
                    }
                }
            }
        }
        std::cout << "  Weight deltas (mean): input->L4=" << (curInputToL4.mean - baseInputToL4.mean)
                  << " L4->L5=" << (curL4ToL5.mean - baseL4ToL5.mean)
                  << " L5->Output=" << (curL5ToOutput.mean - baseL5ToOutput.mean) << std::endl;
        size_t l5OutputChanged = 0;
        double l5OutputDeltaSum = 0.0;
        double l5OutputAbsSum = 0.0;
        double l5OutputAbsMax = 0.0;
        for (uint64_t sid : synL5ToOutputIds) {
            auto syn = datastore.getSynapse(sid);
            if (!syn) continue;
            auto it = baseL5ToOutputWeights.find(sid);
            if (it == baseL5ToOutputWeights.end()) continue;
            double delta = syn->getWeight() - it->second;
            if (std::abs(delta) > 1e-6) {
                l5OutputChanged++;
            }
            l5OutputDeltaSum += delta;
            l5OutputAbsSum += std::abs(delta);
            l5OutputAbsMax = std::max(l5OutputAbsMax, std::abs(delta));
        }
        std::cout << "  L5->Output weight delta: changed=" << l5OutputChanged
                  << "/" << synL5ToOutputIds.size()
                  << " mean=" << (synL5ToOutputIds.empty() ? 0.0 : (l5OutputDeltaSum / synL5ToOutputIds.size()))
                  << " mean|abs|=" << (synL5ToOutputIds.empty() ? 0.0 : (l5OutputAbsSum / synL5ToOutputIds.size()))
                  << " max|abs|=" << l5OutputAbsMax << std::endl;
        std::cout << "  L5->Output weight mean by class: ";
        for (int i = 0; i < NUM_LETTERS; ++i) {
            double meanW = outputWeightCountByClass[i] > 0
                ? (outputWeightSumByClass[i] / static_cast<double>(outputWeightCountByClass[i]))
                : 0.0;
            std::cout << char('A' + i) << "=" << std::fixed << std::setprecision(3) << meanW << " ";
        }
        std::cout << std::endl;
        std::cout.flush();

        if (currentPass == 1 && diagReplayCaptured && diagReplayIdx < trainLoader.size()) {
            auto replayResult = runInference(trainLoader.getImage(diagReplayIdx));
            double replayCosine = cosineSimilarity(diagTrainL5Counts, replayResult.l5Counts);
            std::cout << "  Replay diagnostic: train/test L5 cosine="
                      << std::fixed << std::setprecision(3) << replayCosine
                      << " (train=" << diagTrainL5Pattern.size()
                      << ", test=" << replayResult.l5Pattern.size() << ")" << std::endl;
        }

        double selfSum = 0.0;
        double otherSum = 0.0;
        double marginSum = 0.0;
        size_t sampleCount = 0;
        for (int letter = 0; letter < NUM_LETTERS; ++letter) {
            if (!config.includeLetters[letter]) {
                continue;
            }
            for (const auto& pattern : letterL5Patterns[letter]) {
                double selfSim = centroidSimilarity(pattern, letter);
                double bestOther = 0.0;
                for (int other = 0; other < NUM_LETTERS; ++other) {
                    if (other == letter || !config.includeLetters[other]) continue;
                    bestOther = std::max(bestOther, centroidSimilarity(pattern, other));
                }
                selfSum += selfSim;
                otherSum += bestOther;
                marginSum += (selfSim - bestOther);
                sampleCount++;
            }
        }
        if (sampleCount > 0) {
            std::cout << "  Centroid spread (train patterns): self="
                      << std::fixed << std::setprecision(3) << (selfSum / sampleCount)
                      << " other=" << (otherSum / sampleCount)
                      << " margin=" << (marginSum / sampleCount) << std::endl;
        }

        // ====================================================================
        // Active Classification Testing Phase
        // ====================================================================
        // Disable STDP during testing to prevent weight drift
        networkPropagator->setStdpEnabled(false);
        spikeProcessor->setStdpEnabled(false);

        std::vector<size_t> testIndices;
        testIndices.reserve(testLoader.size());
        for (size_t idx = 0; idx < testLoader.size(); ++idx) {
            int label = testLoader.getImage(idx).label - 1;
            if (label < 0 || label >= NUM_LETTERS || !config.includeLetters[label]) {
                continue;
            }
            testIndices.push_back(idx);
            if (testLimit > 0 && testIndices.size() >= static_cast<size_t>(testLimit)) {
                break;
            }
        }
        size_t numTestImages = testIndices.size();
        std::cout << "  Testing with active classification (" << numTestImages << " images)..." << std::endl;
        std::cout.flush();
        int testCorrect = 0;
        int testTotal = 0;

        for (size_t testPos = 0; testPos < numTestImages; ++testPos) {
            size_t testIdx = testIndices[testPos];
            const auto& emnistImg = testLoader.getImage(testIdx);
            int label = emnistImg.label - 1;

            auto inference = runInference(emnistImg);
            const auto& testL5Pattern = inference.l5Pattern;
            const auto& testL5Counts = inference.l5Counts;

            // Primary: use output spike counts (if any) for a simple vote
            int predictedLabel = -1;
            double maxSimilarity = 0.0;

            // First prefer output spikes (supervised signal should bias these).
            if (enableOutputVote) {
                auto maxIt = std::max_element(inference.outSpikeCounts.begin(), inference.outSpikeCounts.end());
                if (maxIt != inference.outSpikeCounts.end() && *maxIt > 0) {
                    // Require a margin over the runner-up to trust the output vote
                    auto sorted = inference.outSpikeCounts;
                    std::sort(sorted.begin(), sorted.end(), std::greater<int>());
                    int top = sorted[0];
                    int second = sorted[1];
                    if (top >= 5 && top >= static_cast<int>(second * 1.1)) {
                        predictedLabel = static_cast<int>(std::distance(inference.outSpikeCounts.begin(), maxIt));
                        maxSimilarity = *maxIt;
                    }
                }
            }

            // If output layer is silent or inconclusive, fall back to pattern matching
            if (predictedLabel < 0) {
                auto res = findBestMatchingLetter(testL5Counts);
                predictedLabel = res.first;
                maxSimilarity = res.second;
                if (predictedLabel < 0) {
                    auto res2 = findBestMatchingLetterCentroid(testL5Counts);
                    predictedLabel = res2.first;
                    maxSimilarity = res2.second;
                }
            }

            if (testPos < 3) {
                size_t activeOut = 0;
                for (int i = 0; i < NUM_LETTERS; ++i) {
                    if (inference.outSpikeCounts[i] > 0) {
                        activeOut++;
                    }
                }
                std::cout << "    Output spike counts: ";
                for (int i = 0; i < NUM_LETTERS; ++i) {
                    std::cout << char('A' + i) << "=" << inference.outSpikeCounts[i] << " ";
                }
                std::cout << std::endl;
                if (config.enableOutputCompetition &&
                    inference.rawOutSpikeCounts != inference.outSpikeCounts) {
                    std::cout << "    Output spike counts (raw): ";
                    for (int i = 0; i < NUM_LETTERS; ++i) {
                        std::cout << char('A' + i) << "=" << inference.rawOutSpikeCounts[i] << " ";
                    }
                    std::cout << std::endl;
                }
                std::cout << "    Debug test " << testIdx << ": L5 active=" << testL5Pattern.size()
                          << ", output active=" << activeOut << " predicted=" << predictedLabel
                          << " label=" << label << " maxSim=" << maxSimilarity << std::endl;
            }

            testTotal++;
            if (predictedLabel == label) {
                testCorrect++;
            }

            // Progress output every image during testing
            if (testTotal % 5 == 0 || testTotal == 1) {
                double acc = 100.0 * testCorrect / testTotal;
                std::cout << "    Testing: " << testTotal << "/" << numTestImages
                          << " (" << std::fixed << std::setprecision(2) << acc << "%)" << std::endl;
                std::cout.flush();
            }
        }

        // Re-enable STDP for the next training pass
        networkPropagator->setStdpEnabled(true);
        spikeProcessor->setStdpEnabled(true);

        double currentAccuracy = (testTotal > 0) ? (100.0 * testCorrect / testTotal) : 0.0;
        std::cout << "  Pass " << currentPass << " accuracy: " << std::fixed << std::setprecision(2)
                  << currentAccuracy << "% (" << testCorrect << "/" << testTotal << ")" << std::endl;
        passAccuracies.push_back(currentAccuracy);

        // Check for convergence
        if (previousAccuracy >= 0) {
            double accuracyChange = std::abs(currentAccuracy - previousAccuracy);
            if (accuracyChange < ACCURACY_EPSILON) {
                stablePasses++;
                std::cout << "  Accuracy stable for " << stablePasses << " passes" << std::endl;
                if (stablePasses >= STABLE_PASSES_REQUIRED) {
                    std::cout << "\n=== Accuracy Converged ===" << std::endl;
                    break;
                }
            } else {
                stablePasses = 0;
            }
        }
        previousAccuracy = currentAccuracy;
    }

    auto trainingEnd = std::chrono::high_resolution_clock::now();
    double trainingTime = std::chrono::duration<double>(trainingEnd - trainingStart).count();

    std::cout << std::endl;
    std::cout << "\n=== Multi-Pass Training Complete ===" << std::endl;
    std::cout << "Total passes: " << currentPass << std::endl;
    std::cout << "Total patterns learned: " << totalPatternsLearned << std::endl;
    std::cout << "Total training time: " << std::fixed << std::setprecision(2) << trainingTime << " seconds" << std::endl;
    std::cout << "Final accuracy: " << std::fixed << std::setprecision(2) << previousAccuracy << "%" << std::endl;
    if (!passAccuracies.empty()) {
        std::cout << "Pass accuracies: ";
        for (size_t i = 0; i < passAccuracies.size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(2) << passAccuracies[i];
        }
        std::cout << std::endl;
    }

    // ========================================================================
    // Final Comprehensive Testing Phase
    // ========================================================================
    // Ensure STDP is disabled for final testing (no weight drift)
    networkPropagator->setStdpEnabled(false);
    spikeProcessor->setStdpEnabled(false);

    std::vector<size_t> finalTestIndices;
    finalTestIndices.reserve(testLoader.size());
    for (size_t idx = 0; idx < testLoader.size(); ++idx) {
        int label = testLoader.getImage(idx).label - 1;
        if (label < 0 || label >= NUM_LETTERS || !config.includeLetters[label]) {
            continue;
        }
        finalTestIndices.push_back(idx);
        if (testLimit > 0 && finalTestIndices.size() >= static_cast<size_t>(testLimit)) {
            break;
        }
    }
    size_t numFinalTestImages = finalTestIndices.size();
    std::cout << "\n=== Final Testing Phase (" << numFinalTestImages << " images) ===" << std::endl;
    int testCorrect = 0;
    int testTotal = 0;

    auto testingStart = std::chrono::high_resolution_clock::now();

        for (size_t testPos = 0; testPos < numFinalTestImages; ++testPos) {
            size_t testIdx = finalTestIndices[testPos];
            const auto& emnistImg = testLoader.getImage(testIdx);
            int label = emnistImg.label - 1;

        auto inference = runInference(emnistImg);
        const auto& testL5Pattern = inference.l5Pattern;
        const auto& testL5Counts = inference.l5Counts;

        // Prefer output spikes first (with margin requirement)
        auto maxIt = std::max_element(inference.outSpikeCounts.begin(), inference.outSpikeCounts.end());
        int predictedLabel = -1;
        double maxSimilarity = 0.0;
        if (enableOutputVote && maxIt != inference.outSpikeCounts.end() && *maxIt > 0) {
            auto sorted = inference.outSpikeCounts;
            std::sort(sorted.begin(), sorted.end(), std::greater<int>());
            int top = sorted[0];
            int second = sorted[1];
            if (top >= 5 && top >= static_cast<int>(second * 1.1)) {
                predictedLabel = static_cast<int>(std::distance(inference.outSpikeCounts.begin(), maxIt));
                maxSimilarity = *maxIt;
            }
        }

        // Fall back to k-NN / centroid if output silent or inconclusive
        if (predictedLabel < 0) {
            auto res = findBestMatchingLetter(testL5Counts);
            predictedLabel = res.first;
            maxSimilarity = res.second;
            if (predictedLabel < 0) {
                auto res2 = findBestMatchingLetterCentroid(testL5Counts);
                predictedLabel = res2.first;
                maxSimilarity = res2.second;
            }
        }

        // Debug output for first few tests
            if (testPos < 3) {
                std::cout << "  Test " << testIdx << " (label=" << label << "): "
                          << "Predicted=" << predictedLabel
                          << ", MaxSim=" << std::fixed << std::setprecision(3) << maxSimilarity << std::endl;

                // Show centroid sims for all letters
                std::cout << "    Centroid similarities: ";
                for (int i = 0; i < NUM_LETTERS; ++i) {
                    double sim = centroidSimilarity(testL5Counts, i);
                    std::cout << char('A' + i) << "=" << std::fixed << std::setprecision(3) << sim << " ";
                }
                std::cout << std::endl;

                // Show output spike counts (top 5)
                std::vector<std::pair<size_t,int>> spikeRank;
                for (int i = 0; i < NUM_LETTERS; ++i) {
                    spikeRank.push_back({static_cast<size_t>(inference.outSpikeCounts[i]), i});
                }
                std::sort(spikeRank.begin(), spikeRank.end(), [](auto& a, auto& b){ return a.first > b.first; });
                std::cout << "    Output spikes (top 5): ";
                for (int i = 0; i < 5; ++i) {
                    std::cout << char('A' + spikeRank[i].second) << "=" << spikeRank[i].first << " ";
                }
                std::cout << std::endl;

                std::cout << "    L5 stats: " << testL5Pattern.size() << " active neurons, "
                          << inference.totalL5Spikes << " total spikes" << std::endl;

                // Show L5 pattern counts for correct letter vs predicted letter
                std::cout << "    L5 patterns: correct(" << char('A' + label) << ")=" << letterL5Patterns[label].size();
                if (predictedLabel >= 0 && predictedLabel < NUM_LETTERS) {
                    std::cout << ", predicted(" << char('A' + predictedLabel) << ")=" << letterL5Patterns[predictedLabel].size();
                } else {
                    std::cout << ", predicted(none)";
                }
                std::cout << std::endl;
            }

        testTotal++;
        if (predictedLabel == label) {
            testCorrect++;
        }

        if (testTotal % 2000 == 0) {
            double acc = 100.0 * testCorrect / testTotal;
            std::cout << "  Final testing: " << testTotal << "/" << numFinalTestImages
                      << " (" << std::fixed << std::setprecision(2) << acc << "%)" << std::endl;
            std::cout.flush();
        }
    }

    auto testingEnd = std::chrono::high_resolution_clock::now();
    double testingTime = std::chrono::duration<double>(testingEnd - testingStart).count();
    double testAccuracy = (testTotal > 0) ? (100.0 * testCorrect / testTotal) : 0.0;

    std::cout << std::endl;
    std::cout << "\n=== Final Testing Complete ===" << std::endl;
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
    std::cout << "Training passes: " << currentPass << std::endl;
    std::cout << "Total training images: " << imagesProcessed << std::endl;
    std::cout << "Testing images: " << testTotal << std::endl;
    std::cout << "Total patterns learned: " << totalPatternsLearned << std::endl;
    std::cout << "Final accuracy: " << std::fixed << std::setprecision(2) << testAccuracy << "%" << std::endl;
    std::cout << "Convergence: " << (stablePasses >= STABLE_PASSES_REQUIRED ? "Yes" : "No") << std::endl;

    std::cout << "\n=== Experiment Complete ===" << std::endl;

    // Cleanup
    if (recordingManager) {
        delete recordingManager;
    }

    return 0;
}
