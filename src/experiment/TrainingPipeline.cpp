#include "snnfw/experiment/TrainingPipeline.h"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cmath>
#include <set>
#include <random>

namespace snnfw {
namespace experiment {

namespace {
constexpr double kL4ToL5SettleMs = 30.0;
}

// Timing helper for performance profiling
static int g_imageCount = 0;
static std::chrono::steady_clock::time_point g_lastReport;
static double g_totalEncodeMs = 0;
static double g_totalWait1Ms = 0;
static double g_totalL4Ms = 0;
static double g_totalWait2Ms = 0;
static double g_totalL5Ms = 0;
static double g_totalPropMs = 0;
static double g_totalCollectMs = 0;

TrainingPipeline::TrainingPipeline(const ExperimentConfig& config,
                                   declarative::ConstructedNetwork& network)
    : config_(config)
    , network_(network)
    , encoder_(config, network.spikeProcessor, network.propagator)
    , competition_(config)
    , classifier_(config)
    , teacher_(config)
{
    g_lastReport = std::chrono::steady_clock::now();
}

InferenceResult TrainingPipeline::runInference(const EMNISTLoader::Image& image) {
    auto t0 = std::chrono::steady_clock::now();

    InferenceResult result;
    size_t totalL5 = static_cast<size_t>(config_.numColumns) * config_.layer5Neurons;
    result.l5Counts.assign(totalL5, 0);
    result.outSpikeCounts.assign(config_.numClasses, 0);

    auto t1 = std::chrono::steady_clock::now();
    double baseTime = encoder_.encodeAndInject(
        image, network_.inputNeurons, network_.columns, network_.outputPopulations);
    auto t2 = std::chrono::steady_clock::now();
    g_totalEncodeMs += std::chrono::duration<double, std::milli>(t2 - t1).count();

    // Wait for input spikes to propagate
    encoder_.waitForSimTime(baseTime + config_.inputLatencyMs + 5.0, 200.0);
    auto t3 = std::chrono::steady_clock::now();
    g_totalWait1Ms += std::chrono::duration<double, std::milli>(t3 - t2).count();

    // L4 competition
    auto colHasL4 = competition_.runL4Competition(
        network_.columns, encoder_.getLastInputFired(), baseTime, network_.propagator);
    auto t4 = std::chrono::steady_clock::now();
    g_totalL4Ms += std::chrono::duration<double, std::milli>(t4 - t3).count();

    // Wait for L4 signatures (~0-100ms) to propagate into L5 before competition.
    encoder_.waitForSimTime(baseTime + kL4ToL5SettleMs, 200.0);
    auto t5 = std::chrono::steady_clock::now();
    g_totalWait2Ms += std::chrono::duration<double, std::milli>(t5 - t4).count();

    // L5 competition
    auto l5Winners = competition_.runL5Competition(
        network_.columns, colHasL4, baseTime, network_.propagator);
    auto t6 = std::chrono::steady_clock::now();
    g_totalL5Ms += std::chrono::duration<double, std::milli>(t6 - t5).count();

    // Fire L5 winners and output neurons in full propagation mode
    if (config_.enableFullPropagation) {
        int colIdxSeq = 0;
        size_t l5Offset = 0;
        for (auto& col : network_.columns) {
            auto l5It = col.layerNeurons.find("L5");
            if (l5It == col.layerNeurons.end()) { colIdxSeq++; continue; }
            auto& l5Neurons = l5It->second;
            if (colIdxSeq < static_cast<int>(colHasL4.size()) && !colHasL4[colIdxSeq]) {
                l5Offset += l5Neurons.size();
                colIdxSeq++;
                continue;
            }
            int localIdx = 0;
            for (auto& l5Neuron : l5Neurons) {
                if (l5Offset + localIdx < l5Winners.size() &&
                    l5Winners[l5Offset + localIdx] &&
                    l5Neuron->getInhibition() <= config_.l5InhibitThreshold) {
                    double l5FireTime = baseTime + 15.0 + (colIdxSeq * 0.1) + (localIdx * 0.02);
                    l5Neuron->fireSignature(l5FireTime);
                    network_.propagator->fireNeuron(l5Neuron->getId(), l5FireTime);
                    l5Neuron->fireAndAcknowledge(l5FireTime);
                }
                localIdx++;
            }
            l5Offset += l5Neurons.size();
            colIdxSeq++;
        }

        // Fire output neurons that received spikes
        int letterIdx = 0;
        for (auto& pop : network_.outputPopulations) {
            int localIdx = 0;
            for (auto& outNeuron : pop) {
                if (!outNeuron->getSpikes().empty()) {
                    double outFireTime = baseTime + 25.0 + (letterIdx * 0.1) + (localIdx * 0.01);
                    network_.propagator->fireNeuron(outNeuron->getId(), outFireTime);
                    outNeuron->fireAndAcknowledge(outFireTime);
                }
                localIdx++;
            }
            letterIdx++;
        }
    }

    auto t7 = std::chrono::steady_clock::now();

    // Collect L5 counts
    collectL5Counts(result.l5Counts, l5Winners);

    // Collect output spike counts
    for (int i = 0; i < config_.numClasses; ++i) {
        if (i < static_cast<int>(network_.outputPopulations.size())) {
            for (auto& n : network_.outputPopulations[i]) {
                result.outSpikeCounts[i] += static_cast<int>(n->getSpikes().size());
            }
        }
    }
    result.rawOutSpikeCounts = result.outSpikeCounts;
    competition_.applyOutputCompetition(result.outSpikeCounts);

    auto t8 = std::chrono::steady_clock::now();
    g_totalCollectMs += std::chrono::duration<double, std::milli>(t8 - t7).count();
    g_totalPropMs += std::chrono::duration<double, std::milli>(t7 - t6).count();

    // Report timing every 100 images
    g_imageCount++;
    if (g_imageCount % 100 == 0) {
        auto now = std::chrono::steady_clock::now();
        double totalSec = std::chrono::duration<double>(now - g_lastReport).count();
        std::cout << "[PERF] 100 images in " << totalSec << "s ("
                  << (100.0 / totalSec) << " img/s) - "
                  << "Encode:" << (g_totalEncodeMs/100.0) << "ms "
                  << "Wait1:" << (g_totalWait1Ms/100.0) << "ms "
                  << "L4:" << (g_totalL4Ms/100.0) << "ms "
                  << "Wait2:" << (g_totalWait2Ms/100.0) << "ms "
                  << "L5:" << (g_totalL5Ms/100.0) << "ms "
                  << "Prop:" << (g_totalPropMs/100.0) << "ms "
                  << "Collect:" << (g_totalCollectMs/100.0) << "ms"
                  << std::endl;
        g_lastReport = now;
        g_totalEncodeMs = g_totalWait1Ms = g_totalL4Ms = g_totalWait2Ms = 0;
        g_totalL5Ms = g_totalPropMs = g_totalCollectMs = 0;
    }

    return result;
}

void TrainingPipeline::collectL5Counts(std::vector<uint16_t>& counts,
                                        const std::vector<bool>& l5Winners) {
    size_t offset = 0;
    for (auto& col : network_.columns) {
        auto l5It = col.layerNeurons.find("L5");
        if (l5It == col.layerNeurons.end()) continue;
        auto& l5Neurons = l5It->second;
        for (size_t i = 0; i < l5Neurons.size(); ++i) {
            size_t globalIdx = offset + i;
            size_t spikes = l5Neurons[i]->getSpikes().size();
            if (spikes > 0 && globalIdx < l5Winners.size() && l5Winners[globalIdx] &&
                l5Neurons[i]->getInhibition() <= config_.l5InhibitThreshold) {
                if (globalIdx < counts.size()) {
                    counts[globalIdx] = static_cast<uint16_t>(std::min<size_t>(spikes, 65535));
                }
            }
        }
        offset += l5Neurons.size();
    }
}

void TrainingPipeline::applyHomeostasis() {
    for (auto& col : network_.columns) {
        for (auto& [layerName, neurons] : col.layerNeurons) {
            for (auto& n : neurons) {
                n->applyHomeostaticPlasticity();
            }
        }
    }
    for (auto& pop : network_.outputPopulations) {
        for (auto& n : pop) {
            n->applyHomeostaticPlasticity();
        }
    }
}

double TrainingPipeline::run(EMNISTLoader& trainLoader, EMNISTLoader& testLoader) {
    int currentPass = 0;
    double previousAccuracy = -1.0;
    int stablePasses = 0;

    while (currentPass < config_.maxPasses) {
        currentPass++;
        std::cout << "\n--- Training Pass " << currentPass << " ---" << std::endl;

        // Reset classifier patterns for this pass
        classifier_.resetForPass(config_.keepL5History);
        network_.propagator->resetStdpUpdateStats();

        std::vector<int> trainCount(config_.numClasses, 0);
        int passPatterns = 0;
        int passImages = 0;

        std::cout << "  Processing " << trainLoader.size() << " training images..." << std::endl;

        for (size_t imgIdx = 0; imgIdx < trainLoader.size(); ++imgIdx) {
            const auto& emnistImg = trainLoader.getImage(imgIdx);
            int label = emnistImg.label - 1;  // Convert 1-26 to 0-25

            if (label < 0 || label >= config_.numClasses) continue;
            if (!config_.includeClasses.empty() &&
                label < static_cast<int>(config_.includeClasses.size()) &&
                !config_.includeClasses[label]) continue;
            if (trainCount[label] >= config_.trainingExamplesPerClass) continue;

            // Encode and inject spikes
            double baseTime = encoder_.encodeAndInject(
                emnistImg, network_.inputNeurons, network_.columns, network_.outputPopulations);

            // Wait for input propagation
            encoder_.waitForSimTime(baseTime + config_.inputLatencyMs + 5.0, 200.0);

            // L4 competition
            auto colHasL4 = competition_.runL4Competition(
                network_.columns, encoder_.getLastInputFired(), baseTime, network_.propagator);

            // Wait for L4 signatures (~0-100ms) to propagate into L5 before competition.
            encoder_.waitForSimTime(baseTime + kL4ToL5SettleMs, 200.0);

            // L5 competition
            auto l5Winners = competition_.runL5Competition(
                network_.columns, colHasL4, baseTime, network_.propagator);

            // Supervised teaching
            int taught = teacher_.teach(
                label, network_.columns, l5Winners, colHasL4,
                network_.outputPopulations, baseTime, network_.propagator);
            passPatterns += taught;

            // Collect L5 counts and store for classification
            size_t totalL5 = static_cast<size_t>(config_.numColumns) * config_.layer5Neurons;
            L5CountVector firedL5Counts(totalL5, 0);
            collectL5Counts(firedL5Counts, l5Winners);

            // Check if any L5 neurons fired
            bool hasL5Activity = false;
            for (auto c : firedL5Counts) { if (c > 0) { hasL5Activity = true; break; } }
            if (hasL5Activity) {
                classifier_.storePattern(label, firedL5Counts);
            }

            trainCount[label]++;
            passImages++;

            if (passImages % 500 == 0) {
                std::cout << "  Training: " << passImages << " images processed" << std::endl;
            }

            // Check if all classes trained
            bool allTrained = true;
            for (int i = 0; i < config_.numClasses; ++i) {
                if (!config_.includeClasses.empty() &&
                    i < static_cast<int>(config_.includeClasses.size()) &&
                    !config_.includeClasses[i]) continue;
                if (trainCount[i] < config_.trainingExamplesPerClass) {
                    allTrained = false;
                    break;
                }
            }
            if (allTrained) break;
        }

        totalPatternsLearned_ += passPatterns;
        std::cout << "  Pass " << currentPass << " complete: " << passPatterns
                  << " patterns learned, " << passImages << " images" << std::endl;

        // Apply homeostasis at end of pass
        applyHomeostasis();

        // Check for missing class patterns
        if (classifier_.hasMissingClasses()) {
            std::cerr << "  WARNING: Missing L5 patterns for one or more classes." << std::endl;
        }

        // Run testing phase
        double currentAccuracy = runTestingPhase(testLoader);
        std::cout << "  Pass " << currentPass << " accuracy: " << std::fixed
                  << std::setprecision(2) << currentAccuracy << "%" << std::endl;
        passAccuracies_.push_back(currentAccuracy);

        // Check convergence
        if (previousAccuracy >= 0) {
            double accuracyChange = std::abs(currentAccuracy - previousAccuracy);
            if (accuracyChange < config_.accuracyEpsilon) {
                stablePasses++;
                std::cout << "  Accuracy stable for " << stablePasses << " passes" << std::endl;
                if (stablePasses >= config_.stablePassesRequired) {
                    std::cout << "\n=== Accuracy Converged ===" << std::endl;
                    break;
                }
            } else {
                stablePasses = 0;
            }
        }
        previousAccuracy = currentAccuracy;
    }

    return passAccuracies_.empty() ? 0.0 : passAccuracies_.back();
}

double TrainingPipeline::runTestingPhase(EMNISTLoader& testLoader) {
    // Disable STDP during testing
    network_.propagator->setStdpEnabled(false);
    network_.spikeProcessor->setStdpEnabled(false);

    std::vector<size_t> testIndices;
    testIndices.reserve(testLoader.size());
    if (config_.testLimit > 0) {
        std::vector<std::vector<size_t>> byClass(config_.numClasses);
        for (size_t idx = 0; idx < testLoader.size(); ++idx) {
            int label = testLoader.getImage(idx).label - 1;
            if (label < 0 || label >= config_.numClasses) continue;
            if (!config_.includeClasses.empty() &&
                label < static_cast<int>(config_.includeClasses.size()) &&
                !config_.includeClasses[label]) continue;
            byClass[label].push_back(idx);
        }
        std::mt19937 rng(config_.seed);
        std::vector<int> activeClasses;
        for (int cls = 0; cls < config_.numClasses; ++cls) {
            if (!config_.includeClasses.empty() &&
                cls < static_cast<int>(config_.includeClasses.size()) &&
                !config_.includeClasses[cls]) {
                continue;
            }
            if (!byClass[cls].empty()) {
                std::shuffle(byClass[cls].begin(), byClass[cls].end(), rng);
                activeClasses.push_back(cls);
            }
        }
        if (!activeClasses.empty()) {
            size_t cursor = 0;
            while (testIndices.size() < static_cast<size_t>(config_.testLimit)) {
                bool pushed = false;
                for (int cls : activeClasses) {
                    if (cursor < byClass[cls].size()) {
                        testIndices.push_back(byClass[cls][cursor]);
                        pushed = true;
                        if (testIndices.size() >= static_cast<size_t>(config_.testLimit)) {
                            break;
                        }
                    }
                }
                if (!pushed) break;
                cursor++;
            }
        }
    } else {
        for (size_t idx = 0; idx < testLoader.size(); ++idx) {
            int label = testLoader.getImage(idx).label - 1;
            if (label < 0 || label >= config_.numClasses) continue;
            if (!config_.includeClasses.empty() &&
                label < static_cast<int>(config_.includeClasses.size()) &&
                !config_.includeClasses[label]) continue;
            testIndices.push_back(idx);
        }
    }

    size_t numTestImages = testIndices.size();
    std::cout << "  Testing with active classification (" << numTestImages << " images)..." << std::endl;

    int testCorrect = 0;
    int testTotal = 0;
    std::vector<std::vector<int>> confusion(
        config_.numClasses, std::vector<int>(config_.numClasses, 0));
    std::vector<int> unknownByClass(config_.numClasses, 0);

    for (size_t testPos = 0; testPos < numTestImages; ++testPos) {
        size_t testIdx = testIndices[testPos];
        const auto& emnistImg = testLoader.getImage(testIdx);
        int label = emnistImg.label - 1;

        auto inference = runInference(emnistImg);

        int predictedLabel = -1;
        double maxSimilarity = 0.0;

        // First prefer output spikes
        if (config_.enableOutputVote) {
            auto maxIt = std::max_element(inference.outSpikeCounts.begin(),
                                          inference.outSpikeCounts.end());
            if (maxIt != inference.outSpikeCounts.end() && *maxIt > 0) {
                auto sorted = inference.outSpikeCounts;
                std::sort(sorted.begin(), sorted.end(), std::greater<int>());
                int top = sorted[0];
                int second = sorted.size() > 1 ? sorted[1] : 0;
                if (top >= 5 && top >= static_cast<int>(second * 1.1)) {
                    predictedLabel = static_cast<int>(
                        std::distance(inference.outSpikeCounts.begin(), maxIt));
                    maxSimilarity = top;
                }
            }
        }

        // Fall back to k-NN
        if (predictedLabel < 0) {
            auto res = classifier_.classifyKNN(inference.l5Counts);
            predictedLabel = res.first;
            maxSimilarity = res.second;
            if (predictedLabel < 0) {
                auto res2 = classifier_.classifyCentroid(inference.l5Counts);
                predictedLabel = res2.first;
                maxSimilarity = res2.second;
            }
        }

        // DEBUG: Log first few test predictions
        if (testTotal < 5) {
            std::cout << "    DEBUG Test " << testIdx << ": label=" << label
                      << " predicted=" << predictedLabel
                      << " outVote=" << (maxSimilarity > 0 && predictedLabel >= 0 && config_.enableOutputVote ? "YES" : "NO")
                      << " maxSim=" << std::fixed << std::setprecision(3) << maxSimilarity
                      << " L5active=" << std::count_if(inference.l5Counts.begin(), inference.l5Counts.end(),
                                                        [](uint16_t c) { return c > 0; })
                      << std::endl;
        }

        testTotal++;
        if (predictedLabel == label) {
            testCorrect++;
        }
        if (predictedLabel >= 0 && predictedLabel < config_.numClasses) {
            confusion[label][predictedLabel]++;
        } else {
            unknownByClass[label]++;
        }

        if (testTotal % 50 == 0 || testTotal == 1) {
            double acc = 100.0 * testCorrect / testTotal;
            std::cout << "    Testing: " << testTotal << "/" << numTestImages
                      << " (" << std::fixed << std::setprecision(2) << acc << "%)" << std::endl;
        }
    }

    // Re-enable STDP
    network_.propagator->setStdpEnabled(true);
    network_.spikeProcessor->setStdpEnabled(true);

    // Confusion matrix (rows = true, cols = predicted)
    std::cout << "\n  Confusion Matrix (rows=true, cols=pred):" << std::endl;
    std::cout << "     ";
    for (int c = 0; c < config_.numClasses; ++c) {
        char labelChar = (c < 26) ? static_cast<char>('A' + c) : '?';
        std::cout << std::setw(4) << labelChar;
    }
    std::cout << std::setw(6) << "UNK";
    std::cout << std::endl;
    for (int r = 0; r < config_.numClasses; ++r) {
        char labelChar = (r < 26) ? static_cast<char>('A' + r) : '?';
        std::cout << "  " << std::setw(2) << labelChar << " ";
        for (int c = 0; c < config_.numClasses; ++c) {
            std::cout << std::setw(4) << confusion[r][c];
        }
        std::cout << std::setw(6) << unknownByClass[r];
        std::cout << std::endl;
    }

    // Per-class accuracy summary
    std::cout << "\n  Per-class accuracy:" << std::endl;
    for (int cls = 0; cls < config_.numClasses; ++cls) {
        int rowTotal = 0;
        for (int c = 0; c < config_.numClasses; ++c) {
            rowTotal += confusion[cls][c];
        }
        rowTotal += unknownByClass[cls];
        double acc = (rowTotal > 0) ? (100.0 * confusion[cls][cls] / rowTotal) : 0.0;
        char labelChar = (cls < 26) ? static_cast<char>('A' + cls) : '?';
        std::cout << "    " << labelChar << ": " << std::fixed << std::setprecision(2)
                  << acc << "% (" << confusion[cls][cls] << "/" << rowTotal << ")"
                  << std::endl;
    }

    return (testTotal > 0) ? (100.0 * testCorrect / testTotal) : 0.0;
}

} // namespace experiment
} // namespace snnfw
