#include "snnfw/experiment/ExperimentRunner.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <stdexcept>

namespace snnfw {
namespace experiment {

ExperimentRunner::ExperimentRunner(const ExperimentConfig& config)
    : config_(config)
{
}

double ExperimentRunner::run() {
    std::cout << "=== EMNIST Letters Training (SONATA Model) ===" << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();

    // Step 1: Load data
    std::cout << "\n[1/3] Loading EMNIST data..." << std::endl;
    loadData();

    // Step 2: Build network from config
    std::cout << "\n[2/3] Building network from " << config_.networkConfigPath << "..." << std::endl;
    buildNetwork();

    // Step 3: Run training pipeline
    std::cout << "\n[3/3] Starting training pipeline..." << std::endl;
    pipeline_ = std::make_unique<TrainingPipeline>(config_, *network_);
    double finalAccuracy = pipeline_->run(*trainLoader_, *testLoader_);
    passAccuracies_ = pipeline_->getPassAccuracies();

    auto endTime = std::chrono::high_resolution_clock::now();
    double totalTime = std::chrono::duration<double>(endTime - startTime).count();

    std::cout << "\n=== Experiment Complete ===" << std::endl;
    std::cout << "  Final accuracy: " << std::fixed << std::setprecision(2)
              << finalAccuracy << "%" << std::endl;
    std::cout << "  Total passes: " << passAccuracies_.size() << std::endl;
    std::cout << "  Total patterns learned: " << pipeline_->getTotalPatternsLearned() << std::endl;
    std::cout << "  Total time: " << std::fixed << std::setprecision(1) << totalTime << "s" << std::endl;

    // Print per-pass accuracies
    std::cout << "  Per-pass accuracies: ";
    for (size_t i = 0; i < passAccuracies_.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(2) << passAccuracies_[i] << "%";
    }
    std::cout << std::endl;

    return finalAccuracy;
}

void ExperimentRunner::loadData() {
    trainLoader_ = std::make_unique<EMNISTLoader>();
    testLoader_ = std::make_unique<EMNISTLoader>();

    if (!trainLoader_->load(config_.trainImagesPath, config_.trainLabelsPath)) {
        throw std::runtime_error("Failed to load training data from: " +
                                 config_.trainImagesPath);
    }
    std::cout << "  Training set: " << trainLoader_->size() << " images" << std::endl;

    if (!testLoader_->load(config_.testImagesPath, config_.testLabelsPath)) {
        throw std::runtime_error("Failed to load test data from: " +
                                 config_.testImagesPath);
    }
    std::cout << "  Test set: " << testLoader_->size() << " images" << std::endl;
}

void ExperimentRunner::buildNetwork() {
    // Create datastore and factory first (needed by DeclarativeLoader)
    datastore_ = std::make_unique<Datastore>(config_.datastorePath);
    factory_ = std::make_unique<NeuralObjectFactory>();

    // Load and parse the network description
    declarative::DeclarativeLoader loader(*factory_, *datastore_);
    auto ir = loader.parseOnly(config_.networkConfigPath);

    // Sync config from the IR
    syncConfigFromIR(ir);

    // Construct the network from the parsed IR
    network_ = std::make_unique<declarative::ConstructedNetwork>(loader.loadFromIR(ir));

    std::cout << "  Network built: " << network_->inputNeurons.size() << " input neurons, "
              << network_->columns.size() << " columns, "
              << network_->outputPopulations.size() << " output classes" << std::endl;
    std::cout << "  Total neurons: " << network_->allNeuronIds.size() << std::endl;
    std::cout << "  Total synapses: " << network_->allSynapses.size() << std::endl;

    // Start the spike processor
    network_->spikeProcessor->start();
    std::cout << "  Spike processor started" << std::endl;
}

void ExperimentRunner::syncConfigFromIR(const declarative::NetworkIR& ir) {
    // Sync encoder/input parameters from the parsed IR
    config_.pixelThreshold = ir.inputLayer.pixelThreshold;
    config_.inputLatencyMs = ir.inputLayer.latencyMs;

    // Sync architecture parameters from the parsed IR
    config_.numClasses = ir.outputLayer.numClasses;
    config_.neuronsPerOutputClass = ir.outputLayer.neuronsPerClass;
    config_.numColumns = 0;

    // Count columns from the brain hierarchy
    for (auto& hemi : ir.brain.hemispheres) {
        for (auto& lobe : hemi.lobes) {
            for (auto& region : lobe.regions) {
                for (auto& nucleus : region.nuclei) {
                    config_.numColumns += static_cast<int>(nucleus.columns.size());
                    if (nucleus.columnTemplate.has_value()) {
                        auto& tmpl = nucleus.columnTemplate.value();
                        config_.numColumns += static_cast<int>(
                            tmpl.orientations.size() * tmpl.frequencies.size());
                    }
                }
            }
        }
    }

    // Sync simulation parameters
    config_.interImageGapMs = ir.simulation.interImageGapMs;
    config_.l4Keep = ir.simulation.l4Keep;
    config_.l5Keep = ir.simulation.l5Keep;
    if (!config_.hasSimilarityRuntimeOverrides) {
        config_.enableSimilarityCompetition = ir.simulation.enableSimilarityCompetition;
        config_.l4SimilarityWeight = ir.simulation.l4SimilarityWeight;
        config_.l5SimilarityWeight = ir.simulation.l5SimilarityWeight;
        config_.traceSimilarityCompetition = ir.simulation.traceSimilarityCompetition;
    }
    config_.enableL5Inhibition = ir.simulation.enableL5Inhibition;
    config_.enableL5InterColumnInhibition = ir.simulation.enableL5InterColumnInhibition;
    config_.l5InterColumnInhibit = ir.simulation.l5InterColumnInhibit;
    config_.l5InterColumnMinOverlap = ir.simulation.l5InterColumnMinOverlap;
    config_.l5InterColumnWinnerScale = ir.simulation.l5InterColumnWinnerScale;
    config_.l5InterColumnMaxInhibit = ir.simulation.l5InterColumnMaxInhibit;
    config_.l5InterColumnMaxOrientationDeltaDeg =
        ir.simulation.l5InterColumnMaxOrientationDeltaDeg;
    config_.l5InterColumnMaxFrequencyOctaveDelta =
        ir.simulation.l5InterColumnMaxFrequencyOctaveDelta;
    config_.l5InterColumnMaxNeighbors = ir.simulation.l5InterColumnMaxNeighbors;
    config_.maskMinActive = ir.simulation.maskMinActive;
    config_.stdpLtdScale = ir.simulation.stdpLtdScale;
    config_.stdpLtdWindowMs = ir.simulation.stdpLtdWindowMs;
    config_.traceStdp = ir.simulation.traceStdp;
    config_.enableStdpEligibilityGate = ir.simulation.enableStdpEligibilityGate;
    config_.stdpEligibilityMinUpdates = ir.simulation.stdpEligibilityMinUpdates;
    config_.stdpEligibilityMinLtp = ir.simulation.stdpEligibilityMinLtp;
    config_.stdpEligibilityThreshold = ir.simulation.stdpEligibilityThreshold;
    config_.stdpEligibilityLtdPenalty = ir.simulation.stdpEligibilityLtdPenalty;
    config_.numThreads = ir.simulation.spikeProcessorThreads;

    // Determine layer sizes from column template or first column
    for (auto& hemi : ir.brain.hemispheres) {
        for (auto& lobe : hemi.lobes) {
            for (auto& region : lobe.regions) {
                for (auto& nucleus : region.nuclei) {
                    std::vector<declarative::LayerIR> const* layers = nullptr;
                    if (nucleus.columnTemplate.has_value()) {
                        layers = &nucleus.columnTemplate.value().layers;
                    } else if (!nucleus.columns.empty()) {
                        layers = &nucleus.columns.front().layers;
                    }
                    if (layers) {
                        for (auto& layer : *layers) {
                            for (auto& pop : layer.populations) {
                                if (layer.name == "L4" && !pop.gridLayout.empty()) {
                                    // Parse "7x7" -> 7
                                    auto xPos = pop.gridLayout.find('x');
                                    if (xPos != std::string::npos) {
                                        config_.layer4Size = std::stoi(
                                            pop.gridLayout.substr(0, xPos));
                                    }
                                } else if (layer.name == "L5") {
                                    config_.layer5Neurons = pop.count;
                                }
                            }
                        }
                        return; // Got what we need
                    }
                }
            }
        }
    }
}

const std::vector<double>& ExperimentRunner::getPassAccuracies() const {
    return passAccuracies_;
}

} // namespace experiment
} // namespace snnfw
