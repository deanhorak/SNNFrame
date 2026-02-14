/**
 * @file emnist_sonata_training.cpp
 * @brief EMNIST Letters training using a SONATA model description.
 *
 * This is the declarative-model-driven equivalent of emnist_letters_training.cpp.
 * Instead of building the network in C++, the network architecture is loaded
 * from a SONATA circuit_config.json file, and all training/testing logic uses
 * the reusable framework components in snnfw::experiment.
 */

#include "snnfw/experiment/ExperimentRunner.h"
#include "snnfw/Logger.h"
#include <spdlog/spdlog.h>
#include <iostream>
#include <string>
#include <cstdlib>

int main(int argc, char* argv[]) {
    // Set logging level to WARN to reduce verbosity during training
    snnfw::Logger::getInstance().setLevel(spdlog::level::warn);

    snnfw::experiment::ExperimentConfig config;

    // Default paths (can be overridden by command-line args)
    config.networkConfigPath = "configs/emnist_v1_sonata/circuit_config.json";
    config.trainImagesPath = "emnist-letters/emnist-letters-train-images-idx3-ubyte";
    config.trainLabelsPath = "emnist-letters/emnist-letters-train-labels-idx1-ubyte";
    config.testImagesPath  = "emnist-letters/emnist-letters-test-images-idx3-ubyte";
    config.testLabelsPath  = "emnist-letters/emnist-letters-test-labels-idx1-ubyte";
    config.datastorePath   = "./sonata_experiment_db";

    // Training parameters (matching emnist_letters_training.cpp defaults)
    config.numClasses = 26;
    config.trainingExamplesPerClass = 200;
    config.maxPasses = 20;
    config.seed = 42;

    // Architecture (will be overridden by SONATA config)
    config.numColumns = 16;
    config.layer4Size = 7;
    config.layer5Neurons = 80;

    // Neuron parameters
    config.neuronWindow = 500.0;
    config.neuronThreshold = 1.2;
    config.neuronMaxPatterns = 500;
    config.inputLatencyMs = 15.0;
    config.pixelThreshold = 0.4;

    // Competition
    config.l4Keep = 8;
    config.l5Keep = 8;
    config.l4RowDelay = 0.3;
    config.l4ColDelay = 0.2;
    config.l4PostShiftMs = -6.0;
    config.l4PostJitterMs = 2.0;
    config.interImageGapMs = 550.0;

    // L5 inhibition
    config.enableL5Inhibition = true;
    config.l5InhibitLoser = 1.2;
    config.l5InhibitThreshold = 0.5;
    config.l5MinSpikes = 1;
    config.enableL5InterColumnInhibition = true;
    config.l5InterColumnInhibit = 0.02;
    config.l5InterColumnMinOverlap = 0.30;
    config.l5InterColumnWinnerScale = 0.50;
    config.l5InterColumnMaxInhibit = 0.35;
    config.l5InterColumnMaxOrientationDeltaDeg = 90.0;
    config.l5InterColumnMaxFrequencyOctaveDelta = 2.0;
    config.l5InterColumnMaxNeighbors = 64;

    // STDP
    config.stdpLtdScale = 0.3;
    config.stdpLtdWindowMs = 70.0;
    config.traceStdp = true;

    // Classification
    config.knnK = 7;
    config.maxPatternsPerClass = 2048;
    config.keepL5History = true;
    config.enableOutputVote = false;
    config.enableFullPropagation = true;
    config.disableOutputTeach = false;

    // Output competition
    config.enableOutputCompetition = true;
    config.outputCompetitionKeep = 1;
    config.outputCompetitionMinSpikes = 5;

    // Convergence
    config.accuracyEpsilon = 0.001;
    config.stablePassesRequired = 3;

    // Include all 26 letters
    config.includeClasses.assign(26, true);

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config.networkConfigPath = argv[++i];
        } else if (arg == "--train-images" && i + 1 < argc) {
            config.trainImagesPath = argv[++i];
        } else if (arg == "--train-labels" && i + 1 < argc) {
            config.trainLabelsPath = argv[++i];
        } else if (arg == "--test-images" && i + 1 < argc) {
            config.testImagesPath = argv[++i];
        } else if (arg == "--test-labels" && i + 1 < argc) {
            config.testLabelsPath = argv[++i];
        } else if (arg == "--datastore" && i + 1 < argc) {
            config.datastorePath = argv[++i];
        } else if (arg == "--max-passes" && i + 1 < argc) {
            config.maxPasses = std::atoi(argv[++i]);
        } else if (arg == "--examples-per-class" && i + 1 < argc) {
            config.trainingExamplesPerClass = std::atoi(argv[++i]);
        } else if (arg == "--test-limit" && i + 1 < argc) {
            config.testLimit = std::atoi(argv[++i]);
        } else if (arg == "--threads" && i + 1 < argc) {
            config.numThreads = std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            config.seed = std::atoi(argv[++i]);
        } else if (arg == "--no-output-vote") {
            config.enableOutputVote = false;
        } else if (arg == "--output-vote") {
            config.enableOutputVote = true;
        } else if (arg == "--no-output-competition") {
            config.enableOutputCompetition = false;
        } else if (arg == "--no-output-teach") {
            config.disableOutputTeach = true;
        } else if (arg == "--keep-l5-history") {
            config.keepL5History = true;
        } else if (arg == "--no-l5-inter-column-inhibit") {
            config.enableL5InterColumnInhibition = false;
        } else if (arg == "--l5-inter-column-inhibit" && i + 1 < argc) {
            config.enableL5InterColumnInhibition = true;
            config.l5InterColumnInhibit = std::atof(argv[++i]);
        } else if (arg == "--l5-inter-column-min-overlap" && i + 1 < argc) {
            config.l5InterColumnMinOverlap = std::atof(argv[++i]);
        } else if (arg == "--l5-inter-column-winner-scale" && i + 1 < argc) {
            config.l5InterColumnWinnerScale = std::atof(argv[++i]);
        } else if (arg == "--l5-inter-column-max" && i + 1 < argc) {
            config.l5InterColumnMaxInhibit = std::atof(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --config <path>           SONATA circuit_config.json path\n"
                      << "  --train-images <path>     EMNIST training images path\n"
                      << "  --train-labels <path>     EMNIST training labels path\n"
                      << "  --test-images <path>      EMNIST test images path\n"
                      << "  --test-labels <path>      EMNIST test labels path\n"
                      << "  --datastore <path>        Datastore directory path\n"
                      << "  --max-passes <n>          Maximum training passes\n"
                      << "  --examples-per-class <n>  Training examples per class\n"
                      << "  --test-limit <n>          Max test images (0 = all)\n"
                      << "  --threads <n>             Spike processor threads\n"
                      << "  --seed <n>                Random seed\n"
                      << "  --output-vote             Enable output-spike voting\n"
                      << "  --no-output-vote          Disable output-spike voting\n"
                      << "  --no-output-competition   Disable output winner masking\n"
                      << "  --no-output-teach         Disable supervised output teaching\n"
                      << "  --keep-l5-history         Keep class patterns across passes\n"
                      << "  --no-l5-inter-column-inhibit Disable cross-column L5 inhibition\n"
                      << "  --l5-inter-column-inhibit <v>  Set cross-column inhibition strength\n"
                      << "  --l5-inter-column-min-overlap <v> Min RF overlap for inhibition\n"
                      << "  --l5-inter-column-winner-scale <v> Winner inhibition scaling [0..1]\n"
                      << "  --l5-inter-column-max <v>   Max inhibition injected per column\n"
                      << std::endl;
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    try {
        snnfw::experiment::ExperimentRunner runner(config);
        double accuracy = runner.run();
        return (accuracy > 0.0) ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
