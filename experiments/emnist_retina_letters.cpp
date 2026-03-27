#include "snnfw/EMNISTLoader.h"
#include "snnfw/Logger.h"
#include "snnfw/adapters/RetinaAdapter.h"
#include "snnfw/classification/ClassificationStrategy.h"
#include "snnfw/declarative/NativeJSONParser.h"
#include "snnfw/declarative/SONATAParser.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using snnfw::EMNISTLoader;
using snnfw::adapters::BaseAdapter;
using snnfw::adapters::RetinaAdapter;
using snnfw::classification::ClassificationStrategy;
using snnfw::declarative::AdapterConfigIR;
using snnfw::declarative::ClassificationConfigIR;
using snnfw::declarative::ColumnIR;
using snnfw::declarative::ColumnTemplateIR;
using snnfw::declarative::NativeJSONParser;
using snnfw::declarative::NetworkIR;
using snnfw::declarative::SONATAParser;

struct Config {
    std::string configPath;
    std::string trainImagesPath;
    std::string trainLabelsPath;
    std::string testImagesPath;
    std::string testLabelsPath;
    int examplesPerClass = 200;
    int testLimit = 1000;
    unsigned int seed = 42;
    int gridSize = 8;
    std::vector<int> gridSizes{8};
    int numOrientations = 8;
    double edgeThreshold = 0.15;
    double temporalWindowMs = 200.0;
    std::string edgeOperator = "sobel";
    std::string encodingStrategy = "rate";
    std::string classifier = "majority";
    std::string activationMode = "binary";
    std::string hierarchicalGroups;
    std::string hierarchicalCoarseStrategy = "majority";
    std::string hierarchicalFineStrategy = "majority";
    int knnK = 5;
    int hierarchicalCoarseK = 0;
    int hierarchicalFineK = 0;
    double classifierExponent = 1.0;
    bool useFeatures = false;
    std::vector<BaseAdapter::Config> retinaConfigs;
    std::vector<std::vector<int>> focusGroups;
    int focusLimitPerLabel = 0;
    bool focusOnly = false;
    bool bilateralFusion = false;
    std::string stage1Classifier = "majority";
    std::string fusionClassifier = "majority";
    int stage1K = 0;
    int fusionK = 0;
    double stage1Exponent = 1.0;
    double fusionExponent = 1.0;
    int fusionHoldoutPerClass = 0;
    std::string fusionFeatureMode = "interaction";
    double corpusVoteGain = 0.35;
    double corpusMarginGain = 0.50;
    double corpusCentroidGain = 0.0;
    double corpusNeighborGain = 0.0;
    int onlineCorrectionRepeats = 0;
    int onlineExemplarBudgetPerClass = 16;
    double onlineCentroidLr = 0.25;
    double onlinePositiveRewardGain = 0.35;
    double onlineNegativeRewardGain = 1.0;
    int onlineReplayQueueCapacity = 256;
    int onlineReplayDelaySteps = 0;
    int onlineReplayPauseInterval = 1;
    int onlineReplayBatchSize = 1;
    double onlineReplayUncertaintyThreshold = 0.35;
    double onlineEligibilityTraceDecay = 1.0;
    double onlineContextUncertaintyGain = 0.75;
    double onlineContextDisagreementGain = 0.50;
    bool hierarchicalRetinaLayout = false;
    std::vector<std::string> declaredHemisphereOrder;
    std::string fusionPath;
};

struct HemisphereRuntime {
    std::string name;
    std::vector<std::unique_ptr<RetinaAdapter>> retinas;
    std::vector<ClassificationStrategy::LabeledPattern> trainingPatterns;
    std::vector<size_t> trainingSourceIndices;
    std::unique_ptr<ClassificationStrategy> classifier;
    double overallWeight = 1.0;
    std::array<double, 26> classWeights{};
    std::array<double, 26> predictionWeights{};
    std::array<std::vector<double>, 26> classCentroids{};
    std::array<std::vector<size_t>, 26> onlinePatternIndices{};
};

struct FusionRuntime {
    std::vector<ClassificationStrategy::LabeledPattern> trainingPatterns;
    std::array<std::vector<size_t>, 26> onlinePatternIndices{};
};

struct LabelTrace {
    int label = -1;
    double score = 0.0;
};

struct HemisphereDecisionTrace {
    std::vector<double> pattern;
    std::vector<double> confidence;
    std::vector<LabelTrace> topHypotheses;
    int predicted = -1;
    double margin = 0.0;
};

struct BilateralDecisionTrace {
    std::vector<HemisphereDecisionTrace> hemisphereTraces;
    std::vector<double> combinedConfidence;
    std::vector<double> fusionPattern;
    std::vector<LabelTrace> topHypotheses;
    int predicted = -1;
};

struct DecisionContext {
    double uncertainty = 0.0;
    double disagreement = 0.0;
    double plasticity = 1.0;
    double replayPriority = 0.0;
};

struct OnlineSampleRecord {
    size_t imageIndex = 0;
    int truth = -1;
    int initialPredicted = -1;
    int finalPredicted = -1;
    bool correctionSucceeded = false;
};

struct ReplayItem {
    size_t recordIndex = 0;
    int remainingReplays = 0;
    double priority = 0.0;
    double eligibilityScale = 1.0;
    size_t traceAgeSteps = 0;
    size_t readyStep = 0;
    size_t sequence = 0;
};

struct RetinaLayerBinding {
    std::string hemisphere;
    std::string lobe;
    std::string region;
    std::string nucleus;
    std::string column;
    std::string layer;
    std::string path;
};

std::vector<double> buildFusionPattern(std::vector<HemisphereRuntime>& hemispheres,
                                       const EMNISTLoader::Image& image,
                                       const Config& config);

double cosineSimilarity(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size() || a.empty()) {
        return 0.0;
    }

    double dot = 0.0;
    double normA = 0.0;
    double normB = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    if (normA <= 0.0 || normB <= 0.0) {
        return 0.0;
    }
    return dot / (std::sqrt(normA) * std::sqrt(normB));
}

char classToChar(int label) {
    return static_cast<char>('A' + label);
}

std::vector<int> parseCsvInts(const std::string& csv) {
    std::vector<int> values;
    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            values.push_back(std::stoi(token));
        }
    }
    return values;
}

int parseLabelToken(std::string token) {
    token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
    if (token.empty()) {
        throw std::runtime_error("Empty label token in focus group definition");
    }

    if (token.size() == 1 && std::isalpha(static_cast<unsigned char>(token[0]))) {
        const char upper = static_cast<char>(std::toupper(static_cast<unsigned char>(token[0])));
        if (upper < 'A' || upper > 'Z') {
            throw std::runtime_error("Invalid class label token: " + token);
        }
        return upper - 'A';
    }

    const int numeric = std::stoi(token);
    if (numeric >= 0 && numeric < 26) {
        return numeric;
    }
    if (numeric >= 1 && numeric <= 26) {
        return numeric - 1;
    }
    throw std::runtime_error("Label token out of range: " + token);
}

std::vector<std::vector<int>> parseLabelGroups(const std::string& spec) {
    std::vector<std::vector<int>> groups;
    std::stringstream outer(spec);
    std::string groupToken;
    while (std::getline(outer, groupToken, ';')) {
        if (groupToken.empty()) {
            continue;
        }
        std::stringstream inner(groupToken);
        std::string labelToken;
        std::vector<int> group;
        while (std::getline(inner, labelToken, ',')) {
            if (!labelToken.empty()) {
                const int label = parseLabelToken(labelToken);
                if (std::find(group.begin(), group.end(), label) == group.end()) {
                    group.push_back(label);
                }
            }
        }
        if (!group.empty()) {
            groups.push_back(std::move(group));
        }
    }
    return groups;
}

std::vector<int> flattenLabelGroups(const std::vector<std::vector<int>>& groups) {
    std::vector<int> labels;
    for (const auto& group : groups) {
        for (int label : group) {
            if (std::find(labels.begin(), labels.end(), label) == labels.end()) {
                labels.push_back(label);
            }
        }
    }
    std::sort(labels.begin(), labels.end());
    return labels;
}

std::string labelGroupToString(const std::vector<int>& group) {
    std::ostringstream oss;
    for (size_t i = 0; i < group.size(); ++i) {
        if (i > 0) {
            oss << "/";
        }
        oss << classToChar(group[i]);
    }
    return oss.str();
}

void normalizeL2(std::vector<double>& values) {
    double norm = 0.0;
    for (double value : values) {
        norm += value * value;
    }
    if (norm <= 0.0) {
        return;
    }
    norm = std::sqrt(norm);
    for (double& value : values) {
        value /= norm;
    }
}

void normalizeSum(std::vector<double>& values) {
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    if (sum <= 0.0) {
        return;
    }
    for (double& value : values) {
        value /= sum;
    }
}

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::string trim(std::string value) {
    const auto isSpace = [](unsigned char c) { return std::isspace(c) != 0; };
    value.erase(value.begin(),
                std::find_if(value.begin(), value.end(),
                             [&](unsigned char c) { return !isSpace(c); }));
    value.erase(std::find_if(value.rbegin(), value.rend(),
                             [&](unsigned char c) { return !isSpace(c); }).base(),
                value.end());
    return value;
}

std::string normalizeHierarchyPath(const std::string& path) {
    std::stringstream ss(path);
    std::string token;
    std::vector<std::string> parts;
    while (std::getline(ss, token, '/')) {
        token = trim(token);
        if (!token.empty()) {
            parts.push_back(token);
        }
    }

    std::ostringstream oss;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (i > 0) {
            oss << "/";
        }
        oss << parts[i];
    }
    return oss.str();
}

std::vector<ColumnIR> expandColumnTemplateForPaths(const ColumnTemplateIR& tmpl) {
    std::vector<ColumnIR> result;
    for (double orientation : tmpl.orientations) {
        for (double frequency : tmpl.frequencies) {
            ColumnIR column;
            std::string name = tmpl.namingPattern;

            auto pos = name.find("{orientation}");
            if (pos != std::string::npos) {
                std::ostringstream oss;
                if (orientation == static_cast<int>(orientation)) {
                    oss << static_cast<int>(orientation);
                } else {
                    oss << orientation;
                }
                name.replace(pos, 13, oss.str());
            }

            pos = name.find("{frequency}");
            if (pos != std::string::npos) {
                std::ostringstream oss;
                if (frequency == static_cast<int>(frequency)) {
                    oss << static_cast<int>(frequency);
                } else {
                    oss << frequency;
                }
                name.replace(pos, 11, oss.str());
            }

            column.name = name;
            column.properties["orientation"] = orientation;
            column.properties["spatial_frequency"] = frequency;
            column.layers = tmpl.layers;
            result.push_back(std::move(column));
        }
    }
    return result;
}

std::unordered_map<std::string, RetinaLayerBinding> buildRetinaLayerIndex(const NetworkIR& ir) {
    std::unordered_map<std::string, RetinaLayerBinding> index;

    for (const auto& hemisphere : ir.brain.hemispheres) {
        for (const auto& lobe : hemisphere.lobes) {
            for (const auto& region : lobe.regions) {
                for (const auto& nucleus : region.nuclei) {
                    std::vector<ColumnIR> columns = nucleus.columns;
                    if (nucleus.columnTemplate.has_value()) {
                        auto expanded = expandColumnTemplateForPaths(nucleus.columnTemplate.value());
                        columns.insert(columns.end(), expanded.begin(), expanded.end());
                    }

                    for (const auto& column : columns) {
                        for (const auto& layer : column.layers) {
                            RetinaLayerBinding binding;
                            binding.hemisphere = hemisphere.name;
                            binding.lobe = lobe.name;
                            binding.region = region.name;
                            binding.nucleus = nucleus.name;
                            binding.column = column.name;
                            binding.layer = layer.name;
                            binding.path = normalizeHierarchyPath(
                                hemisphere.name + "/" + lobe.name + "/" + region.name + "/" +
                                nucleus.name + "/" + column.name + "/" + layer.name);
                            index[binding.path] = binding;
                        }
                    }
                }
            }
        }
    }

    return index;
}

BaseAdapter::Config makeDefaultRetinaConfig(const Config& config, int gridSize, size_t index) {
    BaseAdapter::Config retinaConfig;
    retinaConfig.name = "emnist_retina_" + std::to_string(index);
    retinaConfig.type = "retina";
    retinaConfig.temporalWindow = config.temporalWindowMs;
    retinaConfig.intParams["grid_size"] = gridSize;
    retinaConfig.intParams["num_orientations"] = config.numOrientations;
    retinaConfig.doubleParams["edge_threshold"] = config.edgeThreshold;
    retinaConfig.doubleParams["neuron_window_size"] = config.temporalWindowMs;
    retinaConfig.doubleParams["neuron_threshold"] = 0.7;
    retinaConfig.intParams["neuron_max_patterns"] = 100;
    retinaConfig.stringParams["edge_operator"] = config.edgeOperator;
    retinaConfig.stringParams["encoding_strategy"] = config.encodingStrategy;
    retinaConfig.stringParams["activation_mode"] = config.activationMode;
    return retinaConfig;
}

void applyClassificationConfig(const ClassificationConfigIR& irConfig, Config& config) {
    if (!irConfig.type.empty()) {
        config.classifier = irConfig.type;
    }
    config.knnK = irConfig.k;
    config.classifierExponent = irConfig.distanceExponent;
    config.stage1Classifier = config.classifier;
    config.fusionClassifier = config.classifier;
    config.stage1K = config.knnK;
    config.fusionK = config.knnK;
    config.stage1Exponent = config.classifierExponent;
    config.fusionExponent = config.classifierExponent;

    const auto groupIt = irConfig.stringParams.find("group_definitions");
    if (groupIt != irConfig.stringParams.end()) {
        config.hierarchicalGroups = groupIt->second;
    }
    const auto coarseStrategyIt = irConfig.stringParams.find("coarse_strategy");
    if (coarseStrategyIt != irConfig.stringParams.end()) {
        config.hierarchicalCoarseStrategy = coarseStrategyIt->second;
    }
    const auto fineStrategyIt = irConfig.stringParams.find("fine_strategy");
    if (fineStrategyIt != irConfig.stringParams.end()) {
        config.hierarchicalFineStrategy = fineStrategyIt->second;
    }
    const auto coarseKIt = irConfig.intParams.find("coarse_k");
    if (coarseKIt != irConfig.intParams.end()) {
        config.hierarchicalCoarseK = coarseKIt->second;
    }
    const auto fineKIt = irConfig.intParams.find("fine_k");
    if (fineKIt != irConfig.intParams.end()) {
        config.hierarchicalFineK = fineKIt->second;
    }
    const auto fusionModeIt = irConfig.stringParams.find("fusion_mode");
    if (fusionModeIt != irConfig.stringParams.end()) {
        std::string value = fusionModeIt->second;
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        config.bilateralFusion = (value == "bilateral");
    }
    const auto stage1StrategyIt = irConfig.stringParams.find("stage1_strategy");
    if (stage1StrategyIt != irConfig.stringParams.end()) {
        config.stage1Classifier = stage1StrategyIt->second;
    }
    const auto fusionStrategyIt = irConfig.stringParams.find("fusion_strategy");
    if (fusionStrategyIt != irConfig.stringParams.end()) {
        config.fusionClassifier = fusionStrategyIt->second;
    }
    const auto fusionFeatureModeIt = irConfig.stringParams.find("fusion_feature_mode");
    if (fusionFeatureModeIt != irConfig.stringParams.end()) {
        config.fusionFeatureMode = fusionFeatureModeIt->second;
    }
    const auto fusionPathIt = irConfig.stringParams.find("fusion_path");
    if (fusionPathIt != irConfig.stringParams.end()) {
        config.fusionPath = normalizeHierarchyPath(fusionPathIt->second);
    }
    const auto stage1KIt = irConfig.intParams.find("stage1_k");
    if (stage1KIt != irConfig.intParams.end()) {
        config.stage1K = stage1KIt->second;
    }
    const auto fusionKIt = irConfig.intParams.find("fusion_k");
    if (fusionKIt != irConfig.intParams.end()) {
        config.fusionK = fusionKIt->second;
    }
    const auto fusionHoldoutIt = irConfig.intParams.find("fusion_holdout_per_class");
    if (fusionHoldoutIt != irConfig.intParams.end()) {
        config.fusionHoldoutPerClass = fusionHoldoutIt->second;
    }
    const auto stage1ExpIt = irConfig.doubleParams.find("stage1_distance_exponent");
    if (stage1ExpIt != irConfig.doubleParams.end()) {
        config.stage1Exponent = stage1ExpIt->second;
    }
    const auto fusionExpIt = irConfig.doubleParams.find("fusion_distance_exponent");
    if (fusionExpIt != irConfig.doubleParams.end()) {
        config.fusionExponent = fusionExpIt->second;
    }
    const auto corpusVoteGainIt = irConfig.doubleParams.find("corpus_vote_gain");
    if (corpusVoteGainIt != irConfig.doubleParams.end()) {
        config.corpusVoteGain = corpusVoteGainIt->second;
    }
    const auto corpusMarginGainIt = irConfig.doubleParams.find("corpus_margin_gain");
    if (corpusMarginGainIt != irConfig.doubleParams.end()) {
        config.corpusMarginGain = corpusMarginGainIt->second;
    }
    const auto corpusCentroidGainIt = irConfig.doubleParams.find("corpus_centroid_gain");
    if (corpusCentroidGainIt != irConfig.doubleParams.end()) {
        config.corpusCentroidGain = corpusCentroidGainIt->second;
    }
    const auto corpusNeighborGainIt = irConfig.doubleParams.find("corpus_neighbor_gain");
    if (corpusNeighborGainIt != irConfig.doubleParams.end()) {
        config.corpusNeighborGain = corpusNeighborGainIt->second;
    }
    const auto onlineRepeatsIt = irConfig.intParams.find("online_correction_repeats");
    if (onlineRepeatsIt != irConfig.intParams.end()) {
        config.onlineCorrectionRepeats = onlineRepeatsIt->second;
    }
    const auto onlineBudgetIt = irConfig.intParams.find("online_exemplar_budget_per_class");
    if (onlineBudgetIt != irConfig.intParams.end()) {
        config.onlineExemplarBudgetPerClass = onlineBudgetIt->second;
    }
    const auto onlineCentroidLrIt = irConfig.doubleParams.find("online_centroid_lr");
    if (onlineCentroidLrIt != irConfig.doubleParams.end()) {
        config.onlineCentroidLr = onlineCentroidLrIt->second;
    }
    const auto onlinePositiveGainIt = irConfig.doubleParams.find("online_positive_reward_gain");
    if (onlinePositiveGainIt != irConfig.doubleParams.end()) {
        config.onlinePositiveRewardGain = onlinePositiveGainIt->second;
    }
    const auto onlineNegativeGainIt = irConfig.doubleParams.find("online_negative_reward_gain");
    if (onlineNegativeGainIt != irConfig.doubleParams.end()) {
        config.onlineNegativeRewardGain = onlineNegativeGainIt->second;
    }
    const auto onlineReplayCapacityIt = irConfig.intParams.find("online_replay_queue_capacity");
    if (onlineReplayCapacityIt != irConfig.intParams.end()) {
        config.onlineReplayQueueCapacity = onlineReplayCapacityIt->second;
    }
    const auto onlineReplayDelayIt = irConfig.intParams.find("online_replay_delay_steps");
    if (onlineReplayDelayIt != irConfig.intParams.end()) {
        config.onlineReplayDelaySteps = onlineReplayDelayIt->second;
    }
    const auto onlineReplayPauseIt = irConfig.intParams.find("online_replay_pause_interval");
    if (onlineReplayPauseIt != irConfig.intParams.end()) {
        config.onlineReplayPauseInterval = onlineReplayPauseIt->second;
    }
    const auto onlineReplayBatchIt = irConfig.intParams.find("online_replay_batch_size");
    if (onlineReplayBatchIt != irConfig.intParams.end()) {
        config.onlineReplayBatchSize = onlineReplayBatchIt->second;
    }
    const auto onlineReplayThresholdIt =
        irConfig.doubleParams.find("online_replay_uncertainty_threshold");
    if (onlineReplayThresholdIt != irConfig.doubleParams.end()) {
        config.onlineReplayUncertaintyThreshold = onlineReplayThresholdIt->second;
    }
    const auto onlineEligibilityDecayIt =
        irConfig.doubleParams.find("online_eligibility_trace_decay");
    if (onlineEligibilityDecayIt != irConfig.doubleParams.end()) {
        config.onlineEligibilityTraceDecay = onlineEligibilityDecayIt->second;
    }
    const auto onlineContextUncertaintyGainIt =
        irConfig.doubleParams.find("online_context_uncertainty_gain");
    if (onlineContextUncertaintyGainIt != irConfig.doubleParams.end()) {
        config.onlineContextUncertaintyGain = onlineContextUncertaintyGainIt->second;
    }
    const auto onlineContextDisagreementGainIt =
        irConfig.doubleParams.find("online_context_disagreement_gain");
    if (onlineContextDisagreementGainIt != irConfig.doubleParams.end()) {
        config.onlineContextDisagreementGain = onlineContextDisagreementGainIt->second;
    }
}

BaseAdapter::Config makeRetinaAdapterConfig(const AdapterConfigIR& adapter,
                                            const Config& defaults,
                                            size_t index) {
    BaseAdapter::Config retinaConfig = makeDefaultRetinaConfig(defaults, defaults.gridSize, index);
    retinaConfig.name = adapter.name.empty()
        ? ("emnist_retina_" + std::to_string(index))
        : adapter.name;
    retinaConfig.type = "retina";
    retinaConfig.temporalWindow = adapter.temporalWindowMs > 0.0
        ? adapter.temporalWindowMs
        : defaults.temporalWindowMs;

    for (const auto& [key, value] : adapter.doubleParams) {
        retinaConfig.doubleParams[key] = value;
    }
    for (const auto& [key, value] : adapter.intParams) {
        retinaConfig.intParams[key] = value;
    }
    for (const auto& [key, value] : adapter.stringParams) {
        retinaConfig.stringParams[key] = value;
    }

    if (retinaConfig.intParams.find("grid_size") == retinaConfig.intParams.end()) {
        retinaConfig.intParams["grid_size"] = defaults.gridSize;
    }
    if (retinaConfig.intParams.find("num_orientations") == retinaConfig.intParams.end()) {
        retinaConfig.intParams["num_orientations"] = defaults.numOrientations;
    }
    if (retinaConfig.doubleParams.find("edge_threshold") == retinaConfig.doubleParams.end()) {
        retinaConfig.doubleParams["edge_threshold"] = defaults.edgeThreshold;
    }
    if (retinaConfig.doubleParams.find("neuron_window_size") == retinaConfig.doubleParams.end()) {
        retinaConfig.doubleParams["neuron_window_size"] = retinaConfig.temporalWindow;
    }
    if (retinaConfig.doubleParams.find("neuron_threshold") == retinaConfig.doubleParams.end()) {
        retinaConfig.doubleParams["neuron_threshold"] = 0.7;
    }
    if (retinaConfig.intParams.find("neuron_max_patterns") == retinaConfig.intParams.end()) {
        retinaConfig.intParams["neuron_max_patterns"] = 100;
    }
    if (retinaConfig.stringParams.find("edge_operator") == retinaConfig.stringParams.end()) {
        retinaConfig.stringParams["edge_operator"] = defaults.edgeOperator;
    }
    if (retinaConfig.stringParams.find("encoding_strategy") == retinaConfig.stringParams.end()) {
        retinaConfig.stringParams["encoding_strategy"] = defaults.encodingStrategy;
    }
    if (retinaConfig.stringParams.find("activation_mode") == retinaConfig.stringParams.end()) {
        retinaConfig.stringParams["activation_mode"] = defaults.activationMode;
    }

    return retinaConfig;
}

NetworkIR parseDeclarativeConfig(const std::string& configPath) {
    NativeJSONParser nativeParser;
    if (nativeParser.canParse(configPath)) {
        return nativeParser.parse(configPath);
    }

    SONATAParser sonataParser;
    if (sonataParser.canParse(configPath)) {
        return sonataParser.parse(configPath);
    }

    throw std::runtime_error("Unsupported declarative config format: " + configPath);
}

void loadDeclarativeConfig(Config& config, const std::string& configPath) {
    const NetworkIR ir = parseDeclarativeConfig(configPath);

    if (!ir.classification.type.empty()) {
        applyClassificationConfig(ir.classification, config);
    }

    const auto layerIndex = buildRetinaLayerIndex(ir);
    const bool wantsHierarchicalLayout =
        !config.fusionPath.empty() ||
        std::any_of(ir.adapters.begin(), ir.adapters.end(), [](const AdapterConfigIR& adapter) {
            auto it = adapter.stringParams.find("attach_path");
            return it != adapter.stringParams.end() && !trim(it->second).empty();
        });

    if (wantsHierarchicalLayout) {
        if (layerIndex.empty()) {
            throw std::runtime_error(
                "Hierarchical Retina declarative layout requested, but the config does not "
                "define any brain layer paths");
        }
        config.hierarchicalRetinaLayout = true;
        config.declaredHemisphereOrder.clear();
        for (const auto& hemisphere : ir.brain.hemispheres) {
            config.declaredHemisphereOrder.push_back(hemisphere.name);
        }
        if (!config.fusionPath.empty() &&
            config.fusionPath != "OutputLayer" &&
            layerIndex.find(config.fusionPath) == layerIndex.end()) {
            throw std::runtime_error(
                "fusion_path '" + config.fusionPath + "' does not resolve to a declared brain layer");
        }
    }

    std::vector<BaseAdapter::Config> retinaConfigs;
    retinaConfigs.reserve(ir.adapters.size());
    for (const auto& adapter : ir.adapters) {
        if (adapter.type != "retina") {
            continue;
        }
        auto retinaConfig = makeRetinaAdapterConfig(adapter, config, retinaConfigs.size());
        if (wantsHierarchicalLayout) {
            const std::string attachPath =
                normalizeHierarchyPath(retinaConfig.getStringParam("attach_path", ""));
            if (attachPath.empty()) {
                throw std::runtime_error(
                    "Retina adapter '" + retinaConfig.name +
                    "' is missing string_params.attach_path while hierarchical Retina layout "
                    "is enabled");
            }
            const auto bindingIt = layerIndex.find(attachPath);
            if (bindingIt == layerIndex.end()) {
                throw std::runtime_error(
                    "Retina adapter '" + retinaConfig.name +
                    "' attach_path '" + attachPath + "' does not resolve to a declared brain layer");
            }
            retinaConfig.stringParams["attach_path"] = attachPath;
            retinaConfig.stringParams["hemisphere"] = bindingIt->second.hemisphere;
            retinaConfig.stringParams["layer_name"] = bindingIt->second.layer;
        }
        retinaConfigs.push_back(std::move(retinaConfig));
    }

    if (!retinaConfigs.empty()) {
        config.retinaConfigs = std::move(retinaConfigs);
        config.gridSizes.clear();
        for (const auto& retinaConfig : config.retinaConfigs) {
            config.gridSizes.push_back(retinaConfig.getIntParam("grid_size", config.gridSize));
        }
        config.gridSize = config.gridSizes.front();
        config.numOrientations =
            config.retinaConfigs.front().getIntParam("num_orientations", config.numOrientations);
        config.edgeThreshold =
            config.retinaConfigs.front().getDoubleParam("edge_threshold", config.edgeThreshold);
        config.temporalWindowMs = config.retinaConfigs.front().temporalWindow;
        config.edgeOperator =
            config.retinaConfigs.front().getStringParam("edge_operator", config.edgeOperator);
        config.encodingStrategy =
            config.retinaConfigs.front().getStringParam("encoding_strategy", config.encodingStrategy);
        config.activationMode =
            config.retinaConfigs.front().getStringParam("activation_mode", config.activationMode);
    }
}

std::vector<BaseAdapter::Config> buildRetinaConfigs(const Config& config) {
    if (!config.retinaConfigs.empty()) {
        return config.retinaConfigs;
    }

    std::vector<BaseAdapter::Config> retinaConfigs;
    retinaConfigs.reserve(config.gridSizes.size());
    for (size_t i = 0; i < config.gridSizes.size(); ++i) {
        retinaConfigs.push_back(makeDefaultRetinaConfig(config, config.gridSizes[i], i));
    }
    return retinaConfigs;
}

std::vector<double> extractPattern(std::vector<std::unique_ptr<RetinaAdapter>>& retinas,
                                   const EMNISTLoader::Image& image,
                                   bool useFeatures,
                                   bool learnPatterns);

std::vector<std::pair<std::string, std::vector<BaseAdapter::Config>>>
groupRetinaConfigsByHemisphere(const std::vector<BaseAdapter::Config>& retinaConfigs,
                              const Config& config) {
    std::vector<std::pair<std::string, std::vector<BaseAdapter::Config>>> groups;
    if (config.hierarchicalRetinaLayout) {
        for (const auto& hemisphereName : config.declaredHemisphereOrder) {
            groups.push_back({hemisphereName, {}});
        }
    }
    for (const auto& retinaConfig : retinaConfigs) {
        const std::string hemisphere =
            retinaConfig.getStringParam("hemisphere", "default");
        auto it = std::find_if(groups.begin(), groups.end(),
                               [&](const auto& group) { return group.first == hemisphere; });
        if (it == groups.end()) {
            groups.push_back({hemisphere, {retinaConfig}});
        } else {
            it->second.push_back(retinaConfig);
        }
    }
    if (config.hierarchicalRetinaLayout) {
        for (auto& group : groups) {
            std::sort(group.second.begin(), group.second.end(),
                      [](const BaseAdapter::Config& lhs, const BaseAdapter::Config& rhs) {
                          return lhs.getStringParam("attach_path", lhs.name) <
                                 rhs.getStringParam("attach_path", rhs.name);
                      });
        }
        groups.erase(std::remove_if(groups.begin(), groups.end(),
                                    [](const auto& group) { return group.second.empty(); }),
                     groups.end());
    }
    return groups;
}

std::unique_ptr<ClassificationStrategy> makeClassifierStrategy(const std::string& type,
                                                               int k,
                                                               double exponent,
                                                               const Config& config) {
    ClassificationStrategy::Config clsConfig;
    clsConfig.name = type;
    clsConfig.k = k;
    clsConfig.numClasses = 26;
    clsConfig.distanceExponent = exponent;
    clsConfig.stringParams["group_definitions"] = config.hierarchicalGroups;
    clsConfig.stringParams["coarse_strategy"] = config.hierarchicalCoarseStrategy;
    clsConfig.stringParams["fine_strategy"] = config.hierarchicalFineStrategy;
    if (config.hierarchicalCoarseK > 0) {
        clsConfig.intParams["coarse_k"] = config.hierarchicalCoarseK;
    }
    if (config.hierarchicalFineK > 0) {
        clsConfig.intParams["fine_k"] = config.hierarchicalFineK;
    }
    return snnfw::classification::ClassificationStrategyFactory::create(type, clsConfig);
}

std::vector<std::unique_ptr<RetinaAdapter>>
createRetinaAdapters(const std::vector<BaseAdapter::Config>& retinaConfigs) {
    std::vector<std::unique_ptr<RetinaAdapter>> retinas;
    retinas.reserve(retinaConfigs.size());
    for (const auto& retinaConfig : retinaConfigs) {
        auto retina = std::make_unique<RetinaAdapter>(retinaConfig);
        if (!retina->initialize()) {
            throw std::runtime_error("Failed to initialize RetinaAdapter '" + retinaConfig.name + "'");
        }
        retinas.push_back(std::move(retina));
    }
    return retinas;
}

struct TrainingSplit {
    std::vector<size_t> stage1Indices;
    std::vector<size_t> fusionIndices;
};

struct EvaluationResult {
    std::array<std::array<int, 26>, 26> initialConfusion{};
    std::array<std::array<int, 26>, 26> confusion{};
    int tested = 0;
    int correct = 0;
    int initialCorrect = 0;
    int correctionEvents = 0;
    int correctionSuccesses = 0;
    int correctionReplays = 0;
    int replayDelaySamples = 0;
    double replayDelaySteps = 0.0;
    double replayEligibilityScaleSum = 0.0;
    double seconds = 0.0;
};

TrainingSplit splitTrainingIndices(const std::array<std::vector<size_t>, 26>& indicesByLabel,
                                   int examplesPerClass,
                                   int fusionHoldoutPerClass,
                                   unsigned int seed,
                                   bool retainFusionInStage1 = false) {
    TrainingSplit split;
    std::mt19937 rng(seed);
    for (size_t label = 0; label < indicesByLabel.size(); ++label) {
        auto shuffled = indicesByLabel[label];
        std::shuffle(shuffled.begin(), shuffled.end(), rng);
        const int keep = examplesPerClass > 0
            ? std::min<int>(examplesPerClass, static_cast<int>(shuffled.size()))
            : static_cast<int>(shuffled.size());
        if (keep <= 0) {
            continue;
        }

        int holdout = fusionHoldoutPerClass;
        if (holdout <= 0) {
            holdout = std::min(100, std::max(20, keep / 10));
        }
        holdout = std::min(holdout, std::max(1, keep / 4));
        if (holdout >= keep) {
            holdout = std::max(0, keep - 1);
        }

        const int stage1Count = retainFusionInStage1 ? keep : (keep - holdout);
        const int fusionStart = keep - holdout;
        split.stage1Indices.insert(split.stage1Indices.end(),
                                   shuffled.begin(),
                                   shuffled.begin() + stage1Count);
        split.fusionIndices.insert(split.fusionIndices.end(),
                                   shuffled.begin() + fusionStart,
                                   shuffled.begin() + keep);
    }

    std::shuffle(split.stage1Indices.begin(), split.stage1Indices.end(), rng);
    std::shuffle(split.fusionIndices.begin(), split.fusionIndices.end(), rng);
    return split;
}

std::vector<ClassificationStrategy::LabeledPattern> buildSupportPatterns(
    const HemisphereRuntime& hemisphere,
    const std::vector<size_t>& excludedSourceIndices) {
    if (excludedSourceIndices.empty()) {
        return hemisphere.trainingPatterns;
    }

    const std::unordered_set<size_t> excluded(excludedSourceIndices.begin(),
                                              excludedSourceIndices.end());
    std::vector<ClassificationStrategy::LabeledPattern> supportPatterns;
    supportPatterns.reserve(hemisphere.trainingPatterns.size());
    for (size_t i = 0; i < hemisphere.trainingPatterns.size(); ++i) {
        if (i < hemisphere.trainingSourceIndices.size() &&
            excluded.find(hemisphere.trainingSourceIndices[i]) != excluded.end()) {
            continue;
        }
        supportPatterns.push_back(hemisphere.trainingPatterns[i]);
    }
    return supportPatterns;
}

bool useOnlineCorrection(const Config& config) {
    return config.onlineCorrectionRepeats > 0;
}

int onlineTraceTopK(const Config& config) {
    return std::max(1, std::min(3, config.knnK > 0 ? config.knnK : 3));
}

int onlineReplayDelaySteps(const Config& config) {
    return std::max(0, config.onlineReplayDelaySteps);
}

int onlineReplayPauseInterval(const Config& config) {
    return std::max(1, config.onlineReplayPauseInterval);
}

int onlineReplayBatchSize(const Config& config) {
    return std::max(1, config.onlineReplayBatchSize);
}

double decayEligibilityTrace(double currentScale, size_t delaySteps, const Config& config) {
    const double decay = std::clamp(config.onlineEligibilityTraceDecay, 0.10, 1.0);
    const double delayed =
        currentScale * std::pow(decay, static_cast<double>(std::max<size_t>(1, delaySteps)));
    return std::clamp(delayed, 0.05, 1.0);
}

DecisionContext buildDecisionContext(const BilateralDecisionTrace& decision, const Config& config) {
    DecisionContext context;
    if (!decision.topHypotheses.empty()) {
        const double best = decision.topHypotheses.front().score;
        const double second =
            decision.topHypotheses.size() > 1 ? decision.topHypotheses[1].score : 0.0;
        const double normalizedMargin =
            best > 1e-6 ? std::clamp((best - second) / best, 0.0, 1.0) : 0.0;
        context.uncertainty = 1.0 - normalizedMargin;
    }

    int validVotes = 0;
    int disagreements = 0;
    for (const auto& hemisphereTrace : decision.hemisphereTraces) {
        if (hemisphereTrace.predicted < 0) {
            continue;
        }
        ++validVotes;
        if (decision.predicted >= 0 && hemisphereTrace.predicted != decision.predicted) {
            ++disagreements;
        }
    }
    context.disagreement = validVotes > 0
        ? static_cast<double>(disagreements) / static_cast<double>(validVotes)
        : 0.0;

    context.plasticity = 1.0 +
        config.onlineContextUncertaintyGain * context.uncertainty +
        config.onlineContextDisagreementGain * context.disagreement;
    context.plasticity = std::clamp(context.plasticity, 0.5, 3.0);
    context.replayPriority = context.plasticity + context.uncertainty + 0.5 * context.disagreement;
    return context;
}

void enqueueReplayItem(std::vector<ReplayItem>& replayQueue,
                       ReplayItem item,
                       const Config& config) {
    if (item.remainingReplays <= 0 || config.onlineReplayQueueCapacity <= 0) {
        return;
    }

    replayQueue.push_back(item);
    if (static_cast<int>(replayQueue.size()) <= config.onlineReplayQueueCapacity) {
        return;
    }

    const auto worstIt = std::min_element(
        replayQueue.begin(), replayQueue.end(),
        [](const ReplayItem& lhs, const ReplayItem& rhs) {
            if (lhs.priority != rhs.priority) {
                return lhs.priority < rhs.priority;
            }
            return lhs.sequence > rhs.sequence;
        });
    if (worstIt != replayQueue.end()) {
        replayQueue.erase(worstIt);
    }
}

bool popReplayItem(std::vector<ReplayItem>& replayQueue,
                   ReplayItem& item,
                   size_t currentStep,
                   bool flushAll = false) {
    if (replayQueue.empty()) {
        return false;
    }

    auto bestIt = replayQueue.end();
    for (auto it = replayQueue.begin(); it != replayQueue.end(); ++it) {
        if (!flushAll && it->readyStep > currentStep) {
            continue;
        }
        if (bestIt == replayQueue.end()) {
            bestIt = it;
            continue;
        }
        if (it->priority > bestIt->priority ||
            (it->priority == bestIt->priority && it->sequence < bestIt->sequence)) {
            bestIt = it;
        }
    }
    if (bestIt == replayQueue.end()) {
        return false;
    }
    item = *bestIt;
    replayQueue.erase(bestIt);
    return true;
}

void recordReplayTiming(EvaluationResult& result, const ReplayItem& item, size_t currentStep) {
    result.replayDelaySamples++;
    result.replayDelaySteps += static_cast<double>(item.traceAgeSteps);
    result.replayEligibilityScaleSum += item.eligibilityScale;
}

ReplayItem makeReplayItem(size_t recordIndex,
                          int remainingReplays,
                          double priority,
                          size_t currentStep,
                          double eligibilityScale,
                          const Config& config,
                          size_t sequence) {
    const size_t delaySteps = static_cast<size_t>(onlineReplayDelaySteps(config));
    return ReplayItem{recordIndex,
                      remainingReplays,
                      priority,
                      delaySteps > 0 ? decayEligibilityTrace(eligibilityScale, delaySteps, config)
                                     : std::clamp(eligibilityScale, 0.05, 1.0),
                      delaySteps,
                      currentStep + delaySteps,
                      sequence};
}

void finalizeEvaluationFromRecords(EvaluationResult& result,
                                   const std::vector<OnlineSampleRecord>& records,
                                   double seconds) {
    result.initialConfusion = {};
    result.confusion = {};
    result.tested = 0;
    result.correct = 0;
    result.initialCorrect = 0;

    for (const auto& record : records) {
        if (record.truth < 0 || record.truth >= 26 ||
            record.initialPredicted < 0 || record.initialPredicted >= 26 ||
            record.finalPredicted < 0 || record.finalPredicted >= 26) {
            continue;
        }
        result.initialConfusion[static_cast<size_t>(record.truth)]
                               [static_cast<size_t>(record.initialPredicted)]++;
        result.confusion[static_cast<size_t>(record.truth)]
                        [static_cast<size_t>(record.finalPredicted)]++;
        result.initialCorrect += (record.initialPredicted == record.truth) ? 1 : 0;
        result.correct += (record.finalPredicted == record.truth) ? 1 : 0;
        result.tested++;
    }
    result.seconds = seconds;
}

void updateCentroid(std::vector<double>& centroid,
                    const std::vector<double>& pattern,
                    double learningRate) {
    if (pattern.empty()) {
        return;
    }
    if (centroid.empty()) {
        centroid = pattern;
        normalizeL2(centroid);
        return;
    }

    const size_t dim = std::min(centroid.size(), pattern.size());
    const double keep = std::clamp(1.0 - learningRate, 0.0, 1.0);
    const double learn = std::clamp(learningRate, 0.0, 1.0);
    for (size_t i = 0; i < dim; ++i) {
        centroid[i] = keep * centroid[i] + learn * pattern[i];
    }
    normalizeL2(centroid);
}

void insertOrReplaceOnlinePattern(
    std::vector<ClassificationStrategy::LabeledPattern>& patterns,
    std::vector<size_t>* sourceIndices,
    std::array<std::vector<size_t>, 26>& onlineIndices,
    const std::vector<double>& pattern,
    int label,
    int budgetPerClass) {
    if (label < 0 || label >= 26 || pattern.empty() || budgetPerClass <= 0) {
        return;
    }

    auto& labelOnlineIndices = onlineIndices[static_cast<size_t>(label)];
    if (static_cast<int>(labelOnlineIndices.size()) < budgetPerClass) {
        patterns.emplace_back(pattern, label);
        labelOnlineIndices.push_back(patterns.size() - 1);
        if (sourceIndices != nullptr) {
            sourceIndices->push_back(std::numeric_limits<size_t>::max());
        }
        return;
    }

    double worstSimilarity = std::numeric_limits<double>::infinity();
    size_t replaceIndex = labelOnlineIndices.front();
    for (size_t candidateIndex : labelOnlineIndices) {
        if (candidateIndex >= patterns.size()) {
            continue;
        }
        const double similarity = cosineSimilarity(pattern, patterns[candidateIndex].pattern);
        if (similarity < worstSimilarity) {
            worstSimilarity = similarity;
            replaceIndex = candidateIndex;
        }
    }
    if (replaceIndex < patterns.size()) {
        patterns[replaceIndex].pattern = pattern;
        patterns[replaceIndex].label = label;
    }
}

std::vector<LabelTrace> collectTopHypotheses(const std::vector<double>& confidence, int k) {
    std::vector<LabelTrace> traces;
    traces.reserve(confidence.size());
    for (size_t i = 0; i < confidence.size(); ++i) {
        traces.push_back({static_cast<int>(i), confidence[i]});
    }
    std::partial_sort(
        traces.begin(),
        traces.begin() + std::min<int>(k, static_cast<int>(traces.size())),
        traces.end(),
        [](const LabelTrace& lhs, const LabelTrace& rhs) { return lhs.score > rhs.score; });
    if (static_cast<int>(traces.size()) > k) {
        traces.resize(static_cast<size_t>(k));
    }
    return traces;
}

HemisphereDecisionTrace inferHemisphereDecision(HemisphereRuntime& hemisphere,
                                                const EMNISTLoader::Image& image,
                                                const Config& config) {
    HemisphereDecisionTrace trace;
    trace.pattern = extractPattern(hemisphere.retinas, image, config.useFeatures, false);
    trace.confidence = hemisphere.classifier->classifyWithConfidence(
        trace.pattern, hemisphere.trainingPatterns, cosineSimilarity);
    normalizeSum(trace.confidence);
    trace.topHypotheses = collectTopHypotheses(trace.confidence, onlineTraceTopK(config));
    if (!trace.topHypotheses.empty()) {
        trace.predicted = trace.topHypotheses.front().label;
        const double best = trace.topHypotheses.front().score;
        const double second = trace.topHypotheses.size() > 1 ? trace.topHypotheses[1].score : 0.0;
        trace.margin = std::max(0.0, best - second);
    }
    return trace;
}

std::vector<double> buildFusionPatternFromTraces(const std::vector<HemisphereDecisionTrace>& traces,
                                                 const Config& config) {
    std::vector<double> fusionPattern;
    std::vector<std::vector<double>> confidences;
    confidences.reserve(traces.size());
    for (const auto& trace : traces) {
        std::vector<double> confidence = trace.confidence;
        normalizeL2(confidence);
        confidences.push_back(std::move(confidence));
    }

    for (const auto& confidence : confidences) {
        fusionPattern.insert(fusionPattern.end(), confidence.begin(), confidence.end());
    }

    std::string mode = config.fusionFeatureMode;
    std::transform(mode.begin(), mode.end(), mode.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (mode == "interaction") {
        for (size_t a = 0; a < confidences.size(); ++a) {
            for (size_t b = a + 1; b < confidences.size(); ++b) {
                const size_t dim = std::min(confidences[a].size(), confidences[b].size());
                for (size_t i = 0; i < dim; ++i) {
                    fusionPattern.push_back(confidences[a][i] * confidences[b][i]);
                }
                for (size_t i = 0; i < dim; ++i) {
                    fusionPattern.push_back(std::abs(confidences[a][i] - confidences[b][i]));
                }
            }
        }
    }

    normalizeL2(fusionPattern);
    return fusionPattern;
}

std::vector<double> buildFusionPattern(std::vector<HemisphereRuntime>& hemispheres,
                                       const EMNISTLoader::Image& image,
                                       const Config& config) {
    std::vector<HemisphereDecisionTrace> traces;
    traces.reserve(hemispheres.size());
    for (auto& hemisphere : hemispheres) {
        traces.push_back(inferHemisphereDecision(hemisphere, image, config));
    }
    return buildFusionPatternFromTraces(traces, config);
}

bool useCorpusCallosumFusion(const Config& config) {
    return toLower(config.fusionClassifier) == "corpus_callosum";
}

void buildHemisphereClassCentroids(HemisphereRuntime& hemisphere) {
    hemisphere.classCentroids = {};

    std::array<int, 26> counts{};
    for (const auto& labeledPattern : hemisphere.trainingPatterns) {
        if (labeledPattern.label < 0 || labeledPattern.label >= 26 ||
            labeledPattern.pattern.empty()) {
            continue;
        }
        auto& centroid = hemisphere.classCentroids[static_cast<size_t>(labeledPattern.label)];
        if (centroid.empty()) {
            centroid.assign(labeledPattern.pattern.size(), 0.0);
        }
        const size_t dim = std::min(centroid.size(), labeledPattern.pattern.size());
        for (size_t i = 0; i < dim; ++i) {
            centroid[i] += labeledPattern.pattern[i];
        }
        counts[static_cast<size_t>(labeledPattern.label)]++;
    }

    for (size_t label = 0; label < hemisphere.classCentroids.size(); ++label) {
        auto& centroid = hemisphere.classCentroids[label];
        if (centroid.empty() || counts[label] <= 0) {
            continue;
        }
        const double invCount = 1.0 / static_cast<double>(counts[label]);
        for (double& value : centroid) {
            value *= invCount;
        }
        normalizeL2(centroid);
    }
}

std::vector<double> computeCentroidEvidence(const HemisphereRuntime& hemisphere,
                                            const std::vector<double>& pattern) {
    std::vector<double> centroidScores(26, 0.0);
    for (size_t label = 0; label < hemisphere.classCentroids.size(); ++label) {
        const auto& centroid = hemisphere.classCentroids[label];
        if (centroid.empty() || centroid.size() != pattern.size()) {
            continue;
        }
        centroidScores[label] = std::max(0.0, cosineSimilarity(pattern, centroid));
    }
    normalizeSum(centroidScores);
    return centroidScores;
}

std::vector<double> computeNeighborSignature(const HemisphereRuntime& hemisphere,
                                             const std::vector<double>& pattern,
                                             int k) {
    std::vector<std::pair<int, double>> neighbors;
    neighbors.reserve(hemisphere.trainingPatterns.size());

    for (size_t i = 0; i < hemisphere.trainingPatterns.size(); ++i) {
        const auto& labeledPattern = hemisphere.trainingPatterns[i];
        if (labeledPattern.label < 0 || labeledPattern.label >= 26) {
            continue;
        }
        neighbors.emplace_back(static_cast<int>(i),
                               cosineSimilarity(pattern, labeledPattern.pattern));
    }

    const int actualK = std::min<int>(std::max(1, k), static_cast<int>(neighbors.size()));
    if (actualK <= 0) {
        return std::vector<double>(26, 0.0);
    }

    std::partial_sort(neighbors.begin(), neighbors.begin() + actualK, neighbors.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });

    std::vector<double> signature(26, 0.0);
    for (int i = 0; i < actualK; ++i) {
        const auto& [index, similarity] = neighbors[static_cast<size_t>(i)];
        const int label = hemisphere.trainingPatterns[static_cast<size_t>(index)].label;
        signature[static_cast<size_t>(label)] += std::max(0.0, similarity);
    }
    normalizeSum(signature);
    return signature;
}

void calibrateCorpusCallosumWeights(std::vector<HemisphereRuntime>& hemispheres,
                                    const EMNISTLoader& loader,
                                    const std::vector<size_t>& indices,
                                    const Config& config) {
    struct CorpusStats {
        double overall = 1.0;
        std::array<double, 26> trueLabelAccuracy{};
        std::array<double, 26> predictionPrecision{};
    };

    std::vector<CorpusStats> rawStats(hemispheres.size());

    for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
        auto& hemisphere = hemispheres[hemisphereIndex];
        const auto supportPatterns = buildSupportPatterns(hemisphere, indices);
        std::array<int, 26> totals{};
        std::array<int, 26> corrects{};
        std::array<int, 26> predictedTotals{};
        std::array<int, 26> predictedCorrects{};
        int totalSamples = 0;
        int totalCorrect = 0;

        for (size_t index : indices) {
            const auto& image = loader.getImage(index);
            const int truth = static_cast<int>(image.label) - 1;
            if (truth < 0 || truth >= 26) {
                continue;
            }
            if (supportPatterns.empty()) {
                continue;
            }

            const auto pattern =
                extractPattern(hemisphere.retinas, image, config.useFeatures, false);
            const auto confidence = hemisphere.classifier->classifyWithConfidence(
                pattern, supportPatterns, cosineSimilarity);
            const int predicted = static_cast<int>(
                std::distance(confidence.begin(),
                              std::max_element(confidence.begin(), confidence.end())));
            totals[static_cast<size_t>(truth)]++;
            predictedTotals[static_cast<size_t>(predicted)]++;
            totalSamples++;
            if (predicted == truth) {
                corrects[static_cast<size_t>(truth)]++;
                predictedCorrects[static_cast<size_t>(predicted)]++;
                totalCorrect++;
            }
        }

        const double overall =
            (static_cast<double>(totalCorrect) + 1.0) /
            (static_cast<double>(std::max(1, totalSamples)) + 2.0);
        rawStats[hemisphereIndex].overall = overall;
        for (size_t label = 0; label < hemisphere.classWeights.size(); ++label) {
            const double classAcc =
                (static_cast<double>(corrects[label]) + 1.0) /
                (static_cast<double>(std::max(1, totals[label])) + 2.0);
            const double predictionPrecision =
                (static_cast<double>(predictedCorrects[label]) + 1.0) /
                (static_cast<double>(std::max(1, predictedTotals[label])) + 2.0);
            rawStats[hemisphereIndex].trueLabelAccuracy[label] = classAcc;
            rawStats[hemisphereIndex].predictionPrecision[label] = predictionPrecision;
        }
    }

    double meanOverall = 0.0;
    for (const auto& stats : rawStats) {
        meanOverall += stats.overall;
    }
    meanOverall /= std::max<size_t>(1, rawStats.size());

    for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
        auto& hemisphere = hemispheres[hemisphereIndex];
        hemisphere.overallWeight =
            std::clamp(rawStats[hemisphereIndex].overall / std::max(1e-6, meanOverall), 0.85, 1.15);
        for (size_t label = 0; label < hemisphere.classWeights.size(); ++label) {
            double meanClassAcc = 0.0;
            double meanPredictionPrecision = 0.0;
            for (const auto& stats : rawStats) {
                meanClassAcc += stats.trueLabelAccuracy[label];
                meanPredictionPrecision += stats.predictionPrecision[label];
            }
            meanClassAcc /= std::max<size_t>(1, rawStats.size());
            meanPredictionPrecision /= std::max<size_t>(1, rawStats.size());
            hemisphere.classWeights[label] = std::clamp(
                rawStats[hemisphereIndex].trueLabelAccuracy[label] /
                    std::max(1e-6, meanClassAcc),
                0.85, 1.15);
            hemisphere.predictionWeights[label] = std::clamp(
                rawStats[hemisphereIndex].predictionPrecision[label] /
                    std::max(1e-6, meanPredictionPrecision),
                0.85, 1.15);
        }
    }
}

const LabelTrace* findLabelTrace(const std::vector<LabelTrace>& traces, int label) {
    for (const auto& trace : traces) {
        if (trace.label == label) {
            return &trace;
        }
    }
    return nullptr;
}

void scaleClamped(double& value, double factor, double minValue, double maxValue) {
    value = std::clamp(value * factor, minValue, maxValue);
}

BilateralDecisionTrace inferCorpusCallosumDecision(std::vector<HemisphereRuntime>& hemispheres,
                                                   const EMNISTLoader::Image& image,
                                                   const Config& config) {
    BilateralDecisionTrace trace;
    trace.hemisphereTraces.reserve(hemispheres.size());
    trace.combinedConfidence.assign(26, 0.0);

    for (auto& hemisphere : hemispheres) {
        trace.hemisphereTraces.push_back(inferHemisphereDecision(hemisphere, image, config));
    }

    for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
        auto& hemisphere = hemispheres[hemisphereIndex];
        const auto& hemisphereTrace = trace.hemisphereTraces[hemisphereIndex];
        std::vector<double> centroidEvidence;
        if (config.corpusCentroidGain > 0.0) {
            centroidEvidence = computeCentroidEvidence(hemisphere, hemisphereTrace.pattern);
        }
        std::vector<double> neighborSignature;
        if (config.corpusNeighborGain > 0.0) {
            const int neighborK = config.stage1K > 0 ? config.stage1K : config.knnK;
            neighborSignature = computeNeighborSignature(hemisphere, hemisphereTrace.pattern, neighborK);
        }

        for (size_t label = 0;
             label < trace.combinedConfidence.size() && label < hemisphereTrace.confidence.size();
             ++label) {
            trace.combinedConfidence[label] += hemisphere.overallWeight *
                                               hemisphere.classWeights[label] *
                                               hemisphereTrace.confidence[label];
            if (!centroidEvidence.empty()) {
                trace.combinedConfidence[label] += config.corpusCentroidGain *
                                                   hemisphere.overallWeight *
                                                   hemisphere.classWeights[label] *
                                                   centroidEvidence[label];
            }
            if (!neighborSignature.empty()) {
                trace.combinedConfidence[label] += config.corpusNeighborGain *
                                                   hemisphere.overallWeight *
                                                   hemisphere.classWeights[label] *
                                                   neighborSignature[label];
            }
        }

        if (hemisphereTrace.predicted >= 0 &&
            static_cast<size_t>(hemisphereTrace.predicted) < trace.combinedConfidence.size()) {
            trace.combinedConfidence[static_cast<size_t>(hemisphereTrace.predicted)] +=
                config.corpusVoteGain *
                hemisphere.overallWeight *
                hemisphere.predictionWeights[static_cast<size_t>(hemisphereTrace.predicted)] *
                (hemisphereTrace.topHypotheses.empty() ? 0.0 : hemisphereTrace.topHypotheses.front().score) *
                (1.0 + config.corpusMarginGain * hemisphereTrace.margin);
        }
    }

    normalizeSum(trace.combinedConfidence);
    trace.topHypotheses = collectTopHypotheses(trace.combinedConfidence, onlineTraceTopK(config));
    if (!trace.topHypotheses.empty()) {
        trace.predicted = trace.topHypotheses.front().label;
    }
    return trace;
}

std::vector<double> buildCorpusCallosumConfidence(std::vector<HemisphereRuntime>& hemispheres,
                                                  const EMNISTLoader::Image& image,
                                                  const Config& config) {
    return inferCorpusCallosumDecision(hemispheres, image, config).combinedConfidence;
}

BilateralDecisionTrace inferFusionDecision(std::vector<HemisphereRuntime>& hemispheres,
                                           const EMNISTLoader::Image& image,
                                           const Config& config,
                                           const ClassificationStrategy& fusionClassifier,
                                           const FusionRuntime& fusionRuntime) {
    BilateralDecisionTrace trace;
    trace.hemisphereTraces.reserve(hemispheres.size());
    for (auto& hemisphere : hemispheres) {
        trace.hemisphereTraces.push_back(inferHemisphereDecision(hemisphere, image, config));
    }
    trace.fusionPattern = buildFusionPatternFromTraces(trace.hemisphereTraces, config);
    trace.combinedConfidence = fusionClassifier.classifyWithConfidence(
        trace.fusionPattern, fusionRuntime.trainingPatterns, cosineSimilarity);
    normalizeSum(trace.combinedConfidence);
    trace.topHypotheses = collectTopHypotheses(trace.combinedConfidence, onlineTraceTopK(config));
    if (!trace.topHypotheses.empty()) {
        trace.predicted = trace.topHypotheses.front().label;
    }
    return trace;
}

void applyRewardToHemisphere(HemisphereRuntime& hemisphere,
                             const HemisphereDecisionTrace& trace,
                             int rewardedLabel,
                             double reward,
                             const Config& config) {
    if (rewardedLabel < 0 || rewardedLabel >= 26 || trace.topHypotheses.empty()) {
        return;
    }

    const double rewardMagnitude = std::clamp(std::abs(reward), 0.0, 3.0);
    const LabelTrace* rewardedTrace = findLabelTrace(trace.topHypotheses, rewardedLabel);

    if (reward > 0.0) {
        if (rewardedTrace == nullptr) {
            return;
        }
        const double rewardGain =
            std::clamp(config.onlinePositiveRewardGain * rewardMagnitude, 0.05, 2.5);
        const double baseLr =
            std::clamp(config.onlineCentroidLr * 0.35 * rewardGain, 0.02, 0.20);
        const double support = std::max(0.05, rewardedTrace->score);
        scaleClamped(hemisphere.classWeights[static_cast<size_t>(rewardedLabel)],
                     1.0 + baseLr * support,
                     0.5, 1.5);
        scaleClamped(hemisphere.overallWeight,
                     1.0 + baseLr * support * 0.4,
                     0.75, 1.25);
        if (trace.predicted == rewardedLabel) {
            scaleClamped(hemisphere.predictionWeights[static_cast<size_t>(rewardedLabel)],
                         1.0 + baseLr * support * (1.0 + trace.margin),
                         0.5, 1.5);
        }
        updateCentroid(hemisphere.classCentroids[static_cast<size_t>(rewardedLabel)],
                       trace.pattern,
                       std::clamp(config.onlineCentroidLr * support * rewardGain, 0.05, 0.45));
        insertOrReplaceOnlinePattern(hemisphere.trainingPatterns,
                                     &hemisphere.trainingSourceIndices,
                                     hemisphere.onlinePatternIndices,
                                     trace.pattern,
                                     rewardedLabel,
                                     config.onlineExemplarBudgetPerClass);
        return;
    }

    const LabelTrace* penalizedTrace = findLabelTrace(trace.topHypotheses, rewardedLabel);
    if (penalizedTrace == nullptr) {
        return;
    }
    const double rewardGain =
        std::clamp(config.onlineNegativeRewardGain * rewardMagnitude, 0.05, 2.5);
    const double baseLr =
        std::clamp(config.onlineCentroidLr * 0.35 * rewardGain, 0.02, 0.20);
    const double support = std::max(0.05, penalizedTrace->score);
    const double penalty = std::clamp(baseLr * support * (1.0 + trace.margin), 0.02, 0.20);
    scaleClamped(hemisphere.classWeights[static_cast<size_t>(rewardedLabel)],
                 1.0 - penalty,
                 0.5, 1.5);
    scaleClamped(hemisphere.overallWeight,
                 1.0 - penalty * 0.35,
                 0.75, 1.25);
    if (trace.predicted == rewardedLabel) {
        scaleClamped(hemisphere.predictionWeights[static_cast<size_t>(rewardedLabel)],
                     1.0 - penalty * 1.15,
                     0.5, 1.5);
    }
}

void applyRewardToFusion(FusionRuntime& fusionRuntime,
                         const std::vector<double>& fusionPattern,
                         int rewardedLabel,
                         double reward,
                         const Config& config) {
    if (reward <= 0.0 ||
        reward * config.onlinePositiveRewardGain < 0.05) {
        return;
    }
    insertOrReplaceOnlinePattern(fusionRuntime.trainingPatterns,
                                 nullptr,
                                 fusionRuntime.onlinePatternIndices,
                                 fusionPattern,
                                 rewardedLabel,
                                 config.onlineExemplarBudgetPerClass);
}

EvaluationResult evaluateCorpusCallosumPatterns(std::vector<HemisphereRuntime>& hemispheres,
                                                const EMNISTLoader& loader,
                                                const std::vector<size_t>& indices,
                                                const Config& config,
                                                const std::string& label) {
    EvaluationResult result;
    const auto start = std::chrono::high_resolution_clock::now();
    const int maxTests = static_cast<int>(indices.size());
    const int progressStep = maxTests >= 1000 ? 200 : (maxTests >= 200 ? 50 : 25);
    std::vector<OnlineSampleRecord> records;
    records.reserve(indices.size());
    std::vector<ReplayItem> replayQueue;
    replayQueue.reserve(static_cast<size_t>(std::max(16, config.onlineReplayQueueCapacity)));
    size_t replaySequence = 0;

    auto processReplayItem = [&](ReplayItem item, size_t currentStep) {
        if (item.recordIndex >= records.size()) {
            return;
        }
        auto& record = records[item.recordIndex];
        const auto& replayImage = loader.getImage(record.imageIndex);
        auto replayDecision = inferCorpusCallosumDecision(hemispheres, replayImage, config);
        if (replayDecision.predicted < 0 || replayDecision.predicted >= 26) {
            return;
        }

        record.finalPredicted = replayDecision.predicted;
        result.correctionReplays++;
        recordReplayTiming(result, item, currentStep);
        const auto replayContext = buildDecisionContext(replayDecision, config);
        const double effectiveReward =
            std::max(0.05, replayContext.plasticity * item.eligibilityScale);

        if (record.finalPredicted == record.truth) {
            if (!record.correctionSucceeded) {
                result.correctionSuccesses++;
                record.correctionSucceeded = true;
            }
            for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
                applyRewardToHemisphere(hemispheres[hemisphereIndex],
                                        replayDecision.hemisphereTraces[hemisphereIndex],
                                        replayDecision.predicted,
                                        effectiveReward,
                                        config);
            }
            return;
        }

        for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
            applyRewardToHemisphere(hemispheres[hemisphereIndex],
                                    replayDecision.hemisphereTraces[hemisphereIndex],
                                    replayDecision.predicted,
                                    -effectiveReward,
                                    config);
        }
        if (item.remainingReplays > 1) {
            enqueueReplayItem(replayQueue,
                              makeReplayItem(item.recordIndex,
                                             item.remainingReplays - 1,
                                             replayContext.replayPriority,
                                             currentStep,
                                             item.eligibilityScale,
                                             config,
                                             replaySequence++),
                              config);
        }
    };

    auto processReplayBudget = [&](size_t currentStep, int budget, bool flushAll = false) {
        if (!flushAll &&
            (currentStep == 0 || (currentStep % static_cast<size_t>(onlineReplayPauseInterval(config))) != 0)) {
            return;
        }
        const int batchBudget = flushAll ? budget : std::min(budget, onlineReplayBatchSize(config));
        ReplayItem item;
        int remaining = batchBudget;
        while (remaining-- > 0 && popReplayItem(replayQueue, item, currentStep, flushAll)) {
            processReplayItem(item, currentStep);
        }
    };

    auto currentAccuracy = [&]() {
        const int currentCorrect = static_cast<int>(std::count_if(
            records.begin(), records.end(), [](const OnlineSampleRecord& record) {
                return record.truth >= 0 && record.truth == record.finalPredicted;
            }));
        return 100.0 * static_cast<double>(currentCorrect) /
               static_cast<double>(std::max<size_t>(1, records.size()));
    };

    for (size_t index : indices) {
        const auto& image = loader.getImage(index);
        const int truth = static_cast<int>(image.label) - 1;
        if (truth < 0 || truth >= 26) {
            continue;
        }

        auto decision = inferCorpusCallosumDecision(hemispheres, image, config);
        const int initialPredicted = decision.predicted;
        if (initialPredicted < 0 || initialPredicted >= 26) {
            continue;
        }
        records.push_back({index, truth, initialPredicted, initialPredicted, false});
        auto& record = records.back();
        const auto context = buildDecisionContext(decision, config);

        if (useOnlineCorrection(config)) {
            if (initialPredicted == truth) {
                for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
                    applyRewardToHemisphere(hemispheres[hemisphereIndex],
                                            decision.hemisphereTraces[hemisphereIndex],
                                            decision.predicted,
                                            std::max(0.25, 0.5 * context.plasticity),
                                            config);
                }
            } else {
                result.correctionEvents++;
                for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
                    applyRewardToHemisphere(hemispheres[hemisphereIndex],
                                            decision.hemisphereTraces[hemisphereIndex],
                                            decision.predicted,
                                            -context.plasticity,
                                            config);
                }
                enqueueReplayItem(replayQueue,
                                  makeReplayItem(records.size() - 1,
                                                 config.onlineCorrectionRepeats,
                                                 context.replayPriority +
                                                     (context.uncertainty >=
                                                              config.onlineReplayUncertaintyThreshold
                                                          ? 0.5
                                                          : 0.0),
                                                 records.size(),
                                                 1.0,
                                                 config,
                                                 replaySequence++),
                                  config);
            }
        }

        processReplayBudget(records.size(), std::numeric_limits<int>::max());

        if (static_cast<int>(records.size()) % progressStep == 0) {
            const double acc = currentAccuracy();
            std::cout << "  " << label << ": " << records.size() << "/" << maxTests
                      << " (" << std::fixed << std::setprecision(2) << acc << "%)" << std::endl;
        }
    }

    processReplayBudget(records.size() + static_cast<size_t>(onlineReplayDelaySteps(config)),
                        std::numeric_limits<int>::max(),
                        true);

    const auto end = std::chrono::high_resolution_clock::now();
    finalizeEvaluationFromRecords(
        result, records, std::chrono::duration<double>(end - start).count());
    return result;
}

std::vector<double> extractPattern(std::vector<std::unique_ptr<RetinaAdapter>>& retinas,
                                   const EMNISTLoader::Image& image,
                                   bool useFeatures,
                                   bool learnPatterns) {
    std::vector<double> combined;

    for (auto& retina : retinas) {
        snnfw::adapters::SensoryAdapter::DataSample sample;
        sample.rawData = image.pixels;
        sample.timestamp = 0.0;

        std::vector<double> part;
        if (useFeatures) {
            auto features = retina->extractFeatures(sample);
            part = std::move(features.features);
        } else {
            retina->processData(sample);
            if (learnPatterns) {
                for (const auto& neuron : retina->getNeurons()) {
                    neuron->learnCurrentPattern();
                }
            }
            part = retina->getActivationPattern();
        }
        normalizeL2(part);
        const double fusionWeight = std::max(0.0, retina->getDoubleParam("fusion_weight", 1.0));
        if (fusionWeight != 1.0) {
            for (double& value : part) {
                value *= fusionWeight;
            }
        }
        combined.insert(combined.end(), part.begin(), part.end());
        retina->clearNeuronStates();
    }

    return combined;
}

std::array<std::vector<size_t>, 26> collectLabelIndices(const EMNISTLoader& loader) {
    std::array<std::vector<size_t>, 26> indicesByLabel;
    for (size_t i = 0; i < loader.size(); ++i) {
        const int label = static_cast<int>(loader.getImage(i).label) - 1;
        if (label >= 0 && label < 26) {
            indicesByLabel[static_cast<size_t>(label)].push_back(i);
        }
    }
    return indicesByLabel;
}

std::vector<size_t> selectStratifiedIndices(const std::array<std::vector<size_t>, 26>& indicesByLabel,
                                            int maxPerClass,
                                            unsigned int seed) {
    std::mt19937 rng(seed);
    std::vector<size_t> selected;

    for (size_t label = 0; label < indicesByLabel.size(); ++label) {
        auto shuffled = indicesByLabel[label];
        std::shuffle(shuffled.begin(), shuffled.end(), rng);
        const int keep = (maxPerClass > 0)
            ? std::min<int>(maxPerClass, static_cast<int>(shuffled.size()))
            : static_cast<int>(shuffled.size());
        selected.insert(selected.end(), shuffled.begin(), shuffled.begin() + keep);
    }

    std::shuffle(selected.begin(), selected.end(), rng);
    return selected;
}

std::vector<size_t> selectBalancedTestIndices(const std::array<std::vector<size_t>, 26>& indicesByLabel,
                                              int testLimit,
                                              unsigned int seed) {
    if (testLimit <= 0) {
        return selectStratifiedIndices(indicesByLabel, 0, seed);
    }

    std::mt19937 rng(seed);
    std::array<std::vector<size_t>, 26> shuffledByLabel = indicesByLabel;
    std::array<size_t, 26> offsets{};
    for (auto& indices : shuffledByLabel) {
        std::shuffle(indices.begin(), indices.end(), rng);
    }

    std::vector<size_t> selected;
    selected.reserve(static_cast<size_t>(testLimit));

    // Round-robin keeps the sample balanced even when the limit is not divisible by 26.
    while (static_cast<int>(selected.size()) < testLimit) {
        bool addedAny = false;
        for (size_t label = 0; label < shuffledByLabel.size(); ++label) {
            if (static_cast<int>(selected.size()) >= testLimit) {
                break;
            }
            auto& indices = shuffledByLabel[label];
            if (offsets[label] < indices.size()) {
                selected.push_back(indices[offsets[label]++]);
                addedAny = true;
            }
        }
        if (!addedAny) {
            break;
        }
    }

    std::shuffle(selected.begin(), selected.end(), rng);
    return selected;
}

std::vector<size_t> selectFocusedTestIndices(const std::array<std::vector<size_t>, 26>& indicesByLabel,
                                             const std::vector<int>& focusLabels,
                                             int limitPerLabel,
                                             unsigned int seed) {
    if (focusLabels.empty()) {
        return {};
    }

    std::mt19937 rng(seed);
    std::array<std::vector<size_t>, 26> shuffledByLabel = indicesByLabel;
    std::array<size_t, 26> offsets{};

    for (int label : focusLabels) {
        auto& indices = shuffledByLabel[static_cast<size_t>(label)];
        std::shuffle(indices.begin(), indices.end(), rng);
        if (limitPerLabel > 0 && static_cast<int>(indices.size()) > limitPerLabel) {
            indices.resize(static_cast<size_t>(limitPerLabel));
        }
    }

    std::vector<size_t> selected;
    while (true) {
        bool addedAny = false;
        for (int label : focusLabels) {
            auto& indices = shuffledByLabel[static_cast<size_t>(label)];
            auto& offset = offsets[static_cast<size_t>(label)];
            if (offset < indices.size()) {
                selected.push_back(indices[offset++]);
                addedAny = true;
            }
        }
        if (!addedAny) {
            break;
        }
    }

    return selected;
}

EvaluationResult evaluatePatterns(std::vector<std::unique_ptr<RetinaAdapter>>& retinas,
                                  const EMNISTLoader& loader,
                                  const std::vector<size_t>& indices,
                                  const Config& config,
                                  const ClassificationStrategy& classifier,
                                  const std::vector<ClassificationStrategy::LabeledPattern>& trainingPatterns,
                                  const std::string& label) {
    EvaluationResult result;
    const auto start = std::chrono::high_resolution_clock::now();
    const int maxTests = static_cast<int>(indices.size());
    const int progressStep = maxTests >= 1000 ? 200 : (maxTests >= 200 ? 50 : 25);

    for (size_t index : indices) {
        const auto& image = loader.getImage(index);
        const int truth = static_cast<int>(image.label) - 1;
        if (truth < 0 || truth >= 26) {
            continue;
        }

        const auto pattern = extractPattern(retinas, image, config.useFeatures, false);
        const int predicted = classifier.classify(pattern, trainingPatterns, cosineSimilarity);
        result.confusion[static_cast<size_t>(truth)][static_cast<size_t>(predicted)]++;
        result.correct += (predicted == truth) ? 1 : 0;
        result.tested++;

        if (result.tested % progressStep == 0) {
            const double acc =
                100.0 * static_cast<double>(result.correct) /
                static_cast<double>(std::max(1, result.tested));
            std::cout << "  " << label << ": " << result.tested << "/" << maxTests
                      << " (" << std::fixed << std::setprecision(2) << acc << "%)" << std::endl;
        }
    }

    const auto end = std::chrono::high_resolution_clock::now();
    result.seconds = std::chrono::duration<double>(end - start).count();
    return result;
}

EvaluationResult evaluateBilateralPatterns(std::vector<HemisphereRuntime>& hemispheres,
                                           const EMNISTLoader& loader,
                                           const std::vector<size_t>& indices,
                                           const Config& config,
                                           const ClassificationStrategy& fusionClassifier,
                                           FusionRuntime& fusionRuntime,
                                           const std::string& label) {
    EvaluationResult result;
    const auto start = std::chrono::high_resolution_clock::now();
    const int maxTests = static_cast<int>(indices.size());
    const int progressStep = maxTests >= 1000 ? 200 : (maxTests >= 200 ? 50 : 25);
    std::vector<OnlineSampleRecord> records;
    records.reserve(indices.size());
    std::vector<ReplayItem> replayQueue;
    replayQueue.reserve(static_cast<size_t>(std::max(16, config.onlineReplayQueueCapacity)));
    size_t replaySequence = 0;

    auto processReplayItem = [&](ReplayItem item, size_t currentStep) {
        if (item.recordIndex >= records.size()) {
            return;
        }
        auto& record = records[item.recordIndex];
        const auto& replayImage = loader.getImage(record.imageIndex);
        auto replayDecision = inferFusionDecision(
            hemispheres, replayImage, config, fusionClassifier, fusionRuntime);
        if (replayDecision.predicted < 0 || replayDecision.predicted >= 26) {
            return;
        }

        record.finalPredicted = replayDecision.predicted;
        result.correctionReplays++;
        recordReplayTiming(result, item, currentStep);
        const auto replayContext = buildDecisionContext(replayDecision, config);
        const double effectiveReward =
            std::max(0.05, replayContext.plasticity * item.eligibilityScale);

        if (record.finalPredicted == record.truth) {
            if (!record.correctionSucceeded) {
                result.correctionSuccesses++;
                record.correctionSucceeded = true;
            }
            for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
                applyRewardToHemisphere(hemispheres[hemisphereIndex],
                                        replayDecision.hemisphereTraces[hemisphereIndex],
                                        replayDecision.predicted,
                                        effectiveReward,
                                        config);
            }
            applyRewardToFusion(fusionRuntime,
                                replayDecision.fusionPattern,
                                replayDecision.predicted,
                                effectiveReward,
                                config);
            return;
        }

        for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
            applyRewardToHemisphere(hemispheres[hemisphereIndex],
                                    replayDecision.hemisphereTraces[hemisphereIndex],
                                    replayDecision.predicted,
                                    -effectiveReward,
                                    config);
        }
        if (item.remainingReplays > 1) {
            enqueueReplayItem(replayQueue,
                              makeReplayItem(item.recordIndex,
                                             item.remainingReplays - 1,
                                             replayContext.replayPriority,
                                             currentStep,
                                             item.eligibilityScale,
                                             config,
                                             replaySequence++),
                              config);
        }
    };

    auto processReplayBudget = [&](size_t currentStep, int budget, bool flushAll = false) {
        if (!flushAll &&
            (currentStep == 0 || (currentStep % static_cast<size_t>(onlineReplayPauseInterval(config))) != 0)) {
            return;
        }
        const int batchBudget = flushAll ? budget : std::min(budget, onlineReplayBatchSize(config));
        ReplayItem item;
        int remaining = batchBudget;
        while (remaining-- > 0 && popReplayItem(replayQueue, item, currentStep, flushAll)) {
            processReplayItem(item, currentStep);
        }
    };

    auto currentAccuracy = [&]() {
        const int currentCorrect = static_cast<int>(std::count_if(
            records.begin(), records.end(), [](const OnlineSampleRecord& record) {
                return record.truth >= 0 && record.truth == record.finalPredicted;
            }));
        return 100.0 * static_cast<double>(currentCorrect) /
               static_cast<double>(std::max<size_t>(1, records.size()));
    };

    for (size_t index : indices) {
        const auto& image = loader.getImage(index);
        const int truth = static_cast<int>(image.label) - 1;
        if (truth < 0 || truth >= 26) {
            continue;
        }

        auto decision = inferFusionDecision(hemispheres, image, config, fusionClassifier, fusionRuntime);
        const int initialPredicted = decision.predicted;
        if (initialPredicted < 0 || initialPredicted >= 26) {
            continue;
        }
        records.push_back({index, truth, initialPredicted, initialPredicted, false});
        const auto context = buildDecisionContext(decision, config);

        if (useOnlineCorrection(config)) {
            if (initialPredicted == truth) {
                for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
                    applyRewardToHemisphere(hemispheres[hemisphereIndex],
                                            decision.hemisphereTraces[hemisphereIndex],
                                            decision.predicted,
                                            std::max(0.25, 0.5 * context.plasticity),
                                            config);
                }
                applyRewardToFusion(fusionRuntime,
                                    decision.fusionPattern,
                                    decision.predicted,
                                    std::max(0.25, 0.5 * context.plasticity),
                                    config);
            } else {
                result.correctionEvents++;
                for (size_t hemisphereIndex = 0; hemisphereIndex < hemispheres.size(); ++hemisphereIndex) {
                    applyRewardToHemisphere(hemispheres[hemisphereIndex],
                                            decision.hemisphereTraces[hemisphereIndex],
                                            decision.predicted,
                                            -context.plasticity,
                                            config);
                }
                enqueueReplayItem(replayQueue,
                                  makeReplayItem(records.size() - 1,
                                                 config.onlineCorrectionRepeats,
                                                 context.replayPriority +
                                                     (context.uncertainty >=
                                                              config.onlineReplayUncertaintyThreshold
                                                          ? 0.5
                                                          : 0.0),
                                                 records.size(),
                                                 1.0,
                                                 config,
                                                 replaySequence++),
                                  config);
            }
        }

        processReplayBudget(records.size(), std::numeric_limits<int>::max());

        if (static_cast<int>(records.size()) % progressStep == 0) {
            const double acc = currentAccuracy();
            std::cout << "  " << label << ": " << records.size() << "/" << maxTests
                      << " (" << std::fixed << std::setprecision(2) << acc << "%)" << std::endl;
        }
    }

    processReplayBudget(records.size() + static_cast<size_t>(onlineReplayDelaySteps(config)),
                        std::numeric_limits<int>::max(),
                        true);

    const auto end = std::chrono::high_resolution_clock::now();
    finalizeEvaluationFromRecords(
        result, records, std::chrono::duration<double>(end - start).count());
    return result;
}

void printTopConfusions(const std::array<std::array<int, 26>, 26>& confusion) {
    struct PairConfusion {
        int a;
        int b;
        int total;
        int aToB;
        int bToA;
    };

    std::vector<PairConfusion> pairs;
    for (int i = 0; i < 26; ++i) {
        for (int j = i + 1; j < 26; ++j) {
            const int total = confusion[i][j] + confusion[j][i];
            if (total > 0) {
                pairs.push_back({i, j, total, confusion[i][j], confusion[j][i]});
            }
        }
    }

    std::sort(pairs.begin(), pairs.end(), [](const PairConfusion& lhs, const PairConfusion& rhs) {
        if (lhs.total != rhs.total) return lhs.total > rhs.total;
        if (lhs.aToB != rhs.aToB) return lhs.aToB > rhs.aToB;
        return lhs.bToA > rhs.bToA;
    });

    std::cout << "\nTop confusion pairs:" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(10, pairs.size()); ++i) {
        const auto& p = pairs[i];
        std::cout << "  " << classToChar(p.a) << "<->" << classToChar(p.b)
                  << ": " << p.total
                  << " (" << classToChar(p.a) << "->" << classToChar(p.b) << "=" << p.aToB
                  << ", " << classToChar(p.b) << "->" << classToChar(p.a) << "=" << p.bToA
                  << ")" << std::endl;
    }
}

void printPerClassAccuracy(const std::array<std::array<int, 26>, 26>& confusion) {
    std::cout << "\nPer-class accuracy:" << std::endl;
    for (int label = 0; label < 26; ++label) {
        int total = 0;
        for (int pred = 0; pred < 26; ++pred) {
            total += confusion[label][pred];
        }
        const double acc = total > 0
            ? (100.0 * static_cast<double>(confusion[label][label]) / static_cast<double>(total))
            : 0.0;
        std::cout << "  " << classToChar(label) << ": "
                  << std::fixed << std::setprecision(2) << acc << "% ("
                  << confusion[label][label] << "/" << total << ")" << std::endl;
    }
}

void printFocusedFamilyReport(const std::array<std::array<int, 26>, 26>& confusion,
                              const std::vector<std::vector<int>>& focusGroups) {
    if (focusGroups.empty()) {
        return;
    }

    std::cout << "\nFocused family report:" << std::endl;
    for (const auto& group : focusGroups) {
        if (group.empty()) {
            continue;
        }

        int total = 0;
        int correct = 0;
        int inFamily = 0;
        for (int truth : group) {
            for (int pred = 0; pred < 26; ++pred) {
                const int count = confusion[static_cast<size_t>(truth)][static_cast<size_t>(pred)];
                total += count;
                if (truth == pred) {
                    correct += count;
                }
                if (std::find(group.begin(), group.end(), pred) != group.end()) {
                    inFamily += count;
                }
            }
        }

        const double familyAcc = total > 0
            ? 100.0 * static_cast<double>(correct) / static_cast<double>(total)
            : 0.0;
        const double familyContainment = total > 0
            ? 100.0 * static_cast<double>(inFamily) / static_cast<double>(total)
            : 0.0;

        std::cout << "  [" << labelGroupToString(group) << "] accuracy="
                  << std::fixed << std::setprecision(2) << familyAcc << "% (" << correct
                  << "/" << total << "), in-family=" << familyContainment << "%" << std::endl;

        for (size_t i = 0; i < group.size(); ++i) {
            for (size_t j = i + 1; j < group.size(); ++j) {
                const int a = group[i];
                const int b = group[j];
                const int totalPair =
                    confusion[static_cast<size_t>(a)][static_cast<size_t>(b)] +
                    confusion[static_cast<size_t>(b)][static_cast<size_t>(a)];
                if (totalPair > 0) {
                    std::cout << "    " << classToChar(a) << "<->" << classToChar(b)
                              << ": " << totalPair
                              << " (" << classToChar(a) << "->" << classToChar(b) << "="
                              << confusion[static_cast<size_t>(a)][static_cast<size_t>(b)]
                              << ", " << classToChar(b) << "->" << classToChar(a) << "="
                              << confusion[static_cast<size_t>(b)][static_cast<size_t>(a)] << ")"
                              << std::endl;
                }
            }
        }

        struct EscapeConfusion {
            int truth;
            int pred;
            int count;
        };
        std::vector<EscapeConfusion> escapes;
        for (int truth : group) {
            for (int pred = 0; pred < 26; ++pred) {
                if (std::find(group.begin(), group.end(), pred) != group.end()) {
                    continue;
                }
                const int count =
                    confusion[static_cast<size_t>(truth)][static_cast<size_t>(pred)];
                if (count > 0) {
                    escapes.push_back({truth, pred, count});
                }
            }
        }
        std::sort(escapes.begin(), escapes.end(),
                  [](const EscapeConfusion& lhs, const EscapeConfusion& rhs) {
                      return lhs.count > rhs.count;
                  });
        for (size_t i = 0; i < std::min<size_t>(3, escapes.size()); ++i) {
            std::cout << "    escape " << classToChar(escapes[i].truth) << "->"
                      << classToChar(escapes[i].pred) << ": " << escapes[i].count << std::endl;
        }
    }
}

void printOnlineCorrectionSummary(const EvaluationResult& result) {
    if (result.tested <= 0 || result.correctionEvents <= 0) {
        return;
    }

    const double initialAccuracy =
        100.0 * static_cast<double>(result.initialCorrect) /
        static_cast<double>(std::max(1, result.tested));
    const double finalAccuracy =
        100.0 * static_cast<double>(result.correct) /
        static_cast<double>(std::max(1, result.tested));
    const double correctionHitRate =
        100.0 * static_cast<double>(result.correctionSuccesses) /
        static_cast<double>(std::max(1, result.correctionEvents));

    std::cout << "  Initial accuracy: " << std::fixed << std::setprecision(2)
              << initialAccuracy << "%" << std::endl;
    std::cout << "  Post-correction accuracy: " << std::fixed << std::setprecision(2)
              << finalAccuracy << "%" << std::endl;
    std::cout << "  Correction events: " << result.correctionEvents
              << ", corrected: " << result.correctionSuccesses
              << " (" << std::fixed << std::setprecision(2) << correctionHitRate << "%)"
              << ", replays: " << result.correctionReplays << std::endl;
    if (result.replayDelaySamples > 0) {
        std::cout << "  Replay timing: avg_delay_steps=" << std::fixed << std::setprecision(2)
                  << (result.replayDelaySteps /
                      static_cast<double>(std::max(1, result.replayDelaySamples)))
                  << ", avg_eligibility="
                  << (result.replayEligibilityScaleSum /
                      static_cast<double>(std::max(1, result.replayDelaySamples)))
                  << std::endl;
    }
}

Config parseArgs(int argc, char* argv[]) {
    Config config;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            config.configPath = argv[++i];
        }
    }

    if (!config.configPath.empty()) {
        loadDeclarativeConfig(config, config.configPath);
    }

    auto applyRetinaIntOverride = [&](const std::string& key, int value) {
        if (config.retinaConfigs.empty()) {
            return;
        }
        for (auto& retinaConfig : config.retinaConfigs) {
            retinaConfig.intParams[key] = value;
        }
    };
    auto applyRetinaDoubleOverride = [&](const std::string& key, double value) {
        if (config.retinaConfigs.empty()) {
            return;
        }
        for (auto& retinaConfig : config.retinaConfigs) {
            retinaConfig.doubleParams[key] = value;
        }
    };
    auto applyRetinaStringOverride = [&](const std::string& key, const std::string& value) {
        if (config.retinaConfigs.empty()) {
            return;
        }
        for (auto& retinaConfig : config.retinaConfigs) {
            retinaConfig.stringParams[key] = value;
        }
    };
    auto applyRetinaTemporalOverride = [&](double value) {
        if (config.retinaConfigs.empty()) {
            return;
        }
        for (auto& retinaConfig : config.retinaConfigs) {
            retinaConfig.temporalWindow = value;
            retinaConfig.doubleParams["neuron_window_size"] = value;
        }
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--config" && i + 1 < argc) {
            ++i;
        } else if (arg == "--train-images" && i + 1 < argc) {
            config.trainImagesPath = argv[++i];
        } else if (arg == "--train-labels" && i + 1 < argc) {
            config.trainLabelsPath = argv[++i];
        } else if (arg == "--test-images" && i + 1 < argc) {
            config.testImagesPath = argv[++i];
        } else if (arg == "--test-labels" && i + 1 < argc) {
            config.testLabelsPath = argv[++i];
        } else if (arg == "--examples-per-class" && i + 1 < argc) {
            config.examplesPerClass = std::atoi(argv[++i]);
        } else if (arg == "--test-limit" && i + 1 < argc) {
            config.testLimit = std::atoi(argv[++i]);
        } else if (arg == "--seed" && i + 1 < argc) {
            config.seed = static_cast<unsigned int>(std::atoi(argv[++i]));
        } else if (arg == "--grid-size" && i + 1 < argc) {
            config.gridSize = std::atoi(argv[++i]);
            config.gridSizes = {config.gridSize};
            config.retinaConfigs.clear();
        } else if (arg == "--grid-sizes" && i + 1 < argc) {
            config.gridSizes = parseCsvInts(argv[++i]);
            if (config.gridSizes.empty()) {
                throw std::runtime_error("grid-sizes must contain at least one integer");
            }
            config.gridSize = config.gridSizes.front();
            config.retinaConfigs.clear();
        } else if (arg == "--num-orientations" && i + 1 < argc) {
            config.numOrientations = std::atoi(argv[++i]);
            applyRetinaIntOverride("num_orientations", config.numOrientations);
        } else if (arg == "--edge-threshold" && i + 1 < argc) {
            config.edgeThreshold = std::atof(argv[++i]);
            applyRetinaDoubleOverride("edge_threshold", config.edgeThreshold);
        } else if (arg == "--temporal-window-ms" && i + 1 < argc) {
            config.temporalWindowMs = std::atof(argv[++i]);
            applyRetinaTemporalOverride(config.temporalWindowMs);
        } else if (arg == "--edge-operator" && i + 1 < argc) {
            config.edgeOperator = argv[++i];
            applyRetinaStringOverride("edge_operator", config.edgeOperator);
        } else if (arg == "--encoding-strategy" && i + 1 < argc) {
            config.encodingStrategy = argv[++i];
            applyRetinaStringOverride("encoding_strategy", config.encodingStrategy);
        } else if (arg == "--classifier" && i + 1 < argc) {
            config.classifier = argv[++i];
        } else if (arg == "--activation-mode" && i + 1 < argc) {
            config.activationMode = argv[++i];
            applyRetinaStringOverride("activation_mode", config.activationMode);
        } else if (arg == "--hierarchical-groups" && i + 1 < argc) {
            config.hierarchicalGroups = argv[++i];
        } else if (arg == "--hierarchical-coarse-strategy" && i + 1 < argc) {
            config.hierarchicalCoarseStrategy = argv[++i];
        } else if (arg == "--hierarchical-fine-strategy" && i + 1 < argc) {
            config.hierarchicalFineStrategy = argv[++i];
        } else if (arg == "--hierarchical-coarse-k" && i + 1 < argc) {
            config.hierarchicalCoarseK = std::atoi(argv[++i]);
        } else if (arg == "--hierarchical-fine-k" && i + 1 < argc) {
            config.hierarchicalFineK = std::atoi(argv[++i]);
        } else if (arg == "--knn-k" && i + 1 < argc) {
            config.knnK = std::atoi(argv[++i]);
        } else if (arg == "--classifier-exponent" && i + 1 < argc) {
            config.classifierExponent = std::atof(argv[++i]);
        } else if (arg == "--online-correction-repeats" && i + 1 < argc) {
            config.onlineCorrectionRepeats = std::atoi(argv[++i]);
        } else if (arg == "--online-exemplar-budget-per-class" && i + 1 < argc) {
            config.onlineExemplarBudgetPerClass = std::atoi(argv[++i]);
        } else if (arg == "--online-centroid-lr" && i + 1 < argc) {
            config.onlineCentroidLr = std::atof(argv[++i]);
        } else if (arg == "--online-positive-reward-gain" && i + 1 < argc) {
            config.onlinePositiveRewardGain = std::atof(argv[++i]);
        } else if (arg == "--online-negative-reward-gain" && i + 1 < argc) {
            config.onlineNegativeRewardGain = std::atof(argv[++i]);
        } else if (arg == "--online-replay-queue-capacity" && i + 1 < argc) {
            config.onlineReplayQueueCapacity = std::atoi(argv[++i]);
        } else if (arg == "--online-replay-delay-steps" && i + 1 < argc) {
            config.onlineReplayDelaySteps = std::atoi(argv[++i]);
        } else if (arg == "--online-replay-pause-interval" && i + 1 < argc) {
            config.onlineReplayPauseInterval = std::atoi(argv[++i]);
        } else if (arg == "--online-replay-batch-size" && i + 1 < argc) {
            config.onlineReplayBatchSize = std::atoi(argv[++i]);
        } else if (arg == "--online-replay-uncertainty-threshold" && i + 1 < argc) {
            config.onlineReplayUncertaintyThreshold = std::atof(argv[++i]);
        } else if (arg == "--online-eligibility-trace-decay" && i + 1 < argc) {
            config.onlineEligibilityTraceDecay = std::atof(argv[++i]);
        } else if (arg == "--online-context-uncertainty-gain" && i + 1 < argc) {
            config.onlineContextUncertaintyGain = std::atof(argv[++i]);
        } else if (arg == "--online-context-disagreement-gain" && i + 1 < argc) {
            config.onlineContextDisagreementGain = std::atof(argv[++i]);
        } else if (arg == "--use-features") {
            config.useFeatures = true;
        } else if (arg == "--use-activations") {
            config.useFeatures = false;
        } else if (arg == "--focus-groups" && i + 1 < argc) {
            config.focusGroups = parseLabelGroups(argv[++i]);
        } else if (arg == "--focus-limit-per-label" && i + 1 < argc) {
            config.focusLimitPerLabel = std::atoi(argv[++i]);
        } else if (arg == "--focus-only") {
            config.focusOnly = true;
        } else if (arg == "--help") {
            std::cout
                << "Usage: emnist_retina_letters [options]\n"
                << "  --config <path>\n"
                << "  --train-images <path>\n"
                << "  --train-labels <path>\n"
                << "  --test-images <path>\n"
                << "  --test-labels <path>\n"
                << "  --examples-per-class <n>\n"
                << "  --test-limit <n>\n"
                << "  --grid-size <n>\n"
                << "  --grid-sizes <n1,n2,...>\n"
                << "  --num-orientations <n>\n"
                << "  --edge-threshold <v>\n"
                << "  --edge-operator sobel|gabor|dog\n"
                << "  --encoding-strategy rate|temporal|population\n"
                << "  --classifier majority|weighted_similarity|weighted_distance|hierarchical\n"
                << "  --activation-mode binary|similarity|hybrid\n"
                << "  --hierarchical-groups <l1,l2;...>\n"
                << "  --hierarchical-coarse-strategy majority|weighted_similarity|weighted_distance\n"
                << "  --hierarchical-fine-strategy majority|weighted_similarity|weighted_distance\n"
                << "  --hierarchical-coarse-k <n>\n"
                << "  --hierarchical-fine-k <n>\n"
                << "  --knn-k <n>\n"
                << "  --classifier-exponent <v>\n"
                << "  --online-correction-repeats <n>\n"
                << "  --online-exemplar-budget-per-class <n>\n"
                << "  --online-centroid-lr <v>\n"
                << "  --online-positive-reward-gain <v>\n"
                << "  --online-negative-reward-gain <v>\n"
                << "  --online-replay-queue-capacity <n>\n"
                << "  --online-replay-delay-steps <n>\n"
                << "  --online-replay-pause-interval <n>\n"
                << "  --online-replay-batch-size <n>\n"
                << "  --online-replay-uncertainty-threshold <v>\n"
                << "  --online-eligibility-trace-decay <v>\n"
                << "  --online-context-uncertainty-gain <v>\n"
                << "  --online-context-disagreement-gain <v>\n"
                << "  --focus-groups <A,B;C,D;...>\n"
                << "  --focus-limit-per-label <n>\n"
                << "  --focus-only\n"
                << "  --use-features | --use-activations\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (config.trainImagesPath.empty() || config.trainLabelsPath.empty() ||
        config.testImagesPath.empty() || config.testLabelsPath.empty()) {
        throw std::runtime_error("EMNIST image/label paths are required");
    }
    if (config.focusOnly && config.focusGroups.empty()) {
        throw std::runtime_error("--focus-only requires --focus-groups");
    }

    return config;
}

} // namespace

int main(int argc, char* argv[]) {
    snnfw::Logger::getInstance().setLevel(spdlog::level::warn);

    try {
        const Config config = parseArgs(argc, argv);
        const auto retinaConfigs = buildRetinaConfigs(config);

        std::cout << "=== EMNIST Retina Classification ===" << std::endl;
        if (!config.configPath.empty()) {
            std::cout << "  Config: " << config.configPath << std::endl;
        }
        std::cout << "  Retina adapters: " << retinaConfigs.size() << std::endl;
        for (const auto& retinaConfig : retinaConfigs) {
            std::cout << "    - " << retinaConfig.name
                      << ": grid="
                      << retinaConfig.getIntParam("grid_size", config.gridSize)
                      << "x" << retinaConfig.getIntParam("grid_size", config.gridSize)
                      << ", orientations="
                      << retinaConfig.getIntParam("num_orientations", config.numOrientations)
                      << ", edge="
                      << retinaConfig.getStringParam("edge_operator", config.edgeOperator)
                      << ", threshold="
                      << retinaConfig.getDoubleParam("edge_threshold", config.edgeThreshold)
                      << ", fusion_weight="
                      << retinaConfig.getDoubleParam("fusion_weight", 1.0)
                      << ", hemisphere="
                      << retinaConfig.getStringParam("hemisphere", "default")
                      << ", rotation="
                      << retinaConfig.getDoubleParam("rotation_deg", 0.0)
                      << ", shift=("
                      << retinaConfig.getDoubleParam("shift_x_px", 0.0) << ","
                      << retinaConfig.getDoubleParam("shift_y_px", 0.0) << ")"
                      << ", encoding="
                      << retinaConfig.getStringParam("encoding_strategy", config.encodingStrategy)
                      << ", activation_mode="
                      << retinaConfig.getStringParam("activation_mode", config.activationMode)
                      << ", attach_path="
                      << retinaConfig.getStringParam("attach_path", "")
                      << std::endl;
        }
        std::cout << "  Classifier=" << config.classifier
                  << ", representation=" << (config.useFeatures ? "features" : "activations")
                  << std::endl;
        if (config.bilateralFusion) {
            std::cout << "  Fusion mode=bilateral"
                      << ", stage1=" << config.stage1Classifier
                      << " (k=" << (config.stage1K > 0 ? config.stage1K : config.knnK) << ")"
                      << ", fusion=" << config.fusionClassifier
                      << " (k=" << (config.fusionK > 0 ? config.fusionK : config.knnK) << ")"
                      << ", holdout/class="
                      << (config.fusionHoldoutPerClass > 0 ? config.fusionHoldoutPerClass : 0)
                      << ", fusion_features=" << config.fusionFeatureMode
                      << ", corpus_vote_gain=" << config.corpusVoteGain
                      << ", corpus_margin_gain=" << config.corpusMarginGain
                      << ", corpus_centroid_gain=" << config.corpusCentroidGain
                      << ", corpus_neighbor_gain=" << config.corpusNeighborGain
                      << ", online_repeats=" << config.onlineCorrectionRepeats
                      << ", online_budget/class=" << config.onlineExemplarBudgetPerClass
                      << ", online_centroid_lr=" << config.onlineCentroidLr
                      << ", online_pos_gain=" << config.onlinePositiveRewardGain
                      << ", online_neg_gain=" << config.onlineNegativeRewardGain
                      << ", replay_capacity=" << config.onlineReplayQueueCapacity
                      << ", replay_delay_steps=" << config.onlineReplayDelaySteps
                      << ", replay_pause_interval=" << config.onlineReplayPauseInterval
                      << ", replay_batch_size=" << config.onlineReplayBatchSize
                      << ", replay_uncertainty=" << config.onlineReplayUncertaintyThreshold
                      << ", eligibility_decay=" << config.onlineEligibilityTraceDecay
                      << ", context_uncertainty_gain=" << config.onlineContextUncertaintyGain
                      << ", context_disagreement_gain=" << config.onlineContextDisagreementGain
                      << ", fusion_path=" << (config.fusionPath.empty() ? "<none>" : config.fusionPath)
                      << std::endl;
        }
        if (config.classifier == "hierarchical") {
            std::cout << "  Hierarchy groups: " << config.hierarchicalGroups
                      << ", coarse=" << config.hierarchicalCoarseStrategy
                      << ", fine=" << config.hierarchicalFineStrategy
                      << ", coarse_k="
                      << (config.hierarchicalCoarseK > 0 ? config.hierarchicalCoarseK : config.knnK)
                      << ", fine_k="
                      << (config.hierarchicalFineK > 0 ? config.hierarchicalFineK : config.knnK)
                      << std::endl;
        }
        if (!config.focusGroups.empty()) {
            std::cout << "  Focus groups: ";
            for (size_t i = 0; i < config.focusGroups.size(); ++i) {
                if (i > 0) {
                    std::cout << ", ";
                }
                std::cout << "[" << labelGroupToString(config.focusGroups[i]) << "]";
            }
            if (config.focusLimitPerLabel > 0) {
                std::cout << " limit/class=" << config.focusLimitPerLabel;
            }
            if (config.focusOnly) {
                std::cout << " (focus only)";
            }
            std::cout << std::endl;
        }

        EMNISTLoader trainLoader(EMNISTLoader::Variant::LETTERS);
        EMNISTLoader testLoader(EMNISTLoader::Variant::LETTERS);
        if (!trainLoader.load(config.trainImagesPath, config.trainLabelsPath) ||
            !testLoader.load(config.testImagesPath, config.testLabelsPath)) {
            std::cerr << "Failed to load EMNIST dataset" << std::endl;
            return 1;
        }

        const auto trainIndicesByLabel = collectLabelIndices(trainLoader);
        const auto testIndicesByLabel = collectLabelIndices(testLoader);
        const auto selectedTestIndices =
            selectBalancedTestIndices(testIndicesByLabel, config.testLimit, config.seed + 1U);
        const auto focusLabels = flattenLabelGroups(config.focusGroups);
        const auto focusedTestIndices =
            selectFocusedTestIndices(testIndicesByLabel, focusLabels, config.focusLimitPerLabel,
                                     config.seed + 7U);

        if (!config.bilateralFusion) {
            auto retinas = createRetinaAdapters(retinaConfigs);
            auto classifier = makeClassifierStrategy(config.classifier, config.knnK,
                                                     config.classifierExponent, config);

            std::vector<ClassificationStrategy::LabeledPattern> trainingPatterns;
            trainingPatterns.reserve(static_cast<size_t>(26 * std::max(1, config.examplesPerClass)));
            const auto selectedTrainIndices =
                selectStratifiedIndices(trainIndicesByLabel, config.examplesPerClass, config.seed);

            const auto trainingStart = std::chrono::high_resolution_clock::now();
            for (size_t i : selectedTrainIndices) {
                const auto& image = trainLoader.getImage(i);
                const int label = static_cast<int>(image.label) - 1;
                if (label < 0 || label >= 26) {
                    continue;
                }
                trainingPatterns.emplace_back(
                    extractPattern(retinas, image, config.useFeatures,
                                   !config.useFeatures && config.activationMode != "binary"),
                    label);
                if (static_cast<int>(trainingPatterns.size()) % 1000 == 0) {
                    std::cout << "  Training patterns: " << trainingPatterns.size() << std::endl;
                }
            }

            std::cout << "  Stored training patterns: " << trainingPatterns.size() << std::endl;
            const auto end = std::chrono::high_resolution_clock::now();
            const double trainingSeconds =
                std::chrono::duration<double>(end - trainingStart).count();

            if (!config.focusOnly) {
                const auto eval = evaluatePatterns(retinas, testLoader, selectedTestIndices, config,
                                                   *classifier, trainingPatterns, "Testing");
                const double accuracy =
                    100.0 * static_cast<double>(eval.correct) /
                    static_cast<double>(std::max(1, eval.tested));

                std::cout << "\n=== Results ===" << std::endl;
                std::cout << "  Accuracy: " << std::fixed << std::setprecision(2) << accuracy
                          << "%" << std::endl;
                std::cout << "  Correct: " << eval.correct << "/" << eval.tested << std::endl;
                std::cout << "  Elapsed: " << std::fixed << std::setprecision(2)
                          << (trainingSeconds + eval.seconds) << "s" << std::endl;

                printPerClassAccuracy(eval.confusion);
                printTopConfusions(eval.confusion);
            }

            if (!config.focusGroups.empty()) {
                const auto focusedEval = evaluatePatterns(
                    retinas, testLoader, focusedTestIndices, config, *classifier,
                    trainingPatterns, "Focus");
                const double focusedAccuracy =
                    100.0 * static_cast<double>(focusedEval.correct) /
                    static_cast<double>(std::max(1, focusedEval.tested));

                std::cout << "\n=== Focused Results ===" << std::endl;
                std::cout << "  Accuracy: " << std::fixed << std::setprecision(2)
                          << focusedAccuracy << "%" << std::endl;
                std::cout << "  Correct: " << focusedEval.correct << "/" << focusedEval.tested
                          << std::endl;
                std::cout << "  Elapsed: " << std::fixed << std::setprecision(2)
                          << focusedEval.seconds << "s" << std::endl;

                printPerClassAccuracy(focusedEval.confusion);
                printTopConfusions(focusedEval.confusion);
                printFocusedFamilyReport(focusedEval.confusion, config.focusGroups);
            }
        } else {
            const auto groupedConfigs = groupRetinaConfigsByHemisphere(retinaConfigs, config);
            if (groupedConfigs.size() < 2) {
                throw std::runtime_error(
                    "bilateral fusion requires at least two hemisphere groups");
            }

            std::vector<HemisphereRuntime> hemispheres;
            hemispheres.reserve(groupedConfigs.size());
            std::cout << "  Bilateral fusion enabled: " << groupedConfigs.size()
                      << " hemisphere groups" << std::endl;
            if (config.hierarchicalRetinaLayout) {
                std::cout << "  Hierarchical Retina layout: enabled" << std::endl;
            }
            for (const auto& [name, configsForHemisphere] : groupedConfigs) {
                HemisphereRuntime hemisphere;
                hemisphere.name = name;
                hemisphere.retinas = createRetinaAdapters(configsForHemisphere);
                hemispheres.push_back(std::move(hemisphere));
                std::cout << "    * " << name << ": " << configsForHemisphere.size()
                          << " retina branches" << std::endl;
                if (config.hierarchicalRetinaLayout) {
                    for (const auto& retinaConfig : configsForHemisphere) {
                        std::cout << "      - " << retinaConfig.name
                                  << " -> " << retinaConfig.getStringParam("attach_path", "<unbound>")
                                  << std::endl;
                    }
                }
            }

            const int stage1K = config.stage1K > 0 ? config.stage1K : config.knnK;
            const int fusionK = config.fusionK > 0 ? config.fusionK : config.knnK;
            const auto split = splitTrainingIndices(trainIndicesByLabel, config.examplesPerClass,
                                                    config.fusionHoldoutPerClass, config.seed,
                                                    useCorpusCallosumFusion(config));

            const auto trainingStart = std::chrono::high_resolution_clock::now();
            for (auto& hemisphere : hemispheres) {
                hemisphere.trainingPatterns.reserve(split.stage1Indices.size());
                hemisphere.trainingSourceIndices.reserve(split.stage1Indices.size());
                for (size_t i : split.stage1Indices) {
                    const auto& image = trainLoader.getImage(i);
                    const int label = static_cast<int>(image.label) - 1;
                    if (label < 0 || label >= 26) {
                        continue;
                    }
                    hemisphere.trainingPatterns.emplace_back(
                        extractPattern(hemisphere.retinas, image, config.useFeatures,
                                       !config.useFeatures && config.activationMode != "binary"),
                        label);
                    hemisphere.trainingSourceIndices.push_back(i);
                    if (static_cast<int>(hemisphere.trainingPatterns.size()) % 1000 == 0) {
                        std::cout << "  Hemisphere " << hemisphere.name
                                  << " stage1 patterns: " << hemisphere.trainingPatterns.size()
                                  << std::endl;
                    }
                }
                hemisphere.classifier = makeClassifierStrategy(
                    config.stage1Classifier, stage1K, config.stage1Exponent, config);
                buildHemisphereClassCentroids(hemisphere);
                std::cout << "  Hemisphere " << hemisphere.name
                          << " stored stage1 patterns: "
                          << hemisphere.trainingPatterns.size() << std::endl;
            }

            std::unique_ptr<ClassificationStrategy> fusionClassifier;
            FusionRuntime fusionRuntime;
            if (useCorpusCallosumFusion(config)) {
                calibrateCorpusCallosumWeights(hemispheres, trainLoader, split.fusionIndices, config);
                std::cout << "  Corpus-callosum calibration samples: "
                          << split.fusionIndices.size() << std::endl;
                for (const auto& hemisphere : hemispheres) {
                    std::cout << "    - " << hemisphere.name
                              << " overall_weight=" << std::fixed << std::setprecision(3)
                              << hemisphere.overallWeight
                              << ", class_bias[I]=" << hemisphere.classWeights[8]
                              << ", class_bias[L]=" << hemisphere.classWeights[11]
                              << ", class_bias[Q]=" << hemisphere.classWeights[16]
                              << ", vote_bias[I]=" << hemisphere.predictionWeights[8]
                              << ", vote_bias[L]=" << hemisphere.predictionWeights[11]
                              << ", vote_bias[Q]=" << hemisphere.predictionWeights[16]
                              << std::endl;
                }
            } else {
                fusionClassifier = makeClassifierStrategy(
                    config.fusionClassifier, fusionK, config.fusionExponent, config);
                fusionRuntime.trainingPatterns.reserve(split.fusionIndices.size());
                for (size_t i : split.fusionIndices) {
                    const auto& image = trainLoader.getImage(i);
                    const int label = static_cast<int>(image.label) - 1;
                    if (label < 0 || label >= 26) {
                        continue;
                    }
                    fusionRuntime.trainingPatterns.emplace_back(
                        buildFusionPattern(hemispheres, image, config), label);
                    if (static_cast<int>(fusionRuntime.trainingPatterns.size()) % 500 == 0) {
                        std::cout << "  Fusion patterns: " << fusionRuntime.trainingPatterns.size()
                                  << std::endl;
                    }
                }

                if (fusionRuntime.trainingPatterns.empty()) {
                    throw std::runtime_error(
                        "bilateral fusion produced no fusion training patterns; increase examples-per-class or reduce fusion_holdout_per_class");
                }

                std::cout << "  Stored fusion patterns: " << fusionRuntime.trainingPatterns.size()
                          << " (mode=" << config.fusionFeatureMode << ")" << std::endl;
            }
            const auto end = std::chrono::high_resolution_clock::now();
            const double trainingSeconds =
                std::chrono::duration<double>(end - trainingStart).count();

            if (!config.focusOnly) {
                const auto eval = useCorpusCallosumFusion(config)
                    ? evaluateCorpusCallosumPatterns(
                          hemispheres, testLoader, selectedTestIndices, config, "Testing")
                    : evaluateBilateralPatterns(
                          hemispheres, testLoader, selectedTestIndices, config, *fusionClassifier,
                          fusionRuntime, "Testing");
                const double accuracy =
                    100.0 * static_cast<double>(eval.correct) /
                    static_cast<double>(std::max(1, eval.tested));

                std::cout << "\n=== Results ===" << std::endl;
                std::cout << "  Accuracy: " << std::fixed << std::setprecision(2) << accuracy
                          << "%" << std::endl;
                std::cout << "  Correct: " << eval.correct << "/" << eval.tested << std::endl;
                std::cout << "  Elapsed: " << std::fixed << std::setprecision(2)
                          << (trainingSeconds + eval.seconds) << "s" << std::endl;
                printOnlineCorrectionSummary(eval);

                printPerClassAccuracy(eval.confusion);
                printTopConfusions(eval.confusion);
            }

            if (!config.focusGroups.empty()) {
                const auto focusedEval = useCorpusCallosumFusion(config)
                    ? evaluateCorpusCallosumPatterns(
                          hemispheres, testLoader, focusedTestIndices, config, "Focus")
                    : evaluateBilateralPatterns(
                          hemispheres, testLoader, focusedTestIndices, config, *fusionClassifier,
                          fusionRuntime, "Focus");
                const double focusedAccuracy =
                    100.0 * static_cast<double>(focusedEval.correct) /
                    static_cast<double>(std::max(1, focusedEval.tested));

                std::cout << "\n=== Focused Results ===" << std::endl;
                std::cout << "  Accuracy: " << std::fixed << std::setprecision(2)
                          << focusedAccuracy << "%" << std::endl;
                std::cout << "  Correct: " << focusedEval.correct << "/" << focusedEval.tested
                          << std::endl;
                std::cout << "  Elapsed: " << std::fixed << std::setprecision(2)
                          << focusedEval.seconds << "s" << std::endl;
                printOnlineCorrectionSummary(focusedEval);

                printPerClassAccuracy(focusedEval.confusion);
                printTopConfusions(focusedEval.confusion);
                printFocusedFamilyReport(focusedEval.confusion, config.focusGroups);
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
