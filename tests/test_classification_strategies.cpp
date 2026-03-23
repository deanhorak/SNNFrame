#include "snnfw/classification/ClassificationStrategy.h"
#include "snnfw/experiment/ExperimentConfig.h"
#include "snnfw/experiment/KNNClassifier.h"
#include <gtest/gtest.h>
#include <cmath>

namespace {

double cosineSimilarity(const std::vector<double>& a, const std::vector<double>& b) {
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

} // namespace

TEST(ClassificationStrategies, HierarchicalKNNClassifiesWithinConfiguredGroup) {
    snnfw::classification::ClassificationStrategy::Config config;
    config.name = "hierarchical";
    config.k = 3;
    config.numClasses = 6;
    config.distanceExponent = 1.0;
    config.stringParams["group_definitions"] = "0,1;2,3";
    config.stringParams["coarse_strategy"] = "majority";
    config.stringParams["fine_strategy"] = "weighted_similarity";
    config.intParams["coarse_k"] = 5;
    config.intParams["fine_k"] = 3;

    auto strategy =
        snnfw::classification::ClassificationStrategyFactory::create("hierarchical", config);

    std::vector<snnfw::classification::ClassificationStrategy::LabeledPattern> trainingPatterns = {
        {{1.0, 0.0}, 0},
        {{0.95, 0.05}, 0},
        {{0.85, 0.15}, 1},
        {{0.80, 0.20}, 1},
        {{0.0, 1.0}, 2},
        {{0.1, 0.9}, 3},
        {{-1.0, 0.0}, 4},
        {{0.0, -1.0}, 5},
    };

    const std::vector<double> testPattern = {0.82, 0.18};
    EXPECT_EQ(strategy->classify(testPattern, trainingPatterns, cosineSimilarity), 1);
}

TEST(ClassificationStrategies, HierarchicalKNNConfidenceZerosOutsideWinningGroup) {
    snnfw::classification::ClassificationStrategy::Config config;
    config.name = "hierarchical";
    config.k = 3;
    config.numClasses = 4;
    config.distanceExponent = 1.0;
    config.stringParams["group_definitions"] = "0,1;2,3";
    config.stringParams["coarse_strategy"] = "weighted_similarity";
    config.stringParams["fine_strategy"] = "majority";
    config.intParams["coarse_k"] = 4;
    config.intParams["fine_k"] = 3;

    auto strategy =
        snnfw::classification::ClassificationStrategyFactory::create("hierarchical", config);

    std::vector<snnfw::classification::ClassificationStrategy::LabeledPattern> trainingPatterns = {
        {{1.0, 0.0}, 0},
        {{0.9, 0.1}, 1},
        {{0.0, 1.0}, 2},
        {{0.1, 0.9}, 3},
    };

    const auto confidence =
        strategy->classifyWithConfidence({0.92, 0.08}, trainingPatterns, cosineSimilarity);

    EXPECT_GT(confidence[0] + confidence[1], 0.99);
    EXPECT_DOUBLE_EQ(confidence[2], 0.0);
    EXPECT_DOUBLE_EQ(confidence[3], 0.0);
}

TEST(ClassificationStrategies, ExperimentKNNClassifierUsesWeightedDistanceMode) {
    snnfw::experiment::ExperimentConfig config;
    config.numClasses = 2;
    config.numColumns = 1;
    config.layer5Neurons = 2;
    config.knnK = 2;
    config.classificationType = "weighted_distance";
    config.knnSimilarityExponent = 1.0;
    config.includeClasses = {true, true};

    snnfw::experiment::KNNClassifier classifier(config);
    classifier.storePattern(0, {10, 0});
    classifier.storePattern(1, {6, 8});

    const auto result = classifier.classifyKNN({9, 1});
    EXPECT_EQ(result.first, 0);
    EXPECT_GT(result.second, 0.9);
}

TEST(ClassificationStrategies, ExperimentKNNClassifierUsesHierarchicalMode) {
    snnfw::experiment::ExperimentConfig config;
    config.numClasses = 4;
    config.numColumns = 1;
    config.layer5Neurons = 2;
    config.knnK = 3;
    config.classificationType = "hierarchical";
    config.knnSimilarityExponent = 1.0;
    config.classificationStringParams["group_definitions"] = "0,1;2,3";
    config.classificationStringParams["coarse_strategy"] = "majority";
    config.classificationStringParams["fine_strategy"] = "weighted_similarity";
    config.classificationIntParams["coarse_k"] = 4;
    config.classificationIntParams["fine_k"] = 3;
    config.includeClasses = {true, true, true, true};

    snnfw::experiment::KNNClassifier classifier(config);
    classifier.storePattern(0, {10, 0});
    classifier.storePattern(0, {9, 1});
    classifier.storePattern(1, {8, 2});
    classifier.storePattern(1, {7, 3});
    classifier.storePattern(2, {0, 10});
    classifier.storePattern(3, {1, 9});

    const auto result = classifier.classifyKNN({8, 2});
    EXPECT_EQ(result.first, 1);
    EXPECT_GT(result.second, 0.5);
}
