#include "snnfw/experiment/KNNClassifier.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace snnfw {
namespace experiment {

KNNClassifier::KNNClassifier(const ExperimentConfig& config)
    : config_(config)
    , numClasses_(config.numClasses)
    , totalL5Neurons_(static_cast<size_t>(config.numColumns) * config.layer5Neurons)
{
    reset();
}

void KNNClassifier::reset() {
    classPatterns_.assign(numClasses_, {});
    classCentroids_.assign(numClasses_, std::vector<int64_t>(totalL5Neurons_, 0));
    classPatternCounts_.assign(numClasses_, 0);
}

void KNNClassifier::resetForPass(bool keepHistory) {
    if (!keepHistory) {
        reset();
    }
}

void KNNClassifier::storePattern(int classLabel, const L5CountVector& counts) {
    if (classLabel < 0 || classLabel >= numClasses_) return;

    // Store for k-NN
    if (classPatterns_[classLabel].size() >= config_.maxPatternsPerClass) {
        classPatterns_[classLabel].erase(classPatterns_[classLabel].begin());
    }
    classPatterns_[classLabel].push_back(counts);

    // Update centroid
    for (size_t idx = 0; idx < counts.size() && idx < totalL5Neurons_; ++idx) {
        if (counts[idx] > 0) {
            classCentroids_[classLabel][idx] += counts[idx];
        }
    }
    classPatternCounts_[classLabel]++;
}

double KNNClassifier::cosineSimilarity(const L5CountVector& a, const L5CountVector& b) {
    if (a.empty() || b.empty() || a.size() != b.size()) return 0.0;
    double dot = 0.0, normA = 0.0, normB = 0.0;
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

double KNNClassifier::centroidSimilarity(const L5CountVector& testCounts, int classLabel) const {
    if (testCounts.empty() || classLabel < 0 || classLabel >= numClasses_) return 0.0;
    if (classPatternCounts_[classLabel] == 0) return 0.0;

    double dot = 0.0, normTest = 0.0, normCentroid = 0.0;
    double invCount = 1.0 / static_cast<double>(classPatternCounts_[classLabel]);
    for (size_t idx = 0; idx < testCounts.size() && idx < totalL5Neurons_; ++idx) {
        double testVal = static_cast<double>(testCounts[idx]);
        double centroidVal = static_cast<double>(classCentroids_[classLabel][idx]) * invCount;
        dot += testVal * centroidVal;
        normTest += testVal * testVal;
        normCentroid += centroidVal * centroidVal;
    }
    if (normTest <= 0.0 || normCentroid <= 0.0) return 0.0;
    return dot / (std::sqrt(normTest) * std::sqrt(normCentroid));
}

std::pair<int, double> KNNClassifier::classifyKNN(const L5CountVector& testCounts) const {
    const int K = config_.knnK;
    std::vector<std::pair<double, int>> allSimilarities;

    for (int cls = 0; cls < numClasses_; ++cls) {
        if (!config_.includeClasses.empty() && cls < static_cast<int>(config_.includeClasses.size())
            && !config_.includeClasses[cls]) continue;
        for (const auto& trainPattern : classPatterns_[cls]) {
            double sim = cosineSimilarity(testCounts, trainPattern);
            allSimilarities.push_back({sim, cls});
        }
    }

    std::sort(allSimilarities.begin(), allSimilarities.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<int> votes(numClasses_, 0);
    double maxSim = 0.0;
    int numVotes = std::min(K, static_cast<int>(allSimilarities.size()));
    for (int i = 0; i < numVotes; ++i) {
        votes[allSimilarities[i].second]++;
        if (i == 0) maxSim = allSimilarities[i].first;
    }

    int bestLabel = -1, maxVoteCount = 0;
    for (int cls = 0; cls < numClasses_; ++cls) {
        if (!config_.includeClasses.empty() && cls < static_cast<int>(config_.includeClasses.size())
            && !config_.includeClasses[cls]) continue;
        if (votes[cls] > maxVoteCount) {
            maxVoteCount = votes[cls];
            bestLabel = cls;
        }
    }
    return {bestLabel, maxSim};
}

std::pair<int, double> KNNClassifier::classifyCentroid(const L5CountVector& testCounts) const {
    int bestLabel = -1;
    double bestSim = -1.0;
    for (int cls = 0; cls < numClasses_; ++cls) {
        if (!config_.includeClasses.empty() && cls < static_cast<int>(config_.includeClasses.size())
            && !config_.includeClasses[cls]) continue;
        double sim = centroidSimilarity(testCounts, cls);
        if (sim > bestSim) { bestSim = sim; bestLabel = cls; }
    }
    return {bestLabel, bestSim};
}

bool KNNClassifier::hasMissingClasses() const {
    for (int cls = 0; cls < numClasses_; ++cls) {
        if (!config_.includeClasses.empty() && cls < static_cast<int>(config_.includeClasses.size())
            && !config_.includeClasses[cls]) continue;
        if (classPatterns_[cls].empty()) return true;
    }
    return false;
}

size_t KNNClassifier::getPatternCount(int classLabel) const {
    if (classLabel < 0 || classLabel >= numClasses_) return 0;
    return classPatterns_[classLabel].size();
}

} // namespace experiment
} // namespace snnfw

