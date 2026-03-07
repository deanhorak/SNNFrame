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
    classLatencySums_.assign(numClasses_, std::vector<double>(totalL5Neurons_, 0.0));
    classLatencyObsCounts_.assign(numClasses_, std::vector<uint32_t>(totalL5Neurons_, 0));
    classPatternCounts_.assign(numClasses_, 0);
    neuronPatternPresence_.assign(totalL5Neurons_, 0);
    totalPatternsSeen_ = 0;
}

void KNNClassifier::resetForPass(bool keepHistory) {
    if (!keepHistory) {
        reset();
    }
}

void KNNClassifier::storePattern(int classLabel, const L5CountVector& counts,
                                 const L5LatencyVector& latencies) {
    if (classLabel < 0 || classLabel >= numClasses_) return;

    // Store for k-NN
    if (classPatterns_[classLabel].size() >= config_.maxPatternsPerClass) {
        classPatterns_[classLabel].erase(classPatterns_[classLabel].begin());
    }
    classPatterns_[classLabel].push_back({counts, latencies});
    totalPatternsSeen_++;

    // Update centroid
    for (size_t idx = 0; idx < counts.size() && idx < totalL5Neurons_; ++idx) {
        if (counts[idx] > 0) {
            classCentroids_[classLabel][idx] += counts[idx];
            neuronPatternPresence_[idx] += 1;
        }
        if (!latencies.empty() && idx < latencies.size()) {
            const uint16_t latency = latencies[idx];
            if (latency != kNoLatency) {
                classLatencySums_[classLabel][idx] += latencyToSignal(latency);
                classLatencyObsCounts_[classLabel][idx] += 1;
            }
        }
    }
    classPatternCounts_[classLabel]++;
}

double KNNClassifier::latencyToSignal(uint16_t latency) const {
    const double window = std::max(1.0, config_.neuronWindow);
    const double l = std::min(static_cast<double>(latency), window);
    return 1.0 - (l / window);
}

double KNNClassifier::temporalFeatureScale(uint16_t latency) const {
    if (!config_.enableTemporalFeatureCoding || latency == kNoLatency) {
        return 1.0;
    }
    const double gain = std::max(0.0, config_.temporalFeatureGain);
    const double power = std::max(0.1, config_.temporalFeaturePower);
    const double signal = std::pow(std::clamp(latencyToSignal(latency), 0.0, 1.0), power);
    return 1.0 + (gain * signal);
}

double KNNClassifier::latencySimilarity(const L5LatencyVector& a, const L5LatencyVector& b) const {
    if (a.empty() || b.empty() || a.size() != b.size()) return 0.0;
    double dot = 0.0, normA = 0.0, normB = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double av = (a[i] == kNoLatency) ? 0.0 : latencyToSignal(a[i]);
        const double bv = (b[i] == kNoLatency) ? 0.0 : latencyToSignal(b[i]);
        dot += av * bv;
        normA += av * av;
        normB += bv * bv;
    }
    if (normA <= 0.0 || normB <= 0.0) return 0.0;
    return dot / (std::sqrt(normA) * std::sqrt(normB));
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

double KNNClassifier::idfWeight(size_t idx) const {
    if (!config_.enableReadoutIdfWeighting) return 1.0;
    if (idx >= neuronPatternPresence_.size()) return 1.0;
    const double n = static_cast<double>(std::max<uint32_t>(1, totalPatternsSeen_));
    const double df = static_cast<double>(neuronPatternPresence_[idx]);
    const double idf = std::log((n + 1.0) / (df + 1.0)) + 1.0;
    const double p = std::max(0.0, config_.readoutIdfPower);
    return std::pow(idf, p);
}

double KNNClassifier::weightedCosineSimilarity(const L5CountVector& a, const L5CountVector& b,
                                               const L5LatencyVector* latA,
                                               const L5LatencyVector* latB) const {
    if (a.empty() || b.empty() || a.size() != b.size()) return 0.0;
    double dot = 0.0, normA = 0.0, normB = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double w = idfWeight(i);
        const uint16_t la = (latA && i < latA->size()) ? (*latA)[i] : kNoLatency;
        const uint16_t lb = (latB && i < latB->size()) ? (*latB)[i] : kNoLatency;
        const double av = static_cast<double>(a[i]) * temporalFeatureScale(la) * w;
        const double bv = static_cast<double>(b[i]) * temporalFeatureScale(lb) * w;
        dot += av * bv;
        normA += av * av;
        normB += bv * bv;
    }
    if (normA <= 0.0 || normB <= 0.0) return 0.0;
    return dot / (std::sqrt(normA) * std::sqrt(normB));
}

double KNNClassifier::latencyCentroidSimilarity(const L5LatencyVector& testLatencies,
                                                int classLabel) const {
    if (testLatencies.empty() || classLabel < 0 || classLabel >= numClasses_) return 0.0;
    if (classPatternCounts_[classLabel] == 0) return 0.0;
    double dot = 0.0, normTest = 0.0, normCentroid = 0.0;
    for (size_t idx = 0; idx < testLatencies.size() && idx < totalL5Neurons_; ++idx) {
        const uint16_t latency = testLatencies[idx];
        const double testVal = (latency == kNoLatency) ? 0.0 : latencyToSignal(latency);
        double centroidVal = 0.0;
        const uint32_t obs = classLatencyObsCounts_[classLabel][idx];
        if (obs > 0) {
            centroidVal = classLatencySums_[classLabel][idx] / static_cast<double>(obs);
        }
        dot += testVal * centroidVal;
        normTest += testVal * testVal;
        normCentroid += centroidVal * centroidVal;
    }
    if (normTest <= 0.0 || normCentroid <= 0.0) return 0.0;
    return dot / (std::sqrt(normTest) * std::sqrt(normCentroid));
}

double KNNClassifier::weightedCentroidSimilarity(const L5CountVector& testCounts,
                                                 int classLabel,
                                                 const L5LatencyVector* testLatencies) const {
    if (testCounts.empty() || classLabel < 0 || classLabel >= numClasses_) return 0.0;
    if (classPatternCounts_[classLabel] == 0) return 0.0;

    double dot = 0.0, normTest = 0.0, normCentroid = 0.0;
    const double invCount = 1.0 / static_cast<double>(classPatternCounts_[classLabel]);
    for (size_t idx = 0; idx < testCounts.size() && idx < totalL5Neurons_; ++idx) {
        const double w = idfWeight(idx);
        const uint16_t testLatency =
            (testLatencies && idx < testLatencies->size()) ? (*testLatencies)[idx] : kNoLatency;
        const double testVal =
            static_cast<double>(testCounts[idx]) * temporalFeatureScale(testLatency) * w;
        const uint32_t obs = classLatencyObsCounts_[classLabel][idx];
        const double centroidTemporalScale = (obs > 0)
            ? temporalFeatureScale(static_cast<uint16_t>(std::lround(
                std::clamp((1.0 - (classLatencySums_[classLabel][idx] / static_cast<double>(obs))) *
                           std::max(1.0, config_.neuronWindow),
                           0.0, 65534.0))))
            : 1.0;
        const double centroidVal =
            static_cast<double>(classCentroids_[classLabel][idx]) * invCount *
            centroidTemporalScale * w;
        dot += testVal * centroidVal;
        normTest += testVal * testVal;
        normCentroid += centroidVal * centroidVal;
    }
    if (normTest <= 0.0 || normCentroid <= 0.0) return 0.0;
    return dot / (std::sqrt(normTest) * std::sqrt(normCentroid));
}

double KNNClassifier::centroidSimilarity(const L5CountVector& testCounts, int classLabel,
                                         const L5LatencyVector& testLatencies) const {
    if (testCounts.empty() || classLabel < 0 || classLabel >= numClasses_) return 0.0;
    if (classPatternCounts_[classLabel] == 0) return 0.0;

    const double countSim = weightedCentroidSimilarity(
        testCounts, classLabel, testLatencies.empty() ? nullptr : &testLatencies);
    if (!config_.enableTemporalLatencyReadout) return countSim;
    const double latSim = latencyCentroidSimilarity(testLatencies, classLabel);
    const double w = std::clamp(config_.temporalLatencyWeight, 0.0, 1.0);
    return (1.0 - w) * countSim + w * latSim;
}

std::pair<int, double> KNNClassifier::classifyKNN(const L5CountVector& testCounts,
                                                  const L5LatencyVector& testLatencies) const {
    const bool hasTestActivity = std::any_of(
        testCounts.begin(), testCounts.end(), [](uint16_t v) { return v > 0; });
    if (!hasTestActivity) {
        return {-1, 0.0};
    }

    const int K = config_.knnK;
    std::vector<std::pair<double, int>> allSimilarities;

    for (int cls = 0; cls < numClasses_; ++cls) {
        if (!config_.includeClasses.empty() && cls < static_cast<int>(config_.includeClasses.size())
            && !config_.includeClasses[cls]) continue;
        for (const auto& trainPattern : classPatterns_[cls]) {
            const double countSim = weightedCosineSimilarity(
                testCounts, trainPattern.counts,
                testLatencies.empty() ? nullptr : &testLatencies,
                trainPattern.latencies.empty() ? nullptr : &trainPattern.latencies);
            double sim = countSim;
            if (config_.enableTemporalLatencyReadout && !testLatencies.empty() &&
                !trainPattern.latencies.empty()) {
                const double latSim = latencySimilarity(testLatencies, trainPattern.latencies);
                const double w = std::clamp(config_.temporalLatencyWeight, 0.0, 1.0);
                sim = (1.0 - w) * countSim + w * latSim;
            }
            allSimilarities.push_back({sim, cls});
        }
    }

    std::sort(allSimilarities.begin(), allSimilarities.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    if (allSimilarities.empty()) {
        return {-1, 0.0};
    }

    std::vector<int> votes(numClasses_, 0);
    std::vector<double> weightedVotes(numClasses_, 0.0);
    std::vector<double> classBestSimilarity(numClasses_, -1.0);
    double maxSim = 0.0;
    int numVotes = std::min(K, static_cast<int>(allSimilarities.size()));
    const double exponent = std::max(0.0, config_.knnSimilarityExponent);
    for (int i = 0; i < numVotes; ++i) {
        const int cls = allSimilarities[i].second;
        const double sim = std::max(0.0, allSimilarities[i].first);
        votes[cls]++;
        classBestSimilarity[cls] = std::max(classBestSimilarity[cls], sim);
        weightedVotes[cls] += config_.enableKnnSimilarityWeightedVote
            ? std::pow(sim, exponent)
            : 1.0;
        if (i == 0) maxSim = sim;
    }
    if (maxSim <= 1e-9) {
        return {-1, 0.0};
    }

    int bestLabel = -1;
    int bestVoteCount = -1;
    double bestWeightedVote = -1.0;
    double bestNeighborSim = -1.0;
    for (int cls = 0; cls < numClasses_; ++cls) {
        if (!config_.includeClasses.empty() && cls < static_cast<int>(config_.includeClasses.size())
            && !config_.includeClasses[cls]) continue;
        const double weightedVote = weightedVotes[cls];
        const int voteCount = votes[cls];
        const double classSim = classBestSimilarity[cls];
        if (weightedVote > bestWeightedVote + 1e-12 ||
            (std::abs(weightedVote - bestWeightedVote) <= 1e-12 &&
             (voteCount > bestVoteCount ||
              (voteCount == bestVoteCount && classSim > bestNeighborSim)))) {
            bestWeightedVote = weightedVote;
            bestVoteCount = voteCount;
            bestNeighborSim = classSim;
            bestLabel = cls;
        }
    }
    return {bestLabel, maxSim};
}

std::pair<int, double> KNNClassifier::classifyCentroid(const L5CountVector& testCounts,
                                                       const L5LatencyVector& testLatencies) const {
    const bool hasTestActivity = std::any_of(
        testCounts.begin(), testCounts.end(), [](uint16_t v) { return v > 0; });
    if (!hasTestActivity) {
        return {-1, 0.0};
    }

    int bestLabel = -1;
    double bestSim = -1.0;
    for (int cls = 0; cls < numClasses_; ++cls) {
        if (!config_.includeClasses.empty() && cls < static_cast<int>(config_.includeClasses.size())
            && !config_.includeClasses[cls]) continue;
        double sim = centroidSimilarity(testCounts, cls, testLatencies);
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
