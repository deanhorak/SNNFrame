#ifndef SNNFW_EXPERIMENT_KNN_CLASSIFIER_H
#define SNNFW_EXPERIMENT_KNN_CLASSIFIER_H

#include "snnfw/experiment/ExperimentConfig.h"
#include <vector>
#include <cstdint>
#include <utility>

namespace snnfw {
namespace experiment {

using L5CountVector = std::vector<uint16_t>;

/**
 * @brief k-NN + centroid classifier using L5 activation vectors.
 *
 * Stores per-class L5 spike-count patterns during training.
 * At test time, classifies using k-NN voting with cosine similarity,
 * falling back to centroid matching.
 */
class KNNClassifier {
public:
    explicit KNNClassifier(const ExperimentConfig& config);

    /// Reset all stored patterns and centroids
    void reset();

    /// Reset patterns for a new pass (optionally keep history)
    void resetForPass(bool keepHistory);

    /// Store an L5 pattern for a given class label
    void storePattern(int classLabel, const L5CountVector& counts);

    /// Classify using k-NN voting. Returns (predictedLabel, similarity).
    std::pair<int, double> classifyKNN(const L5CountVector& testCounts) const;

    /// Classify using centroid matching. Returns (predictedLabel, similarity).
    std::pair<int, double> classifyCentroid(const L5CountVector& testCounts) const;

    /// Compute cosine similarity between two count vectors
    static double cosineSimilarity(const L5CountVector& a, const L5CountVector& b);

    /// Compute centroid similarity for a specific class
    double centroidSimilarity(const L5CountVector& testCounts, int classLabel) const;

    /// Check if any class is missing patterns
    bool hasMissingClasses() const;

    /// Get number of stored patterns for a class
    size_t getPatternCount(int classLabel) const;

private:
    const ExperimentConfig& config_;
    int numClasses_;
    size_t totalL5Neurons_;

    // Per-class k-NN pattern store
    std::vector<std::vector<L5CountVector>> classPatterns_;

    // Per-class centroid accumulators
    std::vector<std::vector<int64_t>> classCentroids_;
    std::vector<int> classPatternCounts_;
};

} // namespace experiment
} // namespace snnfw

#endif // SNNFW_EXPERIMENT_KNN_CLASSIFIER_H

