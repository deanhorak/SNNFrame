#include "snnfw/DendriticPatternMemory.h"

#include <algorithm>
#include <limits>

namespace snnfw {

DendriticPatternMemory::DendriticPatternMemory()
    : DendriticPatternMemory(Config{}) {}

DendriticPatternMemory::DendriticPatternMemory(Config config)
    : config_(config) {
    if (config_.maxPrototypes == 0) {
        config_.maxPrototypes = 1;
    }
}

size_t DendriticPatternMemory::learn(const DendriticSpikeImage& image) {
    size_t bestIndex = 0;
    const double similarity = bestSimilarity(image, &bestIndex);
    if (!prototypes_.empty() && similarity >= config_.similarityThreshold) {
        auto& prototype = prototypes_[bestIndex];
        prototype.support++;
        if (config_.mergeOnReinforcement) {
            prototype.image.mergeUnion(image);
        }
        return bestIndex;
    }

    if (prototypes_.size() < config_.maxPrototypes) {
        prototypes_.push_back({image, 1U});
        return prototypes_.size() - 1U;
    }

    auto weakest = std::min_element(
        prototypes_.begin(),
        prototypes_.end(),
        [](const Prototype& lhs, const Prototype& rhs) {
            return lhs.support < rhs.support;
        });
    *weakest = {image, 1U};
    return static_cast<size_t>(std::distance(prototypes_.begin(), weakest));
}

double DendriticPatternMemory::bestSimilarity(const DendriticSpikeImage& image,
                                              size_t* prototypeIndex) const {
    double best = 0.0;
    size_t bestIndex = 0;
    for (size_t i = 0; i < prototypes_.size(); ++i) {
        const double similarity = prototypes_[i].image.temporalTolerantSimilarity(
            image, config_.temporalToleranceBins);
        if (similarity > best) {
            best = similarity;
            bestIndex = i;
        }
    }
    if (prototypeIndex != nullptr) {
        *prototypeIndex = bestIndex;
    }
    return best;
}

bool DendriticPatternMemory::recognizes(const DendriticSpikeImage& image) const {
    return bestSimilarity(image) >= config_.similarityThreshold;
}

} // namespace snnfw
