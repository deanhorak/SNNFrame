#pragma once

#include "snnfw/DendriticSpikeImage.h"

#include <cstdint>
#include <vector>

namespace snnfw {

class DendriticPatternMemory {
public:
    struct Config {
        double similarityThreshold = 0.65;
        uint16_t temporalToleranceBins = 1;
        size_t maxPrototypes = 64;
        bool mergeOnReinforcement = true;
    };

    struct Prototype {
        DendriticSpikeImage image;
        uint32_t support = 0;
    };

    DendriticPatternMemory();
    explicit DendriticPatternMemory(Config config);

    size_t learn(const DendriticSpikeImage& image);
    double bestSimilarity(const DendriticSpikeImage& image, size_t* prototypeIndex = nullptr) const;
    bool recognizes(const DendriticSpikeImage& image) const;

    size_t prototypeCount() const { return prototypes_.size(); }
    const std::vector<Prototype>& prototypes() const { return prototypes_; }
    void clear() { prototypes_.clear(); }

private:
    Config config_;
    std::vector<Prototype> prototypes_;
};

} // namespace snnfw
