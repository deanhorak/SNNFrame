#pragma once

#include <cstdint>
#include <vector>

namespace snnfw {

class DendriticSpikeImage {
public:
    DendriticSpikeImage() = default;
    DendriticSpikeImage(uint16_t rows, uint16_t timeBins, double binMs = 1.0);

    uint16_t rows() const { return rows_; }
    uint16_t timeBins() const { return timeBins_; }
    double binMs() const { return binMs_; }
    bool empty() const { return spikeCount() == 0; }

    bool addSpike(uint16_t row, double timeMs);
    bool setSpike(uint16_t row, uint16_t timeBin);
    bool hasSpike(uint16_t row, uint16_t timeBin) const;
    void clear();

    size_t spikeCount() const;
    const std::vector<uint64_t>& words() const { return bits_; }

    double jaccardSimilarity(const DendriticSpikeImage& other) const;
    double temporalTolerantSimilarity(const DendriticSpikeImage& other,
                                      uint16_t toleranceBins) const;

    void mergeUnion(const DendriticSpikeImage& other);

private:
    uint16_t rows_ = 0;
    uint16_t timeBins_ = 0;
    double binMs_ = 1.0;
    std::vector<uint64_t> bits_;

    size_t bitCount() const {
        return static_cast<size_t>(rows_) * static_cast<size_t>(timeBins_);
    }
    size_t bitIndex(uint16_t row, uint16_t timeBin) const;
    bool compatibleWith(const DendriticSpikeImage& other) const;
};

} // namespace snnfw
