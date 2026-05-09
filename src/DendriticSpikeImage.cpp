#include "snnfw/DendriticSpikeImage.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace snnfw {

namespace {

size_t popcount(uint64_t value) {
    return static_cast<size_t>(__builtin_popcountll(value));
}

} // namespace

DendriticSpikeImage::DendriticSpikeImage(uint16_t rows,
                                         uint16_t timeBins,
                                         double binMs)
    : rows_(rows),
      timeBins_(timeBins),
      binMs_(binMs > 0.0 ? binMs : 1.0),
      bits_((static_cast<size_t>(rows) * static_cast<size_t>(timeBins) + 63U) / 64U,
            0U) {}

bool DendriticSpikeImage::addSpike(uint16_t row, double timeMs) {
    if (timeMs < 0.0 || binMs_ <= 0.0) {
        return false;
    }
    const double rawBin = std::floor(timeMs / binMs_);
    if (rawBin < 0.0 || rawBin > static_cast<double>(std::numeric_limits<uint16_t>::max())) {
        return false;
    }
    const auto bin = static_cast<uint16_t>(rawBin);
    return setSpike(row, bin);
}

bool DendriticSpikeImage::setSpike(uint16_t row, uint16_t timeBin) {
    if (row >= rows_ || timeBin >= timeBins_) {
        return false;
    }
    const size_t index = bitIndex(row, timeBin);
    bits_[index / 64U] |= (uint64_t{1} << (index % 64U));
    return true;
}

bool DendriticSpikeImage::hasSpike(uint16_t row, uint16_t timeBin) const {
    if (row >= rows_ || timeBin >= timeBins_) {
        return false;
    }
    const size_t index = bitIndex(row, timeBin);
    return (bits_[index / 64U] & (uint64_t{1} << (index % 64U))) != 0U;
}

void DendriticSpikeImage::clear() {
    std::fill(bits_.begin(), bits_.end(), 0U);
}

size_t DendriticSpikeImage::spikeCount() const {
    size_t count = 0;
    for (uint64_t word : bits_) {
        count += popcount(word);
    }
    return count;
}

double DendriticSpikeImage::jaccardSimilarity(const DendriticSpikeImage& other) const {
    if (!compatibleWith(other) || bits_.empty()) {
        return 0.0;
    }

    size_t intersection = 0;
    size_t unionCount = 0;
    for (size_t i = 0; i < bits_.size(); ++i) {
        intersection += popcount(bits_[i] & other.bits_[i]);
        unionCount += popcount(bits_[i] | other.bits_[i]);
    }
    return unionCount == 0 ? 0.0 : static_cast<double>(intersection) /
                                  static_cast<double>(unionCount);
}

double DendriticSpikeImage::temporalTolerantSimilarity(
    const DendriticSpikeImage& other,
    uint16_t toleranceBins) const {
    if (!compatibleWith(other)) {
        return 0.0;
    }

    const size_t thisCount = spikeCount();
    const size_t otherCount = other.spikeCount();
    if (thisCount == 0 || otherCount == 0) {
        return 0.0;
    }

    DendriticSpikeImage matchedOther(rows_, timeBins_, binMs_);
    size_t matches = 0;
    for (uint16_t row = 0; row < rows_; ++row) {
        for (uint16_t bin = 0; bin < timeBins_; ++bin) {
            if (!hasSpike(row, bin)) {
                continue;
            }
            const int begin = std::max<int>(0, static_cast<int>(bin) - toleranceBins);
            const int end =
                std::min<int>(static_cast<int>(timeBins_) - 1,
                              static_cast<int>(bin) + toleranceBins);
            int bestBin = -1;
            int bestDistance = static_cast<int>(toleranceBins) + 1;
            for (int candidate = begin; candidate <= end; ++candidate) {
                if (!other.hasSpike(row, static_cast<uint16_t>(candidate)) ||
                    matchedOther.hasSpike(row, static_cast<uint16_t>(candidate))) {
                    continue;
                }
                const int distance = std::abs(candidate - static_cast<int>(bin));
                if (distance < bestDistance) {
                    bestDistance = distance;
                    bestBin = candidate;
                }
            }
            if (bestBin >= 0) {
                matchedOther.setSpike(row, static_cast<uint16_t>(bestBin));
                matches++;
            }
        }
    }

    const size_t unionCount = thisCount + otherCount - matches;
    return unionCount == 0 ? 0.0 : static_cast<double>(matches) /
                                  static_cast<double>(unionCount);
}

void DendriticSpikeImage::mergeUnion(const DendriticSpikeImage& other) {
    if (!compatibleWith(other)) {
        return;
    }
    for (size_t i = 0; i < bits_.size(); ++i) {
        bits_[i] |= other.bits_[i];
    }
}

size_t DendriticSpikeImage::bitIndex(uint16_t row, uint16_t timeBin) const {
    return static_cast<size_t>(row) * static_cast<size_t>(timeBins_) +
           static_cast<size_t>(timeBin);
}

bool DendriticSpikeImage::compatibleWith(const DendriticSpikeImage& other) const {
    return rows_ == other.rows_ && timeBins_ == other.timeBins_ &&
           std::abs(binMs_ - other.binMs_) < 1e-9 &&
           bits_.size() == other.bits_.size();
}

} // namespace snnfw
