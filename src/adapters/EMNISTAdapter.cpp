#include "snnfw/adapters/EMNISTAdapter.h"
#include <algorithm>
#include <cmath>

namespace snnfw {
namespace adapters {

EMNISTAdapter::EMNISTAdapter(const Config& config)
    : SensoryAdapter(config) {
    pixelThreshold_ = getDoubleParam("pixel_threshold", 0.4);
    inputLatencyMs_ = getDoubleParam("input_latency_ms", 15.0);
    imageRows_ = std::max(1, getIntParam("image_rows", 28));
    imageCols_ = std::max(1, getIntParam("image_cols", 28));
    featureDimension_ = static_cast<size_t>(imageRows_) * static_cast<size_t>(imageCols_);
}

bool EMNISTAdapter::initialize() {
    if (!SensoryAdapter::initialize()) {
        return false;
    }

    neurons_.clear();
    neurons_.reserve(featureDimension_);
    for (size_t i = 0; i < featureDimension_; ++i) {
        neurons_.push_back(std::make_shared<Neuron>(200.0, 0.7, 1, i));
    }
    lastActivationPattern_.assign(featureDimension_, 0.0);
    return true;
}

SensoryAdapter::SpikePattern EMNISTAdapter::processData(const DataSample& data) {
    SpikePattern pattern;
    pattern.timestamp = data.timestamp;
    pattern.duration = inputLatencyMs_;
    pattern.spikeTimes.resize(data.rawData.size());

    if (lastActivationPattern_.size() != data.rawData.size()) {
        lastActivationPattern_.assign(data.rawData.size(), 0.0);
    }
    featureDimension_ = data.rawData.size();

    for (size_t i = 0; i < data.rawData.size(); ++i) {
        const double norm = static_cast<double>(data.rawData[i]) / 255.0;
        lastActivationPattern_[i] = norm;
        if (norm > pixelThreshold_) {
            const double t = (1.0 - norm) * inputLatencyMs_;
            pattern.spikeTimes[i].push_back(t);
        }
    }
    return pattern;
}

SensoryAdapter::FeatureVector EMNISTAdapter::extractFeatures(const DataSample& data) {
    FeatureVector result;
    result.timestamp = data.timestamp;
    result.features.reserve(data.rawData.size());
    for (uint8_t pixel : data.rawData) {
        result.features.push_back(static_cast<double>(pixel) / 255.0);
    }

    if (!result.features.empty()) {
        featureDimension_ = result.features.size();
        lastActivationPattern_ = result.features;
        if (neurons_.size() != featureDimension_) {
            neurons_.clear();
            neurons_.reserve(featureDimension_);
            for (size_t i = 0; i < featureDimension_; ++i) {
                neurons_.push_back(std::make_shared<Neuron>(200.0, 0.7, 1, i));
            }
        }
    }

    return result;
}

SensoryAdapter::SpikePattern EMNISTAdapter::encodeFeatures(const FeatureVector& features) {
    SpikePattern pattern;
    pattern.timestamp = features.timestamp;
    pattern.duration = inputLatencyMs_;
    pattern.spikeTimes.resize(features.features.size());

    for (size_t i = 0; i < features.features.size(); ++i) {
        const double v = features.features[i];
        if (v > pixelThreshold_) {
            const double t = std::max(0.0, (1.0 - v) * inputLatencyMs_);
            pattern.spikeTimes[i].push_back(t);
        }
    }

    return pattern;
}

void EMNISTAdapter::clearNeuronStates() {
    for (auto& neuron : neurons_) {
        neuron->clearSpikes();
    }
    std::fill(lastActivationPattern_.begin(), lastActivationPattern_.end(), 0.0);
}

} // namespace adapters
} // namespace snnfw
