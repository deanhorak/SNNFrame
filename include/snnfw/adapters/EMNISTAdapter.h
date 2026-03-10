#ifndef SNNFW_EMNIST_ADAPTER_H
#define SNNFW_EMNIST_ADAPTER_H

#include "snnfw/adapters/SensoryAdapter.h"
#include <vector>

namespace snnfw {
namespace adapters {

/**
 * @brief Sensory adapter that converts EMNIST image pixels into latency-coded spikes.
 *
 * This adapter preserves the existing experiment behavior:
 * - A pixel fires iff normalized_intensity > pixel_threshold
 * - Spike time is (1 - normalized_intensity) * input_latency_ms
 */
class EMNISTAdapter : public SensoryAdapter {
public:
    explicit EMNISTAdapter(const Config& config);
    ~EMNISTAdapter() override = default;

    bool initialize() override;

    SpikePattern processData(const DataSample& data) override;
    FeatureVector extractFeatures(const DataSample& data) override;
    SpikePattern encodeFeatures(const FeatureVector& features) override;

    const std::vector<std::shared_ptr<Neuron>>& getNeurons() const override {
        return neurons_;
    }

    std::vector<double> getActivationPattern() const override {
        return lastActivationPattern_;
    }

    size_t getNeuronCount() const override {
        return featureDimension_;
    }

    size_t getFeatureDimension() const override {
        return featureDimension_;
    }

    void clearNeuronStates() override;

private:
    double pixelThreshold_ = 0.4;
    double inputLatencyMs_ = 15.0;
    int imageRows_ = 28;
    int imageCols_ = 28;
    size_t featureDimension_ = 28 * 28;

    std::vector<std::shared_ptr<Neuron>> neurons_;
    std::vector<double> lastActivationPattern_;
};

} // namespace adapters
} // namespace snnfw

#endif // SNNFW_EMNIST_ADAPTER_H
