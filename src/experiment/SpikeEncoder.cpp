#include "snnfw/experiment/SpikeEncoder.h"
#include <chrono>
#include <thread>
#include <algorithm>
#include <iostream>

namespace snnfw {
namespace experiment {

SpikeEncoder::SpikeEncoder(const ExperimentConfig& config,
                           std::shared_ptr<SpikeProcessor> spikeProcessor,
                           std::shared_ptr<NetworkPropagator> propagator)
    : config_(config)
    , spikeProcessor_(std::move(spikeProcessor))
    , propagator_(std::move(propagator))
{
    lastEndTime_ = spikeProcessor_->getCurrentTime();
}

double SpikeEncoder::encodeAndInject(
    const EMNISTLoader::Image& image,
    std::vector<std::shared_ptr<Neuron>>& inputNeurons,
    std::vector<declarative::ConstructedNetwork::ColumnGroup>& columns,
    std::vector<std::vector<std::shared_ptr<Neuron>>>& outputPopulations)
{
    // Wait for schedule horizon
    double nowTime = spikeProcessor_->getCurrentTime();
    const double scheduleHorizon = spikeProcessor_->getMaxScheduleAheadMs() - 2.0;
    while (lastEndTime_ + config_.interImageGapMs > nowTime + scheduleHorizon) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        nowTime = spikeProcessor_->getCurrentTime();
    }
    double baseTime = std::max(nowTime, lastEndTime_) + config_.interImageGapMs;
    lastEndTime_ = baseTime + config_.neuronWindow;
    lastBaseTime_ = baseTime;

    // Memory cleanup for all neurons
    for (auto& col : columns) {
        for (auto& [layerName, neurons] : col.layerNeurons) {
            for (auto& n : neurons) {
                n->periodicMemoryCleanup(baseTime);
            }
        }
    }
    for (auto& pop : outputPopulations) {
        for (auto& n : pop) {
            n->periodicMemoryCleanup(baseTime);
        }
    }
    for (auto& n : inputNeurons) {
        n->periodicMemoryCleanup(baseTime);
    }
    // Ensure output spikes from prior images do not leak
    for (auto& pop : outputPopulations) {
        for (auto& n : pop) {
            n->removeSpikesBefore(baseTime);
        }
    }

    // Encode pixels as spikes
    lastInputFired_.assign(inputNeurons.size(), false);
    int spikeCount = 0;
    for (size_t idx = 0; idx < inputNeurons.size() && idx < image.pixels.size(); ++idx) {
        double norm = image.pixels[idx] / 255.0;
        if (norm > config_.pixelThreshold) {
            double fireT = baseTime + (1.0 - norm) * config_.inputLatencyMs;
            inputNeurons[idx]->fireSignature(fireT);
            propagator_->fireNeuron(inputNeurons[idx]->getId(), fireT);
            lastInputFired_[idx] = true;
            spikeCount++;
        }
    }

    static int g_encodeCount = 0;
    if (++g_encodeCount % 100 == 0) {
        std::cout << "[DEBUG] Encoded " << spikeCount << " input spikes for image " << g_encodeCount << std::endl;
    }

    return baseTime;
}

void SpikeEncoder::waitForSimTime(double targetTimeMs, double timeoutMs) {
    auto start = std::chrono::steady_clock::now();
    double startSimTime = spikeProcessor_->getCurrentTime();
    int iterations = 0;
    while (spikeProcessor_->getCurrentTime() < targetTimeMs) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start).count();
        if (elapsed > timeoutMs) {
            std::cout << "[WARN] waitForSimTime timeout after " << elapsed << "ms (target="
                      << targetTimeMs << ", current=" << spikeProcessor_->getCurrentTime() << ")" << std::endl;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        iterations++;
    }

    static int g_waitCount = 0;
    static double g_totalWaitMs = 0;
    static int g_totalIterations = 0;
    auto wallTime = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    g_totalWaitMs += wallTime;
    g_totalIterations += iterations;
    if (++g_waitCount % 200 == 0) {
        double simTimeDelta = spikeProcessor_->getCurrentTime() - startSimTime;
        std::cout << "[DEBUG] waitForSimTime: avg " << (g_totalWaitMs / 200.0) << "ms wall time, "
                  << (g_totalIterations / 200.0) << " iterations, last simDelta=" << simTimeDelta << "ms" << std::endl;
        g_totalWaitMs = 0;
        g_totalIterations = 0;
    }
}

} // namespace experiment
} // namespace snnfw

