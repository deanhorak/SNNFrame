#include "snnfw/experiment/CompetitionManager.h"
#include <algorithm>
#include <cmath>

namespace snnfw {
namespace experiment {

CompetitionManager::CompetitionManager(const ExperimentConfig& config)
    : config_(config)
    , gen_(config.seed)
    , l4PostJitterDist_(0.0, config.l4PostJitterMs)
{
}

void CompetitionManager::setSeed(unsigned int seed) {
    gen_.seed(seed);
}

std::vector<bool> CompetitionManager::runL4Competition(
    std::vector<declarative::ConstructedNetwork::ColumnGroup>& columns,
    const std::vector<bool>& inputFired,
    double baseTime,
    std::shared_ptr<NetworkPropagator> propagator)
{
    std::vector<bool> colHasL4(columns.size(), false);
    int colIdxSeq = 0;

    for (auto& col : columns) {
        auto l4It = col.layerNeurons.find("L4");
        if (l4It == col.layerNeurons.end()) {
            colIdxSeq++;
            continue;
        }
        auto& l4Neurons = l4It->second;

        // maskMinActive gating: skip column if insufficient input activity in its receptive field
        if (config_.maskMinActive > 0 && !col.inputMaskActiveIdx.empty()) {
            int maskedCount = 0;
            for (int idx : col.inputMaskActiveIdx) {
                if (idx >= 0 && idx < static_cast<int>(inputFired.size()) && inputFired[idx]) {
                    maskedCount++;
                    if (maskedCount >= config_.maskMinActive) break;
                }
            }
            if (maskedCount < config_.maskMinActive) {
                colIdxSeq++;
                continue;  // Skip this column entirely
            }
        }

        const int L4_KEEP = std::max(1, config_.l4Keep);
        std::vector<std::pair<size_t, size_t>> ranked;
        ranked.reserve(l4Neurons.size());
        for (size_t i = 0; i < l4Neurons.size(); ++i) {
            ranked.emplace_back(l4Neurons[i]->getSpikes().size(), i);
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        int winners = 0;
        for (size_t i = 0; i < ranked.size() && winners < L4_KEEP; ++i) {
            if (ranked[i].first == 0) break;
            int localIdx = static_cast<int>(ranked[i].second);
            int l4Row = localIdx / config_.layer4Size;
            int l4Col = localIdx % config_.layer4Size;
            double spatialDelay = (l4Row * config_.l4RowDelay) + (l4Col * config_.l4ColDelay);
            double l4Jitter = (config_.l4PostJitterMs > 0.0) ? l4PostJitterDist_(gen_) : 0.0;
            double l4Fire = baseTime + 10.0 + config_.l4PostShiftMs + (colIdxSeq * 0.5)
                            + spatialDelay + l4Jitter;
            if (l4Fire < baseTime) l4Fire = baseTime;
            auto& l4Neuron = l4Neurons[localIdx];
            l4Neuron->fireSignature(l4Fire);
            propagator->fireNeuron(l4Neuron->getId(), l4Fire);
            l4Neuron->fireAndAcknowledge(l4Fire);
            colHasL4[colIdxSeq] = true;
            winners++;
        }
        colIdxSeq++;
    }
    return colHasL4;
}

std::vector<bool> CompetitionManager::runL5Competition(
    std::vector<declarative::ConstructedNetwork::ColumnGroup>& columns,
    const std::vector<bool>& colHasL4,
    double baseTime,
    std::shared_ptr<NetworkPropagator> propagator)
{
    // Count total L5 neurons
    size_t totalL5 = 0;
    for (auto& col : columns) {
        auto it = col.layerNeurons.find("L5");
        if (it != col.layerNeurons.end()) totalL5 += it->second.size();
    }

    const int L5_KEEP = std::max(1, config_.l5Keep);
    std::vector<bool> l5WinnerGlobal(totalL5, false);
    int colIdxSeq = 0;
    size_t l5Offset = 0;

    for (auto& col : columns) {
        auto it = col.layerNeurons.find("L5");
        if (it == col.layerNeurons.end()) { colIdxSeq++; continue; }
        auto& l5Neurons = it->second;

        for (auto& n : l5Neurons) n->resetInhibition();

        if (colIdxSeq < static_cast<int>(colHasL4.size()) && !colHasL4[colIdxSeq]) {
            l5Offset += l5Neurons.size();
            colIdxSeq++;
            continue;
        }

        std::vector<std::pair<size_t, size_t>> ranked;
        ranked.reserve(l5Neurons.size());
        for (size_t i = 0; i < l5Neurons.size(); ++i) {
            ranked.emplace_back(l5Neurons[i]->getSpikes().size(), i);
        }
        std::sort(ranked.begin(), ranked.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        int winners = 0;
        std::vector<bool> l5WinnerLocal(l5Neurons.size(), false);
        for (size_t i = 0; i < ranked.size() && winners < L5_KEEP; ++i) {
            if (ranked[i].first < static_cast<size_t>(config_.l5MinSpikes)) break;
            l5WinnerGlobal[l5Offset + ranked[i].second] = true;
            l5WinnerLocal[ranked[i].second] = true;
            winners++;
        }
        if (config_.enableL5Inhibition) {
            for (size_t i = 0; i < l5Neurons.size(); ++i) {
                if (!l5WinnerLocal[i]) {
                    l5Neurons[i]->applyInhibition(config_.l5InhibitLoser);
                }
            }
        }
        l5Offset += l5Neurons.size();
        colIdxSeq++;
    }
    return l5WinnerGlobal;
}

void CompetitionManager::applyOutputCompetition(std::vector<int>& counts) {
    if (!config_.enableOutputCompetition) return;
    if (config_.outputCompetitionKeep <= 0) return;
    int numClasses = static_cast<int>(counts.size());
    std::vector<std::pair<int, int>> ranked;
    ranked.reserve(numClasses);
    for (int i = 0; i < numClasses; ++i) {
        ranked.emplace_back(counts[i], i);
    }
    std::sort(ranked.begin(), ranked.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    if (ranked.empty() || ranked.front().first < config_.outputCompetitionMinSpikes) return;
    int keep = std::min(config_.outputCompetitionKeep, numClasses);
    for (int i = keep; i < numClasses; ++i) {
        counts[ranked[i].second] = 0;
    }
}

} // namespace experiment
} // namespace snnfw

