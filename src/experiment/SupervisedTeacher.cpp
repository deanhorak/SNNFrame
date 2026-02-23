#include "snnfw/experiment/SupervisedTeacher.h"
#include "snnfw/declarative/NetworkConstructor.h"
#include <algorithm>

namespace snnfw {
namespace experiment {

namespace {
bool meetsStdpEligibility(const ExperimentConfig& config,
                          const std::shared_ptr<NetworkPropagator>& propagator,
                          uint64_t neuronId) {
    if (!config.enableStdpEligibilityGate || !propagator) {
        return true;
    }

    const auto stats = propagator->getNeuronStdpEligibility(neuronId);
    const uint64_t minUpdates = static_cast<uint64_t>(
        std::max(0, config.stdpEligibilityMinUpdates));
    const uint64_t minLtp = static_cast<uint64_t>(
        std::max(0, config.stdpEligibilityMinLtp));

    if (stats.totalUpdates < minUpdates || stats.ltpUpdates < minLtp) {
        return false;
    }
    return stats.score(config.stdpEligibilityLtdPenalty) >= config.stdpEligibilityThreshold;
}
} // namespace

SupervisedTeacher::SupervisedTeacher(const ExperimentConfig& config)
    : config_(config)
{
}

int SupervisedTeacher::teach(
    int classLabel,
    std::vector<declarative::ConstructedNetwork::ColumnGroup>& columns,
    const std::vector<bool>& l5WinnerGlobal,
    const std::vector<bool>& colHasL4,
    std::vector<std::vector<std::shared_ptr<Neuron>>>& outputPopulations,
    double baseTime,
    std::shared_ptr<NetworkPropagator> propagator)
{
    int patternsLearned = 0;

    if (config_.enableFullPropagation) {
        // Fire L5 winners in full propagation mode so downstream STDP can occur
        bool hasEligibleL5Winner = false;
        int colIdxSeq = 0;
        size_t l5Offset = 0;
        for (auto& col : columns) {
            auto l5It = col.layerNeurons.find("L5");
            if (l5It == col.layerNeurons.end()) {
                colIdxSeq++;
                continue;
            }
            auto& l5Neurons = l5It->second;

            if (colIdxSeq < static_cast<int>(colHasL4.size()) && !colHasL4[colIdxSeq]) {
                l5Offset += l5Neurons.size();
                colIdxSeq++;
                continue;
            }

            int localIdx = 0;
            for (auto& l5Neuron : l5Neurons) {
                size_t globalIdx = l5Offset + localIdx;
                if (globalIdx < l5WinnerGlobal.size() && l5WinnerGlobal[globalIdx] &&
                    l5Neuron->getInhibition() <= config_.l5InhibitThreshold) {
                    double l5FireTime = baseTime + 15.0 + (colIdxSeq * 0.1) + (localIdx * 0.02);
                    l5Neuron->fireSignature(l5FireTime);
                    propagator->fireNeuron(l5Neuron->getId(), l5FireTime);
                    l5Neuron->fireAndAcknowledge(l5FireTime);
                    const bool eligible = meetsStdpEligibility(config_, propagator, l5Neuron->getId());
                    hasEligibleL5Winner = hasEligibleL5Winner || eligible;
                    if (eligible) {
                        l5Neuron->learnCurrentPattern();
                    }
                }
                localIdx++;
            }
            l5Offset += l5Neurons.size();
            colIdxSeq++;
        }

        // Supervised teaching signal: force the correct output population
        if (!config_.disableOutputTeach &&
            classLabel >= 0 && classLabel < static_cast<int>(outputPopulations.size())) {
            double teachTime = baseTime + 30.0;
            for (auto& outputNeuron : outputPopulations[classLabel]) {
                outputNeuron->fireSignature(teachTime);
                propagator->fireNeuron(outputNeuron->getId(), teachTime);
                outputNeuron->fireAndAcknowledge(teachTime);
                bool outputEligible = meetsStdpEligibility(config_, propagator, outputNeuron->getId());
                if (config_.enableStdpEligibilityGate && !outputEligible && hasEligibleL5Winner) {
                    // When output STDP traces lag, allow supervised write if upstream winners
                    // for this image already satisfied STDP eligibility.
                    outputEligible = true;
                }
                if (outputEligible) {
                    outputNeuron->learnCurrentPattern();
                    patternsLearned++;
                }
            }
        }
    }

    return patternsLearned;
}

} // namespace experiment
} // namespace snnfw
