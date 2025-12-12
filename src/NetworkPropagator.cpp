#include "snnfw/NetworkPropagator.h"
#include "snnfw/ActionPotential.h"
#include "snnfw/RetrogradeActionPotential.h"
#include "snnfw/ActivityMonitor.h"
#include "snnfw/RecordingManager.h"
#include "snnfw/Logger.h"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace snnfw {

NetworkPropagator::NetworkPropagator(std::shared_ptr<SpikeProcessor> spikeProcessor)
    : spikeProcessor_(spikeProcessor),
      stdpAPlus_(0.01),
      stdpAMinus_(0.012),
      stdpTauPlus_(20.0),
      stdpTauMinus_(20.0),
      activityMonitor_(nullptr),
      recordingManager_(nullptr) {
    if (!spikeProcessor_) {
        SNNFW_ERROR("NetworkPropagator: SpikeProcessor cannot be null");
        throw std::invalid_argument("SpikeProcessor cannot be null");
    }
    SNNFW_INFO("NetworkPropagator: Initialized with STDP parameters (A+={}, A-={}, τ+={}, τ-={})",
               stdpAPlus_, stdpAMinus_, stdpTauPlus_, stdpTauMinus_);
}

void NetworkPropagator::setActivityMonitor(ActivityMonitor* monitor) {
    activityMonitor_ = monitor;
    if (monitor) {
        SNNFW_INFO("NetworkPropagator: Activity monitor attached for automatic recording");
    } else {
        SNNFW_INFO("NetworkPropagator: Activity monitor detached");
    }
}

void NetworkPropagator::setRecordingManager(RecordingManager* manager) {
    recordingManager_ = manager;
    if (manager) {
        SNNFW_INFO("NetworkPropagator: Recording manager attached for direct recording");
    } else {
        SNNFW_INFO("NetworkPropagator: Recording manager detached");
    }
}

void NetworkPropagator::registerNeuron(const std::shared_ptr<Neuron>& neuron) {
    if (!neuron) {
        SNNFW_ERROR("NetworkPropagator: Cannot register null neuron");
        return;
    }

    std::lock_guard<std::mutex> lock(neuronMutex_);
    neuronRegistry_[neuron->getId()] = neuron;
    SNNFW_DEBUG("NetworkPropagator: Registered neuron {}", neuron->getId());
}

void NetworkPropagator::registerAxon(const std::shared_ptr<Axon>& axon) {
    if (!axon) {
        SNNFW_ERROR("NetworkPropagator: Cannot register null axon");
        return;
    }

    std::lock_guard<std::mutex> lock(axonMutex_);
    axonRegistry_[axon->getId()] = axon;
    SNNFW_DEBUG("NetworkPropagator: Registered axon {}", axon->getId());
}

void NetworkPropagator::registerSynapse(const std::shared_ptr<Synapse>& synapse) {
    if (!synapse) {
        SNNFW_ERROR("NetworkPropagator: Cannot register null synapse");
        return;
    }

    std::lock_guard<std::mutex> lock(synapseMutex_);
    synapseRegistry_[synapse->getId()] = synapse;

    // Build reverse index: dendrite ID -> synapses
    uint64_t dendriteId = synapse->getDendriteId();
    dendriteToSynapsesIndex_[dendriteId].push_back(synapse);

    // Also register with SpikeProcessor for retrograde spike delivery
    spikeProcessor_->registerSynapse(synapse);

    SNNFW_DEBUG("NetworkPropagator: Registered synapse {} targeting dendrite {}",
                synapse->getId(), dendriteId);
}

void NetworkPropagator::registerDendrite(const std::shared_ptr<Dendrite>& dendrite) {
    if (!dendrite) {
        SNNFW_ERROR("NetworkPropagator: Cannot register null dendrite");
        return;
    }

    std::lock_guard<std::mutex> lock(dendriteMutex_);
    dendriteRegistry_[dendrite->getId()] = dendrite;
    
    // Also register with SpikeProcessor for delivery
    spikeProcessor_->registerDendrite(dendrite);
    
    SNNFW_DEBUG("NetworkPropagator: Registered dendrite {}", dendrite->getId());
}

int NetworkPropagator::fireNeuron(uint64_t neuronId, double firingTime) {
    // Record neuron firing if recording manager is attached (before checking axons/synapses)
    if (recordingManager_) {
        RecordedSpike recordedSpike;
        // Clamp negative times to 0 to avoid uint64 overflow
        recordedSpike.timestamp = static_cast<uint64_t>(std::max(0.0, firingTime));
        recordedSpike.sourceNeuronId = neuronId;
        recordedSpike.targetNeuronId = 0;  // No specific target for neuron-level recording
        recordedSpike.synapseId = 0;  // No specific synapse for neuron-level recording
        recordingManager_->recordSpike(recordedSpike);
    }

    // Get the neuron
    std::shared_ptr<Neuron> neuron;
    {
        std::lock_guard<std::mutex> lock(neuronMutex_);
        auto it = neuronRegistry_.find(neuronId);
        if (it == neuronRegistry_.end()) {
            SNNFW_WARN("NetworkPropagator: Neuron {} not found", neuronId);
            return -1;
        }
        neuron = it->second;
    }

    // Get the neuron's axon
    uint64_t axonId = neuron->getAxonId();
    if (axonId == 0) {
        SNNFW_DEBUG("NetworkPropagator: Neuron {} has no axon", neuronId);
        return 0;
    }

    std::shared_ptr<Axon> axon;
    {
        std::lock_guard<std::mutex> lock(axonMutex_);
        auto it = axonRegistry_.find(axonId);
        if (it == axonRegistry_.end()) {
            SNNFW_WARN("NetworkPropagator: Axon {} not found for neuron {}", axonId, neuronId);
            return -1;
        }
        axon = it->second;
    }

    // Get all synapses connected to this axon
    const auto& synapseIds = axon->getSynapseIds();
    if (synapseIds.empty()) {
        SNNFW_DEBUG("NetworkPropagator: Axon {} has no synapses", axonId);
        return 0;
    }

    // Get the neuron's temporal signature (unique spike pattern)
    const auto& temporalSignature = neuron->getTemporalSignature();
    if (temporalSignature.empty()) {
        SNNFW_WARN("NetworkPropagator: Neuron {} has no temporal signature", neuronId);
        return 0;
    }

    int spikesScheduled = 0;
    double minDelay = std::numeric_limits<double>::max();
    double maxDelay = std::numeric_limits<double>::min();

    // For each synapse, create and schedule multiple action potentials based on temporal signature
    for (uint64_t synapseId : synapseIds) {
        std::shared_ptr<Synapse> synapse;
        {
            std::lock_guard<std::mutex> lock(synapseMutex_);
            auto it = synapseRegistry_.find(synapseId);
            if (it == synapseRegistry_.end()) {
                SNNFW_WARN("NetworkPropagator: Synapse {} not found", synapseId);
                continue;
            }
            synapse = it->second;
        }

        double baseDelay = synapse->getDelay();
        double amplitude = synapse->getWeight();

        // Schedule one spike for each time offset in the temporal signature
        for (double timeOffset : temporalSignature) {
            double totalDelay = baseDelay + timeOffset;
            double arrivalTime = firingTime + totalDelay;

            // Track delay statistics
            minDelay = std::min(minDelay, totalDelay);
            maxDelay = std::max(maxDelay, totalDelay);

            auto actionPotential = std::make_shared<ActionPotential>(
                synapseId,
                synapse->getDendriteId(),
                arrivalTime,
                amplitude,
                firingTime  // Set dispatch time to when neuron fired
            );

            // Schedule for delivery
            if (spikeProcessor_->scheduleSpike(actionPotential)) {
                spikesScheduled++;
                SNNFW_TRACE("NetworkPropagator: Scheduled spike from neuron {} via synapse {} to dendrite {} at time {:.3f}ms (offset: {:.3f}ms, dispatch: {:.3f}ms)",
                           neuronId, synapseId, synapse->getDendriteId(), arrivalTime, timeOffset, firingTime);

                // Record spike in activity monitor if attached
                if (activityMonitor_) {
                    activityMonitor_->recordSpike(actionPotential, arrivalTime);
                }
            } else {
                SNNFW_WARN("NetworkPropagator: Failed to schedule spike from neuron {} via synapse {}",
                          neuronId, synapseId);
            }
        }

        // Schedule retrograde action potentials for STDP
        // These travel back to the synapse to update weights based on timing
        // Use the same delay as forward spikes (retrograde signals also take time to propagate)
        double retrogradeArrivalTime = firingTime + baseDelay;

        auto retrogradeAP = std::make_shared<RetrogradeActionPotential>(
            synapseId,
            neuronId,
            retrogradeArrivalTime,
            firingTime,  // dispatchTime (when the forward spike was sent)
            firingTime   // lastFiringTime (when this neuron fired)
        );

        if (spikeProcessor_->scheduleRetrogradeSpike(retrogradeAP)) {
            SNNFW_TRACE("NetworkPropagator: Scheduled retrograde spike from neuron {} to synapse {} at time {:.3f}ms",
                       neuronId, synapseId, retrogradeArrivalTime);
        } else {
            SNNFW_WARN("NetworkPropagator: Failed to schedule retrograde spike from neuron {} to synapse {}",
                      neuronId, synapseId);
        }
    }

    SNNFW_DEBUG("NetworkPropagator: Neuron {} fired at {:.3f}ms, scheduled {} spikes across {} synapses (delay range: {:.3f}-{:.3f}ms, temporal spread: {:.1f}ms)",
               neuronId, firingTime, spikesScheduled, synapseIds.size(), minDelay, maxDelay,
               temporalSignature.back() - temporalSignature.front());

    return spikesScheduled;
}

bool NetworkPropagator::deliverSpikeToNeuron(uint64_t neuronId, uint64_t synapseId, double spikeTime, double amplitude, double dispatchTime) {
    std::shared_ptr<Neuron> neuron;
    {
        std::lock_guard<std::mutex> lock(neuronMutex_);
        auto it = neuronRegistry_.find(neuronId);
        if (it == neuronRegistry_.end()) {
            SNNFW_WARN("NetworkPropagator: Cannot deliver spike to neuron {} - not found", neuronId);
            return false;
        }
        neuron = it->second;
    }

    // Insert spike into neuron's buffer
    // Note: amplitude could be used to modulate the spike, but for now we just insert the spike time
    neuron->insertSpike(spikeTime);

    // Record the incoming spike for STDP (with dispatch time)
    neuron->recordIncomingSpike(synapseId, spikeTime, dispatchTime);

    SNNFW_TRACE("NetworkPropagator: Delivered spike to neuron {} at time {:.3f}ms (amplitude: {:.3f}, synapse: {}, dispatch: {:.3f}ms)",
               neuronId, spikeTime, amplitude, synapseId, dispatchTime);

    return true;
}

std::shared_ptr<Neuron> NetworkPropagator::getNeuron(uint64_t neuronId) const {
    std::lock_guard<std::mutex> lock(neuronMutex_);
    auto it = neuronRegistry_.find(neuronId);
    return (it != neuronRegistry_.end()) ? it->second : nullptr;
}

std::shared_ptr<Axon> NetworkPropagator::getAxon(uint64_t axonId) const {
    std::lock_guard<std::mutex> lock(axonMutex_);
    auto it = axonRegistry_.find(axonId);
    return (it != axonRegistry_.end()) ? it->second : nullptr;
}

std::shared_ptr<Synapse> NetworkPropagator::getSynapse(uint64_t synapseId) const {
    std::lock_guard<std::mutex> lock(synapseMutex_);
    auto it = synapseRegistry_.find(synapseId);
    return (it != synapseRegistry_.end()) ? it->second : nullptr;
}

std::shared_ptr<Dendrite> NetworkPropagator::getDendrite(uint64_t dendriteId) const {
    std::lock_guard<std::mutex> lock(dendriteMutex_);
    auto it = dendriteRegistry_.find(dendriteId);
    return (it != dendriteRegistry_.end()) ? it->second : nullptr;
}

std::vector<double> NetworkPropagator::computeLayerActivation(const std::vector<uint64_t>& neuronIds) const {
    std::vector<double> activations;
    activations.reserve(neuronIds.size());

    for (uint64_t neuronId : neuronIds) {
        std::shared_ptr<Neuron> neuron;
        {
            std::lock_guard<std::mutex> lock(neuronMutex_);
            auto it = neuronRegistry_.find(neuronId);
            if (it != neuronRegistry_.end()) {
                neuron = it->second;
            }
        }

        if (neuron) {
            // Get the best similarity between current spikes and learned patterns
            double activation = neuron->getBestSimilarity();
            activations.push_back(activation);
        } else {
            SNNFW_WARN("NetworkPropagator: Neuron {} not found when computing activation", neuronId);
            activations.push_back(0.0);
        }
    }

    return activations;
}

void NetworkPropagator::clearAllSpikes() {
    std::lock_guard<std::mutex> lock(neuronMutex_);
    for (auto& pair : neuronRegistry_) {
        pair.second->clearSpikes();
    }
    SNNFW_DEBUG("NetworkPropagator: Cleared all spike buffers");
}

size_t NetworkPropagator::getNeuronCount() const {
    std::lock_guard<std::mutex> lock(neuronMutex_);
    return neuronRegistry_.size();
}

size_t NetworkPropagator::getSynapseCount() const {
    std::lock_guard<std::mutex> lock(synapseMutex_);
    return synapseRegistry_.size();
}

void NetworkPropagator::sendAcknowledgment(const std::shared_ptr<SpikeAcknowledgment>& acknowledgment) {
    if (!acknowledgment) {
        SNNFW_ERROR("NetworkPropagator: Cannot send null acknowledgment");
        return;
    }

    // Apply STDP to the synapse
    double timeDifference = acknowledgment->getTimeDifference();
    uint64_t synapseId = acknowledgment->getSynapseId();

    if (applySTDP(synapseId, timeDifference)) {
        SNNFW_TRACE("NetworkPropagator: Applied STDP to synapse {} (Δt = {:.3f}ms)",
                   synapseId, timeDifference);
    } else {
        SNNFW_WARN("NetworkPropagator: Failed to apply STDP to synapse {}", synapseId);
    }
}

bool NetworkPropagator::applySTDP(uint64_t synapseId, double timeDifference) {
    std::shared_ptr<Synapse> synapse;
    {
        std::lock_guard<std::mutex> lock(synapseMutex_);
        auto it = synapseRegistry_.find(synapseId);
        if (it == synapseRegistry_.end()) {
            SNNFW_WARN("NetworkPropagator: Synapse {} not found for STDP update", synapseId);
            return false;
        }
        synapse = it->second;
    }

    // Classic STDP learning rule:
    // Δw = A+ * exp(-Δt / τ+)  if Δt > 0 (LTP: pre before post)
    // Δw = -A- * exp(Δt / τ-)  if Δt < 0 (LTD: post before pre)

    double weightChange = 0.0;

    if (timeDifference > 0) {
        // LTP: strengthen synapse (pre-synaptic spike arrived before post-synaptic spike)
        weightChange = stdpAPlus_ * std::exp(-timeDifference / stdpTauPlus_);
    } else if (timeDifference < 0) {
        // LTD: weaken synapse (post-synaptic spike arrived before pre-synaptic spike)
        weightChange = -stdpAMinus_ * std::exp(timeDifference / stdpTauMinus_);
    }
    // If timeDifference == 0, no change

    if (weightChange != 0.0) {
        double oldWeight = synapse->getWeight();
        double newWeight = oldWeight + weightChange;

        // Clamp weight to [0, 2] range to prevent runaway growth/decay
        newWeight = std::max(0.0, std::min(2.0, newWeight));

        synapse->setWeight(newWeight);

        SNNFW_TRACE("NetworkPropagator: STDP update for synapse {}: Δt={:.3f}ms, Δw={:.6f}, weight: {:.4f} → {:.4f}",
                   synapseId, timeDifference, weightChange, oldWeight, newWeight);
    }

    return true;
}

void NetworkPropagator::setSTDPParameters(double aPlus, double aMinus, double tauPlus, double tauMinus) {
    stdpAPlus_ = aPlus;
    stdpAMinus_ = aMinus;
    stdpTauPlus_ = tauPlus;
    stdpTauMinus_ = tauMinus;

    SNNFW_INFO("NetworkPropagator: Updated STDP parameters (A+={}, A-={}, τ+={}, τ-={})",
               stdpAPlus_, stdpAMinus_, stdpTauPlus_, stdpTauMinus_);
}

void NetworkPropagator::applyRewardModulatedSTDP(uint64_t neuronId, double rewardFactor) {
    // Get the target neuron
    std::shared_ptr<Neuron> neuron;
    {
        std::lock_guard<std::mutex> lock(neuronMutex_);
        auto it = neuronRegistry_.find(neuronId);
        if (it == neuronRegistry_.end()) {
            SNNFW_WARN("NetworkPropagator: Cannot apply reward-modulated STDP - neuron {} not found", neuronId);
            return;
        }
        neuron = it->second;
    }

    // Find all synapses targeting this neuron's dendrites using reverse index
    // PERFORMANCE: O(k) where k = number of dendrites, instead of O(n) where n = total synapses
    std::vector<std::shared_ptr<Synapse>> targetSynapses;
    {
        std::lock_guard<std::mutex> lock(synapseMutex_);
        for (uint64_t dendriteId : neuron->getDendriteIds()) {
            auto it = dendriteToSynapsesIndex_.find(dendriteId);
            if (it != dendriteToSynapsesIndex_.end()) {
                // Append all synapses targeting this dendrite
                targetSynapses.insert(targetSynapses.end(), it->second.begin(), it->second.end());
            }
        }
    }

    // Apply reward-modulated weight changes
    int updatedCount = 0;
    for (const auto& synapse : targetSynapses) {
        double currentWeight = synapse->getWeight();

        // Reward-modulated learning: strengthen synapses proportional to reward
        // Positive reward (>1.0) = strengthen, negative reward (<1.0) = weaken
        double weightChange = stdpAPlus_ * (rewardFactor - 1.0);
        double newWeight = currentWeight + weightChange;

        // Clamp to [0, 2] range
        newWeight = std::max(0.0, std::min(2.0, newWeight));

        synapse->setWeight(newWeight);
        updatedCount++;

        SNNFW_TRACE("NetworkPropagator: Reward-modulated STDP - synapse {} weight: {:.4f} → {:.4f} (reward: {:.2f})",
                   synapse->getId(), currentWeight, newWeight, rewardFactor);
    }

    SNNFW_DEBUG("NetworkPropagator: Applied reward-modulated STDP to {} synapses targeting neuron {} (reward: {:.2f})",
                updatedCount, neuronId, rewardFactor);
}

} // namespace snnfw

