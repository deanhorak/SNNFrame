#ifndef SNNFW_NETWORK_PROPAGATOR_H
#define SNNFW_NETWORK_PROPAGATOR_H

#include "snnfw/Neuron.h"
#include "snnfw/Axon.h"
#include "snnfw/Synapse.h"
#include "snnfw/Dendrite.h"
#include "snnfw/SpikeProcessor.h"
#include "snnfw/SpikeAcknowledgment.h"
#include <memory>
#include <map>
#include <vector>
#include <mutex>

namespace snnfw {

/**
 * @brief NetworkPropagator manages forward propagation of spikes through a multi-layer network
 *
 * This class coordinates the propagation of action potentials through explicit synaptic
 * connections in a biologically plausible manner. It maintains registries of all neural
 * objects and provides methods for:
 * - Registering neurons, axons, synapses, and dendrites
 * - Propagating spikes from source neurons through their axons and synapses
 * - Delivering spikes to target neurons via dendrites
 * - Computing layer activations based on spike patterns
 *
 * Architecture:
 * - Maintains registries for all neural objects (neurons, axons, synapses, dendrites)
 * - Uses SpikeProcessor for temporal spike delivery with delays
 * - Supports layer-by-layer forward propagation
 * - Thread-safe for concurrent access
 *
 * Biological Motivation:
 * In biological neural networks, action potentials propagate from the soma through
 * the axon, across synapses (with delays and weight modulation), and into dendrites
 * of downstream neurons. This class simulates that process explicitly.
 */
class NetworkPropagator {
public:
    /**
     * @brief Constructor
     * @param spikeProcessor Shared pointer to the spike processor for temporal delivery
     */
    explicit NetworkPropagator(std::shared_ptr<SpikeProcessor> spikeProcessor);

    /**
     * @brief Register a neuron with the propagator
     * @param neuron Shared pointer to the neuron
     */
    void registerNeuron(const std::shared_ptr<Neuron>& neuron);

    /**
     * @brief Register an axon with the propagator
     * @param axon Shared pointer to the axon
     */
    void registerAxon(const std::shared_ptr<Axon>& axon);

    /**
     * @brief Register a synapse with the propagator
     * @param synapse Shared pointer to the synapse
     */
    void registerSynapse(const std::shared_ptr<Synapse>& synapse);

    /**
     * @brief Register a dendrite with the propagator
     * @param dendrite Shared pointer to the dendrite
     */
    void registerDendrite(const std::shared_ptr<Dendrite>& dendrite);

    /**
     * @brief Fire a neuron and propagate spikes through its axon
     * 
     * When a neuron fires, this method:
     * 1. Gets the neuron's axon
     * 2. For each synapse connected to the axon:
     *    - Creates an ActionPotential with appropriate delay and weight
     *    - Schedules it for delivery via SpikeProcessor
     *
     * @param neuronId ID of the neuron that is firing
     * @param firingTime Time when the neuron fires (in ms)
     * @return Number of spikes scheduled, or -1 if neuron/axon not found
     */
    int fireNeuron(uint64_t neuronId, double firingTime);

    /**
     * @brief Deliver a spike to a target neuron via dendrite
     *
     * This is called by the dendrite when it receives an action potential.
     * It inserts the spike into the target neuron's spike buffer and records
     * the incoming spike for STDP.
     *
     * @param neuronId ID of the target neuron
     * @param synapseId ID of the synapse that delivered the spike
     * @param spikeTime Time of the spike (in ms)
     * @param amplitude Amplitude of the spike (modulated by synaptic weight)
     * @param dispatchTime Time when the spike was originally dispatched (in ms)
     * @return true if delivered successfully, false if neuron not found
     */
    bool deliverSpikeToNeuron(uint64_t neuronId, uint64_t synapseId, double spikeTime, double amplitude, double dispatchTime = 0.0);

    /**
     * @brief Send an acknowledgment from a postsynaptic neuron to update a synapse
     *
     * This implements the feedback mechanism for STDP. When a neuron fires, it sends
     * acknowledgments to all synapses that contributed spikes within the temporal window.
     * The synapse uses the timing information to adjust its weight.
     *
     * @param acknowledgment Shared pointer to the spike acknowledgment
     */
    void sendAcknowledgment(const std::shared_ptr<SpikeAcknowledgment>& acknowledgment);

    /**
     * @brief Apply STDP weight update to a synapse
     *
     * Uses the classic STDP learning rule:
     * - If Δt > 0 (pre before post): LTP (strengthen synapse)
     * - If Δt < 0 (post before pre): LTD (weaken synapse)
     * - Magnitude decreases exponentially with |Δt|
     *
     * @param synapseId ID of the synapse to update
     * @param timeDifference Δt = t_post - t_pre (in ms)
     * @return true if synapse was updated, false if not found
     */
    bool applySTDP(uint64_t synapseId, double timeDifference);

    /**
     * @brief Get a neuron by ID
     * @param neuronId ID of the neuron
     * @return Shared pointer to the neuron, or nullptr if not found
     */
    std::shared_ptr<Neuron> getNeuron(uint64_t neuronId) const;

    /**
     * @brief Get an axon by ID
     * @param axonId ID of the axon
     * @return Shared pointer to the axon, or nullptr if not found
     */
    std::shared_ptr<Axon> getAxon(uint64_t axonId) const;

    /**
     * @brief Get a synapse by ID
     * @param synapseId ID of the synapse
     * @return Shared pointer to the synapse, or nullptr if not found
     */
    std::shared_ptr<Synapse> getSynapse(uint64_t synapseId) const;

    /**
     * @brief Get a dendrite by ID
     * @param dendriteId ID of the dendrite
     * @return Shared pointer to the dendrite, or nullptr if not found
     */
    std::shared_ptr<Dendrite> getDendrite(uint64_t dendriteId) const;

    /**
     * @brief Compute activation vector for a layer of neurons
     * 
     * For each neuron in the layer, computes the best similarity between
     * its current spike pattern and its learned reference patterns.
     *
     * @param neuronIds Vector of neuron IDs in the layer
     * @return Vector of activation values (one per neuron)
     */
    std::vector<double> computeLayerActivation(const std::vector<uint64_t>& neuronIds) const;

    /**
     * @brief Clear all spike buffers in registered neurons
     */
    void clearAllSpikes();

    /**
     * @brief Get the number of registered neurons
     * @return Number of neurons in the registry
     */
    size_t getNeuronCount() const;

    /**
     * @brief Get the number of registered synapses
     * @return Number of synapses in the registry
     */
    size_t getSynapseCount() const;

    /**
     * @brief Set STDP learning parameters
     * @param aPlus LTP amplitude (default: 0.01)
     * @param aMinus LTD amplitude (default: 0.012)
     * @param tauPlus LTP time constant in ms (default: 20.0)
     * @param tauMinus LTD time constant in ms (default: 20.0)
     */
    void setSTDPParameters(double aPlus, double aMinus, double tauPlus, double tauMinus);

    /**
     * @brief Apply reward-modulated STDP to synapses targeting a specific neuron
     * @param neuronId ID of the target neuron
     * @param rewardFactor Reward multiplier (1.0 = normal STDP, >1.0 = enhanced learning, <1.0 = reduced learning)
     *
     * This strengthens synapses that contributed to the neuron's activation when reward is positive.
     * Used for supervised learning where correct classifications receive reward.
     */
    void applyRewardModulatedSTDP(uint64_t neuronId, double rewardFactor);

    /**
     * @brief Set activity monitor for automatic spike recording
     * @param monitor Pointer to activity monitor (nullptr to disable)
     */
    void setActivityMonitor(class ActivityMonitor* monitor);

    /**
     * @brief Set recording manager for direct spike recording (bypasses ActivityMonitor)
     * @param manager Pointer to recording manager (nullptr to disable)
     */
    void setRecordingManager(class RecordingManager* manager);

private:
    // Spike processor for temporal delivery
    std::shared_ptr<SpikeProcessor> spikeProcessor_;

    // Registries for neural objects
    std::map<uint64_t, std::shared_ptr<Neuron>> neuronRegistry_;
    std::map<uint64_t, std::shared_ptr<Axon>> axonRegistry_;
    std::map<uint64_t, std::shared_ptr<Synapse>> synapseRegistry_;
    std::map<uint64_t, std::shared_ptr<Dendrite>> dendriteRegistry_;

    // Reverse index: dendrite ID -> list of synapses targeting that dendrite
    // This enables O(1) lookup of synapses by dendrite instead of O(n) iteration
    std::map<uint64_t, std::vector<std::shared_ptr<Synapse>>> dendriteToSynapsesIndex_;

    // Mutexes for thread-safe access
    mutable std::mutex neuronMutex_;
    mutable std::mutex axonMutex_;
    mutable std::mutex synapseMutex_;
    mutable std::mutex dendriteMutex_;

    // STDP learning parameters
    double stdpAPlus_;      ///< LTP amplitude (default: 0.01)
    double stdpAMinus_;     ///< LTD amplitude (default: 0.012)
    double stdpTauPlus_;    ///< LTP time constant in ms (default: 20.0)
    double stdpTauMinus_;   ///< LTD time constant in ms (default: 20.0)

    // Activity monitoring (optional)
    class ActivityMonitor* activityMonitor_;  ///< Optional activity monitor for recording
    class RecordingManager* recordingManager_;  ///< Optional recording manager for direct recording
};

} // namespace snnfw

#endif // SNNFW_NETWORK_PROPAGATOR_H

