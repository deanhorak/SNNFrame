#include <gtest/gtest.h>
#include "snnfw/Synapse.h"
#include "snnfw/Axon.h"
#include "snnfw/Dendrite.h"
#include "snnfw/Neuron.h"
#include "snnfw/ActionPotential.h"
#include "snnfw/SpikeProcessor.h"
#include "snnfw/NetworkPropagator.h"
#include "snnfw/NeuralObjectFactory.h"
#include <thread>
#include <chrono>

using namespace snnfw;

// ============================================================================
// Test 1: Synapse Weight Updates
// ============================================================================

TEST(SynapseTest, WeightGetterSetter) {
    Synapse synapse(1, 2, 1.0, 1.0, 100);
    
    EXPECT_EQ(synapse.getAxonId(), 1);
    EXPECT_EQ(synapse.getDendriteId(), 2);
    EXPECT_DOUBLE_EQ(synapse.getWeight(), 1.0);
    
    synapse.setWeight(1.5);
    EXPECT_DOUBLE_EQ(synapse.getWeight(), 1.5);
    
    synapse.modifyWeight(0.3);
    EXPECT_DOUBLE_EQ(synapse.getWeight(), 1.8);
    
    synapse.modifyWeight(-0.5);
    EXPECT_DOUBLE_EQ(synapse.getWeight(), 1.3);
}

TEST(SynapseTest, DelayGetterSetter) {
    Synapse synapse(1, 2, 1.0, 2.5, 100);
    
    EXPECT_DOUBLE_EQ(synapse.getDelay(), 2.5);
    
    synapse.setDelay(5.0);
    EXPECT_DOUBLE_EQ(synapse.getDelay(), 5.0);
}

// ============================================================================
// Test 2: ActionPotential Creation
// ============================================================================

TEST(ActionPotentialTest, CreationAndProperties) {
    auto ap = std::make_shared<ActionPotential>(
        123,    // synapseId
        456,    // dendriteId
        10.5,   // scheduledTime
        0.8     // amplitude
    );
    
    EXPECT_EQ(ap->getSynapseId(), 123);
    EXPECT_EQ(ap->getDendriteId(), 456);
    EXPECT_DOUBLE_EQ(ap->getScheduledTime(), 10.5);
    EXPECT_DOUBLE_EQ(ap->getAmplitude(), 0.8);
    EXPECT_STREQ(ap->getEventType(), "ActionPotential");
}

// ============================================================================
// Test 3: Neuron Spike Tracking for STDP
// ============================================================================

TEST(NeuronTest, IncomingSpikeTracking) {
    auto neuron = std::make_shared<Neuron>(200.0, 0.7, 20, 1);

    // Record incoming spikes
    neuron->recordIncomingSpike(100, 10.0);
    neuron->recordIncomingSpike(101, 15.0);
    neuron->recordIncomingSpike(102, 20.0);

    // Verify spikes are tracked (we can't directly access the deque,
    // but we can test via fireAndAcknowledge)
    // This will be tested in the integration tests
}

TEST(NeuronTest, OldSpikeClearing) {
    auto neuron = std::make_shared<Neuron>(200.0, 0.7, 20, 1);

    // Record spikes at different times
    neuron->recordIncomingSpike(100, 10.0);
    neuron->recordIncomingSpike(101, 50.0);
    neuron->recordIncomingSpike(102, 100.0);
    neuron->recordIncomingSpike(103, 250.0);  // This should trigger clearing

    // Spikes older than 200ms from 250.0 should be cleared
    // (i.e., spikes before 50.0 should be gone)
    // We'll verify this in integration tests
}

// ============================================================================
// Test 4: STDP Weight Updates
// ============================================================================

TEST(STDPTest, LongTermPotentiation) {
    auto spikeProcessor = std::make_shared<SpikeProcessor>(1);
    auto propagator = std::make_shared<NetworkPropagator>(spikeProcessor);
    
    // Set STDP parameters
    propagator->setSTDPParameters(0.01, 0.012, 20.0, 20.0);
    
    // Create a synapse with initial weight 1.0
    auto synapse = std::make_shared<Synapse>(1, 2, 1.0, 1.0, 100);
    propagator->registerSynapse(synapse);
    
    double initialWeight = synapse->getWeight();
    
    // Apply LTP: presynaptic spike arrives BEFORE postsynaptic spike
    // timeDifference = t_post - t_pre = 10ms (positive)
    propagator->applySTDP(100, 10.0);
    
    // Weight should increase
    EXPECT_GT(synapse->getWeight(), initialWeight);
}

TEST(STDPTest, LongTermDepression) {
    auto spikeProcessor = std::make_shared<SpikeProcessor>(1);
    auto propagator = std::make_shared<NetworkPropagator>(spikeProcessor);
    
    propagator->setSTDPParameters(0.01, 0.012, 20.0, 20.0);
    
    auto synapse = std::make_shared<Synapse>(1, 2, 1.0, 1.0, 100);
    propagator->registerSynapse(synapse);
    
    double initialWeight = synapse->getWeight();
    
    // Apply LTD: presynaptic spike arrives AFTER postsynaptic spike
    // timeDifference = t_post - t_pre = -10ms (negative)
    propagator->applySTDP(100, -10.0);
    
    // Weight should decrease
    EXPECT_LT(synapse->getWeight(), initialWeight);
}

TEST(STDPTest, WeightClamping) {
    auto spikeProcessor = std::make_shared<SpikeProcessor>(1);
    auto propagator = std::make_shared<NetworkPropagator>(spikeProcessor);
    
    propagator->setSTDPParameters(0.5, 0.5, 20.0, 20.0);  // Large learning rates
    
    // Test upper bound clamping
    auto synapse1 = std::make_shared<Synapse>(1, 2, 1.9, 1.0, 100);
    propagator->registerSynapse(synapse1);
    
    // Apply strong LTP multiple times
    for (int i = 0; i < 10; ++i) {
        propagator->applySTDP(100, 5.0);
    }
    
    // Weight should be clamped at 2.0
    EXPECT_LE(synapse1->getWeight(), 2.0);
    
    // Test lower bound clamping
    auto synapse2 = std::make_shared<Synapse>(3, 4, 0.1, 1.0, 101);
    propagator->registerSynapse(synapse2);
    
    // Apply strong LTD multiple times
    for (int i = 0; i < 10; ++i) {
        propagator->applySTDP(101, -5.0);
    }
    
    // Weight should be clamped at 0.0
    EXPECT_GE(synapse2->getWeight(), 0.0);
}

TEST(STDPTest, ExponentialDecay) {
    auto spikeProcessor = std::make_shared<SpikeProcessor>(1);
    auto propagator = std::make_shared<NetworkPropagator>(spikeProcessor);
    
    propagator->setSTDPParameters(0.01, 0.012, 20.0, 20.0);
    
    auto synapse1 = std::make_shared<Synapse>(1, 2, 1.0, 1.0, 100);
    auto synapse2 = std::make_shared<Synapse>(3, 4, 1.0, 1.0, 101);
    propagator->registerSynapse(synapse1);
    propagator->registerSynapse(synapse2);
    
    // Apply STDP with different time differences
    propagator->applySTDP(100, 5.0);   // Small time difference
    propagator->applySTDP(101, 50.0);  // Large time difference
    
    double weight1 = synapse1->getWeight();
    double weight2 = synapse2->getWeight();
    
    // Smaller time difference should produce larger weight change
    EXPECT_GT(weight1 - 1.0, weight2 - 1.0);
}

// ============================================================================
// Test 5: Integration Test - Single Synapse Spike Propagation
// ============================================================================

TEST(IntegrationTest, SingleSynapseSpikePropagation) {
    // Create spike processor and network propagator with 200ms buffer (to accommodate temporal signatures)
    auto spikeProcessor = std::make_shared<SpikeProcessor>(200);
    auto propagator = std::make_shared<NetworkPropagator>(spikeProcessor);

    // Create neural objects using factory
    NeuralObjectFactory factory;

    auto preNeuron = factory.createNeuron(200.0, 0.7, 20);
    auto postNeuron = factory.createNeuron(200.0, 0.7, 20);
    auto axon = factory.createAxon(preNeuron->getId());
    auto dendrite = factory.createDendrite(postNeuron->getId());
    auto synapse = factory.createSynapse(axon->getId(), dendrite->getId(), 1.0, 1.0);

    // Connect neuron to axon and axon to synapse
    preNeuron->setAxonId(axon->getId());
    axon->addSynapse(synapse->getId());

    // Register everything
    propagator->registerNeuron(preNeuron);
    propagator->registerNeuron(postNeuron);
    propagator->registerAxon(axon);
    propagator->registerSynapse(synapse);
    propagator->registerDendrite(dendrite);

    preNeuron->setNetworkPropagator(propagator);
    postNeuron->setNetworkPropagator(propagator);
    dendrite->setNetworkPropagator(propagator);

    spikeProcessor->registerDendrite(dendrite);
    spikeProcessor->start();

    // Give the spike processor time to start
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Fire the presynaptic neuron
    double currentTime = spikeProcessor->getCurrentTime();
    double firingTime = currentTime + 5.0;

    std::cout << "DEBUG: Current time: " << currentTime << std::endl;
    std::cout << "DEBUG: Firing time: " << firingTime << std::endl;
    std::cout << "DEBUG: Axon ID: " << axon->getId() << std::endl;
    std::cout << "DEBUG: Synapse count: " << axon->getSynapseCount() << std::endl;
    std::cout << "DEBUG: Neuron axon ID: " << preNeuron->getAxonId() << std::endl;

    int spikesScheduled = propagator->fireNeuron(preNeuron->getId(), firingTime);

    std::cout << "DEBUG: Spikes scheduled: " << spikesScheduled << std::endl;

    // Wait for spike to propagate
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Check that postsynaptic neuron received the spike
    auto spikes = postNeuron->getSpikes();
    std::cout << "DEBUG: Postsynaptic spikes received: " << spikes.size() << std::endl;

    EXPECT_GT(spikes.size(), 0);

    spikeProcessor->stop();
}

// ============================================================================
// Test 6: Integration Test - STDP Learning
// ============================================================================

TEST(IntegrationTest, STDPLearning) {
    auto spikeProcessor = std::make_shared<SpikeProcessor>(200);  // 200ms buffer (to accommodate temporal signatures)
    auto propagator = std::make_shared<NetworkPropagator>(spikeProcessor);
    propagator->setSTDPParameters(0.05, 0.05, 20.0, 20.0);

    NeuralObjectFactory factory;

    auto preNeuron = factory.createNeuron(200.0, 0.7, 20);
    auto postNeuron = factory.createNeuron(200.0, 0.7, 20);
    auto axon = factory.createAxon(preNeuron->getId());
    auto dendrite = factory.createDendrite(postNeuron->getId());
    auto synapse = factory.createSynapse(axon->getId(), dendrite->getId(), 1.0, 1.0);

    // Connect neuron to axon and axon to synapse
    preNeuron->setAxonId(axon->getId());
    axon->addSynapse(synapse->getId());

    propagator->registerNeuron(preNeuron);
    propagator->registerNeuron(postNeuron);
    propagator->registerAxon(axon);
    propagator->registerSynapse(synapse);
    propagator->registerDendrite(dendrite);

    preNeuron->setNetworkPropagator(propagator);
    postNeuron->setNetworkPropagator(propagator);
    dendrite->setNetworkPropagator(propagator);

    spikeProcessor->registerDendrite(dendrite);
    spikeProcessor->start();

    double initialWeight = synapse->getWeight();

    // Get temporal signature to calculate proper timing
    const auto& signature = preNeuron->getTemporalSignature();
    double firstSpikeOffset = signature.empty() ? 0.0 : signature.front();
    double lastSpikeOffset = signature.empty() ? 0.0 : signature.back();

    std::cout << "DEBUG: Presynaptic neuron temporal signature (" << signature.size() << " spikes): ";
    for (double offset : signature) {
        std::cout << offset << "ms ";
    }
    std::cout << std::endl;

    // Test LTP: pre fires, then post fires AFTER all presynaptic spikes arrive
    double currentTime = spikeProcessor->getCurrentTime();
    double preFiringTime = currentTime + 5.0;
    double synapticDelay = 1.0;  // From synapse creation
    double firstSpikeArrival = preFiringTime + synapticDelay + firstSpikeOffset;
    double lastSpikeArrival = preFiringTime + synapticDelay + lastSpikeOffset;
    double postFiringTime = lastSpikeArrival + 10.0;  // Fire 10ms after last presynaptic spike arrives

    std::cout << "DEBUG: Current time: " << currentTime << "ms" << std::endl;
    std::cout << "DEBUG: Pre fires at: " << preFiringTime << "ms" << std::endl;
    std::cout << "DEBUG: First spike arrives at: " << firstSpikeArrival << "ms" << std::endl;
    std::cout << "DEBUG: Last spike arrives at: " << lastSpikeArrival << "ms" << std::endl;
    std::cout << "DEBUG: Post fires at: " << postFiringTime << "ms" << std::endl;

    propagator->fireNeuron(preNeuron->getId(), preFiringTime);
    std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(lastSpikeArrival - currentTime + 20)));

    postNeuron->fireAndAcknowledge(postFiringTime);
    std::this_thread::sleep_for(std::chrono::milliseconds(30));  // Wait for STDP processing

    // Weight should have increased (all presynaptic spikes arrived before postsynaptic firing)
    std::cout << "DEBUG: Initial weight: " << initialWeight << ", Final weight: " << synapse->getWeight() << std::endl;
    EXPECT_GT(synapse->getWeight(), initialWeight);

    spikeProcessor->stop();
}

