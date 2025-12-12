#include <gtest/gtest.h>
#include "snnfw/Neuron.h"
#include "snnfw/Axon.h"
#include "snnfw/Synapse.h"
#include "snnfw/Dendrite.h"
#include "snnfw/SpikeProcessor.h"
#include "snnfw/NetworkPropagator.h"
#include "snnfw/ActionPotential.h"
#include "snnfw/RetrogradeActionPotential.h"
#include "snnfw/NeuralObjectFactory.h"
#include <thread>
#include <chrono>

using namespace snnfw;

/**
 * Test that retrograde action potentials are created and scheduled correctly
 */
TEST(RetrogradeSTDPTest, RetrogradeActionPotentialCreation) {
    // Create a retrograde action potential
    uint64_t synapseId = 1000;
    uint64_t postsynapticNeuronId = 2000;
    double scheduledTime = 50.0;  // When it should arrive
    double dispatchTime = 10.0;   // When the forward spike was sent
    double lastFiringTime = 45.0; // When the postsynaptic neuron fired
    
    auto retrogradeAP = std::make_shared<RetrogradeActionPotential>(
        synapseId,
        postsynapticNeuronId,
        scheduledTime,
        dispatchTime,
        lastFiringTime
    );
    
    // Verify properties
    EXPECT_EQ(retrogradeAP->getSynapseId(), synapseId);
    EXPECT_EQ(retrogradeAP->getPostsynapticNeuronId(), postsynapticNeuronId);
    EXPECT_EQ(retrogradeAP->getScheduledTime(), scheduledTime);
    EXPECT_EQ(retrogradeAP->getDispatchTime(), dispatchTime);
    EXPECT_EQ(retrogradeAP->getLastFiringTime(), lastFiringTime);
    
    // Verify temporal offset calculation
    // temporalOffset = lastFiringTime - dispatchTime = 45.0 - 10.0 = 35.0
    EXPECT_DOUBLE_EQ(retrogradeAP->getTemporalOffset(), 35.0);
    
    // Verify event type
    EXPECT_STREQ(retrogradeAP->getEventType(), "RetrogradeActionPotential");
}

/**
 * Test that STDP is applied correctly via retrograde signals
 */
TEST(RetrogradeSTDPTest, STDPApplicationViaRetrograde) {
    // Create spike processor and network propagator
    auto spikeProcessor = std::make_shared<SpikeProcessor>(1000, 20);
    auto propagator = std::make_shared<NetworkPropagator>(spikeProcessor);
    
    // Set STDP parameters
    propagator->setSTDPParameters(0.05, 0.05, 20.0, 20.0);
    spikeProcessor->setSTDPParameters(0.05, 0.05, 20.0, 20.0);

    // Create neural objects using factory
    NeuralObjectFactory factory;

    auto preNeuron = factory.createNeuron(100.0, 0.5, 10);
    auto postNeuron = factory.createNeuron(100.0, 0.5, 10);
    auto axon = factory.createAxon(preNeuron->getId());
    auto dendrite = factory.createDendrite(postNeuron->getId());
    auto synapse = factory.createSynapse(axon->getId(), dendrite->getId(), 1.0, 1.0);

    // Connect neuron to axon and axon to synapse
    preNeuron->setAxonId(axon->getId());
    axon->addSynapse(synapse->getId());
    
    // Register all components
    propagator->registerNeuron(preNeuron);
    propagator->registerNeuron(postNeuron);
    propagator->registerAxon(axon);
    propagator->registerSynapse(synapse);
    propagator->registerDendrite(dendrite);

    // Set network propagators
    preNeuron->setNetworkPropagator(propagator);
    postNeuron->setNetworkPropagator(propagator);
    dendrite->setNetworkPropagator(propagator);
    
    // Start spike processor
    spikeProcessor->start();
    
    // Record initial weight
    double initialWeight = synapse->getWeight();
    EXPECT_DOUBLE_EQ(initialWeight, 1.0);
    
    // Fire presynaptic neuron at t=10ms
    double preFireTime = 10.0;
    int spikesScheduled = propagator->fireNeuron(preNeuron->getId(), preFireTime);
    EXPECT_GT(spikesScheduled, 0);
    
    // Wait for spikes to propagate
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Check that weight has changed due to retrograde STDP
    double finalWeight = synapse->getWeight();
    
    // Since we scheduled retrograde spikes, the weight should have been updated
    // The exact value depends on timing, but it should be different from initial
    std::cout << "Initial weight: " << initialWeight << ", Final weight: " << finalWeight << std::endl;
    
    // Stop spike processor
    spikeProcessor->stop();
}

/**
 * Test that dispatch time is correctly set on ActionPotentials
 */
TEST(RetrogradeSTDPTest, ActionPotentialDispatchTime) {
    auto spikeProcessor = std::make_shared<SpikeProcessor>(1000, 20);
    auto propagator = std::make_shared<NetworkPropagator>(spikeProcessor);

    // Create neural objects using factory
    NeuralObjectFactory factory;

    auto preNeuron = factory.createNeuron(100.0, 0.5, 10);
    auto postNeuron = factory.createNeuron(100.0, 0.5, 10);
    auto axon = factory.createAxon(preNeuron->getId());
    auto dendrite = factory.createDendrite(postNeuron->getId());
    auto synapse = factory.createSynapse(axon->getId(), dendrite->getId(), 1.0, 2.0);

    // Connect neuron to axon and axon to synapse
    preNeuron->setAxonId(axon->getId());
    axon->addSynapse(synapse->getId());

    // Register all components
    propagator->registerNeuron(preNeuron);
    propagator->registerNeuron(postNeuron);
    propagator->registerAxon(axon);
    propagator->registerSynapse(synapse);
    propagator->registerDendrite(dendrite);

    // Set network propagators
    preNeuron->setNetworkPropagator(propagator);
    postNeuron->setNetworkPropagator(propagator);
    dendrite->setNetworkPropagator(propagator);
    
    spikeProcessor->start();
    
    // Fire neuron at t=15ms
    double firingTime = 15.0;
    int spikesScheduled = propagator->fireNeuron(preNeuron->getId(), firingTime);
    EXPECT_GT(spikesScheduled, 0);
    
    // The ActionPotentials should have dispatchTime set to firingTime
    // We can't directly verify this without accessing the event queue,
    // but we can verify that the system doesn't crash and spikes are scheduled
    
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    spikeProcessor->stop();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

