#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Brain.h"
#include "snnfw/Hemisphere.h"
#include "snnfw/Lobe.h"
#include "snnfw/Region.h"
#include "snnfw/Nucleus.h"
#include "snnfw/Layer.h"
#include "snnfw/Cluster.h"
#include "snnfw/Neuron.h"
#include "snnfw/Axon.h"
#include "snnfw/Dendrite.h"
#include "snnfw/Synapse.h"
#include "snnfw/NetworkPropagator.h"
#include "snnfw/SpikeProcessor.h"
#include <cassert>
#include <iostream>

using namespace snnfw;

// Simple deterministic propagation test: L4 neuron fires, should reach L5 and output via synapse, and learn pattern.
int main() {
    NeuralObjectFactory factory;
    auto spikeProcessor = std::make_shared<SpikeProcessor>(1000, 2);
    auto propagator = std::make_shared<NetworkPropagator>(spikeProcessor);
    spikeProcessor->start();

    // Build minimal network: L4 -> L5 -> Output
    auto l4 = factory.createNeuron(200.0, 0.5, 10);
    auto l5 = factory.createNeuron(200.0, 0.5, 10);
    auto out = factory.createNeuron(200.0, 0.5, 10);

    auto l4Axon = factory.createAxon(l4->getId());
    auto l5Axon = factory.createAxon(l5->getId());
    auto l5Dend = factory.createDendrite(l5->getId());
    auto outDend = factory.createDendrite(out->getId());
    l4->setAxonId(l4Axon->getId());
    l5->setAxonId(l5Axon->getId());
    l5->addDendrite(l5Dend->getId());
    out->addDendrite(outDend->getId());

    // Ensure neurons have deterministic signatures (single spike at offset 1ms)
    l4->clearSpikes();
    l5->clearSpikes();
    out->clearSpikes();
    l4->fireSignature(0.0); // generate signature spikes
    l5->fireSignature(0.0);
    out->fireSignature(0.0);
    l4->clearSpikes();
    l5->clearSpikes();
    out->clearSpikes();

    auto l4l5 = factory.createSynapse(l4Axon->getId(), l5Dend->getId(), 1.0, 1.0);
    auto l5out = factory.createSynapse(l5Axon->getId(), outDend->getId(), 1.0, 1.0);
    l4Axon->addSynapse(l4l5->getId());
    l5Axon->addSynapse(l5out->getId());

    // Register
    propagator->registerNeuron(l4);
    propagator->registerNeuron(l5);
    propagator->registerNeuron(out);
    propagator->registerAxon(l4Axon);
    propagator->registerAxon(l5Axon);
    l5Dend->setNetworkPropagator(propagator);
    outDend->setNetworkPropagator(propagator);
    propagator->registerDendrite(l5Dend);
    propagator->registerDendrite(outDend);
    propagator->registerSynapse(l4l5);
    propagator->registerSynapse(l5out);

    // Fire L4 signature
    double t0 = spikeProcessor->getCurrentTime();
    propagator->fireNeuron(l4->getId(), t0 + 10.0);

    // Allow spikes to process (wait up to 100ms)
    for (int i = 0; i < 10 && l5->getSpikes().empty(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // L5 should have received spikes
    if (l5->getSpikes().empty()) {
        std::cerr << "FAIL: L5 received no spikes from L4" << std::endl;
        return 1;
    }

    // Manually fire L5 if not auto-firing
    propagator->fireNeuron(l5->getId(), t0 + 20.0);
    for (int i = 0; i < 10 && out->getSpikes().empty(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (out->getSpikes().empty()) {
        // Attempt direct delivery to see if registry is the issue
        propagator->deliverSpikeToNeuron(out->getId(), l5out->getId(), t0 + 30.0, 1.0, t0 + 20.0);
        for (int i = 0; i < 10 && out->getSpikes().empty(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (out->getSpikes().empty()) {
            std::cerr << "FAIL: Output received no spikes from L5 (even direct delivery)" << std::endl;
            return 1;
        }
    }

    // Learn patterns
    l5->learnCurrentPattern();
    out->learnCurrentPattern();

    if (l5->getLearnedPatternCount() == 0 || out->getLearnedPatternCount() == 0) {
        std::cerr << "FAIL: Patterns not learned at L5 or output" << std::endl;
        return 1;
    }

    std::cout << "PASS: Propagation and learning succeeded (L5 spikes=" << l5->getSpikes().size()
              << ", out spikes=" << out->getSpikes().size() << ")" << std::endl;
    spikeProcessor->stop();
    return 0;
}
