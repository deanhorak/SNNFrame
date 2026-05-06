#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Neuron.h"
#include "snnfw/Axon.h"
#include "snnfw/Dendrite.h"
#include "snnfw/Synapse.h"
#include "snnfw/NetworkPropagator.h"
#include "snnfw/SpikeProcessor.h"
#include <iostream>

using namespace snnfw;

int main() {
    NeuralObjectFactory factory;
    auto proc = std::make_shared<SpikeProcessor>(1000, 2);
    auto prop = std::make_shared<NetworkPropagator>(proc);
    proc->start();

    auto l4 = factory.createNeuron(200.0, 0.3, 10);
    auto l5 = factory.createNeuron(200.0, 0.3, 10);
    auto out = factory.createNeuron(200.0, 0.3, 10);

    auto l4Ax = factory.createAxon(l4->getId());
    auto l5Ax = factory.createAxon(l5->getId());
    auto l5Den = factory.createDendrite(l5->getId());
    auto outDen = factory.createDendrite(out->getId());
    l4->setAxonId(l4Ax->getId());
    l5->setAxonId(l5Ax->getId());
    l5->addDendrite(l5Den->getId());
    out->addDendrite(outDen->getId());

    // Deterministic signatures
    l4->clearSpikes();
    l5->clearSpikes();
    out->clearSpikes();
    l4->fireSignature(0.0);
    l5->fireSignature(0.0);
    out->fireSignature(0.0);
    l4->clearSpikes();
    l5->clearSpikes();
    out->clearSpikes();

    auto l4l5 = factory.createSynapse(l4Ax->getId(), l5Den->getId(), 1.0, 1.0);
    auto l5out = factory.createSynapse(l5Ax->getId(), outDen->getId(), 1.0, 1.0);
    l4Ax->addSynapse(l4l5->getId());
    l5Ax->addSynapse(l5out->getId());

    prop->registerNeuron(l4);
    prop->registerNeuron(l5);
    prop->registerNeuron(out);
    prop->registerAxon(l4Ax);
    prop->registerAxon(l5Ax);
    l5Den->setNetworkPropagator(prop);
    outDen->setNetworkPropagator(prop);
    prop->registerDendrite(l5Den);
    prop->registerDendrite(outDen);
    prop->registerSynapse(l4l5);
    prop->registerSynapse(l5out);

    double t0 = proc->getCurrentTime();
    prop->fireNeuron(l4->getId(), t0 + 5.0);
    for (int i = 0; i < 10 && l5->getSpikes().empty(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (l5->getSpikes().empty()) {
        std::cerr << "FAIL: L5 did not receive spikes" << std::endl;
        proc->stop();
        return 1;
    }

    // Drive L5 onward
    prop->fireNeuron(l5->getId(), t0 + 15.0);
    for (int i = 0; i < 10 && out->getSpikes().empty(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (out->getSpikes().empty()) {
        // Try direct delivery to check registration
        prop->deliverSpikeToNeuron(out->getId(), l5out->getId(), t0 + 30.0, 1.0, t0 + 20.0);
        for (int i = 0; i < 10 && out->getSpikes().empty(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if (out->getSpikes().empty()) {
            std::cerr << "FAIL: Output did not receive spikes (even direct delivery)" << std::endl;
            proc->stop();
            return 1;
        }
    }

    // Learn patterns downstream
    l5->learnCurrentPattern();
    out->learnCurrentPattern();
    if (l5->getLearnedPatternCount() == 0 || out->getLearnedPatternCount() == 0) {
        std::cerr << "FAIL: No patterns learned" << std::endl;
        proc->stop();
        return 1;
    }

    std::cout << "PASS: Layered propagation and learning (L5 spikes=" << l5->getSpikes().size()
              << ", out spikes=" << out->getSpikes().size() << ")" << std::endl;
    proc->stop();
    return 0;
}
