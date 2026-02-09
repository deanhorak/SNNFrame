#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Neuron.h"
#include "snnfw/Axon.h"
#include "snnfw/Dendrite.h"
#include "snnfw/Synapse.h"
#include "snnfw/NetworkPropagator.h"
#include "snnfw/SpikeProcessor.h"
#include <chrono>
#include <iostream>
#include <thread>

// This test validates that firing an L4 neuron delivers spikes to L5 even if the fire time is in the past.
// It guards against the prior regression where scheduleSpike dropped past-time events.
int main() {
    using namespace snnfw;

    NeuralObjectFactory factory;
    auto proc = std::make_shared<SpikeProcessor>(1000, 2);
    auto prop = std::make_shared<NetworkPropagator>(proc);
    proc->start();

    auto l4 = factory.createNeuron(200.0, 0.5, 10);
    auto l5 = factory.createNeuron(200.0, 0.5, 10);
    auto ax = factory.createAxon(l4->getId());
    auto den = factory.createDendrite(l5->getId());
    l4->setAxonId(ax->getId());
    l5->addDendrite(den->getId());
    auto syn = factory.createSynapse(ax->getId(), den->getId(), 1.0, 1.0);
    ax->addSynapse(syn->getId());

    prop->registerNeuron(l4);
    prop->registerNeuron(l5);
    prop->registerAxon(ax);
    den->setNetworkPropagator(prop);
    prop->registerDendrite(den);
    prop->registerSynapse(syn);

    double t0 = proc->getCurrentTime();
    // Fire in the past to ensure the clamping logic still delivers
    prop->fireNeuron(l4->getId(), t0 - 5.0);

    // Allow background processor to deliver spikes (wait up to 100ms)
    for (int i = 0; i < 10 && l5->getSpikes().empty(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (l5->getSpikes().empty()) {
        std::cerr << "FAIL: L4->L5 spike not delivered when firing time is in the past" << std::endl;
        proc->stop();
        return 1;
    }

    std::cout << "PASS: L4->L5 spike delivered (count=" << l5->getSpikes().size() << ")" << std::endl;
    proc->stop();
    return 0;
}
