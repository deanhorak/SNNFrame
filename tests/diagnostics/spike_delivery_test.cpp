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

    auto n1 = factory.createNeuron(200.0, 0.5, 10);
    auto n2 = factory.createNeuron(200.0, 0.5, 10);
    auto ax = factory.createAxon(n1->getId());
    auto den = factory.createDendrite(n2->getId());
    n1->setAxonId(ax->getId());
    n2->addDendrite(den->getId());
    auto syn = factory.createSynapse(ax->getId(), den->getId(), 1.0, 1.0);
    ax->addSynapse(syn->getId());

    prop->registerNeuron(n1);
    prop->registerNeuron(n2);
    prop->registerAxon(ax);
    den->setNetworkPropagator(prop);
    prop->registerDendrite(den);
    prop->registerSynapse(syn);

    double t0 = proc->getCurrentTime();
    prop->fireNeuron(n1->getId(), t0 + 5.0);
    // Allow background processor to deliver spikes (wait up to 100ms)
    for (int i = 0; i < 10 && n2->getSpikes().empty(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (n2->getSpikes().empty()) {
        std::cerr << "FAIL: spike not delivered to postsynaptic neuron" << std::endl;
        proc->stop();
        return 1;
    }

    std::cout << "PASS: spike delivered (count=" << n2->getSpikes().size() << ")" << std::endl;
    proc->stop();
    return 0;
}
