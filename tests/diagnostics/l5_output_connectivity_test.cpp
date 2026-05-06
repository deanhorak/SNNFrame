#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Neuron.h"
#include "snnfw/Axon.h"
#include "snnfw/Dendrite.h"
#include "snnfw/Synapse.h"
#include <iostream>
#include <random>
#include <vector>

// This test mirrors the L5 -> output wiring with sparse probabilities and ensures no L5 axon is left without a synapse.
int main() {
    using namespace snnfw;

    constexpr int NUM_LETTERS = 26;
    const int neuronsPerOutputClass = 3;
    const double outputConnectProb = 0.002;
    const double outputInitialWeight = 0.002;
    const double maxWeight = 0.5;

    NeuralObjectFactory factory;
    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis(0.0, 1.0);

    // Build a handful of L5 neurons
    std::vector<std::shared_ptr<Neuron>> l5Neurons;
    for (int i = 0; i < 50; ++i) {
        auto n = factory.createNeuron(200.0, 0.5, 10);
        auto ax = factory.createAxon(n->getId());
        auto den = factory.createDendrite(n->getId());
        n->setAxonId(ax->getId());
        n->addDendrite(den->getId());
        l5Neurons.push_back(n);
    }

    // Build output populations
    std::vector<std::vector<std::shared_ptr<Neuron>>> outputPop(NUM_LETTERS);
    for (int i = 0; i < NUM_LETTERS; ++i) {
        for (int j = 0; j < neuronsPerOutputClass; ++j) {
            auto n = factory.createNeuron(200.0, 0.5, 10);
            auto ax = factory.createAxon(n->getId());
            auto den = factory.createDendrite(n->getId());
            n->setAxonId(ax->getId());
            n->addDendrite(den->getId());
            outputPop[i].push_back(n);
        }
    }

    // Wire L5 -> output with sparse prob
    int outputSynapses = 0;
    std::vector<std::shared_ptr<Synapse>> allSyn;
    for (auto& l5 : l5Neurons) {
        auto ax = factory.createAxon(l5->getId()); // stand-in axon object to collect synapses
        for (int cls = 0; cls < NUM_LETTERS; ++cls) {
            for (auto& out : outputPop[cls]) {
                if (dis(gen) < outputConnectProb) {
                    auto syn = factory.createSynapse(ax->getId(), out->getDendriteIds()[0], outputInitialWeight, maxWeight);
                    ax->addSynapse(syn->getId());
                    allSyn.push_back(syn);
                    outputSynapses++;
                }
            }
        }
        // Safeguard: ensure at least one synapse
        if (ax->getSynapseCount() == 0) {
            int cls = gen() % NUM_LETTERS;
            if (!outputPop[cls].empty()) {
                int nidx = gen() % outputPop[cls].size();
                auto& out = outputPop[cls][nidx];
                auto syn = factory.createSynapse(ax->getId(), out->getDendriteIds()[0], outputInitialWeight, maxWeight);
                ax->addSynapse(syn->getId());
                allSyn.push_back(syn);
                outputSynapses++;
            }
        }
        if (ax->getSynapseCount() == 0) {
            std::cerr << "FAIL: L5 axon " << ax->getId() << " has zero synapses after safeguard\n";
            return 1;
        }
    }

    std::cout << "PASS: L5->output wiring created " << outputSynapses << " synapses, no empty axons" << std::endl;
    return 0;
}
