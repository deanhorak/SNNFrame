#include "snnfw/Neuron.h"
#include <iostream>

using namespace snnfw;

int main() {
    // Low threshold neuron to ensure firing
    Neuron neuron(200.0, 0.2, 10);

    // Learn a simple pattern with explicit time offsets
    neuron.insertSpike(10.0);
    neuron.insertSpike(20.0);
    neuron.learnCurrentPattern();
    neuron.clearSpikes();

    // Present similar pattern
    neuron.insertSpike(10.0);
    neuron.insertSpike(20.0);

    if (!neuron.checkShouldFire()) {
        std::cerr << "FAIL: neuron did not signal fire on matching pattern" << std::endl;
        return 1;
    }

    neuron.fireAndAcknowledge(30.0);
    if (neuron.getSpikes().empty()) {
        std::cerr << "FAIL: neuron did not emit spike after fireAndAcknowledge" << std::endl;
        return 1;
    }

    std::cout << "PASS: postsynaptic firing rule and emit OK" << std::endl;
    return 0;
}
