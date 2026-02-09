#include "snnfw/Neuron.h"
#include <iostream>

using namespace snnfw;

int main() {
    Neuron neuron(200.0, 0.999, 5);

    // Learn base pattern
    neuron.insertSpike(5.0);
    neuron.insertSpike(10.0);
    neuron.learnCurrentPattern();
    neuron.clearSpikes();

    // Similar pattern should fire
    neuron.insertSpike(5.0);
    neuron.insertSpike(10.0);
    if (!neuron.checkShouldFire()) {
        std::cerr << "FAIL: did not fire on similar pattern" << std::endl;
        return 1;
    }

    neuron.clearSpikes();
    // Dissimilar pattern should not fire
    neuron.insertSpike(180.0);
    neuron.insertSpike(190.0);
    if (neuron.checkShouldFire()) {
        std::cerr << "FAIL: fired on dissimilar pattern" << std::endl;
        return 1;
    }

    std::cout << "PASS: pattern learning and similarity thresholds OK" << std::endl;
    return 0;
}
