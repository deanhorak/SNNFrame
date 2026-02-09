#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <set>
#include <string>
#include <vector>

// This diagnostic checks that the retinal encoding (pixel -> spike time)
// produces temporally distinct spike patterns for different local stimuli.
// We mirror the experiment's encoding: pixels above a threshold fire once
// at baseTime + (1 - norm) * 5ms. The goal is to ensure each receptive
// field yields unique temporal codes that downstream layers can learn.

struct Stimulus {
    std::string name;
    std::array<double, 16> intensities; // 4x4 patch, row-major, 0..1
};

struct EncodedPattern {
    std::set<int> active;
    std::vector<double> spikeTimes; // size 16, -1 for silent
};

EncodedPattern encode(const Stimulus& stim, double baseTime) {
    EncodedPattern out;
    out.spikeTimes.assign(16, -1.0);

    for (int i = 0; i < 16; ++i) {
        double norm = stim.intensities[i];
        if (norm > 0.25) { // same threshold used in training path
            double fireT = baseTime + (1.0 - norm) * 5.0;
            out.spikeTimes[i] = fireT;
            out.active.insert(i);
        }
    }
    return out;
}

double jaccard(const std::set<int>& a, const std::set<int>& b) {
    if (a.empty() && b.empty()) return 1.0;
    if (a.empty() || b.empty()) return 0.0;
    std::set<int> inter;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                          std::inserter(inter, inter.begin()));
    std::set<int> uni;
    std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                   std::inserter(uni, uni.begin()));
    return static_cast<double>(inter.size()) / static_cast<double>(uni.size());
}

// Minimum temporal separation between any two spikes in a pattern
double min_time_delta(const EncodedPattern& p) {
    std::vector<double> times;
    for (double t : p.spikeTimes) {
        if (t >= 0.0) times.push_back(t);
    }
    if (times.size() < 2) return std::numeric_limits<double>::infinity();
    std::sort(times.begin(), times.end());
    double minDelta = std::numeric_limits<double>::infinity();
    for (size_t i = 1; i < times.size(); ++i) {
        minDelta = std::min(minDelta, times[i] - times[i - 1]);
    }
    return minDelta;
}

int main() {
    const double baseTime = 100.0;

    // Four distinct stimuli within the same 4x4 receptive field.
    std::vector<Stimulus> stimuli = {
        {"horizontal_top",
         {0.90, 0.80, 0.70, 0.60,
          0.05, 0.05, 0.05, 0.05,
          0.05, 0.05, 0.05, 0.05,
          0.05, 0.05, 0.05, 0.05}},
        {"diagonal",
         {0.85, 0.05, 0.05, 0.05,
          0.05, 0.65, 0.05, 0.05,
          0.05, 0.05, 0.45, 0.05,
          0.05, 0.05, 0.05, 0.35}},
        {"center_spot",
         {0.05, 0.05, 0.05, 0.05,
          0.05, 0.95, 0.85, 0.05,
          0.05, 0.80, 0.70, 0.05,
          0.05, 0.05, 0.05, 0.05}},
        {"checkerboard",
         {0.90, 0.10, 0.80, 0.10,
          0.10, 0.70, 0.10, 0.60,
          0.85, 0.10, 0.75, 0.10,
          0.10, 0.65, 0.10, 0.55}}
    };

    std::vector<EncodedPattern> patterns;
    patterns.reserve(stimuli.size());
    for (const auto& s : stimuli) {
        patterns.push_back(encode(s, baseTime));
    }

    // 1) Each stimulus should have at least one active neuron
    for (size_t i = 0; i < patterns.size(); ++i) {
        if (patterns[i].active.empty()) {
            std::cerr << "FAIL: stimulus '" << stimuli[i].name
                      << "' produced no spikes" << std::endl;
            return 1;
        }
    }

    // 2) Pairwise Jaccard overlap of active sets should be low (< 0.6) to ensure spatial distinctness.
    for (size_t i = 0; i < patterns.size(); ++i) {
        for (size_t j = i + 1; j < patterns.size(); ++j) {
            double jac = jaccard(patterns[i].active, patterns[j].active);
            if (jac >= 0.6) {
                std::cerr << "FAIL: stimuli '" << stimuli[i].name << "' and '"
                          << stimuli[j].name << "' overlap too much (Jaccard="
                          << jac << ")" << std::endl;
                return 1;
            }
        }
    }

    // 3) Within each pattern, spike times must be temporally separated (>=0.2ms).
    for (size_t i = 0; i < patterns.size(); ++i) {
        double minDelta = min_time_delta(patterns[i]);
        if (minDelta < 0.2) {
            std::cerr << "FAIL: stimulus '" << stimuli[i].name
                      << "' has insufficient temporal separation (min delta="
                      << minDelta << "ms)" << std::endl;
            return 1;
        }
    }

    std::cout << "PASS: retinal encoding produces distinct spatial and temporal spike patterns" << std::endl;
    return 0;
}
