#include "snnfw/Neuron.h"
#include <array>
#include <set>
#include <vector>
#include <iostream>
#include <cmath>

using namespace snnfw;

// Reuse globals from experiment for simplicity
constexpr int NUM_LETTERS = 26;
extern std::array<std::array<int, 1280>, NUM_LETTERS> letterL5Counts;
extern std::array<int, NUM_LETTERS> letterPatternCounts;
std::array<std::array<int, 1280>, NUM_LETTERS> letterL5Counts{};
std::array<int, NUM_LETTERS> letterPatternCounts{};

std::set<size_t> makePattern(std::initializer_list<size_t> idxs) {
    return std::set<size_t>(idxs.begin(), idxs.end());
}

std::pair<int, double> findBestMatchingLetterCentroid(const std::set<size_t>& testPattern);
std::pair<int, double> findBestMatchingLetter(const std::set<size_t>& testPattern);

double centroidSimilarity(const std::set<size_t>& testPattern, int letter) {
    if (testPattern.empty() || letterPatternCounts[letter] == 0) return 0.0;
    double score = 0.0;
    for (size_t idx : testPattern) {
        if (idx < letterL5Counts[letter].size()) {
            score += static_cast<double>(letterL5Counts[letter][idx]) / letterPatternCounts[letter];
        }
    }
    return score / testPattern.size();
}

std::pair<int, double> findBestMatchingLetterCentroid(const std::set<size_t>& testPattern) {
    int best = -1; double bestSim = -1.0;
    for (int l = 0; l < NUM_LETTERS; ++l) {
        double s = centroidSimilarity(testPattern, l);
        if (s > bestSim) { bestSim = s; best = l; }
    }
    return {best, bestSim};
}

double jaccardSimilarity(const std::set<size_t>& a, const std::set<size_t>& b) {
    if (a.empty() && b.empty()) return 1.0;
    if (a.empty() || b.empty()) return 0.0;
    size_t inter = 0;
    auto ia = a.begin(); auto ib = b.begin();
    while (ia != a.end() && ib != b.end()) {
        if (*ia == *ib) { ++inter; ++ia; ++ib; }
        else if (*ia < *ib) ++ia; else ++ib;
    }
    size_t uni = a.size() + b.size() - inter;
    return uni == 0 ? 1.0 : static_cast<double>(inter) / uni;
}

int main() {
    // Seed centroid counts: letter A has high counts on 1,2,3; letter B on 10,11,12
    letterPatternCounts.fill(0);
    for (int l = 0; l < NUM_LETTERS; ++l) letterL5Counts[l].fill(0);

    for (int i = 0; i < 10; ++i) { letterL5Counts[0][1]++; letterL5Counts[0][2]++; letterL5Counts[0][3]++; }
    letterPatternCounts[0] = 10;
    for (int i = 0; i < 10; ++i) { letterL5Counts[1][10]++; letterL5Counts[1][11]++; letterL5Counts[1][12]++; }
    letterPatternCounts[1] = 10;

    auto testA = makePattern({1,2,3});
    auto testB = makePattern({10,11,12});
    auto testNoise = makePattern({100,200});

    auto ca = findBestMatchingLetterCentroid(testA);
    auto cb = findBestMatchingLetterCentroid(testB);
    auto cn = findBestMatchingLetterCentroid(testNoise);

    bool pass = (ca.first == 0 && cb.first == 1 && cn.first != -1);
    if (!pass) {
        std::cerr << "FAIL centroid: A->" << ca.first << " B->" << cb.first << " noise->" << cn.first << std::endl;
        return 1;
    }

    std::cout << "PASS centroid classifier sanity (A sim=" << ca.second << ", B sim=" << cb.second << ")" << std::endl;
    return 0;
}
