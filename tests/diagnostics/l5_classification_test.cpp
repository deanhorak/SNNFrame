#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Neuron.h"
#include <iostream>
#include <set>
#include <vector>

// Minimal stand-alone test of centroid and k-NN classification on L5 patterns.
// Builds two labeled patterns, queries a test pattern close to A and expects A.

// Helpers copied from training code (simplified)
double jaccardSimilarity(const std::set<size_t>& a, const std::set<size_t>& b) {
    if (a.empty() && b.empty()) return 1.0;
    if (a.empty() || b.empty()) return 0.0;
    size_t inter = 0;
    size_t uni = a.size() + b.size();
    for (auto v : a) {
        if (b.count(v)) inter++;
    }
    uni -= inter;
    return static_cast<double>(inter) / static_cast<double>(uni);
}

int main() {
    const int NUM_LETTERS = 2; // A and B
    std::vector<std::set<size_t>> patternsA = {
        {1,2,3,10,11}, {2,3,4,11,12}, {1,3,5,10,12}
    };
    std::vector<std::set<size_t>> patternsB = {
        {100,101,102,110}, {101,102,103,111}
    };

    // Build centroid counts
    const size_t TOTAL_L5 = 512;
    std::vector<std::vector<int>> counts(NUM_LETTERS, std::vector<int>(TOTAL_L5, 0));
    std::vector<int> patternCounts(NUM_LETTERS, 0);

    for (auto& p : patternsA) {
        for (auto idx : p) if (idx < TOTAL_L5) counts[0][idx]++;
        patternCounts[0]++;
    }
    for (auto& p : patternsB) {
        for (auto idx : p) if (idx < TOTAL_L5) counts[1][idx]++;
        patternCounts[1]++;
    }

    auto centroidSimilarity = [&](const std::set<size_t>& test, int letter){
        if (test.empty() || patternCounts[letter]==0) return 0.0;
        double score=0.0;
        for (auto idx : test) if (idx < TOTAL_L5) score += static_cast<double>(counts[letter][idx]) / patternCounts[letter];
        return score / test.size();
    };

    auto knnClassify = [&](const std::set<size_t>& test){
        std::vector<std::pair<double,int>> sims;
        for (auto& p : patternsA) sims.push_back({jaccardSimilarity(test,p),0});
        for (auto& p : patternsB) sims.push_back({jaccardSimilarity(test,p),1});
        std::sort(sims.begin(), sims.end(), [](auto&a,auto&b){return a.first>b.first;});
        int votes[NUM_LETTERS]={0};
        int K = std::min(3, (int)sims.size());
        for (int i=0;i<K;++i) votes[sims[i].second]++;
        int best=0; if (votes[1]>votes[0]) best=1;
        double maxSim = sims.empty()?0.0:sims[0].first;
        return std::make_pair(best,maxSim);
    };

    // Query close to A
    std::set<size_t> testPattern = {2,3,4,10};
    auto knn = knnClassify(testPattern);
    auto cenA = centroidSimilarity(testPattern,0);
    auto cenB = centroidSimilarity(testPattern,1);

    bool pass = (knn.first==0) && (cenA > cenB);
    if (!pass) {
        std::cerr << "FAIL: expected A; knn="<<knn.first<<" cenA="<<cenA<<" cenB="<<cenB<<"\n";
        return 1;
    }
    std::cout << "PASS: L5 classification favors A (knn="<<knn.first<<", cenA="<<cenA<<", cenB="<<cenB<<")\n";
    return 0;
}
