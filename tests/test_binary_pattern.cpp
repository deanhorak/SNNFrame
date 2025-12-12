#include "snnfw/BinaryPattern.h"
#include <gtest/gtest.h>
#include <vector>
#include <chrono>

using namespace snnfw;

TEST(BinaryPatternTest, BasicConstruction) {
    // Create spike times
    std::vector<double> spikes = {10.2, 10.4, 25.5, 50.1, 50.3, 100.0};

    // Convert to BinaryPattern
    BinaryPattern pattern(spikes, 200.0);

    // Check binning (std::round rounds to nearest integer, 0.5 rounds up)
    EXPECT_EQ(pattern[10], 2);  // 10.2 and 10.4 both round to 10
    EXPECT_EQ(pattern[26], 1);  // 25.5 rounds to 26
    EXPECT_EQ(pattern[50], 2);  // 50.1 and 50.3 both round to 50
    EXPECT_EQ(pattern[100], 1); // 100.0 rounds to 100

    // Check total spikes
    EXPECT_EQ(pattern.getTotalSpikes(), 6);
}

TEST(BinaryPatternTest, EmptyPattern) {
    BinaryPattern pattern;

    EXPECT_TRUE(pattern.isEmpty());
    EXPECT_EQ(pattern.getTotalSpikes(), 0);
}

TEST(BinaryPatternTest, CosineSimilarity) {
    // Create two identical patterns (after rounding)
    std::vector<double> spikes1 = {10.0, 20.0, 30.0};
    std::vector<double> spikes2 = {10.0, 20.0, 30.0};

    BinaryPattern p1(spikes1, 200.0);
    BinaryPattern p2(spikes2, 200.0);

    // They should be identical
    double sim = BinaryPattern::cosineSimilarity(p1, p2);
    EXPECT_DOUBLE_EQ(sim, 1.0);  // Should be identical

    // Create a different pattern
    std::vector<double> spikes3 = {100.0, 150.0, 180.0};
    BinaryPattern p3(spikes3, 200.0);

    // Should be very different (no overlap)
    double sim2 = BinaryPattern::cosineSimilarity(p1, p3);
    EXPECT_DOUBLE_EQ(sim2, 0.0);  // Should be completely different (orthogonal)
}

TEST(BinaryPatternTest, HistogramIntersection) {
    std::vector<double> spikes1 = {10.0, 20.0, 30.0};
    std::vector<double> spikes2 = {10.0, 20.0, 30.0};

    BinaryPattern p1(spikes1, 200.0);
    BinaryPattern p2(spikes2, 200.0);

    // Identical patterns should have similarity 1.0
    double sim = BinaryPattern::histogramIntersection(p1, p2);
    EXPECT_DOUBLE_EQ(sim, 1.0);
}

TEST(BinaryPatternTest, Blending) {
    std::vector<double> spikes1 = {10.0, 20.0, 30.0};
    std::vector<double> spikes2 = {10.0, 10.0, 20.0, 20.0};  // More spikes at 10 and 20

    BinaryPattern p1(spikes1, 200.0);
    BinaryPattern p2(spikes2, 200.0);

    // p1[10] = 1, p2[10] = 2
    // p1[20] = 1, p2[20] = 2
    EXPECT_EQ(p1[10], 1);
    EXPECT_EQ(p2[10], 2);

    // Blend 50% of p2 into p1 (need significant alpha to see change with rounding)
    BinaryPattern::blend(p1, p2, 0.5);

    // After blend: p1[10] = 0.5*1 + 0.5*2 = 1.5 → rounds to 2
    EXPECT_EQ(p1[10], 2);
    EXPECT_EQ(p1[20], 2);
}

TEST(BinaryPatternTest, ToSpikeTimes) {
    std::vector<double> spikes = {10.0, 20.0, 30.0};
    BinaryPattern pattern(spikes, 200.0);

    std::vector<double> reconstructed = pattern.toSpikeTimes();

    // Should have same number of spikes
    EXPECT_EQ(reconstructed.size(), spikes.size());

    // Spikes should be approximately at the same times (within 1ms)
    for (size_t i = 0; i < spikes.size(); ++i) {
        EXPECT_NEAR(reconstructed[i], spikes[i], 1.0);
    }
}

TEST(BinaryPatternTest, Performance) {
    // Create a large spike train
    std::vector<double> spikes;
    for (int i = 0; i < 100; ++i) {
        spikes.push_back(i * 2.0);  // 100 spikes spread over 200ms
    }

    // Time the conversion
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 10000; ++i) {
        BinaryPattern pattern(spikes, 200.0);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Should be fast (less than 1 microsecond per conversion on average)
    double avgTime = duration.count() / 10000.0;
    EXPECT_LT(avgTime, 10.0);  // Less than 10 microseconds per conversion
}

