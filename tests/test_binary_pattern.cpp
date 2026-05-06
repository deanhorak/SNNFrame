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

    // Bins are relative to the earliest spike (normalized to 0ms).
    // The two spikes at 10.x become bin 0 after normalization.
    EXPECT_EQ(pattern[0], 2);
    EXPECT_EQ(pattern[15], 1);   // 25.5 → relative 15.3 → bin 15
    EXPECT_EQ(pattern[40], 2);   // 50.x → relative ~40
    EXPECT_EQ(pattern[90], 1);   // 100 → relative 89.8 → bin 90

    // Check total spikes
    EXPECT_EQ(pattern.getTotalSpikes(), 6);
}

TEST(BinaryPatternTest, WindowSizeScalingKeepsSpikes) {
    // With windowSize > PATTERN_SIZE, spikes should not be silently dropped.
    // The representation is fixed-size (200 bins), so times are scaled/compressed.
    std::vector<double> spikes = {0.0, 100.0, 250.0, 499.0};

    BinaryPattern pattern(spikes, 500.0);

    // All spikes are within the 500ms window and should be represented somewhere.
    EXPECT_EQ(pattern.getTotalSpikes(), spikes.size());

    // The very last spike should land in the last bin (or close to it after scaling).
    // We clamp, so it must not be out-of-range.
    EXPECT_GE(pattern[BinaryPattern::PATTERN_SIZE - 1], 1);
}

TEST(BinaryPatternTest, AbsoluteLatencyModePreservesSingleSpikeTiming) {
    BinaryPattern early({10.0}, 200.0, false);
    BinaryPattern late({80.0}, 200.0, false);

    EXPECT_EQ(early[10], 1);
    EXPECT_EQ(late[80], 1);
    EXPECT_LT(BinaryPattern::cosineSimilarity(early, late), 1.0);
}

TEST(BinaryPatternTest, FirstSpikeRelativeModeCollapsesSingleSpikeLatency) {
    BinaryPattern early({10.0}, 200.0, true);
    BinaryPattern late({80.0}, 200.0, true);

    EXPECT_EQ(early[0], 1);
    EXPECT_EQ(late[0], 1);
    EXPECT_DOUBLE_EQ(BinaryPattern::cosineSimilarity(early, late), 1.0);
}

TEST(BinaryPatternTest, EmptyPattern) {
    BinaryPattern pattern;

    EXPECT_TRUE(pattern.isEmpty());
    EXPECT_EQ(pattern.getTotalSpikes(), 0);
}

TEST(BinaryPatternTest, CosineSimilarity) {
    // Create two identical patterns (after normalization)
    std::vector<double> spikes1 = {10.0, 20.0, 30.0};
    std::vector<double> spikes2 = {10.0, 20.0, 30.0};

    BinaryPattern p1(spikes1, 200.0);
    BinaryPattern p2(spikes2, 200.0);

    // They should be identical
    double sim = BinaryPattern::cosineSimilarity(p1, p2);
    EXPECT_DOUBLE_EQ(sim, 1.0);  // Should be identical

    // Create a different pattern (different relative structure)
    std::vector<double> spikes3 = {10.0, 40.0, 80.0};
    BinaryPattern p3(spikes3, 200.0);

    // Should be lower than identical, but non-zero because the first bin overlaps
    double sim2 = BinaryPattern::cosineSimilarity(p1, p3);
    EXPECT_LT(sim2, 1.0);
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

    // After normalization, earliest spike is at 10 → bin 0.
    // p1[0] = 1, p2[0] = 2; p1[10] = 1, p2[10] = 2
    EXPECT_EQ(p1[0], 1);
    EXPECT_EQ(p2[0], 2);

    // Blend 50% of p2 into p1 (need significant alpha to see change with rounding)
    BinaryPattern::blend(p1, p2, 0.5);

    // After blend: p1[0] = 0.5*1 + 0.5*2 = 1.5 → rounds to 2
    EXPECT_EQ(p1[0], 2);
    EXPECT_EQ(p1[10], 2);
}

TEST(BinaryPatternTest, ToSpikeTimes) {
    std::vector<double> spikes = {10.0, 20.0, 30.0};
    BinaryPattern pattern(spikes, 200.0);

    std::vector<double> reconstructed = pattern.toSpikeTimes();

    // Should have same number of spikes
    EXPECT_EQ(reconstructed.size(), spikes.size());

    // Times are relative to the first spike (normalized to 0), at bin centers.
    std::vector<double> expected = {0.5, 10.5, 20.5};
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(reconstructed[i], expected[i], 1e-6);
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
