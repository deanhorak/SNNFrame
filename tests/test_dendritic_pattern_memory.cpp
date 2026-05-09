#include "snnfw/DendriticPatternMemory.h"

#include <gtest/gtest.h>

using snnfw::DendriticPatternMemory;
using snnfw::DendriticSpikeImage;

TEST(DendriticSpikeImageTests, EncodesRowsAndTimeBins) {
    DendriticSpikeImage image(4, 16, 1.0);

    EXPECT_TRUE(image.addSpike(1, 3.0));
    EXPECT_TRUE(image.addSpike(3, 9.0));
    EXPECT_FALSE(image.addSpike(4, 1.0));
    EXPECT_FALSE(image.addSpike(1, 65536.0));

    EXPECT_TRUE(image.hasSpike(1, 3));
    EXPECT_TRUE(image.hasSpike(3, 9));
    EXPECT_FALSE(image.hasSpike(1, 4));
    EXPECT_EQ(image.spikeCount(), 2U);
}

TEST(DendriticSpikeImageTests, RejectsIncompatibleTemporalResolution) {
    DendriticSpikeImage oneMs(4, 16, 1.0);
    DendriticSpikeImage halfMs(4, 16, 0.5);

    oneMs.setSpike(1, 4);
    halfMs.setSpike(1, 4);

    EXPECT_DOUBLE_EQ(oneMs.jaccardSimilarity(halfMs), 0.0);
    EXPECT_DOUBLE_EQ(oneMs.temporalTolerantSimilarity(halfMs, 1), 0.0);

    oneMs.mergeUnion(halfMs);
    EXPECT_EQ(oneMs.spikeCount(), 1U);
}

TEST(DendriticSpikeImageTests, TemporalToleranceMatchesJitteredSpikeImages) {
    DendriticSpikeImage learned(4, 16, 1.0);
    learned.setSpike(0, 2);
    learned.setSpike(1, 5);
    learned.setSpike(2, 9);

    DendriticSpikeImage jittered(4, 16, 1.0);
    jittered.setSpike(0, 3);
    jittered.setSpike(1, 6);
    jittered.setSpike(2, 8);

    EXPECT_DOUBLE_EQ(learned.jaccardSimilarity(jittered), 0.0);
    EXPECT_GT(learned.temporalTolerantSimilarity(jittered, 1), 0.99);
}

TEST(DendriticPatternMemoryTests, LearnsAndRecognizesJitteredTemporalRaster) {
    DendriticPatternMemory memory({0.70, 1, 8, false});

    DendriticSpikeImage pattern(5, 20, 1.0);
    pattern.setSpike(0, 2);
    pattern.setSpike(2, 7);
    pattern.setSpike(4, 13);

    DendriticSpikeImage jittered(5, 20, 1.0);
    jittered.setSpike(0, 3);
    jittered.setSpike(2, 6);
    jittered.setSpike(4, 14);

    memory.learn(pattern);

    EXPECT_EQ(memory.prototypeCount(), 1U);
    EXPECT_TRUE(memory.recognizes(jittered));
}

TEST(DendriticPatternMemoryTests, SeparatesDifferentRowTimeStructure) {
    DendriticPatternMemory memory({0.70, 1, 8, false});

    DendriticSpikeImage pattern(5, 20, 1.0);
    pattern.setSpike(0, 2);
    pattern.setSpike(2, 7);
    pattern.setSpike(4, 13);

    DendriticSpikeImage different(5, 20, 1.0);
    different.setSpike(1, 2);
    different.setSpike(3, 7);
    different.setSpike(4, 18);

    memory.learn(pattern);

    EXPECT_FALSE(memory.recognizes(different));
}

TEST(DendriticPatternMemoryTests, ReinforcesSimilarPatternInsteadOfAddingPrototype) {
    DendriticPatternMemory memory({0.70, 1, 8, false});

    DendriticSpikeImage pattern(3, 12, 1.0);
    pattern.setSpike(0, 1);
    pattern.setSpike(1, 5);

    DendriticSpikeImage jittered(3, 12, 1.0);
    jittered.setSpike(0, 2);
    jittered.setSpike(1, 4);

    memory.learn(pattern);
    const size_t index = memory.learn(jittered);

    ASSERT_EQ(memory.prototypeCount(), 1U);
    EXPECT_EQ(index, 0U);
    EXPECT_EQ(memory.prototypes().front().support, 2U);
}
