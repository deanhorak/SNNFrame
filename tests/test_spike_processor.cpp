#include <gtest/gtest.h>
#include "snnfw/SpikeProcessor.h"
#include "snnfw/ActionPotential.h"
#include "snnfw/Dendrite.h"
#include "snnfw/Logger.h"
#include <spdlog/spdlog.h>
#include <memory>
#include <thread>
#include <chrono>

using namespace snnfw;

// ============================================================================
// SpikeProcessor Tests
// ============================================================================

class SpikeProcessorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize logger for tests (suppress output)
        Logger::getInstance().initialize("test_spike_processor.log", spdlog::level::warn);
    }
    
    void TearDown() override {}
};

TEST_F(SpikeProcessorTest, ConstructorAndDefaults) {
    SpikeProcessor processor;
    
    EXPECT_FALSE(processor.isRunning());
    EXPECT_DOUBLE_EQ(processor.getCurrentTime(), 0.0);
    EXPECT_DOUBLE_EQ(processor.getTimeStep(), 1.0);
}

TEST_F(SpikeProcessorTest, CustomConstructor) {
    SpikeProcessor processor(5000, 8);
    
    EXPECT_FALSE(processor.isRunning());
    EXPECT_DOUBLE_EQ(processor.getCurrentTime(), 0.0);
}

TEST_F(SpikeProcessorTest, StartAndStop) {
    SpikeProcessor processor(1000, 2);
    
    processor.start();
    EXPECT_TRUE(processor.isRunning());
    
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    processor.stop();
    EXPECT_FALSE(processor.isRunning());
}

TEST_F(SpikeProcessorTest, StartTwice) {
    SpikeProcessor processor(1000, 2);
    
    processor.start();
    EXPECT_TRUE(processor.isRunning());
    
    // Starting again should not crash
    processor.start();
    EXPECT_TRUE(processor.isRunning());
    
    processor.stop();
}

TEST_F(SpikeProcessorTest, StopWithoutStart) {
    SpikeProcessor processor(1000, 2);
    
    // Stopping without starting should not crash
    EXPECT_NO_THROW(processor.stop());
    EXPECT_FALSE(processor.isRunning());
}

TEST_F(SpikeProcessorTest, SetTimeStep) {
    SpikeProcessor processor;
    
    processor.setTimeStep(0.5);
    EXPECT_DOUBLE_EQ(processor.getTimeStep(), 0.5);
    
    processor.setTimeStep(2.0);
    EXPECT_DOUBLE_EQ(processor.getTimeStep(), 2.0);
}

TEST_F(SpikeProcessorTest, RegisterDendrite) {
    SpikeProcessor processor;
    
    auto dendrite1 = std::make_shared<Dendrite>(100, 2001);
    auto dendrite2 = std::make_shared<Dendrite>(101, 2002);
    
    EXPECT_NO_THROW(processor.registerDendrite(dendrite1));
    EXPECT_NO_THROW(processor.registerDendrite(dendrite2));
}

TEST_F(SpikeProcessorTest, RegisterNullDendrite) {
    SpikeProcessor processor;
    
    // Should handle null gracefully
    EXPECT_NO_THROW(processor.registerDendrite(nullptr));
}

TEST_F(SpikeProcessorTest, UnregisterDendrite) {
    SpikeProcessor processor;
    
    auto dendrite = std::make_shared<Dendrite>(100, 2001);
    processor.registerDendrite(dendrite);
    
    EXPECT_NO_THROW(processor.unregisterDendrite(2001));
}

TEST_F(SpikeProcessorTest, UnregisterNonexistentDendrite) {
    SpikeProcessor processor;
    
    // Should handle gracefully
    EXPECT_NO_THROW(processor.unregisterDendrite(9999));
}

TEST_F(SpikeProcessorTest, ScheduleSpike) {
    SpikeProcessor processor(1000, 2);
    
    auto ap = std::make_shared<ActionPotential>(100, 2001, 5.0);
    
    bool scheduled = processor.scheduleSpike(ap);
    EXPECT_TRUE(scheduled);
}

TEST_F(SpikeProcessorTest, ScheduleNullSpike) {
    SpikeProcessor processor(1000, 2);
    
    bool scheduled = processor.scheduleSpike(nullptr);
    EXPECT_FALSE(scheduled);
}

TEST_F(SpikeProcessorTest, ScheduleSpikeInPast) {
    SpikeProcessor processor(1000, 2);
    
    processor.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Try to schedule spike in the past
    auto ap = std::make_shared<ActionPotential>(100, 2001, 0.0);
    bool scheduled = processor.scheduleSpike(ap);
    
    processor.stop();
    
    // Should fail because time is in the past
    EXPECT_FALSE(scheduled);
}

TEST_F(SpikeProcessorTest, ScheduleSpikeTooFarInFuture) {
    SpikeProcessor processor(100, 2);  // Only 100ms buffer
    
    // Try to schedule spike beyond buffer
    auto ap = std::make_shared<ActionPotential>(100, 2001, 200.0);
    bool scheduled = processor.scheduleSpike(ap);
    
    EXPECT_FALSE(scheduled);
}

TEST_F(SpikeProcessorTest, GetPendingSpikeCount) {
    SpikeProcessor processor(1000, 2);
    
    EXPECT_EQ(processor.getPendingSpikeCount(), 0);
    
    auto ap1 = std::make_shared<ActionPotential>(100, 2001, 5.0);
    auto ap2 = std::make_shared<ActionPotential>(101, 2002, 10.0);
    auto ap3 = std::make_shared<ActionPotential>(102, 2003, 15.0);
    
    processor.scheduleSpike(ap1);
    processor.scheduleSpike(ap2);
    processor.scheduleSpike(ap3);
    
    EXPECT_EQ(processor.getPendingSpikeCount(), 3);
}

TEST_F(SpikeProcessorTest, TimeAdvancement) {
    SpikeProcessor processor(1000, 2);
    
    double initialTime = processor.getCurrentTime();
    EXPECT_DOUBLE_EQ(initialTime, 0.0);
    
    processor.start();
    
    // Wait for time to advance
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    double currentTime = processor.getCurrentTime();
    
    processor.stop();
    
    // Time should have advanced
    EXPECT_GT(currentTime, initialTime);
}

TEST_F(SpikeProcessorTest, SpikeDelivery) {
    SpikeProcessor processor(1000, 2);
    
    // Create and register dendrite
    auto dendrite = std::make_shared<Dendrite>(100, 2001);
    processor.registerDendrite(dendrite);
    
    // Schedule spike for near future
    auto ap = std::make_shared<ActionPotential>(100, 2001, 5.0);
    processor.scheduleSpike(ap);
    
    EXPECT_EQ(processor.getPendingSpikeCount(), 1);
    
    processor.start();
    
    // Wait for spike to be delivered
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    processor.stop();
    
    // Spike should have been delivered (count should be 0)
    EXPECT_EQ(processor.getPendingSpikeCount(), 0);
}

TEST_F(SpikeProcessorTest, MultipleSpikesDelivery) {
    SpikeProcessor processor(1000, 4);
    
    // Create and register multiple dendrites
    auto dendrite1 = std::make_shared<Dendrite>(100, 2001);
    auto dendrite2 = std::make_shared<Dendrite>(101, 2002);
    auto dendrite3 = std::make_shared<Dendrite>(102, 2003);
    
    processor.registerDendrite(dendrite1);
    processor.registerDendrite(dendrite2);
    processor.registerDendrite(dendrite3);
    
    // Schedule multiple spikes
    for (int i = 0; i < 10; ++i) {
        auto ap = std::make_shared<ActionPotential>(100 + i, 2001 + (i % 3), 5.0 + i);
        processor.scheduleSpike(ap);
    }
    
    EXPECT_EQ(processor.getPendingSpikeCount(), 10);
    
    processor.start();
    
    // Wait for all spikes to be delivered
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    processor.stop();
    
    // All spikes should have been delivered
    EXPECT_EQ(processor.getPendingSpikeCount(), 0);
}

TEST_F(SpikeProcessorTest, SpikeDeliveryToUnregisteredDendrite) {
    SpikeProcessor processor(1000, 2);
    
    // Schedule spike to non-existent dendrite
    auto ap = std::make_shared<ActionPotential>(100, 9999, 5.0);
    processor.scheduleSpike(ap);
    
    processor.start();
    
    // Should not crash even though dendrite doesn't exist
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    EXPECT_NO_THROW(processor.stop());
}

TEST_F(SpikeProcessorTest, ParallelSpikeDelivery) {
    SpikeProcessor processor(1000, 8);  // 8 delivery threads
    
    // Create and register many dendrites
    std::vector<std::shared_ptr<Dendrite>> dendrites;
    for (int i = 0; i < 100; ++i) {
        auto dendrite = std::make_shared<Dendrite>(100 + i, 2001 + i);
        dendrites.push_back(dendrite);
        processor.registerDendrite(dendrite);
    }
    
    // Schedule many spikes
    for (int i = 0; i < 100; ++i) {
        auto ap = std::make_shared<ActionPotential>(100 + i, 2001 + i, 5.0);
        processor.scheduleSpike(ap);
    }
    
    EXPECT_EQ(processor.getPendingSpikeCount(), 100);
    
    processor.start();
    
    // Wait for all spikes to be delivered
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    processor.stop();
    
    // All spikes should have been delivered
    EXPECT_EQ(processor.getPendingSpikeCount(), 0);
}

TEST_F(SpikeProcessorTest, DestructorStopsProcessor) {
    {
        SpikeProcessor processor(1000, 2);
        processor.start();
        EXPECT_TRUE(processor.isRunning());
        
        // Destructor should stop the processor
    }
    
    // If we get here without hanging, destructor worked correctly
    SUCCEED();
}

// Main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

