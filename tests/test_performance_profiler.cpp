#include <gtest/gtest.h>
#include "snnfw/PerformanceProfiler.h"
#include <thread>
#include <chrono>

using namespace snnfw;

class PerformanceProfilerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Nothing to set up
    }
    
    void TearDown() override {
        // Nothing to tear down
    }
    
    // Helper function to simulate work
    void simulateWork(int milliseconds) {
        std::this_thread::sleep_for(std::chrono::milliseconds(milliseconds));
    }
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(PerformanceProfilerTest, ConstructorAndDefaults) {
    PerformanceProfiler profiler;
    
    EXPECT_FALSE(profiler.isProfiling());
    EXPECT_TRUE(profiler.getAllMetrics().empty());
}

TEST_F(PerformanceProfilerTest, StartAndStopProfiling) {
    PerformanceProfiler profiler;
    
    EXPECT_FALSE(profiler.isProfiling());
    
    profiler.startProfiling();
    EXPECT_TRUE(profiler.isProfiling());
    
    profiler.stopProfiling();
    EXPECT_FALSE(profiler.isProfiling());
}

TEST_F(PerformanceProfilerTest, StartTwice) {
    PerformanceProfiler profiler;
    
    profiler.startProfiling();
    EXPECT_TRUE(profiler.isProfiling());
    
    // Starting again should not crash
    profiler.startProfiling();
    EXPECT_TRUE(profiler.isProfiling());
    
    profiler.stopProfiling();
}

TEST_F(PerformanceProfilerTest, StopWithoutStart) {
    PerformanceProfiler profiler;
    
    // Stopping without starting should not crash
    EXPECT_NO_THROW(profiler.stopProfiling());
    EXPECT_FALSE(profiler.isProfiling());
}

// ============================================================================
// Manual Timing Tests
// ============================================================================

TEST_F(PerformanceProfilerTest, ManualTimingSingleOperation) {
    PerformanceProfiler profiler;
    profiler.startProfiling();
    
    profiler.startOperation("TestOp");
    simulateWork(10);
    profiler.endOperation("TestOp");
    
    auto metrics = profiler.getMetrics("TestOp");
    EXPECT_EQ(metrics.name, "TestOp");
    EXPECT_EQ(metrics.callCount, 1);
    EXPECT_GT(metrics.totalTimeMs, 0.0);
    EXPECT_GT(metrics.avgTimeMs, 0.0);
}

TEST_F(PerformanceProfilerTest, ManualTimingMultipleCalls) {
    PerformanceProfiler profiler;
    profiler.startProfiling();
    
    for (int i = 0; i < 5; ++i) {
        profiler.startOperation("TestOp");
        simulateWork(5);
        profiler.endOperation("TestOp");
    }
    
    auto metrics = profiler.getMetrics("TestOp");
    EXPECT_EQ(metrics.callCount, 5);
    EXPECT_GT(metrics.totalTimeMs, 0.0);
    EXPECT_GT(metrics.avgTimeMs, 0.0);
    EXPECT_GT(metrics.minTimeMs, 0.0);
    EXPECT_GT(metrics.maxTimeMs, 0.0);
}

TEST_F(PerformanceProfilerTest, ManualTimingWithComponentInfo) {
    PerformanceProfiler profiler;
    profiler.startProfiling();
    
    profiler.startOperation("TestOp", 12345, "Neuron");
    simulateWork(5);
    profiler.endOperation("TestOp");
    
    auto metrics = profiler.getMetrics("TestOp");
    EXPECT_EQ(metrics.componentId, 12345);
    EXPECT_EQ(metrics.componentType, "Neuron");
}

TEST_F(PerformanceProfilerTest, EndOperationWithoutStart) {
    PerformanceProfiler profiler;
    profiler.startProfiling();
    
    // Should not crash
    EXPECT_NO_THROW(profiler.endOperation("NonExistent"));
}

TEST_F(PerformanceProfilerTest, OperationsWhenNotProfiling) {
    PerformanceProfiler profiler;
    // Not started
    
    profiler.startOperation("TestOp");
    profiler.endOperation("TestOp");
    
    // Should not record anything
    EXPECT_TRUE(profiler.getAllMetrics().empty());
}

// ============================================================================
// RAII Timer Tests
// ============================================================================

TEST_F(PerformanceProfilerTest, ScopedTimerBasic) {
    PerformanceProfiler profiler;
    profiler.startProfiling();
    
    {
        auto timer = profiler.startTimer("ScopedOp");
        simulateWork(10);
    } // Timer stops here
    
    auto metrics = profiler.getMetrics("ScopedOp");
    EXPECT_EQ(metrics.callCount, 1);
    EXPECT_GT(metrics.totalTimeMs, 0.0);
}

TEST_F(PerformanceProfilerTest, ScopedTimerManualStop) {
    PerformanceProfiler profiler;
    profiler.startProfiling();
    
    {
        auto timer = profiler.startTimer("ScopedOp");
        simulateWork(5);
        timer.stop();
        simulateWork(5);  // This should not be counted
    }
    
    auto metrics = profiler.getMetrics("ScopedOp");
    EXPECT_EQ(metrics.callCount, 1);
    // Time should be closer to 5ms than 10ms
    EXPECT_LT(metrics.totalTimeMs, 15.0);
}

TEST_F(PerformanceProfilerTest, ScopedTimerMove) {
    PerformanceProfiler profiler;
    profiler.startProfiling();
    
    {
        auto timer1 = profiler.startTimer("ScopedOp");
        auto timer2 = std::move(timer1);
        simulateWork(5);
    }
    
    auto metrics = profiler.getMetrics("ScopedOp");
    EXPECT_EQ(metrics.callCount, 1);
}

// ============================================================================
// Metrics Tests
// ============================================================================

TEST_F(PerformanceProfilerTest, GetMetricsNonExistent) {
    PerformanceProfiler profiler;
    profiler.startProfiling();
    
    auto metrics = profiler.getMetrics("NonExistent");
    EXPECT_EQ(metrics.callCount, 0);
    EXPECT_EQ(metrics.totalTimeMs, 0.0);
}

TEST_F(PerformanceProfilerTest, GetAllMetrics) {
    PerformanceProfiler profiler;
    profiler.startProfiling();
    
    profiler.startOperation("Op1");
    profiler.endOperation("Op1");
    
    profiler.startOperation("Op2");
    profiler.endOperation("Op2");
    
    auto allMetrics = profiler.getAllMetrics();
    EXPECT_EQ(allMetrics.size(), 2);
    EXPECT_NE(allMetrics.find("Op1"), allMetrics.end());
    EXPECT_NE(allMetrics.find("Op2"), allMetrics.end());
}

// ============================================================================
// Spike Processing Tests
// ============================================================================

TEST_F(PerformanceProfilerTest, RecordSpikesProcessed) {
    PerformanceProfiler profiler;
    profiler.startProfiling();
    
    profiler.recordSpikesProcessed(10);
    profiler.recordSpikesProcessed(5);
    
    auto snapshot = profiler.getLatestSnapshot();
    EXPECT_EQ(snapshot.spikesProcessed, 15);
}

// ============================================================================
// Memory Tracking Tests
// ============================================================================

TEST_F(PerformanceProfilerTest, RecordMemoryUsage) {
    PerformanceProfiler profiler;
    profiler.startProfiling();
    
    profiler.startOperation("TestOp");
    profiler.endOperation("TestOp");
    
    profiler.recordMemoryUsage(1024 * 1024, "TestOp");  // 1 MB
    
    auto metrics = profiler.getMetrics("TestOp");
    EXPECT_EQ(metrics.currentMemoryBytes, 1024 * 1024);
    EXPECT_EQ(metrics.peakMemoryBytes, 1024 * 1024);
}

TEST_F(PerformanceProfilerTest, RecordMemoryUsagePeak) {
    PerformanceProfiler profiler;
    profiler.startProfiling();
    
    profiler.startOperation("TestOp");
    profiler.endOperation("TestOp");
    
    profiler.recordMemoryUsage(1024 * 1024, "TestOp");
    profiler.recordMemoryUsage(2 * 1024 * 1024, "TestOp");
    profiler.recordMemoryUsage(512 * 1024, "TestOp");
    
    auto metrics = profiler.getMetrics("TestOp");
    EXPECT_EQ(metrics.currentMemoryBytes, 512 * 1024);
    EXPECT_EQ(metrics.peakMemoryBytes, 2 * 1024 * 1024);
}

// ============================================================================
// Snapshot Tests
// ============================================================================

TEST_F(PerformanceProfilerTest, GetSnapshotEmpty) {
    PerformanceProfiler profiler;
    profiler.startProfiling();
    
    auto snapshot = profiler.getSnapshot(0.0, 100.0);
    EXPECT_EQ(snapshot.startTime, 0.0);
    EXPECT_EQ(snapshot.endTime, 100.0);
    EXPECT_TRUE(snapshot.metrics.empty());
}

TEST_F(PerformanceProfilerTest, GetSnapshotWithData) {
    PerformanceProfiler profiler;
    profiler.startProfiling();
    
    profiler.startOperation("Op1");
    simulateWork(5);
    profiler.endOperation("Op1");
    
    simulateWork(10);  // Wait a bit
    
    auto snapshot = profiler.getLatestSnapshot(100.0);
    EXPECT_FALSE(snapshot.metrics.empty());
    EXPECT_GT(snapshot.totalCpuTimeMs, 0.0);
}

TEST_F(PerformanceProfilerTest, GetLatestSnapshot) {
    PerformanceProfiler profiler;
    profiler.startProfiling();

    profiler.startOperation("Op1");
    simulateWork(5);  // Add some delay
    profiler.endOperation("Op1");

    auto snapshot = profiler.getLatestSnapshot();
    EXPECT_GT(snapshot.wallClockTimeMs, 0.0);
}

// ============================================================================
// Bottleneck Analysis Tests
// ============================================================================

TEST_F(PerformanceProfilerTest, AnalyzeBottlenecksEmpty) {
    PerformanceProfiler profiler;
    profiler.startProfiling();
    
    auto analysis = profiler.analyzeBottlenecks(10);
    EXPECT_TRUE(analysis.topBottlenecks.empty());
    EXPECT_EQ(analysis.totalProfiledTimeMs, 0.0);
}

TEST_F(PerformanceProfilerTest, AnalyzeBottlenecksWithData) {
    PerformanceProfiler profiler;
    profiler.startProfiling();
    
    // Create operations with different execution times
    profiler.startOperation("SlowOp");
    simulateWork(20);
    profiler.endOperation("SlowOp");
    
    profiler.startOperation("FastOp");
    simulateWork(2);
    profiler.endOperation("FastOp");
    
    auto analysis = profiler.analyzeBottlenecks(10);
    EXPECT_EQ(analysis.topBottlenecks.size(), 2);
    
    // First bottleneck should be SlowOp
    EXPECT_EQ(analysis.topBottlenecks[0].operationName, "SlowOp");
    EXPECT_GT(analysis.topBottlenecks[0].percentageOfTotal, 
              analysis.topBottlenecks[1].percentageOfTotal);
}

TEST_F(PerformanceProfilerTest, AnalyzeBottlenecksTopN) {
    PerformanceProfiler profiler;
    profiler.startProfiling();
    
    for (int i = 0; i < 10; ++i) {
        std::string opName = "Op" + std::to_string(i);
        profiler.startOperation(opName);
        simulateWork(1);
        profiler.endOperation(opName);
    }
    
    auto analysis = profiler.analyzeBottlenecks(5);
    EXPECT_EQ(analysis.topBottlenecks.size(), 5);
}

// ============================================================================
// Alert Callback Tests
// ============================================================================

TEST_F(PerformanceProfilerTest, RegisterAndUnregisterAlertCallback) {
    PerformanceProfiler profiler;
    profiler.startProfiling();
    
    bool callbackCalled = false;
    auto callback = [&callbackCalled](const std::string& op, double time, const std::string& type) {
        callbackCalled = true;
    };
    
    uint64_t id = profiler.registerAlertCallback(callback, 5.0);
    EXPECT_GT(id, 0);
    
    profiler.startOperation("SlowOp");
    simulateWork(10);  // Exceeds threshold
    profiler.endOperation("SlowOp");
    
    EXPECT_TRUE(callbackCalled);
    
    profiler.unregisterAlertCallback(id);
}

TEST_F(PerformanceProfilerTest, AlertCallbackThreshold) {
    PerformanceProfiler profiler;
    profiler.startProfiling();
    
    int callbackCount = 0;
    auto callback = [&callbackCount](const std::string& op, double time, const std::string& type) {
        callbackCount++;
    };
    
    profiler.registerAlertCallback(callback, 10.0);
    
    profiler.startOperation("FastOp");
    simulateWork(2);  // Below threshold
    profiler.endOperation("FastOp");
    
    profiler.startOperation("SlowOp");
    simulateWork(15);  // Above threshold
    profiler.endOperation("SlowOp");
    
    EXPECT_EQ(callbackCount, 1);  // Only SlowOp should trigger
}

// ============================================================================
// Reset Tests
// ============================================================================

TEST_F(PerformanceProfilerTest, Reset) {
    PerformanceProfiler profiler;
    profiler.startProfiling();
    
    profiler.startOperation("TestOp");
    profiler.endOperation("TestOp");
    profiler.recordSpikesProcessed(10);
    
    EXPECT_FALSE(profiler.getAllMetrics().empty());
    
    profiler.reset();
    
    EXPECT_TRUE(profiler.getAllMetrics().empty());
    auto snapshot = profiler.getLatestSnapshot();
    EXPECT_EQ(snapshot.spikesProcessed, 0);
}

// ============================================================================
// Report Generation Tests
// ============================================================================

TEST_F(PerformanceProfilerTest, GenerateReportEmpty) {
    PerformanceProfiler profiler;
    profiler.startProfiling();
    
    std::string report = profiler.generateReport();
    EXPECT_FALSE(report.empty());
    EXPECT_NE(report.find("No profiling data"), std::string::npos);
}

TEST_F(PerformanceProfilerTest, GenerateReportWithData) {
    PerformanceProfiler profiler;
    profiler.startProfiling();
    
    profiler.startOperation("TestOp");
    profiler.endOperation("TestOp");
    
    std::string report = profiler.generateReport();
    EXPECT_FALSE(report.empty());
    EXPECT_NE(report.find("TestOp"), std::string::npos);
}

TEST_F(PerformanceProfilerTest, ExportToJson) {
    PerformanceProfiler profiler;
    profiler.startProfiling();
    
    profiler.startOperation("TestOp");
    profiler.endOperation("TestOp");
    
    std::string json = profiler.exportToJson();
    EXPECT_FALSE(json.empty());
    EXPECT_NE(json.find("TestOp"), std::string::npos);
    EXPECT_NE(json.find("metrics"), std::string::npos);
}

// Main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

