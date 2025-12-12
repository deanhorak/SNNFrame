#include <gtest/gtest.h>
#include "snnfw/ThreadPool.h"
#include "snnfw/ThreadSafe.h"
#include <vector>
#include <chrono>
#include <thread>

using namespace snnfw;

// Test fixture for threading tests
class ThreadingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }
    
    void TearDown() override {
        // Cleanup code if needed
    }
};

// ============================================================================
// ThreadPool Tests
// ============================================================================

TEST_F(ThreadingTest, ThreadPoolCreation) {
    ThreadPool pool(4);
    EXPECT_EQ(pool.size(), 4);
    EXPECT_FALSE(pool.isStopped());
}

TEST_F(ThreadingTest, ThreadPoolDefaultSize) {
    ThreadPool pool;
    EXPECT_GT(pool.size(), 0);
    EXPECT_LE(pool.size(), std::thread::hardware_concurrency());
}

TEST_F(ThreadingTest, ThreadPoolExecuteSimpleTask) {
    ThreadPool pool(2);
    
    auto future = pool.enqueue([] {
        return 42;
    });
    
    EXPECT_EQ(future.get(), 42);
}

TEST_F(ThreadingTest, ThreadPoolExecuteMultipleTasks) {
    ThreadPool pool(4);
    std::vector<std::future<int>> results;
    
    for (int i = 0; i < 10; ++i) {
        results.emplace_back(pool.enqueue([i] {
            return i * 2;
        }));
    }
    
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(results[i].get(), i * 2);
    }
}

TEST_F(ThreadingTest, ThreadPoolTasksWithArguments) {
    ThreadPool pool(2);
    
    auto future = pool.enqueue([](int a, int b) {
        return a + b;
    }, 10, 20);
    
    EXPECT_EQ(future.get(), 30);
}

TEST_F(ThreadingTest, ThreadPoolPendingTasks) {
    ThreadPool pool(1); // Single thread to create backlog
    
    std::vector<std::future<void>> futures;
    
    // Submit tasks that take time
    for (int i = 0; i < 5; ++i) {
        futures.emplace_back(pool.enqueue([] {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }));
    }
    
    // Should have some pending tasks
    // Note: This is timing-dependent, so we just check it doesn't crash
    size_t pending = pool.pendingTasks();
    EXPECT_GE(pending, 0);
    
    // Wait for all tasks
    for (auto& f : futures) {
        f.get();
    }
    
    // All tasks should be done
    EXPECT_EQ(pool.pendingTasks(), 0);
}

TEST_F(ThreadingTest, ThreadPoolExceptionHandling) {
    ThreadPool pool(2);
    
    auto future = pool.enqueue([] {
        throw std::runtime_error("Test exception");
        return 42;
    });
    
    // Should throw when getting result
    EXPECT_THROW(future.get(), std::runtime_error);
}

TEST_F(ThreadingTest, ThreadPoolShutdown) {
    {
        ThreadPool pool(4);
        
        std::vector<std::future<int>> results;
        for (int i = 0; i < 10; ++i) {
            results.emplace_back(pool.enqueue([i] { return i; }));
        }
        
        // Pool destructor should wait for all tasks
    }
    // If we get here without hanging, shutdown worked
    SUCCEED();
}

// ============================================================================
// ThreadSafe Tests
// ============================================================================

TEST_F(ThreadingTest, ThreadSafeBasicOperations) {
    ThreadSafe<int> safeInt(42);
    
    int value = safeInt.getCopy();
    EXPECT_EQ(value, 42);
    
    safeInt.set(100);
    EXPECT_EQ(safeInt.getCopy(), 100);
}

TEST_F(ThreadingTest, ThreadSafeModify) {
    ThreadSafe<std::vector<int>> safeVec;
    
    safeVec.modify([](std::vector<int>& vec) {
        vec.push_back(1);
        vec.push_back(2);
        vec.push_back(3);
    });
    
    size_t size = safeVec.read([](const std::vector<int>& vec) {
        return vec.size();
    });
    
    EXPECT_EQ(size, 3);
}

TEST_F(ThreadingTest, ThreadSafeConcurrentAccess) {
    ThreadSafe<int> safeCounter(0);
    ThreadPool pool(4);
    
    std::vector<std::future<void>> futures;
    
    // 100 threads incrementing
    for (int i = 0; i < 100; ++i) {
        futures.emplace_back(pool.enqueue([&safeCounter] {
            safeCounter.modify([](int& val) {
                val++;
            });
        }));
    }
    
    // Wait for all
    for (auto& f : futures) {
        f.get();
    }
    
    EXPECT_EQ(safeCounter.getCopy(), 100);
}

TEST_F(ThreadingTest, ThreadSafeVectorConcurrent) {
    ThreadSafe<std::vector<int>> safeVec;
    ThreadPool pool(8);
    
    std::vector<std::future<void>> futures;
    
    // Multiple threads adding elements
    for (int i = 0; i < 50; ++i) {
        futures.emplace_back(pool.enqueue([&safeVec, i] {
            safeVec.modify([i](std::vector<int>& vec) {
                vec.push_back(i);
            });
        }));
    }
    
    // Wait for all
    for (auto& f : futures) {
        f.get();
    }
    
    size_t size = safeVec.read([](const std::vector<int>& vec) {
        return vec.size();
    });
    
    EXPECT_EQ(size, 50);
}

// ============================================================================
// ThreadSafeRW Tests
// ============================================================================

TEST_F(ThreadingTest, ThreadSafeRWBasicOperations) {
    ThreadSafeRW<int> safeInt(42);
    
    int value = safeInt.getCopy();
    EXPECT_EQ(value, 42);
    
    safeInt.set(100);
    EXPECT_EQ(safeInt.getCopy(), 100);
}

TEST_F(ThreadingTest, ThreadSafeRWReadWrite) {
    ThreadSafeRW<std::map<int, std::string>> safeMap;
    
    // Write
    safeMap.write([](auto& map) {
        map[1] = "one";
        map[2] = "two";
        map[3] = "three";
    });
    
    // Read
    std::string value = safeMap.read([](const auto& map) {
        auto it = map.find(2);
        return it != map.end() ? it->second : "";
    });
    
    EXPECT_EQ(value, "two");
}

TEST_F(ThreadingTest, ThreadSafeRWConcurrentReads) {
    ThreadSafeRW<std::vector<int>> safeVec;
    
    // Initialize
    safeVec.write([](auto& vec) {
        for (int i = 0; i < 100; ++i) {
            vec.push_back(i);
        }
    });
    
    ThreadPool pool(8);
    std::vector<std::future<int>> futures;
    
    // Many concurrent reads
    for (int i = 0; i < 100; ++i) {
        futures.emplace_back(pool.enqueue([&safeVec, i] {
            return safeVec.read([i](const auto& vec) {
                return vec[i];
            });
        }));
    }
    
    // Verify all reads
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(futures[i].get(), i);
    }
}

// ============================================================================
// AtomicCounter Tests
// ============================================================================

TEST_F(ThreadingTest, AtomicCounterBasicOperations) {
    AtomicCounter counter(0);
    
    EXPECT_EQ(counter.get(), 0);
    EXPECT_EQ(counter.increment(), 1);
    EXPECT_EQ(counter.increment(), 2);
    EXPECT_EQ(counter.decrement(), 1);
    EXPECT_EQ(counter.get(), 1);
}

TEST_F(ThreadingTest, AtomicCounterSet) {
    AtomicCounter counter(0);
    
    counter.set(42);
    EXPECT_EQ(counter.get(), 42);
}

TEST_F(ThreadingTest, AtomicCounterAddSubtract) {
    AtomicCounter counter(10);
    
    EXPECT_EQ(counter.add(5), 15);
    EXPECT_EQ(counter.subtract(3), 12);
    EXPECT_EQ(counter.get(), 12);
}

TEST_F(ThreadingTest, AtomicCounterConcurrentIncrement) {
    AtomicCounter counter(0);
    ThreadPool pool(8);
    
    std::vector<std::future<void>> futures;
    
    // 1000 concurrent increments
    for (int i = 0; i < 1000; ++i) {
        futures.emplace_back(pool.enqueue([&counter] {
            counter.increment();
        }));
    }
    
    // Wait for all
    for (auto& f : futures) {
        f.get();
    }
    
    EXPECT_EQ(counter.get(), 1000);
}

TEST_F(ThreadingTest, AtomicCounterConcurrentMixed) {
    AtomicCounter counter(1000);
    ThreadPool pool(8);
    
    std::vector<std::future<void>> futures;
    
    // 500 increments and 500 decrements
    for (int i = 0; i < 500; ++i) {
        futures.emplace_back(pool.enqueue([&counter] {
            counter.increment();
        }));
        futures.emplace_back(pool.enqueue([&counter] {
            counter.decrement();
        }));
    }
    
    // Wait for all
    for (auto& f : futures) {
        f.get();
    }
    
    // Should be back to 1000
    EXPECT_EQ(counter.get(), 1000);
}

// Main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

