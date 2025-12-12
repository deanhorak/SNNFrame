#include "snnfw/ThreadPool.h"
#include "snnfw/ThreadSafe.h"
#include "snnfw/Logger.h"
#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Neuron.h"
#include "snnfw/Cluster.h"
#include <spdlog/spdlog.h>
#include <iostream>
#include <vector>
#include <map>
#include <chrono>
#include <random>
#include <sstream>

using namespace snnfw;

// Example 1: Basic ThreadPool usage
void example1_BasicThreadPool() {
    SNNFW_INFO("=== Example 1: Basic ThreadPool Usage ===");
    
    // Create a thread pool with 4 worker threads
    ThreadPool pool(4);
    
    // Submit some simple tasks
    std::vector<std::future<int>> results;
    
    for (int i = 0; i < 10; ++i) {
        results.emplace_back(
            pool.enqueue([i] {
                std::ostringstream oss;
                oss << std::this_thread::get_id();
                SNNFW_DEBUG("Task {} executing on thread {}", i, oss.str());
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                return i * i;
            })
        );
    }
    
    // Collect results
    SNNFW_INFO("Waiting for {} tasks to complete...", results.size());
    for (size_t i = 0; i < results.size(); ++i) {
        int result = results[i].get();
        SNNFW_INFO("Task {} result: {}", i, result);
    }
    
    SNNFW_INFO("All tasks completed\n");
}

// Example 2: ThreadSafe container
void example2_ThreadSafeContainer() {
    SNNFW_INFO("=== Example 2: ThreadSafe Container ===");
    
    ThreadSafe<std::vector<int>> safeVector;
    ThreadPool pool(4);
    
    // Multiple threads adding to the vector
    std::vector<std::future<void>> tasks;
    
    for (int i = 0; i < 20; ++i) {
        tasks.emplace_back(
            pool.enqueue([&safeVector, i] {
                safeVector.modify([i](std::vector<int>& vec) {
                    vec.push_back(i);
                    SNNFW_DEBUG("Added {} to vector", i);
                });
            })
        );
    }
    
    // Wait for all additions
    for (auto& task : tasks) {
        task.get();
    }
    
    // Read the final size
    size_t size = safeVector.read([](const std::vector<int>& vec) {
        return vec.size();
    });
    
    SNNFW_INFO("Final vector size: {}", size);
    SNNFW_INFO("Vector contents: {}", 
        safeVector.read([](const std::vector<int>& vec) {
            std::string result = "[";
            for (size_t i = 0; i < vec.size(); ++i) {
                result += std::to_string(vec[i]);
                if (i < vec.size() - 1) result += ", ";
            }
            result += "]";
            return result;
        })
    );
    SNNFW_INFO("");
}

// Example 3: ThreadSafeRW for read-heavy workloads
void example3_ReadWriteLock() {
    SNNFW_INFO("=== Example 3: Read-Write Lock ===");
    
    ThreadSafeRW<std::map<int, std::string>> safeMap;
    ThreadPool pool(8);
    
    // Initialize map
    safeMap.write([](auto& map) {
        for (int i = 0; i < 10; ++i) {
            map[i] = "value_" + std::to_string(i);
        }
    });
    
    std::vector<std::future<void>> tasks;
    
    // Many readers (can run concurrently)
    for (int i = 0; i < 50; ++i) {
        tasks.emplace_back(
            pool.enqueue([&safeMap, i] {
                auto value = safeMap.read([i](const auto& map) {
                    auto it = map.find(i % 10);
                    return it != map.end() ? it->second : "not found";
                });
                SNNFW_DEBUG("Read operation {}: {}", i, value);
            })
        );
    }
    
    // Few writers (exclusive access)
    for (int i = 0; i < 5; ++i) {
        tasks.emplace_back(
            pool.enqueue([&safeMap, i] {
                safeMap.write([i](auto& map) {
                    map[i] = "updated_" + std::to_string(i);
                    SNNFW_DEBUG("Write operation: updated key {}", i);
                });
            })
        );
    }
    
    // Wait for all operations
    for (auto& task : tasks) {
        task.get();
    }
    
    SNNFW_INFO("Completed {} read/write operations", tasks.size());
    SNNFW_INFO("");
}

// Example 4: AtomicCounter
void example4_AtomicCounter() {
    SNNFW_INFO("=== Example 4: Atomic Counter ===");
    
    AtomicCounter counter(0);
    ThreadPool pool(8);
    
    std::vector<std::future<void>> tasks;
    
    // Multiple threads incrementing
    for (int i = 0; i < 100; ++i) {
        tasks.emplace_back(
            pool.enqueue([&counter] {
                counter.increment();
            })
        );
    }
    
    // Wait for all increments
    for (auto& task : tasks) {
        task.get();
    }
    
    SNNFW_INFO("Final counter value: {} (expected: 100)", counter.get());
    SNNFW_INFO("");
}

// Example 5: Parallel neuron processing
void example5_ParallelNeuronProcessing() {
    SNNFW_INFO("=== Example 5: Parallel Neuron Processing ===");

    // Create factory and cluster
    NeuralObjectFactory factory;
    auto cluster = factory.createCluster();

    // Create neurons and store them in a map for access
    std::map<uint64_t, std::shared_ptr<Neuron>> neuronMap;

    for (int i = 0; i < 10; ++i) {
        auto neuron = factory.createNeuron(50.0, 0.95, 20);
        neuronMap[neuron->getId()] = neuron;
        cluster->addNeuron(neuron->getId());
    }

    SNNFW_INFO("Created cluster with {} neuron IDs", cluster->size());

    // Process neurons in parallel
    ThreadPool pool(4);
    std::vector<std::future<void>> tasks;

    // Simulate spike insertion in parallel
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 100.0);

    // Get all neuron IDs from cluster
    const auto& neuronIds = cluster->getNeuronIds();

    for (uint64_t neuronId : neuronIds) {
        tasks.emplace_back(
            pool.enqueue([&neuronMap, neuronId, &gen, &dis] {
                if (neuronMap.count(neuronId)) {
                    auto neuron = neuronMap[neuronId];
                    // Insert random spikes
                    for (int j = 0; j < 5; ++j) {
                        double time = dis(gen);
                        neuron->insertSpike(time);
                    }

                    // Learn pattern
                    neuron->learnCurrentPattern();

                    SNNFW_DEBUG("Processed neuron {}", neuron->getId());
                }
            })
        );
    }

    // Wait for all processing
    for (auto& task : tasks) {
        task.get();
    }

    SNNFW_INFO("Parallel neuron processing complete");
    SNNFW_INFO("");
}

// Example 6: Task dependencies and chaining
void example6_TaskChaining() {
    SNNFW_INFO("=== Example 6: Task Chaining ===");
    
    ThreadPool pool(4);
    
    // First task
    auto future1 = pool.enqueue([] {
        SNNFW_INFO("Task 1: Computing...");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return 10;
    });
    
    // Second task depends on first
    int result1 = future1.get();
    auto future2 = pool.enqueue([result1] {
        SNNFW_INFO("Task 2: Using result from Task 1: {}", result1);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return result1 * 2;
    });
    
    // Third task depends on second
    int result2 = future2.get();
    auto future3 = pool.enqueue([result2] {
        SNNFW_INFO("Task 3: Using result from Task 2: {}", result2);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return result2 + 5;
    });
    
    int finalResult = future3.get();
    SNNFW_INFO("Final result: {}", finalResult);
    SNNFW_INFO("");
}

int main() {
    // Initialize logger
    Logger::getInstance().initialize("threading_example.log", spdlog::level::info);
    
    SNNFW_INFO("=== SNNFW Threading Examples ===");
    SNNFW_INFO("Hardware concurrency: {} threads\n", std::thread::hardware_concurrency());
    
    try {
        example1_BasicThreadPool();
        example2_ThreadSafeContainer();
        example3_ReadWriteLock();
        example4_AtomicCounter();
        example5_ParallelNeuronProcessing();
        example6_TaskChaining();
        
        SNNFW_INFO("=== All Examples Completed Successfully ===");
    } catch (const std::exception& e) {
        SNNFW_ERROR("Exception caught: {}", e.what());
        return 1;
    }
    
    return 0;
}

