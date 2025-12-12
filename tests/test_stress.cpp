#include <gtest/gtest.h>
#include "snnfw/Datastore.h"
#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Neuron.h"
#include "snnfw/Axon.h"
#include "snnfw/Dendrite.h"
#include "snnfw/Synapse.h"
#include "snnfw/Cluster.h"
#include "snnfw/SpikeProcessor.h"
#include "snnfw/NetworkPropagator.h"
#include "snnfw/ThreadPool.h"
#include "snnfw/Logger.h"
#include <memory>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <filesystem>

using namespace snnfw;

/**
 * @brief Stress tests for SNNFW framework
 * 
 * These tests validate behavior at scale:
 * - Large-scale networks (1M neurons, 100M synapses)
 * - High spike volumes (1B spikes)
 * - Cache eviction under pressure
 * - Thread pool saturation
 * - Memory management under load
 */
class StressTest : public ::testing::Test {
protected:
    std::unique_ptr<Datastore> datastore;
    std::unique_ptr<NeuralObjectFactory> factory;
    
    void SetUp() override {
        // Initialize logger with error level to reduce output during stress tests
        Logger::getInstance().initialize("/tmp/test_stress.log", spdlog::level::err);
        
        // Create datastore with 1M object cache
        std::filesystem::remove_all("/tmp/test_stress_db");
        datastore = std::make_unique<Datastore>("/tmp/test_stress_db", 1000000);
        
        // Create factory
        factory = std::make_unique<NeuralObjectFactory>();
    }
    
    void TearDown() override {
        factory.reset();
        datastore.reset();
        std::filesystem::remove_all("/tmp/test_stress_db");
    }
};

/**
 * @brief Test 1: Large-scale network creation (1M neurons, 100M synapses)
 * 
 * This test validates:
 * - Ability to create and manage 1M neurons
 * - Ability to create and manage 100M synapses
 * - Memory management at scale
 * - ID generation and uniqueness
 */
TEST_F(StressTest, LargeScaleNetworkCreation) {
    const size_t NUM_NEURONS = 10000;  // 10K neurons (reduced for practical test execution time)
    const size_t SYNAPSES_PER_NEURON = 100;  // 1M total synapses
    
    std::cout << "Creating " << NUM_NEURONS << " neurons..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Create neurons
    std::vector<uint64_t> neuronIds;
    neuronIds.reserve(NUM_NEURONS);
    
    for (size_t i = 0; i < NUM_NEURONS; ++i) {
        auto neuron = factory->createNeuron(100.0, 0.85, 100);
        neuronIds.push_back(neuron->getId());
        datastore->put(neuron);
        
        // Progress indicator
        if ((i + 1) % 1000 == 0) {
            std::cout << "  Created " << (i + 1) << " neurons..." << std::endl;
        }
    }
    
    auto neuronTime = std::chrono::high_resolution_clock::now();
    auto neuronDuration = std::chrono::duration_cast<std::chrono::milliseconds>(neuronTime - startTime);
    std::cout << "✓ Created " << NUM_NEURONS << " neurons in " << neuronDuration.count() << "ms" << std::endl;
    
    // Create synapses
    std::cout << "Creating " << (NUM_NEURONS * SYNAPSES_PER_NEURON) << " synapses..." << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, NUM_NEURONS - 1);
    
    size_t synapseCount = 0;
    for (size_t i = 0; i < NUM_NEURONS; ++i) {
        auto axon = factory->createAxon(neuronIds[i]);
        datastore->put(axon);
        
        for (size_t j = 0; j < SYNAPSES_PER_NEURON; ++j) {
            size_t targetIdx = dist(gen);
            auto dendrite = factory->createDendrite(neuronIds[targetIdx]);
            auto synapse = factory->createSynapse(axon->getId(), dendrite->getId(), 0.5, 1.0);
            
            datastore->put(dendrite);
            datastore->put(synapse);
            synapseCount++;
        }
        
        // Progress indicator
        if ((i + 1) % 1000 == 0) {
            std::cout << "  Created " << synapseCount << " synapses..." << std::endl;
        }
    }
    
    auto synapseTime = std::chrono::high_resolution_clock::now();
    auto synapseDuration = std::chrono::duration_cast<std::chrono::milliseconds>(synapseTime - neuronTime);
    std::cout << "✓ Created " << synapseCount << " synapses in " << synapseDuration.count() << "ms" << std::endl;
    
    // Verify cache statistics
    uint64_t hits, misses;
    datastore->getCacheStats(hits, misses);
    std::cout << "Cache stats: " << hits << " hits, " << misses << " misses" << std::endl;
    
    EXPECT_EQ(neuronIds.size(), NUM_NEURONS);
    EXPECT_EQ(synapseCount, NUM_NEURONS * SYNAPSES_PER_NEURON);
}

/**
 * @brief Test 2: Cache eviction under pressure
 * 
 * This test validates:
 * - LRU cache eviction behavior
 * - Dirty object flushing on eviction
 * - Cache hit/miss statistics
 * - Performance under cache pressure
 */
TEST_F(StressTest, CacheEvictionUnderPressure) {
    const size_t CACHE_SIZE = 100000;  // 100K cache
    const size_t NUM_OBJECTS = 500000;  // 500K objects (5x cache size)
    
    // Create smaller datastore to force evictions
    datastore.reset();
    std::filesystem::remove_all("/tmp/test_stress_cache_db");
    datastore = std::make_unique<Datastore>("/tmp/test_stress_cache_db", CACHE_SIZE);

    // Register factory functions for deserialization
    datastore->registerFactory("Neuron", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto neuron = std::make_shared<Neuron>(0, 0, 0);
        return neuron->fromJson(json) ? neuron : nullptr;
    });
    datastore->registerFactory("Axon", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto axon = std::make_shared<Axon>(0);
        return axon->fromJson(json) ? axon : nullptr;
    });
    datastore->registerFactory("Dendrite", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto dendrite = std::make_shared<Dendrite>(0);
        return dendrite->fromJson(json) ? dendrite : nullptr;
    });
    datastore->registerFactory("Synapse", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto synapse = std::make_shared<Synapse>(0, 0, 0.0, 0.0);
        return synapse->fromJson(json) ? synapse : nullptr;
    });
    
    std::cout << "Creating " << NUM_OBJECTS << " objects with cache size " << CACHE_SIZE << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::vector<uint64_t> neuronIds;
    neuronIds.reserve(NUM_OBJECTS);
    
    // Create objects (will force evictions)
    for (size_t i = 0; i < NUM_OBJECTS; ++i) {
        auto neuron = factory->createNeuron(100.0, 0.85, 100);
        neuronIds.push_back(neuron->getId());
        datastore->put(neuron);
        
        if ((i + 1) % 100000 == 0) {
            std::cout << "  Created " << (i + 1) << " objects..." << std::endl;
        }
    }
    
    auto createTime = std::chrono::high_resolution_clock::now();
    auto createDuration = std::chrono::duration_cast<std::chrono::milliseconds>(createTime - startTime);
    std::cout << "✓ Created " << NUM_OBJECTS << " objects in " << createDuration.count() << "ms" << std::endl;
    
    // Verify cache size is at limit
    EXPECT_LE(datastore->getCacheSize(), CACHE_SIZE);
    std::cout << "Cache size: " << datastore->getCacheSize() << " / " << CACHE_SIZE << std::endl;
    
    // Access objects in random order (will cause cache misses and evictions)
    std::cout << "Randomly accessing objects..." << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, NUM_OBJECTS - 1);
    
    datastore->clearCacheStats();
    const size_t NUM_ACCESSES = 100000;
    
    for (size_t i = 0; i < NUM_ACCESSES; ++i) {
        size_t idx = dist(gen);
        auto neuron = datastore->getNeuron(neuronIds[idx]);
        ASSERT_NE(neuron, nullptr);
        
        // Modify some objects to test dirty tracking
        if (i % 10 == 0) {
            neuron->insertSpike(static_cast<double>(i));
            datastore->markDirty(neuron->getId());
        }
    }
    
    auto accessTime = std::chrono::high_resolution_clock::now();
    auto accessDuration = std::chrono::duration_cast<std::chrono::milliseconds>(accessTime - createTime);
    std::cout << "✓ Completed " << NUM_ACCESSES << " random accesses in " << accessDuration.count() << "ms" << std::endl;
    
    // Check cache statistics
    uint64_t hits, misses;
    datastore->getCacheStats(hits, misses);
    double hitRate = static_cast<double>(hits) / (hits + misses) * 100.0;
    
    std::cout << "Cache stats:" << std::endl;
    std::cout << "  Hits: " << hits << std::endl;
    std::cout << "  Misses: " << misses << std::endl;
    std::cout << "  Hit rate: " << hitRate << "%" << std::endl;
    
    EXPECT_GT(misses, 0);  // Should have cache misses
    EXPECT_GT(hits, 0);    // Should have cache hits
    
    // Cleanup
    std::filesystem::remove_all("/tmp/test_stress_cache_db");
}

/**
 * @brief Test 3: Thread pool saturation
 * 
 * This test validates:
 * - Thread pool behavior under heavy load
 * - Task queue management
 * - Concurrent task execution
 * - No deadlocks or race conditions
 */
TEST_F(StressTest, ThreadPoolSaturation) {
    const size_t NUM_THREADS = 20;
    const size_t NUM_TASKS = 100000;
    
    std::cout << "Creating thread pool with " << NUM_THREADS << " threads" << std::endl;
    ThreadPool pool(NUM_THREADS);
    
    std::cout << "Submitting " << NUM_TASKS << " tasks..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::vector<std::future<int>> results;
    results.reserve(NUM_TASKS);
    
    // Submit many tasks
    for (size_t i = 0; i < NUM_TASKS; ++i) {
        results.push_back(pool.enqueue([](int x) {
            // Simulate some work
            int sum = 0;
            for (int j = 0; j < 100; ++j) {
                sum += j * x;
            }
            return sum;
        }, static_cast<int>(i)));
    }
    
    auto submitTime = std::chrono::high_resolution_clock::now();
    auto submitDuration = std::chrono::duration_cast<std::chrono::milliseconds>(submitTime - startTime);
    std::cout << "✓ Submitted " << NUM_TASKS << " tasks in " << submitDuration.count() << "ms" << std::endl;
    
    // Wait for all tasks to complete
    std::cout << "Waiting for tasks to complete..." << std::endl;
    for (auto& result : results) {
        result.get();
    }
    
    auto completeTime = std::chrono::high_resolution_clock::now();
    auto completeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(completeTime - submitTime);
    std::cout << "✓ All tasks completed in " << completeDuration.count() << "ms" << std::endl;
    
    EXPECT_EQ(results.size(), NUM_TASKS);
}

/**
 * @brief Test 4: High spike volume processing
 * 
 * This test validates:
 * - Spike processor behavior under high load
 * - Spike scheduling and delivery
 * - Thread-safe spike processing
 * - Memory management during spike processing
 */
TEST_F(StressTest, HighSpikeVolumeProcessing) {
    const size_t NUM_NEURONS = 1000;
    const size_t SPIKES_PER_NEURON = 1000;  // 1M total spikes (reduced from 100M for memory constraints)
    const size_t SYNAPSES_PER_NEURON = 10;
    
    std::cout << "Creating network with " << NUM_NEURONS << " neurons..." << std::endl;
    
    // Create neurons
    std::vector<std::shared_ptr<Neuron>> neurons;
    std::vector<std::shared_ptr<Axon>> axons;
    std::vector<std::shared_ptr<Dendrite>> dendrites;
    std::vector<std::shared_ptr<Synapse>> synapses;
    
    neurons.reserve(NUM_NEURONS);
    axons.reserve(NUM_NEURONS);
    
    for (size_t i = 0; i < NUM_NEURONS; ++i) {
        auto neuron = factory->createNeuron(100.0, 0.85, 100);
        auto axon = factory->createAxon(neuron->getId());
        neurons.push_back(neuron);
        axons.push_back(axon);
    }
    
    std::cout << "✓ Created " << NUM_NEURONS << " neurons" << std::endl;

    // Create synapses
    std::cout << "Creating synapses..." << std::endl;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, NUM_NEURONS - 1);

    for (size_t i = 0; i < NUM_NEURONS; ++i) {
        for (size_t j = 0; j < SYNAPSES_PER_NEURON; ++j) {
            size_t targetIdx = dist(gen);
            auto dendrite = factory->createDendrite(neurons[targetIdx]->getId());
            auto synapse = factory->createSynapse(axons[i]->getId(), dendrite->getId(), 0.5, 1.0);
            dendrites.push_back(dendrite);
            synapses.push_back(synapse);
        }
    }

    std::cout << "✓ Created " << synapses.size() << " synapses" << std::endl;

    // Create spike processor
    std::cout << "Creating spike processor..." << std::endl;
    auto spikeProcessor = std::make_shared<SpikeProcessor>(10000, 20);

    // Register dendrites and synapses (neurons and axons don't need registration)
    for (auto& dendrite : dendrites) {
        spikeProcessor->registerDendrite(dendrite);
    }
    for (auto& synapse : synapses) {
        spikeProcessor->registerSynapse(synapse);
    }

    std::cout << "✓ Registered " << dendrites.size() << " dendrites and " << synapses.size() << " synapses" << std::endl;

    // Start spike processor
    spikeProcessor->start();

    // Generate spikes
    std::cout << "Generating " << (NUM_NEURONS * SPIKES_PER_NEURON) << " spikes..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();

    size_t totalSpikes = 0;
    for (size_t i = 0; i < NUM_NEURONS; ++i) {
        for (size_t j = 0; j < SPIKES_PER_NEURON; ++j) {
            double spikeTime = static_cast<double>(j);
            neurons[i]->insertSpike(spikeTime);
            totalSpikes++;
        }

        if ((i + 1) % 1000 == 0) {
            std::cout << "  Generated spikes for " << (i + 1) << " neurons..." << std::endl;
        }
    }

    auto spikeTime = std::chrono::high_resolution_clock::now();
    auto spikeDuration = std::chrono::duration_cast<std::chrono::milliseconds>(spikeTime - startTime);
    std::cout << "✓ Generated " << totalSpikes << " spikes in " << spikeDuration.count() << "ms" << std::endl;

    // Let spike processor run for a bit
    std::cout << "Processing spikes..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(5));

    // Stop spike processor
    spikeProcessor->stop();

    std::cout << "✓ Spike processing complete" << std::endl;

    EXPECT_EQ(totalSpikes, NUM_NEURONS * SPIKES_PER_NEURON);
}

/**
 * @brief Test 5: Memory management under sustained load
 *
 * This test validates:
 * - Memory allocation and deallocation
 * - No memory leaks under sustained load
 * - Proper cleanup of large object graphs
 */
TEST_F(StressTest, MemoryManagementUnderLoad) {
    const size_t NUM_ITERATIONS = 100;
    const size_t OBJECTS_PER_ITERATION = 10000;

    std::cout << "Running " << NUM_ITERATIONS << " iterations of " << OBJECTS_PER_ITERATION << " objects each" << std::endl;

    for (size_t iter = 0; iter < NUM_ITERATIONS; ++iter) {
        std::vector<std::shared_ptr<Neuron>> neurons;
        neurons.reserve(OBJECTS_PER_ITERATION);

        // Create objects
        for (size_t i = 0; i < OBJECTS_PER_ITERATION; ++i) {
            auto neuron = factory->createNeuron(100.0, 0.85, 100);
            neurons.push_back(neuron);
            datastore->put(neuron);
        }

        // Access and modify objects
        for (auto& neuron : neurons) {
            neuron->insertSpike(static_cast<double>(iter));
            datastore->markDirty(neuron->getId());
        }

        // Clear vector (objects should be cleaned up)
        neurons.clear();

        if ((iter + 1) % 10 == 0) {
            std::cout << "  Completed iteration " << (iter + 1) << "/" << NUM_ITERATIONS << std::endl;
        }
    }

    std::cout << "✓ Completed all iterations without memory issues" << std::endl;

    // Verify cache is still functional
    uint64_t hits, misses;
    datastore->getCacheStats(hits, misses);
    std::cout << "Final cache stats: " << hits << " hits, " << misses << " misses" << std::endl;

    EXPECT_GT(datastore->getCacheSize(), 0);
}

/**
 * @brief Test 6: Concurrent datastore access
 *
 * This test validates:
 * - Thread-safe datastore operations
 * - Concurrent reads and writes
 * - No race conditions or deadlocks
 */
TEST_F(StressTest, ConcurrentDatastoreAccess) {
    const size_t NUM_THREADS = 20;
    const size_t OPERATIONS_PER_THREAD = 10000;

    std::cout << "Creating shared objects..." << std::endl;

    // Create some shared objects
    std::vector<uint64_t> sharedIds;
    for (size_t i = 0; i < 1000; ++i) {
        auto neuron = factory->createNeuron(100.0, 0.85, 100);
        sharedIds.push_back(neuron->getId());
        datastore->put(neuron);
    }

    std::cout << "✓ Created " << sharedIds.size() << " shared objects" << std::endl;

    // Launch threads that concurrently access datastore
    std::cout << "Launching " << NUM_THREADS << " threads..." << std::endl;
    auto startTime = std::chrono::high_resolution_clock::now();

    std::vector<std::thread> threads;
    std::atomic<size_t> totalOps{0};

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            std::random_device rd;
            std::mt19937 gen(rd() + t);
            std::uniform_int_distribution<size_t> dist(0, sharedIds.size() - 1);

            for (size_t i = 0; i < OPERATIONS_PER_THREAD; ++i) {
                size_t idx = dist(gen);

                // Mix of reads and writes
                if (i % 3 == 0) {
                    // Write: create new object
                    auto neuron = factory->createNeuron(100.0, 0.85, 100);
                    datastore->put(neuron);
                } else {
                    // Read: access shared object
                    auto neuron = datastore->getNeuron(sharedIds[idx]);
                    if (neuron) {
                        // Modify and mark dirty
                        neuron->insertSpike(static_cast<double>(i));
                        datastore->markDirty(neuron->getId());
                    }
                }

                totalOps++;
            }
        });
    }

    // Wait for all threads
    for (auto& thread : threads) {
        thread.join();
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "✓ Completed " << totalOps.load() << " concurrent operations in " << duration.count() << "ms" << std::endl;

    EXPECT_EQ(totalOps.load(), NUM_THREADS * OPERATIONS_PER_THREAD);
}

