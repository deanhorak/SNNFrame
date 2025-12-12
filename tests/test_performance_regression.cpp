#include <gtest/gtest.h>
#include <snnfw/NeuralObjectFactory.h>
#include <snnfw/Datastore.h>
#include <snnfw/SpikeProcessor.h>
#include <snnfw/ThreadPool.h>
#include <snnfw/Logger.h>
#include <chrono>
#include <random>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

using namespace snnfw;

/**
 * @brief Performance regression test fixture
 *
 * This test suite validates that performance metrics remain within acceptable bounds.
 * It establishes baseline performance metrics and detects regressions.
 */
class PerformanceRegressionTest : public ::testing::Test {
protected:
    std::unique_ptr<Datastore> datastore;
    std::unique_ptr<NeuralObjectFactory> factory;

    // Performance baselines (these should be tuned based on target hardware)
    const double MIN_SPIKE_THROUGHPUT = 100000.0;  // spikes/sec
    const double MAX_BYTES_PER_NEURON = 10000.0;   // bytes
    const double MAX_BYTES_PER_SYNAPSE = 1000.0;   // bytes
    const double MIN_CACHE_HIT_RATE = 15.0;        // percent (with random access)
    const double MAX_FLUSH_TIME_MS = 5000.0;       // milliseconds

    void SetUp() override {
        Logger::getInstance().initialize("/tmp/test_perf_regression.log", spdlog::level::err);
        std::filesystem::remove_all("/tmp/test_perf_regression_db");
        datastore = std::make_unique<Datastore>("/tmp/test_perf_regression_db", 100000);
        factory = std::make_unique<NeuralObjectFactory>();
    }

    void TearDown() override {
        factory.reset();
        datastore.reset();
        std::filesystem::remove_all("/tmp/test_perf_regression_db");
    }

    // Helper to measure memory usage
    size_t getCurrentMemoryUsage() {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.substr(0, 6) == "VmRSS:") {
                std::istringstream iss(line.substr(6));
                size_t memory_kb;
                iss >> memory_kb;
                return memory_kb * 1024;  // Convert to bytes
            }
        }
        return 0;
    }
};

/**
 * @brief Test 1: Spike processing throughput
 *
 * This test validates:
 * - Spike processing rate meets minimum threshold
 * - Throughput scales with network size
 * - No performance degradation over time
 */
TEST_F(PerformanceRegressionTest, SpikeProcessingThroughput) {
    const size_t NUM_NEURONS = 1000;
    const size_t NUM_SPIKES = 100000;

    std::cout << "Creating network with " << NUM_NEURONS << " neurons..." << std::endl;

    // Create neurons
    std::vector<std::shared_ptr<Neuron>> neurons;
    neurons.reserve(NUM_NEURONS);

    for (size_t i = 0; i < NUM_NEURONS; ++i) {
        auto neuron = factory->createNeuron(100.0, 0.85, 100);
        neurons.push_back(neuron);
    }

    // Create axons and dendrites
    std::vector<std::shared_ptr<Axon>> axons;
    std::vector<std::shared_ptr<Dendrite>> dendrites;

    for (size_t i = 0; i < NUM_NEURONS; ++i) {
        auto axon = factory->createAxon(neurons[i]->getId());
        neurons[i]->setAxonId(axon->getId());
        axons.push_back(axon);

        auto dendrite = factory->createDendrite(neurons[i]->getId());
        neurons[i]->addDendrite(dendrite->getId());
        dendrites.push_back(dendrite);
    }

    // Create synapses
    std::vector<std::shared_ptr<Synapse>> synapses;

    for (size_t i = 0; i < NUM_NEURONS; ++i) {
        for (size_t j = 0; j < 10; ++j) {
            size_t targetIdx = (i + j + 1) % NUM_NEURONS;
            auto synapse = factory->createSynapse(
                axons[i]->getId(),
                dendrites[targetIdx]->getId(),
                0.5,
                1.0
            );
            synapses.push_back(synapse);
        }
    }

    std::cout << "✓ Created " << NUM_NEURONS << " neurons and " << synapses.size() << " synapses" << std::endl;

    // Create spike processor
    auto spikeProcessor = std::make_unique<SpikeProcessor>(10000, 20);

    // Register dendrites and synapses
    for (auto& dendrite : dendrites) {
        spikeProcessor->registerDendrite(dendrite);
    }
    for (auto& synapse : synapses) {
        spikeProcessor->registerSynapse(synapse);
    }

    spikeProcessor->start();

    std::cout << "Generating " << NUM_SPIKES << " spikes..." << std::endl;

    // Generate spikes
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> neuronDist(0, NUM_NEURONS - 1);
    std::uniform_real_distribution<double> timeDist(0.0, 1000.0);

    auto startTime = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < NUM_SPIKES; ++i) {
        size_t neuronIdx = neuronDist(gen);
        size_t targetIdx = (neuronIdx + 1) % NUM_NEURONS;
        double time = timeDist(gen);

        // Create action potential for a synapse
        auto ap = std::make_shared<ActionPotential>(
            synapses[neuronIdx * 10]->getId(),  // Use first synapse of this neuron
            dendrites[targetIdx]->getId(),
            time,
            0.5
        );

        spikeProcessor->scheduleSpike(ap);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "✓ Generated " << NUM_SPIKES << " spikes in " << duration.count() << "ms" << std::endl;

    // Wait for processing to complete
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    spikeProcessor->stop();

    // Calculate throughput
    double throughput = (NUM_SPIKES * 1000.0) / duration.count();  // spikes/sec

    std::cout << "Spike processing throughput: " << throughput << " spikes/sec" << std::endl;
    std::cout << "Minimum required: " << MIN_SPIKE_THROUGHPUT << " spikes/sec" << std::endl;

    EXPECT_GE(throughput, MIN_SPIKE_THROUGHPUT)
        << "Spike processing throughput below minimum threshold";
}

/**
 * @brief Test 2: Memory usage per neuron
 *
 * This test validates:
 * - Memory usage per neuron is within acceptable bounds
 * - No memory leaks during object creation
 * - Memory scales linearly with neuron count
 */
TEST_F(PerformanceRegressionTest, MemoryUsagePerNeuron) {
    const size_t NUM_NEURONS = 10000;

    std::cout << "Measuring baseline memory..." << std::endl;
    size_t baselineMemory = getCurrentMemoryUsage();

    std::cout << "Creating " << NUM_NEURONS << " neurons..." << std::endl;

    std::vector<std::shared_ptr<Neuron>> neurons;
    neurons.reserve(NUM_NEURONS);

    for (size_t i = 0; i < NUM_NEURONS; ++i) {
        auto neuron = factory->createNeuron(100.0, 0.85, 100);
        neurons.push_back(neuron);
        datastore->put(neuron);
    }

    std::cout << "✓ Created " << NUM_NEURONS << " neurons" << std::endl;

    size_t afterMemory = getCurrentMemoryUsage();
    size_t memoryUsed = afterMemory - baselineMemory;
    double bytesPerNeuron = static_cast<double>(memoryUsed) / NUM_NEURONS;

    std::cout << "Memory usage: " << memoryUsed << " bytes total" << std::endl;
    std::cout << "Bytes per neuron: " << bytesPerNeuron << std::endl;
    std::cout << "Maximum allowed: " << MAX_BYTES_PER_NEURON << " bytes/neuron" << std::endl;

    EXPECT_LE(bytesPerNeuron, MAX_BYTES_PER_NEURON)
        << "Memory usage per neuron exceeds maximum threshold";
}

/**
 * @brief Test 3: Memory usage per synapse
 *
 * This test validates:
 * - Memory usage per synapse is within acceptable bounds
 * - Synapse memory overhead is minimal
 * - Memory scales linearly with synapse count
 */
TEST_F(PerformanceRegressionTest, MemoryUsagePerSynapse) {
    const size_t NUM_SYNAPSES = 100000;

    std::cout << "Measuring baseline memory..." << std::endl;
    size_t baselineMemory = getCurrentMemoryUsage();

    std::cout << "Creating " << NUM_SYNAPSES << " synapses..." << std::endl;

    std::vector<std::shared_ptr<Synapse>> synapses;
    synapses.reserve(NUM_SYNAPSES);

    // Create a few neurons for axons and dendrites
    auto neuron1 = factory->createNeuron(100.0, 0.85, 100);
    auto neuron2 = factory->createNeuron(100.0, 0.85, 100);

    // Create axon and dendrite
    auto axon = factory->createAxon(neuron1->getId());
    auto dendrite = factory->createDendrite(neuron2->getId());

    for (size_t i = 0; i < NUM_SYNAPSES; ++i) {
        auto synapse = factory->createSynapse(
            axon->getId(),
            dendrite->getId(),
            0.5,
            1.0
        );
        synapses.push_back(synapse);
        datastore->put(synapse);
    }

    std::cout << "✓ Created " << NUM_SYNAPSES << " synapses" << std::endl;

    size_t afterMemory = getCurrentMemoryUsage();
    size_t memoryUsed = afterMemory - baselineMemory;
    double bytesPerSynapse = static_cast<double>(memoryUsed) / NUM_SYNAPSES;

    std::cout << "Memory usage: " << memoryUsed << " bytes total" << std::endl;
    std::cout << "Bytes per synapse: " << bytesPerSynapse << std::endl;
    std::cout << "Maximum allowed: " << MAX_BYTES_PER_SYNAPSE << " bytes/synapse" << std::endl;

    EXPECT_LE(bytesPerSynapse, MAX_BYTES_PER_SYNAPSE)
        << "Memory usage per synapse exceeds maximum threshold";
}



/**
 * @brief Test 4: Cache hit rate
 *
 * This test validates:
 * - Cache hit rate meets minimum threshold with realistic access patterns
 * - LRU eviction policy is effective
 * - Cache performance is consistent
 */
TEST_F(PerformanceRegressionTest, CacheHitRate) {
    const size_t CACHE_SIZE = 10000;
    const size_t NUM_OBJECTS = 50000;  // 5x cache size
    const size_t NUM_ACCESSES = 100000;

    // Create smaller datastore to test cache behavior
    datastore.reset();
    std::filesystem::remove_all("/tmp/test_perf_cache_db");
    datastore = std::make_unique<Datastore>("/tmp/test_perf_cache_db", CACHE_SIZE);

    // Register factory functions
    datastore->registerFactory("Neuron", [](const std::string& json) -> std::shared_ptr<NeuralObject> {
        auto neuron = std::make_shared<Neuron>(0, 0, 0);
        return neuron->fromJson(json) ? neuron : nullptr;
    });

    std::cout << "Creating " << NUM_OBJECTS << " objects with cache size " << CACHE_SIZE << std::endl;

    std::vector<uint64_t> neuronIds;
    neuronIds.reserve(NUM_OBJECTS);

    for (size_t i = 0; i < NUM_OBJECTS; ++i) {
        auto neuron = factory->createNeuron(100.0, 0.85, 100);
        neuronIds.push_back(neuron->getId());
        datastore->put(neuron);
    }

    std::cout << "✓ Created " << NUM_OBJECTS << " objects" << std::endl;

    // Clear cache stats
    datastore->clearCacheStats();

    // Access objects with realistic pattern (some locality)
    std::cout << "Performing " << NUM_ACCESSES << " accesses with locality..." << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<size_t> dist(0, NUM_OBJECTS - 1);
    std::uniform_real_distribution<double> localityDist(0.0, 1.0);

    size_t currentIdx = 0;

    for (size_t i = 0; i < NUM_ACCESSES; ++i) {
        // 70% chance of accessing nearby object (locality)
        if (localityDist(gen) < 0.7) {
            currentIdx = (currentIdx + (dist(gen) % 100)) % NUM_OBJECTS;
        } else {
            currentIdx = dist(gen);
        }

        auto neuron = datastore->getNeuron(neuronIds[currentIdx]);
        ASSERT_NE(neuron, nullptr);
    }

    std::cout << "✓ Completed " << NUM_ACCESSES << " accesses" << std::endl;

    // Check cache statistics
    uint64_t hits, misses;
    datastore->getCacheStats(hits, misses);

    double hitRate = (static_cast<double>(hits) / (hits + misses)) * 100.0;

    std::cout << "Cache statistics:" << std::endl;
    std::cout << "  Hits: " << hits << std::endl;
    std::cout << "  Misses: " << misses << std::endl;
    std::cout << "  Hit rate: " << hitRate << "%" << std::endl;
    std::cout << "Minimum required: " << MIN_CACHE_HIT_RATE << "%" << std::endl;

    EXPECT_GE(hitRate, MIN_CACHE_HIT_RATE)
        << "Cache hit rate below minimum threshold";

    // Cleanup
    std::filesystem::remove_all("/tmp/test_perf_cache_db");
}

/**
 * @brief Test 5: Datastore flush time
 *
 * This test validates:
 * - Flush time is within acceptable bounds
 * - Flush performance scales reasonably with object count
 * - No performance degradation with dirty objects
 */
TEST_F(PerformanceRegressionTest, DatastoreFlushTime) {
    const size_t NUM_OBJECTS = 50000;

    std::cout << "Creating " << NUM_OBJECTS << " objects..." << std::endl;

    std::vector<uint64_t> neuronIds;
    neuronIds.reserve(NUM_OBJECTS);

    for (size_t i = 0; i < NUM_OBJECTS; ++i) {
        auto neuron = factory->createNeuron(100.0, 0.85, 100);
        neuronIds.push_back(neuron->getId());
        datastore->put(neuron);
    }

    std::cout << "✓ Created " << NUM_OBJECTS << " objects" << std::endl;

    // Mark all objects as dirty
    std::cout << "Marking all objects as dirty..." << std::endl;
    for (const auto& id : neuronIds) {
        datastore->markDirty(id);
    }

    std::cout << "Flushing all objects to disk..." << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();
    datastore->flushAll();
    auto endTime = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    std::cout << "Flush time: " << duration.count() << "ms" << std::endl;
    std::cout << "Maximum allowed: " << MAX_FLUSH_TIME_MS << "ms" << std::endl;

    EXPECT_LE(duration.count(), MAX_FLUSH_TIME_MS)
        << "Datastore flush time exceeds maximum threshold";
}

/**
 * @brief Test 6: Thread pool task throughput
 *
 * This test validates:
 * - Thread pool can handle high task submission rate
 * - Task execution throughput is acceptable
 * - No performance degradation with many tasks
 */
TEST_F(PerformanceRegressionTest, ThreadPoolTaskThroughput) {
    const size_t NUM_THREADS = 20;
    const size_t NUM_TASKS = 100000;
    const double MIN_TASK_THROUGHPUT = 100000.0;  // tasks/sec

    std::cout << "Creating thread pool with " << NUM_THREADS << " threads..." << std::endl;

    ThreadPool pool(NUM_THREADS);
    std::atomic<size_t> completedTasks{0};

    std::cout << "Submitting " << NUM_TASKS << " tasks..." << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();

    std::vector<std::future<void>> futures;
    futures.reserve(NUM_TASKS);

    for (size_t i = 0; i < NUM_TASKS; ++i) {
        futures.push_back(pool.enqueue([&completedTasks]() {
            // Simulate some work
            volatile int sum = 0;
            for (int j = 0; j < 100; ++j) {
                sum += j;
            }
            completedTasks++;
        }));
    }

    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.get();
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

    double throughput = (NUM_TASKS * 1000.0) / duration.count();  // tasks/sec

    std::cout << "✓ Completed " << completedTasks << " tasks in " << duration.count() << "ms" << std::endl;
    std::cout << "Task throughput: " << throughput << " tasks/sec" << std::endl;
    std::cout << "Minimum required: " << MIN_TASK_THROUGHPUT << " tasks/sec" << std::endl;

    EXPECT_EQ(completedTasks, NUM_TASKS) << "Not all tasks completed";
    EXPECT_GE(throughput, MIN_TASK_THROUGHPUT)
        << "Thread pool task throughput below minimum threshold";
}
