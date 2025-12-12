#include "snnfw/SpikeProcessor.h"
#include "snnfw/ActivityMonitor.h"
#include "snnfw/Logger.h"
#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace snnfw {

SpikeProcessor::SpikeProcessor(size_t timeSliceCount, size_t deliveryThreads)
    : numTimeSlices(timeSliceCount),
      numDeliveryThreads(deliveryThreads),
      timeStep(1.0),
      running(false),
      stopRequested(false),
      currentTime(0.0),
      currentSliceIndex(0),
      realTimeSync(false),  // Default to false (fast mode) - can be enabled with setRealTimeSync(true)
      totalLoopTime(0.0),
      maxLoopTime(0.0),
      loopCount(0),
      accumulatedDrift(0.0),
      stdpAPlus(0.01),
      stdpAMinus(0.012),
      stdpTauPlus(20.0),
      stdpTauMinus(20.0),
      activityMonitor_(nullptr) {

    // Initialize event queue with empty vectors for each time slice
    eventQueue.resize(numTimeSlices);

    // Create thread pool for spike delivery
    threadPool = std::make_unique<ThreadPool>(numDeliveryThreads);

    SNNFW_INFO("SpikeProcessor created: {} time slices, {} delivery threads, real-time sync: {}",
               numTimeSlices, numDeliveryThreads, realTimeSync);
}

SpikeProcessor::~SpikeProcessor() {
    stop();
}

void SpikeProcessor::start() {
    if (running.load()) {
        SNNFW_WARN("SpikeProcessor already running");
        return;
    }

    stopRequested.store(false);
    running.store(true);

    // Reset timing statistics
    {
        std::lock_guard<std::mutex> lock(statsMutex);
        totalLoopTime = 0.0;
        maxLoopTime = 0.0;
        loopCount = 0;
        accumulatedDrift = 0.0;
    }

    // Record wall-clock start time
    startWallTime = std::chrono::steady_clock::now();

    // Start the background processing thread
    processingThread = std::thread(&SpikeProcessor::processingLoop, this);

    SNNFW_INFO("SpikeProcessor started (real-time sync: {})", realTimeSync);
}

void SpikeProcessor::stop() {
    if (!running.load()) {
        return;
    }

    SNNFW_INFO("SpikeProcessor stopping...");
    
    stopRequested.store(true);
    cv.notify_all();
    
    if (processingThread.joinable()) {
        processingThread.join();
    }
    
    // Join all active delivery threads
    {
        std::lock_guard<std::mutex> lock(deliveryThreadsMutex);
        SNNFW_INFO("SpikeProcessor: Joining {} active delivery threads...", activeDeliveryThreads.size());
        for (auto& thread : activeDeliveryThreads) {
            if (thread.joinable()) {
                thread.join();
            }
        }
        activeDeliveryThreads.clear();
    }
    
    running.store(false);

    SNNFW_INFO("SpikeProcessor stopped. Final time: {:.3f}ms", currentTime.load());
}

void SpikeProcessor::setActivityMonitor(ActivityMonitor* monitor) {
    activityMonitor_ = monitor;
    if (monitor) {
        SNNFW_INFO("SpikeProcessor: Activity monitor attached for automatic recording");
    } else {
        SNNFW_INFO("SpikeProcessor: Activity monitor detached");
    }
}

bool SpikeProcessor::scheduleSpike(const std::shared_ptr<ActionPotential>& actionPotential) {
    if (!actionPotential) {
        SNNFW_ERROR("SpikeProcessor: Cannot schedule null action potential");
        return false;
    }

    int sliceIndex = getTimeSliceIndex(actionPotential->getScheduledTime());
    
    if (sliceIndex < 0) {
        SNNFW_WARN("SpikeProcessor: Spike scheduled for time {:.3f}ms is out of range (current: {:.3f}ms, max: {:.3f}ms)",
                   actionPotential->getScheduledTime(),
                   currentTime.load(),
                   currentTime.load() + numTimeSlices * timeStep);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(queueMutex);
        eventQueue[sliceIndex].push_back(actionPotential);
    }

    SNNFW_TRACE("SpikeProcessor: Scheduled spike for time {:.3f}ms (slice {})",
                actionPotential->getScheduledTime(), sliceIndex);
    
    return true;
}

bool SpikeProcessor::scheduleRetrogradeSpike(const std::shared_ptr<RetrogradeActionPotential>& retrogradeAP) {
    if (!retrogradeAP) {
        SNNFW_ERROR("SpikeProcessor: Cannot schedule null retrograde action potential");
        return false;
    }

    int sliceIndex = getTimeSliceIndex(retrogradeAP->getScheduledTime());

    if (sliceIndex < 0) {
        SNNFW_WARN("SpikeProcessor: Retrograde spike scheduled for time {:.3f}ms is out of range (current: {:.3f}ms, max: {:.3f}ms)",
                   retrogradeAP->getScheduledTime(),
                   currentTime.load(),
                   currentTime.load() + numTimeSlices * timeStep);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(queueMutex);
        eventQueue[sliceIndex].push_back(retrogradeAP);
    }

    SNNFW_TRACE("SpikeProcessor: Scheduled retrograde spike for time {:.3f}ms (slice {})",
                retrogradeAP->getScheduledTime(), sliceIndex);

    return true;
}

void SpikeProcessor::registerDendrite(const std::shared_ptr<Dendrite>& dendrite) {
    if (!dendrite) {
        SNNFW_ERROR("SpikeProcessor: Cannot register null dendrite");
        return;
    }

    std::lock_guard<std::mutex> lock(dendriteRegistryMutex);
    dendriteRegistry[dendrite->getId()] = dendrite;

    SNNFW_DEBUG("SpikeProcessor: Registered dendrite {} (total: {})",
                dendrite->getId(), dendriteRegistry.size());
}

void SpikeProcessor::registerSynapse(const std::shared_ptr<Synapse>& synapse) {
    if (!synapse) {
        SNNFW_ERROR("SpikeProcessor: Cannot register null synapse");
        return;
    }

    std::lock_guard<std::mutex> lock(synapseRegistryMutex);
    synapseRegistry[synapse->getId()] = synapse;

    SNNFW_DEBUG("SpikeProcessor: Registered synapse {} (total: {})",
                synapse->getId(), synapseRegistry.size());
}

void SpikeProcessor::unregisterDendrite(uint64_t dendriteId) {
    std::lock_guard<std::mutex> lock(dendriteRegistryMutex);
    auto it = dendriteRegistry.find(dendriteId);

    if (it != dendriteRegistry.end()) {
        dendriteRegistry.erase(it);
        SNNFW_DEBUG("SpikeProcessor: Unregistered dendrite {} (remaining: {})",
                    dendriteId, dendriteRegistry.size());
    } else {
        SNNFW_WARN("SpikeProcessor: Dendrite {} not found for unregistration", dendriteId);
    }
}

size_t SpikeProcessor::getPendingSpikeCount() const {
    std::lock_guard<std::mutex> lock(queueMutex);
    size_t total = 0;
    for (const auto& slice : eventQueue) {
        total += slice.size();
    }
    return total;
}

size_t SpikeProcessor::getSpikeCountAtSlice(size_t timeSliceIndex) const {
    if (timeSliceIndex >= numTimeSlices) {
        return 0;
    }
    
    std::lock_guard<std::mutex> lock(queueMutex);
    return eventQueue[timeSliceIndex].size();
}

int SpikeProcessor::getTimeSliceIndex(double timeMs) const {
    double currentTimeMs = currentTime.load();
    
    // Check if time is in the past
    if (timeMs < currentTimeMs) {
        return -1;
    }
    
    // Calculate relative time from current time
    double relativeTime = timeMs - currentTimeMs;
    
    // Calculate slice index
    size_t relativeSliceIndex = static_cast<size_t>(relativeTime / timeStep);
    
    // Check if it's within our buffer
    if (relativeSliceIndex >= numTimeSlices) {
        return -1;
    }
    
    // Calculate absolute slice index (with wraparound)
    size_t absoluteSliceIndex = (currentSliceIndex + relativeSliceIndex) % numTimeSlices;
    
    return static_cast<int>(absoluteSliceIndex);
}

void SpikeProcessor::processingLoop() {
    SNNFW_INFO("SpikeProcessor: Processing loop started");

    auto loopStartTime = std::chrono::steady_clock::now();

    while (!stopRequested.load()) {
        auto iterationStart = std::chrono::steady_clock::now();

        // Cleanup completed delivery threads
        cleanupCompletedThreads();

        // Kick off async delivery for current time slice (non-blocking)
        deliverSliceAsync(currentSliceIndex, currentTime.load());

        // Advance time (atomic double doesn't have fetch_add in C++17, so we use store)
        double newTime = currentTime.load() + timeStep;
        currentTime.store(newTime);
        currentSliceIndex = (currentSliceIndex + 1) % numTimeSlices;

        auto iterationEnd = std::chrono::steady_clock::now();

        // Calculate how long this iteration took
        auto iterationDuration = std::chrono::duration_cast<std::chrono::microseconds>(
            iterationEnd - iterationStart);
        double iterationTimeUs = static_cast<double>(iterationDuration.count());

        // Update timing statistics
        {
            std::lock_guard<std::mutex> lock(statsMutex);
            totalLoopTime += iterationTimeUs;
            maxLoopTime = std::max(maxLoopTime, iterationTimeUs);
            loopCount++;
        }

        if (realTimeSync) {
            // Real-time synchronization: each timeslice should take exactly 1ms of wall-clock time
            // Calculate expected wall-clock time for this simulation time
            double expectedWallTimeMs = currentTime.load();
            auto expectedWallTime = startWallTime +
                std::chrono::microseconds(static_cast<int64_t>(expectedWallTimeMs * 1000.0));

            auto now = std::chrono::steady_clock::now();

            // Calculate drift
            auto drift = std::chrono::duration_cast<std::chrono::microseconds>(now - expectedWallTime);
            double driftMs = static_cast<double>(drift.count()) / 1000.0;

            {
                std::lock_guard<std::mutex> lock(statsMutex);
                accumulatedDrift = driftMs;
            }

            // If we're ahead of schedule, sleep to maintain real-time sync
            if (drift.count() < 0) {
                // We're ahead - sleep for the remaining time
                auto sleepDuration = std::chrono::microseconds(-drift.count());
                std::this_thread::sleep_for(sleepDuration);
            } else if (driftMs > 10.0) {
                // We're falling behind by more than 10ms - log a warning
                size_t activeThreads = getActiveDeliveryThreadCount();
                SNNFW_WARN("SpikeProcessor: Falling behind real-time by {:.2f}ms at simulation time {:.1f}ms ({} active delivery threads)",
                          driftMs, currentTime.load(), activeThreads);
            }

            // Log timing info periodically (every 1000 iterations = 1 second of sim time)
            if (loopCount % 1000 == 0) {
                double avgLoopTime = totalLoopTime / loopCount;
                SNNFW_DEBUG("SpikeProcessor: Sim time: {:.1f}ms, Avg loop: {:.1f}μs, Max loop: {:.1f}μs, Drift: {:.2f}ms",
                           currentTime.load(), avgLoopTime, maxLoopTime, driftMs);
            }
        } else {
            // Non-real-time mode: run as fast as possible
            // IMPORTANT: Wait for all delivery threads to complete before advancing time
            // This ensures spike propagation completes before the simulation time advances,
            // preventing "out of range" errors when callbacks try to schedule new spikes
            waitForDeliveryThreads();

            // Just a tiny sleep to prevent CPU spinning if there's no work
            if (iterationTimeUs < 10.0) {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        }
    }

    SNNFW_INFO("SpikeProcessor: Processing loop ended at simulation time {:.3f}ms",
               currentTime.load());

    // Log final statistics
    double avgLoopTime, maxLoop, drift;
    getTimingStats(avgLoopTime, maxLoop, drift);
    SNNFW_INFO("SpikeProcessor: Final stats - Avg loop: {:.1f}μs, Max loop: {:.1f}μs, Final drift: {:.2f}ms",
               avgLoopTime, maxLoop, drift);
}

void SpikeProcessor::deliverSliceAsync(size_t sliceIndex, double simTime) {
    // Extract events for this time slice
    std::vector<std::shared_ptr<EventObject>> eventsToDeliver;
    {
        std::lock_guard<std::mutex> lock(queueMutex);
        eventsToDeliver = std::move(eventQueue[sliceIndex]);
        eventQueue[sliceIndex].clear();
    }

    if (eventsToDeliver.empty()) {
        return;
    }

    SNNFW_TRACE("SpikeProcessor: Async delivering {} events at time {:.3f}ms",
                eventsToDeliver.size(), simTime);

    // Spawn a new thread to handle this timeslice's delivery
    // This thread will further divide work among the thread pool
    std::thread deliveryThread([this, eventsToDeliver = std::move(eventsToDeliver), simTime]() {
        // Divide events evenly among thread pool workers
        size_t eventsPerThread = (eventsToDeliver.size() + numDeliveryThreads - 1) / numDeliveryThreads;

        std::vector<std::future<void>> deliveryTasks;

        for (size_t threadIdx = 0; threadIdx < numDeliveryThreads; ++threadIdx) {
            size_t startIdx = threadIdx * eventsPerThread;
            size_t endIdx = std::min(startIdx + eventsPerThread, eventsToDeliver.size());

            if (startIdx >= eventsToDeliver.size()) {
                break;
            }

            // Submit delivery task to thread pool
            deliveryTasks.emplace_back(
                threadPool->enqueue([this, &eventsToDeliver, startIdx, endIdx]() {
                    for (size_t i = startIdx; i < endIdx; ++i) {
                        const auto& event = eventsToDeliver[i];

                        // Check event type and deliver accordingly
                        const char* eventType = event->getEventType();

                        if (strcmp(eventType, "ActionPotential") == 0) {
                            // Forward spike - deliver to dendrite
                            auto spike = std::static_pointer_cast<ActionPotential>(event);

                            std::shared_ptr<Dendrite> dendrite;
                            {
                                std::lock_guard<std::mutex> lock(dendriteRegistryMutex);
                                auto it = dendriteRegistry.find(spike->getDendriteId());
                                if (it != dendriteRegistry.end()) {
                                    dendrite = it->second;
                                }
                            }

                            if (dendrite) {
                                dendrite->receiveSpike(spike);

                                // Record spike in activity monitor if attached
                                if (activityMonitor_) {
                                    activityMonitor_->recordSpike(spike, spike->getScheduledTime());
                                }
                            } else {
                                SNNFW_WARN("SpikeProcessor: Dendrite {} not found for spike delivery",
                                           spike->getDendriteId());
                            }
                        }
                        else if (strcmp(eventType, "RetrogradeActionPotential") == 0) {
                            // Retrograde spike - deliver to synapse for STDP
                            auto retrogradeSpike = std::static_pointer_cast<RetrogradeActionPotential>(event);

                            std::shared_ptr<Synapse> synapse;
                            {
                                std::lock_guard<std::mutex> lock(synapseRegistryMutex);
                                auto it = synapseRegistry.find(retrogradeSpike->getSynapseId());
                                if (it != synapseRegistry.end()) {
                                    synapse = it->second;
                                }
                            }

                            if (synapse) {
                                // Apply STDP based on temporal offset
                                double temporalOffset = retrogradeSpike->getTemporalOffset();
                                applySTDPToSynapse(synapse, temporalOffset);
                            } else {
                                SNNFW_WARN("SpikeProcessor: Synapse {} not found for retrograde spike delivery",
                                           retrogradeSpike->getSynapseId());
                            }
                        }
                        else {
                            SNNFW_WARN("SpikeProcessor: Unknown event type: {}", eventType);
                        }
                    }
                })
            );
        }

        // Wait for all delivery tasks to complete
        for (auto& task : deliveryTasks) {
            task.get();
        }

        SNNFW_TRACE("SpikeProcessor: Completed async delivery for time {:.3f}ms", simTime);
    });

    // Add to active threads list
    {
        std::lock_guard<std::mutex> lock(deliveryThreadsMutex);
        activeDeliveryThreads.push_back(std::move(deliveryThread));
    }
}

void SpikeProcessor::cleanupCompletedThreads() {
    std::lock_guard<std::mutex> lock(deliveryThreadsMutex);
    
    // Join and remove completed threads
    auto it = activeDeliveryThreads.begin();
    while (it != activeDeliveryThreads.end()) {
        if (it->joinable()) {
            // Try to join with zero timeout (non-blocking check)
            // Since C++ doesn't have try_join, we'll just keep accumulating
            // and clean them up periodically or on shutdown
            ++it;
        } else {
            it = activeDeliveryThreads.erase(it);
        }
    }
    
    // Limit the number of threads we keep around
    // Join older threads if we have too many
    const size_t MAX_THREADS = 100;
    while (activeDeliveryThreads.size() > MAX_THREADS && !activeDeliveryThreads.empty()) {
        if (activeDeliveryThreads.front().joinable()) {
            activeDeliveryThreads.front().join();
        }
        activeDeliveryThreads.pop_front();
    }
}

size_t SpikeProcessor::getActiveDeliveryThreadCount() const {
    std::lock_guard<std::mutex> lock(deliveryThreadsMutex);
    return activeDeliveryThreads.size();
}

void SpikeProcessor::waitForDeliveryThreads() {
    std::lock_guard<std::mutex> lock(deliveryThreadsMutex);

    // Join all active delivery threads
    for (auto& thread : activeDeliveryThreads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    activeDeliveryThreads.clear();
}

void SpikeProcessor::getTimingStats(double& avgLoopTime, double& maxLoop, double& driftMs) const {
    std::lock_guard<std::mutex> lock(statsMutex);

    if (loopCount > 0) {
        avgLoopTime = totalLoopTime / loopCount;
    } else {
        avgLoopTime = 0.0;
    }

    maxLoop = maxLoopTime;
    driftMs = accumulatedDrift;
}

void SpikeProcessor::setSTDPParameters(double aPlus, double aMinus, double tauPlus, double tauMinus) {
    stdpAPlus = aPlus;
    stdpAMinus = aMinus;
    stdpTauPlus = tauPlus;
    stdpTauMinus = tauMinus;

    SNNFW_INFO("SpikeProcessor: Updated STDP parameters (A+={}, A-={}, τ+={}, τ-={})",
               stdpAPlus, stdpAMinus, stdpTauPlus, stdpTauMinus);
}

void SpikeProcessor::applySTDPToSynapse(std::shared_ptr<Synapse> synapse, double temporalOffset) {
    if (!synapse) {
        return;
    }

    // Classic STDP learning rule:
    // temporalOffset = lastFiringTime - dispatchTime
    // If temporalOffset >= 0: neuron fired AFTER spike was sent → LTP (strengthen)
    // If temporalOffset < 0: neuron fired BEFORE spike was sent → LTD (weaken)

    double weightChange = 0.0;

    if (temporalOffset >= 0) {
        // LTP: strengthen synapse (post-synaptic neuron fired after spike arrived)
        weightChange = stdpAPlus * std::exp(-temporalOffset / stdpTauPlus);
    } else {
        // LTD: weaken synapse (post-synaptic neuron fired before spike arrived)
        weightChange = -stdpAMinus * std::exp(temporalOffset / stdpTauMinus);
    }

    if (weightChange != 0.0) {
        double oldWeight = synapse->getWeight();
        double newWeight = oldWeight + weightChange;

        // Clamp weight to [0, 2] range to prevent runaway growth/decay
        newWeight = std::max(0.0, std::min(2.0, newWeight));

        synapse->setWeight(newWeight);

        SNNFW_TRACE("SpikeProcessor: STDP update for synapse {}: temporalOffset={:.3f}ms, Δw={:.6f}, weight: {:.4f} → {:.4f}",
                   synapse->getId(), temporalOffset, weightChange, oldWeight, newWeight);
    }
}

} // namespace snnfw

