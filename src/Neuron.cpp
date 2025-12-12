#include "snnfw/Neuron.h"
#include "snnfw/Logger.h"
#include "snnfw/learning/PatternUpdateStrategy.h"
#include "snnfw/NetworkPropagator.h"
#include "snnfw/SpikeAcknowledgment.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <algorithm>
#include <random>

using json = nlohmann::json;

namespace snnfw {

Neuron::Neuron(double windowSizeMs, double similarityThreshold, size_t maxReferencePatterns, uint64_t neuronId)
    : NeuralObject(neuronId),
      windowSize(windowSizeMs),
      threshold(similarityThreshold),
      maxPatterns(maxReferencePatterns),
      axonId(0),
      similarityMetric_(SimilarityMetric::COSINE),  // Default to cosine similarity
      inhibition_(0.0),
      firingRate_(0.0),
      targetFiringRate_(5.0),  // Default target: 5 Hz
      intrinsicExcitability_(1.0),
      lastFiringTime_(-1000.0),
      firingCount_(0),
      firingWindowStart_(0.0) {
    // Generate unique temporal signature for this neuron
    generateTemporalSignature();
}

void Neuron::insertSpike(double spikeTime) {
    std::lock_guard<std::mutex> lock(spikesMutex_);
    spikes.push_back(spikeTime);
    removeOldSpikesUnsafe(spikeTime);

    // NOTE: Removed shouldFire() check here because it causes issues during training
    // When learning patterns, we don't want the neuron to fire based on previously
    // learned patterns. Use getBestSimilarity() or checkShouldFire() explicitly
    // during testing/inference instead.

    // if (shouldFire()) {
    //     SNNFW_INFO("Neuron {} fires a new spike at time: {}", getId(), spikeTime);
    // }
}

void Neuron::learnCurrentPattern() {
    // Copy spikes under lock to avoid holding lock during pattern learning
    std::vector<double> spikesCopy;
    {
        std::lock_guard<std::mutex> lock(spikesMutex_);
        if (spikes.empty()) {
            SNNFW_DEBUG("Neuron {}: Cannot learn pattern - no spikes in window", getId());
            return;
        }
        spikesCopy = spikes;
    }

    // Convert current spike window to BinaryPattern (200 bytes, fixed size)
    BinaryPattern newPattern(spikesCopy, windowSize);

    SNNFW_DEBUG("Neuron {}: Converting {} spike times to BinaryPattern ({} total spikes)",
                getId(), spikesCopy.size(), newPattern.getTotalSpikes());

    // If a pattern update strategy is set, use it directly with BinaryPattern
    if (patternStrategy_) {
        // Create similarity metric lambda using the selected metric
        auto similarityMetric = [this](const BinaryPattern& a, const BinaryPattern& b) {
            return this->computeSimilarity(a, b);
        };

        // Call the BinaryPattern version of updatePatterns (efficient, no conversion!)
        patternStrategy_->updatePatterns(referencePatterns, newPattern, similarityMetric);

        SNNFW_DEBUG("Neuron {}: Updated patterns using {} strategy ({} patterns stored)",
                    getId(), patternStrategy_->getName(), referencePatterns.size());
        return;
    }

    // Default behavior - work directly with BinaryPattern (MUCH more efficient!)
    if (referencePatterns.size() < maxPatterns) {
        referencePatterns.push_back(newPattern);
        SNNFW_DEBUG("Neuron {}: Learned new BinaryPattern ({} total spikes, {} patterns stored)",
                    getId(), newPattern.getTotalSpikes(), referencePatterns.size());
    } else {
        // Find most similar pattern using selected similarity metric
        int bestIndex = -1;
        double bestSim = -1.0;

        for (size_t i = 0; i < referencePatterns.size(); ++i) {
            double sim = computeSimilarity(referencePatterns[i], newPattern);
            if (sim > bestSim) {
                bestSim = sim;
                bestIndex = static_cast<int>(i);
            }
        }

        if (bestIndex != -1 && bestSim >= threshold) {
            // Blend new pattern into most similar existing pattern
            BinaryPattern::blend(referencePatterns[bestIndex], newPattern, 0.2);
            SNNFW_DEBUG("Neuron {}: Blended new pattern into pattern #{} (similarity={:.3f})",
                        getId(), bestIndex, bestSim);
        } else {
            // Replace random pattern (novel pattern, low similarity to all existing)
            size_t randIndex = rand() % referencePatterns.size();
            referencePatterns[randIndex] = newPattern;
            SNNFW_DEBUG("Neuron {}: Replaced pattern #{} with new pattern (best similarity={:.3f})",
                        getId(), randIndex, bestSim);
        }
    }
}

void Neuron::setPatternUpdateStrategy(std::shared_ptr<learning::PatternUpdateStrategy> strategy) {
    patternStrategy_ = strategy;
    SNNFW_INFO("Neuron {}: Set pattern update strategy to {}", getId(), strategy ? strategy->getName() : "default");
}

void Neuron::printSpikes() const {
    std::lock_guard<std::mutex> lock(spikesMutex_);
    std::string spikesStr;
    for (double spikeTime : spikes) {
        spikesStr += std::to_string(spikeTime) + " ";
    }
    SNNFW_INFO("Neuron {}: Current spikes in window: {}", getId(), spikesStr);
}

void Neuron::printReferencePatterns() const {
    SNNFW_INFO("Neuron {}: Stored reference patterns ({})", getId(), referencePatterns.size());
    for (size_t i = 0; i < referencePatterns.size(); ++i) {
        SNNFW_INFO("  Pattern #{}: {}", i, referencePatterns[i].toString());
    }
}

void Neuron::removeOldSpikes(double currentTime) {
    std::lock_guard<std::mutex> lock(spikesMutex_);
    removeOldSpikesUnsafe(currentTime);
}

void Neuron::removeOldSpikesUnsafe(double currentTime) {
    // Note: Caller must hold spikesMutex_ lock
    while (!spikes.empty() && (currentTime - spikes.front() > windowSize)) {
        spikes.erase(spikes.begin());
    }
}

// Convert spike pattern to temporal histogram (fuzzy representation)
// Uses cumulative distribution function (CDF) instead of histogram
// to preserve temporal ordering information
std::vector<double> Neuron::spikeToHistogram(const std::vector<double>& pattern) const {
    const int bins = 50;  // More bins for better temporal resolution
    std::vector<double> cdf(bins, 0.0);

    if (pattern.empty()) return cdf;

    // Sort spikes for CDF computation
    std::vector<double> sortedSpikes = pattern;
    std::sort(sortedSpikes.begin(), sortedSpikes.end());

    double binSize = windowSize / bins;

    // Build cumulative distribution function
    // CDF[i] = fraction of spikes that occurred before time i*binSize
    size_t spikeIdx = 0;
    for (int bin = 0; bin < bins; ++bin) {
        double binEnd = (bin + 1) * binSize;

        // Count spikes up to this bin
        while (spikeIdx < sortedSpikes.size() && sortedSpikes[spikeIdx] < binEnd) {
            spikeIdx++;
        }

        // Normalize by total spike count
        cdf[bin] = static_cast<double>(spikeIdx) / sortedSpikes.size();
    }

    return cdf;
}

// Compute similarity between two CDFs using hybrid approach:
// 1. Direct CDF similarity (captures overall temporal distribution)
// 2. First derivative similarity (captures spike rate changes)
// 3. L1 distance penalty (penalizes different distributions)
double Neuron::histogramSimilarity(const std::vector<double>& cdf1, const std::vector<double>& cdf2) const {
    if (cdf1.size() < 2 || cdf2.size() < 2) return 0.0;

    // 1. Cosine similarity on CDFs directly
    double dot_cdf = 0.0, norm1_cdf = 0.0, norm2_cdf = 0.0;
    for (size_t i = 0; i < cdf1.size(); ++i) {
        dot_cdf += cdf1[i] * cdf2[i];
        norm1_cdf += cdf1[i] * cdf1[i];
        norm2_cdf += cdf2[i] * cdf2[i];
    }
    double cdf_sim = 0.0;
    if (norm1_cdf > 1e-10 && norm2_cdf > 1e-10) {
        cdf_sim = dot_cdf / (std::sqrt(norm1_cdf) * std::sqrt(norm2_cdf));
    }

    // 2. Compute first derivatives (spike rate) and their similarity
    std::vector<double> deriv1, deriv2;
    for (size_t i = 0; i < cdf1.size() - 1; ++i) {
        deriv1.push_back(cdf1[i+1] - cdf1[i]);
        deriv2.push_back(cdf2[i+1] - cdf2[i]);
    }

    double dot_deriv = 0.0, norm1_deriv = 0.0, norm2_deriv = 0.0;
    for (size_t i = 0; i < deriv1.size(); ++i) {
        dot_deriv += deriv1[i] * deriv2[i];
        norm1_deriv += deriv1[i] * deriv1[i];
        norm2_deriv += deriv2[i] * deriv2[i];
    }
    double deriv_sim = 0.0;
    if (norm1_deriv > 1e-10 && norm2_deriv > 1e-10) {
        deriv_sim = dot_deriv / (std::sqrt(norm1_deriv) * std::sqrt(norm2_deriv));
    }

    // 3. L1 distance (lower is better, so invert it)
    double l1_dist = 0.0;
    for (size_t i = 0; i < cdf1.size(); ++i) {
        l1_dist += std::abs(cdf1[i] - cdf2[i]);
    }
    double l1_sim = 1.0 / (1.0 + l1_dist);  // Convert distance to similarity

    // Weighted combination: emphasize derivative (temporal structure)
    return 0.3 * cdf_sim + 0.5 * deriv_sim + 0.2 * l1_sim;
}

double Neuron::computeSimilarity(const BinaryPattern& a, const BinaryPattern& b) const {
    switch (similarityMetric_) {
        case SimilarityMetric::COSINE:
            return BinaryPattern::cosineSimilarity(a, b);
        case SimilarityMetric::HISTOGRAM:
            return BinaryPattern::histogramIntersection(a, b);
        case SimilarityMetric::EUCLIDEAN:
            return BinaryPattern::euclideanSimilarity(a, b);
        case SimilarityMetric::CORRELATION:
            return BinaryPattern::correlationSimilarity(a, b);
        case SimilarityMetric::WAVEFORM:
            return BinaryPattern::waveformSimilarity(a, b);
        default:
            return BinaryPattern::cosineSimilarity(a, b);
    }
}

bool Neuron::shouldFire() const {
    // Convert current spikes to BinaryPattern for comparison
    BinaryPattern currentPattern(spikes, windowSize);

    for (const auto& refPattern : referencePatterns) {
        // Compare using selected similarity metric
        double similarity = computeSimilarity(currentPattern, refPattern);
        if (similarity >= threshold) {
            return true;
        }
    }
    return false;
}

double Neuron::getBestSimilarity() const {
    // Copy spikes under lock
    std::vector<double> spikesCopy;
    {
        std::lock_guard<std::mutex> lock(spikesMutex_);
        if (spikes.empty() || referencePatterns.empty()) {
            return 0.0;
        }
        spikesCopy = spikes;
    }

    // Convert current spikes to BinaryPattern
    BinaryPattern currentPattern(spikesCopy, windowSize);

    double bestSim = -1.0;

    // Compare with all reference patterns using selected similarity metric
    for (const auto& refPattern : referencePatterns) {
        // Skip empty reference patterns
        if (refPattern.isEmpty()) continue;

        // Compute similarity using selected metric
        double similarity = computeSimilarity(currentPattern, refPattern);

        if (similarity > bestSim) {
            bestSim = similarity;
        }
    }
    return (bestSim < 0.0) ? 0.0 : bestSim;
}

int Neuron::findMostSimilarPattern(const std::vector<double>& newPattern) const {
    // Convert newPattern to BinaryPattern for comparison
    BinaryPattern newBinaryPattern(newPattern, windowSize);

    int bestIndex = -1;
    double bestSim = -1.0;

    for (size_t i = 0; i < referencePatterns.size(); ++i) {
        const auto& ref = referencePatterns[i];

        // Use selected similarity metric
        double sim = computeSimilarity(ref, newBinaryPattern);
        if (sim > bestSim) {
            bestSim = sim;
            bestIndex = static_cast<int>(i);
        }
    }
    return bestIndex;
}

void Neuron::blendPattern(std::vector<double>& target, const std::vector<double>& newPattern, double alpha) {
    for (size_t i = 0; i < target.size(); ++i) {
        target[i] = (1.0 - alpha) * target[i] + alpha * newPattern[i];
    }
}

double Neuron::cosineSimilarity(const std::vector<double>& a, const std::vector<double>& b) {
    double dot = 0.0, normA = 0.0, normB = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }
    if (normA == 0.0 || normB == 0.0) return 0.0;
    return dot / (std::sqrt(normA) * std::sqrt(normB));
}

// Compute spike distance using a simplified Victor-Purpura-like metric
// Returns a distance (lower is better), which we convert to similarity
double Neuron::spikeDistance(const std::vector<double>& spikes1, const std::vector<double>& spikes2) const {
    // Cost parameter: how much it costs to shift a spike by 1ms
    const double q = 0.5;

    // Handle empty cases
    if (spikes1.empty() && spikes2.empty()) return 0.0;
    if (spikes1.empty()) return static_cast<double>(spikes2.size());
    if (spikes2.empty()) return static_cast<double>(spikes1.size());

    // Simplified greedy matching: for each spike in spikes1, find closest in spikes2
    double totalCost = 0.0;
    std::vector<bool> matched2(spikes2.size(), false);

    for (double s1 : spikes1) {
        double minDist = std::numeric_limits<double>::max();
        int bestMatch = -1;

        // Find closest unmatched spike in spikes2
        for (size_t j = 0; j < spikes2.size(); ++j) {
            if (!matched2[j]) {
                double dist = std::abs(s1 - spikes2[j]);
                if (dist < minDist) {
                    minDist = dist;
                    bestMatch = static_cast<int>(j);
                }
            }
        }

        // Cost is either: shift cost (q * time_diff) or delete+insert cost (2.0)
        if (bestMatch >= 0) {
            double shiftCost = q * minDist;
            double deleteInsertCost = 2.0;

            if (shiftCost < deleteInsertCost) {
                totalCost += shiftCost;
                matched2[bestMatch] = true;
            } else {
                totalCost += deleteInsertCost;
            }
        } else {
            totalCost += 1.0;  // Delete cost
        }
    }

    // Add cost for unmatched spikes in spikes2 (insert cost)
    for (bool matched : matched2) {
        if (!matched) totalCost += 1.0;
    }

    return totalCost;
}

void Neuron::addDendrite(uint64_t dendriteId) {
    // Check if dendrite is already connected
    auto it = std::find(dendriteIds.begin(), dendriteIds.end(), dendriteId);
    if (it == dendriteIds.end()) {
        dendriteIds.push_back(dendriteId);
        SNNFW_DEBUG("Neuron {}: Added dendrite {} (total: {})",
                    getId(), dendriteId, dendriteIds.size());
    } else {
        SNNFW_WARN("Neuron {}: Dendrite {} already connected", getId(), dendriteId);
    }
}

bool Neuron::removeDendrite(uint64_t dendriteId) {
    auto it = std::find(dendriteIds.begin(), dendriteIds.end(), dendriteId);
    if (it != dendriteIds.end()) {
        dendriteIds.erase(it);
        SNNFW_DEBUG("Neuron {}: Removed dendrite {} (remaining: {})",
                    getId(), dendriteId, dendriteIds.size());
        return true;
    }
    SNNFW_WARN("Neuron {}: Dendrite {} not found for removal", getId(), dendriteId);
    return false;
}

std::string Neuron::toJson() const {
    json j;
    j["type"] = "Neuron";
    j["id"] = getId();
    j["windowSize"] = windowSize;
    j["threshold"] = threshold;
    j["maxPatterns"] = maxPatterns;
    j["axonId"] = axonId;
    j["dendriteIds"] = dendriteIds;
    j["spikes"] = spikes;

    // Serialize position if set
    if (hasPosition()) {
        const Position3D& pos = getPosition();
        j["position"] = {
            {"x", pos.x},
            {"y", pos.y},
            {"z", pos.z}
        };
    }

    // Serialize BinaryPatterns as arrays of spike counts
    json patternsJson = json::array();
    for (const auto& pattern : referencePatterns) {
        json patternJson = json::array();
        for (size_t i = 0; i < BinaryPattern::PATTERN_SIZE; ++i) {
            patternJson.push_back(pattern[i]);
        }
        patternsJson.push_back(patternJson);
    }
    j["referencePatterns"] = patternsJson;

    return j.dump();
}

bool Neuron::fromJson(const std::string& jsonStr) {
    try {
        json j = json::parse(jsonStr);

        // Validate type
        if (j["type"] != "Neuron") {
            SNNFW_ERROR("Invalid type in JSON: expected 'Neuron', got '{}'", j["type"].get<std::string>());
            return false;
        }

        // Deserialize ID from base class
        setId(j["id"]);

        // Deserialize position if present
        if (j.contains("position")) {
            float x = j["position"]["x"].get<float>();
            float y = j["position"]["y"].get<float>();
            float z = j["position"]["z"].get<float>();
            setPosition(x, y, z);
        } else {
            clearPosition();
        }

        // Deserialize fields
        windowSize = j["windowSize"];
        threshold = j["threshold"];
        maxPatterns = j["maxPatterns"];
        axonId = j["axonId"];
        dendriteIds = j["dendriteIds"].get<std::vector<uint64_t>>();
        spikes = j["spikes"].get<std::vector<double>>();

        // Deserialize BinaryPatterns from arrays of spike counts
        referencePatterns.clear();
        if (j.contains("referencePatterns")) {
            for (const auto& patternJson : j["referencePatterns"]) {
                BinaryPattern pattern;
                for (size_t i = 0; i < BinaryPattern::PATTERN_SIZE && i < patternJson.size(); ++i) {
                    pattern[i] = patternJson[i].get<uint8_t>();
                }
                referencePatterns.push_back(pattern);
            }
        }

        return true;
    } catch (const std::exception& e) {
        SNNFW_ERROR("Failed to deserialize Neuron from JSON: {}", e.what());
        return false;
    }
}

void Neuron::recordIncomingSpike(uint64_t synapseId, double spikeTime, double dispatchTime) {
    std::lock_guard<std::mutex> lock(incomingSpikesMutex_);

    // Add the incoming spike to our tracking deque
    incomingSpikes_.emplace_back(synapseId, spikeTime, dispatchTime);

    // Clean up old spikes outside the temporal window
    clearOldIncomingSpikes(spikeTime);

    SNNFW_TRACE("Neuron {}: Recorded incoming spike from synapse {} at time {:.3f}ms (dispatch: {:.3f}ms, total tracked: {})",
                getId(), synapseId, spikeTime, dispatchTime, incomingSpikes_.size());
}

int Neuron::fireAndAcknowledge(double firingTime) {
    auto propagator = networkPropagator_.lock();
    if (!propagator) {
        SNNFW_WARN("Neuron {}: Cannot send acknowledgments - NetworkPropagator not set", getId());
        return 0;
    }

    int acknowledgmentCount = 0;

    // Send acknowledgments to all presynaptic neurons that contributed spikes
    // within the temporal window
    // Note: We make a copy of the spikes under lock to avoid holding the lock during network operations
    std::vector<IncomingSpike> spikesToAcknowledge;
    {
        std::lock_guard<std::mutex> lock(incomingSpikesMutex_);
        spikesToAcknowledge.assign(incomingSpikes_.begin(), incomingSpikes_.end());
    }

    for (const auto& incomingSpike : spikesToAcknowledge) {
        // Create acknowledgment with timing information
        auto ack = std::make_shared<SpikeAcknowledgment>(
            incomingSpike.synapseId,
            getId(),  // postsynaptic neuron ID
            firingTime,
            incomingSpike.arrivalTime
        );

        // Send acknowledgment back through the NetworkPropagator
        propagator->sendAcknowledgment(ack);
        acknowledgmentCount++;

        SNNFW_TRACE("Neuron {}: Sent acknowledgment for synapse {} (Δt = {:.3f}ms)",
                    getId(), incomingSpike.synapseId, ack->getTimeDifference());
    }

    // Update firing rate statistics for homeostatic plasticity
    updateFiringRate(firingTime);

    SNNFW_DEBUG("Neuron {}: Fired at {:.3f}ms and sent {} acknowledgments",
                getId(), firingTime, acknowledgmentCount);

    return acknowledgmentCount;
}

void Neuron::clearOldIncomingSpikes(double currentTime) {
    // Note: Caller must hold incomingSpikesMutex_ lock
    // Remove spikes that are older than the temporal window
    while (!incomingSpikes_.empty() &&
           (currentTime - incomingSpikes_.front().arrivalTime > windowSize)) {
        incomingSpikes_.pop_front();
    }
}

void Neuron::periodicMemoryCleanup(double currentTime) {
    // Clear old spikes from the rolling window and shrink container
    size_t spikesCount = 0;
    {
        std::lock_guard<std::mutex> lock(spikesMutex_);
        removeOldSpikesUnsafe(currentTime);
        spikes.shrink_to_fit();
        spikesCount = spikes.size();
    }

    // Clear old incoming spikes for STDP and shrink container
    size_t incomingSpikesCount = 0;
    {
        std::lock_guard<std::mutex> lock(incomingSpikesMutex_);
        clearOldIncomingSpikes(currentTime);
        incomingSpikes_.shrink_to_fit();
        incomingSpikesCount = incomingSpikes_.size();
    }

    // Shrink pattern storage to fit (BinaryPattern is fixed size, so just shrink the vector)
    referencePatterns.shrink_to_fit();
    // Note: Each BinaryPattern is already fixed at 200 bytes, no need to shrink individual patterns

    SNNFW_TRACE("Neuron {}: Memory cleanup - {} spikes, {} incoming spikes, {} patterns ({}KB pattern memory)",
                getId(), spikesCount, incomingSpikesCount, referencePatterns.size(),
                (referencePatterns.size() * 200) / 1024);
}

void Neuron::generateTemporalSignature() {
    // Use neuron ID as seed for reproducibility
    std::mt19937 rng(getId());

    // Generate 1-10 spikes
    std::uniform_int_distribution<int> spikeCountDist(1, 10);
    int numSpikes = spikeCountDist(rng);

    // Spread spikes over 0-100ms
    std::uniform_real_distribution<double> timingDist(0.0, 100.0);

    temporalSignature_.clear();
    temporalSignature_.reserve(numSpikes);

    for (int i = 0; i < numSpikes; ++i) {
        temporalSignature_.push_back(timingDist(rng));
    }

    // Sort the offsets for easier processing
    std::sort(temporalSignature_.begin(), temporalSignature_.end());

    SNNFW_TRACE("Neuron {}: Generated temporal signature with {} spikes over {:.1f}ms",
                getId(), numSpikes,
                temporalSignature_.empty() ? 0.0 : (temporalSignature_.back() - temporalSignature_.front()));
}

void Neuron::fireSignature(double baseTime) {
    // Insert spikes according to this neuron's unique temporal signature
    for (double offset : temporalSignature_) {
        insertSpike(baseTime + offset);
    }

    SNNFW_TRACE("Neuron {}: Fired signature pattern with {} spikes starting at {:.3f}ms",
                getId(), temporalSignature_.size(), baseTime);
}

void Neuron::applyInhibition(double amount) {
    inhibition_ += amount;
    SNNFW_TRACE("Neuron {}: Applied inhibition {:.4f}, total inhibition: {:.4f}",
                getId(), amount, inhibition_);
}

double Neuron::getActivation() const {
    double similarity = getBestSimilarity();
    if (similarity < 0.0) {
        return 0.0;  // No patterns learned yet
    }
    // Activation = (similarity * intrinsic excitability) - inhibition (clamped to [0, 1])
    double activation = (similarity * intrinsicExcitability_) - inhibition_;
    return std::max(0.0, std::min(1.0, activation));
}

void Neuron::updateFiringRate(double currentTime) {
    const double MEASUREMENT_WINDOW = 1000.0;  // 1 second window in ms

    // Initialize window if needed
    if (firingWindowStart_ == 0.0) {
        firingWindowStart_ = currentTime;
    }

    // Record this firing
    firingCount_++;
    lastFiringTime_ = currentTime;

    // Calculate firing rate if we have enough time elapsed
    double elapsed = currentTime - firingWindowStart_;
    if (elapsed >= MEASUREMENT_WINDOW) {
        // Calculate rate in Hz (firings per second)
        firingRate_ = (firingCount_ / elapsed) * 1000.0;

        // Reset window
        firingCount_ = 0;
        firingWindowStart_ = currentTime;

        SNNFW_TRACE("Neuron {}: Firing rate updated to {:.2f} Hz (target: {:.2f} Hz)",
                    getId(), firingRate_, targetFiringRate_);
    }
}

void Neuron::applyHomeostaticPlasticity() {
    const double LEARNING_RATE = 0.01;  // How quickly to adjust excitability
    const double MIN_EXCITABILITY = 0.5;
    const double MAX_EXCITABILITY = 2.0;

    if (firingRate_ == 0.0) {
        return;  // Not enough data yet
    }

    // Calculate error: how far are we from target?
    double error = targetFiringRate_ - firingRate_;

    // Adjust intrinsic excitability based on error
    // If firing too little (error > 0), increase excitability
    // If firing too much (error < 0), decrease excitability
    double adjustment = LEARNING_RATE * error;
    intrinsicExcitability_ += adjustment;

    // Clamp to reasonable bounds
    intrinsicExcitability_ = std::max(MIN_EXCITABILITY, std::min(MAX_EXCITABILITY, intrinsicExcitability_));

    SNNFW_DEBUG("Neuron {}: Homeostatic adjustment - firing rate: {:.2f} Hz, target: {:.2f} Hz, excitability: {:.3f}",
                getId(), firingRate_, targetFiringRate_, intrinsicExcitability_);
}

} // namespace snnfw