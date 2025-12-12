/**
 * @file RetinaAdapter.cpp
 * @brief Implementation of RetinaAdapter for visual processing with pluggable strategies
 *
 * This file implements the RetinaAdapter class, which provides biologically-inspired
 * visual processing for spiking neural networks. The adapter uses:
 * - Spatial grid decomposition to divide images into receptive fields
 * - Pluggable edge detection operators (Sobel, Gabor, DoG)
 * - Pluggable spike encoding strategies (Rate, Temporal, Population)
 * - Orientation-selective neurons mimicking V1 simple cells
 *
 * Performance on MNIST:
 * - 8×8 grid + Sobel + Rate: 94.63% accuracy (current best)
 * - 7×7 grid + Sobel + Rate: 92.71% accuracy
 * - Higher spatial resolution (more regions) improves accuracy
 * - Sobel operator outperforms Gabor for sharp-edged digits
 *
 * Key Design Decisions:
 * - Region size calculated as imageSize / gridSize (integer division)
 * - For 28×28 MNIST: 8×8 grid → 3×3 pixel regions (optimal)
 * - Each region has numOrientations neurons (typically 8)
 * - Total neurons = gridSize² × numOrientations (e.g., 8×8×8 = 512)
 */

#include "snnfw/adapters/RetinaAdapter.h"
#include "snnfw/features/EdgeOperator.h"
#include "snnfw/features/SobelOperator.h"
#include "snnfw/features/GaborOperator.h"
#include "snnfw/features/DoGOperator.h"
#include "snnfw/encoding/EncodingStrategy.h"
#include "snnfw/encoding/RateEncoder.h"
#include "snnfw/encoding/TemporalEncoder.h"
#include "snnfw/encoding/PopulationEncoder.h"
#include "snnfw/Logger.h"
#include <algorithm>
#include <cmath>

namespace snnfw {
namespace adapters {

/**
 * @brief Construct a RetinaAdapter with configuration
 *
 * Initializes the adapter with pluggable edge detection and encoding strategies.
 * The constructor:
 * 1. Loads configuration parameters (grid size, orientations, thresholds)
 * 2. Creates the edge operator (Sobel, Gabor, or DoG) based on config
 * 3. Creates the encoding strategy (Rate, Temporal, or Population) based on config
 * 4. Prepares for neuron creation (done in initialize())
 *
 * @param config Configuration containing all adapter parameters
 */
RetinaAdapter::RetinaAdapter(const Config& config)
    : SensoryAdapter(config)
    , gridSize_(0)
    , regionSize_(0)
    , numOrientations_(0)
    , edgeThreshold_(0.15)
    , temporalWindow_(100.0)
    , neuronWindowSize_(200.0)
    , neuronThreshold_(0.7)
    , neuronMaxPatterns_(100)
    , imageRows_(0)
    , imageCols_(0)
{
    // Load configuration parameters
    gridSize_ = getIntParam("grid_size", 7);
    numOrientations_ = getIntParam("num_orientations", 8);
    edgeThreshold_ = getDoubleParam("edge_threshold", 0.15);
    temporalWindow_ = config.temporalWindow > 0 ? config.temporalWindow : 100.0;

    neuronWindowSize_ = getDoubleParam("neuron_window_size", 200.0);
    neuronThreshold_ = getDoubleParam("neuron_threshold", 0.7);
    neuronMaxPatterns_ = getIntParam("neuron_max_patterns", 100);

    // Create edge operator
    std::string edgeOperatorType = getStringParam("edge_operator", "sobel");
    features::EdgeOperator::Config edgeConfig;
    edgeConfig.name = edgeOperatorType;
    edgeConfig.numOrientations = numOrientations_;
    edgeConfig.edgeThreshold = edgeThreshold_;

    // Copy edge operator parameters from config
    if (config.stringParams.count("edge_operator_params") > 0) {
        // Parse nested params if needed - for now use direct params
    }
    edgeConfig.doubleParams["wavelength"] = getDoubleParam("wavelength", 4.0);
    edgeConfig.doubleParams["sigma"] = getDoubleParam("sigma", 2.0);
    edgeConfig.doubleParams["gamma"] = getDoubleParam("gamma", 0.5);
    edgeConfig.doubleParams["phase_offset"] = getDoubleParam("phase_offset", 0.0);
    edgeConfig.doubleParams["sigma1"] = getDoubleParam("sigma1", 1.0);
    edgeConfig.doubleParams["sigma2"] = getDoubleParam("sigma2", 1.6);
    edgeConfig.intParams["kernel_size"] = getIntParam("kernel_size", 5);

    edgeOperator_ = features::EdgeOperatorFactory::create(edgeOperatorType, edgeConfig);

    // Create encoding strategy
    std::string encodingType = getStringParam("encoding_strategy", "rate");
    encoding::EncodingStrategy::Config encodingConfig;
    encodingConfig.name = encodingType;
    encodingConfig.temporalWindow = temporalWindow_;
    encodingConfig.baselineTime = 0.0;
    encodingConfig.intensityScale = temporalWindow_;

    // Copy encoding parameters
    encodingConfig.doubleParams["timing_jitter"] = getDoubleParam("timing_jitter", 0.0);
    encodingConfig.doubleParams["min_spike_interval"] = getDoubleParam("min_spike_interval", 5.0);
    encodingConfig.doubleParams["tuning_width"] = getDoubleParam("tuning_width", 0.3);
    encodingConfig.doubleParams["min_response"] = getDoubleParam("min_response", 0.1);
    encodingConfig.intParams["dual_spike_mode"] = getIntParam("dual_spike_mode", 0);
    encodingConfig.intParams["population_size"] = getIntParam("population_size", 5);

    encodingStrategy_ = encoding::EncodingStrategyFactory::create(encodingType, encodingConfig);

    SNNFW_INFO("RetinaAdapter '{}': grid={}x{}, orientations={}, threshold={}, edge={}, encoding={}",
               getName(), gridSize_, gridSize_, numOrientations_, edgeThreshold_,
               edgeOperatorType, encodingType);
}

bool RetinaAdapter::initialize() {
    if (!SensoryAdapter::initialize()) {
        return false;
    }
    
    createNeurons();
    
    SNNFW_INFO("RetinaAdapter '{}': initialized with {} neurons", 
               getName(), neurons_.size());
    return true;
}

/**
 * @brief Create the neuron population for the retina
 *
 * Creates a 2D grid of orientation-selective neurons:
 * - Each spatial region has numOrientations_ neurons (one per orientation)
 * - Total neurons = gridSize² × numOrientations_
 * - Example: 8×8 grid with 8 orientations = 512 neurons
 *
 * Neuron organization:
 * - neuronGrid_[region][orientation] = neuron for that (region, orientation) pair
 * - neurons_ = flat list of all neurons for easy iteration
 *
 * Each neuron stores temporal spike patterns for pattern-based learning.
 */
void RetinaAdapter::createNeurons() {
    // Clear existing neurons completely before creating new ones
    neurons_.clear();
    neuronGrid_.clear();

    int numRegions = gridSize_ * gridSize_;
    neuronGrid_.resize(numRegions);

    int neuronId = 0;
    for (int region = 0; region < numRegions; ++region) {
        neuronGrid_[region].resize(numOrientations_);
        for (int orient = 0; orient < numOrientations_; ++orient) {
            auto neuron = std::make_shared<Neuron>(
                neuronWindowSize_,      // Temporal window for pattern learning (ms)
                neuronThreshold_,       // Similarity threshold for pattern matching
                neuronMaxPatterns_,     // Maximum patterns to store per neuron
                neuronId++              // Unique neuron ID
            );
            neuronGrid_[region][orient] = neuron;
            neurons_.push_back(neuron);
        }
    }
}

/**
 * @brief Extract a rectangular region from the image
 *
 * Extracts a regionSize × regionSize block of pixels from the image.
 * The region is specified by its row and column indices in the grid.
 *
 * Region calculation:
 * - startRow = regionRow × regionSize
 * - startCol = regionCol × regionSize
 * - For 28×28 image with 8×8 grid: regionSize = 28/8 = 3 pixels
 *
 * @param image Input image
 * @param regionRow Row index of region in grid (0 to gridSize-1)
 * @param regionCol Column index of region in grid (0 to gridSize-1)
 * @return Flattened vector of pixel values (regionSize² elements)
 */
std::vector<uint8_t> RetinaAdapter::extractRegion(const Image& image,
                                                   int regionRow,
                                                   int regionCol) const {
    std::vector<uint8_t> region(regionSize_ * regionSize_);

    int startRow = regionRow * regionSize_;
    int startCol = regionCol * regionSize_;

    for (int r = 0; r < regionSize_; ++r) {
        for (int c = 0; c < regionSize_; ++c) {
            int imgRow = startRow + r;
            int imgCol = startCol + c;
            region[r * regionSize_ + c] = image.getPixel(imgRow, imgCol);
        }
    }

    return region;
}

/**
 * @brief Extract edge features from a region using pluggable edge operator
 *
 * Delegates edge detection to the configured EdgeOperator (Sobel, Gabor, or DoG).
 * The operator computes edge strength at multiple orientations.
 *
 * Performance comparison (MNIST):
 * - Sobel: 94.63% accuracy (best for sharp edges)
 * - Gabor: 87.20% accuracy (designed for natural images)
 * - DoG: Not yet tested
 *
 * @param region Pixel values of the region (regionSize² elements)
 * @param regionSize Size of the region (width = height)
 * @return Edge strengths for each orientation (numOrientations_ elements, range [0,1])
 */
std::vector<double> RetinaAdapter::extractEdgeFeatures(const std::vector<uint8_t>& region,
                                                        int regionSize) const {
    // Use pluggable edge operator (Sobel, Gabor, or DoG)
    return edgeOperator_->extractEdges(region, regionSize);
}

std::vector<double> RetinaAdapter::featuresToSpikes(const std::vector<double>& features) const {
    return featuresToSpikeTimes(features, temporalWindow_);
}

SensoryAdapter::SpikePattern RetinaAdapter::processData(const DataSample& data) {
    // Convert raw data to image
    Image image;
    image.pixels = data.rawData;
    
    // Infer image dimensions if not set
    if (imageRows_ == 0 || imageCols_ == 0) {
        // Assume square image
        int totalPixels = data.rawData.size();
        imageRows_ = imageCols_ = static_cast<int>(std::sqrt(totalPixels));
        regionSize_ = imageRows_ / gridSize_;
        
        SNNFW_INFO("RetinaAdapter '{}': inferred image size {}x{}, region size {}", 
                   getName(), imageRows_, imageCols_, regionSize_);
    }
    
    image.rows = imageRows_;
    image.cols = imageCols_;
    
    // Extract features and encode as spikes
    auto features = extractFeatures(data);
    return encodeFeatures(features);
}

SensoryAdapter::FeatureVector RetinaAdapter::extractFeatures(const DataSample& data) {
    FeatureVector result;
    result.timestamp = data.timestamp;
    
    // Convert to image
    Image image;
    image.pixels = data.rawData;
    image.rows = imageRows_;
    image.cols = imageCols_;
    
    // Extract features for each region
    for (int row = 0; row < gridSize_; ++row) {
        for (int col = 0; col < gridSize_; ++col) {
            auto region = extractRegion(image, row, col);
            auto edgeFeatures = extractEdgeFeatures(region, regionSize_);
            
            // Add to feature vector
            for (double feature : edgeFeatures) {
                result.features.push_back(feature);
            }
        }
    }
    
    return result;
}

SensoryAdapter::SpikePattern RetinaAdapter::encodeFeatures(const FeatureVector& features) {
    SpikePattern pattern;
    pattern.timestamp = features.timestamp;
    pattern.duration = temporalWindow_;
    pattern.spikeTimes.resize(neurons_.size());

    // Clear all neurons first
    clearNeuronStates();

    // Get neurons per feature from encoding strategy
    int neuronsPerFeature = encodingStrategy_->getNeuronsPerFeature();

    // Encode features as spikes and insert into neurons
    size_t featureIdx = 0;
    for (int row = 0; row < gridSize_; ++row) {
        for (int col = 0; col < gridSize_; ++col) {
            for (int orient = 0; orient < numOrientations_; ++orient) {
                double featureValue = features.features[featureIdx++];

                // Use encoding strategy to generate spike times
                std::vector<double> spikeTimes = encodingStrategy_->encode(featureValue, orient);

                // For rate/temporal encoding (1 neuron per feature), insert into single neuron
                if (neuronsPerFeature == 1) {
                    auto& neuron = neuronGrid_[row * gridSize_ + col][orient];
                    int neuronIdx = row * gridSize_ * numOrientations_ +
                                   col * numOrientations_ + orient;

                    for (double spikeTime : spikeTimes) {
                        neuron->insertSpike(spikeTime);
                        pattern.spikeTimes[neuronIdx].push_back(spikeTime);
                    }
                } else {
                    // For population encoding (multiple neurons per feature)
                    // Each spike goes to a different neuron in the population
                    // Note: This requires neuron structure to support population encoding
                    // For now, we'll distribute spikes across the single neuron
                    // TODO: Extend neuron grid to support population encoding
                    auto& neuron = neuronGrid_[row * gridSize_ + col][orient];
                    int neuronIdx = row * gridSize_ * numOrientations_ +
                                   col * numOrientations_ + orient;

                    for (double spikeTime : spikeTimes) {
                        neuron->insertSpike(spikeTime);
                        pattern.spikeTimes[neuronIdx].push_back(spikeTime);
                    }
                }
            }
        }
    }

    return pattern;
}

std::vector<double> RetinaAdapter::getActivationPattern() const {
    std::vector<double> activations(neurons_.size(), 0.0);

    for (size_t i = 0; i < neurons_.size(); ++i) {
        // PERFORMANCE OPTIMIZATION: During training, neurons accumulate patterns
        // and getBestSimilarity() becomes very expensive (100 pattern comparisons per neuron).
        // For 1536 neurons × 100 patterns = 153,600 comparisons per image!
        //
        // Instead, use a simple spike-based activation:
        // - If neuron has spikes: activation = 1.0 (binary activation)
        // - If no spikes: activation = 0.0
        //
        // This is appropriate for training where we just need to know which neurons
        // are active, not their similarity to learned patterns.
        //
        // For inference/testing, you may want to use getBestSimilarity() instead.

        activations[i] = neurons_[i]->getSpikes().empty() ? 0.0 : 1.0;
    }

    return activations;
}

void RetinaAdapter::clearNeuronStates() {
    for (auto& neuron : neurons_) {
        neuron->clearSpikes();
    }
}

std::shared_ptr<Neuron> RetinaAdapter::getNeuronAt(int row, int col, int orientation) const {
    if (row < 0 || row >= gridSize_ || col < 0 || col >= gridSize_ || 
        orientation < 0 || orientation >= numOrientations_) {
        return nullptr;
    }
    
    return neuronGrid_[row * gridSize_ + col][orientation];
}

std::vector<double> RetinaAdapter::processImage(const Image& image) {
    static int callCount = 0;
    callCount++;

    DataSample sample;
    sample.rawData = image.pixels;
    sample.timestamp = 0.0;

    // Set image dimensions
    imageRows_ = image.rows;
    imageCols_ = image.cols;
    regionSize_ = imageRows_ / gridSize_;

    if (callCount <= 6) SNNFW_INFO("processImage call #{}: About to call processData()", callCount);
    // Process and get activation pattern
    processData(sample);
    if (callCount <= 6) SNNFW_INFO("processImage call #{}: processData() done, calling getActivationPattern()", callCount);
    auto result = getActivationPattern();
    if (callCount <= 6) SNNFW_INFO("processImage call #{}: getActivationPattern() done", callCount);
    return result;
}

} // namespace adapters
} // namespace snnfw

