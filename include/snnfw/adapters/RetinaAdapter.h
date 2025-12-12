#ifndef SNNFW_RETINA_ADAPTER_H
#define SNNFW_RETINA_ADAPTER_H

#include "snnfw/adapters/SensoryAdapter.h"
#include "snnfw/Neuron.h"
#include "snnfw/features/EdgeOperator.h"
#include "snnfw/encoding/EncodingStrategy.h"
#include <vector>
#include <memory>
#include <cmath>

namespace snnfw {
namespace adapters {

/**
 * @brief Retina adapter for processing visual input with pluggable edge detection and encoding
 *
 * The RetinaAdapter mimics the early visual processing in the retina and V1 cortex:
 * - Spatial grid decomposition (receptive fields)
 * - Pluggable edge detection with multiple orientations (simple cells)
 * - Pluggable spike encoding strategies (rate, temporal, population coding)
 * - Population of orientation-selective neurons
 *
 * Architecture:
 * - Input: Grayscale images (any size, typically 28×28 for MNIST)
 * - Spatial Grid: Divides image into regions (e.g., 8×8 grid for optimal MNIST accuracy)
 * - Edge Detectors: Multiple orientations per region (8 orientations recommended)
 * - Neurons: One neuron per (region, orientation) pair
 * - Output: Spike patterns encoding edge features
 *
 * Performance (MNIST):
 * - 8×8 grid + Sobel + Rate: 94.63% accuracy (current best)
 * - 7×7 grid + Sobel + Rate: 92.71% accuracy
 * - 8×8 grid + Gabor + Rate: 87.20% accuracy (Gabor worse for sharp edges)
 *
 * Configuration Parameters:
 * - grid_size: Number of regions per dimension (8 recommended for MNIST)
 * - num_orientations: Number of edge orientations (8 recommended)
 * - edge_threshold: Minimum edge strength to generate spikes (0.15 default)
 * - temporal_window: Duration of spike pattern in ms (200.0 default)
 * - neuron_window_size: Temporal window for pattern learning in ms (200.0 default)
 * - neuron_threshold: Similarity threshold for pattern matching (0.7 default)
 * - neuron_max_patterns: Maximum patterns per neuron (100 default)
 * - edge_operator: Type of edge operator ("sobel", "gabor", "dog") [default: "sobel"]
 * - encoding_strategy: Type of encoding ("rate", "temporal", "population") [default: "rate"]
 *
 * Edge Operator Parameters (in edge_operator_params):
 * - wavelength: Gabor wavelength (default: 4.0)
 * - sigma: Gabor/DoG sigma (default: 2.0)
 * - gamma: Gabor aspect ratio (default: 0.5)
 * - kernel_size: Filter kernel size (default: 5)
 *
 * Encoding Strategy Parameters (in encoding_params):
 * - dual_spike_mode: Temporal encoder dual spike (default: false)
 * - population_size: Population encoder size (default: 5)
 *
 * Usage:
 * @code
 * // For optimal MNIST accuracy (94.63%)
 * BaseAdapter::Config config;
 * config.type = "retina";
 * config.name = "visual_cortex";
 * config.setIntParam("grid_size", 8);  // 8×8 grid for best accuracy
 * config.setIntParam("num_orientations", 8);
 * config.setDoubleParam("edge_threshold", 0.15);
 * config.setStringParam("edge_operator", "sobel");  // Best for MNIST
 * config.setStringParam("encoding_strategy", "rate");
 *
 * RetinaAdapter retina(config);
 * retina.initialize();
 *
 * // Process image
 * SensoryAdapter::DataSample sample;
 * sample.rawData = imagePixels; // 28×28 grayscale MNIST image
 * auto spikes = retina.processData(sample);
 *
 * // Get activation pattern for classification
 * auto activations = retina.getActivationPattern();  // 512-dimensional vector (8×8×8)
 * @endcode
 */
class RetinaAdapter : public SensoryAdapter {
public:
    /**
     * @brief Image structure for visual input
     */
    struct Image {
        std::vector<uint8_t> pixels;  ///< Pixel values (0-255)
        int rows;                      ///< Image height
        int cols;                      ///< Image width
        
        uint8_t getPixel(int row, int col) const {
            if (row < 0 || row >= rows || col < 0 || col >= cols) return 0;
            return pixels[row * cols + col];
        }
        
        double getNormalizedPixel(int row, int col) const {
            return static_cast<double>(getPixel(row, col)) / 255.0;
        }
    };

    /**
     * @brief Constructor
     * @param config Adapter configuration
     */
    explicit RetinaAdapter(const Config& config);

    /**
     * @brief Destructor
     */
    ~RetinaAdapter() override = default;

    /**
     * @brief Initialize the adapter and create neuron population
     */
    bool initialize() override;

    /**
     * @brief Process image data and generate spike patterns
     * @param data Input data (must contain image pixels)
     * @return Spike pattern from all neurons
     */
    SpikePattern processData(const DataSample& data) override;

    /**
     * @brief Extract edge features from image
     * @param data Input image data
     * @return Feature vector containing edge strengths
     */
    FeatureVector extractFeatures(const DataSample& data) override;

    /**
     * @brief Encode edge features as spike patterns
     * @param features Edge feature vector
     * @return Spike pattern encoding the features
     */
    SpikePattern encodeFeatures(const FeatureVector& features) override;

    /**
     * @brief Get the neuron population
     */
    const std::vector<std::shared_ptr<Neuron>>& getNeurons() const override {
        return neurons_;
    }

    /**
     * @brief Get activation pattern from neurons
     * @return Vector of activation values (one per neuron)
     */
    std::vector<double> getActivationPattern() const override;

    /**
     * @brief Get neuron count
     */
    size_t getNeuronCount() const override {
        return neurons_.size();
    }

    /**
     * @brief Get feature dimension
     */
    size_t getFeatureDimension() const override {
        return gridSize_ * gridSize_ * numOrientations_;
    }

    /**
     * @brief Clear all neuron states
     */
    void clearNeuronStates() override;

    /**
     * @brief Get neuron at specific grid position and orientation
     * @param row Grid row
     * @param col Grid column
     * @param orientation Orientation index (0 to numOrientations-1)
     * @return Shared pointer to neuron, or nullptr if invalid indices
     */
    std::shared_ptr<Neuron> getNeuronAt(int row, int col, int orientation) const;

    /**
     * @brief Process image and get activation pattern (convenience method)
     * @param image Input image
     * @return Activation pattern vector
     */
    std::vector<double> processImage(const Image& image);

private:
    // Configuration parameters
    int gridSize_;              ///< Grid size (e.g., 7 for 7×7)
    int regionSize_;            ///< Region size in pixels
    int numOrientations_;       ///< Number of edge orientations
    double edgeThreshold_;      ///< Minimum edge strength
    double temporalWindow_;     ///< Spike pattern duration (ms)

    // Neuron parameters
    double neuronWindowSize_;   ///< Neuron temporal window (ms)
    double neuronThreshold_;    ///< Neuron similarity threshold
    int neuronMaxPatterns_;     ///< Max patterns per neuron

    // Pluggable strategies
    std::unique_ptr<features::EdgeOperator> edgeOperator_;      ///< Edge detection strategy
    std::unique_ptr<encoding::EncodingStrategy> encodingStrategy_; ///< Spike encoding strategy

    // Neuron population
    // Structure: neurons_[region_row * gridSize + region_col][orientation]
    std::vector<std::vector<std::shared_ptr<Neuron>>> neuronGrid_;
    std::vector<std::shared_ptr<Neuron>> neurons_;  ///< Flat list for easy access

    // Image dimensions (set during first processData call)
    int imageRows_;
    int imageCols_;

    /**
     * @brief Extract a region from the image
     * @param image Input image
     * @param regionRow Region row index
     * @param regionCol Region column index
     * @return Vector of pixel values in the region
     */
    std::vector<uint8_t> extractRegion(const Image& image, int regionRow, int regionCol) const;

    /**
     * @brief Extract edge features for a region
     * @param region Region pixel values
     * @param regionSize Region size in pixels
     * @return Vector of edge strengths (one per orientation)
     */
    std::vector<double> extractEdgeFeatures(const std::vector<uint8_t>& region, 
                                            int regionSize) const;

    /**
     * @brief Convert feature values to spike times
     * @param features Feature values (0.0 to 1.0)
     * @return Vector of spike times
     */
    std::vector<double> featuresToSpikes(const std::vector<double>& features) const;

    /**
     * @brief Create neuron population
     */
    void createNeurons();
};

} // namespace adapters
} // namespace snnfw

#endif // SNNFW_RETINA_ADAPTER_H

