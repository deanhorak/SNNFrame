#include "snnfw/features/EdgeOperator.h"
#include "snnfw/features/SobelOperator.h"
#include "snnfw/features/GaborOperator.h"
#include "snnfw/features/DoGOperator.h"
#include <stdexcept>
#include <algorithm>

namespace snnfw {
namespace features {

std::unique_ptr<EdgeOperator> EdgeOperatorFactory::create(
    const std::string& type,
    const EdgeOperator::Config& config) {
    
    // Convert type to lowercase for case-insensitive comparison
    std::string lowerType = type;
    std::transform(lowerType.begin(), lowerType.end(), lowerType.begin(), ::tolower);
    
    if (lowerType == "sobel") {
        return std::make_unique<SobelOperator>(config);
    } else if (lowerType == "gabor") {
        return std::make_unique<GaborOperator>(config);
    } else if (lowerType == "dog" || lowerType == "difference_of_gaussians") {
        return std::make_unique<DoGOperator>(config);
    } else {
        throw std::invalid_argument("Unknown edge operator type: " + type);
    }
}

std::vector<std::string> EdgeOperatorFactory::getAvailableOperators() {
    return {
        "sobel",
        "gabor",
        "dog",
        "difference_of_gaussians"
    };
}

} // namespace features
} // namespace snnfw

