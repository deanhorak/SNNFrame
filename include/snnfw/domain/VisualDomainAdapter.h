#ifndef SNNFW_DOMAIN_VISUALDOMAINADAPTER_H
#define SNNFW_DOMAIN_VISUALDOMAINADAPTER_H

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace snnfw::domain {

struct VisualStimulus {
    std::vector<uint8_t> pixels;
    int label = -1;
    int rows = 0;
    int cols = 0;
    double timestamp = 0.0;

    uint8_t getPixel(int row, int col) const {
        return pixels[static_cast<size_t>(row * cols + col)];
    }

    double getNormalizedPixel(int row, int col) const {
        return static_cast<double>(getPixel(row, col)) / 255.0;
    }
};

struct VisualDomainConfig {
    std::string source = "emnist";
    std::string variant = "letters";
    bool applyTransform = true;
};

class VisualDomainAdapter {
public:
    virtual ~VisualDomainAdapter() = default;

    virtual bool load(const std::string& imageFile,
                      const std::string& labelFile,
                      size_t maxImages = 0) = 0;
    virtual size_t size() const = 0;
    virtual const VisualStimulus& getStimulus(size_t index) const = 0;
    virtual int numClasses() const = 0;
    virtual std::vector<std::string> classNames() const = 0;
    virtual std::string domainName() const = 0;
};

std::unique_ptr<VisualDomainAdapter> createVisualDomainAdapter(const VisualDomainConfig& config);

} // namespace snnfw::domain

#endif // SNNFW_DOMAIN_VISUALDOMAINADAPTER_H
