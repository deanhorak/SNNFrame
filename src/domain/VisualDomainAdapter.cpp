#include "snnfw/domain/VisualDomainAdapter.h"

#include "snnfw/EMNISTLoader.h"
#include "snnfw/MNISTLoader.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace snnfw::domain {

namespace {

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return value;
}

std::vector<std::string> makeDigitClassNames(int count) {
    std::vector<std::string> names;
    names.reserve(static_cast<size_t>(std::max(0, count)));
    for (int i = 0; i < count; ++i) {
        names.push_back(std::to_string(i));
    }
    return names;
}

std::vector<std::string> makeLetterClassNames() {
    std::vector<std::string> names;
    names.reserve(26);
    for (char c = 'A'; c <= 'Z'; ++c) {
        names.emplace_back(1, c);
    }
    return names;
}

class EMNISTDomainAdapter final : public VisualDomainAdapter {
public:
    explicit EMNISTDomainAdapter(const VisualDomainConfig& config)
        : variant_(parseVariant(config.variant)),
          applyTransform_(config.applyTransform),
          loader_(variant_) {}

    bool load(const std::string& imageFile,
              const std::string& labelFile,
              size_t maxImages = 0) override {
        if (!loader_.load(imageFile, labelFile, maxImages, applyTransform_)) {
            return false;
        }

        stimuli_.clear();
        stimuli_.reserve(loader_.size());
        for (size_t i = 0; i < loader_.size(); ++i) {
            const auto& image = loader_.getImage(i);
            VisualStimulus stimulus;
            stimulus.pixels = image.pixels;
            stimulus.rows = image.rows;
            stimulus.cols = image.cols;
            stimulus.timestamp = static_cast<double>(i);
            stimulus.label = normalizeLabel(image.label);
            stimuli_.push_back(std::move(stimulus));
        }
        return true;
    }

    size_t size() const override {
        return stimuli_.size();
    }

    const VisualStimulus& getStimulus(size_t index) const override {
        return stimuli_[index];
    }

    int numClasses() const override {
        return loader_.getNumClasses();
    }

    std::vector<std::string> classNames() const override {
        switch (variant_) {
            case EMNISTLoader::Variant::LETTERS:
                return makeLetterClassNames();
            case EMNISTLoader::Variant::DIGITS:
                return makeDigitClassNames(10);
            case EMNISTLoader::Variant::BALANCED:
            case EMNISTLoader::Variant::BYMERGE:
                return makeDigitClassNames(loader_.getNumClasses());
            case EMNISTLoader::Variant::BYCLASS:
                return makeDigitClassNames(loader_.getNumClasses());
            default:
                return makeDigitClassNames(loader_.getNumClasses());
        }
    }

    std::string domainName() const override {
        return "EMNIST " + loader_.getVariantName();
    }

private:
    static EMNISTLoader::Variant parseVariant(const std::string& variant) {
        const std::string lowered = toLower(variant);
        if (lowered.empty() || lowered == "letters") {
            return EMNISTLoader::Variant::LETTERS;
        }
        if (lowered == "digits") {
            return EMNISTLoader::Variant::DIGITS;
        }
        if (lowered == "balanced") {
            return EMNISTLoader::Variant::BALANCED;
        }
        if (lowered == "byclass") {
            return EMNISTLoader::Variant::BYCLASS;
        }
        if (lowered == "bymerge") {
            return EMNISTLoader::Variant::BYMERGE;
        }
        throw std::runtime_error("Unsupported EMNIST variant: " + variant);
    }

    int normalizeLabel(uint8_t rawLabel) const {
        if (variant_ == EMNISTLoader::Variant::LETTERS) {
            return static_cast<int>(rawLabel) - 1;
        }
        return static_cast<int>(rawLabel);
    }

    EMNISTLoader::Variant variant_;
    bool applyTransform_ = true;
    EMNISTLoader loader_;
    std::vector<VisualStimulus> stimuli_;
};

class MNISTDomainAdapter final : public VisualDomainAdapter {
public:
    explicit MNISTDomainAdapter(const VisualDomainConfig&) {}

    bool load(const std::string& imageFile,
              const std::string& labelFile,
              size_t maxImages = 0) override {
        if (!loader_.load(imageFile, labelFile, maxImages)) {
            return false;
        }

        stimuli_.clear();
        stimuli_.reserve(loader_.size());
        for (size_t i = 0; i < loader_.size(); ++i) {
            const auto& image = loader_.getImage(i);
            VisualStimulus stimulus;
            stimulus.pixels = image.pixels;
            stimulus.rows = image.rows;
            stimulus.cols = image.cols;
            stimulus.timestamp = static_cast<double>(i);
            stimulus.label = static_cast<int>(image.label);
            stimuli_.push_back(std::move(stimulus));
        }
        return true;
    }

    size_t size() const override {
        return stimuli_.size();
    }

    const VisualStimulus& getStimulus(size_t index) const override {
        return stimuli_[index];
    }

    int numClasses() const override {
        return 10;
    }

    std::vector<std::string> classNames() const override {
        return makeDigitClassNames(10);
    }

    std::string domainName() const override {
        return "MNIST Digits";
    }

private:
    MNISTLoader loader_;
    std::vector<VisualStimulus> stimuli_;
};

} // namespace

std::unique_ptr<VisualDomainAdapter> createVisualDomainAdapter(const VisualDomainConfig& config) {
    const std::string source = toLower(config.source);
    if (source.empty() || source == "emnist") {
        return std::make_unique<EMNISTDomainAdapter>(config);
    }
    if (source == "mnist") {
        return std::make_unique<MNISTDomainAdapter>(config);
    }
    throw std::runtime_error("Unsupported visual domain source: " + config.source);
}

} // namespace snnfw::domain
