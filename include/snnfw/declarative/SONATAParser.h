#ifndef SNNFW_DECLARATIVE_SONATA_PARSER_H
#define SNNFW_DECLARATIVE_SONATA_PARSER_H

#include "snnfw/declarative/FormatParser.h"
#include <nlohmann/json.hpp>
#include <string>
#include <map>

namespace snnfw {
namespace declarative {

/**
 * @class SONATAParser
 * @brief Parser for SONATA format neural network descriptions
 *
 * SONATA (Scalable Open Network Architecture TemplAte) is a standard format
 * for representing large-scale neural network models developed by the Blue Brain
 * Project and Allen Institute.
 *
 * The parser reads a circuit_config.json file which points to:
 *   - nodes HDF5 files (neuron populations with attributes)
 *   - edges HDF5 files (synaptic connectivity with weights/delays)
 *
 * File detection: circuit_config.json or *.sonata.json
 *
 * The mapping from SONATA to NetworkIR:
 *   - Node populations → Populations within a default hierarchy
 *   - Node attributes (window_size_ms, similarity_threshold) → NeuronParamsIR
 *   - Edge populations → ProjectionIR
 *   - Edge attributes (weight, delay) → Projection weight/delay
 */
class SONATAParser : public FormatParser {
public:
    NetworkIR parse(const std::string& primaryFile,
                    const std::string& auxiliaryFile = "") override;

    bool canParse(const std::string& filename) const override;
    std::string formatName() const override { return "sonata"; }

    /// Parse from an already-loaded circuit config JSON (useful for testing)
    NetworkIR parseCircuitConfig(const nlohmann::json& config,
                                 const std::string& baseDir = ".");

private:
    /// Resolve manifest variables like $BASE_DIR, $NETWORK_DIR
    std::string resolveManifest(const std::string& path,
                                const std::map<std::string, std::string>& manifest) const;

    /// Parse the manifest section of circuit_config.json
    std::map<std::string, std::string> parseManifest(const nlohmann::json& config) const;

    /// Load node populations from HDF5 and add to IR
    void loadNodes(const std::string& nodesFile,
                   const std::string& nodeTypesFile,
                   NetworkIR& ir) const;

    /// Load edge populations from HDF5 and add to IR
    void loadEdges(const std::string& edgesFile,
                   const std::string& edgeTypesFile,
                   NetworkIR& ir) const;

    /// Build a default hierarchy wrapper around flat populations
    void buildDefaultHierarchy(NetworkIR& ir,
                               const std::string& networkName) const;
};

} // namespace declarative
} // namespace snnfw

#endif // SNNFW_DECLARATIVE_SONATA_PARSER_H

