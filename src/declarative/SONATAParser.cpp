#include "snnfw/declarative/SONATAParser.h"
#include "snnfw/Logger.h"
#include <bbp/sonata/nodes.h>
#include <bbp/sonata/edges.h>
#include <fstream>
#include <filesystem>
#include <algorithm>

namespace snnfw {
namespace declarative {

bool SONATAParser::canParse(const std::string& filename) const {
    // Detect circuit_config.json or *.sonata.json
    namespace fs = std::filesystem;
    auto fn = fs::path(filename).filename().string();
    if (fn == "circuit_config.json") return true;
    const std::string ext = ".sonata.json";
    if (fn.size() >= ext.size()) {
        return fn.compare(fn.size() - ext.size(), ext.size(), ext) == 0;
    }
    return false;
}

NetworkIR SONATAParser::parse(const std::string& primaryFile,
                               const std::string& /*auxiliaryFile*/) {
    std::ifstream file(primaryFile);
    if (!file.is_open()) {
        throw ParseError("Cannot open SONATA config file: " + primaryFile);
    }
    nlohmann::json config;
    try {
        file >> config;
    } catch (const nlohmann::json::parse_error& e) {
        throw ParseError("JSON parse error in " + primaryFile + ": " + e.what());
    }

    auto baseDir = std::filesystem::path(primaryFile).parent_path().string();
    if (baseDir.empty()) baseDir = ".";

    return parseCircuitConfig(config, baseDir);
}

NetworkIR SONATAParser::parseCircuitConfig(const nlohmann::json& config,
                                            const std::string& baseDir) {
    NetworkIR ir;
    ir.sourceFormat = "sonata";

    // Parse manifest for variable resolution
    auto manifest = parseManifest(config);
    if (manifest.find("$BASE_DIR") == manifest.end()) {
        manifest["$BASE_DIR"] = baseDir;
    }

    // Extract network name
    std::string networkName = config.value("network_name", "SONATANetwork");

    // Parse SNNFrame-specific extensions if present
    if (config.contains("snnframe")) {
        const auto& snnfw = config["snnframe"];
        if (snnfw.contains("neuron_params")) {
            for (auto& [key, val] : snnfw["neuron_params"].items()) {
                NeuronParamsIR params;
                params.name = key;
                params.windowSizeMs = val.value("window_size_ms", 500.0);
                params.similarityThreshold = val.value("similarity_threshold", 0.93);
                params.maxReferencePatterns = val.value("max_reference_patterns", 500);
                params.similarityMetric = val.value("similarity_metric", "cosine");
                ir.neuronParamSets[key] = params;
            }
        }
        if (snnfw.contains("input_layer")) {
            const auto& il = snnfw["input_layer"];
            ir.inputLayer.rows = il.value("rows", 28);
            ir.inputLayer.cols = il.value("cols", 28);
            ir.inputLayer.latencyMs = il.value("latency_ms", 15.0);
        }
        if (snnfw.contains("output_layer")) {
            const auto& ol = snnfw["output_layer"];
            ir.outputLayer.numClasses = ol.value("num_classes", 26);
            ir.outputLayer.neuronsPerClass = ol.value("neurons_per_class", 3);
        }
        if (snnfw.contains("gabor")) {
            const auto& g = snnfw["gabor"];
            ir.gabor.freqLow = g.value("freq_low", 8.0);
            ir.gabor.freqHigh = g.value("freq_high", 3.0);
            ir.gabor.sigma = g.value("sigma", 2.0);
            ir.gabor.gamma = g.value("gamma", 0.5);
            ir.gabor.threshold = g.value("threshold", 0.5);
            ir.gabor.kernelSize = g.value("kernel_size", 7);
        }
    }

    // Load nodes and edges from HDF5 files if specified
    if (config.contains("networks")) {
        const auto& networks = config["networks"];
        if (networks.contains("nodes")) {
            for (const auto& nodeEntry : networks["nodes"]) {
                std::string nodesFile = resolveManifest(
                    nodeEntry.value("nodes_file", ""), manifest);
                std::string nodeTypesFile = resolveManifest(
                    nodeEntry.value("node_types_file", ""), manifest);
                if (!nodesFile.empty()) {
                    loadNodes(nodesFile, nodeTypesFile, ir);
                }
            }
        }
        if (networks.contains("edges")) {
            for (const auto& edgeEntry : networks["edges"]) {
                std::string edgesFile = resolveManifest(
                    edgeEntry.value("edges_file", ""), manifest);
                std::string edgeTypesFile = resolveManifest(
                    edgeEntry.value("edge_types_file", ""), manifest);
                if (!edgesFile.empty()) {
                    loadEdges(edgesFile, edgeTypesFile, ir);
                }
            }
        }
    }

    // Build default hierarchy if none was created by node loading
    if (ir.brain.hemispheres.empty()) {
        buildDefaultHierarchy(ir, networkName);
    }

    return ir;
}

std::map<std::string, std::string> SONATAParser::parseManifest(
        const nlohmann::json& config) const {
    std::map<std::string, std::string> manifest;
    if (config.contains("manifest")) {
        for (auto& [key, val] : config["manifest"].items()) {
            manifest[key] = val.get<std::string>();
        }
    }
    return manifest;
}

std::string SONATAParser::resolveManifest(
        const std::string& path,
        const std::map<std::string, std::string>& manifest) const {
    std::string resolved = path;
    // Iteratively resolve variables (handles nested references like $NETWORK_DIR -> $BASE_DIR/networks)
    bool changed = true;
    int iterations = 0;
    while (changed && iterations < 10) {
        changed = false;
        for (const auto& [var, val] : manifest) {
            auto pos = resolved.find(var);
            if (pos != std::string::npos) {
                resolved.replace(pos, var.size(), val);
                changed = true;
            }
        }
        ++iterations;
    }
    return resolved;
}

void SONATAParser::loadNodes(const std::string& nodesFile,
                              const std::string& /*nodeTypesFile*/,
                              NetworkIR& ir) const {
    if (!std::filesystem::exists(nodesFile)) {
        SNNFW_WARN("SONATA nodes file not found: {}", nodesFile);
        return;
    }

    try {
        bbp::sonata::NodeStorage storage(nodesFile);
        auto populationNames = storage.populationNames();

        for (const auto& popName : populationNames) {
            auto population = storage.openPopulation(popName);
            auto selection = population->selectAll();
            size_t nodeCount = selection.flatSize();

            SNNFW_INFO("SONATA: Loading node population '{}' with {} nodes",
                       popName, nodeCount);

            // Read SNNFrame-specific attributes if available
            auto attrNames = population->attributeNames();

            NeuronParamsIR params;
            params.name = popName + "_params";

            // Try to read SNNFrame neuron parameters from node attributes
            bool hasWindowSize = attrNames.count("window_size_ms") > 0;
            bool hasThreshold = attrNames.count("similarity_threshold") > 0;
            bool hasMaxPatterns = attrNames.count("max_patterns") > 0;

            if (hasWindowSize) {
                auto values = population->getAttribute<double>("window_size_ms", selection);
                if (!values.empty()) params.windowSizeMs = values[0];
            }
            if (hasThreshold) {
                auto values = population->getAttribute<double>("similarity_threshold", selection);
                if (!values.empty()) params.similarityThreshold = values[0];
            }
            if (hasMaxPatterns) {
                auto values = population->getAttribute<uint64_t>("max_patterns", selection);
                if (!values.empty()) params.maxReferencePatterns = static_cast<int>(values[0]);
            }

            ir.neuronParamSets[params.name] = params;

            // Create a population IR entry (will be placed in hierarchy by buildDefaultHierarchy)
            PopulationIR popIR;
            popIR.name = popName;
            popIR.count = static_cast<int>(nodeCount);
            popIR.neuronParams = params.name;

            // Store in a temporary layer (buildDefaultHierarchy will organize)
            LayerIR layer;
            layer.name = popName;
            layer.populations.push_back(popIR);

            // Add as a column in the default hierarchy
            ColumnIR col;
            col.name = popName + "_col";
            col.layers.push_back(layer);

            // Check for orientation/frequency attributes for column properties
            if (attrNames.count("orientation") > 0) {
                auto orientations = population->getAttribute<double>("orientation", selection);
                if (!orientations.empty()) {
                    col.properties["orientation"] = orientations[0];
                }
            }

            // Store columns temporarily - buildDefaultHierarchy will place them
            if (ir.brain.hemispheres.empty()) {
                buildDefaultHierarchy(ir, "SONATANetwork");
            }
            auto& nucleus = ir.brain.hemispheres[0].lobes[0].regions[0].nuclei[0];
            nucleus.columns.push_back(col);
        }
    } catch (const std::exception& e) {
        SNNFW_ERROR("Failed to load SONATA nodes from '{}': {}", nodesFile, e.what());
        throw ParseError("Failed to load SONATA nodes: " + std::string(e.what()));
    }
}

void SONATAParser::loadEdges(const std::string& edgesFile,
                              const std::string& /*edgeTypesFile*/,
                              NetworkIR& ir) const {
    if (!std::filesystem::exists(edgesFile)) {
        SNNFW_WARN("SONATA edges file not found: {}", edgesFile);
        return;
    }

    try {
        bbp::sonata::EdgeStorage storage(edgesFile);
        auto populationNames = storage.populationNames();

        for (const auto& popName : populationNames) {
            auto population = storage.openPopulation(popName);
            auto selection = population->selectAll();
            size_t edgeCount = selection.flatSize();

            SNNFW_INFO("SONATA: Loading edge population '{}' with {} edges",
                       popName, edgeCount);

            // Read edge attributes
            auto attrNames = population->attributeNames();

            ProjectionIR proj;
            proj.name = popName;
            proj.pattern = "explicit";  // SONATA has explicit connectivity
            proj.scope = "global";

            // Source and target come from the population metadata
            proj.source = population->source();
            proj.target = population->target();

            // Read weight and delay if available
            if (attrNames.count("weight") > 0) {
                auto weights = population->getAttribute<double>("weight", selection);
                if (!weights.empty()) {
                    // Use average weight as the projection weight
                    double sum = 0.0;
                    for (double w : weights) sum += w;
                    proj.weight = sum / weights.size();
                }
            }
            if (attrNames.count("delay") > 0) {
                auto delays = population->getAttribute<double>("delay", selection);
                if (!delays.empty()) {
                    double sum = 0.0;
                    for (double d : delays) sum += d;
                    proj.delay = sum / delays.size();
                }
            }

            // Estimate connectivity probability
            // (edges / (source_count * target_count))
            proj.probability = 1.0;  // Explicit connectivity

            proj.synapseGroup = popName;
            ir.projections.push_back(proj);
        }
    } catch (const std::exception& e) {
        SNNFW_ERROR("Failed to load SONATA edges from '{}': {}", edgesFile, e.what());
        throw ParseError("Failed to load SONATA edges: " + std::string(e.what()));
    }
}

void SONATAParser::buildDefaultHierarchy(NetworkIR& ir,
                                          const std::string& networkName) const {
    // Create a minimal hierarchy: Brain → Hemisphere → Lobe → Region → Nucleus
    BrainIR brain;
    brain.name = networkName;

    HemisphereIR hemi;
    hemi.name = "Default";

    LobeIR lobe;
    lobe.name = "Primary";

    RegionIR region;
    region.name = "Main";

    NucleusIR nucleus;
    nucleus.name = "Core";

    region.nuclei.push_back(nucleus);
    lobe.regions.push_back(region);
    hemi.lobes.push_back(lobe);
    brain.hemispheres.push_back(hemi);

    ir.brain = brain;
}

} // namespace declarative
} // namespace snnfw

