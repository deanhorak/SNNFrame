#include "snnfw/declarative/NetworkConstructor.h"
#include "snnfw/NetworkBuilder.h"
#include "snnfw/ConnectivityBuilder.h"
#include "snnfw/ConnectivityPattern.h"
#include "snnfw/Logger.h"
#include <regex>
#include <sstream>
#include <algorithm>
#include <stdexcept>

namespace snnfw {
namespace declarative {

NetworkConstructor::NetworkConstructor(NeuralObjectFactory& factory, Datastore& datastore)
    : factory_(factory), datastore_(datastore) {}

ConstructedNetwork NetworkConstructor::construct(const NetworkIR& ir) {
    // Validate first
    auto errors = ir.getValidationErrors();
    if (!errors.empty()) {
        std::string msg = "NetworkIR validation failed:\n";
        for (const auto& e : errors) {
            msg += "  - " + e + "\n";
        }
        throw std::runtime_error(msg);
    }

    ConstructedNetwork result;
    result.sourceIR = ir;

    SNNFW_INFO("NetworkConstructor: Phase 1 - Building hierarchy...");
    buildHierarchy(ir, result);

    SNNFW_INFO("NetworkConstructor: Phase 2 - Creating neurons...");
    createNeurons(ir, result);

    SNNFW_INFO("NetworkConstructor: Phase 3 - Creating connectivity...");
    createConnectivity(ir, result);

    SNNFW_INFO("NetworkConstructor: Phase 4 - Initializing runtime...");
    initializeRuntime(ir, result);

    SNNFW_INFO("NetworkConstructor: Construction complete. {} neurons, {} synapses, {} columns",
               result.allNeuronIds.size(), result.allSynapses.size(), result.columns.size());

    return result;
}

// ============================================================================
// Phase 1: Build hierarchy
// ============================================================================
void NetworkConstructor::buildHierarchy(const NetworkIR& ir, ConstructedNetwork& result) {
    NetworkBuilder builder(factory_, datastore_, false);  // no auto-validate

    builder.createBrain(ir.brain.name);

    for (const auto& hemiIR : ir.brain.hemispheres) {
        builder.addHemisphere(hemiIR.name);
        for (const auto& lobeIR : hemiIR.lobes) {
            builder.addLobe(lobeIR.name);
            for (const auto& regionIR : lobeIR.regions) {
                builder.addRegion(regionIR.name);
                for (const auto& nucleusIR : regionIR.nuclei) {
                    builder.addNucleus(nucleusIR.name);

                    // Expand column template if present
                    std::vector<ColumnIR> allColumns = nucleusIR.columns;
                    if (nucleusIR.columnTemplate.has_value()) {
                        auto expanded = expandColumnTemplate(nucleusIR.columnTemplate.value());
                        allColumns.insert(allColumns.end(), expanded.begin(), expanded.end());
                    }

                    for (const auto& colIR : allColumns) {
                        builder.addColumn(colIR.name);
                        for (const auto& layerIR : colIR.layers) {
                            builder.addLayer(layerIR.name);
                            // Populations are created in Phase 2
                            builder.up();  // back to column
                        }
                        builder.up();  // back to nucleus
                    }
                    builder.up();  // back to region
                }
                builder.up();  // back to lobe
            }
            builder.up();  // back to hemisphere
        }
        builder.up();  // back to brain
    }

    result.brain = builder.build();
}

// ============================================================================
// Phase 2: Create neurons
// ============================================================================
void NetworkConstructor::createNeurons(const NetworkIR& ir, ConstructedNetwork& result) {
    // Create input layer neurons
    {
        const auto& il = ir.inputLayer;
        auto params = il.neuronParams;
        int totalInput = il.rows * il.cols;
        result.inputNeurons.reserve(totalInput);
        for (int i = 0; i < totalInput; ++i) {
            auto neuron = factory_.createNeuron(
                params.windowSizeMs, params.similarityThreshold, params.maxReferencePatterns);
            result.inputNeurons.push_back(neuron);
            result.allNeuronIds.push_back(neuron->getId());
        }
        SNNFW_INFO("  Created {} input neurons ({}x{})", totalInput, il.rows, il.cols);
    }

    // Create output layer neurons
    {
        const auto& ol = ir.outputLayer;
        auto params = ol.neuronParams;
        result.outputPopulations.resize(ol.numClasses);
        for (int c = 0; c < ol.numClasses; ++c) {
            result.outputPopulations[c].reserve(ol.neuronsPerClass);
            for (int n = 0; n < ol.neuronsPerClass; ++n) {
                auto neuron = factory_.createNeuron(
                    params.windowSizeMs, params.similarityThreshold, params.maxReferencePatterns);
                result.outputPopulations[c].push_back(neuron);
                result.allNeuronIds.push_back(neuron->getId());
            }
        }
        SNNFW_INFO("  Created {} output neurons ({} classes x {} per class)",
                   ol.numClasses * ol.neuronsPerClass, ol.numClasses, ol.neuronsPerClass);
    }

    // Create column neurons from hierarchy
    for (const auto& hemiIR : ir.brain.hemispheres) {
        for (const auto& lobeIR : hemiIR.lobes) {
            for (const auto& regionIR : lobeIR.regions) {
                for (const auto& nucleusIR : regionIR.nuclei) {
                    // Combine explicit columns + expanded template
                    std::vector<ColumnIR> allColumns = nucleusIR.columns;
                    if (nucleusIR.columnTemplate.has_value()) {
                        auto expanded = expandColumnTemplate(nucleusIR.columnTemplate.value());
                        allColumns.insert(allColumns.end(), expanded.begin(), expanded.end());
                    }

                    for (const auto& colIR : allColumns) {
                        ConstructedNetwork::ColumnGroup cg;
                        cg.name = colIR.name;

                        // Extract orientation/frequency from properties
                        if (colIR.properties.count("orientation")) {
                            cg.orientation = colIR.properties.at("orientation");
                        }
                        if (colIR.properties.count("spatial_frequency")) {
                            cg.spatialFrequency = colIR.properties.at("spatial_frequency");
                        }

                        // Create neurons for each layer
                        for (const auto& layerIR : colIR.layers) {
                            for (const auto& popIR : layerIR.populations) {
                                auto params = resolveNeuronParams(popIR.neuronParams, ir);
                                std::vector<std::shared_ptr<Neuron>> layerNeurons;
                                layerNeurons.reserve(popIR.count);
                                for (int i = 0; i < popIR.count; ++i) {
                                    auto neuron = factory_.createNeuron(
                                        params.windowSizeMs,
                                        params.similarityThreshold,
                                        params.maxReferencePatterns);
                                    layerNeurons.push_back(neuron);
                                    result.allNeuronIds.push_back(neuron->getId());
                                }
                                cg.layerNeurons[layerIR.name] = std::move(layerNeurons);
                            }
                        }

                        result.columns.push_back(std::move(cg));
                    }
                }
            }
        }
    }

    SNNFW_INFO("  Created {} cortical columns with {} total neuron IDs",
               result.columns.size(), result.allNeuronIds.size());
}

// ============================================================================
// Phase 3: Create connectivity
// ============================================================================
void NetworkConstructor::createConnectivity(const NetworkIR& ir, ConstructedNetwork& result) {
    ConnectivityBuilder connBuilder(factory_, datastore_, false);

    for (const auto& proj : ir.projections) {
        auto pattern = createPattern(proj);
        auto synapseGroup = mapSynapseGroup(proj.synapseGroup);

        // Try to resolve as special paths first (InputGrid, OutputLayer)
        auto srcSpecial = resolveSpecialPath(proj.source, result);
        auto tgtSpecial = resolveSpecialPath(proj.target, result);

        if (!srcSpecial.empty() && !tgtSpecial.empty()) {
            // Both are special (unlikely but supported)
            auto stats = connBuilder.connect(srcSpecial, tgtSpecial, *pattern);
            auto& created = connBuilder.getCreatedSynapses();
            for (auto& syn : created) {
                result.allSynapses.push_back(syn);
                if (!proj.synapseGroup.empty()) {
                    result.synapseGroups[proj.synapseGroup].push_back(syn);
                }
            }
            result.allAxons.insert(result.allAxons.end(),
                connBuilder.getCreatedAxons().begin(), connBuilder.getCreatedAxons().end());
            result.allDendrites.insert(result.allDendrites.end(),
                connBuilder.getCreatedDendrites().begin(), connBuilder.getCreatedDendrites().end());
            SNNFW_INFO("  Projection '{}': {} synapses", proj.name, stats.synapsesCreated);
        } else if (!srcSpecial.empty() || !tgtSpecial.empty()) {
            // One special, one glob - connect to/from each column
            auto columnSrc = srcSpecial.empty() ? resolveNeuronPath(proj.source, result)
                : std::vector<std::vector<std::shared_ptr<Neuron>>>{srcSpecial};
            auto columnTgt = tgtSpecial.empty() ? resolveNeuronPath(proj.target, result)
                : std::vector<std::vector<std::shared_ptr<Neuron>>>{tgtSpecial};

            size_t totalSynapses = 0;
            // For scope "intra_column": pair up src[i] with tgt[i]
            // For others: connect all to all
            if (proj.scope == "intra_column" && columnSrc.size() == columnTgt.size()) {
                for (size_t i = 0; i < columnSrc.size(); ++i) {
                    connBuilder.clearCreatedObjects();
                    auto stats = connBuilder.connect(columnSrc[i], columnTgt[i], *pattern);
                    for (auto& syn : connBuilder.getCreatedSynapses()) {
                        result.allSynapses.push_back(syn);
                        if (!proj.synapseGroup.empty()) {
                            result.synapseGroups[proj.synapseGroup].push_back(syn);
                        }
                    }
                    result.allAxons.insert(result.allAxons.end(),
                        connBuilder.getCreatedAxons().begin(), connBuilder.getCreatedAxons().end());
                    result.allDendrites.insert(result.allDendrites.end(),
                        connBuilder.getCreatedDendrites().begin(), connBuilder.getCreatedDendrites().end());
                    totalSynapses += stats.synapsesCreated;
                }
            } else {
                // Flatten all and connect
                std::vector<std::shared_ptr<Neuron>> flatSrc, flatTgt;
                for (auto& v : columnSrc) flatSrc.insert(flatSrc.end(), v.begin(), v.end());
                for (auto& v : columnTgt) flatTgt.insert(flatTgt.end(), v.begin(), v.end());
                connBuilder.clearCreatedObjects();
                auto stats = connBuilder.connect(flatSrc, flatTgt, *pattern);
                for (auto& syn : connBuilder.getCreatedSynapses()) {
                    result.allSynapses.push_back(syn);
                    if (!proj.synapseGroup.empty()) {
                        result.synapseGroups[proj.synapseGroup].push_back(syn);
                    }
                }
                result.allAxons.insert(result.allAxons.end(),
                    connBuilder.getCreatedAxons().begin(), connBuilder.getCreatedAxons().end());
                result.allDendrites.insert(result.allDendrites.end(),
                    connBuilder.getCreatedDendrites().begin(), connBuilder.getCreatedDendrites().end());
                totalSynapses = stats.synapsesCreated;
            }
            SNNFW_INFO("  Projection '{}': {} synapses ({})", proj.name, totalSynapses, proj.scope);
        } else {
            // Both are glob paths
            auto columnSrc = resolveNeuronPath(proj.source, result);
            auto columnTgt = resolveNeuronPath(proj.target, result);
            size_t totalSynapses = 0;

            if (proj.scope == "intra_column" && columnSrc.size() == columnTgt.size()) {
                for (size_t i = 0; i < columnSrc.size(); ++i) {
                    connBuilder.clearCreatedObjects();
                    auto stats = connBuilder.connect(columnSrc[i], columnTgt[i], *pattern);
                    for (auto& syn : connBuilder.getCreatedSynapses()) {
                        result.allSynapses.push_back(syn);
                        if (!proj.synapseGroup.empty()) {
                            result.synapseGroups[proj.synapseGroup].push_back(syn);
                        }
                    }
                    result.allAxons.insert(result.allAxons.end(),
                        connBuilder.getCreatedAxons().begin(), connBuilder.getCreatedAxons().end());
                    result.allDendrites.insert(result.allDendrites.end(),
                        connBuilder.getCreatedDendrites().begin(), connBuilder.getCreatedDendrites().end());
                    totalSynapses += stats.synapsesCreated;
                }
            } else {
                std::vector<std::shared_ptr<Neuron>> flatSrc, flatTgt;
                for (auto& v : columnSrc) flatSrc.insert(flatSrc.end(), v.begin(), v.end());
                for (auto& v : columnTgt) flatTgt.insert(flatTgt.end(), v.begin(), v.end());
                connBuilder.clearCreatedObjects();
                auto stats = connBuilder.connect(flatSrc, flatTgt, *pattern);
                for (auto& syn : connBuilder.getCreatedSynapses()) {
                    result.allSynapses.push_back(syn);
                    if (!proj.synapseGroup.empty()) {
                        result.synapseGroups[proj.synapseGroup].push_back(syn);
                    }
                }
                result.allAxons.insert(result.allAxons.end(),
                    connBuilder.getCreatedAxons().begin(), connBuilder.getCreatedAxons().end());
                result.allDendrites.insert(result.allDendrites.end(),
                    connBuilder.getCreatedDendrites().begin(), connBuilder.getCreatedDendrites().end());
                totalSynapses = stats.synapsesCreated;
            }
            SNNFW_INFO("  Projection '{}': {} synapses ({})", proj.name, totalSynapses, proj.scope);
        }
    }

    SNNFW_INFO("  Total: {} synapses, {} axons, {} dendrites",
               result.allSynapses.size(), result.allAxons.size(), result.allDendrites.size());
}

// ============================================================================
// Phase 4: Initialize runtime
// ============================================================================
void NetworkConstructor::initializeRuntime(const NetworkIR& ir, ConstructedNetwork& result) {
    const auto& sim = ir.simulation;

    // Create SpikeProcessor
    result.spikeProcessor = std::make_shared<SpikeProcessor>(
        sim.spikeProcessorTimeSlices, sim.spikeProcessorThreads);

    // Create NetworkPropagator
    result.propagator = std::make_shared<NetworkPropagator>(result.spikeProcessor);

    // Register all dendrites with SpikeProcessor
    for (const auto& dendrite : result.allDendrites) {
        result.spikeProcessor->registerDendrite(dendrite);
    }

    // Register all neurons with NetworkPropagator
    for (const auto& id : result.allNeuronIds) {
        // We need actual Neuron objects - collect from all groups
        // Input neurons
        for (const auto& n : result.inputNeurons) {
            if (n->getId() == id) {
                result.propagator->registerNeuron(n);
                goto next_id;
            }
        }
        // Output neurons
        for (const auto& pop : result.outputPopulations) {
            for (const auto& n : pop) {
                if (n->getId() == id) {
                    result.propagator->registerNeuron(n);
                    goto next_id;
                }
            }
        }
        // Column neurons
        for (const auto& col : result.columns) {
            for (const auto& [layerName, neurons] : col.layerNeurons) {
                for (const auto& n : neurons) {
                    if (n->getId() == id) {
                        result.propagator->registerNeuron(n);
                        goto next_id;
                    }
                }
            }
        }
        next_id:;
    }

    // Register axons
    for (const auto& axon : result.allAxons) {
        result.propagator->registerAxon(axon);
    }

    // Register dendrites
    for (const auto& dendrite : result.allDendrites) {
        result.propagator->registerDendrite(dendrite);
    }

    // Register synapses and assign synapse groups
    for (const auto& synapse : result.allSynapses) {
        result.propagator->registerSynapse(synapse);
    }

    // Register synapse groups
    for (const auto& [groupName, synapses] : result.synapseGroups) {
        auto group = mapSynapseGroup(groupName);
        for (const auto& syn : synapses) {
            result.propagator->registerSynapseGroup(syn->getId(), group);
        }
    }

    SNNFW_INFO("  Runtime initialized: SpikeProcessor({} slices, {} threads), "
               "NetworkPropagator registered {} neurons",
               sim.spikeProcessorTimeSlices, sim.spikeProcessorThreads,
               result.allNeuronIds.size());
}

// ============================================================================
// Helper: Expand column template
// ============================================================================
std::vector<ColumnIR> NetworkConstructor::expandColumnTemplate(const ColumnTemplateIR& tmpl) {
    std::vector<ColumnIR> result;

    for (double orient : tmpl.orientations) {
        for (double freq : tmpl.frequencies) {
            ColumnIR col;

            // Generate name from pattern
            std::string name = tmpl.namingPattern;
            // Replace {orientation} token
            auto pos = name.find("{orientation}");
            if (pos != std::string::npos) {
                // Format as integer if whole number, else decimal
                std::ostringstream oss;
                if (orient == static_cast<int>(orient)) {
                    oss << static_cast<int>(orient);
                } else {
                    oss << orient;
                }
                name.replace(pos, 13, oss.str());
            }
            // Replace {frequency} token
            pos = name.find("{frequency}");
            if (pos != std::string::npos) {
                std::string freqStr = (freq >= 5.0) ? "High" : "Low";
                name.replace(pos, 11, freqStr);
            }

            col.name = name;
            col.properties["orientation"] = orient;
            col.properties["spatial_frequency"] = freq;
            col.layers = tmpl.layers;  // Copy the layer blueprint

            result.push_back(std::move(col));
        }
    }

    return result;
}

// ============================================================================
// Helper: Resolve neuron path
// ============================================================================
std::vector<std::vector<std::shared_ptr<Neuron>>>
NetworkConstructor::resolveNeuronPath(const std::string& path, const ConstructedNetwork& result) {
    // Path format: "RegionName/*/LayerName" or "RegionName/*/LayerName/PopName"
    // The * means "for each column"
    std::vector<std::vector<std::shared_ptr<Neuron>>> resolved;

    // Split path by /
    std::vector<std::string> parts;
    std::istringstream iss(path);
    std::string part;
    while (std::getline(iss, part, '/')) {
        if (!part.empty()) parts.push_back(part);
    }

    if (parts.size() < 3) {
        SNNFW_WARN("Cannot resolve path '{}': need at least Region/*/Layer", path);
        return resolved;
    }

    // Find the layer name (last non-wildcard component, or after the *)
    std::string layerName;
    for (size_t i = 0; i < parts.size(); ++i) {
        if (parts[i] == "*" && i + 1 < parts.size()) {
            layerName = parts[i + 1];
            break;
        }
    }

    if (layerName.empty()) {
        SNNFW_WARN("Cannot resolve path '{}': no layer name found after '*'", path);
        return resolved;
    }

    // Collect neurons from each column's matching layer
    for (const auto& col : result.columns) {
        auto it = col.layerNeurons.find(layerName);
        if (it != col.layerNeurons.end()) {
            resolved.push_back(it->second);
        }
    }

    return resolved;
}

// ============================================================================
// Helper: Resolve special paths (InputGrid, OutputLayer)
// ============================================================================
std::vector<std::shared_ptr<Neuron>>
NetworkConstructor::resolveSpecialPath(const std::string& path, const ConstructedNetwork& result) {
    if (path == result.sourceIR.inputLayer.name || path == "InputGrid") {
        return result.inputNeurons;
    }
    if (path == result.sourceIR.outputLayer.name || path == "OutputLayer") {
        std::vector<std::shared_ptr<Neuron>> flat;
        for (const auto& pop : result.outputPopulations) {
            flat.insert(flat.end(), pop.begin(), pop.end());
        }
        return flat;
    }
    return {};  // Not a special path
}

// ============================================================================
// Helper: Create connectivity pattern
// ============================================================================
std::unique_ptr<ConnectivityPattern>
NetworkConstructor::createPattern(const ProjectionIR& proj) {
    if (proj.pattern == "random_sparse") {
        return std::make_unique<RandomSparsePattern>(proj.probability, proj.weight, proj.delay);
    } else if (proj.pattern == "all_to_all") {
        return std::make_unique<AllToAllPattern>(proj.weight, proj.delay);
    } else if (proj.pattern == "one_to_one") {
        return std::make_unique<OneToOnePattern>(proj.weight, proj.delay);
    } else if (proj.pattern == "many_to_one") {
        return std::make_unique<ManyToOnePattern>(proj.weight, proj.delay);
    } else if (proj.pattern == "topographic") {
        return std::make_unique<TopographicPattern>(1.0, proj.weight, proj.delay);
    } else {
        // Default to random sparse
        SNNFW_WARN("Unknown pattern type '{}', defaulting to random_sparse", proj.pattern);
        return std::make_unique<RandomSparsePattern>(proj.probability, proj.weight, proj.delay);
    }
}

// ============================================================================
// Helper: Map synapse group string to enum
// ============================================================================
NetworkPropagator::SynapseGroup
NetworkConstructor::mapSynapseGroup(const std::string& groupName) {
    if (groupName == "InputToL4") return NetworkPropagator::SynapseGroup::InputToL4;
    if (groupName == "L4ToL5") return NetworkPropagator::SynapseGroup::L4ToL5;
    if (groupName == "L5ToOutput") return NetworkPropagator::SynapseGroup::L5ToOutput;
    return NetworkPropagator::SynapseGroup::Unknown;
}

// ============================================================================
// Helper: Resolve neuron params by name
// ============================================================================
NeuronParamsIR NetworkConstructor::resolveNeuronParams(const std::string& name, const NetworkIR& ir) {
    if (!name.empty()) {
        auto it = ir.neuronParamSets.find(name);
        if (it != ir.neuronParamSets.end()) {
            return it->second;
        }
        SNNFW_WARN("Neuron params '{}' not found, using defaults", name);
    }
    return NeuronParamsIR{};  // defaults
}

} // namespace declarative
} // namespace snnfw

