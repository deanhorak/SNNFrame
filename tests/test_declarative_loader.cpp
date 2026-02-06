#include <gtest/gtest.h>
#include "snnfw/declarative/DeclarativeLoader.h"
#include "snnfw/declarative/NativeJSONParser.h"
#include "snnfw/declarative/NetworkIR.h"
#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Datastore.h"
#include "snnfw/Logger.h"
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <spdlog/spdlog.h>

using namespace snnfw;
using namespace snnfw::declarative;

// ============================================================================
// Test fixture
// ============================================================================
class DeclarativeLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        Logger::getInstance().initialize("test_declarative.log", spdlog::level::warn);
        dbPath_ = "test_declarative_db_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed());
        factory_ = std::make_unique<NeuralObjectFactory>();
        datastore_ = std::make_unique<Datastore>(dbPath_);
    }
    void TearDown() override {
        datastore_.reset();
        factory_.reset();
        std::filesystem::remove_all(dbPath_);
    }
    std::string dbPath_;
    std::unique_ptr<NeuralObjectFactory> factory_;
    std::unique_ptr<Datastore> datastore_;
};

// ============================================================================
// NetworkIR validation tests
// ============================================================================
TEST_F(DeclarativeLoaderTest, MinimalValidIR_Validates) {
    // Build a minimal valid IR with complete hierarchy
    NetworkIR ir;
    ir.brain.name = "TestBrain";

    // Neuron params
    NeuronParamsIR params;
    params.name = "default";
    ir.neuronParamSets["default"] = params;

    // Input/output layers need valid sizes
    ir.inputLayer.rows = 2;
    ir.inputLayer.cols = 2;
    ir.outputLayer.numClasses = 2;
    ir.outputLayer.neuronsPerClass = 1;

    // Build minimal hierarchy: brain → hemisphere → lobe → region → nucleus → column → layer
    LayerIR layer;
    layer.name = "L4";
    PopulationIR pop;
    pop.name = "neurons";
    pop.count = 4;
    pop.neuronParams = "default";
    layer.populations.push_back(pop);

    ColumnIR col;
    col.name = "Col0";
    col.layers.push_back(layer);

    NucleusIR nucleus;
    nucleus.name = "Nuc0";
    nucleus.columns.push_back(col);

    RegionIR region;
    region.name = "Reg0";
    region.nuclei.push_back(nucleus);

    LobeIR lobe;
    lobe.name = "Lobe0";
    lobe.regions.push_back(region);

    HemisphereIR hemi;
    hemi.name = "Left";
    hemi.lobes.push_back(lobe);

    ir.brain.hemispheres.push_back(hemi);

    EXPECT_TRUE(ir.validate()) << "Minimal valid IR should pass validation";
}

TEST_F(DeclarativeLoaderTest, IR_MissingBrainName_Invalid) {
    NetworkIR ir;
    // brain name is empty by default
    auto errors = ir.getValidationErrors();
    EXPECT_FALSE(errors.empty());
}

// ============================================================================
// NativeJSONParser tests
// ============================================================================
TEST_F(DeclarativeLoaderTest, Parser_CanParse_SnnfJson) {
    NativeJSONParser parser;
    EXPECT_TRUE(parser.canParse("network.snnf.json"));
    EXPECT_TRUE(parser.canParse("/path/to/my_network.snnf.json"));
    EXPECT_FALSE(parser.canParse("network.json"));
    EXPECT_FALSE(parser.canParse("network.sonata.json"));
    EXPECT_FALSE(parser.canParse("network.hoc"));
}

TEST_F(DeclarativeLoaderTest, Parser_FormatName) {
    NativeJSONParser parser;
    EXPECT_EQ(parser.formatName(), "snnframe_json");
}

TEST_F(DeclarativeLoaderTest, Parser_ParseMinimalJson) {
    nlohmann::json root = {
        {"snnframe_version", "1.0"},
        {"neuron_params", {
            {"default", {
                {"window_size_ms", 200.0},
                {"similarity_threshold", 0.8},
                {"max_reference_patterns", 100}
            }}
        }},
        {"input_layer", {
            {"rows", 14},
            {"cols", 14}
        }},
        {"output_layer", {
            {"num_classes", 10},
            {"neurons_per_class", 2}
        }},
        {"brain", {
            {"name", "TestBrain"},
            {"hemispheres", nlohmann::json::array({
                {{"name", "Left"}, {"lobes", nlohmann::json::array()}}
            })}
        }}
    };

    NativeJSONParser parser;
    auto ir = parser.parseJson(root);

    EXPECT_EQ(ir.formatVersion, "1.0");
    EXPECT_EQ(ir.sourceFormat, "snnframe_json");
    EXPECT_EQ(ir.neuronParamSets.size(), 1);
    EXPECT_EQ(ir.neuronParamSets["default"].windowSizeMs, 200.0);
    EXPECT_EQ(ir.inputLayer.rows, 14);
    EXPECT_EQ(ir.inputLayer.cols, 14);
    EXPECT_EQ(ir.outputLayer.numClasses, 10);
    EXPECT_EQ(ir.outputLayer.neuronsPerClass, 2);
    EXPECT_EQ(ir.brain.name, "TestBrain");
    EXPECT_EQ(ir.brain.hemispheres.size(), 1);
}

TEST_F(DeclarativeLoaderTest, Parser_ParseProjections) {
    nlohmann::json root = {
        {"brain", {{"name", "B"}, {"hemispheres", nlohmann::json::array()}}},
        {"projections", nlohmann::json::array({
            {
                {"name", "L4toL5"},
                {"source", "V1/*/L4"},
                {"target", "V1/*/L5"},
                {"pattern", "random_sparse"},
                {"probability", 0.25},
                {"weight", 0.5},
                {"max_weight", 2.0},
                {"delay", 1.0},
                {"scope", "intra_column"},
                {"synapse_group", "L4ToL5"}
            }
        })}
    };

    NativeJSONParser parser;
    auto ir = parser.parseJson(root);

    ASSERT_EQ(ir.projections.size(), 1);
    EXPECT_EQ(ir.projections[0].name, "L4toL5");
    EXPECT_EQ(ir.projections[0].source, "V1/*/L4");
    EXPECT_EQ(ir.projections[0].pattern, "random_sparse");
    EXPECT_DOUBLE_EQ(ir.projections[0].probability, 0.25);
    EXPECT_EQ(ir.projections[0].scope, "intra_column");
    EXPECT_EQ(ir.projections[0].synapseGroup, "L4ToL5");
}

TEST_F(DeclarativeLoaderTest, Parser_ParseColumnTemplate) {
    nlohmann::json root = {
        {"brain", {
            {"name", "B"},
            {"hemispheres", nlohmann::json::array({
                {{"name", "Left"}, {"lobes", nlohmann::json::array({
                    {{"name", "Occipital"}, {"regions", nlohmann::json::array({
                        {{"name", "V1"}, {"nuclei", nlohmann::json::array({
                            {{"name", "Features"}, {"column_template", {
                                {"template_name", "CorticalColumn"},
                                {"orientations", {0.0, 45.0, 90.0, 135.0}},
                                {"frequencies", {3.0, 8.0}},
                                {"layers", nlohmann::json::array({
                                    {{"name", "L4"}, {"populations", nlohmann::json::array({
                                        {{"name", "L4_n"}, {"count", 49}, {"neuron_params", "default"}}
                                    })}}
                                })}
                            }}}
                        })}}
                    })}}
                })}}
            })}
        }}
    };

    NativeJSONParser parser;
    auto ir = parser.parseJson(root);

    auto& nuc = ir.brain.hemispheres[0].lobes[0].regions[0].nuclei[0];
    EXPECT_EQ(nuc.name, "Features");
    ASSERT_TRUE(nuc.columnTemplate.has_value());
    EXPECT_EQ(nuc.columnTemplate->orientations.size(), 4);
    EXPECT_EQ(nuc.columnTemplate->frequencies.size(), 2);
    EXPECT_EQ(nuc.columnTemplate->layers.size(), 1);
    EXPECT_EQ(nuc.columnTemplate->layers[0].populations[0].count, 49);
}

TEST_F(DeclarativeLoaderTest, Parser_ParseSaccades) {
    nlohmann::json root = {
        {"brain", {{"name", "B"}, {"hemispheres", nlohmann::json::array()}}},
        {"saccades", {
            {"enabled", true},
            {"num_fixations", 3},
            {"regions", nlohmann::json::array({
                {{"name", "top"}, {"row_start", 0}, {"row_end", 13}, {"col_start", 0}, {"col_end", 27}},
                {{"name", "center"}, {"row_start", 7}, {"row_end", 20}, {"col_start", 7}, {"col_end", 20}}
            })}
        }}
    };

    NativeJSONParser parser;
    auto ir = parser.parseJson(root);

    EXPECT_TRUE(ir.saccades.enabled);
    EXPECT_EQ(ir.saccades.numFixations, 3);
    ASSERT_EQ(ir.saccades.regions.size(), 2);
    EXPECT_EQ(ir.saccades.regions[0].name, "top");
    EXPECT_EQ(ir.saccades.regions[0].rowEnd, 13);
    EXPECT_EQ(ir.saccades.regions[1].name, "center");
}

TEST_F(DeclarativeLoaderTest, Parser_ParseGabor) {
    nlohmann::json root = {
        {"brain", {{"name", "B"}, {"hemispheres", nlohmann::json::array()}}},
        {"gabor", {
            {"freq_low", 6.0},
            {"freq_high", 2.5},
            {"sigma", 1.5},
            {"gamma", 0.4},
            {"threshold", 0.3},
            {"kernel_size", 5}
        }}
    };

    NativeJSONParser parser;
    auto ir = parser.parseJson(root);

    EXPECT_DOUBLE_EQ(ir.gabor.freqLow, 6.0);
    EXPECT_DOUBLE_EQ(ir.gabor.freqHigh, 2.5);
    EXPECT_DOUBLE_EQ(ir.gabor.sigma, 1.5);
    EXPECT_EQ(ir.gabor.kernelSize, 5);
}

// ============================================================================
// DeclarativeLoader tests
// ============================================================================
TEST_F(DeclarativeLoaderTest, Loader_RegisteredFormats) {
    DeclarativeLoader loader(*factory_, *datastore_);
    auto formats = loader.getRegisteredFormats();
    ASSERT_EQ(formats.size(), 1);
    EXPECT_EQ(formats[0], "snnframe_json");
}

TEST_F(DeclarativeLoaderTest, Loader_ParseOnly_InvalidFile) {
    DeclarativeLoader loader(*factory_, *datastore_);
    EXPECT_THROW(loader.parseOnly("nonexistent.snnf.json"), ParseError);
}

TEST_F(DeclarativeLoaderTest, Loader_ParseOnly_UnsupportedFormat) {
    DeclarativeLoader loader(*factory_, *datastore_);
    EXPECT_THROW(loader.parseOnly("network.hoc"), ParseError);
}

// ============================================================================
// NetworkConstructor tests (small network)
// ============================================================================
TEST_F(DeclarativeLoaderTest, Constructor_SmallNetwork) {
    // Build a minimal IR programmatically
    NetworkIR ir;
    ir.sourceFormat = "test";
    ir.brain.name = "SmallBrain";

    // Neuron params
    NeuronParamsIR params;
    params.name = "default";
    params.windowSizeMs = 100.0;
    params.similarityThreshold = 0.8;
    params.maxReferencePatterns = 50;
    ir.neuronParamSets["default"] = params;

    // Input layer
    ir.inputLayer.rows = 4;
    ir.inputLayer.cols = 4;
    ir.inputLayer.neuronParams = params;

    // Output layer
    ir.outputLayer.numClasses = 3;
    ir.outputLayer.neuronsPerClass = 2;
    ir.outputLayer.neuronParams = params;

    // Simple hierarchy: Brain -> Hemisphere -> Lobe -> Region -> Nucleus -> 2 columns
    HemisphereIR hemi;
    hemi.name = "Left";
    LobeIR lobe;
    lobe.name = "TestLobe";
    RegionIR region;
    region.name = "TestRegion";
    NucleusIR nucleus;
    nucleus.name = "TestNucleus";

    // Two explicit columns with L4 and L5
    for (int c = 0; c < 2; ++c) {
        ColumnIR col;
        col.name = "Col" + std::to_string(c);
        col.properties["orientation"] = c * 45.0;

        LayerIR l4;
        l4.name = "L4";
        PopulationIR l4Pop;
        l4Pop.name = "L4_neurons";
        l4Pop.count = 4;
        l4Pop.neuronParams = "default";
        l4.populations.push_back(l4Pop);
        col.layers.push_back(l4);

        LayerIR l5;
        l5.name = "L5";
        PopulationIR l5Pop;
        l5Pop.name = "L5_neurons";
        l5Pop.count = 3;
        l5Pop.neuronParams = "default";
        l5.populations.push_back(l5Pop);
        col.layers.push_back(l5);

        nucleus.columns.push_back(col);
    }

    region.nuclei.push_back(nucleus);
    lobe.regions.push_back(region);
    hemi.lobes.push_back(lobe);
    ir.brain.hemispheres.push_back(hemi);

    // Projections
    ProjectionIR inputToL4;
    inputToL4.name = "Input_to_L4";
    inputToL4.source = "InputGrid";
    inputToL4.target = "TestRegion/*/L4";
    inputToL4.pattern = "all_to_all";
    inputToL4.weight = 0.5;
    inputToL4.delay = 1.0;
    inputToL4.scope = "global";
    inputToL4.synapseGroup = "InputToL4";
    ir.projections.push_back(inputToL4);

    ProjectionIR l4ToL5;
    l4ToL5.name = "L4_to_L5";
    l4ToL5.source = "TestRegion/*/L4";
    l4ToL5.target = "TestRegion/*/L5";
    l4ToL5.pattern = "all_to_all";
    l4ToL5.weight = 0.3;
    l4ToL5.delay = 1.0;
    l4ToL5.scope = "intra_column";
    l4ToL5.synapseGroup = "L4ToL5";
    ir.projections.push_back(l4ToL5);

    // Validate
    ASSERT_TRUE(ir.validate()) << "IR should be valid";

    // Construct
    NetworkConstructor constructor(*factory_, *datastore_);
    auto network = constructor.construct(ir);

    // Verify input neurons
    EXPECT_EQ(network.inputNeurons.size(), 16);  // 4x4

    // Verify output neurons
    EXPECT_EQ(network.outputPopulations.size(), 3);
    for (const auto& pop : network.outputPopulations) {
        EXPECT_EQ(pop.size(), 2);
    }

    // Verify columns
    EXPECT_EQ(network.columns.size(), 2);
    for (const auto& col : network.columns) {
        EXPECT_TRUE(col.layerNeurons.count("L4") > 0);
        EXPECT_TRUE(col.layerNeurons.count("L5") > 0);
        EXPECT_EQ(col.layerNeurons.at("L4").size(), 4);
        EXPECT_EQ(col.layerNeurons.at("L5").size(), 3);
    }

    // Verify synapse creation
    EXPECT_GT(network.allSynapses.size(), 0);
    EXPECT_GT(network.allAxons.size(), 0);
    EXPECT_GT(network.allDendrites.size(), 0);

    // Verify synapse groups
    EXPECT_TRUE(network.synapseGroups.count("InputToL4") > 0);
    EXPECT_TRUE(network.synapseGroups.count("L4ToL5") > 0);

    // Verify total neuron count
    // Input: 16 + Output: 3*2=6 + Columns: 2*(4+3)=14 = 36
    EXPECT_EQ(network.allNeuronIds.size(), 36);

    // Verify runtime was initialized
    EXPECT_TRUE(network.spikeProcessor != nullptr);
    EXPECT_TRUE(network.propagator != nullptr);
    EXPECT_TRUE(network.brain != nullptr);
}

