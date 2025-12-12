#include <gtest/gtest.h>
#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Brain.h"
#include "snnfw/Hemisphere.h"
#include "snnfw/Lobe.h"
#include "snnfw/Region.h"
#include "snnfw/Nucleus.h"
#include "snnfw/Column.h"
#include "snnfw/Layer.h"

using namespace snnfw;

// ============================================================================
// Layer Tests
// ============================================================================

TEST(LayerTests, ConstructorAssignsId) {
    Layer layer(600000000000000ULL);
    EXPECT_EQ(layer.getId(), 600000000000000ULL);
}

TEST(LayerTests, AddCluster) {
    Layer layer(600000000000000ULL);
    layer.addCluster(500000000000000ULL);
    layer.addCluster(500000000000001ULL);
    
    EXPECT_EQ(layer.size(), 2);
}

TEST(LayerTests, GetClusterId) {
    Layer layer(600000000000000ULL);
    layer.addCluster(500000000000000ULL);
    layer.addCluster(500000000000001ULL);
    
    EXPECT_EQ(layer.getClusterId(0), 500000000000000ULL);
    EXPECT_EQ(layer.getClusterId(1), 500000000000001ULL);
    EXPECT_EQ(layer.getClusterId(2), 0ULL);  // Out of range
}

TEST(LayerTests, RemoveCluster) {
    Layer layer(600000000000000ULL);
    layer.addCluster(500000000000000ULL);
    layer.addCluster(500000000000001ULL);
    layer.addCluster(500000000000002ULL);
    
    EXPECT_TRUE(layer.removeCluster(500000000000001ULL));
    EXPECT_EQ(layer.size(), 2);
    EXPECT_EQ(layer.getClusterId(0), 500000000000000ULL);
    EXPECT_EQ(layer.getClusterId(1), 500000000000002ULL);
    
    EXPECT_FALSE(layer.removeCluster(999999999999999ULL));  // Not found
}

TEST(LayerTests, Clear) {
    Layer layer(600000000000000ULL);
    layer.addCluster(500000000000000ULL);
    layer.addCluster(500000000000001ULL);
    
    layer.clear();
    EXPECT_EQ(layer.size(), 0);
}

TEST(LayerTests, GetClusterIds) {
    Layer layer(600000000000000ULL);
    layer.addCluster(500000000000000ULL);
    layer.addCluster(500000000000001ULL);
    
    const auto& ids = layer.getClusterIds();
    EXPECT_EQ(ids.size(), 2);
    EXPECT_EQ(ids[0], 500000000000000ULL);
    EXPECT_EQ(ids[1], 500000000000001ULL);
}

// ============================================================================
// Column Tests
// ============================================================================

TEST(ColumnTests, ConstructorAssignsId) {
    Column column(700000000000000ULL);
    EXPECT_EQ(column.getId(), 700000000000000ULL);
}

TEST(ColumnTests, AddLayer) {
    Column column(700000000000000ULL);
    column.addLayer(600000000000000ULL);
    column.addLayer(600000000000001ULL);
    
    EXPECT_EQ(column.size(), 2);
}

TEST(ColumnTests, GetLayerId) {
    Column column(700000000000000ULL);
    column.addLayer(600000000000000ULL);
    column.addLayer(600000000000001ULL);
    
    EXPECT_EQ(column.getLayerId(0), 600000000000000ULL);
    EXPECT_EQ(column.getLayerId(1), 600000000000001ULL);
    EXPECT_EQ(column.getLayerId(2), 0ULL);  // Out of range
}

TEST(ColumnTests, RemoveLayer) {
    Column column(700000000000000ULL);
    column.addLayer(600000000000000ULL);
    column.addLayer(600000000000001ULL);
    column.addLayer(600000000000002ULL);
    
    EXPECT_TRUE(column.removeLayer(600000000000001ULL));
    EXPECT_EQ(column.size(), 2);
    EXPECT_EQ(column.getLayerId(0), 600000000000000ULL);
    EXPECT_EQ(column.getLayerId(1), 600000000000002ULL);
    
    EXPECT_FALSE(column.removeLayer(999999999999999ULL));  // Not found
}

TEST(ColumnTests, Clear) {
    Column column(700000000000000ULL);
    column.addLayer(600000000000000ULL);
    column.addLayer(600000000000001ULL);
    
    column.clear();
    EXPECT_EQ(column.size(), 0);
}

TEST(ColumnTests, GetLayerIds) {
    Column column(700000000000000ULL);
    column.addLayer(600000000000000ULL);
    column.addLayer(600000000000001ULL);
    
    const auto& ids = column.getLayerIds();
    EXPECT_EQ(ids.size(), 2);
    EXPECT_EQ(ids[0], 600000000000000ULL);
    EXPECT_EQ(ids[1], 600000000000001ULL);
}

// ============================================================================
// Nucleus Tests
// ============================================================================

TEST(NucleusTests, ConstructorAssignsId) {
    Nucleus nucleus(800000000000000ULL);
    EXPECT_EQ(nucleus.getId(), 800000000000000ULL);
}

TEST(NucleusTests, AddColumn) {
    Nucleus nucleus(800000000000000ULL);
    nucleus.addColumn(700000000000000ULL);
    nucleus.addColumn(700000000000001ULL);
    
    EXPECT_EQ(nucleus.size(), 2);
}

TEST(NucleusTests, GetColumnId) {
    Nucleus nucleus(800000000000000ULL);
    nucleus.addColumn(700000000000000ULL);
    nucleus.addColumn(700000000000001ULL);
    
    EXPECT_EQ(nucleus.getColumnId(0), 700000000000000ULL);
    EXPECT_EQ(nucleus.getColumnId(1), 700000000000001ULL);
    EXPECT_EQ(nucleus.getColumnId(2), 0ULL);  // Out of range
}

TEST(NucleusTests, RemoveColumn) {
    Nucleus nucleus(800000000000000ULL);
    nucleus.addColumn(700000000000000ULL);
    nucleus.addColumn(700000000000001ULL);
    nucleus.addColumn(700000000000002ULL);
    
    EXPECT_TRUE(nucleus.removeColumn(700000000000001ULL));
    EXPECT_EQ(nucleus.size(), 2);
    EXPECT_EQ(nucleus.getColumnId(0), 700000000000000ULL);
    EXPECT_EQ(nucleus.getColumnId(1), 700000000000002ULL);
    
    EXPECT_FALSE(nucleus.removeColumn(999999999999999ULL));  // Not found
}

TEST(NucleusTests, Clear) {
    Nucleus nucleus(800000000000000ULL);
    nucleus.addColumn(700000000000000ULL);
    nucleus.addColumn(700000000000001ULL);
    
    nucleus.clear();
    EXPECT_EQ(nucleus.size(), 0);
}

TEST(NucleusTests, GetColumnIds) {
    Nucleus nucleus(800000000000000ULL);
    nucleus.addColumn(700000000000000ULL);
    nucleus.addColumn(700000000000001ULL);
    
    const auto& ids = nucleus.getColumnIds();
    EXPECT_EQ(ids.size(), 2);
    EXPECT_EQ(ids[0], 700000000000000ULL);
    EXPECT_EQ(ids[1], 700000000000001ULL);
}

// ============================================================================
// Region Tests
// ============================================================================

TEST(RegionTests, ConstructorAssignsId) {
    Region region(900000000000000ULL);
    EXPECT_EQ(region.getId(), 900000000000000ULL);
}

TEST(RegionTests, AddNucleus) {
    Region region(900000000000000ULL);
    region.addNucleus(800000000000000ULL);
    region.addNucleus(800000000000001ULL);
    
    EXPECT_EQ(region.size(), 2);
}

TEST(RegionTests, GetNucleusId) {
    Region region(900000000000000ULL);
    region.addNucleus(800000000000000ULL);
    region.addNucleus(800000000000001ULL);
    
    EXPECT_EQ(region.getNucleusId(0), 800000000000000ULL);
    EXPECT_EQ(region.getNucleusId(1), 800000000000001ULL);
    EXPECT_EQ(region.getNucleusId(2), 0ULL);  // Out of range
}

TEST(RegionTests, RemoveNucleus) {
    Region region(900000000000000ULL);
    region.addNucleus(800000000000000ULL);
    region.addNucleus(800000000000001ULL);
    region.addNucleus(800000000000002ULL);
    
    EXPECT_TRUE(region.removeNucleus(800000000000001ULL));
    EXPECT_EQ(region.size(), 2);
    EXPECT_EQ(region.getNucleusId(0), 800000000000000ULL);
    EXPECT_EQ(region.getNucleusId(1), 800000000000002ULL);
    
    EXPECT_FALSE(region.removeNucleus(999999999999999ULL));  // Not found
}

TEST(RegionTests, Clear) {
    Region region(900000000000000ULL);
    region.addNucleus(800000000000000ULL);
    region.addNucleus(800000000000001ULL);
    
    region.clear();
    EXPECT_EQ(region.size(), 0);
}

TEST(RegionTests, GetNucleusIds) {
    Region region(900000000000000ULL);
    region.addNucleus(800000000000000ULL);
    region.addNucleus(800000000000001ULL);
    
    const auto& ids = region.getNucleusIds();
    EXPECT_EQ(ids.size(), 2);
    EXPECT_EQ(ids[0], 800000000000000ULL);
    EXPECT_EQ(ids[1], 800000000000001ULL);
}

// ============================================================================
// Lobe Tests
// ============================================================================

TEST(LobeTests, ConstructorAssignsId) {
    Lobe lobe(1000000000000000ULL);
    EXPECT_EQ(lobe.getId(), 1000000000000000ULL);
}

TEST(LobeTests, AddRegion) {
    Lobe lobe(1000000000000000ULL);
    lobe.addRegion(900000000000000ULL);
    lobe.addRegion(900000000000001ULL);
    
    EXPECT_EQ(lobe.size(), 2);
}

TEST(LobeTests, GetRegionId) {
    Lobe lobe(1000000000000000ULL);
    lobe.addRegion(900000000000000ULL);
    lobe.addRegion(900000000000001ULL);

    EXPECT_EQ(lobe.getRegionId(0), 900000000000000ULL);
    EXPECT_EQ(lobe.getRegionId(1), 900000000000001ULL);
    EXPECT_EQ(lobe.getRegionId(2), 0ULL);  // Out of range
}

TEST(LobeTests, RemoveRegion) {
    Lobe lobe(1000000000000000ULL);
    lobe.addRegion(900000000000000ULL);
    lobe.addRegion(900000000000001ULL);
    lobe.addRegion(900000000000002ULL);

    EXPECT_TRUE(lobe.removeRegion(900000000000001ULL));
    EXPECT_EQ(lobe.size(), 2);
    EXPECT_EQ(lobe.getRegionId(0), 900000000000000ULL);
    EXPECT_EQ(lobe.getRegionId(1), 900000000000002ULL);

    EXPECT_FALSE(lobe.removeRegion(999999999999999ULL));  // Not found
}

TEST(LobeTests, Clear) {
    Lobe lobe(1000000000000000ULL);
    lobe.addRegion(900000000000000ULL);
    lobe.addRegion(900000000000001ULL);

    lobe.clear();
    EXPECT_EQ(lobe.size(), 0);
}

TEST(LobeTests, GetRegionIds) {
    Lobe lobe(1000000000000000ULL);
    lobe.addRegion(900000000000000ULL);
    lobe.addRegion(900000000000001ULL);

    const auto& ids = lobe.getRegionIds();
    EXPECT_EQ(ids.size(), 2);
    EXPECT_EQ(ids[0], 900000000000000ULL);
    EXPECT_EQ(ids[1], 900000000000001ULL);
}

// ============================================================================
// Hemisphere Tests
// ============================================================================

TEST(HemisphereTests, ConstructorAssignsId) {
    Hemisphere hemisphere(1100000000000000ULL);
    EXPECT_EQ(hemisphere.getId(), 1100000000000000ULL);
}

TEST(HemisphereTests, AddLobe) {
    Hemisphere hemisphere(1100000000000000ULL);
    hemisphere.addLobe(1000000000000000ULL);
    hemisphere.addLobe(1000000000000001ULL);

    EXPECT_EQ(hemisphere.size(), 2);
}

TEST(HemisphereTests, GetLobeId) {
    Hemisphere hemisphere(1100000000000000ULL);
    hemisphere.addLobe(1000000000000000ULL);
    hemisphere.addLobe(1000000000000001ULL);

    EXPECT_EQ(hemisphere.getLobeId(0), 1000000000000000ULL);
    EXPECT_EQ(hemisphere.getLobeId(1), 1000000000000001ULL);
    EXPECT_EQ(hemisphere.getLobeId(2), 0ULL);  // Out of range
}

TEST(HemisphereTests, RemoveLobe) {
    Hemisphere hemisphere(1100000000000000ULL);
    hemisphere.addLobe(1000000000000000ULL);
    hemisphere.addLobe(1000000000000001ULL);
    hemisphere.addLobe(1000000000000002ULL);

    EXPECT_TRUE(hemisphere.removeLobe(1000000000000001ULL));
    EXPECT_EQ(hemisphere.size(), 2);
    EXPECT_EQ(hemisphere.getLobeId(0), 1000000000000000ULL);
    EXPECT_EQ(hemisphere.getLobeId(1), 1000000000000002ULL);

    EXPECT_FALSE(hemisphere.removeLobe(999999999999999ULL));  // Not found
}

TEST(HemisphereTests, Clear) {
    Hemisphere hemisphere(1100000000000000ULL);
    hemisphere.addLobe(1000000000000000ULL);
    hemisphere.addLobe(1000000000000001ULL);

    hemisphere.clear();
    EXPECT_EQ(hemisphere.size(), 0);
}

TEST(HemisphereTests, GetLobeIds) {
    Hemisphere hemisphere(1100000000000000ULL);
    hemisphere.addLobe(1000000000000000ULL);
    hemisphere.addLobe(1000000000000001ULL);

    const auto& ids = hemisphere.getLobeIds();
    EXPECT_EQ(ids.size(), 2);
    EXPECT_EQ(ids[0], 1000000000000000ULL);
    EXPECT_EQ(ids[1], 1000000000000001ULL);
}

// ============================================================================
// Brain Tests
// ============================================================================

TEST(BrainTests, ConstructorAssignsId) {
    Brain brain(1200000000000000ULL);
    EXPECT_EQ(brain.getId(), 1200000000000000ULL);
}

TEST(BrainTests, AddHemisphere) {
    Brain brain(1200000000000000ULL);
    brain.addHemisphere(1100000000000000ULL);
    brain.addHemisphere(1100000000000001ULL);

    EXPECT_EQ(brain.size(), 2);
}

TEST(BrainTests, GetHemisphereId) {
    Brain brain(1200000000000000ULL);
    brain.addHemisphere(1100000000000000ULL);
    brain.addHemisphere(1100000000000001ULL);

    EXPECT_EQ(brain.getHemisphereId(0), 1100000000000000ULL);
    EXPECT_EQ(brain.getHemisphereId(1), 1100000000000001ULL);
    EXPECT_EQ(brain.getHemisphereId(2), 0ULL);  // Out of range
}

TEST(BrainTests, RemoveHemisphere) {
    Brain brain(1200000000000000ULL);
    brain.addHemisphere(1100000000000000ULL);
    brain.addHemisphere(1100000000000001ULL);
    brain.addHemisphere(1100000000000002ULL);

    EXPECT_TRUE(brain.removeHemisphere(1100000000000001ULL));
    EXPECT_EQ(brain.size(), 2);
    EXPECT_EQ(brain.getHemisphereId(0), 1100000000000000ULL);
    EXPECT_EQ(brain.getHemisphereId(1), 1100000000000002ULL);

    EXPECT_FALSE(brain.removeHemisphere(999999999999999ULL));  // Not found
}

TEST(BrainTests, Clear) {
    Brain brain(1200000000000000ULL);
    brain.addHemisphere(1100000000000000ULL);
    brain.addHemisphere(1100000000000001ULL);

    brain.clear();
    EXPECT_EQ(brain.size(), 0);
}

TEST(BrainTests, GetHemisphereIds) {
    Brain brain(1200000000000000ULL);
    brain.addHemisphere(1100000000000000ULL);
    brain.addHemisphere(1100000000000001ULL);

    const auto& ids = brain.getHemisphereIds();
    EXPECT_EQ(ids.size(), 2);
    EXPECT_EQ(ids[0], 1100000000000000ULL);
    EXPECT_EQ(ids[1], 1100000000000001ULL);
}

// ============================================================================
// Factory Integration Tests
// ============================================================================

TEST(FactoryIntegrationTests, CreateLayerWithCorrectIdRange) {
    NeuralObjectFactory factory;
    auto layer = factory.createLayer();

    EXPECT_GE(layer->getId(), 600000000000000ULL);
    EXPECT_LE(layer->getId(), 699999999999999ULL);
}

TEST(FactoryIntegrationTests, CreateColumnWithCorrectIdRange) {
    NeuralObjectFactory factory;
    auto column = factory.createColumn();

    EXPECT_GE(column->getId(), 700000000000000ULL);
    EXPECT_LE(column->getId(), 799999999999999ULL);
}

TEST(FactoryIntegrationTests, CreateNucleusWithCorrectIdRange) {
    NeuralObjectFactory factory;
    auto nucleus = factory.createNucleus();

    EXPECT_GE(nucleus->getId(), 800000000000000ULL);
    EXPECT_LE(nucleus->getId(), 899999999999999ULL);
}

TEST(FactoryIntegrationTests, CreateRegionWithCorrectIdRange) {
    NeuralObjectFactory factory;
    auto region = factory.createRegion();

    EXPECT_GE(region->getId(), 900000000000000ULL);
    EXPECT_LE(region->getId(), 999999999999999ULL);
}

TEST(FactoryIntegrationTests, CreateLobeWithCorrectIdRange) {
    NeuralObjectFactory factory;
    auto lobe = factory.createLobe();

    EXPECT_GE(lobe->getId(), 1000000000000000ULL);
    EXPECT_LE(lobe->getId(), 1099999999999999ULL);
}

TEST(FactoryIntegrationTests, CreateHemisphereWithCorrectIdRange) {
    NeuralObjectFactory factory;
    auto hemisphere = factory.createHemisphere();

    EXPECT_GE(hemisphere->getId(), 1100000000000000ULL);
    EXPECT_LE(hemisphere->getId(), 1199999999999999ULL);
}

TEST(FactoryIntegrationTests, CreateBrainWithCorrectIdRange) {
    NeuralObjectFactory factory;
    auto brain = factory.createBrain();

    EXPECT_GE(brain->getId(), 1200000000000000ULL);
    EXPECT_LE(brain->getId(), 1299999999999999ULL);
}

TEST(FactoryIntegrationTests, TypeIdentificationForLayer) {
    NeuralObjectFactory factory;
    auto layer = factory.createLayer();

    EXPECT_EQ(NeuralObjectFactory::getObjectType(layer->getId()),
              NeuralObjectFactory::ObjectType::LAYER);
}

TEST(FactoryIntegrationTests, TypeIdentificationForColumn) {
    NeuralObjectFactory factory;
    auto column = factory.createColumn();

    EXPECT_EQ(NeuralObjectFactory::getObjectType(column->getId()),
              NeuralObjectFactory::ObjectType::COLUMN);
}

TEST(FactoryIntegrationTests, TypeIdentificationForNucleus) {
    NeuralObjectFactory factory;
    auto nucleus = factory.createNucleus();

    EXPECT_EQ(NeuralObjectFactory::getObjectType(nucleus->getId()),
              NeuralObjectFactory::ObjectType::NUCLEUS);
}

TEST(FactoryIntegrationTests, TypeIdentificationForRegion) {
    NeuralObjectFactory factory;
    auto region = factory.createRegion();

    EXPECT_EQ(NeuralObjectFactory::getObjectType(region->getId()),
              NeuralObjectFactory::ObjectType::REGION);
}

TEST(FactoryIntegrationTests, TypeIdentificationForLobe) {
    NeuralObjectFactory factory;
    auto lobe = factory.createLobe();

    EXPECT_EQ(NeuralObjectFactory::getObjectType(lobe->getId()),
              NeuralObjectFactory::ObjectType::LOBE);
}

TEST(FactoryIntegrationTests, TypeIdentificationForHemisphere) {
    NeuralObjectFactory factory;
    auto hemisphere = factory.createHemisphere();

    EXPECT_EQ(NeuralObjectFactory::getObjectType(hemisphere->getId()),
              NeuralObjectFactory::ObjectType::HEMISPHERE);
}

TEST(FactoryIntegrationTests, TypeIdentificationForBrain) {
    NeuralObjectFactory factory;
    auto brain = factory.createBrain();

    EXPECT_EQ(NeuralObjectFactory::getObjectType(brain->getId()),
              NeuralObjectFactory::ObjectType::BRAIN);
}

TEST(FactoryIntegrationTests, BuildCompleteHierarchy) {
    NeuralObjectFactory factory;

    // Create brain
    auto brain = factory.createBrain();

    // Create hemispheres
    auto leftHemisphere = factory.createHemisphere();
    auto rightHemisphere = factory.createHemisphere();
    brain->addHemisphere(leftHemisphere->getId());
    brain->addHemisphere(rightHemisphere->getId());

    // Create lobes in left hemisphere
    auto frontalLobe = factory.createLobe();
    leftHemisphere->addLobe(frontalLobe->getId());

    // Create region in frontal lobe
    auto motorCortex = factory.createRegion();
    frontalLobe->addRegion(motorCortex->getId());

    // Create nucleus in motor cortex
    auto nucleus = factory.createNucleus();
    motorCortex->addNucleus(nucleus->getId());

    // Create column in nucleus
    auto column = factory.createColumn();
    nucleus->addColumn(column->getId());

    // Create layer in column
    auto layer = factory.createLayer();
    column->addLayer(layer->getId());

    // Create cluster in layer
    auto cluster = factory.createCluster();
    layer->addCluster(cluster->getId());

    // Verify hierarchy
    EXPECT_EQ(brain->size(), 2);
    EXPECT_EQ(leftHemisphere->size(), 1);
    EXPECT_EQ(frontalLobe->size(), 1);
    EXPECT_EQ(motorCortex->size(), 1);
    EXPECT_EQ(nucleus->size(), 1);
    EXPECT_EQ(column->size(), 1);
    EXPECT_EQ(layer->size(), 1);

    // Verify IDs are correct
    EXPECT_EQ(brain->getHemisphereId(0), leftHemisphere->getId());
    EXPECT_EQ(leftHemisphere->getLobeId(0), frontalLobe->getId());
    EXPECT_EQ(frontalLobe->getRegionId(0), motorCortex->getId());
    EXPECT_EQ(motorCortex->getNucleusId(0), nucleus->getId());
    EXPECT_EQ(nucleus->getColumnId(0), column->getId());
    EXPECT_EQ(column->getLayerId(0), layer->getId());
    EXPECT_EQ(layer->getClusterId(0), cluster->getId());
}

TEST(FactoryIntegrationTests, MultipleObjectsIncrementIds) {
    NeuralObjectFactory factory;

    auto layer1 = factory.createLayer();
    auto layer2 = factory.createLayer();
    auto layer3 = factory.createLayer();

    EXPECT_EQ(layer2->getId(), layer1->getId() + 1);
    EXPECT_EQ(layer3->getId(), layer2->getId() + 1);
}

TEST(FactoryIntegrationTests, ObjectCountTracking) {
    NeuralObjectFactory factory;

    factory.createLayer();
    factory.createLayer();
    factory.createColumn();
    factory.createNucleus();
    factory.createNucleus();
    factory.createNucleus();

    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::LAYER), 2);
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::COLUMN), 1);
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::NUCLEUS), 3);
}

TEST(FactoryIntegrationTests, ResetClearsHierarchicalObjects) {
    NeuralObjectFactory factory;

    factory.createLayer();
    factory.createColumn();
    factory.createNucleus();
    factory.createRegion();
    factory.createLobe();
    factory.createHemisphere();
    factory.createBrain();

    factory.reset();

    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::LAYER), 0);
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::COLUMN), 0);
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::NUCLEUS), 0);
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::REGION), 0);
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::LOBE), 0);
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::HEMISPHERE), 0);
    EXPECT_EQ(factory.getObjectCount(NeuralObjectFactory::ObjectType::BRAIN), 0);

    // Verify IDs reset to start of range
    auto newLayer = factory.createLayer();
    EXPECT_EQ(newLayer->getId(), 600000000000000ULL);
}

TEST(FactoryIntegrationTests, GetObjectTypeName) {
    EXPECT_STREQ(NeuralObjectFactory::getObjectTypeName(NeuralObjectFactory::ObjectType::LAYER), "Layer");
    EXPECT_STREQ(NeuralObjectFactory::getObjectTypeName(NeuralObjectFactory::ObjectType::COLUMN), "Column");
    EXPECT_STREQ(NeuralObjectFactory::getObjectTypeName(NeuralObjectFactory::ObjectType::NUCLEUS), "Nucleus");
    EXPECT_STREQ(NeuralObjectFactory::getObjectTypeName(NeuralObjectFactory::ObjectType::REGION), "Region");
    EXPECT_STREQ(NeuralObjectFactory::getObjectTypeName(NeuralObjectFactory::ObjectType::LOBE), "Lobe");
    EXPECT_STREQ(NeuralObjectFactory::getObjectTypeName(NeuralObjectFactory::ObjectType::HEMISPHERE), "Hemisphere");
    EXPECT_STREQ(NeuralObjectFactory::getObjectTypeName(NeuralObjectFactory::ObjectType::BRAIN), "Brain");
}

TEST(FactoryIntegrationTests, GetObjectTypeNameFromId) {
    NeuralObjectFactory factory;

    auto layer = factory.createLayer();
    auto column = factory.createColumn();
    auto brain = factory.createBrain();

    EXPECT_STREQ(NeuralObjectFactory::getObjectTypeName(layer->getId()), "Layer");
    EXPECT_STREQ(NeuralObjectFactory::getObjectTypeName(column->getId()), "Column");
    EXPECT_STREQ(NeuralObjectFactory::getObjectTypeName(brain->getId()), "Brain");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

