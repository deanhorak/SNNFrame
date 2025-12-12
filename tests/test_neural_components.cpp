#include <gtest/gtest.h>
#include "snnfw/EventObject.h"
#include "snnfw/ActionPotential.h"
#include "snnfw/Axon.h"
#include "snnfw/Dendrite.h"
#include "snnfw/Synapse.h"
#include <memory>

using namespace snnfw;

// ============================================================================
// EventObject Tests
// ============================================================================

class EventObjectTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(EventObjectTest, ConstructorAndGetters) {
    EventObject event(10.5);
    EXPECT_DOUBLE_EQ(event.getScheduledTime(), 10.5);
    EXPECT_STREQ(event.getEventType(), "EventObject");
}

TEST_F(EventObjectTest, SetScheduledTime) {
    EventObject event(10.0);
    event.setScheduledTime(20.5);
    EXPECT_DOUBLE_EQ(event.getScheduledTime(), 20.5);
}

TEST_F(EventObjectTest, DefaultConstructor) {
    EventObject event;
    EXPECT_DOUBLE_EQ(event.getScheduledTime(), 0.0);
}

// ============================================================================
// ActionPotential Tests
// ============================================================================

class ActionPotentialTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ActionPotentialTest, ConstructorAndGetters) {
    ActionPotential ap(100, 200, 15.5, 0.8);
    
    EXPECT_EQ(ap.getSynapseId(), 100);
    EXPECT_EQ(ap.getDendriteId(), 200);
    EXPECT_DOUBLE_EQ(ap.getScheduledTime(), 15.5);
    EXPECT_DOUBLE_EQ(ap.getAmplitude(), 0.8);
    EXPECT_STREQ(ap.getEventType(), "ActionPotential");
}

TEST_F(ActionPotentialTest, DefaultAmplitude) {
    ActionPotential ap(100, 200, 15.5);
    EXPECT_DOUBLE_EQ(ap.getAmplitude(), 1.0);
}

TEST_F(ActionPotentialTest, SetAmplitude) {
    ActionPotential ap(100, 200, 15.5, 0.5);
    ap.setAmplitude(0.9);
    EXPECT_DOUBLE_EQ(ap.getAmplitude(), 0.9);
}

TEST_F(ActionPotentialTest, InheritsFromEventObject) {
    ActionPotential ap(100, 200, 15.5);
    EventObject* eventPtr = &ap;
    EXPECT_DOUBLE_EQ(eventPtr->getScheduledTime(), 15.5);
}

// ============================================================================
// Axon Tests
// ============================================================================

class AxonTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(AxonTest, ConstructorAndGetters) {
    Axon axon(42, 1001);
    
    EXPECT_EQ(axon.getSourceNeuronId(), 42);
    EXPECT_EQ(axon.getId(), 1001);
}

TEST_F(AxonTest, DefaultAxonId) {
    Axon axon(42);
    EXPECT_EQ(axon.getSourceNeuronId(), 42);
    EXPECT_EQ(axon.getId(), 0);
}

TEST_F(AxonTest, SetSourceNeuronId) {
    Axon axon(42, 1001);
    axon.setSourceNeuronId(99);
    EXPECT_EQ(axon.getSourceNeuronId(), 99);
}

TEST_F(AxonTest, InheritsFromNeuralObject) {
    Axon axon(42, 1001);
    NeuralObject* neuralPtr = &axon;
    EXPECT_EQ(neuralPtr->getId(), 1001);
}

TEST_F(AxonTest, AddSynapse) {
    Axon axon(42, 1001);

    axon.addSynapse(100);
    EXPECT_EQ(axon.getSynapseCount(), 1);

    axon.addSynapse(101);
    axon.addSynapse(102);
    EXPECT_EQ(axon.getSynapseCount(), 3);
}

TEST_F(AxonTest, GetSynapseIds) {
    Axon axon(42, 1001);

    axon.addSynapse(100);
    axon.addSynapse(101);

    const auto& synapseIds = axon.getSynapseIds();
    EXPECT_EQ(synapseIds.size(), 2);
    EXPECT_EQ(synapseIds[0], 100);
    EXPECT_EQ(synapseIds[1], 101);
}

TEST_F(AxonTest, RemoveSynapse) {
    Axon axon(42, 1001);

    axon.addSynapse(100);
    axon.addSynapse(101);
    axon.addSynapse(102);

    EXPECT_TRUE(axon.removeSynapse(101));
    EXPECT_EQ(axon.getSynapseCount(), 2);

    const auto& synapseIds = axon.getSynapseIds();
    EXPECT_EQ(synapseIds[0], 100);
    EXPECT_EQ(synapseIds[1], 102);
}

TEST_F(AxonTest, RemoveNonexistentSynapse) {
    Axon axon(42, 1001);

    axon.addSynapse(100);
    EXPECT_FALSE(axon.removeSynapse(999));
    EXPECT_EQ(axon.getSynapseCount(), 1);
}

TEST_F(AxonTest, AddDuplicateSynapse) {
    Axon axon(42, 1001);

    axon.addSynapse(100);
    axon.addSynapse(100);  // Duplicate

    // Should not add duplicate
    EXPECT_EQ(axon.getSynapseCount(), 1);
}

// ============================================================================
// Dendrite Tests
// ============================================================================

class DendriteTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(DendriteTest, ConstructorAndGetters) {
    Dendrite dendrite(42, 2001);
    
    EXPECT_EQ(dendrite.getTargetNeuronId(), 42);
    EXPECT_EQ(dendrite.getId(), 2001);
    EXPECT_EQ(dendrite.getSynapseCount(), 0);
}

TEST_F(DendriteTest, DefaultDendriteId) {
    Dendrite dendrite(42);
    EXPECT_EQ(dendrite.getTargetNeuronId(), 42);
    EXPECT_EQ(dendrite.getId(), 0);
}

TEST_F(DendriteTest, SetTargetNeuronId) {
    Dendrite dendrite(42, 2001);
    dendrite.setTargetNeuronId(99);
    EXPECT_EQ(dendrite.getTargetNeuronId(), 99);
}

TEST_F(DendriteTest, AddSynapse) {
    Dendrite dendrite(42, 2001);
    
    dendrite.addSynapse(100);
    EXPECT_EQ(dendrite.getSynapseCount(), 1);
    
    dendrite.addSynapse(101);
    dendrite.addSynapse(102);
    EXPECT_EQ(dendrite.getSynapseCount(), 3);
}

TEST_F(DendriteTest, GetSynapseIds) {
    Dendrite dendrite(42, 2001);
    
    dendrite.addSynapse(100);
    dendrite.addSynapse(101);
    dendrite.addSynapse(102);
    
    const auto& synapseIds = dendrite.getSynapseIds();
    EXPECT_EQ(synapseIds.size(), 3);
    EXPECT_EQ(synapseIds[0], 100);
    EXPECT_EQ(synapseIds[1], 101);
    EXPECT_EQ(synapseIds[2], 102);
}

TEST_F(DendriteTest, AddDuplicateSynapse) {
    Dendrite dendrite(42, 2001);
    
    dendrite.addSynapse(100);
    dendrite.addSynapse(100);  // Duplicate
    
    // Should only be added once
    EXPECT_EQ(dendrite.getSynapseCount(), 1);
}

TEST_F(DendriteTest, RemoveSynapse) {
    Dendrite dendrite(42, 2001);
    
    dendrite.addSynapse(100);
    dendrite.addSynapse(101);
    dendrite.addSynapse(102);
    
    bool removed = dendrite.removeSynapse(101);
    EXPECT_TRUE(removed);
    EXPECT_EQ(dendrite.getSynapseCount(), 2);
    
    const auto& synapseIds = dendrite.getSynapseIds();
    EXPECT_EQ(synapseIds[0], 100);
    EXPECT_EQ(synapseIds[1], 102);
}

TEST_F(DendriteTest, RemoveNonexistentSynapse) {
    Dendrite dendrite(42, 2001);
    
    dendrite.addSynapse(100);
    
    bool removed = dendrite.removeSynapse(999);
    EXPECT_FALSE(removed);
    EXPECT_EQ(dendrite.getSynapseCount(), 1);
}

TEST_F(DendriteTest, ReceiveSpike) {
    Dendrite dendrite(42, 2001);
    
    auto ap = std::make_shared<ActionPotential>(100, 2001, 10.5, 0.8);
    
    // Should not crash
    EXPECT_NO_THROW(dendrite.receiveSpike(ap));
}

TEST_F(DendriteTest, ReceiveNullSpike) {
    Dendrite dendrite(42, 2001);
    
    // Should handle null gracefully
    EXPECT_NO_THROW(dendrite.receiveSpike(nullptr));
}

TEST_F(DendriteTest, InheritsFromNeuralObject) {
    Dendrite dendrite(42, 2001);
    NeuralObject* neuralPtr = &dendrite;
    EXPECT_EQ(neuralPtr->getId(), 2001);
}

// ============================================================================
// Synapse Tests
// ============================================================================

class SynapseTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(SynapseTest, ConstructorAndGetters) {
    Synapse synapse(1001, 2001, 0.5, 1.5, 3001);
    
    EXPECT_EQ(synapse.getAxonId(), 1001);
    EXPECT_EQ(synapse.getDendriteId(), 2001);
    EXPECT_DOUBLE_EQ(synapse.getWeight(), 0.5);
    EXPECT_DOUBLE_EQ(synapse.getDelay(), 1.5);
    EXPECT_EQ(synapse.getId(), 3001);
}

TEST_F(SynapseTest, DefaultParameters) {
    Synapse synapse(1001, 2001);
    
    EXPECT_EQ(synapse.getAxonId(), 1001);
    EXPECT_EQ(synapse.getDendriteId(), 2001);
    EXPECT_DOUBLE_EQ(synapse.getWeight(), 1.0);
    EXPECT_DOUBLE_EQ(synapse.getDelay(), 1.0);
    EXPECT_EQ(synapse.getId(), 0);
}

TEST_F(SynapseTest, SetWeight) {
    Synapse synapse(1001, 2001, 0.5);
    
    synapse.setWeight(0.8);
    EXPECT_DOUBLE_EQ(synapse.getWeight(), 0.8);
}

TEST_F(SynapseTest, SetDelay) {
    Synapse synapse(1001, 2001, 0.5, 1.0);
    
    synapse.setDelay(2.5);
    EXPECT_DOUBLE_EQ(synapse.getDelay(), 2.5);
}

TEST_F(SynapseTest, ModifyWeight) {
    Synapse synapse(1001, 2001, 0.5);
    
    synapse.modifyWeight(0.2);
    EXPECT_DOUBLE_EQ(synapse.getWeight(), 0.7);
    
    synapse.modifyWeight(-0.3);
    EXPECT_DOUBLE_EQ(synapse.getWeight(), 0.4);
}

TEST_F(SynapseTest, InheritsFromNeuralObject) {
    Synapse synapse(1001, 2001, 0.5, 1.5, 3001);
    NeuralObject* neuralPtr = &synapse;
    EXPECT_EQ(neuralPtr->getId(), 3001);
}

// Main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

