#include <gtest/gtest.h>
#include "snnfw/Assertions.h"
#include "snnfw/Datastore.h"
#include "snnfw/NeuralObjectFactory.h"
#include "snnfw/Logger.h"
#include <vector>
#include <filesystem>

using namespace snnfw;

class AssertionsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize logger
        Logger::getInstance().initialize("/tmp/test_assertions.log", spdlog::level::warn);

        // Create datastore
        std::filesystem::remove_all("/tmp/test_assertions_db");
        datastore = std::make_unique<Datastore>("/tmp/test_assertions_db", 1000000);
        
        // Create factory
        factory = std::make_unique<NeuralObjectFactory>();
        
        // Reset assertion config to non-strict mode for most tests
        AssertionConfig::getInstance().setStrictMode(false);
        AssertionConfig::getInstance().setThrowOnError(false);
    }
    
    void TearDown() override {
        factory.reset();
        datastore.reset();
        std::filesystem::remove_all("/tmp/test_assertions_db");
    }
    
    std::unique_ptr<Datastore> datastore;
    std::unique_ptr<NeuralObjectFactory> factory;
};

// Test basic assertion that passes
TEST_F(AssertionsTest, AssertPassingCondition) {
    EXPECT_NO_THROW({
        SNNFW_ASSERT(true, "This should not fail");
        SNNFW_ASSERT(1 == 1, "Math works");
        SNNFW_ASSERT(5 > 3, "Five is greater than three");
    });
}

// Test basic assertion that fails in non-strict mode
TEST_F(AssertionsTest, AssertFailingConditionNonStrict) {
    AssertionConfig::getInstance().setStrictMode(false);
    AssertionConfig::getInstance().setThrowOnError(false);
    
    // Should log error but not throw
    EXPECT_NO_THROW({
        SNNFW_ASSERT(false, "This assertion fails but doesn't throw");
    });
}

// Test basic assertion that fails in strict mode
TEST_F(AssertionsTest, AssertFailingConditionStrict) {
    AssertionConfig::getInstance().setStrictMode(true);
    AssertionConfig::getInstance().setThrowOnError(true);
    
    // Should throw AssertionError
    EXPECT_THROW({
        SNNFW_ASSERT(false, "This assertion fails and throws");
    }, AssertionError);
}

// Test assertion with formatted message
TEST_F(AssertionsTest, AssertFormattedMessage) {
    AssertionConfig::getInstance().setThrowOnError(true);
    
    try {
        SNNFW_ASSERT(false, "Value {} is not equal to {}", 5, 10);
        FAIL() << "Should have thrown AssertionError";
    } catch (const AssertionError& e) {
        std::string msg = e.getMessage();
        EXPECT_NE(msg.find("5"), std::string::npos);
        EXPECT_NE(msg.find("10"), std::string::npos);
    }
}

// Test ID existence check - passing
TEST_F(AssertionsTest, RequireIdExistsPass) {
    auto neuron = factory->createNeuron(100, 0.85, 10);
    datastore->put(neuron);
    
    EXPECT_NO_THROW({
        SNNFW_REQUIRE_ID_EXISTS(neuron->getId(), *datastore);
    });
}

// Test ID existence check - failing in non-strict mode
TEST_F(AssertionsTest, RequireIdExistsFailNonStrict) {
    AssertionConfig::getInstance().setThrowOnError(false);
    
    EXPECT_NO_THROW({
        SNNFW_REQUIRE_ID_EXISTS(999999, *datastore);
    });
}

// Test ID existence check - failing in strict mode
TEST_F(AssertionsTest, RequireIdExistsFailStrict) {
    AssertionConfig::getInstance().setThrowOnError(true);
    
    EXPECT_THROW({
        SNNFW_REQUIRE_ID_EXISTS(999999, *datastore);
    }, AssertionError);
}

// Test null pointer check - passing
TEST_F(AssertionsTest, RequireNotNullPass) {
    auto neuron = factory->createNeuron(100, 0.85, 10);
    
    EXPECT_NO_THROW({
        SNNFW_REQUIRE_NOT_NULL(neuron, "neuron");
    });
}

// Test null pointer check - failing
TEST_F(AssertionsTest, RequireNotNullFail) {
    AssertionConfig::getInstance().setThrowOnError(true);
    
    std::shared_ptr<Neuron> nullNeuron = nullptr;
    
    EXPECT_THROW({
        SNNFW_REQUIRE_NOT_NULL(nullNeuron, "nullNeuron");
    }, AssertionError);
}

// Test range check - passing
TEST_F(AssertionsTest, RequireRangePass) {
    int value = 50;
    
    EXPECT_NO_THROW({
        SNNFW_REQUIRE_RANGE(value, 0, 100, "value");
        SNNFW_REQUIRE_RANGE(value, 50, 50, "exact value");
    });
}

// Test range check - failing below minimum
TEST_F(AssertionsTest, RequireRangeFailBelowMin) {
    AssertionConfig::getInstance().setThrowOnError(true);
    
    int value = -5;
    
    EXPECT_THROW({
        SNNFW_REQUIRE_RANGE(value, 0, 100, "value");
    }, AssertionError);
}

// Test range check - failing above maximum
TEST_F(AssertionsTest, RequireRangeFailAboveMax) {
    AssertionConfig::getInstance().setThrowOnError(true);
    
    int value = 150;
    
    EXPECT_THROW({
        SNNFW_REQUIRE_RANGE(value, 0, 100, "value");
    }, AssertionError);
}

// Test positive value check - passing
TEST_F(AssertionsTest, RequirePositivePass) {
    double value = 5.5;
    
    EXPECT_NO_THROW({
        SNNFW_REQUIRE_POSITIVE(value, "value");
    });
}

// Test positive value check - failing with zero
TEST_F(AssertionsTest, RequirePositiveFailZero) {
    AssertionConfig::getInstance().setThrowOnError(true);
    
    double value = 0.0;
    
    EXPECT_THROW({
        SNNFW_REQUIRE_POSITIVE(value, "value");
    }, AssertionError);
}

// Test positive value check - failing with negative
TEST_F(AssertionsTest, RequirePositiveFailNegative) {
    AssertionConfig::getInstance().setThrowOnError(true);
    
    double value = -5.5;
    
    EXPECT_THROW({
        SNNFW_REQUIRE_POSITIVE(value, "value");
    }, AssertionError);
}

// Test non-negative value check - passing
TEST_F(AssertionsTest, RequireNonNegativePass) {
    double value1 = 5.5;
    double value2 = 0.0;
    
    EXPECT_NO_THROW({
        SNNFW_REQUIRE_NON_NEGATIVE(value1, "value1");
        SNNFW_REQUIRE_NON_NEGATIVE(value2, "value2");
    });
}

// Test non-negative value check - failing
TEST_F(AssertionsTest, RequireNonNegativeFail) {
    AssertionConfig::getInstance().setThrowOnError(true);
    
    double value = -0.1;
    
    EXPECT_THROW({
        SNNFW_REQUIRE_NON_NEGATIVE(value, "value");
    }, AssertionError);
}

// Test container not empty check - passing
TEST_F(AssertionsTest, RequireNotEmptyPass) {
    std::vector<int> vec = {1, 2, 3};
    
    EXPECT_NO_THROW({
        SNNFW_REQUIRE_NOT_EMPTY(vec, "vec");
    });
}

// Test container not empty check - failing
TEST_F(AssertionsTest, RequireNotEmptyFail) {
    AssertionConfig::getInstance().setThrowOnError(true);
    
    std::vector<int> vec;
    
    EXPECT_THROW({
        SNNFW_REQUIRE_NOT_EMPTY(vec, "vec");
    }, AssertionError);
}

// Test container size limit check - passing
TEST_F(AssertionsTest, RequireSizeLimitPass) {
    std::vector<int> vec = {1, 2, 3};
    
    EXPECT_NO_THROW({
        SNNFW_REQUIRE_SIZE_LIMIT(vec, 10, "vec");
        SNNFW_REQUIRE_SIZE_LIMIT(vec, 3, "vec");
    });
}

// Test container size limit check - failing
TEST_F(AssertionsTest, RequireSizeLimitFail) {
    AssertionConfig::getInstance().setThrowOnError(true);
    
    std::vector<int> vec = {1, 2, 3, 4, 5};
    
    EXPECT_THROW({
        SNNFW_REQUIRE_SIZE_LIMIT(vec, 3, "vec");
    }, AssertionError);
}

// Test ID range check - passing
TEST_F(AssertionsTest, RequireIdRangePass) {
    uint64_t neuronId = 100000000000000;
    
    EXPECT_NO_THROW({
        SNNFW_REQUIRE_ID_RANGE(neuronId, 100000000000000, 199999999999999, "Neuron");
    });
}

// Test ID range check - failing
TEST_F(AssertionsTest, RequireIdRangeFail) {
    AssertionConfig::getInstance().setThrowOnError(true);
    
    uint64_t invalidId = 999;
    
    EXPECT_THROW({
        SNNFW_REQUIRE_ID_RANGE(invalidId, 100000000000000, 199999999999999, "Neuron");
    }, AssertionError);
}

// Test unconditional failure
TEST_F(AssertionsTest, UnconditionalFail) {
    AssertionConfig::getInstance().setThrowOnError(true);
    
    EXPECT_THROW({
        SNNFW_FAIL("This should always fail");
    }, AssertionError);
}

// Test AssertionError properties
TEST_F(AssertionsTest, AssertionErrorProperties) {
    AssertionConfig::getInstance().setThrowOnError(true);
    
    try {
        SNNFW_ASSERT(false, "Test error message");
        FAIL() << "Should have thrown AssertionError";
    } catch (const AssertionError& e) {
        EXPECT_EQ(e.getMessage(), "Test error message");
        EXPECT_NE(e.getFile().find("test_assertions.cpp"), std::string::npos);
        EXPECT_GT(e.getLine(), 0);
        EXPECT_EQ(e.getCondition(), "false");
    }
}

// Test runtime mode switching
TEST_F(AssertionsTest, RuntimeModeSwitch) {
    // Start in non-strict mode
    AssertionConfig::getInstance().setThrowOnError(false);
    EXPECT_NO_THROW({
        SNNFW_ASSERT(false, "Non-strict mode");
    });
    
    // Switch to strict mode
    AssertionConfig::getInstance().setThrowOnError(true);
    EXPECT_THROW({
        SNNFW_ASSERT(false, "Strict mode");
    }, AssertionError);
    
    // Switch back to non-strict mode
    AssertionConfig::getInstance().setThrowOnError(false);
    EXPECT_NO_THROW({
        SNNFW_ASSERT(false, "Non-strict mode again");
    });
}

