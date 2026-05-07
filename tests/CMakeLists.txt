# Test CMakeLists.txt
# Build GTest from source to ensure ABI compatibility with the project compiler
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
# Prevent overriding the parent project's compiler/linker settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# Disable install of GTest
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

set(GTest_FOUND TRUE)

if(GTest_FOUND)
    # Test executables
    add_executable(test_neuron test_neuron.cpp)
    target_link_libraries(test_neuron
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_cluster test_cluster.cpp)
    target_link_libraries(test_cluster
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_threading test_threading.cpp)
    target_link_libraries(test_threading
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_neural_components test_neural_components.cpp)
    target_link_libraries(test_neural_components
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_spike_processor test_spike_processor.cpp)
    target_link_libraries(test_spike_processor
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_neural_object_factory test_neural_object_factory.cpp)
    target_link_libraries(test_neural_object_factory
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_hierarchical_structures test_hierarchical_structures.cpp)
    target_link_libraries(test_hierarchical_structures
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_datastore test_datastore.cpp)
    target_link_libraries(test_datastore
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_adapters AdapterTests.cpp)
    target_link_libraries(test_adapters
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_spike_propagation test_spike_propagation.cpp)
    target_link_libraries(test_spike_propagation
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_retrograde_stdp test_retrograde_stdp.cpp)
    target_link_libraries(test_retrograde_stdp
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_binary_pattern test_binary_pattern.cpp)
    target_link_libraries(test_binary_pattern
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_dendritic_pattern_memory test_dendritic_pattern_memory.cpp)
    target_link_libraries(test_dendritic_pattern_memory
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_network_inspector test_network_inspector.cpp)
    target_link_libraries(test_network_inspector
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_activity_monitor test_activity_monitor.cpp)
    target_link_libraries(test_activity_monitor
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_performance_profiler test_performance_profiler.cpp)
    target_link_libraries(test_performance_profiler
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_network_validator test_network_validator.cpp)
    target_link_libraries(test_network_validator
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_assertions test_assertions.cpp)
    target_link_libraries(test_assertions
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_integration test_integration.cpp)
    target_link_libraries(test_integration
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_stress test_stress.cpp)
    target_link_libraries(test_stress
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_performance_regression test_performance_regression.cpp)
    target_link_libraries(test_performance_regression
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_network_builder test_network_builder.cpp)
    target_link_libraries(test_network_builder
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_connectivity test_connectivity.cpp)
    target_link_libraries(test_connectivity
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_model_importer test_model_importer.cpp)
    target_link_libraries(test_model_importer
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_declarative_loader test_declarative_loader.cpp)
    target_link_libraries(test_declarative_loader
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_classification_strategies test_classification_strategies.cpp)
    target_link_libraries(test_classification_strategies
        snnfw
        GTest::gtest
        GTest::gtest_main
    )

    # Note: test_main.cpp disabled until Main class is re-implemented
    # add_executable(test_main test_main.cpp)
    # target_link_libraries(test_main
    #     ${PROJECT_NAME}
    #     GTest::gtest
    #     GTest::gtest_main
    # )

    # Add tests
    add_test(NAME NeuronTests COMMAND test_neuron)
    add_test(NAME ClusterTests COMMAND test_cluster)
    add_test(NAME ThreadingTests COMMAND test_threading)
    add_test(NAME NeuralComponentsTests COMMAND test_neural_components)
    add_test(NAME SpikeProcessorTests COMMAND test_spike_processor)
    add_test(NAME NeuralObjectFactoryTests COMMAND test_neural_object_factory)
    add_test(NAME HierarchicalStructuresTests COMMAND test_hierarchical_structures)
    add_test(NAME DatastoreTests COMMAND test_datastore)
    add_test(NAME AdapterTests COMMAND test_adapters)
    add_test(NAME SpikePropagationTests COMMAND test_spike_propagation)
    add_test(NAME RetrogradeSTDPTests COMMAND test_retrograde_stdp)
    add_test(NAME BinaryPatternTests COMMAND test_binary_pattern)
    add_test(NAME DendriticPatternMemoryTests COMMAND test_dendritic_pattern_memory)
    add_test(NAME NetworkInspectorTests COMMAND test_network_inspector)
    add_test(NAME ActivityMonitorTests COMMAND test_activity_monitor)
    add_test(NAME PerformanceProfilerTests COMMAND test_performance_profiler)
    add_test(NAME NetworkValidatorTests COMMAND test_network_validator)
    add_test(NAME AssertionsTests COMMAND test_assertions)
    add_test(NAME IntegrationTests COMMAND test_integration)
    add_test(NAME StressTests COMMAND test_stress)
    add_test(NAME PerformanceRegressionTests COMMAND test_performance_regression)
    add_test(NAME NetworkBuilderTests COMMAND test_network_builder)
    add_test(NAME ConnectivityTests COMMAND test_connectivity)
    add_test(NAME ModelImporterTests COMMAND test_model_importer)
    add_test(NAME DeclarativeLoaderTests COMMAND test_declarative_loader)
    add_test(NAME ClassificationStrategyTests COMMAND test_classification_strategies)
    # add_test(NAME MainTests COMMAND test_main)

    # Many developers have Conda on their PATH, which can inject an older libstdc++
    # into the runtime search path and prevent GCC-built binaries from running.
    # Prefer the system runtime via LD_LIBRARY_PATH, while still allowing RUNPATH
    # to resolve project/built dependencies (e.g., libsonata).
    set(_SNNFW_SYSTEM_LD_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu:/usr/lib:/lib/x86_64-linux-gnu:/lib")
    set(_SNNFW_ALL_TESTS
        NeuronTests
        ClusterTests
        ThreadingTests
        NeuralComponentsTests
        SpikeProcessorTests
        NeuralObjectFactoryTests
        HierarchicalStructuresTests
        DatastoreTests
        AdapterTests
        SpikePropagationTests
        RetrogradeSTDPTests
        BinaryPatternTests
        DendriticPatternMemoryTests
        NetworkInspectorTests
        ActivityMonitorTests
        PerformanceProfilerTests
        NetworkValidatorTests
        AssertionsTests
        IntegrationTests
        StressTests
        PerformanceRegressionTests
        NetworkBuilderTests
        ConnectivityTests
        ModelImporterTests
        DeclarativeLoaderTests
        ClassificationStrategyTests
    )

    foreach(_test_name IN LISTS _SNNFW_ALL_TESTS)
        set_tests_properties(${_test_name}
            PROPERTIES
                ENVIRONMENT "LD_LIBRARY_PATH=${_SNNFW_SYSTEM_LD_LIBRARY_PATH}"
        )
    endforeach()
else()
    message(STATUS "Google Test not found. Skipping tests.")
endif()

# Diagnostics/smoke tests that don't require GTest
add_subdirectory(diagnostics)
