# Test CMakeLists.txt
find_package(GTest QUIET)

if(GTest_FOUND)
    # Test executables
    add_executable(test_neuron test_neuron.cpp)
    target_link_libraries(test_neuron
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_cluster test_cluster.cpp)
    target_link_libraries(test_cluster
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_threading test_threading.cpp)
    target_link_libraries(test_threading
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_neural_components test_neural_components.cpp)
    target_link_libraries(test_neural_components
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_spike_processor test_spike_processor.cpp)
    target_link_libraries(test_spike_processor
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_neural_object_factory test_neural_object_factory.cpp)
    target_link_libraries(test_neural_object_factory
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_hierarchical_structures test_hierarchical_structures.cpp)
    target_link_libraries(test_hierarchical_structures
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_datastore test_datastore.cpp)
    target_link_libraries(test_datastore
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_adapters AdapterTests.cpp)
    target_link_libraries(test_adapters
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_spike_propagation test_spike_propagation.cpp)
    target_link_libraries(test_spike_propagation
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_retrograde_stdp test_retrograde_stdp.cpp)
    target_link_libraries(test_retrograde_stdp
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_binary_pattern test_binary_pattern.cpp)
    target_link_libraries(test_binary_pattern
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_network_inspector test_network_inspector.cpp)
    target_link_libraries(test_network_inspector
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_activity_monitor test_activity_monitor.cpp)
    target_link_libraries(test_activity_monitor
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_performance_profiler test_performance_profiler.cpp)
    target_link_libraries(test_performance_profiler
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_network_validator test_network_validator.cpp)
    target_link_libraries(test_network_validator
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_assertions test_assertions.cpp)
    target_link_libraries(test_assertions
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_integration test_integration.cpp)
    target_link_libraries(test_integration
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_stress test_stress.cpp)
    target_link_libraries(test_stress
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_performance_regression test_performance_regression.cpp)
    target_link_libraries(test_performance_regression
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_network_builder test_network_builder.cpp)
    target_link_libraries(test_network_builder
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_connectivity test_connectivity.cpp)
    target_link_libraries(test_connectivity
        ${PROJECT_NAME}
        GTest::gtest
        GTest::gtest_main
    )

    add_executable(test_model_importer test_model_importer.cpp)
    target_link_libraries(test_model_importer
        ${PROJECT_NAME}
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
    # add_test(NAME MainTests COMMAND test_main)
else()
    message(STATUS "Google Test not found. Skipping tests.")
endif()
