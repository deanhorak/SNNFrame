add_executable(snnfw_diag_propagation propagation_test.cpp)
target_link_libraries(snnfw_diag_propagation PRIVATE snnfw)

add_executable(snnfw_diag_classifier classifier_test.cpp)
target_link_libraries(snnfw_diag_classifier PRIVATE snnfw)

add_executable(snnfw_diag_spike_delivery spike_delivery_test.cpp)
target_link_libraries(snnfw_diag_spike_delivery PRIVATE snnfw)

add_executable(snnfw_diag_postsynaptic postsynaptic_fire_test.cpp)
target_link_libraries(snnfw_diag_postsynaptic PRIVATE snnfw)

add_executable(snnfw_diag_layered layered_propagation_test.cpp)
target_link_libraries(snnfw_diag_layered PRIVATE snnfw)

add_executable(snnfw_diag_pattern pattern_learning_test.cpp)
add_executable(snnfw_diag_l4_l5_delivery l4_l5_delivery_test.cpp)
add_executable(snnfw_diag_l5_output_connectivity l5_output_connectivity_test.cpp)
add_executable(snnfw_diag_l5_classification l5_classification_test.cpp)
add_executable(snnfw_diag_retina_encoding retina_encoding_test.cpp)
target_link_libraries(snnfw_diag_pattern PRIVATE snnfw)
target_link_libraries(snnfw_diag_l4_l5_delivery PRIVATE snnfw)
target_link_libraries(snnfw_diag_l5_output_connectivity PRIVATE snnfw)
target_link_libraries(snnfw_diag_l5_classification PRIVATE snnfw)
target_link_libraries(snnfw_diag_retina_encoding PRIVATE snnfw)

add_test(NAME diag_spike_delivery COMMAND snnfw_diag_spike_delivery)
add_test(NAME diag_postsynaptic COMMAND snnfw_diag_postsynaptic)
add_test(NAME diag_layered COMMAND snnfw_diag_layered)
add_test(NAME diag_pattern COMMAND snnfw_diag_pattern)
add_test(NAME diag_l4_l5_delivery COMMAND snnfw_diag_l4_l5_delivery)
add_test(NAME diag_l5_output_connectivity COMMAND snnfw_diag_l5_output_connectivity)
add_test(NAME diag_l5_classification COMMAND snnfw_diag_l5_classification)
add_test(NAME diag_propagation COMMAND snnfw_diag_propagation)
add_test(NAME diag_classifier COMMAND snnfw_diag_classifier)
add_test(NAME diag_retina_encoding COMMAND snnfw_diag_retina_encoding)

# Force system libstdc++ to avoid conda shims breaking runtime
set(_SNNFW_DIAG_LD_LIBRARY_PATH "/usr/lib/x86_64-linux-gnu:/usr/lib:/lib/x86_64-linux-gnu:/lib")
set_tests_properties(
    diag_spike_delivery
    diag_postsynaptic
    diag_layered
    diag_pattern
    diag_l4_l5_delivery
    diag_l5_output_connectivity
    diag_l5_classification
    diag_propagation
    diag_classifier
    diag_retina_encoding
    PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=${_SNNFW_DIAG_LD_LIBRARY_PATH}"
)
