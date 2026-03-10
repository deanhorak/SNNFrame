# EMNIST Letters Training Experiment (71.93% accuracy)
add_executable(emnist_letters_training emnist_letters_training.cpp)
target_link_libraries(emnist_letters_training PRIVATE snnfw)
target_include_directories(emnist_letters_training PRIVATE ${CMAKE_SOURCE_DIR}/include)

# EMNIST Letters Visualization Experiment
add_executable(emnist_letters_visualized emnist_letters_visualized.cpp)
target_link_libraries(emnist_letters_visualized PRIVATE snnfw)
target_include_directories(emnist_letters_visualized PRIVATE ${CMAKE_SOURCE_DIR}/include)

# EMNIST SONATA-driven Training Experiment
add_executable(emnist_sonata_training emnist_sonata_training.cpp)
target_link_libraries(emnist_sonata_training PRIVATE snnfw)
target_include_directories(emnist_sonata_training PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Interneuron TCP adapter demo
add_executable(interneuron_adapter_demo interneuron_adapter_demo.cpp)
target_link_libraries(interneuron_adapter_demo PRIVATE snnfw)
target_include_directories(interneuron_adapter_demo PRIVATE ${CMAKE_SOURCE_DIR}/include)
