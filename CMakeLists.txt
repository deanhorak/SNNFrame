cmake_minimum_required(VERSION 3.16)

project(SNNFrame
    VERSION 1.0.0
    DESCRIPTION "Spiking Neural Network Framework"
    LANGUAGES C CXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find threading library
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Set build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Find OpenMP for parallelization
find_package(OpenMP REQUIRED)

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra -Wpedantic -fsanitize=address -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address")

# Include directories
include_directories(include)

# Find spdlog
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
    message(STATUS "spdlog not found via find_package, attempting FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.12.0
    )
    FetchContent_MakeAvailable(spdlog)
endif()

# Fetch nlohmann/json for serialization
message(STATUS "Fetching nlohmann/json...")
include(FetchContent)
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_GetProperties(nlohmann_json)
if(NOT nlohmann_json_POPULATED)
    FetchContent_Populate(nlohmann_json)
    add_subdirectory(${nlohmann_json_SOURCE_DIR} ${nlohmann_json_BINARY_DIR})
endif()
set(nlohmann_json_DIR ${nlohmann_json_BINARY_DIR})

# Fetch HighFive (HDF5 C++ wrapper)
message(STATUS "Fetching HighFive...")
option(HIGHFIVE_USE_BOOST "Enable Boost support" OFF)
option(HIGHFIVE_USE_EIGEN "Enable Eigen support" OFF)
option(HIGHFIVE_BUILD_DOCS "Build documentation" OFF)
option(HIGHFIVE_EXAMPLES "Build examples" OFF)
option(HIGHFIVE_UNIT_TESTS "Build unit tests" OFF)
FetchContent_Declare(
    HighFive
    GIT_REPOSITORY https://github.com/BlueBrain/HighFive.git
    GIT_TAG v2.9.0
)
FetchContent_GetProperties(HighFive)
if(NOT highfive_POPULATED)
    FetchContent_Populate(HighFive)
    add_subdirectory(${highfive_SOURCE_DIR} ${highfive_BINARY_DIR})
endif()
set(HighFive_DIR ${highfive_BINARY_DIR})

# Fetch libsonata for SONATA format support
message(STATUS "Fetching libsonata...")
option(SONATA_TESTS "Build libsonata tests" OFF)
option(SONATA_PYTHON "Build libsonata Python bindings" OFF)
option(EXTLIB_FROM_SUBMODULES "Use git submodules for external libraries" OFF)

set(CMAKE_CXX_FLAGS_BACKUP "${CMAKE_CXX_FLAGS}")
string(REPLACE "-Werror" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

FetchContent_Declare(
    libsonata
    GIT_REPOSITORY https://github.com/BlueBrain/libsonata.git
    GIT_TAG v0.1.27
)
FetchContent_MakeAvailable(libsonata)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_BACKUP}")

# Find RocksDB for persistent storage
message(STATUS "Looking for RocksDB...")
find_package(RocksDB QUIET CONFIG)
if(RocksDB_FOUND)
    message(STATUS "Found RocksDB via find_package")
else()
    message(STATUS "RocksDB not found via find_package, attempting to find system library")
    find_path(ROCKSDB_INCLUDE_DIR rocksdb/db.h PATHS /usr/include /usr/local/include)
    find_library(ROCKSDB_LIBRARY NAMES rocksdb PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)

    if(ROCKSDB_INCLUDE_DIR AND ROCKSDB_LIBRARY)
        message(STATUS "Found RocksDB library: ${ROCKSDB_LIBRARY}")
        message(STATUS "Found RocksDB include: ${ROCKSDB_INCLUDE_DIR}")
        set(ROCKSDB_LIBRARIES ${ROCKSDB_LIBRARY})
        set(ROCKSDB_INCLUDE_DIRS ${ROCKSDB_INCLUDE_DIR})
        set(ROCKSDB_FOUND TRUE)
    else()
        message(WARNING "RocksDB not found. Please install it:")
        message(WARNING "  Ubuntu/Debian: sudo apt-get install librocksdb-dev")
        message(WARNING "  macOS: brew install rocksdb")
        set(ROCKSDB_FOUND FALSE)
    endif()
endif()

# Visualization dependencies
set(GLFW_DIR ${CMAKE_SOURCE_DIR}/third_party/glfw)
set(GLAD_DIR ${CMAKE_SOURCE_DIR}/third_party/glad)
set(GLM_DIR ${CMAKE_SOURCE_DIR}/third_party/glm)
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/third_party/imgui)

# GLFW
if(EXISTS ${GLFW_DIR})
    message(STATUS "Found GLFW in third_party")
    option(GLFW_BUILD_DOCS "Build the GLFW documentation" OFF)
    option(GLFW_BUILD_TESTS "Build the GLFW test programs" OFF)
    option(GLFW_BUILD_EXAMPLES "Build the GLFW example programs" OFF)
    add_subdirectory(${GLFW_DIR})
else()
    message(WARNING "GLFW not found in third_party - visualization will not be available")
endif()

# GLAD
if(EXISTS ${GLAD_DIR})
    message(STATUS "Found GLAD in third_party")
    add_library(glad STATIC ${GLAD_DIR}/src/glad.c)
    target_include_directories(glad PUBLIC ${GLAD_DIR}/include)
else()
    message(WARNING "GLAD not found in third_party - visualization will not be available")
endif()

# ImGui
if(EXISTS ${IMGUI_DIR})
    message(STATUS "Found ImGui in third_party")
    add_library(imgui STATIC
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_demo.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
        ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
    )
    target_include_directories(imgui PUBLIC ${IMGUI_DIR} ${IMGUI_DIR}/backends)
    target_link_libraries(imgui PUBLIC glfw glad)
else()
    message(WARNING "ImGui not found in third_party - visualization will not be available")
endif()

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(experiments)
