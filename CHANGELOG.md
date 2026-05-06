# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-02-10

### Added

- **Experiment Framework Components**
  - `ExperimentConfig`: Centralized configuration structure for all experiment parameters
  - `SpikeEncoder`: Converts pixel intensities to spike times with memory management
  - `CompetitionManager`: Winner-take-all competition in L4, L5, and output layers with maskMinActive gating
  - `KNNClassifier`: k-NN and centroid-based classification using L5 activation patterns
  - `SupervisedTeacher`: Forces correct output neurons to fire and learn during training
  - `TrainingPipeline`: Multi-pass training orchestrator with convergence detection
  - `ExperimentRunner`: Top-level coordinator that loads config, builds network, runs training

- **Connectivity Pattern Extensions**
  - `TiledReceptiveFieldPattern`: Tile-based receptive fields for spatial locality
  - Support for `tiles_per_side`, `tiles_per_column`, `input_grid_size`, `target_grid_size` parameters
  - Deterministic tile selection per column with patch-based connectivity within tiles
  - `maskMinActive` threshold gating for column participation in competition

- **SONATA Format Extensions**
  - Full model description support in `snnframe` extension section
  - Brain hierarchy parsing (Brain, Hemisphere, Lobe, Region, Nucleus, Column, ColumnTemplate, Layer, Population)
  - Projection parsing with connectivity rules and synapse groups
  - Simulation config parsing (spike_processor, stdp, competition)
  - Saccade configuration with fixation regions
  - Enhanced input_layer and output_layer parsing with nested neuron_params

- **Example SONATA Model**
  - `configs/emnist_v1_sonata/circuit_config.json`: Full 6-layer V1 architecture for EMNIST letters
  - 16 cortical columns (8 orientations × 2 frequencies)
  - Complete layer populations and projections
  - Gabor filter and saccade configuration

### Performance

- **Spike Delivery Optimizations** (~15% performance improvement)
  - Eliminated thread creation overhead by submitting work directly to thread pool
  - Used `shared_ptr` to avoid copying event vectors for each task
  - Optimized `Neuron::insertSpike()` to track max spike time (O(1) instead of O(n))
  - Added synchronous delivery for small batches (<50 events) to avoid thread pool overhead
  - Added performance instrumentation for bottleneck identification

### Changed

- Updated all documentation to include declarative loading and experiment framework
- Enhanced `FORMAT_REFERENCE.md` with tiled_receptive_field pattern documentation
- Updated `DEVELOPER_MANUAL.md` with experiment framework usage patterns

## [1.1.0] - 2026-02-06

### Added

- **Declarative Network Loading System**
  - Define neural networks in configuration files instead of writing C++ code
  - `DeclarativeLoader` API: single entry-point that auto-detects format and constructs the network
  - `NetworkIR`: common intermediate representation for all formats
  - `NetworkConstructor`: four-phase construction (hierarchy → neurons → connectivity → runtime)
  - Extensible `FormatParser` interface for adding new formats

- **Native JSON Format** (`.snnf.json`)
  - Full SNNFrame configuration including column templates, path-based connectivity
  - Named neuron parameter sets (define once, reference many times)
  - Column template expansion: parameterized generation (e.g., 8 orientations × 2 frequencies = 16 columns)
  - Path-based projection targets using glob patterns (e.g., `V1/*/L4`)
  - Gabor filter, saccade, and simulation configuration
  - Synapse group support for differential STDP treatment

- **SONATA Format** (`circuit_config.json`, `.sonata.json`)
  - Blue Brain Project / Allen Institute HDF5-based format
  - Reads `circuit_config.json` with manifest variable resolution (`$BASE_DIR`, `$NETWORK_DIR`)
  - Loads node populations from HDF5 files via libsonata
  - Loads edge populations with weight, delay, and connectivity
  - SNNFrame extensions via `"snnframe"` section in circuit_config.json

- **NeuroML Format** (`.nml`, `.neuroml`)
  - NeuroML v2 XML-based community standard
  - Cell type definitions with `snnfw:` property extensions for SNNFrame-specific parameters
  - Synapse type definitions with weight and delay properties
  - Population and projection parsing from `<network>` elements
  - Uses RapidXML for high-performance XML parsing

- **HOC Format** (`.hoc`)
  - NEURON simulator scripting language parser
  - Extracts `begintemplate`/`endtemplate` blocks for cell type definitions
  - Parses `for` loops to determine instantiation counts
  - Parses `new NetCon(...)` and `connect` statements for connectivity
  - Proper depth-tracking parenthesis matching for nested expressions

- **Example Configuration Files**
  - `configs/emnist_v1_network.snnf.json` — full EMNIST experiment in native JSON
  - `configs/example_sonata/circuit_config.json` — SONATA format example
  - `configs/example_network.nml` — NeuroML format example
  - `configs/example_network.hoc` — HOC format example

- **Comprehensive Tests**
  - 32 unit tests for declarative loader system (all passing)
  - Tests for NetworkIR validation, all four parsers, NetworkConstructor, and DeclarativeLoader
  - GTest built from source (v1.14.0 via FetchContent) to resolve ABI mismatch with Conda-installed version

### Fixed

- **GTest ABI Mismatch**: Replaced Conda-installed GTest (v1.10.0) with FetchContent-built GTest v1.14.0 to fix ABI incompatibility with GCC 13, resolving test linking failures across the entire repository

## [1.0.0] - 2026-01-10

### Added

- **STDP Enable/Disable Feature**: Runtime control to enable/disable STDP learning in both `NetworkPropagator` and `SpikeProcessor`
  - `setStdpEnabled(bool)` method to control STDP learning
  - `isStdpEnabled()` method to query current STDP state
  - Early-exit guards in `applySTDP()` and `applySTDPToSynapse()` methods
  - Atomic flag for thread-safe access to STDP state
  - Prevents weight drift during inference/testing phases
  - Unit tests verifying STDP blocking when disabled

- **Core Framework**
  - Hierarchical neural organization (7-level hierarchy)
  - 6-layer canonical cortical microcircuit
  - Spike-Timing-Dependent Plasticity (STDP) with retrograde signaling
  - Multi-threaded spike processing with configurable thread pool
  - RocksDB-backed persistent storage with LRU caching
  - Temporal pattern matching and learning

- **Advanced Features**
  - Multi-column networks with orientation/frequency selectivity
  - Saccade-based attention mechanism
  - Real-time 3D visualization with OpenGL and ImGui
  - Activity recording and playback in binary format
  - Comprehensive activity monitoring with hierarchical tracking
  - Flexible encoding strategies (rate, temporal, population)
  - Edge detection operators (Gabor, Sobel, DoG)
  - Multiple classification strategies
  - Pattern learning strategies

- **Performance**
  - Multi-threaded spike delivery
  - Real-time synchronization option
  - Efficient datastore with LRU cache
  - Release builds with -O3 optimization and LTO

- **Testing & Documentation**
  - Comprehensive unit test suite (20+ test files)
  - Integration tests for network construction
  - Performance regression tests
  - Stress tests for large networks
  - 15+ working code examples
  - Complete API documentation
  - Configuration guides and quick start guides

### Performance Achievements

- **EMNIST Letters Classification**: ~90% accuracy on 26-letter classification
  - Multi-column architecture with 8 orientations × 2 frequencies
  - Cosine similarity-based pattern matching
  - STDP frozen during testing to prevent weight drift
  - Saccade-based attention mechanism
  - Training: ~60 minutes (5,200 images)
  - Testing: ~22 minutes (20,800 images)

### Supported Platforms

- Linux (Ubuntu 18.04+, Debian 10+, CentOS 7+)
- macOS (10.14+ Intel and Apple Silicon)
- Windows (10+ with MSVC 2017+ or MinGW)

### Requirements

- C++17 compatible compiler
- CMake 3.16+
- RocksDB development libraries
- OpenGL 4.5+ (for visualization)

## [Unreleased]

### Planned Features

- GPU acceleration with CUDA
- Distributed training across multiple machines
- Advanced visualization features
- Additional encoding strategies
- Performance optimizations for larger networks
- Declarative experiment runner (load config + run training/testing with no C++ coding)
- SWC morphology format parser
- PyNN format parser

