# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-01-07

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

