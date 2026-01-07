# Troubleshooting Guide

## Build Issues

### CMake Configuration Fails

**Error**: `Could not find RocksDB`

**Solutions**:
1. Install RocksDB development libraries:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install librocksdb-dev
   
   # macOS
   brew install rocksdb
   ```

2. Specify RocksDB path explicitly:
   ```bash
   cmake -DROCKSDB_ROOT=/path/to/rocksdb ..
   ```

### Compilation Errors

**Error**: `undefined reference to 'rocksdb::...'`

**Solutions**:
1. Ensure RocksDB is properly installed
2. Check CMakeLists.txt links RocksDB correctly
3. Try clean rebuild:
   ```bash
   rm -rf build
   mkdir build && cd build
   cmake ..
   make clean
   make -j$(nproc)
   ```

**Error**: `C++17 features not supported`

**Solutions**:
1. Update compiler:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install g++-9  # or higher
   
   # macOS
   brew install gcc@11
   ```

2. Specify compiler explicitly:
   ```bash
   cmake -DCMAKE_CXX_COMPILER=g++-11 ..
   ```

## Runtime Issues

### Segmentation Fault

**Error**: `Segmentation fault (core dumped)`

**Solutions**:
1. Run with debug symbols:
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Debug ..
   make
   gdb ./your_program
   ```

2. Check for null pointers:
   ```cpp
   if (!neuron) {
       std::cerr << "Neuron is null!" << std::endl;
       return;
   }
   ```

3. Verify object lifetime:
   - Ensure shared_ptr objects aren't deleted prematurely
   - Check for use-after-free bugs

### Memory Leaks

**Error**: Memory usage grows unbounded

**Solutions**:
1. Use valgrind to detect leaks:
   ```bash
   valgrind --leak-check=full ./your_program
   ```

2. Check for circular references in shared_ptr
3. Ensure proper cleanup in destructors
4. Verify datastore is flushed periodically

### Slow Performance

**Error**: Program runs slower than expected

**Solutions**:
1. Build in Release mode:
   ```bash
   cmake -DCMAKE_BUILD_TYPE=Release ..
   make -j$(nproc)
   ```

2. Increase thread count:
   ```cpp
   auto processor = std::make_shared<SpikeProcessor>(10000, 32);  // 32 threads
   ```

3. Profile with perf:
   ```bash
   perf record -g ./your_program
   perf report
   ```

4. Check datastore cache size:
   ```cpp
   datastore->setCacheSize(1000000);  // Increase cache
   ```

## Network Issues

### Network Not Learning

**Error**: Weights don't change during training

**Solutions**:
1. Verify STDP is enabled:
   ```cpp
   if (!propagator->isStdpEnabled()) {
       propagator->setStdpEnabled(true);
   }
   ```

2. Check STDP parameters:
   ```cpp
   propagator->setSTDPParameters(0.05, 0.05, 20.0, 20.0);
   ```

3. Verify spike injection:
   ```cpp
   for (auto& neuron : neurons) {
       neuron->injectSpike(currentTime);
   }
   ```

4. Check spike timing:
   - Pre- and post-synaptic spikes must occur within time window
   - Default window is 20ms

### Weights Drifting During Testing

**Error**: Weights change during inference

**Solutions**:
1. Disable STDP before testing:
   ```cpp
   propagator->setStdpEnabled(false);
   processor->setStdpEnabled(false);
   ```

2. Verify both components have STDP disabled
3. Check for residual spike activity

### Poor Classification Accuracy

**Error**: Network accuracy is lower than expected

**Solutions**:
1. Increase training time:
   ```cpp
   for (int epoch = 0; epoch < 100; ++epoch) {
       // Train on all samples
   }
   ```

2. Tune STDP parameters:
   ```cpp
   propagator->setSTDPParameters(0.01, 0.012, 30.0, 30.0);
   ```

3. Adjust encoding parameters:
   ```cpp
   RateEncoder encoder(100.0);  // Adjust max rate
   ```

4. Verify network architecture:
   - Check layer sizes
   - Verify connectivity patterns
   - Ensure sufficient neurons

## Visualization Issues

### Window Doesn't Open

**Error**: Visualization window fails to open

**Solutions**:
1. Verify OpenGL support:
   ```bash
   glxinfo | grep "OpenGL version"
   ```

2. Install graphics drivers:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install libgl1-mesa-glx
   ```

3. Check GLFW installation:
   ```bash
   pkg-config --modversion glfw3
   ```

### Rendering is Slow

**Error**: Visualization frame rate is low

**Solutions**:
1. Reduce visualization detail:
   ```cpp
   vizManager.setDetailLevel(LOW);
   ```

2. Disable unnecessary visualizations:
   ```cpp
   vizManager.disableActivityHeatmap();
   ```

3. Update graphics drivers
4. Use headless mode for large networks

## Data Storage Issues

### Datastore Corruption

**Error**: `RocksDB error: Corruption detected`

**Solutions**:
1. Backup data
2. Repair database:
   ```cpp
   datastore->repair();
   ```

3. Rebuild from scratch if necessary

### Disk Space Issues

**Error**: `No space left on device`

**Solutions**:
1. Clear old recordings:
   ```bash
   rm -rf recordings/*.bin
   ```

2. Reduce cache size:
   ```cpp
   datastore->setCacheSize(100000);
   ```

3. Enable compression:
   ```cpp
   datastore->enableCompression();
   ```

## Getting Help

If you encounter issues not covered here:

1. Check existing [GitHub Issues](https://github.com/deanhorak/SNNFrame/issues)
2. Review [API Reference](API_REFERENCE.md)
3. Check [Architecture Guide](ARCHITECTURE.md)
4. Review example code in `examples/`
5. Open a new issue with:
   - Error message and stack trace
   - Minimal reproducible example
   - System information (OS, compiler, dependencies)
   - Steps to reproduce

## See Also

- [Installation Guide](../INSTALLATION.md)
- [Quick Start](../QUICK_START.md)
- [API Reference](API_REFERENCE.md)

