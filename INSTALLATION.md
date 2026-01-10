# SNNFrame Installation Guide

## System Requirements

### Minimum
- C++17 compatible compiler
- CMake 3.16+
- 4GB RAM
- 2GB disk space

### Recommended
- GCC 9+ or Clang 10+
- CMake 3.20+
- 8GB+ RAM
- 10GB disk space (for datasets)
- NVIDIA GPU with CUDA support (optional, for future GPU acceleration)

## Supported Platforms

- **Linux**: Ubuntu 18.04+, Debian 10+, CentOS 7+
- **macOS**: 10.14+ (Intel and Apple Silicon)
- **Windows**: Windows 10+ with MSVC 2017+ or MinGW

## Installation Steps

### 1. Install System Dependencies

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    librocksdb-dev \
    libglfw3-dev \
    libglew-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libxext-dev \
    libxkbcommon-dev
```

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake rocksdb glfw3 glew
```

#### CentOS/RHEL
```bash
sudo yum groupinstall -y "Development Tools"
sudo yum install -y \
    cmake \
    rocksdb-devel \
    glfw-devel \
    glew-devel \
    libXrandr-devel \
    libXinerama-devel \
    libXcursor-devel \
    libXi-devel
```

### 2. Clone the Repository

```bash
git clone https://github.com/deanhorak/SNNFrame.git
cd SNNFrame
```

### 3. Build the Framework

```bash
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build (using all available cores)
make -j$(nproc)

# Optional: Install to system
sudo make install
```

### 4. Verify Installation

```bash
# Run tests
ctest --output-on-failure

# Try running an example
./emnist_letters_training --help
```

### 5. Run the High-Performance Example

The framework includes a high-performance EMNIST letters classification experiment achieving ~90% accuracy:

```bash
cd build

# Headless training (recommended for first run)
./emnist_letters_training

# With real-time visualization
./emnist_letters_visualized
```

**Expected Results**:
- Training time: ~60 minutes on full dataset
- Testing time: ~22 minutes on full test set
- Accuracy: ~90% on 26-letter classification
- Network: Multi-column architecture with 8 orientations × 2 frequencies

## Build Options

### Release Build (Default)
```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Debug Build
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)
```

### Custom Installation Path
```bash
cmake -DCMAKE_INSTALL_PREFIX=/custom/path ..
make -j$(nproc)
sudo make install
```

## Troubleshooting

### RocksDB Not Found
```bash
# Ubuntu/Debian
sudo apt-get install librocksdb-dev

# macOS
brew install rocksdb

# Or build from source
git clone https://github.com/facebook/rocksdb.git
cd rocksdb
make shared_lib
sudo make install
```

### GLFW/GLAD Not Found
The framework includes GLFW, GLAD, and GLM in `third_party/`. If CMake can't find them:

```bash
# Ensure third_party directory exists
ls -la third_party/

# If missing, copy from original repository
cp -r /path/to/original/snnfw/third_party/* third_party/
```

### Compilation Errors

#### C++17 Not Supported
Update your compiler:
```bash
# Ubuntu/Debian
sudo apt-get install g++-9

# macOS
brew install gcc@11
```

#### Missing OpenGL Headers
```bash
# Ubuntu/Debian
sudo apt-get install libgl1-mesa-dev

# macOS
# Usually included with Xcode Command Line Tools
xcode-select --install
```

### Runtime Issues

#### Library Path Issues
```bash
# Set library path before running
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
./emnist_letters_training
```

#### GLIBCXX Version Mismatch
```bash
# Use system libraries instead of Anaconda
unset LD_LIBRARY_PATH
./emnist_letters_training
```

## Docker Installation (Optional)

Create a `Dockerfile`:

```dockerfile
FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    build-essential cmake git \
    librocksdb-dev libglfw3-dev libglew-dev

WORKDIR /app
COPY . .

RUN mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc)

ENTRYPOINT ["./build/emnist_letters_training"]
```

Build and run:
```bash
docker build -t snnframe .
docker run --rm snnframe
```

## Verifying the Installation

### Quick Test
```bash
cd build
./emnist_letters_training --help
```

### Full Test Suite
```bash
cd build
ctest --output-on-failure -V
```

### Example Run
```bash
cd build
# This will train and test on EMNIST letters
./emnist_letters_training
```

## Next Steps

1. Read `README.md` for an overview
2. Check `docs/QUICK_START.md` for your first program
3. Review `examples/` for code samples
4. Explore `experiments/` for advanced usage

## Getting Help

- Check `docs/` for detailed documentation
- Review `examples/` for code samples
- Open an issue on GitHub
- Check existing issues for solutions

---

**Last Updated**: 2025-12-12
