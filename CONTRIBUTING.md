# Contributing to SNNFrame

Thank you for your interest in contributing to SNNFrame! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- **Clear title and description**
- **Steps to reproduce** the issue
- **Expected vs actual behavior**
- **Environment details** (OS, compiler version, dependencies)
- **Code samples** or test cases if applicable
- **Error messages** and stack traces

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, include:

- **Clear title and description**
- **Use case** and motivation
- **Proposed solution** or API design
- **Alternatives considered**
- **Impact** on existing functionality

### Pull Requests

1. **Fork the repository** and create your branch from `master`
2. **Follow the coding style** (see below)
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Ensure all tests pass** (`ctest --output-on-failure`)
6. **Write clear commit messages**
7. **Submit the pull request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/SNNFrame.git
cd SNNFrame

# Add upstream remote
git remote add upstream https://github.com/deanhorak/SNNFrame.git

# Create a feature branch
git checkout -b feature/my-new-feature

# Build and test
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make -j$(nproc)
ctest --output-on-failure
```

## Coding Style

### C++ Guidelines

- **C++17 standard** - use modern C++ features appropriately
- **Header guards** - use `#ifndef SNNFW_CLASSNAME_H` format
- **Naming conventions**:
  - Classes: `PascalCase` (e.g., `NetworkBuilder`)
  - Functions/methods: `camelCase` (e.g., `createNeuron`)
  - Member variables: `camelCase_` with trailing underscore (e.g., `neuronCount_`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_NEURONS`)
- **Indentation**: 4 spaces (no tabs)
- **Line length**: Prefer 100 characters max
- **Comments**: Use Doxygen-style comments for public APIs

### Example

```cpp
/**
 * @brief Creates a new neuron with specified parameters
 * @param windowSizeMs Temporal window size in milliseconds
 * @param threshold Similarity threshold for pattern matching
 * @return Shared pointer to the created neuron
 */
std::shared_ptr<Neuron> createNeuron(double windowSizeMs, double threshold);
```

## Testing

### Writing Tests

- Place tests in `tests/` directory
- Use Google Test framework
- Name test files `test_<component>.cpp`
- Group related tests using `TEST()` macros
- Test both success and failure cases

### Running Tests

```bash
cd build
ctest --output-on-failure  # Run all tests
./tests/test_neuron        # Run specific test
```

## Documentation

- Update relevant `.md` files in the root and `docs/` directory
- Add code examples for new features
- Update API documentation in header files
- Keep documentation concise and accurate

## Commit Messages

Follow conventional commit format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Example:
```
feat(stdp): add runtime enable/disable for STDP learning

- Add setStdpEnabled() and isStdpEnabled() methods to NetworkPropagator
- Add setStdpEnabled() and isStdpEnabled() methods to SpikeProcessor
- Add early exit guards in applySTDP() methods
- Add unit tests for STDP enable/disable functionality

Closes #123
```

## Review Process

1. Maintainers will review your PR within a few days
2. Address any requested changes
3. Once approved, a maintainer will merge your PR
4. Your contribution will be included in the next release

## Questions?

- Open an issue for questions
- Check existing documentation
- Review example code in `examples/` and `experiments/`

Thank you for contributing to SNNFrame! 🧠⚡

