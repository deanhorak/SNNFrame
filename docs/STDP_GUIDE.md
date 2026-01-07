# STDP Learning Guide

## Overview

Spike-Timing-Dependent Plasticity (STDP) is a learning mechanism where synaptic weights are modified based on the relative timing of pre- and post-synaptic spikes.

## STDP Learning Rule

### Mathematical Formulation

For a synapse connecting pre-synaptic neuron to post-synaptic neuron:

**Δt = t_post - t_pre** (time difference in milliseconds)

**If Δt > 0** (pre-spike before post-spike):
```
Δw = A+ × exp(-Δt / τ+)  [LTP - strengthen]
```

**If Δt < 0** (post-spike before pre-spike):
```
Δw = -A- × exp(Δt / τ-)  [LTD - weaken]
```

Where:
- **A+**: LTP amplitude (default: 0.01)
- **A-**: LTD amplitude (default: 0.012)
- **τ+**: LTP time constant in ms (default: 20.0)
- **τ-**: LTD time constant in ms (default: 20.0)

### Weight Bounds

Weights are clamped to [0, 2] range to prevent runaway growth or decay.

## Implementation in SNNFrame

### Two STDP Pathways

SNNFrame implements STDP in two components:

1. **NetworkPropagator**: Handles retrograde signaling and acknowledgment-based STDP
2. **SpikeProcessor**: Handles retrograde spike delivery and temporal offset-based STDP

Both pathways respect the STDP enable/disable flag.

### Retrograde Signaling

```
1. Pre-synaptic spike arrives at synapse
2. Spike propagates to post-synaptic neuron
3. Post-synaptic neuron fires
4. Retrograde signal travels back to synapse
5. STDP applied based on temporal offset
```

## Configuration

### Setting STDP Parameters

```cpp
auto propagator = std::make_shared<NetworkPropagator>(spikeProcessor);
auto processor = spikeProcessor;

// Configure STDP parameters
propagator->setSTDPParameters(0.05, 0.05, 20.0, 20.0);  // A+, A-, τ+, τ-
processor->setSTDPParameters(0.05, 0.05, 20.0, 20.0);
```

### Enabling/Disabling STDP

```cpp
// Training mode: STDP enabled (default)
propagator->setStdpEnabled(true);
processor->setStdpEnabled(true);

// Inference mode: STDP disabled to prevent weight drift
propagator->setStdpEnabled(false);
processor->setStdpEnabled(false);

// Query STDP state
bool enabled = propagator->isStdpEnabled();
```

## Training vs Inference

### Training Phase

```cpp
// Enable STDP for learning
networkPropagator->setStdpEnabled(true);
spikeProcessor->setStdpEnabled(true);

// Train on data
for (const auto& sample : trainingData) {
    // Inject spikes
    // Process spikes
    // Weights are updated via STDP
}
```

### Inference Phase

```cpp
// Disable STDP to prevent weight drift
networkPropagator->setStdpEnabled(false);
spikeProcessor->setStdpEnabled(false);

// Test on data
for (const auto& sample : testData) {
    // Inject spikes
    // Process spikes
    // Weights remain frozen
}
```

## Best Practices

### 1. Always Disable STDP During Testing

Weight drift during testing can cause inconsistent results:

```cpp
// Before testing
networkPropagator->setStdpEnabled(false);
spikeProcessor->setStdpEnabled(false);

// Run tests
// ...

// After testing, re-enable for next training pass
networkPropagator->setStdpEnabled(true);
spikeProcessor->setStdpEnabled(true);
```

### 2. Tune STDP Parameters

Different tasks may require different STDP parameters:

```cpp
// Conservative learning (slower)
propagator->setSTDPParameters(0.01, 0.012, 20.0, 20.0);

// Aggressive learning (faster)
propagator->setSTDPParameters(0.05, 0.06, 20.0, 20.0);

// Longer time window
propagator->setSTDPParameters(0.01, 0.012, 50.0, 50.0);
```

### 3. Monitor Weight Changes

Track weight statistics during training:

```cpp
// Before training
double initialWeight = synapse->getWeight();

// After training
double finalWeight = synapse->getWeight();
double weightChange = finalWeight - initialWeight;
```

### 4. Use Trace-Based STDP for Complex Patterns

For temporal pattern learning, enable trace-based STDP:

```cpp
propagator->setTraceStdpEnabled(true);
propagator->setStdpLtdWindowMs(20.0);  // LTD time window
```

## Troubleshooting

### Weights Not Changing

**Problem**: Weights remain constant during training

**Solutions**:
1. Verify STDP is enabled: `propagator->isStdpEnabled()`
2. Check STDP parameters are non-zero
3. Ensure pre- and post-synaptic spikes occur within time window
4. Verify synapses are registered with propagator

### Weights Drifting During Testing

**Problem**: Weights change during inference

**Solutions**:
1. Disable STDP before testing: `setStdpEnabled(false)`
2. Verify both NetworkPropagator and SpikeProcessor have STDP disabled
3. Check for any remaining spike activity

### Unstable Learning

**Problem**: Weights oscillate or diverge

**Solutions**:
1. Reduce learning rates (A+, A-)
2. Increase time constants (τ+, τ-)
3. Check weight bounds are appropriate
4. Verify input spike patterns are reasonable

## References

- Bi, G. Q., & Poo, M. M. (1998). "Synaptic modifications in cultured hippocampal neurons: dependence on spike timing, synaptic strength, and postsynaptic cell type." Journal of Neuroscience, 18(24), 10464-10472.

- Markram, H., Lübke, J., Frotscher, M., & Sakmann, B. (1997). "Regulation of synaptic efficacy by coincidence of postsynaptic APs and EPSCs." Science, 275(5297), 213-215.

## See Also

- [API Reference](API_REFERENCE.md)
- [Architecture Guide](ARCHITECTURE.md)
- [Examples](../examples/)

