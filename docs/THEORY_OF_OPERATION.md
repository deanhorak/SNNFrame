# Theory of Operation: Spike Generation and Temporal Pattern Interpretation in SNNFrame

## Overview

The SNNFrame EMNIST experiment implements a biologically-inspired visual processing pipeline that converts pixel intensities into temporal spike patterns, processes them through a 6-layer cortical microcircuit, and learns to classify letters through spike-timing-dependent plasticity (STDP). This document describes how the system works from image presentation to classification.

---

## Phase 1: Image Presentation and Spike Encoding

### **Input Encoding (SpikeEncoder::encodeAndInject)**

When a 28×28 EMNIST letter image is presented:

1. **Pixel-to-Spike Conversion** (`src/experiment/SpikeEncoder.cpp` lines 63-72):
   ```
   For each pixel (784 total):
   - Normalize intensity: norm = pixel_value / 255.0
   - If norm > threshold (0.4):
     * Fire time = baseTime + (1.0 - norm) × inputLatencyMs
     * Brighter pixels → earlier spikes
     * Darker pixels → later spikes (or no spike if below threshold)
   ```

2. **Temporal Encoding Strategy**:
   - **Latency coding**: Pixel intensity is encoded as spike timing
   - Bright pixels (norm ≈ 1.0) fire almost immediately at `baseTime`
   - Dim pixels (norm ≈ 0.4) fire ~15ms later at `baseTime + 15ms`
   - This creates a temporal wave where edges/features emerge first

3. **Spike Injection**:
   - Each input neuron fires its **temporal signature** (1-10 spikes spread over 0-100ms)
   - `fireSignature(fireT)` inserts multiple spikes at `fireT + offset[i]`
   - This creates a unique temporal pattern for each active pixel

**Example**: For the letter "A":
- Edge pixels (high contrast) → ~200 spikes firing at baseTime + 0-5ms
- Interior pixels (lower intensity) → ~50 spikes firing at baseTime + 10-15ms
- Total: ~250 input spikes with temporal structure

---

## Phase 2: Tiled Receptive Fields and L4 Competition

### **Spatial Tiling** (`configs/emnist_v1_sonata/circuit_config.json` lines 141-150)

The 28×28 input is divided into a 4×4 grid of 7×7 tiles:

```
Tile Layout:
┌─────┬─────┬─────┬─────┐
│ T0  │ T1  │ T2  │ T3  │  Each tile = 7×7 pixels (49 neurons)
├─────┼─────┼─────┼─────┤
│ T4  │ T5  │ T6  │ T7  │  Total: 16 tiles
├─────┼─────┼─────┼─────┤
│ T8  │ T9  │ T10 │ T11 │
├─────┼─────┼─────┼─────┤
│ T12 │ T13 │ T14 │ T15 │
└─────┴─────┴─────┴─────┘
```

Each of the 36 cortical columns (12 orientations × 3 frequencies) selects **3 tiles** deterministically based on its column index. This provides:
- **Spatial locality**: Each column focuses on specific regions
- **Redundancy**: Multiple columns cover each tile
- **Feature selectivity**: Different orientations/frequencies respond to different patterns

### **L4 Spike Propagation** (NetworkPropagator::fireNeuron)

When input neurons fire:

1. **Synaptic Transmission** (`src/NetworkPropagator.cpp` lines 194-240):
   ```
   For each synapse from input → L4:
   - Base delay: 1.5ms
   - Temporal signature: 1-10 spike offsets (0-100ms)
   - Arrival time = fireTime + delay + offset[i]
   - Amplitude = weight × intrinsic_excitability
   ```

2. **Dendrite Reception** (`src/Dendrite.cpp` lines 36-66):
   - Dendrite receives ActionPotential event
   - Delivers spike to target L4 neuron via `deliverSpikeToNeuron()`
   - Neuron's spike buffer accumulates: `insertSpike(arrivalTime)`

3. **Pattern Accumulation** (`src/Neuron.cpp` lines 37-48):
   ```
   L4 neuron spike buffer (500ms window):
   [baseTime+2.1, baseTime+3.7, baseTime+5.2, ..., baseTime+98.4]
   
   - Each arriving spike adds to the temporal pattern
   - Old spikes (>500ms) are automatically removed
   - Pattern represents "what the neuron has seen recently"
   ```

### **L4 Winner-Take-All Competition** (CompetitionManager::runL4Competition)

After input spikes propagate (~20ms):

1. **Activation Calculation**:
   ```
   For each L4 neuron in 7×7 grid (49 neurons per column):
   - Convert spike buffer → BinaryPattern (200-byte representation)
   - Compare to learned reference patterns using cosine similarity
   - Activation = (best_similarity × intrinsic_excitability) - inhibition
   ```

2. **Column Gating** (`maskMinActive = 6`):
   - Count active input neurons in column's tiles
   - If < 6 active inputs → column is suppressed (no L4 competition)
   - This prevents spurious activity from noise

3. **Winner Selection**:
   - Find L4 neuron with highest activation in each column
   - Winner fires its temporal signature at `baseTime + 15ms`
   - Losers receive lateral inhibition (suppressed for this image)

**Result**: ~36 L4 winners (one per active column), each firing 1-10 spikes

---

## Phase 3: L5 Integration and Pattern Learning

### **L4 → L5 Propagation** (30ms settling time)

L4 winners propagate through sparse connectivity (0.2% probability):

```
L4 neuron fires → 
  Synapses (weight=0.1, delay=1.0ms) → 
    L5 dendrites receive spikes →
      L5 neurons accumulate temporal patterns
```

### **L5 Winner-Take-All Competition**

After L4 signatures propagate:

1. **Activation**: Same as L4 (cosine similarity to learned patterns)
2. **Winner Selection**: Top 80 L5 neurons across all active columns
3. **Firing**: Winners fire at `baseTime + 15ms + column_offset`

### **Supervised Teaching Signal** (SupervisedTeacher::teach)

**This is the key learning mechanism:**

1. **L5 Pattern Learning** (`src/experiment/SupervisedTeacher.cpp` lines 42-55):
   ```
   For each L5 winner:
   - propagator->fireNeuron(l5Id, fireTime)  // Trigger synaptic propagation
   - l5Neuron->fireAndAcknowledge(fireTime)  // Send STDP acknowledgments
   - l5Neuron->learnCurrentPattern()         // Store spike pattern
   ```

2. **Output Layer Teaching** (lines 57-68):
   ```
   For the CORRECT letter class (e.g., class 0 = 'A'):
   - Force all 4 output neurons for class 'A' to fire at baseTime + 30ms
   - Each output neuron:
     * fireSignature(teachTime)           // Insert temporal signature
     * fireAndAcknowledge(teachTime)      // Trigger STDP
     * learnCurrentPattern()              // Store L5→Output pattern
   ```

**Critical Insight**: The output neurons learn to associate the **L5 activation pattern** (which represents the visual features) with the **correct letter label**.

---

## Phase 4: STDP Learning - The Retrograde Signaling Mechanism

### **How Synaptic Weights Are Updated**

When a neuron fires and acknowledges (`src/Neuron.cpp` lines 542-584):

1. **Acknowledgment Creation**:
   ```
   For each incoming spike in the temporal window:
   - Create SpikeAcknowledgment with:
     * synapseId
     * postsynaptic firing time
     * presynaptic arrival time
     * Δt = post_fire_time - pre_arrival_time
   ```

2. **Retrograde Spike Delivery** (`src/SpikeProcessor.cpp` lines 522-543):
   ```
   RetrogradeActionPotential travels back to synapse:
   - Temporal offset = Δt
   - If Δt > 0: Post fired AFTER pre → LTP (strengthen)
   - If Δt < 0: Post fired BEFORE pre → LTD (weaken)
   ```

3. **Weight Update** (`src/SpikeProcessor.cpp` lines 741-778):
   ```
   Classic STDP rule:

   If Δt ≥ 0 (LTP):
     Δw = A+ × exp(-Δt / τ+)
     Δw = 0.01 × exp(-Δt / 20ms)

   If Δt < 0 (LTD):
     Δw = -A- × exp(Δt / τ-)
     Δw = -0.012 × exp(Δt / 20ms)

   new_weight = clamp(old_weight + Δw, 0.0, 2.0)
   ```

**Example STDP Timeline**:
```
t=0ms:    Input neuron fires
t=1.5ms:  Spike arrives at L4 dendrite
t=2.0ms:  L4 neuron accumulates spike in buffer
t=15ms:   L4 neuron fires (winner)
t=16ms:   L4 sends acknowledgment back
          Δt = 16 - 1.5 = 14.5ms → LTP
          Δw = 0.01 × exp(-14.5/20) ≈ +0.0048
          Weight: 0.1 → 0.1048 (strengthened!)
```

---

## Phase 5: Pattern Storage and Recognition

### **BinaryPattern Representation** (Neuron::learnCurrentPattern)

When `learnCurrentPattern()` is called (`src/Neuron.cpp` lines 60-124):

1. **Spike Buffer → BinaryPattern Conversion**:
   ```
   Spike times: [2.1, 5.3, 12.7, 45.2, 67.8, 89.3, 102.4]

   → Divide 500ms window into 200 bins (2.5ms each)
   → Binary vector: [0,1,0,1,0,0,0,1,0,0,...,1,0,0,1,0]
   → Compact 200-byte representation
   ```

2. **Pattern Storage Strategy**:
   ```
   If < 500 patterns stored:
     - Add new pattern to library

   Else:
     - Find most similar existing pattern (cosine similarity)
     - If similarity ≥ threshold (1.2):
         Blend new pattern into existing (20% weight)
     - Else:
         Replace random pattern (novel pattern)
   ```

3. **Cosine Similarity Metric**:
   ```
   similarity = (A · B) / (||A|| × ||B||)

   Where A and B are binary spike patterns
   - Perfect match: similarity = 1.0
   - Orthogonal: similarity = 0.0
   - Threshold: 1.2 (allows some variation)
   ```

---

## Phase 6: Classification During Testing

### **Inference Without Teaching** (TrainingPipeline::runInference)

During testing (STDP disabled):

1. **Same encoding and competition** as training
2. **L5 activation vector** collected (36 columns × 80 neurons = 2,880 dimensions)
3. **k-NN Classification** (KNNClassifier::classifyKNN):
   ```
   For each stored training pattern:
     - Compute cosine similarity to current L5 activation
     - Find k=5 nearest neighbors
     - Vote: majority class wins

   If no clear winner:
     - Fall back to centroid classification
     - Compare to average L5 pattern per class
   ```

4. **Confusion Matrix** (`src/experiment/TrainingPipeline.cpp` lines 388-480):
   ```
   Rows = True label
   Cols = Predicted label

   Example output:
        A   B   C   D   E   F  ...  UNK
     A  45   2   1   0   0   0  ...   2
     B   1  43   3   1   0   0  ...   2
     C   0   2  44   1   1   0  ...   2
     ...

   Per-class accuracy:
     A: 90.00% (45/50)
     B: 86.00% (43/50)
     C: 88.00% (44/50)
   ```

---

## Summary: Complete Information Flow

```
EMNIST Image (28×28 pixels)
    ↓ [Latency encoding: bright→early, dim→late]
Input Layer (784 neurons × 1-10 spikes each)
    ↓ [Tiled receptive fields: 4×4 tiles, 3 tiles/column]
L4 Layer (36 columns × 49 neurons, 7×7 grid)
    ↓ [Winner-take-all: 1 winner/column if ≥6 inputs active]
L4 Winners (~36 neurons fire temporal signatures)
    ↓ [Sparse connectivity: 0.2% probability, 30ms settling]
L5 Layer (36 columns × 80 neurons)
    ↓ [Winner-take-all: top 80 neurons globally]
L5 Winners (80 neurons fire + learn patterns)
    ↓ [Supervised teaching: force correct output class]
Output Layer (26 classes × 4 neurons = 104 neurons)
    ↓ [STDP: strengthen L5→Output synapses for correct class]
Learned Association: L5 pattern ↔ Letter label
    ↓ [Testing: k-NN on L5 activation vectors]
Classification: ~90% accuracy on 26-class EMNIST
```

---

## Key Innovations

1. **Temporal Signatures**: Each neuron has a unique 1-10 spike pattern, creating rich temporal structure
2. **Tiled Receptive Fields**: Spatial locality + redundancy for robust feature detection
3. **Dual Competition**: L4 (local) and L5 (global) winner-take-all for hierarchical feature selection
4. **Supervised STDP**: Forcing correct outputs during training binds visual features to labels
5. **BinaryPattern Efficiency**: 200-byte compact representation enables fast similarity computation
6. **Confusion Matrix**: Detailed per-class performance analysis reveals misclassification patterns

---

## Biological Plausibility

The SNNFrame architecture incorporates several biologically-inspired principles:

### **Cortical Microcircuit**
- 6-layer organization mirrors mammalian neocortex
- L4 as primary input layer (thalamic input in biology)
- L2/3 for local integration
- L5 for output/decision making
- L6 for feedback modulation

### **Spike-Timing-Dependent Plasticity (STDP)**
- Hebbian learning: "neurons that fire together, wire together"
- Temporal causality: pre-before-post strengthens, post-before-pre weakens
- Exponential decay windows match biological measurements
- Weight bounds prevent runaway potentiation/depression

### **Temporal Coding**
- Latency coding observed in visual cortex
- Temporal signatures create unique neuron identities
- Spike patterns encode information beyond rate coding
- 500ms integration window matches cortical timescales

### **Lateral Inhibition**
- Winner-take-all competition implements biological lateral inhibition
- Sparse coding: only most active neurons fire
- Energy efficiency: reduces metabolic cost

### **Homeostatic Plasticity**
- Intrinsic excitability adjusts to maintain target firing rates
- Prevents runaway excitation or silence
- Stabilizes learning over long timescales

---

## Performance Characteristics

### **Network Scale**
- **Input**: 784 neurons (28×28 grid)
- **Cortical**: 10,256 neurons (36 columns × 6 layers)
- **Output**: 104 neurons (26 classes × 4 neurons)
- **Total**: 11,144 neurons
- **Synapses**: ~109,000 connections

### **Computational Efficiency**
- **Spike delivery**: ~15% optimized through thread pool and batching
- **Pattern matching**: O(1) spike insertion with cached max time
- **Memory**: 200-byte BinaryPattern vs. variable-length spike lists
- **Throughput**: ~38ms per image (including 30ms settling time)

### **Learning Performance**
- **Training**: 2,600 images (100 per class × 26 classes)
- **Testing**: 1,000-4,000 images
- **Accuracy**: ~90% on 26-class EMNIST letters
- **Convergence**: 3-5 training passes typical
- **Patterns stored**: ~500 per neuron (adaptive replacement)

---

## Configuration Parameters

Key parameters from `configs/emnist_v1_sonata/circuit_config.json`:

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `window_size_ms` | 500.0 | Temporal integration window |
| `similarity_threshold` | 1.2 | Pattern recognition threshold |
| `max_reference_patterns` | 500 | Pattern library size per neuron |
| `similarity_metric` | cosine | Pattern comparison method |
| `input_latency_ms` | 15.0 | Max spike latency for encoding |
| `pixel_threshold` | 0.4 | Min intensity to generate spike |
| `tiles_per_side` | 4 | Spatial tiling grid size |
| `tiles_per_column` | 3 | Tiles per cortical column |
| `maskMinActive` | 6 | Min inputs for column activation |
| `stdp_a_plus` | 0.01 | LTP learning rate |
| `stdp_a_minus` | 0.012 | LTD learning rate |
| `stdp_tau_plus` | 20.0 | LTP time constant (ms) |
| `stdp_tau_minus` | 20.0 | LTD time constant (ms) |

---

## Future Directions

### **Potential Enhancements**
1. **Unsupervised Learning**: Remove supervised teaching, rely on self-organization
2. **Recurrent Connections**: Add L6→L4 feedback for context modulation
3. **Attention Mechanisms**: Dynamic tile selection based on saliency
4. **Multi-Scale Processing**: Multiple frequency bands for different features
5. **Online Learning**: Continuous adaptation without discrete training phases
6. **Neuromorphic Hardware**: Deploy on SpiNNaker or Loihi for real-time processing

### **Research Questions**
- Can the network learn orientation selectivity without pre-defined columns?
- How does performance scale with more classes (full alphabet + digits)?
- What is the minimal network size for 90% accuracy?
- Can transfer learning work across different datasets?
- How robust is the system to noise and occlusions?

---

## References

### **Key Concepts**
- **STDP**: Bi & Poo (1998), "Synaptic modifications in cultured hippocampal neurons"
- **Cortical Microcircuit**: Douglas & Martin (2004), "Neuronal circuits of the neocortex"
- **Temporal Coding**: Thorpe et al. (1996), "Spike-based strategies for rapid processing"
- **Lateral Inhibition**: Yuille & Grzywacz (1989), "A winner-take-all mechanism"

### **Implementation Details**
- See `docs/DEVELOPER_MANUAL.md` for API usage
- See `docs/API_REFERENCE.md` for class documentation
- See `docs/STDP_GUIDE.md` for STDP configuration
- See `docs/FORMAT_REFERENCE.md` for SONATA model specification

---

**Document Version**: 1.0
**Last Updated**: 2026-02-14
**Author**: SNNFrame Development Team

