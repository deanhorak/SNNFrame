# SNNFrame: Theory of Operation

**Version**: 2.0
**Branch**: SimilarityRefinement
**Date**: 2026-02-22

---

## Table of Contents

1. [System Purpose](#1-system-purpose)
2. [Internal Framework Structure](#2-internal-framework-structure)
3. [Network Architecture](#3-network-architecture)
4. [Complete Information Flow](#4-complete-information-flow)
5. [Spike Encoding Algorithm](#5-spike-encoding-algorithm)
6. [Spike Propagation Engine](#6-spike-propagation-engine)
7. [L4 Competition Algorithm](#7-l4-competition-algorithm)
8. [L5 Competition Algorithm](#8-l5-competition-algorithm)
9. [STDP Learning — All Variants](#9-stdp-learning--all-variants)
10. [Pattern Learning and BinaryPattern](#10-pattern-learning-and-binarypattern)
11. [Classification System](#11-classification-system)
12. [Training vs Inference Modes](#12-training-vs-inference-modes)
13. [Homeostatic Plasticity](#13-homeostatic-plasticity)
14. [Configuration Reference](#14-configuration-reference)
15. [Current State Assessment](#15-current-state-assessment)
16. [Recommended Next Steps](#16-recommended-next-steps)

---

## 1. System Purpose

SNNFrame is a biologically-inspired Spiking Neural Network (SNN) framework for image classification, specifically targeting the EMNIST handwritten letters dataset (26 classes, A–Z). Unlike rate-coded artificial neural networks, SNNFrame encodes information in the **timing** of spikes, uses **spike-timing-dependent plasticity (STDP)** for unsupervised synaptic learning, and organizes computation into cortical-column structures that mirror mammalian visual cortex.

The primary experiment (`emnist_sonata_training`) loads a network from a SONATA circuit description (`configs/emnist_v1_sonata/circuit_config.json`), trains on up to 200 EMNIST images per class per pass for up to 20 passes, then classifies test images using a k-NN readout over learned L5 population vectors. Reported accuracy targets ~90% on 26-class EMNIST letters.

---

## 2. Internal Framework Structure

### 2.1 Namespace and File Organization

```
snnfw/                         Core SNN framework
  Neuron                       Neural unit with spike buffer and pattern library
  Axon                         Carries spikes from one neuron outward
  Dendrite                     Receives spikes and delivers to a target neuron
  Synapse                      Weighted connection with delay (Axon→Dendrite)
  SpikeProcessor               Event queue, scheduler, STDP engine, thread pool
  NetworkPropagator            Fires neurons, routes spikes, applies STDP
  BinaryPattern                Compact 200-bin temporal spike representation
  NetworkPropagator            Fires neurons, routes spikes, applies STDP

snnfw/experiment/              High-level experiment framework
  ExperimentConfig             All hyperparameters in one struct
  ExperimentRunner             Top-level: loads network, runs TrainingPipeline
  SpikeEncoder                 Encodes EMNIST pixels → input spikes
  CompetitionManager           L4 and L5 winner-take-all logic
  SupervisedTeacher            Forces correct output during training
  KNNClassifier                k-NN readout over L5 activation vectors
  TrainingPipeline             Main training/testing loop

snnfw/declarative/             Network construction from config files
  SONATAParser                 Parses SONATA circuit_config.json
  NetworkConstructor           Builds Neuron/Axon/Dendrite/Synapse objects
  NetworkIR                    Intermediate representation of network topology

experiments/                   Entry points
  emnist_sonata_training.cpp   Primary experiment: declarative SONATA model
  emnist_letters_training.cpp  Legacy: programmatic network construction
```

### 2.2 Key Data Structures

| Structure | Purpose |
|-----------|---------|
| `Neuron` | Stores spike buffer (rolling 500 ms window), pattern library (up to 500 BinaryPatterns), intrinsic excitability, inhibition level, temporal signature |
| `BinaryPattern` | 200-element `uint8_t` array representing spike presence in time bins; compact enough for fast cosine similarity |
| `ActionPotential` | Event in the spike queue: `{synapseId, arrivalTimeMs, amplitude}` |
| `RetrogradeActionPotential` | Backward event for STDP: `{synapseId, postFireTime, preArrivalTime}` |
| `SpikeAcknowledgment` | Created by `Neuron::fireAndAcknowledge`; pairs pre-spike arrival time with post-fire time |
| `ConstructedNetwork` | Built by `NetworkConstructor`: holds `inputNeurons`, `columns` (with layered neurons), `outputPopulations`, `propagator`, `spikeProcessor` |
| `ExperimentConfig` | ~60 hyperparameters controlling every aspect of the pipeline |

---

## 3. Network Architecture

### 3.1 Layers and Populations

The SONATA config builds a network with the following populations:

```
Input Layer      784 neurons (28×28 pixel grid)
                 Each neuron fires when pixel intensity > threshold (0.4)

L4 Layer         numColumns × (layer4Size²) neurons
                 Default: 36 columns × 49 neurons = 1,764 neurons
                 Role: local feature detectors, one column per spatial+orientation tile

L5 Layer         numColumns × layer5Neurons neurons
                 Default: 36 columns × 80 neurons = 2,880 neurons
                 Role: higher-order integrators, global competition, classification vector

Output Layer     numClasses × neuronsPerOutputClass neurons
                 Default: 26 × 3 = 78 neurons
                 Role: class-specific population; forced to fire during supervised training
```

### 3.2 Cortical Columns

Each column has a **receptive field** — a subset of input tiles. The 28×28 image is divided into a 4×4 grid of 7×7 tiles (16 tiles total). Each column is assigned `tilesPerColumn` (default 3) tiles. Columns also have **orientation** and **spatial frequency** attributes from the SONATA config. These are used later for inter-column inhibition feature gating.

### 3.3 Connectivity

| Connection | Probability | Initial Weight | Delay |
|------------|-------------|---------------|-------|
| Input → L4 | Tiled RF (all active pixels in column's tiles) | 0.1 | 1.5 ms |
| L4 → L5    | 0.2% random sparse | 0.1 | 1.0 ms |
| L5 → Output | 2% random sparse | 0.02 | 1.0 ms |

Weights are bounded: `[0.0, 2.0]` via STDP clamping. All connections are **one-directional**; retrograde signaling for STDP travels on a separate `RetrogradeActionPotential` channel.

---

## 4. Complete Information Flow

```
┌──────────────────────────────────────────────────────────────┐
│  EMNIST Image (28×28, uint8 pixels, label 1–26)              │
└───────────────────────┬──────────────────────────────────────┘
                        │ SpikeEncoder::encodeAndInject()
                        ▼
┌──────────────────────────────────────────────────────────────┐
│  Input Layer (784 neurons)                                   │
│  Latency coding: norm = pixel/255; fireTime = base +         │
│                  (1 - norm) × inputLatencyMs                 │
│  Each firing neuron emits its temporal signature (1–10 spks) │
└───────────────────────┬──────────────────────────────────────┘
                        │ SpikeProcessor delivers ActionPotentials
                        │ via 24 threads; Input→L4 synapses
                        ▼
┌──────────────────────────────────────────────────────────────┐
│  L4 Layer (36 columns × 49 neurons)                          │
│  Each L4 neuron accumulates spikes in a 500 ms buffer        │
│  STDP runs on Input→L4 synapses continuously                 │
└───────────────────────┬──────────────────────────────────────┘
                        │ waitForSimTime(base + inputLatencyMs + 5 ms)
                        ▼
┌──────────────────────────────────────────────────────────────┐
│  L4 Competition (CompetitionManager::runL4Competition)       │
│  Per column: hybrid score = spike_count + similarity × scale │
│  Top l4Keep (8) winners fire temporal signatures             │
│  Losers receive inhibition                                   │
│  Columns with < maskMinActive (6) active inputs suppressed   │
└───────────────────────┬──────────────────────────────────────┘
                        │ waitForSimTime(base + kL4ToL5SettleMs ~30 ms)
                        │ L4→L5 synapses carry L4 signatures
                        ▼
┌──────────────────────────────────────────────────────────────┐
│  L5 Layer (36 columns × 80 neurons)                          │
│  Receives L4 winner spikes; accumulates in 500 ms buffer     │
│  STDP runs on L4→L5 synapses                                 │
└───────────────────────┬──────────────────────────────────────┘
                        │ CompetitionManager::runL5Competition
                        ▼
┌──────────────────────────────────────────────────────────────┐
│  L5 Competition                                              │
│  Intra-column: top l5Keep (8) winners per column             │
│  Inter-column inhibition based on RF overlap + feature match │
│  Global winner flags: vector<bool> l5WinnerGlobal            │
└───────────────────────┬──────────────────────────────────────┘
              ┌─────────┴─────────┐
    Training  │                   │  Inference
              ▼                   ▼
┌─────────────────────┐  ┌───────────────────────────────────┐
│ SupervisedTeacher   │  │  collectL5Readout()               │
│ L5 winners fire +   │  │  L5CountVector + L5LatencyVector  │
│ learnCurrentPattern │  │  applyL5DivisiveNormalization()   │
│ Correct output pop  │  └──────────────┬────────────────────┘
│ forced to fire at   │                 │
│ base + 30 ms        │                 ▼
│ STDP on L5→Output   │  ┌───────────────────────────────────┐
└─────────────────────┘  │  KNNClassifier::classifyKNN()     │
                         │  Weighted cosine sim (IDF opt.)   │
                         │  k=7 nearest neighbors vote       │
                         │  Fallback: classifyCentroid()     │
                         │  Optional: pairwise disambiguation │
                         └──────────────┬────────────────────┘
                                        ▼
                              Predicted class (0–25)
                              Confusion matrix update
```

---

## 5. Spike Encoding Algorithm

**File**: `src/experiment/SpikeEncoder.cpp`

### 5.1 Latency Coding

```
norm      = pixel_value / 255.0
threshold = config.pixelThreshold  (default 0.4)

if norm > threshold:
    fireTime = baseTime + (1.0 - norm) × inputLatencyMs
    inputNeurons[idx]->fireSignature(fireTime)
    propagator->fireNeuron(inputNeurons[idx]->getId(), fireTime)
```

- Bright pixel (norm=1.0): fires at `baseTime + 0 ms` — first to arrive at L4
- Dim pixel (norm=0.41): fires at `baseTime + 14.85 ms` — last to arrive
- Pixels below threshold: **silent** (no spike generated)
- Typical EMNIST letter: ~200–450 active pixels out of 784

### 5.2 Temporal Signature

`fireSignature(fireT)` does not fire a single spike. Each neuron has a stored **temporal signature** — a list of time offsets (1–10 values, each 0–100 ms). The neuron inserts spikes at `fireT + offset[i]` into its own spike buffer, and `fireNeuron()` schedules those as ActionPotentials downstream through all outgoing synapses.

This creates rich temporal structure: a single "firing event" for one pixel produces a burst of 1–10 spikes spread across up to 100 ms.

### 5.3 Schedule Horizon and Inter-Image Gap

The encoder enforces a 550 ms gap between image presentations (`interImageGapMs`). It blocks until the SpikeProcessor's current simulation time is close enough to the next image's scheduled start, preventing schedule-ahead overflow in the 10,000-slot time-slice buffer.

### 5.4 Memory Cleanup

Before each image, `periodicMemoryCleanup(baseTime)` is called on every neuron to evict spike records older than the integration window, and `removeSpikesBefore(baseTime)` purges output neuron buffers to prevent leakage from prior images.

---

## 6. Spike Propagation Engine

**Files**: `src/SpikeProcessor.cpp`, `src/NetworkPropagator.cpp`

### 6.1 Time-Sliced Event Queue

`SpikeProcessor` maintains a circular array of **10,000 time slices**, each representing 1 ms of simulation time. Spikes are scheduled into future slices. A background thread advances the current time slice, calling `deliverSliceAsync()` for each 1 ms window.

```
scheduleSpike(synapseId, arrivalTimeMs, amplitude):
    slotIndex = (int)arrivalTimeMs % 10000
    eventQueue[slotIndex].push(ActionPotential{...})

deliverSliceAsync(slice):
    for each event in slice:
        if event is ActionPotential:
            dendrite->receiveSpike(event)   // → neuron spike buffer
        if event is RetrogradeActionPotential:
            applySTDPToSynapse(event)
```

### 6.2 Thread Pool

Spike delivery uses a **24-thread pool** (`numThreads`). Each time slice's events are distributed across threads for parallel delivery. Neuron spike buffers use mutex locking for thread safety.

### 6.3 fireNeuron() — The Core Propagation Method

**File**: `src/NetworkPropagator.cpp`, method `fireNeuron(neuronId, fireTime)`

```
1. Retrieve neuron's temporal signature (list of time offsets)
2. For each outgoing Axon of the neuron:
   For each Synapse on that Axon:
     For each offset in temporal signature:
       arrivalTime = fireTime + synapse.delay + offset
       scheduleSpike(synapse.id, arrivalTime, synapse.weight)
3. If not trace-STDP mode:
   Schedule RetrogradeActionPotential for each recent pre-spike
   (enables classic STDP on presynaptic synapses)
```

### 6.4 deliverSpikeToNeuron()

```
1. neuron->insertSpike(arrivalTime)       // Add to rolling 500 ms buffer
2. Record incoming spike for STDP eligibility tracking
3. If trace-STDP enabled: update pre-synaptic trace
```

### 6.5 Event Type Optimization

Event type discrimination uses first-character comparison (optimized from `strcmp`):
```cpp
if (eventType[0] == 'A')  // ActionPotential
if (eventType[0] == 'R')  // RetrogradeActionPotential
```

---

## 7. L4 Competition Algorithm

**File**: `src/experiment/CompetitionManager.cpp`, method `runL4Competition()`

### 7.1 Column Gating

For each column, count how many of the column's assigned input neurons fired (`lastInputFired[]` from encoder). If fewer than `maskMinActive` (6) fired, the column is suppressed — no L4 competition runs, `colHasL4[col] = false`.

### 7.2 Hybrid Scoring

For each L4 neuron in an active column:

```
spikeCount  = neuron->getSpikeCount(baseTime, window)
similarity  = neuron->getBestSimilarity()  // vs. learned BinaryPatterns

score = (1 - l4SimilarityWeight) × spikeCount
      + l4SimilarityWeight × similarity × maxSpikeCount
```

Default `l4SimilarityWeight = 0.25`, so it's 75% spike-count driven, 25% pattern-match driven.

### 7.3 Winner Selection and Firing

- Sort neurons by score; top `l4Keep` (8) win
- Winners fire at a position-dependent time:
  ```
  fireTime = baseTime + l4PostShiftMs + row × l4RowDelay + col × l4ColDelay + jitter
  ```
  Default: `baseTime - 6 ms + row×0.3 + col×0.2 + uniform(-1, +1) ms`
- Winners call `propagator->fireNeuron()` — scheduling L4→L5 spikes
- Losers receive `l5InhibitLoser` (1.2) inhibition units

### 7.4 Spatial Delay Rationale

The row/column delays create a mild spatial sweep: neurons near the top-left of the L4 grid fire slightly earlier than bottom-right neurons. This preserves a spatial ordering in the temporal domain, potentially giving L5 neurons timing cues about spatial structure.

---

## 8. L5 Competition Algorithm

**File**: `src/experiment/CompetitionManager.cpp`, method `runL5Competition()`

### 8.1 Intra-Column Competition

Same hybrid scoring as L4 but with `l5SimilarityWeight = 0.60` (majority pattern-match driven). Top `l5Keep` (8) winners per column advance.

### 8.2 Inter-Column Inhibition

After intra-column competition, columns inhibit each other based on three criteria:

```
For each pair of columns (A, B) where A won:
  rfOverlap = |tilesA ∩ tilesB| / |tilesA ∪ tilesB|
  orientDelta = |orientationA - orientationB|  (degrees)
  freqDelta = |log2(freqA / freqB)|  (octaves)

  if rfOverlap >= l5InterColumnMinOverlap (0.30)
  and orientDelta <= l5InterColumnMaxOrientationDeltaDeg (90°)
  and freqDelta <= l5InterColumnMaxFrequencyOctaveDelta (2.0):
      inhibit = l5InterColumnInhibit × rfOverlap × winnerScore / l5InterColumnWinnerScale
      inhibit = min(inhibit, l5InterColumnMaxInhibit)
      apply to losers in column B
```

This prevents two nearby columns with similar feature tuning from both fully activating — enforcing sparser, more discriminative L5 representations.

### 8.3 Global Winner Flags

After all column competitions, the result is a flat `vector<bool> l5WinnerGlobal` of size `numColumns × layer5Neurons`. This vector is the **L5 population code** — the read-out used for both supervised teaching and k-NN classification.

---

## 9. STDP Learning — All Variants

**Files**: `src/SpikeProcessor.cpp` (applySTDPToSynapse), `src/NetworkPropagator.cpp` (applySTDP, trace methods)

### 9.1 Classic STDP

Triggered by `RetrogradeActionPotential`. The post-synaptic neuron fires and sends a retrograde signal back to each pre-synaptic synapse with the time difference `Δt = postFireTime - preArrivalTime`.

```
If Δt ≥ 0 (pre before post → causal → LTP):
    Δw = stdpAPlus × exp(-Δt / stdpTauPlus)
    default: 0.01 × exp(-Δt / 20 ms)

If Δt < 0 (post before pre → acausal → LTD):
    Δw = -stdpAMinus × ltdScale × exp(Δt / stdpTauMinus)
    default: -0.012 × 0.3 × exp(Δt / 20 ms)

new_weight = clamp(old_weight + Δw, 0.0, 2.0)
```

LTD is deliberately scaled down (`stdpLtdScale = 0.3`) to bias toward potentiation and prevent runaway depression.

### 9.2 Trace-Based STDP

When `traceStdp = true` (default), each synapse maintains a **pre-synaptic eligibility trace** that decays exponentially. When a post-synaptic neuron fires, the trace value at each synapse determines the weight update magnitude rather than reconstructing exact timing differences. This is more biologically plausible and computationally efficient.

### 9.3 STDP Eligibility Gate

Before a neuron's pattern is stored (via `learnCurrentPattern()`), its STDP history is checked:

```
stats = propagator->getNeuronStdpEligibility(neuronId)

Gate passes if:
  stats.totalUpdates >= stdpEligibilityMinUpdates  (default 1)
  stats.ltpUpdates   >= stdpEligibilityMinLtp      (default 0)
  stats.score(ltdPenalty) >= stdpEligibilityThreshold  (default -0.002)

where score = ltpUpdates - ltdPenalty × ltdUpdates
```

Neurons that haven't received meaningful STDP updates (never responded to input) are blocked from storing patterns. This prevents the pattern library from filling with noise.

### 9.4 STDP Statistics Tracking

Per-synapse-group stats are maintained (InputToL4, L4ToL5, L5ToOutput):
- `totalUpdates`, `ltpUpdates`, `ltdUpdates`
- `weightSum`, `weightMin`, `weightMax`
- Reset via `resetStdpUpdateStats()` at each training pass start

---

## 10. Pattern Learning and BinaryPattern

**Files**: `src/BinaryPattern.cpp`, `src/Neuron.cpp`

### 10.1 BinaryPattern Construction

`BinaryPattern(spikeTimes, windowMs, numBins=200)` converts a spike time list into a fixed-length binary vector:

```
binWidth = windowMs / numBins   (default: 500 ms / 200 = 2.5 ms per bin)

For each spike time t:
  bin = (int)((t - windowStart) / binWidth)
  pattern[bin] = 1

Result: uint8_t[200] — 200 bytes representing 500 ms of activity
```

The window is **anchored to the trailing edge** of the current time, so recent spikes map to the highest bins. This preserves latency information across the temporal axis.

### 10.2 Similarity Metrics

BinaryPattern supports five similarity metrics (configured via `similarity_metric` in SONATA config):

| Metric | Formula | Notes |
|--------|---------|-------|
| `cosine` (default) | `(A·B) / (‖A‖ × ‖B‖)` | Angle between spike vectors |
| `histogram_intersection` | `Σ min(aᵢ,bᵢ) / Σ max(aᵢ,bᵢ)` | Jaccard-like |
| `euclidean` | `1 / (1 + ‖A-B‖)` | Distance-based |
| `correlation` | Pearson r, shifted to [0,1] | Mean-subtracted dot product |
| `waveform` | Cross-correlation with Gaussian smoothing | Most compute-intensive |

### 10.3 Pattern Storage Strategy (learnCurrentPattern)

`Neuron::learnCurrentPattern()` is called when a neuron is designated a winner:

```
1. Convert current spike buffer → BinaryPattern (newPat)

2. If patterns.size() < neuronMaxPatterns (500):
     patterns.push_back(newPat)
     return

3. Find best matching existing pattern:
     bestSim = max cosine_similarity(newPat, p) for p in patterns

4. If bestSim >= neuronThreshold (1.2):
     // Near-duplicate: blend into existing
     blend(bestMatch, newPat, alpha=0.20)
     // target = 0.8 × target + 0.2 × source

5. Else:
     // Novel pattern: replace a random entry
     patterns[random_index] = newPat
```

The blend operation (`BinaryPattern::blend`) is a weighted average: each bin of the target pattern moves 20% toward the new pattern. This implements **online prototype updating** — frequently-seen patterns consolidate while novel patterns displace unused slots.

### 10.4 getBestSimilarity()

Returns the maximum cosine similarity between the neuron's current spike buffer (as BinaryPattern) and all stored reference patterns. Used during competition to score neurons.

---

## 11. Classification System

**File**: `src/experiment/KNNClassifier.cpp`

### 11.1 Pattern Storage During Training

For each training image, after L5 competition, `storePattern(label, l5Counts, l5Latencies)` is called:

- **k-NN store**: appends the L5 count vector to `classPatterns_[label]` (capped at `maxPatternsPerClass = 2048`)
- **Centroid accumulation**: `classCentroids_[label][i] += counts[i]` for all L5 neurons
- **Latency accumulation**: running sum and observation count for latency-based similarity
- **IDF tracking**: `neuronPatternPresence_[i]++` for every L5 neuron that fired

### 11.2 IDF Weighting

Inspired by TF-IDF in text retrieval. L5 neurons that fire for **every** class (ubiquitous, low discriminative value) get low weight; neurons that fire for **only a few** classes get high weight:

```
idf(i) = (log((N+1) / (df_i+1)) + 1) ^ idfPower

where N = total training patterns seen
      df_i = number of patterns where neuron i was active
      idfPower = readoutIdfPower (default 1.0)
```

Applied in `weightedCosineSimilarity()` when `enableReadoutIdfWeighting = true` (default: off).

### 11.3 k-NN Classification

```
classifyKNN(testCounts, testLatencies):
  1. For each stored training pattern (all classes):
       sim = weightedCosineSimilarity(testCounts, trainCounts)
       if enableTemporalLatencyReadout:
         latSim = latencySimilarity(testLatencies, trainLatencies)
         sim = (1-w) × sim + w × latSim  [w = temporalLatencyWeight = 0.35]

  2. Sort all (sim, class) pairs descending

  3. Take top K=7 neighbors; vote by class majority

  4. Return (winningClass, topSimilarity)
```

### 11.4 Centroid Fallback

If k-NN returns no winner (empty pattern store), `classifyCentroid()` compares against per-class centroid vectors (average L5 activation across all training examples). Same weighted cosine similarity, using running centroid.

### 11.5 Pairwise Disambiguation

For historically confused letter pairs (T/I, G/Q, I/L), a post-processing step can force a choice if the centroid similarity difference exceeds a configured margin:

```
if predictedLabel ∈ {classA, classB}:
    simA = centroidSimilarity(testCounts, classA)
    simB = centroidSimilarity(testCounts, classB)
    if abs(simA - simB) >= margin:
        predictedLabel = argmax(simA, simB)
```

Currently disabled by default (`enablePairDisambiguation = false`).

### 11.6 Output Spike Vote (Optional)

When `enableOutputVote = true`, the output layer spike counts are checked first. If the top output population fired ≥ 5 spikes **and** had ≥ 1.1× more spikes than the second-place population, it wins without using k-NN. This is currently disabled in the SONATA experiment.

### 11.7 L5 Divisive Normalization (Optional)

Before classification, per-column L5 activity can be normalized:

```
for each column:
    colSum = sum of L5 spike counts in this column
    scale = l5DivisiveTargetPerColumn / colSum
    counts[i] = round(counts[i] × scale)  for each neuron in column
```

This equalizes activity across columns with different overall firing rates. Currently disabled (`enableL5DivisiveNormalization = false`).

---

## 12. Training vs Inference Modes

### 12.1 Training Pass (run())

```
For each training image (up to trainingExamplesPerClass=200 per class):
  1. resetNeuronStdpEligibility()
  2. encodeAndInject(image)
  3. waitForSimTime(base + inputLatencyMs + 5 ms)
  4. runL4Competition() → colHasL4[]
  5. waitForSimTime(base + kL4ToL5SettleMs)
  6. runL5Competition() → l5Winners[]
  7. SupervisedTeacher::teach(label, ...) → patterns learned
  8. collectL5Readout() → L5CountVector
  9. applyL5DivisiveNormalization()
  10. if hasL5Activity: classifier.storePattern(label, counts, latencies)

After all training images:
  applyHomeostasis()  → adjust intrinsic excitability

Then: runTestingPhase() → accuracy + confusion matrix
```

STDP is **enabled** during training. The supervised teach step forces the correct output class, which drives STDP on L5→Output synapses.

### 12.2 Inference (runInference())

```
1. STDP disabled (propagator->setStdpEnabled(false))
2. encodeAndInject(image)
3. waitForSimTime(base + inputLatencyMs + 5 ms)
4. runL4Competition()
5. waitForSimTime(base + kL4ToL5SettleMs)
6. runL5Competition()
7. if enableFullPropagation: fire L5 winners → L5→Output spikes
8. collectL5Readout()
9. collect output spike counts
10. applyOutputCompetition() (if enabled)
11. return InferenceResult{l5Counts, l5Latencies, outSpikeCounts}
```

No supervised teaching. No pattern learning. No STDP.

### 12.3 Convergence Detection

After each training pass, if `|currentAccuracy - previousAccuracy| < accuracyEpsilon (0.001%)` for `stablePassesRequired (3)` consecutive passes, training terminates early.

---

## 13. Homeostatic Plasticity

**File**: `src/Neuron.cpp`, method `applyHomeostaticPlasticity()`

Applied once per training pass (not per image). Each neuron tracks its recent firing rate and compares to a target:

```
if firingRate > targetRate:
    intrinsicExcitability -= homeostaticRate    // reduce excitability
if firingRate < targetRate:
    intrinsicExcitability += homeostaticRate    // increase excitability

intrinsicExcitability = clamp(intrinsicExcitability, minExcitability, maxExcitability)
```

Target rates by layer:
- L4: `l4TargetRate = 8.0`
- L5: `l5TargetRate = 5.0`
- Output: `outputTargetRate = 2.0`

Intrinsic excitability acts as a gain multiplier on incoming spike amplitudes, ensuring no layer goes permanently silent or saturates.

---

## 14. Configuration Reference

All parameters live in `ExperimentConfig`. Key groups:

### 14.1 Network Architecture
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `numColumns` | 16 (SONATA overrides to 36) | Number of cortical columns |
| `layer4Size` | 7 | L4 grid side (7×7 = 49 neurons/column) |
| `layer5Neurons` | 80 | L5 neurons per column |
| `numClasses` | 26 | Output classes (A–Z) |
| `neuronsPerOutputClass` | 3 | Output neurons per class |

### 14.2 Encoding
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `pixelThreshold` | 0.4 | Min normalized intensity to fire |
| `inputLatencyMs` | 15.0 | Max spike latency (bright=0ms, dim=15ms) |
| `interImageGapMs` | 550.0 | Gap between image presentations |
| `neuronWindow` | 500.0 | Spike buffer integration window (ms) |

### 14.3 Competition
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `maskMinActive` | 6 | Min inputs to activate a column |
| `l4Keep` | 8 | L4 winners per column |
| `l5Keep` | 8 | L5 winners per column |
| `l4SimilarityWeight` | 0.25 | Pattern match fraction in L4 score |
| `l5SimilarityWeight` | 0.60 | Pattern match fraction in L5 score |
| `l4RowDelay` / `l4ColDelay` | 0.3 / 0.2 ms | Spatial firing delay gradients |
| `l4PostShiftMs` | -6.0 | Global time shift for L4 winner firing |

### 14.4 STDP
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `traceStdp` | true | Use trace-based (vs classic) STDP |
| `stdpLtdScale` | 0.3 | LTD magnitude scale factor |
| `stdpLtdWindowMs` | 70.0 | LTD eligibility window |
| `enableStdpEligibilityGate` | true | Require STDP history before learning |
| `stdpEligibilityThreshold` | -0.002 | Min STDP score to allow pattern storage |

### 14.5 Classification
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `knnK` | 7 | Neighbors in k-NN vote |
| `maxPatternsPerClass` | 2048 | Max k-NN training patterns per class |
| `keepL5History` | true | Retain patterns across training passes |
| `enableTemporalLatencyReadout` | false | Fuse latency similarity in readout |
| `enableReadoutIdfWeighting` | false | IDF-weight L5 neurons in similarity |
| `enableL5DivisiveNormalization` | false | Per-column L5 activity normalization |
| `enablePairDisambiguation` | false | Post-hoc T/I, G/Q, I/L refinement |

---

## 15. Current State Assessment

### 15.1 What Is Working
- **End-to-end pipeline**: Image → spikes → L4 → L5 → k-NN classification executes reliably
- **STDP and supervised teaching**: Weights update; pattern libraries grow during training
- **Confusion matrix**: Per-class accuracy breakdown visible after each test phase
- **Convergence detection**: Training stops when accuracy stabilizes
- **Inter-column inhibition**: Feature-gated cross-column suppression implemented

### 15.2 Known Architectural Tensions

1. **L4 competition is partially redundant with STDP**: L4 neurons fire in competition *and* receive STDP updates from input. If competition winners change across passes, STDP receives conflicting training signal.

2. **Supervised teaching dominates**: The teacher forces specific neurons to fire regardless of their natural response, which can overpower unsupervised STDP organization in L4. The system is more "supervised with spike timing" than truly bio-inspired unsupervised learning.

3. **k-NN pattern accumulation grows unbounded** (up to 2048/class × 26 = 53,248 patterns). Classification inference time scales linearly with this.

4. **550 ms inter-image gap**: Most of this gap is dead simulation time (spikes finish within ~100 ms). This is the dominant speed bottleneck — ~5 images/second throughput.

5. **L5 similarity weight (0.60)**: Because L5 patterns are sparse and noisy early in training, high similarity weight can cause early passes to score many neurons as zero (no learned patterns yet). The system partially relies on spike-count drive early on, then transitions to similarity-driven selection.

6. **Output layer effectiveness**: `enableOutputVote` is disabled; the output layer is trained but not used during inference. The output STDP signal may therefore be training L5→Output weights that go unused.

7. **Pattern library replacement is random**: When a neuron's library is full and a novel pattern arrives, a *random* slot is replaced. This discards potentially informative old patterns indiscriminately.

---

## 16. Recommended Next Steps

Based on the analysis above, the following improvements are ordered by expected impact-to-effort ratio:

### 16.1 High Impact / Lower Effort

**A. Enable output spike voting during inference**
Re-enable `enableOutputVote` and tune `outputCompetitionMinSpikes`. The output layer is already being trained — if it learned correctly, it provides a cleaner classification signal than k-NN over the full L5 vector.

**B. Reduce inter-image gap**
The 550 ms gap is conservative. Profile actual spike completion time per image (it's ~80–100 ms). Reduce `interImageGapMs` to 150–200 ms to 3–4× training throughput.

**C. Enable IDF weighting in readout**
`enableReadoutIdfWeighting = true` with `readoutIdfPower = 1.0` re-weights L5 neurons by discriminative value at zero training cost. This is a pure inference-time improvement.

**D. Enable pairwise disambiguation for known confusions**
The confusion matrix reveals which letter pairs are chronically confused. Set non-zero margins for those pairs (T/I, G/Q, I/L) to capture easy wins.

### 16.2 Medium Impact / Medium Effort

**E. Replace random pattern eviction with LRU or worst-match eviction**
When the pattern library is full, evict the least recently used pattern (or the one with highest overlap with others) rather than a random slot. This preserves diversity.

**F. Adaptive L5 similarity weight schedule**
Start pass 1 with low `l5SimilarityWeight` (0.1 — mostly spike-count driven) and ramp it up each pass as patterns accumulate. Prevents early passes from under-activating L5.

**G. Profile and reduce k-NN inference time**
With 2048 patterns × 26 classes = 53,248 dot products per test image, k-NN is O(N) and slow. Switch to a centroid-only classifier after training stabilizes, or cluster stored patterns per class to reduce N.

**H. Diagnose the L4 competition timing**
The `l4PostShiftMs = -6.0` means L4 fires at `baseTime - 6 ms`, which is *before* the image's latency window finishes. Verify that L4 winners are actually seeing the complete input pattern before firing, not just the earliest bright-pixel spikes.

### 16.3 Architectural Changes

**I. Decouple unsupervised L4 STDP from supervised L5 teaching**
Let L4 self-organize purely via STDP for the first N passes (no competition, no teaching). Then freeze L4 weights and train L5+output with supervision. This is a staged learning approach used in deep SNNs.

**J. Replace k-NN with a trained linear readout (SVM or logistic)**
The L5 population vector is a fixed-dimension feature vector. A linear SVM trained on it would generalize better than k-NN and classify in O(1) per image.

**K. Add recurrent L6→L4 feedback**
The current network is purely feedforward. Adding L6→L4 top-down connections would allow learned letter-level expectations to gate L4 feature detection — biologically motivated and potentially powerful for ambiguous letters.

**L. Temporal latency readout**
Enable `enableTemporalLatencyReadout` and tune `temporalLatencyWeight`. First-spike latency carries information about input intensity distribution that spike-count alone discards.

---

**Document Version**: 2.0
**Last Updated**: 2026-02-22
**Status**: Living document — update when architecture changes


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

