# RabbitHoles

This document records lines of work that consumed substantial time without producing a viable mainline result. The point is not to forbid future work. The point is to avoid repeating the same failed sequence without a new technical reason.

## Current Mainline To Protect

These are the benchmark paths worth keeping as the active reference surface:

- Unilateral Retina static: `86.17%`
- Bilateral Retina static: `87.29%`
- Bilateral Retina continuous: `87.56%` initial, `88.44%` post-correction

These live in:

- `configs/emnist_retina_experimental.sonata.json`
- `configs/emnist_retina_bilateral_experimental.sonata.json`
- `configs/emnist_retina_bilateral_continuous.sonata.json`
- `experiments/emnist_retina_letters.cpp`

Any new experimental vision path should be compared against these, not treated as a replacement by default.

## Rabbit Holes

### 1. Bolting Graph Runtime Into The Retina Benchmark Harness

What we tried:

- adding graph execution directly into `experiments/emnist_retina_letters.cpp`
- mixing mature Retina benchmarking with experimental instantiated-network execution

Why it was not productive:

- it coupled a stable benchmark harness to an immature research path
- it made it too easy to keep tuning graph behavior inside the wrong executable
- it obscured whether regressions came from the model or the harness

What to do instead:

- keep `emnist_retina_letters` as the stable bilateral Retina benchmark
- put graph-native experiments in separate executables or branches

Status:

- removed from the active branch surface

### 2. Treating The Multistage Graph Path As A Near-Term Replacement

What we tried:

- a separate multistage declarative cortical experiment
- stage1 -> association -> higher/fusion -> decision graph paths
- repeated config tuning to lift its accuracy

Why it was not productive:

- the path never became competitive with the working Retina baselines
- it stayed in the low single-digit / low double-digit range instead of approaching the high-80s baseline
- the representation was not class-separable enough upstream, so downstream tuning had very little value

What to do instead:

- keep this as a research track only
- require stage-wise separability evidence before spending more time on decision/readout tuning

### 3. Running A Laminar `L4 -> L2_3 -> L5` Config On The Wrong Runtime

What we tried:

- using a runtime path that effectively assumed `L4 -> L5`
- driving a more corticalized laminar config through it anyway

Why it was not productive:

- the runtime/config mismatch killed signal before it reached the intended downstream stage
- this looked like a model problem at first, but it was partly an execution-path mismatch

Lesson:

- before tuning a cortical config, verify that the runtime actually supports the declared laminar flow

### 4. Tuning The Decision Microcircuit Before Upstream Representations Were Separable

What we tried:

- decision nuclei
- excitatory/inhibitory settling
- winner fatigue
- homeostasis
- decision-stage recurrent tuning

Why it was not productive:

- the decision stage often received dead, weak, or poorly differentiated upstream drive
- the result was attractor collapse onto a few classes rather than improved recognition
- this was downstream tuning on top of an upstream representation problem

Lesson:

- do not tune decision dynamics until higher->readout and readout->decision separability has been measured and shown to be usable

### 5. Widening The Higher-Visual Bank Without Fixing Selectivity

What we tried:

- larger higher-visual banks
- richer higher-visual fan-in
- broader convergence into readout or decision

Why it was not productive:

- extra width increased complexity and search space without improving class separation
- broader fan-in often increased collapse, not discrimination
- stronger throughput into decision just amplified weak or wrong structure

Lesson:

- do not widen higher visual blindly
- fix selectivity before adding capacity

### 6. Stronger Feedforward / Stronger Readout Dynamics As A Substitute For Better Representation

What we tried:

- stronger `higher -> readout`
- stronger `readout -> decision`
- more recurrent readout cycles
- stronger readout inhibition
- tighter readout top-k and stricter readout targets

Why it was not productive:

- stronger drive changed which wrong basin won
- longer settling often made collapse worse on the next benchmark gate
- local readout tuning did not solve the missing class-separable upstream code

Lesson:

- do not keep increasing gain, recurrence, or inhibition when the real problem is upstream representation quality

### 7. Adding More Structural Hierarchy Before Verifying Propagation

What we tried:

- retinotopic tiled stage1
- tiled association stages
- distributed higher-visual banks
- direct class decision stages

Why it was not productive:

- structure became more biologically shaped, but useful differentiated drive still did not reliably reach the decision stage
- architecture complexity outran instrumentation

Lesson:

- before adding another area or stage, verify:
  - synapses exist where expected
  - spikes arrive in the right time window
  - post-synaptic drive is nonzero
  - class separation improves at that boundary

## Reusable Work Worth Keeping

Not everything from the failed path was wasted. These pieces are reusable and should be retained:

- exact path-backed declarative group resolution in `NetworkConstructor`
- wildcard/exact hierarchy path support for layer and population resolution
- incoming-drive-aware competition scoring in `CompetitionManager`
- bilateral Retina declarative hierarchy with `attach_path` and `fusion_path`
- bilateral continuous-learning benchmark path

These improve the framework or the benchmark surface without committing us to the failed graph path.

## Rules For Revisiting This Area

Do not reopen the multistage graph path unless at least one of these is true:

- there is a new runtime with native support for the declared laminar computation
- there is new instrumentation showing meaningful class separation at `higher -> readout`
- there is a new biologically motivated representation stage with a concrete reason to expect separation gains

If we do revisit it, the order should be:

1. verify propagation and timing
2. measure stage-wise separability
3. improve representation
4. only then tune decision/readout dynamics

## Practical Default

For current vision work:

- use the bilateral Retina static and continuous paths as the mainline benchmark
- treat graph-native cortical vision as experimental research, not the default execution path
