# RabbitHoles

This document records lines of work that consumed substantial time without producing a viable mainline result. The point is not to forbid future work. The point is to avoid repeating the same failed sequence without a new technical reason.

## Current Mainline To Protect

These are the benchmark paths worth keeping as the active reference surface:

- Unilateral Retina static: `86.17%`
- Bilateral Retina static: `87.29%`
- Bilateral Retina continuous: `87.50%` initial, `88.85%` post-correction
- CIFAR-10 bilateral natural-features experimental reference: `36.60%` on `1000/class, 5000` test

These live in:

- `configs/emnist_retina_experimental.sonata.json`
- `configs/emnist_retina_bilateral_experimental.sonata.json`
- `configs/emnist_retina_bilateral_continuous.sonata.json`
- `configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json`
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

### 8. Treating CIFAR Fusion Tuning As The Main Bottleneck

What we tried:

- stronger disagreement arbitration
- margin-preference fusion tweaks
- corpus weighting variants on the CIFAR natural-image path

Why it was not productive:

- CIFAR errors were mostly cases where both hemispheres were already wrong
- fusion had limited headroom relative to the upstream representation problem
- the benchmark stayed flat while complexity increased

Lesson:

- do not spend time on fusion tuning when branch and hemisphere separability are already weak
- first measure how many errors are even recoverable by better arbitration

### 9. Tuning CIFAR Branch Aggregation When All Branches Are Already Collapsing The Same Way

What we tried:

- branch-level vote attribution
- checking whether one branch was dominating the wrong hemisphere label

Why it was not productive:

- all three branches in each hemisphere were usually biased toward the same wrong basin
- changing aggregation logic would not fix a branch representation that is already collapsed upstream

Lesson:

- if branch diagnostics show the same wrong target across all branches, stop tuning hemisphere aggregation
- fix the front-end representation instead

### 10. Repeated Natural-Image Front-End Tweaks Without Reviving The Edge Path

What we tried:

- chromaticity-only color opponency
- chromaticity plus gray-world color adaptation
- local contrast normalization on top of that path
- retinotopic and patchbank CIFAR variants
- luminance-only edge variants
- weaker thresholds and orientation reweighting
- small-kernel Gabor swaps
- overlapping edge receptive fields
- a separate coarser edge-analysis grid

Why it was not productive:

- some small-sample gates moved, but large-sample results did not hold
- several variants regressed sharply despite looking more biologically shaped
- the key diagnostic never changed: orientation slices stayed effectively dead (`raw_l2=0`, `raw_active=0%`)
- in that phase, the only CIFAR gains that generalized came from richer low-frequency appearance channels, not from the edge/orientation tweaks we were testing

Lesson:

- do not keep swapping edge operators or receptive-field geometry when orientation diagnostics remain zero
- before claiming a front-end improvement, verify that the orientation slice actually carries nonzero energy
- if the edge path is still zero, treat appearance-bank features as the real signal and stop tuning edge operators

### 11. Reviving The Entire CIFAR Edge Path At Once

What we tried:

- patch-local edge normalization on every branch
- larger edge-analysis patches on every branch
- reviving orientation signal globally once diagnostics showed the operator scale mismatch

Why it was not productive:

- it solved the wrong subproblem too aggressively
- orientation signal came alive, but overall accuracy regressed sharply
- the classifier collapsed toward a new wrong basin instead of improving (`ship` became dominant on the all-branch edge revival run)
- this showed that a globally revived coarse edge path can overpower the appearance path rather than complement it

Lesson:

- do not turn on coarse normalized edge analysis everywhere at once
- treat revived natural-image edge signal as a supplemental branch-level cue, not the main code path
- the only edge revival that held was the constrained hybrid on the `g10` branches, with reduced `orientation_feature_gain`

### 12. Temporal Coarse-To-Fine Inference Before The Coarse Cue Proved Useful

What we tried:

- a two-pass CIFAR inference path that masked deferred branch orientation slices on an initial coarse pass
- blending coarse and fine classifier confidence with explicit weights
- a narrower variant that deferred only the supplemental `g10` coarse-edge branches

Why it was not productive:

- even the narrower selective version regressed on the controlled CIFAR gate
- the coarse pass shifted confidence toward the wrong appearance-heavy basin instead of improving the final decision
- the branch representation was not yet stable enough for temporal staging to help; it just reweighted the same weak evidence

Lesson:

- do not add temporal coarse-to-fine arbitration until the supplemental coarse branch is already improving the representation by itself
- when testing temporal staging, start with a proven branch-level gain and verify that the coarse pass helps the same classes rather than just amplifying existing collapse

### 13. Broad Feature-Group Divisive Normalization On The CIFAR Hybrid Path

What we tried:

- per-region divisive normalization over orientation, chromatic, and appearance feature groups
- weak cross-group coupling so dominant appearance channels would be compressed before hemisphere classification

Why it was not productive:

- the controlled CIFAR gate regressed slightly instead of improving
- it changed class balance, but did not improve the underlying fusion or hemisphere separability enough to matter
- the current hybrid path appears to benefit more from preserving the existing appearance-plus-supplemental-edge balance than from globally re-scaling feature groups

Lesson:

- do not apply broad group-wise divisive normalization across all CIFAR branches without evidence that one group is truly saturating the classifier
- if divisive normalization is revisited, test it only on the active supplemental branch first, not on the whole natural-features path

### 14. Weak Local Contour Integration On The Active `g10` Branch

What we tried:

- a minimal branch-local contour facilitation step on the CIFAR hybrid path
- support applied only to neighboring `g10` subfields
- orientation-specific lateral support kept weak and branch-local

Why it was not productive:

- the controlled CIFAR gate still regressed slightly instead of improving
- it moved some class balances but did not improve the underlying representation enough to beat the current hybrid baseline
- the existing `g10` supplemental edge branch already appears close to the useful signal limit under the current classifier

Lesson:

- do not keep tuning branch-local contour support on the current CIFAR hybrid path without a larger representational change
- local contour integration is not the next lever if it cannot beat the gate even when constrained to the one branch that already carries live orientation signal

### 15. Stage-1 Branch Support Banks On The CIFAR Hybrid Path

What we tried:

- appended branch-local similarity features derived from multiple per-class support prototypes
- kept the bilateral scaffold and fusion path unchanged
- tested a minimal bank size (`2` support sets per class per branch) before any larger sweep

Why it was not productive:

- the small CIFAR gate improved only marginally
- the larger CIFAR run ended at exactly the same accuracy as the current promoted baseline (`36.60%`)
- that means the added representation complexity did not buy a real gain on the benchmark that matters

Lesson:

- do not add extra stage-1 branch-bank machinery unless it clears the larger natural-image benchmark, not just the small gate
- if a representation change ties the baseline at full scale, treat it as a dead end and keep the simpler path

### 16. Broad Mid-Scale Appearance Pyramids On Every CIFAR Branch

What we tried:

- replaced the pooled `appearance_bank` auxiliary path with a larger quadrant-aware appearance pyramid on every branch
- kept the bilateral scaffold and the hybrid `g10` edge branch unchanged

Why it was not productive:

- the controlled CIFAR gate collapsed sharply instead of improving
- the broader auxiliary vector distorted the working balance between appearance and the supplemental `g10` edge path
- adding mid-scale appearance detail everywhere at once overwhelmed the current classifier rather than preserving useful structure

Lesson:

- do not expand the auxiliary appearance code globally across all CIFAR branches in one step
- if mid-scale appearance is revisited, it needs to be branch-selective or classifier-aware from the start, not a blanket replacement of the current appearance bank

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
- use the CIFAR natural-features path as the current natural-image reference, not as proof that the edge path is solved
- treat graph-native cortical vision as experimental research, not the default execution path
