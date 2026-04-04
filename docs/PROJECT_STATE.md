# Project State - 2026-04-04

This document captures the current project state with enough detail for a new conversation to resume work without re-discovering recent context.

## Repository Snapshot

- Branch: `ContinuousLearning`
- Date: `2026-04-04`
- Stable CIFAR baseline reference commit: `7eef39b` (`Promote CIFAR hybrid edge retina path`)
- Stable branch intent:
  - keep the bilateral Retina path as the mainline
  - keep rabbit-hole experiments documented instead of merged
  - prefer bounded, measured experiments over broad architecture rewrites

## Local Artifacts At Capture Time

Local untracked artifacts currently present:
- `data/`
- `.codex`
- `docs/cifar10_retina_experiment.png`
- `scripts/render_cifar10_retina_experiment_diagram.py`

These are local/session artifacts unless explicitly promoted.

## Stable, Verified Baselines

### EMNIST

- Static bilateral Retina, full dataset:
  - config: `configs/emnist_retina_bilateral_experimental.sonata.json`
  - result: `88.51%` (`18410/20800`)
  - log: `build/emnist_retina_bilateral_full_all.log`
- Continuous bilateral Retina, full benchmark:
  - config: `configs/emnist_retina_bilateral_continuous.sonata.json`
  - result: `87.50% -> 88.85%`
  - correct: `4620/5200`
  - log: `build/retina_continuous_parallel_3200_5200.log`

### MNIST

- Static bilateral Retina, full dataset:
  - config: `configs/mnist_retina_bilateral_experimental.sonata.json`
  - result: `96.88%` (`9688/10000`)
  - log: `build/mnist_retina_bilateral_full_all.log`

### CIFAR-10

Current promoted CIFAR reference:
- config: `configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json`
- sampled gate: `30.00%` (`300/1000`)
  - log: `build/cifar10_retina_natural_features_promoted_200_1000.log`
- larger sampled benchmark: `36.60%` (`1830/5000`)
  - log: `build/cifar10_retina_natural_features_hybrid_g10_edge5_1000_5000.log`

This `36.60%` result is the current CIFAR benchmark to beat.

## Mainline Architecture

### Shared Bilateral Retina Scaffold

The working architecture across EMNIST, MNIST, and CIFAR is still the same bilateral scaffold:

- input domain adapter
- left transformed view
- right transformed view
- three Retina branches per hemisphere:
  - `SobelG9`
  - `SobelG10`
  - `DogG9`
- hemisphere-local stage-1 classifier
- interhemispheric corpus-callosum-style fusion
- final classification

### Domain Adapter Layer

The Retina path is now domain-driven rather than EMNIST-hardcoded.

Implemented adapters:
- `emnist`
- `mnist`
- `cifar10`

Primary files:
- `include/snnfw/domain/VisualDomainAdapter.h`
- `src/domain/VisualDomainAdapter.cpp`

### CIFAR-Specific Promoted Path

The current best CIFAR path keeps the bilateral scaffold but uses a richer natural-image front end:

- `representation = features`
- color-aware natural-image appearance features
- `appearance_bank` auxiliary path
- constrained coarse normalized edge revival only on `left_retina_g10` and `right_retina_g10`
- no broad edge revival on `g9` or `DoG`

That promoted CIFAR result was committed in:
- `7eef39b` (`Promote CIFAR hybrid edge retina path`)

## Recent Dead Ends Already Documented

See `docs/RabbitHoles.md` for full detail.

Recent CIFAR dead ends already established:
- broad fusion/arbitration tuning was not the bottleneck
- branch aggregation tuning was not the bottleneck
- broad edge revival across all branches regressed accuracy
- retinotopic / patchbank / pyramid front-end expansions regressed accuracy
- temporal coarse-to-fine reweighting did not hold up
- broad divisive normalization regressed slightly
- weak local contour integration on `g10` regressed slightly
- stage-1 branch support banks tied the large benchmark and were rejected

Practical rule: do not revisit any of those without new evidence.

## Current Experimental Track In Progress

The user explicitly chose to continue beyond `36.60%` by opening a separate, bounded natural-image track rather than mutating the promoted CIFAR baseline blindly.

### Active Idea: Stage-1 Shape/Surface Split

This track is experimental and not promoted into the CIFAR baseline config.

Goal:
- keep the bilateral scaffold unchanged
- split stage-1 classifier input into two streams before hemisphere classification:
  - `shape` stream = non-auxiliary features
  - `surface` stream = auxiliary appearance-like features
- normalize each stream separately and combine them with configurable weights

#### Code Changes Currently in `experiments/emnist_retina_letters.cpp`

Added config fields:
- `stage1_stream_split_mode`
- `stage1_shape_stream_weight`
- `stage1_surface_stream_weight`

Added stage-1 stream support:
- raw training patterns retained separately from classifier-space patterns
- per-hemisphere `shape` / `surface` index lists derived from `PatternSlice`s
- classifier patterns built as:
  - separate L2 normalization of `shape` and `surface`
  - weighted concatenation of both streams

Inference path changes:
- raw Retina pattern still extracted normally
- `classifierPattern` built from raw pattern using the shape/surface split
- hemisphere classification uses `classifierPattern`
- corpus-callosum evidence uses `classifierPattern` when present
- online positive updates also use `classifierPattern`

Diagnostics changes:
- branch and hemisphere separability diagnostics continue to run on raw patterns when available, so raw representation diagnostics remain interpretable

#### Small-Gate Result

Shape/surface gate config:
- `build/cifar10_retina_bilateral_natural_features_shape_surface.sonata.json`

Small gate command that succeeded:
- `EXAMPLES_PER_CLASS=200 TEST_LIMIT=1000 LOG_PATH=build/cifar10_retina_natural_features_shape_surface_200_1000.log CONFIG_PATH=build/cifar10_retina_bilateral_natural_features_shape_surface.sonata.json scripts/run_cifar10_retina_bilateral_natural.sh`

Result:
- `30.70%` (`307/1000`)
- baseline for comparison: `30.00%` (`300/1000`)

Interpretation:
- this is a real sampled improvement (`+0.70 pts`)
- it is the only currently active experimental track worth checking further

### Large-Run Status: Unresolved / Needs Investigation

Large-run command attempted twice:
- `EXAMPLES_PER_CLASS=1000 TEST_LIMIT=5000 LOG_PATH=build/cifar10_retina_natural_features_shape_surface_1000_5000.log CONFIG_PATH=build/cifar10_retina_bilateral_natural_features_shape_surface.sonata.json scripts/run_cifar10_retina_bilateral_natural.sh`
- retry log: `build/cifar10_retina_natural_features_shape_surface_1000_5000_retry.log`

Observed behavior on both attempts:
- only startup lines were written
- logs ended after:
  - Retina config summary
  - `Hemisphere Left Hemisphere stored stage1 patterns: 10000`
  - `Hemisphere Right Hemisphere stored stage1 patterns: 10000`
- no final accuracy summary was emitted
- no long-running `emnist_retina_letters` process remained afterward

Interpretation:
- the large shape/surface run did **not** produce a valid benchmark result yet
- before promoting or rejecting the track, this failure mode needs to be explained

Likely next debugging target:
- inspect whether the run is exiting early due to:
  - silent failure after stage-1 pattern storage
  - resource exhaustion
  - an unhandled path specific to larger CIFAR runs with transformed classifier patterns

## Recommended Resume Point

If a new conversation needs to pick this up, start here:

1. Do **not** modify the promoted CIFAR baseline.
2. Work only on the active shape/surface experiment in `experiments/emnist_retina_letters.cpp`.
3. First resolve why the large `1000/class, 5000 test` shape/surface run exits after startup without a final result.
4. Only after a valid large result exists, compare it to the current CIFAR benchmark to beat:
   - `36.60%` from `build/cifar10_retina_natural_features_hybrid_g10_edge5_1000_5000.log`
5. Promotion rule:
   - if large shape/surface result `> 36.60%`, promote it
   - otherwise revert it and keep the current baseline

## Files Most Relevant To Resume Work

Primary code:
- `experiments/emnist_retina_letters.cpp`
- `src/adapters/RetinaAdapter.cpp`
- `include/snnfw/adapters/RetinaAdapter.h`
- `src/domain/VisualDomainAdapter.cpp`
- `include/snnfw/domain/VisualDomainAdapter.h`

Primary configs:
- `configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json`
- `build/cifar10_retina_bilateral_natural_features_shape_surface.sonata.json`
- `configs/emnist_retina_bilateral_continuous.sonata.json`
- `configs/emnist_retina_bilateral_experimental.sonata.json`
- `configs/mnist_retina_bilateral_experimental.sonata.json`

Primary reference docs:
- `README.md`
- `docs/RabbitHoles.md`
- `docs/PR_SUMMARY_CONTINUOUS_LEARNING.md`

## Commands That Matter

Build:
- `cmake --build build -j8 --target emnist_retina_letters`

Current CIFAR promoted baseline gate:
- `EXAMPLES_PER_CLASS=200 TEST_LIMIT=1000 LOG_PATH=build/cifar10_retina_natural_features_promoted_200_1000.log CONFIG_PATH=configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json scripts/run_cifar10_retina_bilateral_natural.sh`

Current CIFAR promoted larger sampled benchmark:
- `EXAMPLES_PER_CLASS=1000 TEST_LIMIT=5000 LOG_PATH=build/cifar10_retina_natural_features_hybrid_g10_edge5_1000_5000.log CONFIG_PATH=configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json scripts/run_cifar10_retina_bilateral_natural.sh`

Shape/surface small gate:
- `EXAMPLES_PER_CLASS=200 TEST_LIMIT=1000 LOG_PATH=build/cifar10_retina_natural_features_shape_surface_200_1000.log CONFIG_PATH=build/cifar10_retina_bilateral_natural_features_shape_surface.sonata.json scripts/run_cifar10_retina_bilateral_natural.sh`

Shape/surface large run attempted but not validated:
- `EXAMPLES_PER_CLASS=1000 TEST_LIMIT=5000 LOG_PATH=build/cifar10_retina_natural_features_shape_surface_1000_5000.log CONFIG_PATH=build/cifar10_retina_bilateral_natural_features_shape_surface.sonata.json scripts/run_cifar10_retina_bilateral_natural.sh`

## Bottom Line

- EMNIST and MNIST paths are stable and strong.
- CIFAR has improved from `10.40%` grayscale to `36.60%` on the current promoted natural-image feature path.
- The only active experimental track worth pursuing right now is the stage-1 `shape_surface` split.
- That track showed a small positive gate (`30.70%`) but does **not yet have a valid large-run result**.
- The immediate next task is to resolve the incomplete large-run behavior before doing anything else.
