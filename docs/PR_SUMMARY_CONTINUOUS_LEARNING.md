# ContinuousLearning PR Summary

## Title
Add reward-driven continuous learning for the bilateral Retina EMNIST benchmark

## What changed
- Added reward-driven online adaptation to `experiments/emnist_retina_letters.cpp`
- Added delayed replay queue for misclassified or uncertain samples
- Added context-gated plasticity based on uncertainty and hemisphere disagreement
- Added bounded hybrid updates:
  - online centroid updates
  - bounded online exemplar insertion
- Added separate before/after reporting for continuous-learning evaluation
- Added the promoted continuous config:
  - `configs/emnist_retina_bilateral_continuous.sonata.json`
- Added runner script:
  - `scripts/run_emnist_retina_bilateral_continuous.sh`
- Updated `README.md` to separate static vs continuous benchmark references

## Why
The static Retina benchmarks measure fixed-memory generalization. The framework also needs an explicit continuous-learning mode that updates from scalar reward during ongoing experience, without switching back to oracle training inside the learner.

## Benchmark categories
- Static unilateral Retina
- Static bilateral Retina
- Continuous bilateral Retina

These are separate benchmark categories. The continuous benchmark includes online reward-driven adaptation during evaluation and is not directly comparable to the static held-out metric.

## Reference results
- Static unilateral Retina:
  - `86.17%`
- Static bilateral Retina:
  - `87.29%`
- Continuous bilateral Retina:
  - initial `87.50%`
  - post-correction `88.85%`

## Continuous benchmark details
- Dataset: EMNIST Letters
- Training: `3200/class`
- Test: `5200`
- Seed: `42`
- Result held across replicate:
  - `4620/5200`
  - initial `87.50%`
  - post-correction `88.85%`
  - correction events `650`
  - corrected `70`
  - replays `1849`
  - replay timing `avg_delay_steps=4.00`, `avg_eligibility=0.40`
  - elapsed `2448.60s` with parallel hemisphere processing
  - prior serial elapsed `4769.88s`
  - runtime reduction `48.67%` (`1.95x`)

## Main implementation points
- Reward signal is scalar inside the learner
- The EMNIST label is only used by the benchmark harness to generate reward sign and report metrics
- Online adaptation uses:
  - positive reward gain
  - negative reward gain
  - replay queue capacity
  - replay delay and pause scheduling
  - eligibility-trace decay
  - uncertainty threshold
  - context uncertainty gain
  - context disagreement gain

## Main files
- `experiments/emnist_retina_letters.cpp`
- `configs/emnist_retina_bilateral_continuous.sonata.json`
- `scripts/run_emnist_retina_bilateral_continuous.sh`
- `README.md`
