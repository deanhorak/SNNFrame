# Project State - 2026-04-06

This document captures the current project state so a new conversation can resume work without re-discovering recent context.

## Repository Snapshot

- Branch: `ContinuousLearning`
- Date: `2026-04-06`
- Stable CIFAR baseline reference commit: `7eef39b` (`Promote CIFAR hybrid edge retina path`)
- Stable branch intent:
  - protect the bilateral Retina mainline
  - keep failed experiment tracks documented instead of promoted
  - require separability and propagation diagnostics before any new architecture change

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
- sampled gate: `33.80%` (`338/1000`)
  - log: `build/cifar10_retina_continuous_replay_gate_delay0_200_1000.log`
- larger sampled benchmark: `39.44%` (`1972/5000`)
  - log: `build/cifar10_retina_continuous_replay_delay0_1000_5000.log`
- confirmatory rerun with `seed=43`: `39.04%` (`1952/5000`)
  - log: `build/cifar10_retina_continuous_replay_delay0_1000_5000_confirm_seed43.log`

This confirmed immediate-replay result is now the CIFAR benchmark to beat.

Latest closed CIFAR follow-up:
- bounded `g10` sensory triplet/BCM plasticity follow-up on the promoted immediate-replay surface:
  - implementation: added an opt-in sensory-layer-only feature-gain rule in `src/adapters/RetinaAdapter.cpp` and `include/snnfw/adapters/RetinaAdapter.h`, with per-feature fast/slow triplet traces, a BCM-style sliding threshold, and training-only gain updates that freeze for evaluation; added g10-only CLI overrides in `experiments/retina_classification.cpp`; classifier-space triplet/voltage/BCM, replay, fusion, `g9`, and `dog_g9` were left untouched
  - strict same-build protected audit reference:
    - config: `configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json`
    - smoke: `32.00%` (`64/200`)
    - log: `build/cifar10_sensory_triplet_bcm_baseline_flow_audit_20_200.log`
    - summary: `build/flow_audit_sensory_triplet_bcm_baseline_testing_summary.json`
  - sensory triplet/BCM probe:
    - config: `configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json` plus CLI overrides `--g10-sensory-triplet-bcm-enabled --g10-sensory-triplet-bcm-learning-rate 0.015 --g10-sensory-triplet-bcm-ltp 0.12 --g10-sensory-triplet-bcm-ltd 0.04 --g10-sensory-triplet-fast-decay 0.80 --g10-sensory-triplet-slow-decay 0.97 --g10-sensory-bcm-threshold-decay 0.985 --g10-sensory-bcm-target-activation 0.08`
    - smoke: `32.50%` (`65/200`)
    - log: `build/cifar10_sensory_triplet_bcm_flow_audit_20_200.log`
    - summary: `build/flow_audit_sensory_triplet_bcm_testing_summary.json`
  - status: mixed and not promotable under the strict audit gate; there is no justified `50/500` follow-up for this bounded setting
  - read: this is the first recent upstream/local-plasticity probe to clear the protected smoke headline, but the support metrics do not justify scaling. Initial accuracy improved (`29.50%` vs `28.50%`), final smoke improved by one sample (`65/200` vs `64/200`), both hemisphere `topk_purity` scores improved (`left 0.186667` vs `0.184444`, `right 0.187778` vs `0.185556`), and audited `g10` orientation L2 increased (`left 0.920339` vs `0.913151`, `right 0.920196` vs `0.912764`). However both audited `g10` post-margins worsened (`left -0.00647548` vs `-0.00592188`, `right -0.00665063` vs `-0.00593531`), `interaction_centroid_accuracy` dropped from `29.0%` to `25.5%`, `confidence_concat_centroid_accuracy` dropped from `30.0%` to `26.5%`, and replay success fell from `7` to `6`. Treat this as a weak signal that sensory-layer gain plasticity can move the surface, not as a candidate to scale without a stronger pass criterion
- bounded `g10` burst/tonic LGN relay follow-up on the promoted immediate-replay surface:
  - implementation: added an opt-in local-salience burst/tonic mode switch to the existing `lgn_relay_enabled` center-surround path in `src/adapters/RetinaAdapter.cpp` and `include/snnfw/adapters/RetinaAdapter.h`, plus g10-only CLI overrides in `experiments/retina_classification.cpp`; the protected config was not changed
  - strict same-build protected audit reference:
    - config: `configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json`
    - smoke: `32.00%` (`64/200`)
    - log: `build/cifar10_burst_tonic_lgn_baseline_flow_audit_20_200.log`
    - summary: `build/flow_audit_burst_tonic_lgn_baseline_testing_summary.json`
  - burst/tonic probe:
    - config: `configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json` plus CLI overrides `--g10-lgn-burst-tonic-enabled --g10-lgn-burst-threshold 0.04 --g10-lgn-burst-extra-strength 0.12 --g10-lgn-burst-slope 8.0 --g10-lgn-burst-neuromodulator 1.0`
    - smoke: `31.50%` (`63/200`)
    - log: `build/cifar10_burst_tonic_lgn_flow_audit_20_200.log`
    - summary: `build/flow_audit_burst_tonic_lgn_testing_summary.json`
  - status: closed under the strict gate; there is no justified `50/500` follow-up for this local-salience burst/tonic relay policy
  - read: the probe was not a raw-drive collapse and even improved initial accuracy (`30.00%` vs `28.50%`) plus left hemisphere `topk_purity` (`0.186111` vs `0.184444`). But it still failed the scale criteria: final smoke fell to `31.50%`, stage-1 view accuracy fell from `26.50%`/`25.00%` to `24.50%`/`24.50%`, `interaction_centroid_accuracy` dropped from `29.0%` to `24.5%`, `confidence_concat_centroid_accuracy` dropped from `30.0%` to `26.0%`, right hemisphere `topk_purity` fell from `0.185556` to `0.181667`, and replay success fell from `7` to `3`
- bounded guided training-time saccade follow-up on the promoted immediate-replay surface:
  - implementation: added an opt-in guided fixation selector in `experiments/retina_classification.cpp`, enabled only by CLI/JSON flags, that scores candidate fixation offsets with the existing `g10` orientation/auxiliary energy and selects fixation centers with a small inhibition-of-return penalty; the protected config was not changed
  - strict same-build protected audit reference:
    - config: `configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json`
    - smoke: `32.00%` (`64/200`)
    - log: `build/cifar10_guided_saccades_baseline_flow_audit_20_200.log`
    - summary: `build/flow_audit_guided_saccades_baseline_testing_summary.json`
  - guided-saccade probe:
    - config: `configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json` plus CLI overrides `--guided-saccades-enabled --guided-saccade-fixations 4 --guided-saccade-candidate-peaks 8 --guided-saccade-jitter-px 0.45 --guided-saccade-zoom 1.03 --guided-saccade-ior-strength 0.35`
    - smoke: `29.00%` (`58/200`)
    - log: `build/cifar10_guided_saccades_flow_audit_20_200.log`
    - summary: `build/flow_audit_guided_saccades_testing_summary.json`
  - status: closed immediately under the strict gate; there is no justified `50/500` follow-up
  - read: the probe did not collapse the local `g10` code, but it still failed the task objective. Left hemisphere `topk_purity` improved (`0.191111` vs `0.184444`), right `topk_purity` only tied (`0.185556` vs `0.185556`), both audited `g10` post-margins moved only slightly toward zero, and orientation energy was unchanged (`left 0.913151`, `right 0.912764`). The metrics that matter for scaling regressed: final smoke dropped from `32.00%` to `29.00%`, stage-1 view accuracy dropped from `26.50%`/`25.00%` to `25.50%`/`23.50%`, `interaction_centroid_accuracy` dropped from `29.0%` to `27.0%`, and `confidence_concat_centroid_accuracy` dropped from `30.0%` to `27.0%`
- bounded `g10` contour-support auxiliary follow-up on the promoted immediate-replay surface:
  - implementation: added an opt-in `contour_support_bank` auxiliary mode in `src/adapters/RetinaAdapter.cpp` and `include/snnfw/adapters/RetinaAdapter.h`, deriving six same-dimensional `g10` channels for collinear continuation, cocircular support, end-stopping, junctionness, and signed border evidence after the stronger `orientation_energy + complex_cell_enabled` path; enabled it only on `left_retina_g10` and `right_retina_g10` in `configs/cifar10_retina_bilateral_natural_features_contour_support_experimental.sonata.json`
  - strict same-build protected audit reference:
    - config: `configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json`
    - smoke: `32.00%` (`64/200`)
    - log: `build/cifar10_retina_baseline_flow_audit_for_contour_support_20_200.log`
    - summary: `build/flow_audit_baseline_contour_support_smoke_testing_summary.json`
  - valid strict probe:
    - config: `configs/cifar10_retina_bilateral_natural_features_contour_support_experimental.sonata.json`
    - smoke: `30.00%` (`60/200`)
    - log: `build/cifar10_retina_contour_support_flow_audit_fix2_20_200.log`
    - summary: `build/flow_audit_contour_support_smoke_fix2_testing_summary.json`
  - status: closed immediately under the strict gate; there is no justified `50/500` follow-up
  - read: this was not a total miss internally. The valid probe improved both hemisphere purity scores (`left 0.186111` vs `0.184444`, `right 0.187778` vs `0.185556`) and made both audited `g10` post-margins materially less negative (`left -0.00287558` vs `-0.00592188`, `right -0.00301946` vs `-0.00593531`). But it still failed the actual pass criteria that matter end-to-end: smoke only reached `30.00%`, below the required `>32.0%`, and `interaction_centroid_accuracy` slipped to `28.5%`, below the required `29.0%`, even though `confidence_concat_centroid_accuracy` held at `30.0%`
- bounded `g10` complex-cell pooling + divisive-normalization follow-up on the promoted immediate-replay surface:
  - implementation: added an opt-in same-dimensional `complex_cell_enabled` stage in `src/adapters/RetinaAdapter.cpp` and `include/snnfw/adapters/RetinaAdapter.h` that pools `g10` orientation responses across pooled/subfield blocks and applies orientation-wise divisive normalization before the existing band selection, then enabled it only on `left_retina_g10` and `right_retina_g10` in `configs/cifar10_retina_bilateral_natural_features_complex_cell_experimental.sonata.json` and the milder `configs/cifar10_retina_bilateral_natural_features_complex_cell_tune1_experimental.sonata.json`
  - fresh protected smoke on the same build:
    - config: `configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json`
    - smoke: `32.00%` (`64/200`)
    - log: `build/cifar10_retina_baseline_smoke_20_200_rerun_for_complex_cell.log`
  - bounded default probe:
    - config: `configs/cifar10_retina_bilateral_natural_features_complex_cell_experimental.sonata.json`
    - smoke: `31.50%` (`63/200`)
    - log: `build/cifar10_retina_complex_cell_smoke_20_200.log`
  - milder retune:
    - config: `configs/cifar10_retina_bilateral_natural_features_complex_cell_tune1_experimental.sonata.json`
    - smoke: `29.50%` (`59/200`)
    - log: `build/cifar10_retina_complex_cell_tune1_smoke_20_200.log`
  - audit follow-up on the best probe:
    - config: `configs/cifar10_retina_bilateral_natural_features_complex_cell_experimental.sonata.json`
    - smoke: `31.50%` (`63/200`)
    - log: `build/cifar10_retina_complex_cell_flow_audit_smoke_20_200.log`
    - summary: `build/flow_audit_complex_cell_smoke_testing_summary.json`
  - status: closed for promotion on the current CIFAR surface; the best probe missed the protected smoke, so there is no justified `50/500` gate
  - read: this was not a pure upstream failure. The default probe improved `g10` branch-local separability and hemisphere combined centroid proxies relative to the protected smoke, and the audit showed less-negative `g10` margins than the protected audited smoke (`left post_margin -0.00469` vs `-0.00592`, `right -0.00492` vs `-0.00594`). But the current end-to-end surface still lost the task objective: stage-1 view accuracy fell to `23.50%`/`23.00%`, the audited fusion alternatives worsened (`interaction 27.5%` vs `29.0%`, `confidence-concat 27.0%` vs `30.0%`), and the probe finished below the protected `32.00%` smoke
- bounded `g10` raw quadrature-Gabor feature-code follow-up on the promoted immediate-replay surface:
  - implementation: added an opt-in `quadrature_gabor` edge operator in `src/features/QuadratureGaborOperator.cpp` and `include/snnfw/features/QuadratureGaborOperator.h`, then enabled it only on `left_retina_g10` and `right_retina_g10` while leaving `g9`, `dog_g9`, replay, and corpus-callosum fusion unchanged
  - fresh protected smoke on the same build:
    - config: `configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json`
    - smoke: `32.00%` (`64/200`)
    - log: `build/cifar10_retina_baseline_smoke_20_200_rerun_for_quadrature_gabor.log`
  - bounded default probe:
    - config: `configs/cifar10_retina_bilateral_natural_features_quadrature_gabor_experimental.sonata.json`
    - smoke: `22.50%` (`45/200`)
    - log: `build/cifar10_retina_quadrature_gabor_smoke_20_200.log`
  - permissive retune:
    - config: `configs/cifar10_retina_bilateral_natural_features_quadrature_gabor_tune1_experimental.sonata.json`
    - smoke: `22.50%` (`45/200`)
    - log: `build/cifar10_retina_quadrature_gabor_tune1_smoke_20_200.log`
  - status: closed for promotion on the current CIFAR surface; both bounded probes lost badly to the protected immediate-replay smoke, so there is no justified `50/500` gate
  - read: unlike the prior orientation-energy probe, this was a direct upstream collapse. On the current `5x5` `g10` edge-analysis patches, the quadrature-Gabor code drove `g10` orientation support to effectively zero after thresholding (`g10/orientation raw_active=0.00%` on both hemispheres), left the branch dominated by the auxiliary appearance bank, dropped stage-1 view accuracy to `18.50%` and `21.00%`, and pulled fusion centroid accuracy down to `20.00%`
- bounded `g10` raw orientation-energy feature-code follow-up on the promoted immediate-replay surface:
  - implementation: added an opt-in `orientation_energy` edge operator in `src/features/OrientationEnergyOperator.cpp` and `include/snnfw/features/OrientationEnergyOperator.h`, then enabled it only on `left_retina_g10` and `right_retina_g10` while leaving `g9`, `dog_g9`, replay, and corpus-callosum fusion unchanged
  - fresh protected smoke on the same build:
    - config: `configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json`
    - smoke: `32.00%` (`64/200`)
    - log: `build/cifar10_retina_baseline_smoke_20_200_rerun_for_orientation_energy.log`
  - bounded default probe:
    - config: `configs/cifar10_retina_bilateral_natural_features_orientation_energy_experimental.sonata.json`
    - smoke: `30.00%` (`60/200`)
    - log: `build/cifar10_retina_orientation_energy_smoke_20_200.log`
  - sharper retune:
    - config: `configs/cifar10_retina_bilateral_natural_features_orientation_energy_tune1_experimental.sonata.json`
    - smoke: `30.00%` (`60/200`)
    - log: `build/cifar10_retina_orientation_energy_tune1_smoke_20_200.log`
  - status: closed for promotion on the current CIFAR surface; both bounded probes lost to the protected immediate-replay smoke, so there is no justified `50/500` gate
  - read: this was not a pure upstream failure. The raw `g10` branch improved materially, with left/right `g10` centroid accuracy moving from `21.00%` and `18.00%` on the protected smoke to `26.00%` and `27.00%` on the default probe, and the hemisphere combined centroid proxies rising from `30.00%`/`27.00%` to `37.50%`/`37.00%`. But the current interaction-fusion surface did not cash that out into task accuracy: stage-1 view accuracy only rose to `27.50%`/`27.50%`, fusion centroid accuracy only reached `30.00%`, and final post-correction accuracy still stayed below the protected smoke
- bounded magnocellular/parvocellular-like LGN relay split on the promoted immediate-replay surface:
  - implementation: opt-in dual-relay band-image construction in `src/adapters/RetinaAdapter.cpp`, enabled only on `left_retina_g10` and `right_retina_g10` via `lgn_parallel_relay_enabled`, with coarse achromatic magno-like relay images blended into the existing low-frequency `g10` path and a finer parvo-like relay feeding the higher-detail and auxiliary inputs while leaving the protected baseline config unchanged
  - fresh protected smoke on the same build:
    - config: `configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json`
    - smoke: `32.00%` (`64/200`)
    - log: `build/cifar10_retina_baseline_smoke_20_200_rerun_for_magno_parvo.log`
  - bounded default split:
    - config: `configs/cifar10_retina_bilateral_natural_features_magno_parvo_lgn_experimental.sonata.json`
    - smoke: `23.50%` (`47/200`)
    - log: `build/cifar10_retina_magno_parvo_lgn_smoke_20_200.log`
  - conservative retune:
    - config: `configs/cifar10_retina_bilateral_natural_features_magno_parvo_lgn_tune1_experimental.sonata.json`
    - smoke: `27.00%` (`54/200`)
    - log: `build/cifar10_retina_magno_parvo_lgn_tune1_smoke_20_200.log`
  - status: closed for promotion on the current CIFAR surface; both bounded probes lost clearly to the protected immediate-replay smoke, so there is no justified `50/500` gate
  - read: the split kept `g10` orientation alive but weakened it; the default run drove `raw_active` for `g10` down into the mid-`50%` range and cut stage-1 view accuracy to `19.00%` and `21.50%`, while the conservative retune only recovered to `21.00%` per hemisphere view and still lost badly at fusion
- bounded `g10`-only fixation-aware stage-1 memory follow-up on the promoted immediate-replay surface:
  - implementation: opt-in stage-1 fixation-memory aggregation in `experiments/retina_classification.cpp`, enabled only on `left_retina_g10` and `right_retina_g10` via `stage1_fixation_memory_mode = mean_max_summary` while leaving `g9`, `dog_g9`, replay, and corpus-callosum fusion unchanged
  - fresh protected control gate with `flow_audit`:
    - config: `configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json`
    - gate: `33.70%` (`337/1000`)
    - log: `build/flow_audit_baseline_gate_200_1000_rerun.log`
    - summary: `build/flow_audit_baseline_gate_200_1000_rerun_testing_summary.json`
  - bounded fixation-aware probe:
    - config: `configs/cifar10_retina_bilateral_natural_features_g10_fixation_memory_experimental.sonata.json`
    - gate: `33.80%` (`338/1000`)
    - log: `build/flow_audit_g10_fixation_memory_gate_200_1000.log`
    - summary: `build/flow_audit_g10_fixation_memory_gate_200_1000_testing_summary.json`
  - status: closed for promotion on the current CIFAR surface; it did not clear the required audit gate, so there is no justified `1000/5000` run
  - read: the probe materially improved hemisphere neighbor purity (`0.237889 -> 0.275667` left, `0.239556 -> 0.274000` right), but it slightly worsened both `g10` post-normalization margins (`-0.00387905 -> -0.00389373` left, `-0.00376781 -> -0.00378284` right) and weakened the offline fusion proxies (`interaction` centroid accuracy `35.9 -> 33.3`, `confidence-concat` `35.7 -> 33.4`), so the remaining ceiling still looks like raw `g10` feature quality rather than fixation averaging
- bounded explicit late-fused `ON/OFF + luminance` `g10` branch split on the promoted immediate-replay surface:
  - implementation: opt-in luminance-branch gating in `src/adapters/RetinaAdapter.cpp`, with each baseline `g10` branch replaced by separate `ON`-dominant, `OFF`-dominant, and achromatic-luminance adapters using `luminance_stream_bank` and their own late-fused hierarchy paths while keeping the protected baseline config unchanged
  - bounded default split:
    - config: `configs/cifar10_retina_bilateral_natural_features_onoff_luminance_experimental.sonata.json`
    - smoke: `27.50%` (`55/200`)
    - log: `build/cifar10_retina_onoff_luminance_smoke_20_200_rerun.log`
  - status: closed for promotion on the current CIFAR surface; it lost clearly to the protected immediate-replay smoke (`32.50%`)
  - read: the explicit branches were real but weak, with left-side centroid accuracy only `22.50%` (`ON`), `21.50%` (`OFF`), and `21.00%` (luminance), right-side centroid accuracy only `18.50%`, `19.00%`, and `18.00%`, and fusion centroid accuracy collapsing to `22.50%`
- bounded explicit transient/sustained `g10` branch split on the promoted immediate-replay surface:
  - implementation: opt-in temporal-stream branch gating in `src/adapters/RetinaAdapter.cpp`, with each baseline `g10` branch replaced by separate transient and sustained adapters using `appearance_stream_bank` and their own late-fused hierarchy paths while keeping the protected baseline config unchanged
  - bounded default split:
    - config: `configs/cifar10_retina_bilateral_natural_features_transient_sustained_experimental.sonata.json`
    - smoke: `29.50%` (`59/200`)
    - log: `build/cifar10_retina_transient_sustained_smoke_20_200.log`
  - status: closed for promotion on the current CIFAR surface; it lost clearly to the protected immediate-replay smoke (`32.50%`)
  - read: the explicit temporal branches stayed alive, but they did not improve usable stage-1 evidence and the fused representation degraded; `left_retina_g10_transient` and `left_retina_g10_sustained` only reached `20.50%` and `20.00%` centroid accuracy, the right-side pair only `18.00%` and `18.00%`, and fusion centroid accuracy collapsed to `22.50%`
- bounded explicit `g10` foveal/peripheral branch split on the promoted immediate-replay surface:
  - implementation: opt-in region-masked branch support in `src/adapters/RetinaAdapter.cpp`, with each baseline `g10` branch replaced by separate foveal and peripheral adapters with independent attach paths and late fusion while keeping the protected baseline config unchanged
  - bounded default split:
    - config: `configs/cifar10_retina_bilateral_natural_features_foveal_peripheral_experimental.sonata.json`
    - smoke: `32.00%` (`64/200`)
    - log: `build/cifar10_retina_foveal_peripheral_smoke_20_200.log`
  - milder retune:
    - config: `configs/cifar10_retina_bilateral_natural_features_foveal_peripheral_tune1_experimental.sonata.json`
    - smoke: `31.00%` (`62/200`)
    - log: `build/cifar10_retina_foveal_peripheral_tune1_smoke_20_200.log`
  - status: closed for promotion on the current CIFAR surface; even the better split stayed below the protected immediate-replay smoke (`32.50%`)
  - read: the explicit split was real and branch-local, but dividing the current `g10` geography into central and peripheral sub-branches reduced usable separability instead of producing a better late-fused upstream code
- bounded temporal coarse-to-fine transient/sustained follow-up on the promoted immediate-replay surface:
  - implementation: opt-in `g10`-only dual-pass temporal arbitration in `src/adapters/RetinaAdapter.cpp`, using `appearance_stream_bank` auxiliaries to support low-frequency coarse drive and sustained fine-detail drive while keeping the protected baseline config unchanged
  - bounded selective `g10` probe:
    - config: `configs/cifar10_retina_bilateral_natural_features_temporal_coarse_fine_experimental.sonata.json`
    - smoke: `31.00%` (`62/200`)
    - log: `build/cifar10_retina_temporal_coarse_fine_smoke_20_200.log`
  - status: closed for promotion on the current CIFAR surface; the corrected smoke only tied the protected immediate-replay smoke (`32.50%`), and parity is not enough to spend a `50/500` gate
  - read: the temporal gate stayed alive, but it did not add enough usable signal to beat the protected smoke; the raw testing line slipped to `31.00%`, while `left_retina_g10` stayed at `21.50%` centroid accuracy and `right_retina_g10` stayed at `18.00%`
- bounded eccentricity-dependent retinal sampling / cortical-magnification follow-up on the promoted immediate-replay surface:
  - implementation: opt-in nonuniform receptive-field sampling in `src/adapters/RetinaAdapter.cpp`, enabled only on the two `g10` branches while keeping the protected baseline config unchanged
  - default selective `g10` probe:
    - config: `configs/cifar10_retina_bilateral_natural_features_eccentricity_sampling_experimental.sonata.json`
    - smoke: `28.50%` (`57/200`)
    - log: `build/cifar10_retina_eccentricity_sampling_smoke_20_200.log`
  - milder retune:
    - config: `configs/cifar10_retina_bilateral_natural_features_eccentricity_sampling_tune1_experimental.sonata.json`
    - smoke: `28.50%` (`57/200`)
    - log: `build/cifar10_retina_eccentricity_sampling_tune1_smoke_20_200.log`
  - status: closed for promotion on the current CIFAR surface; both bounded settings lost clearly to the protected immediate-replay smoke (`32.50%`)
  - read: the current `g10` representation is fragile to spatial remapping; this same-dimensional magnification probe reduced `g10` separability instead of improving useful foveal specialization
- bounded V2-like contextual grouping follow-up on the promoted immediate-replay surface:
  - implementation: opt-in per-retina contextual contour grouping, surround suppression, coarse-to-fine bias, and internal border-ownership competition applied before per-branch normalization in `experiments/retina_classification.cpp`
  - default selective `g10` probe:
    - config: `configs/cifar10_retina_bilateral_natural_features_contextual_grouping_experimental.sonata.json`
    - smoke: `30.50%` (`61/200`)
    - log: `build/cifar10_retina_contextual_grouping_smoke_20_200.log`
  - milder retune:
    - config: `configs/cifar10_retina_bilateral_natural_features_contextual_grouping_tune1_experimental.sonata.json`
    - smoke: `30.00%` (`60/200`)
    - log: `build/cifar10_retina_contextual_grouping_tune1_smoke_20_200.log`
  - status: closed for promotion on the current CIFAR surface; both bounded settings lost to the protected immediate-replay smoke (`32.50%`)
  - read: this same-dimensional modulation of the existing `g10` map hurts `g10` branch separability instead of building a useful proto-object representation
- bounded BCM-style metaplasticity follow-up on the promoted immediate-replay surface:
  - implementation: config-gated adaptation of the existing reward-driven centroid/weight/exemplar path in `experiments/retina_classification.cpp`, tested via CLI overrides on `configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json`
  - bounded default setting:
    - `online_bcm_metaplasticity_ltp = 0.45`
    - `online_bcm_metaplasticity_ltd = 0.35`
    - `online_bcm_metaplasticity_decay = 0.96`
    - `online_bcm_metaplasticity_target = 0.28`
    - smoke: `32.50%` (`65/200`)
    - log: `build/cifar10_retina_bcm_metaplasticity_smoke_20_200.log`
  - higher-LTP retune:
    - `online_bcm_metaplasticity_ltp = 0.65`
    - `online_bcm_metaplasticity_ltd = 0.20`
    - `online_bcm_metaplasticity_decay = 0.94`
    - `online_bcm_metaplasticity_target = 0.24`
    - smoke: `32.00%` (`64/200`)
    - log: `build/cifar10_retina_bcm_metaplasticity_smoke_tune1_20_200.log`
  - moderate retune:
    - `online_bcm_metaplasticity_ltp = 0.55`
    - `online_bcm_metaplasticity_ltd = 0.25`
    - `online_bcm_metaplasticity_decay = 0.97`
    - `online_bcm_metaplasticity_target = 0.26`
    - smoke: `32.50%` (`65/200`)
    - log: `build/cifar10_retina_bcm_metaplasticity_smoke_tune2_20_200.log`
  - status: closed for promotion on the current CIFAR surface; the best settings only tied the protected immediate-replay smoke (`32.50%`), and the project rule is to not spend a `50/500` gate on parity alone
  - cleanup: the opt-in BCM code path was removed after evaluation so the working surface stays aligned with the protected immediate-replay baseline
- bounded voltage-dependent local-plasticity follow-up on the promoted immediate-replay surface:
  - implementation: config-gated path in `experiments/retina_classification.cpp`, tested via CLI overrides on `configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json`
  - default bounded setting:
    - `online_voltage_plasticity_gain = 0.06`
    - `online_voltage_plasticity_ltp = 0.14`
    - `online_voltage_plasticity_ltd = 0.08`
    - `online_voltage_plasticity_decay = 0.94`
    - `online_voltage_plasticity_threshold = 0.30`
    - smoke: `31.00%` (`62/200`)
    - log: `build/cifar10_retina_voltage_plasticity_smoke_20_200.log`
  - milder bounded setting:
    - `online_voltage_plasticity_gain = 0.04`
    - `online_voltage_plasticity_ltp = 0.10`
    - `online_voltage_plasticity_ltd = 0.05`
    - `online_voltage_plasticity_decay = 0.96`
    - `online_voltage_plasticity_threshold = 0.24`
    - smoke: `31.50%` (`63/200`)
    - log: `build/cifar10_retina_voltage_plasticity_smoke_tune1_20_200.log`
  - permissive low-LTD retune:
    - `online_voltage_plasticity_gain = 0.08`
    - `online_voltage_plasticity_ltp = 0.12`
    - `online_voltage_plasticity_ltd = 0.03`
    - `online_voltage_plasticity_decay = 0.92`
    - `online_voltage_plasticity_threshold = 0.22`
    - smoke: `31.00%` (`62/200`)
    - log: `build/cifar10_retina_voltage_plasticity_smoke_tune2_20_200.log`
  - status: closed; no voltage-dependent setting cleared the protected immediate-replay smoke (`32.50%`), so the follow-up did not justify a `50/500` gate
- bounded replay-parameter sweep on the promoted immediate-replay surface:
  - queue-capacity sweep at `50/class, 500 test`:
    - `128`: `31.40%` (`157/500`)
    - log: `build/cifar10_retina_replay_queue128_gate_50_500.log`
    - `256`: `31.60%` (`158/500`)
    - log: `build/cifar10_retina_replay_queue256_gate_50_500.log`
    - `512`: `31.60%` (`158/500`)
    - log: `build/cifar10_retina_replay_queue512_gate_50_500.log`
  - reward-gain sweep at `50/class, 500 test`:
    - `online_positive_reward_gain = 0.35`, `online_negative_reward_gain = 1.1`: `31.60%` (`158/500`)
    - log: `build/cifar10_retina_reward_pos035_neg110_gate_50_500.log`
    - `online_positive_reward_gain = 0.65`, `online_negative_reward_gain = 1.1`: `31.60%` (`158/500`)
    - log: `build/cifar10_retina_reward_pos065_neg110_gate_50_500.log`
    - `online_positive_reward_gain = 0.5`, `online_negative_reward_gain = 0.95`: `31.60%` (`158/500`)
    - log: `build/cifar10_retina_reward_pos050_neg095_gate_50_500.log`
    - `online_positive_reward_gain = 0.5`, `online_negative_reward_gain = 1.25`: `31.60%` (`158/500`)
    - log: `build/cifar10_retina_reward_pos050_neg125_gate_50_500.log`
- config: `configs/cifar10_retina_bilateral_natural_features_upstream_saccades_lgn_stream_experimental.sonata.json`
- sampled gate: `32.50%` (`325/1000`)
  - log: `build/cifar10_retina_upstream_saccades_lgn_stream_gate_200_1000.log`
- larger sampled benchmark: `38.24%` (`1912/5000`)
  - log: `build/cifar10_retina_upstream_saccades_lgn_stream_1000_5000.log`
- confirmatory rerun with `seed=43`: `37.68%` (`1884/5000`)
  - log: `build/cifar10_retina_upstream_saccades_lgn_stream_1000_5000_confirm_seed43.log`
- explicit `g10` luminance/`ON/OFF` follow-up:
  - config: `configs/cifar10_retina_bilateral_natural_features_upstream_saccades_lgn_luminance_stream_experimental.sonata.json`
  - smoke after fixing the auxiliary wiring bug: `28.50%` (`57/200`)
  - log: `build/cifar10_retina_upstream_luminance_stream_smoke_20_200_fix1.log`
  - `50/class, 500 test`: `28.80%` (`144/500`)
  - log: `build/cifar10_retina_upstream_luminance_stream_gate_50_500.log`
  - `200/class, 1000 test`: `32.10%` (`321/1000`)
  - log: `build/cifar10_retina_upstream_luminance_stream_gate_200_1000.log`
- training-only spike-domain augmentation follow-up:
  - config: `configs/cifar10_retina_bilateral_natural_features_training_augmentation_experimental.sonata.json`
  - default `2`-variant smoke: `27.00%` (`54/200`)
  - log: `build/cifar10_retina_training_augmentation_smoke_20_200.log`
  - milder `1`-variant smoke via CLI override:
    - `--training-augmentation-variants 1 --training-augmentation-shift-px 0.35 --training-augmentation-rotation-deg 1.25 --training-augmentation-noise-std 0.01`
    - `24.50%` (`49/200`)
    - log: `build/cifar10_retina_training_augmentation_smoke_tune1_20_200.log`
- curriculum plus review scheduling follow-up:
  - config: `configs/cifar10_retina_bilateral_natural_features_curriculum_review_experimental.sonata.json`
  - default curriculum plus `10%` review smoke: `26.00%` (`52/200`)
  - log: `build/cifar10_retina_curriculum_review_smoke_20_200.log`
  - curriculum plus `5%` review smoke via CLI override:
    - `--training-review-fraction 0.05`
    - `27.00%` (`54/200`)
    - log: `build/cifar10_retina_curriculum_review_smoke_tune1_20_200.log`
  - curriculum-only smoke via CLI override:
    - `--training-review-fraction 0`
    - `29.50%` (`59/200`)
    - log: `build/cifar10_retina_curriculum_only_smoke_20_200.log`
  - curriculum-only `50/class, 500 test` gate:
    - `28.40%` (`142/500`)
    - log: `build/cifar10_retina_curriculum_only_gate_50_500.log`

The continuous replay / eligibility follow-up is now promoted locally. Immediate replay (`delay=0`) cleared the `50/500`, `200/1000`, and `1000/5000` gates, and the confirmatory `seed=43` rerun also stayed above the old protected `38.16%` reference. Delayed replay variants (`2`, `4`, `8` steps) remain non-promotable and should be treated as closed on the current CIFAR surface. The first bounded queue-capacity and reward-gain sweeps on top of the promoted replay surface were neutral on the decisive `50/500` gate, so those scalar replay-parameter nudges should also be treated as exhausted for now. The bounded reward-gated STDP prototype follow-up improved the small gate but still missed the decisive `200/1000` gate (`33.60%` vs protected `33.80%`), so it is also not promotable. The bounded triplet-STDP follow-up was weaker still: two bounded settings both reached `32.00%` at smoke but only tied the protected `50/500` gate at `31.60%`. The bounded voltage-dependent local-plasticity follow-up failed earlier: three bounded settings only reached `31.00%`, `31.50%`, and `31.00%` at smoke. The bounded BCM-style metaplasticity follow-up was the strongest of the post-replay plasticity variants, but it still only tied the protected immediate-replay smoke at `32.50%` and did not earn a `50/500` gate under the repo's own parity rule. On the current CIFAR surface, those first-order classifier-space local-plasticity and metaplasticity add-ons should all be treated as closed for promotion. The older stream-bank, luminance/`ON/OFF`, augmentation, and curriculum/review follow-ups remain non-promotable for the reasons already documented, and the newer late-fused `ON/OFF + luminance` `g10` branch split also closes well below the protected smoke.

## Mainline Architecture

### Shared Bilateral Retina Scaffold

The working architecture across EMNIST, MNIST, and CIFAR remains:

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

The Retina path is domain-driven rather than EMNIST-hardcoded.

Implemented adapters:
- `emnist`
- `mnist`
- `cifar10`

Primary files:
- `include/snnfw/domain/VisualDomainAdapter.h`
- `src/domain/VisualDomainAdapter.cpp`

### CIFAR Promoted Path

The current best CIFAR path keeps the bilateral scaffold but uses a richer natural-image front end:

- `representation = features`
- color-aware natural-image appearance features
- `appearance_bank` auxiliary path on all branches
- constrained coarse normalized edge revival only on `left_retina_g10` and `right_retina_g10`
- `4` training-time saccadic fixations with mild jitter and zoom
- a mild LGN-like center-surround relay only on `left_retina_g10` and `right_retina_g10`
- online continuous-learning correction with immediate replay (`online_correction_repeats = 3`, replay queue `384`, delay `0`)
- no broad edge revival on `g9` or `DoG`

Promotion notes:
- the previous promoted CIFAR result was committed in `7eef39b` (`Promote CIFAR hybrid edge retina path`)
- the current protected local reference is the `training-time saccades + g10-only LGN + immediate replay` update and is not yet committed in this workspace

## Closed Experimental Track

### Stage-1 Shape/Surface Split

This track is now closed and should not be treated as active.

What it tried:
- keep the bilateral scaffold unchanged
- split stage-1 classifier input into:
  - `shape` stream = non-auxiliary features
  - `surface` stream = auxiliary appearance-like features
- normalize the streams separately and reweight them before hemisphere classification

Observed results:
- small gate:
  - config: `build/cifar10_retina_bilateral_natural_features_shape_surface.sonata.json`
  - log: `build/cifar10_retina_natural_features_shape_surface_200_1000.log`
  - result: `30.70%` (`307/1000`)
- full validation run:
  - log: `build/cifar10_retina_natural_features_shape_surface_1000_5000_final.log`
  - result: `35.80%` (`1790/5000`)
  - elapsed: `5730.87s`

Interpretation:
- the small gate gain was real but did not survive the benchmark that matters
- the full run finished after the large-run failure mode was debugged, so this is a valid rejection, not an unresolved experiment
- the track finished `0.80` points below the protected CIFAR baseline of `36.60%`

Cleanup state:
- the experiment is documented as a rabbit hole in `docs/RabbitHoles.md`
- the dedicated `shape_surface` path has been removed from the active benchmark harness in `experiments/retina_classification.cpp`
- the promoted CIFAR baseline remains unchanged

## Candidate Next Directions

These are unvalidated ideas only. They stay subordinate to the protected mainline and must clear diagnostics before any promotion discussion.

## Current CIFAR Promotion Trail

The winning CIFAR follow-up is now the immediate-replay promotion path built on top of `training-time saccades + g10-only LGN`.

- protected config: `configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json`
- frozen winning config: `configs/cifar10_retina_bilateral_natural_features_upstream_saccades_lgn_experimental.sonata.json`
- scope:
  - keep `g9` and `DogG9` on the promoted `appearance_bank` path
  - apply mild LGN relay only on `left_retina_g10` and `right_retina_g10`
  - enable `4` training-time saccades globally
  - enable online correction with immediate replay (`delay=0`)

Observed results:
- broad all-branch upstream bundle:
  - config: `configs/cifar10_retina_bilateral_natural_features_upstream_experimental.sonata.json`
  - small gate: `21.80%` (`109/500`)
  - log: `build/cifar10_retina_upstream_gate_50_500.log`
- same-sample comparison against the promoted baseline:
  - promoted baseline at `50/class, 500 test`: `28.00%` (`140/500`)
  - log: `build/cifar10_retina_baseline_gate_50_500.log`
  - selective upstream at `50/class, 500 test`: `29.20%` (`146/500`)
  - log: `build/cifar10_retina_upstream_selective_gate_50_500.log`
- larger selective gate:
  - `32.00%` (`320/1000`)
  - log: `build/cifar10_retina_upstream_selective_gate_200_1000.log`
- combined follow-up:
  - config: `configs/cifar10_retina_bilateral_natural_features_upstream_saccades_lgn_experimental.sonata.json`
  - `200/class, 1000 test`: `32.30%` (`323/1000`)
  - log: `build/cifar10_retina_upstream_saccades_lgn_gate_200_1000.log`
  - `1000/class, 5000 test`: `38.16%` (`1908/5000`)
  - log: `build/cifar10_retina_upstream_saccades_lgn_1000_5000.log`
- continuous replay follow-up on top of that promoted surface:
  - config: `configs/cifar10_retina_bilateral_natural_features_continuous_replay_experimental.sonata.json`
  - immediate replay smoke at `20/class, 200 test`: `32.50%` (`65/200`)
  - log: `build/cifar10_retina_continuous_replay_smoke_delay0_20_200.log`
  - immediate replay `50/class, 500 test`: `31.60%` (`158/500`)
  - log: `build/cifar10_retina_continuous_replay_gate_delay0_50_500.log`
  - immediate replay `200/class, 1000 test`: `33.80%` (`338/1000`)
  - log: `build/cifar10_retina_continuous_replay_gate_delay0_200_1000.log`
  - immediate replay `1000/class, 5000 test`: `39.44%` (`1972/5000`)
  - log: `build/cifar10_retina_continuous_replay_delay0_1000_5000.log`
  - confirmatory immediate-replay rerun with `seed=43`: `39.04%` (`1952/5000`)
  - log: `build/cifar10_retina_continuous_replay_delay0_1000_5000_confirm_seed43.log`
  - delayed replay smoke sweep:
    - `delay=2`: `32.00%` (`64/200`)
    - log: `build/cifar10_retina_continuous_replay_smoke_delay2_20_200.log`
    - `delay=4`: `31.50%` (`63/200`)
    - log: `build/cifar10_retina_continuous_replay_smoke_delay4_20_200.log`
    - `delay=8`: `31.50%` (`63/200`)
    - log: `build/cifar10_retina_continuous_replay_smoke_delay8_20_200.log`
- branch-selective stream-bank follow-up on top of that surface:
  - config: `configs/cifar10_retina_bilateral_natural_features_upstream_saccades_lgn_stream_experimental.sonata.json`
  - `20/class, 200 test`: `29.50%` (`59/200`)
  - log: `build/cifar10_retina_upstream_saccades_lgn_stream_smoke_20_200.log`
  - paired baseline at `50/class, 500 test`: `28.40%` (`142/500`)
  - log: `build/cifar10_retina_baseline_gate_50_500_current.log`
  - stream-bank follow-up at `50/class, 500 test`: `28.60%` (`143/500`)
  - log: `build/cifar10_retina_upstream_saccades_lgn_stream_gate_50_500.log`
  - `200/class, 1000 test`: `32.50%` (`325/1000`)
  - log: `build/cifar10_retina_upstream_saccades_lgn_stream_gate_200_1000.log`
  - `1000/class, 5000 test`: `38.24%` (`1912/5000`)
  - log: `build/cifar10_retina_upstream_saccades_lgn_stream_1000_5000.log`
  - confirmatory `1000/class, 5000 test` rerun with `seed=43`: `37.68%` (`1884/5000`)
  - log: `build/cifar10_retina_upstream_saccades_lgn_stream_1000_5000_confirm_seed43.log`
- explicit luminance plus sustained/transient `ON/OFF` split on `g10` only:
  - config: `configs/cifar10_retina_bilateral_natural_features_upstream_saccades_lgn_luminance_stream_experimental.sonata.json`
  - first smoke exposed a mode-wiring bug where the new auxiliary slice stayed zero; fixed locally in `src/adapters/RetinaAdapter.cpp`
  - fixed smoke at `20/class, 200 test`: `28.50%` (`57/200`)
  - log: `build/cifar10_retina_upstream_luminance_stream_smoke_20_200_fix1.log`
  - `50/class, 500 test`: `28.80%` (`144/500`)
  - log: `build/cifar10_retina_upstream_luminance_stream_gate_50_500.log`
  - `200/class, 1000 test`: `32.10%` (`321/1000`)
  - log: `build/cifar10_retina_upstream_luminance_stream_gate_200_1000.log`
- ablations against the promoted `200/class, 1000 test` gate:
  - promoted baseline: `30.00%` (`300/1000`)
  - log: `build/cifar10_retina_natural_features_promoted_200_1000.log`
  - `appearance_stream_bank only` on `g10`: `29.90%` (`299/1000`)
  - log: `build/cifar10_retina_upstream_appearance_stream_only_gate_200_1000.log`
  - `LGN only` on `g10`: `31.10%` (`311/1000`)
  - log: `build/cifar10_retina_upstream_lgn_only_gate_200_1000.log`
  - `homeostasis only` on `g10`: `30.30%` (`303/1000`)
  - log: `build/cifar10_retina_upstream_homeostasis_only_gate_200_1000.log`
  - `saccades only`: `31.90%` (`319/1000`)
  - log: `build/cifar10_retina_upstream_saccades_only_gate_200_1000.log`

Interpretation:
- the broad upstream bundle regressed when applied across every branch
- the narrowed `g10`-only upstream probe recovered and exceeded the historical promoted-path `200/1000` gate
- the first composition that cleared the larger sampled benchmark is `training-time saccades + g10-only LGN`
- on the standard large benchmark it finished `1.56` points above the previous protected `36.60%` reference
- adding `appearance_stream_bank` back only on `g10`, on top of the winning `saccades + g10-only LGN` surface, produced a second valid full-benchmark gain
- the stream-bank follow-up beat the protected local reference on the first sampled run, but failed confirmation on a second `1000/5000` run (`37.68%` vs the protected `38.16%`)
- replacing the `g10` auxiliary path with an explicit luminance plus sustained/transient `ON/OFF` split kept the new auxiliary slice alive (`raw_active` around `42.8%`) but still lost the decisive `200/1000` gate (`32.10%` vs `32.30%`)
- `g9` and `DogG9` orientation remain dead by design and are still contributing mainly through auxiliary appearance channels
- `g10` branch-level separability is weaker than the promoted path, so the current gain appears to come from better bilateral complementarity rather than stronger single-branch classifiers
- on the stream-bank follow-up, `g10` orientation stayed alive (`raw_active` around `68%`) and the full-benchmark lift appears to come from slightly better late fusion behavior rather than a dramatic stage-1 separability jump
- the stronger pattern now is that static `g10` auxiliary-bank variants can move small gates, but they are not yet producing stable benchmark gains over the protected baseline
- the ablation ordering is now clear:
  - `saccades only` carries most of the selective gain
  - `LGN only` adds a smaller but real lift
  - `homeostasis only` is marginal
  - `appearance_stream_bank only` does not help by itself
- the bounded replay/eligibility sweep then produced the first confirmed full-benchmark gain over that surface
- immediate replay is the only delayed-plasticity setting currently worth keeping; `delay=2`, `4`, and `8` already lose at smoke
- the immediate-replay path now defines the protected local CIFAR reference surface
- the stream-bank follow-up should stay unpromoted; the full-benchmark edge did not survive confirmation
- the luminance/`ON/OFF` follow-up should also stay unpromoted; it lost the decisive larger gate
- training-only spike-domain augmentation beyond the existing `4`-fixation saccade path should also stay unpromoted; both bounded smoke probes regressed below the protected `29.50%` reference and did not justify the required `50/500` gate
- curriculum plus review scheduling should also stay unpromoted; the review pass regressed immediately, and the best surviving curriculum-only ablation only tied the protected `50/500` gate instead of beating it

### Active Inference Probe

The first target-directed trans-saccadic inference probe is now implemented in
`experiments/retina_classification.cpp`, but it is not promotable.

What was added:
- inference-time `2-3` fixation support on top of the promoted CIFAR baseline
- saliency-ranked fixation candidates with inhibition-of-return
- early stop on bilateral uncertainty
- optional region-block remapping of Retina patterns into a common frame
- script passthrough so `scripts/run_cifar10_retina_bilateral_natural.sh` can forward extra CLI flags

Current sampled results:
- protected baseline at `20/class, 200 test`:
  - `29.50%` (`59/200`)
  - log: `build/cifar10_retina_baseline_smoke_20_200.log`
- active inference, wider `3`-fixation probe with remapping:
  - command flags:
    - `--active-inference-enabled --active-inference-fixations 3 --active-inference-shift-px 3.2 --active-inference-zoom 1.08 --active-inference-uncertainty-threshold 0.32 --active-inference-ior-strength 0.35`
  - `27.00%` (`54/200`)
  - log: `build/cifar10_retina_active_inference_smoke_v2_20_200.log`
- active inference, conservative `2`-fixation probe without remapping:
  - command flags:
    - `--active-inference-enabled --active-inference-fixations 2 --active-inference-remap-enabled 0 --active-inference-shift-px 1.6 --active-inference-zoom 1.05 --active-inference-uncertainty-threshold 0.55 --active-inference-ior-strength 0.25`
  - `27.50%` (`55/200`)
  - log: `build/cifar10_retina_active_inference_tune2_20_200.log`
- active inference, tile-saliency fixation policy with the same conservative settings:
  - `27.00%` (`54/200`)
  - log: `build/cifar10_retina_active_inference_tilepolicy_20_200.log`

Interpretation:
- the code path is real and end-to-end
- the original ring-offset policy regressed
- replacing it with a tile-saliency target selector still regressed on the same smoke gate
- do not spend the `200/class, 1000 test` budget on this path yet
- active inference is no longer the leading next move
- if it is revisited, it needs a materially different fixation policy or evidence-accumulation scheme, not more small policy sweeps

### 1. Upstream Retina / Pre-Cortical Changes

Highest-leverage candidates:
- parallel transient/sustained `ON/OFF` ganglion-style channels plus a dedicated luminance pathway
- a minimal LGN-like relay nucleus with center-surround sharpening and slow gain control
- dynamic saccadic sampling during training presentations using the existing saccade support

Why this is the most plausible next area:
- CIFAR still shows weak upstream separability outside the constrained `g10` edge revival
- several failed rabbit holes were downstream tuning on top of weak representation
- the orientation-dead diagnostics argue for fixing the pre-cortical code before touching fusion again

### 2. Cortical Refinements Only After Diagnostics Pass

Candidates:
- slow homeostatic plasticity on `L4` and `L2/3` excitatory populations
- one cautious `V2`-like convergent region only after verified stage-boundary propagation and separability gains

Rule:
- do not add another area until the new upstream signal is measurably alive and class-separable

### 3. Continuous-Learning / Plasticity Upgrades

Candidates:
- a materially different neuromodulator-gated local plasticity rule only if it is not another first-order classifier-space prototype/evidence nudge

Current read:
- the bounded CIFAR replay/eligibility sweep succeeded and confirmed, but only with immediate replay; delaying replay to `2`, `4`, or `8` steps weakened the result
- the bounded queue-capacity and reward-gain sweeps on top of that surface were neutral on the protected `50/500` gate
- the bounded reward-gated STDP prototype follow-up did improve the `50/500` gate (`32.00%` vs `31.60%`) but still missed the protected `200/1000` gate (`33.60%` vs `33.80%`)
- the bounded triplet-STDP follow-up did not improve on that: two bounded settings both reached `32.00%` at smoke but only tied the protected `50/500` gate at `31.60%`
- the bounded voltage-dependent local-plasticity follow-up failed earlier than either of those: three bounded settings only reached `31.00%`, `31.50%`, and `31.00%` at smoke versus the protected immediate-replay smoke of `32.50%`
- the bounded BCM-style metaplasticity follow-up improved the post-correction count relative to the protected smoke, but it still only tied the protected immediate-replay smoke at `32.50%` and therefore did not justify a `50/500` gate
- the correct next move is now only a materially different local plasticity rule, not more first-order replay scalar tuning or another classifier-space prototype/evidence or metaplastic gating add-on

Why these are attractive:
- they build on the strongest differentiator already working in the repo
- they do not require falling back into the decision-microcircuit rabbit holes

### 4. Low-Risk Training Protocol Changes

Candidates:
- scaling the current protected CIFAR baseline to `500-1000` examples/class with the existing multi-threaded path

Closed follow-up:
- on-the-fly spike-domain augmentation beyond the existing training-time saccades regressed immediately on the protected CIFAR surface and should stay closed unless there is a materially different augmentation policy
- curriculum scheduling plus a second low-rate review pass did not produce a protected-gate win; the review pass regressed and curriculum-only only tied the `50/500` reference

## Immediate Next Experiment Candidate

If a new conversation wants the next bounded step, the current best candidate is:

1. Do not replace the protected CIFAR baseline blindly.
2. Keep `training-time saccades + g10-only LGN + immediate replay` as the protected CIFAR reference surface.
3. Do not spend more CIFAR budget on delayed replay variants; `delay=2`, `4`, and `8` already lost to immediate replay at smoke.
4. Do not spend more CIFAR budget on simple queue-capacity or scalar reward-gain sweeps; they were neutral on the protected `50/500` gate.
5. Treat the bounded reward-gated STDP prototype follow-up as closed for promotion: it reached `32.00%` at `50/500` but only `33.60%` at `200/1000`, below the protected `33.80%`.
6. Treat the bounded triplet-STDP follow-up as closed for promotion: both bounded settings reached `32.00%` at smoke but only tied the protected `50/500` gate at `31.60%`.
7. Treat the bounded voltage-dependent local-plasticity follow-up as closed for promotion: three bounded settings only reached `31.00%`, `31.50%`, and `31.00%` at smoke, below the protected immediate-replay smoke of `32.50%`.
8. Treat the bounded BCM-style metaplasticity follow-up as closed for promotion on the current CIFAR surface: its best settings only tied the protected immediate-replay smoke at `32.50%`, and parity is not enough to spend a `50/500` gate.
9. There is no more justified CIFAR budget for first-order classifier-space prototype/evidence or metaplastic gating add-ons on this surface.
10. Treat the bounded contextual-grouping follow-up as closed for promotion on the current CIFAR surface: both bounded `g10`-only probes regressed immediately at smoke (`30.50%`, `30.00%`) against the protected `32.50%`.
11. Treat the bounded eccentricity-dependent retinal-sampling follow-up as closed for promotion on the current CIFAR surface: both bounded `g10`-only probes regressed immediately at smoke (`28.50%`, `28.50%`) against the protected `32.50%`.
12. Treat the bounded same-dimensional temporal coarse-to-fine follow-up as closed for promotion on the current CIFAR surface: its corrected smoke only tied the protected immediate-replay smoke at `32.50%`, while the raw testing line slipped to `31.00%`.
13. Treat the bounded explicit `g10` foveal/peripheral split as closed for promotion on the current CIFAR surface: its better smoke reached only `32.00%`, and the milder retune slipped to `31.00%`.
14. Treat the bounded explicit transient/sustained `g10` branch split as closed for promotion on the current CIFAR surface: the default split finished at only `29.50%` smoke and degraded fusion despite live branch activity.
15. Treat the bounded explicit late-fused `ON/OFF + luminance` `g10` branch split as closed for promotion on the current CIFAR surface: the clean rerun only reached `27.50%` smoke and fusion centroid accuracy collapsed to `22.50%` despite live sub-branches.
16. There is no justified next CIFAR experiment inside the current `g10` branch-splitting family on this protected surface.
17. Treat the bounded `g10`-only fixation-aware stage-1 memory follow-up as closed for promotion on the current CIFAR surface: the audited `200/1000` probe only edged the protected gate from `33.70%` to `33.80%`, but it failed the actual gate requirement by slightly worsening both `g10` post-normalization margins even while hemisphere `topk_purity` improved.
18. Treat the bounded magnocellular/parvocellular-like LGN relay split as closed for promotion on the current CIFAR surface: both the default probe (`23.50%`) and the conservative retune (`27.00%`) lost clearly to the protected `32.00%` smoke on the same build.
19. If CIFAR work continues, do not spend more budget on direct quadrature-Gabor/simple-cell swaps at the current `g10` patch size; the current `5x5` support is too weak for this operator family on this surface.
20. Treat the bounded neuromodulator-gated eligibility plus success-consolidation replay follow-up as closed for promotion on the current CIFAR surface: it improved the strict audited smoke from `32.00%` to `33.50%`, but both audited `g10` post-margins stayed flat and the same-build `50/500` gate regressed from `30.80%` to `30.40%`. Do not spend more CIFAR budget on more uncertainty/reward/delay reshuffles of the current replay path.
21. The later raw `g10` orientation-energy, quadrature-Gabor, contour-support, guided-saccade, burst/tonic relay, sensory triplet/BCM, prediction-error feedback, and neuromodulator replay probes are now all closed or non-promotable on this surface. If CIFAR work continues, the next candidate should be a materially different intermediate representation/stage boundary with explicit separability diagnostics, not another near-neighbor retune of the current `g10` map or another scalar decision-path or replay-policy reweighting rule.
22. Treat the bounded pooled branch-summary convergent-stage follow-up as closed for promotion on the current CIFAR surface: the first materially different hemisphere-internal stage boundary regressed badly at strict audited smoke (`24.50%` vs protected `32.00%`), collapsed stage-1 view accuracy into the mid-teens, and drove best-neighbor similarity near `0.99`. Do not spend more CIFAR budget on retuning pooled branch-summary conjunctions.
23. Treat the bounded topographic local-continuity/junction stage follow-up as closed for promotion on the current CIFAR surface: it preserved region order and avoided the worst pooled-summary collapse, but it still regressed badly at strict audited smoke (`25.00%` vs protected `32.00%`), weakened stage-1 view accuracy (`22.50%`/`17.00%`), and left best-neighbor similarity too high (`~0.974`). Do not spend more CIFAR budget on retuning this fixed local topographic transform.

## Recommended Resume Point

If a new conversation needs to pick this up, start here:

1. Do not modify the protected CIFAR baseline blindly.
2. Treat the promoted immediate-replay CIFAR result (`39.44%`, confirmed by `39.04%` with `seed=43`) as the protected CIFAR reference surface.
3. Treat delayed replay variants (`2`, `4`, `8` steps) as weaker than immediate replay on the current CIFAR surface.
4. Treat simple queue-capacity and scalar reward-gain sweeps as neutral on the current replay baseline unless there is a materially different interaction to test.
5. Treat the bounded reward-gated STDP prototype follow-up as closed for promotion purposes unless there is a materially different design than the current prototype-evidence path.
6. Treat `shape_surface`, inference-time active vision, the unconfirmed `g10` stream-bank comeback, the explicit `g10` luminance/`ON/OFF` split, the training-only augmentation follow-up, and the curriculum/review follow-up as closed for promotion purposes unless there is a materially new reason to revisit them.
7. Treat the bounded triplet-STDP follow-up as closed for promotion purposes unless there is a materially different design than the current classifier-space triplet-trace path.
8. Treat the bounded voltage-dependent local-plasticity follow-up as closed for promotion purposes unless there is a materially different design than the current classifier-space depolarization-gated prototype path.
9. Treat the bounded BCM-style metaplasticity follow-up as closed for promotion purposes unless there is a materially different design than the current classifier-space BCM-gated centroid/exemplar path.
10. Treat the bounded contextual-grouping follow-up as closed for promotion purposes unless there is a materially different representation than modulation of the existing `g10` map; the current `g10`-only grouping probe regressed immediately at smoke.
11. Treat the bounded eccentricity-dependent retinal-sampling follow-up as closed for promotion purposes unless there is a materially different representation than the current same-dimensional `g10` remap; the present magnification probe regressed immediately at smoke.
12. Treat the bounded same-dimensional temporal coarse-to-fine follow-up as closed for promotion purposes unless there is a materially different representation than the current `g10` map; the present dual-pass probe only tied the protected corrected smoke and did not justify a `50/500` gate.
13. Treat the bounded explicit `g10` foveal/peripheral split as closed for promotion purposes unless there is a materially different representation than the current `g10` map; the present central/peripheral branch split stayed below the protected corrected smoke and did not justify a `50/500` gate.
14. Treat the bounded explicit transient/sustained `g10` branch split as closed for promotion purposes unless there is a materially different representation than the current `g10` map; the present late-fused temporal split finished at only `29.50%` smoke and did not justify a `50/500` gate.
15. Treat the bounded explicit late-fused `ON/OFF + luminance` `g10` branch split as closed for promotion purposes unless there is a materially different pre-cortical dynamic than the current `g10` map; the clean rerun only reached `27.50%` smoke and fusion degraded badly.
16. Treat the bounded `g10`-only fixation-aware stage-1 memory follow-up as closed for promotion purposes unless there is a materially different representation than `mean`, `per-fixation exemplar`, or `mean + max` aggregation of the current `g10` code; the audited `200/1000` probe slightly improved final accuracy and hemisphere `topk_purity`, but it still worsened both `g10` post-normalization margins and weakened the fusion proxies.
17. Treat the bounded magnocellular/parvocellular-like LGN relay split as closed for promotion purposes unless there is a materially different relay computation than the current band-mixed coarse/fine blend; the default and conservative smoke probes both lost badly to the protected baseline.
18. Treat the bounded raw `g10` orientation-energy follow-up as closed for promotion purposes unless there is a materially different representation around it than the current gradient-histogram plus tensor-sharpening formulation; it improved branch-local `g10` centroid accuracy and hemisphere combined centroid proxies, but both bounded smokes still finished at only `30.00%`, below the protected `32.00%` smoke on the same build.
19. Treat the bounded raw `g10` quadrature-Gabor follow-up as closed for promotion purposes on the current patch geometry; both bounded smokes collapsed to `22.50%`, and `g10` orientation activity effectively vanished on the current `5x5` edge-analysis patches.
20. Stop spending time on additional CIFAR training-order tweaks, replay scalar nudges, more first-order classifier-space plasticity/metaplasticity nudges, more small retunes of the current contextual-grouping formulation, more same-dimensional spatial remapping of the current `g10` map, more same-dimensional temporal arbitration on the current `g10` map, more geography-only `g10` splits, more transient/sustained `g10` branch splits, more `ON/OFF + luminance` branch splits of the current map, more fixation-aggregation retunes of the current `g10` map, more band-mixed coarse/fine LGN relay retunes, more small retunes of the current orientation-energy/tensor-histogram `g10` code, or more direct quadrature-Gabor swaps on the current `g10` patch size before a materially different upstream representation clears the protected gates.
21. Treat the bounded operator-matched `g10` contour/border auxiliary follow-up as closed for promotion purposes on the current CIFAR surface: the valid strict smoke only reached `30.00%`, and `interaction_centroid_accuracy` slipped to `28.5%`, even though both hemisphere `topk_purity` scores and both audited `g10` post-margins improved. Do not spend more CIFAR budget on more same-dimensional `contour_support_bank` retunes of the current `g10` map.
22. Treat the bounded guided training-time saccade follow-up as closed for promotion purposes on the current CIFAR surface: `4` guided fixations chosen from `8` `g10`-energy-scored candidates regressed immediately at strict smoke (`29.00%` vs protected `32.00%`), fusion proxies regressed (`interaction 27.0%` vs `29.0%`, `confidence-concat 27.0%` vs `30.0%`), right hemisphere `topk_purity` only tied, and audited `g10` orientation energy was unchanged. Do not scale this selector to `8` or `48` fixations without a materially different policy.
23. Treat the bounded `g10` burst/tonic LGN relay follow-up as closed for promotion purposes on the current CIFAR surface: the local-salience mode switch improved initial accuracy (`30.00%` vs `28.50%`) but missed the final smoke gate (`31.50%` vs protected `32.00%`), weakened stage-1 view accuracy, degraded fusion proxies (`interaction 24.5%` vs `29.0%`, `confidence-concat 26.0%` vs `30.0%`), reduced right hemisphere `topk_purity`, and cut replay successes from `7` to `3`. Do not scale this relay policy without a materially different gating objective.
24. Treat the bounded `g10` sensory triplet/BCM plasticity follow-up as mixed but non-promotable on the current CIFAR surface: it improved the smoke headline by one sample (`32.50%` vs `32.00%`) and improved hemisphere `topk_purity`, but worsened both audited `g10` post-margins and degraded fusion proxies (`interaction 25.5%` vs `29.0%`, `confidence-concat 26.5%` vs `30.0%`). Do not spend a `50/500` gate on this exact sensory-gain rule unless a stricter repeat also preserves fusion separability.
25. Treat the bounded prediction-error-gated corpus-callosum feedback proxy as closed for promotion purposes on the current CIFAR surface: both the whole-hemisphere evidence scale and the targeted predicted-label expectation-suppression form tied the protected strict smoke at `32.00%`, with unchanged fusion alternatives, unchanged replay successes, unchanged hemisphere `topk_purity`, and unchanged audited `g10` margins/orientation energy. Do not spend a `50/500` gate on more scalar class-centroid prediction-error suppression in the decision path.
26. Treat the bounded neuromodulator-gated eligibility plus variable-delay replay consolidation follow-up as closed for promotion purposes on the current CIFAR surface: it slightly improved the strict audited smoke (`33.50%` vs `32.00%`) while leaving audited `g10` margins and hemisphere centroid margins effectively unchanged, then lost the direct same-build `50/500` gate (`30.40%` vs `30.80%`). Do not spend more CIFAR budget on more uncertainty/reward/delay retunes of the current replay path.
27. Treat the bounded pooled branch-summary convergent-stage follow-up as closed for promotion purposes on the current CIFAR surface: it kept the raw branch audits unchanged but collapsed the hemisphere classifier surface (`24.50%` smoke, left/right stage-1 `15.00%`/`16.50%`, `interaction_centroid_accuracy 21.5%`, `confidence-concat 20.0%`, `best_neighbor_similarity ~0.988`). Do not spend more CIFAR budget on more pooled-summary conjunction retunes of the current branch outputs.
28. Treat the bounded topographic local-continuity/junction stage follow-up as closed for promotion purposes on the current CIFAR surface: it was a stricter attempt to preserve spatial structure than Rabbit Hole #46, but it still failed the protected audited smoke (`25.00%`), degraded fusion proxies (`interaction 23.0%`, `confidence-concat 23.0%`), and kept best-neighbor similarity far too high (`~0.974`). Do not spend more CIFAR budget on more fixed local continuity/junction retunes of the current branch outputs.
29. Treat the bounded classifier-side figure-ground residual append as closed for promotion purposes on the current CIFAR surface: the same-build strict audited smoke collapsed from the protected `32.00%` control to `10.00%`, both hemispheres collapsed to a single class, `interaction_centroid_accuracy` fell from `29.0%` to `10.0%`, and both hemisphere `topk_purity` scores roughly halved. Do not spend more CIFAR budget on gain-retuning this direct raw-plus-figure-ground concatenation.
30. Treat the bounded same-dimensional figure-ground mask follow-up as closed for promotion purposes on the current CIFAR surface: it preserved the classifier space and slightly improved left hemisphere purity and nearest-neighbor separation, but it still lost the protected strict smoke (`31.00%` vs `32.00%`), weakened right stage-1 accuracy (`19.50%` vs `25.00%`), and degraded fusion confidence (`interaction 28.0%`, `confidence-concat 27.0%`). Do not spend a `50/500` gate on gain-retuning this exact raw-pattern mask.
31. Treat the bounded separate figure-ground object-memory vote follow-up as closed for promotion purposes on the current CIFAR surface: the bank was live (`200` figure-ground object-memory patterns per hemisphere) and right stage-1 view accuracy improved slightly (`25.00%` -> `26.50%`), but the same-build strict smoke still regressed from `32.00%` to `30.50%`, fusion proxies weakened (`interaction 28.5%`, `confidence-concat 28.0%`), and both hemisphere `topk_purity` plus best-neighbor similarity stayed unchanged. Do not spend a `50/500` gate on gain-retuning or keep-fraction retuning this exact separate vote.
32. Treat the first bounded recurrent sensory-state settling follow-up as not promotable on the current CIFAR surface: it moved both stage-1 storage and inference onto a settled multi-fixation state with figure-ground, continuity, and early callosal support inside the update, but the same-build strict audited smoke still regressed from `32.00%` to `31.00%`, stage-1 view accuracy weakened (`26.50%/25.00%` -> `24.50%/20.50%`), `confidence-concat` fusion regressed (`30.0%` -> `28.0%`), and best-neighbor similarity rose from about `0.888` to `0.909` in both hemispheres. Do not spend a `50/500` gate on this exact recurrent gain setting.
33. Treat the bounded recurrent state + population-centroid readout follow-up as not promotable on the current CIFAR surface: keeping the same recurrent settled state but swapping the hemisphere readout from exemplar matching to class centroids regressed the same-build strict audited smoke further, from `32.00%` to `29.50%`, and degraded both fusion proxies (`interaction 23.0%`, `confidence-concat 23.0%`) while best-neighbor similarity remained elevated near `0.909`. Do not spend a `50/500` gate on this exact centroid-readout formulation.
34. Treat the bounded recurrent state + object-state attractor readout follow-up as not promotable on the current CIFAR surface: keeping the same recurrent settled state but replacing the hemisphere readout with an explicit attractor-style object-state bank regressed the same-build strict audited smoke from `32.00%` to `27.50%`, dropped initial accuracy to `20.50%`, and degraded both fusion proxies (`interaction 18.0%`, `confidence-concat 18.5%`) while best-neighbor similarity again stayed elevated near `0.909`. Do not spend a `50/500` gate on this exact attractor-readout formulation.
35. If CIFAR work continues from here, the next step can no longer be another small auxiliary-bank retune on the current `g10` surface, another high-energy fixation selector over the same `g10` map, another local-salience center-surround relay gain tweak, another unconstrained sensory gain-plasticity rule that improves raw orientation while degrading fusion, another class-centroid confidence reweighting rule, another scalar replay-policy reweighting rule, another pooled branch-summary conjunction stage, another fixed local continuity/junction stage, another direct figure-ground append into the current classifier vector, another direct same-dimensional figure-ground mask over the whole hemisphere pattern, another parallel figure-ground object-memory vote over sparsified raw regions, another same-readout retune of the first recurrent sensory-state gain setting, another centroid readout over the same settled state, or another mild gain/unit-count retune of this first object-state attractor readout. It has to preserve more explicit spatial/topographic structure in a materially different way, or else be a measurement-first pause rather than another near-neighbor upstream tweak.

## Files Most Relevant To Resume Work

Primary code:
- `experiments/retina_classification.cpp`
- `src/adapters/RetinaAdapter.cpp`
- `include/snnfw/adapters/RetinaAdapter.h`
- `src/features/OrientationEnergyOperator.cpp`
- `include/snnfw/features/OrientationEnergyOperator.h`
- `src/features/QuadratureGaborOperator.cpp`
- `include/snnfw/features/QuadratureGaborOperator.h`
- `src/domain/VisualDomainAdapter.cpp`
- `include/snnfw/domain/VisualDomainAdapter.h`

Primary configs:
- `configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json`
- `configs/cifar10_retina_bilateral_natural_features_continuous_replay_experimental.sonata.json`
- `configs/cifar10_retina_bilateral_natural_features_orientation_energy_experimental.sonata.json`
- `configs/cifar10_retina_bilateral_natural_features_orientation_energy_tune1_experimental.sonata.json`
- `configs/cifar10_retina_bilateral_natural_features_quadrature_gabor_experimental.sonata.json`
- `configs/cifar10_retina_bilateral_natural_features_quadrature_gabor_tune1_experimental.sonata.json`
- `configs/cifar10_retina_bilateral_natural_features_contextual_grouping_experimental.sonata.json`
- `configs/cifar10_retina_bilateral_natural_features_eccentricity_sampling_experimental.sonata.json`
- `configs/cifar10_retina_bilateral_natural_features_training_augmentation_experimental.sonata.json`
- `configs/cifar10_retina_bilateral_natural_features_curriculum_review_experimental.sonata.json`
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
- `EXAMPLES_PER_CLASS=200 TEST_LIMIT=1000 LOG_PATH=build/cifar10_retina_continuous_replay_gate_delay0_200_1000.log CONFIG_PATH=configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json scripts/run_cifar10_retina_bilateral_natural.sh`

Current CIFAR promoted larger sampled benchmark:
- `EXAMPLES_PER_CLASS=1000 TEST_LIMIT=5000 LOG_PATH=build/cifar10_retina_continuous_replay_delay0_1000_5000.log CONFIG_PATH=configs/cifar10_retina_bilateral_natural_features_experimental.sonata.json scripts/run_cifar10_retina_bilateral_natural.sh`

## Bottom Line

- EMNIST and MNIST are stable.
- CIFAR mainline is now the bilateral Retina natural-features path with `training-time saccades + g10-only LGN + immediate replay` at `39.44%`, confirmed by a `39.04%` rerun with `seed=43`.
- `shape_surface` is no longer an active experiment. It finished at `35.80%` and is closed.
- Target-directed trans-saccadic inference is now implemented as an experimental path, but the first sampled probes regress and should not be scaled yet.
- The bounded contextual-grouping follow-up is now also closed at smoke; if this area is revisited, it should be via a materially different figure-ground representation or an eccentricity-dependent upstream probe, not another mild modulation of the existing `g10` feature map.
- The bounded eccentricity-dependent retinal-sampling follow-up is now also closed at smoke; if this area is revisited, it should be through a materially different foveal representation than a same-dimensional `g10` remap.
- The bounded explicit late-fused `ON/OFF + luminance` `g10` branch split is now also closed at smoke; if this area is revisited, it should be through materially different pre-cortical relay dynamics rather than more late-fused splitting of the current `g10` map.
- The bounded `g10`-only fixation-aware stage-1 memory follow-up is now also closed at the audited `200/1000` gate; if this area is revisited, it should be through materially different raw `g10` feature code or pre-cortical relay dynamics, not another averaging-vs-summary reshuffle of the same fixation set.
- The bounded magnocellular/parvocellular-like LGN relay split is now also closed at smoke; if this area is revisited, it should be through materially different pre-cortical computation than a coarse/fine band blend on the current `g10` map.
- The bounded `g10` complex-cell pooling + divisive-normalization follow-up is now also closed at smoke; if this area is revisited, it should be through an operator-matched contour/border representation on top of that improved local code, not another same-dimensional pooling or divisive-normalization retune.
- The bounded operator-matched `g10` contour-support auxiliary follow-up is now also closed under a strict audited smoke gate; it improved hemisphere purity and made both audited `g10` post-margins less negative, but it still finished at only `30.00%` and let `interaction_centroid_accuracy` slip to `28.5%`, so there is no justified `50/500` follow-up for more same-dimensional contour-support-bank retunes on the current `g10` map.
- The bounded prediction-error-gated corpus-callosum feedback proxy is now also closed under a strict audited smoke gate; both bounded decision-path forms tied the protected `32.00%` smoke and left the fusion/replay/hemisphere/`g10` audit metrics unchanged, so there is no justified `50/500` follow-up for more scalar class-centroid prediction-error suppression in the decision path.
- The bounded neuromodulator-gated eligibility plus variable-delay replay consolidation follow-up is now also closed after a direct same-build gate: it improved the strict audited smoke to `33.50%`, but left the audited upstream metrics essentially flat and then regressed the direct `50/500` gate from `30.80%` to `30.40%`, so there is no justified follow-up for more scalar replay-policy retunes of the current continuous-learning path.
- The bounded pooled branch-summary convergent-stage follow-up is now also closed under a strict audited smoke gate: it was the first materially different hemisphere-internal stage boundary on top of the protected baseline, but it collapsed the usable classifier surface to `24.50%`, cut left/right stage-1 accuracy to `15.00%`/`16.50%`, and drove best-neighbor similarity near `0.99` while the raw branch audits stayed unchanged. If a higher-order stage is revisited, it must preserve explicit spatial/topographic structure rather than pooling branch summaries this aggressively.
- The bounded topographic local-continuity/junction stage follow-up is now also closed under a strict audited smoke gate: it preserved region order and was less collapsed than the pooled-summary proxy, but it still finished at only `25.00%`, left raw branch audits unchanged, and kept neighbor similarity too close to `1.0` to justify any larger gate. If a higher-order stage is revisited, it must be materially different from both the pooled-summary proxy and this fixed local topographic transform.
- The bounded classifier-side figure-ground residual append is now also closed under a strict audited smoke gate: although the explicit figure-ground state is more fixation-stable than raw `g10`, directly appending it to the current hemisphere classifier vector collapsed both hemispheres and fusion to a single label (`10.00%` smoke, `interaction_centroid_accuracy 10.0%`, `topk_purity ~0.093`). If figure-ground is revisited, it must enter through a gated or object-level boundary rather than direct concatenation.
- The bounded same-dimensional figure-ground mask is now also closed under a strict audited smoke gate: it was far safer than direct concatenation and slightly improved left hemisphere purity plus nearest-neighbor separation, but it still finished below the protected control (`31.00%` vs `32.00%`) and weakened both right-view and fusion confidence metrics. If figure-ground is revisited, it must be through a more selective gating or object-memory boundary than direct whole-pattern masking.
- The bounded separate figure-ground object-memory vote is now also closed under a strict audited smoke gate: it was live and did not collapse the class space, but it still regressed the protected control (`30.50%` vs `32.00%`), weakened fusion confidence (`interaction 28.5%`, `confidence-concat 28.0%`), and left both hemisphere purity and nearest-neighbor structure unchanged. If figure-ground is revisited, it must be through a materially different object representation or selective association mechanism than this parallel sparsified-region vote.
