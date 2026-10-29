# Optional generalization analysis

Profile: `custom`. These results do not certify core gates.

| Axis | Supported | Violated | Not assessed | Numerically unresolved |
|---|---:|---:|---:|---:|
| noise | 0 | 42 | 0 | 0 |
| data | 2 | 16 | 0 | 0 |
| prior | 36 | 6 | 0 | 0 |
| score | 12 | 3 | 0 | 0 |
| trained | 72 | 0 | 0 | 0 |
| interaction | 22 | 2 | 0 | 0 |

Full manifest completed: **True**. Full manifest passed: **False**.

Case execution time: 416.5 seconds. Process peak RSS: 133.9 MiB (process-wide high-water mark, not per-case allocation).

## Settings

```json
{
  "steps": [
    128,
    256
  ],
  "particles": [
    64,
    128
  ],
  "seeds": [
    0,
    1
  ],
  "training_seeds": [
    0,
    1,
    2
  ],
  "training": {
    "updates": 20,
    "width": 16,
    "depth": 1,
    "batch": 32,
    "learning_rate": 0.002,
    "validation_size": 32,
    "validate_every": 10
  },
  "empirical_points": 64
}
```

## Interpretation

Supported means expected implementation checks and declared exact-target controls passed, not universal robustness of approximate priors or learned scores.

There are 150 within-axis paired comparisons and 148 baseline-arm comparisons. Effects and descriptive intervals are in the separate axis summaries. Training uncertainty is nested: sampler seeds are averaged within checkpoints before comparing independent training seeds.

Interaction results are a separate appendix in `interaction-summary.json`, not isolated one-axis effects. Held-out raw-data quality is a population proxy, not an exact-target identity. Velocity moments, component responsibility masses, and particle refinements are recorded per case.

Core prerequisite gates are not certified by these artifacts; smoke/custom budgets are not full-study evidence.

Cross-arm endpoint transport: 64/64 paired configurations passed, maximum scaled residual `1.25e-13`. This excludes intentionally independent coordinate errors.

## Expansion budget

```json
{
  "manifest_cases": 4224,
  "sampler_particle_step_ratio_per_case": 6103.515625,
  "calibration_particle_pair_ratio": 11718.75,
  "empirical_component_ratio": 32.0,
  "limitation": "Operation-count ratios, not wall-clock predictions. Different data, batching, training widths/budgets and cache reuse change runtime. Time a full-budget case before expanding; these ratios do not establish runtime or clear prerequisite gates."
}
```

## Unmet checks

- `noise-gaussian-83f69783998b`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-gaussian-83f69783998b.json`.
- `noise-gaussian-0a94ee169ddb`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-gaussian-0a94ee169ddb.json`.
- `noise-gaussian-a43c2e55ad27`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-gaussian-a43c2e55ad27.json`.
- `noise-fitted_moons-d1aad3a4047f`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-fitted_moons-d1aad3a4047f.json`.
- `noise-fitted_moons-bff19319d0d3`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-fitted_moons-bff19319d0d3.json`.
- `noise-fitted_moons-563efd157763`: naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-fitted_moons-563efd157763.json`.
- `noise-gaussian-8e12380d8c70`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-gaussian-8e12380d8c70.json`.
- `noise-gaussian-2dd2493dd5ef`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-gaussian-2dd2493dd5ef.json`.
- `noise-gaussian-9f072d4c1138`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-gaussian-9f072d4c1138.json`.
- `noise-fitted_moons-aaa59f0bf618`: naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-fitted_moons-aaa59f0bf618.json`.
- `noise-fitted_moons-f6b72e4d7078`: naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-fitted_moons-f6b72e4d7078.json`.
- `noise-fitted_moons-d89568b81df9`: naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-fitted_moons-d89568b81df9.json`.
- `noise-gaussian-ea9a5a165a7a`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-gaussian-ea9a5a165a7a.json`.
- `noise-gaussian-b609b647de24`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-gaussian-b609b647de24.json`.
- `noise-gaussian-e508b343b81f`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-gaussian-e508b343b81f.json`.
- `noise-fitted_moons-5ec2924082c0`: naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-fitted_moons-5ec2924082c0.json`.
- `noise-fitted_moons-7ad39cf363ef`: naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-fitted_moons-7ad39cf363ef.json`.
- `noise-fitted_moons-3cc9177b3d74`: naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-fitted_moons-3cc9177b3d74.json`.
- `noise-gaussian-e5ef11b5c21f`: ode: exact-target bands, fixed: exact-target bands; see `../cases/noise-gaussian-e5ef11b5c21f.json`.
- `noise-gaussian-98b26d89cace`: ode: exact-target bands, fixed: exact-target bands; see `../cases/noise-gaussian-98b26d89cace.json`.
- `noise-gaussian-eb6526228257`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-gaussian-eb6526228257.json`.
- `noise-fitted_moons-9c49ba1a7526`: fixed: exact-target bands; see `../cases/noise-fitted_moons-9c49ba1a7526.json`.
- `noise-fitted_moons-7fbd3230d719`: fixed: exact-target bands; see `../cases/noise-fitted_moons-7fbd3230d719.json`.
- `noise-fitted_moons-73ce3520def7`: naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-fitted_moons-73ce3520def7.json`.
- `noise-gaussian-253f27b9fc27`: ode: exact-target bands, fixed: exact-target bands; see `../cases/noise-gaussian-253f27b9fc27.json`.
- `noise-gaussian-cde549a61078`: ode: exact-target bands, fixed: exact-target bands; see `../cases/noise-gaussian-cde549a61078.json`.
- `noise-gaussian-2f9d44fdd8ca`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-gaussian-2f9d44fdd8ca.json`.
- `noise-fitted_moons-4486d1cf143a`: fixed: exact-target bands; see `../cases/noise-fitted_moons-4486d1cf143a.json`.
- `noise-fitted_moons-595ec692aed0`: fixed: exact-target bands; see `../cases/noise-fitted_moons-595ec692aed0.json`.
- `noise-fitted_moons-fb51999a906e`: naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-fitted_moons-fb51999a906e.json`.
- `noise-gaussian-4fb435654f22`: ode: exact-target bands, fixed: exact-target bands; see `../cases/noise-gaussian-4fb435654f22.json`.
- `noise-gaussian-f2ec146950cd`: ode: exact-target bands, fixed: exact-target bands; see `../cases/noise-gaussian-f2ec146950cd.json`.
- `noise-gaussian-fbf319f4a9b5`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-gaussian-fbf319f4a9b5.json`.
- `noise-fitted_moons-f2ca1a4b1cb7`: fixed: exact-target bands; see `../cases/noise-fitted_moons-f2ca1a4b1cb7.json`.
- `noise-fitted_moons-e1c75ee57476`: fixed: exact-target bands; see `../cases/noise-fitted_moons-e1c75ee57476.json`.
- `noise-fitted_moons-a2fab51458ce`: naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-fitted_moons-a2fab51458ce.json`.
- `noise-gaussian-e9a01508ef77`: ode: exact-target bands, fixed: exact-target bands; see `../cases/noise-gaussian-e9a01508ef77.json`.
- `noise-gaussian-0837a3530d97`: ode: exact-target bands, fixed: exact-target bands; see `../cases/noise-gaussian-0837a3530d97.json`.
- `noise-gaussian-1d5f5e800294`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-gaussian-1d5f5e800294.json`.
- `noise-fitted_moons-e22884b30e48`: fixed: exact-target bands; see `../cases/noise-fitted_moons-e22884b30e48.json`.
- `noise-fitted_moons-754c845abdde`: fixed: exact-target bands; see `../cases/noise-fitted_moons-754c845abdde.json`.
- `noise-fitted_moons-6201bf79f893`: naive: exact-target bands, fixed: exact-target bands; see `../cases/noise-fitted_moons-6201bf79f893.json`.
- `data-gaussian-2c5a052357d6`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/data-gaussian-2c5a052357d6.json`.
- `data-gaussian-83a85def2c23`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/data-gaussian-83a85def2c23.json`.
- `data-gaussian-ebb084efdd96`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/data-gaussian-ebb084efdd96.json`.
- `data-separated-311066e609d0`: naive: exact-target bands, fixed: exact-target bands; see `../cases/data-separated-311066e609d0.json`.
- `data-separated-6f64e084d340`: naive: exact-target bands, fixed: exact-target bands; see `../cases/data-separated-6f64e084d340.json`.
- `data-separated-125195feb268`: naive: exact-target bands, fixed: exact-target bands; see `../cases/data-separated-125195feb268.json`.
- `data-overlapping-ad352c3d9c81`: naive: exact-target bands, fixed: exact-target bands; see `../cases/data-overlapping-ad352c3d9c81.json`.
- `data-overlapping-df2bc3cbca3b`: naive: exact-target bands, fixed: exact-target bands; see `../cases/data-overlapping-df2bc3cbca3b.json`.
- `data-overlapping-fddb8f75eed4`: naive: exact-target bands, fixed: exact-target bands; see `../cases/data-overlapping-fddb8f75eed4.json`.
- `data-imbalanced-26c029636e2a`: naive: exact-target bands, fixed: exact-target bands; see `../cases/data-imbalanced-26c029636e2a.json`.
- `data-imbalanced-befeb49419ae`: naive: exact-target bands, fixed: exact-target bands; see `../cases/data-imbalanced-befeb49419ae.json`.
- `data-fitted_moons-800413f27e27`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/data-fitted_moons-800413f27e27.json`.
- `data-fitted_moons-814ed0df514b`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/data-fitted_moons-814ed0df514b.json`.
- `data-fitted_moons-b12b211aac4c`: naive: exact-target bands, fixed: exact-target bands; see `../cases/data-fitted_moons-b12b211aac4c.json`.
- `data-raw_moons-eb3d748ac6cc`: ode: exact-target bands, ode: step refinement, naive: exact-target bands, fixed: exact-target bands; see `../cases/data-raw_moons-eb3d748ac6cc.json`.
- `data-raw_moons-19931da79db8`: ode: exact-target bands, ode: step refinement, naive: exact-target bands, fixed: exact-target bands; see `../cases/data-raw_moons-19931da79db8.json`.
- `prior-gaussian-75a33600075b`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/prior-gaussian-75a33600075b.json`.
- `prior-gaussian-21bc6746a57a`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/prior-gaussian-21bc6746a57a.json`.
- `prior-gaussian-3d475faa7092`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/prior-gaussian-3d475faa7092.json`.
- `prior-separated-f5b44c0cebfa`: naive: exact-target bands, fixed: exact-target bands; see `../cases/prior-separated-f5b44c0cebfa.json`.
- `prior-separated-b24c7ee5aeb4`: naive: exact-target bands, fixed: exact-target bands; see `../cases/prior-separated-b24c7ee5aeb4.json`.
- `prior-separated-bbe9c7d26319`: naive: exact-target bands, fixed: exact-target bands; see `../cases/prior-separated-bbe9c7d26319.json`.
- `score-gaussian-c873ed98ccd7`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/score-gaussian-c873ed98ccd7.json`.
- `score-gaussian-2e5c990379d5`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/score-gaussian-2e5c990379d5.json`.
- `score-gaussian-7af2e0302a88`: ode: exact-target bands, naive: exact-target bands, fixed: exact-target bands; see `../cases/score-gaussian-7af2e0302a88.json`.
- `interaction-imbalanced-7cc5ebc1fc2a`: fixed: exact-target bands; see `../cases/interaction-imbalanced-7cc5ebc1fc2a.json`.
- `interaction-imbalanced-74034a9a83a8`: fixed: exact-target bands; see `../cases/interaction-imbalanced-74034a9a83a8.json`.

## Limitations

- Smoke settings are not main-study acceptance.
- Core prerequisite gates are not certified by this runner.
- Empirical target exactness is relative to frozen training data.
- Calibration bands are not formal equivalence intervals.
- Training seeds, not nested sampler seeds, are training replicates.

See `assumption-ledger.json` for case-level identity, model-choice, and empirical-requirement outcomes. No finite grid proves universal robustness.
