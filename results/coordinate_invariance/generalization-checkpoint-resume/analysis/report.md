# Optional generalization analysis

Profile: `custom`. These results do not certify core gates.

| Axis | Supported | Violated | Not assessed | Numerically unresolved |
|---|---:|---:|---:|---:|
| noise | 0 | 0 | 0 | 0 |
| data | 0 | 0 | 0 | 0 |
| prior | 0 | 0 | 0 | 0 |
| score | 0 | 0 | 0 | 0 |
| trained | 6 | 0 | 0 | 0 |
| interaction | 0 | 0 | 0 | 0 |

Full manifest completed: **True**. Full manifest passed: **True**.

Case execution time: 7.4 seconds. Process peak RSS: 80.7 MiB (process-wide high-water mark, not per-case allocation).

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

There are 3 within-axis paired comparisons and 3 baseline-arm comparisons. Effects and descriptive intervals are in the separate axis summaries. Training uncertainty is nested: sampler seeds are averaged within checkpoints before comparing independent training seeds.

Interaction results are a separate appendix in `interaction-summary.json`, not isolated one-axis effects. Held-out raw-data quality is a population proxy, not an exact-target identity. Velocity moments, component responsibility masses, and particle refinements are recorded per case.

Core prerequisite gates are not certified by these artifacts; smoke/custom budgets are not full-study evidence.

Cross-arm endpoint transport: 3/3 paired configurations passed, maximum scaled residual `3.18e-15`. This excludes intentionally independent coordinate errors.

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


## Limitations

- Smoke settings are not main-study acceptance.
- Core prerequisite gates are not certified by this runner.
- Empirical target exactness is relative to frozen training data.
- Calibration bands are not formal equivalence intervals.
- Training seeds, not nested sampler seeds, are training replicates.

See `assumption-ledger.json` for case-level identity, model-choice, and empirical-requirement outcomes. No finite grid proves universal robustness.
