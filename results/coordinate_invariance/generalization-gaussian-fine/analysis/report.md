# Optional generalization analysis

Profile: `custom`. These results do not certify core gates.

| Axis | Supported | Violated | Not assessed | Numerically unresolved |
|---|---:|---:|---:|---:|
| noise | 1 | 0 | 0 | 0 |
| data | 0 | 0 | 0 | 0 |
| prior | 0 | 0 | 0 | 0 |
| score | 0 | 0 | 0 | 0 |
| trained | 0 | 0 | 0 | 0 |
| interaction | 0 | 0 | 0 | 0 |

Full manifest completed: **True**. Full manifest passed: **True**.

Case execution time: 580.3 seconds. Process peak RSS: 100.4 MiB (process-wide high-water mark, not per-case allocation).

## Settings

```json
{
  "steps": [
    1000,
    2000,
    4000,
    8000
  ],
  "particles": [
    512,
    2000
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
    "updates": 12000,
    "width": 128,
    "depth": 3,
    "batch": 512,
    "learning_rate": 0.002,
    "validation_size": 512,
    "validate_every": 1000
  },
  "empirical_points": 2048
}
```

## Interpretation

Supported means expected implementation checks and declared exact-target controls passed, not universal robustness of approximate priors or learned scores.

There are 0 within-axis paired comparisons and 0 baseline-arm comparisons. Effects and descriptive intervals are in the separate axis summaries. Training uncertainty is nested: sampler seeds are averaged within checkpoints before comparing independent training seeds.

Interaction results are a separate appendix in `interaction-summary.json`, not isolated one-axis effects. Held-out raw-data quality is a population proxy, not an exact-target identity. Velocity moments, component responsibility masses, and particle refinements are recorded per case.

Core prerequisite gates are not certified by these artifacts; smoke/custom budgets are not full-study evidence.

## Expansion budget

```json
{
  "manifest_cases": 4224,
  "sampler_particle_step_ratio_per_case": 10.0,
  "calibration_particle_pair_ratio": 48.0,
  "empirical_component_ratio": 1.0,
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
