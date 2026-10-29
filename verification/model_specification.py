"""Check the recorded primary model, not sampler conformance or physical truth.

Run: python verification/model_specification.py [path/to/spec.json]
Exit code 0 means all six specification categories pass.
"""
import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_SPEC = Path(__file__).resolve().parents[1] / "coordinate_invariance_model_spec.json"


def compare(actual, expected, path):
    """Return field-specific failures against the Section 1.1 contract."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected an object"]
        errors = [f"{path}.{key}: missing" for key in expected.keys() - actual.keys()]
        errors += [f"{path}.{key}: unknown field" for key in actual.keys() - expected.keys()]
        for key in expected.keys() & actual.keys():
            errors.extend(compare(actual[key], expected[key], f"{path}.{key}"))
        return sorted(errors)
    if isinstance(expected, np.ndarray):
        if (not isinstance(actual, list)
                or not all(isinstance(row, list)
                           and all(type(item) in (int, float) for item in row)
                           for row in actual)):
            return [f"{path}: expected a finite numeric matrix"]
        try:
            value = np.asarray(actual, dtype=float)
        except (TypeError, ValueError, OverflowError):
            return [f"{path}: expected a finite numeric matrix"]
        if (value.shape != expected.shape or not np.isfinite(value).all()
                or not np.allclose(value, expected, rtol=1e-12, atol=1e-14)):
            return [f"{path}: expected {expected.tolist()}, got {actual!r}"]
        return []
    if isinstance(expected, float):
        valid = (type(actual) in (int, float)
                 and abs(actual) <= float(np.finfo(float).max)
                 and np.isclose(actual, expected, rtol=1e-12, atol=1e-14))
    else:
        valid = type(actual) is type(expected) and actual == expected
    return [] if valid else [f"{path}: expected {expected!r}, got {actual!r}"]


def expected_specification():
    """Independent executable transcription of the primary model in Section 1.1."""
    identity = np.eye(4)
    B = np.kron(np.eye(2), [[0., 1.], [-4., 0.]])
    J = np.kron(np.eye(2), [[0., 1.], [-1., 0.]])
    A = np.diag([2., 0.5, 2., 0.5])
    inverse = np.linalg.inv(A)
    arms = {
        "example_scope": "representative nonorthogonal squeeze; not the full transform grid",
        "A": A,
    }
    for name, transform, Q, factor in [
        ("baseline", identity, identity, identity),
        ("transported", A, A @ A.T, A),
        ("reisotropized", A, identity, identity),
    ]:
        pullback = identity if name == "baseline" else inverse
        arms[name] = {
            "clean_transform": transform,
            "B": transform @ B @ np.linalg.inv(transform),
            "Q": Q,
            "noise_factor": factor,
            "initial_conditional_covariance": 0.01**2 * Q,
            "pulled_back_Q": pullback @ Q @ pullback.T,
        }
    return {
        "units": {
            "system": "dimensionless",
            "reference_length": 1.,
            "reference_time": 1.,
            "reference_mass": 1.,
            "dimensionless_mass": 1.,
            "momentum_equals_velocity": True,
            "reference_scales_fixed_across_arms": True,
            "transform_acts_after_nondimensionalization": True,
            "physical_calibration": "not established",
        },
        "coordinates": {
            "dimension": 4,
            "order": ["x1", "v1", "x2", "v2"],
            "matrix_pilot_order": ["x1", "v1"],
            "row_sample_transform": "Y = Z @ A.T",
            "omega": 2.,
            "B": B,
            "J": J,
        },
        "covariance": {
            "Q_policy": "fixed symmetric positive definite",
            "conditional": "C(t) = sigma(t)^2 Q",
            "total": "M(t) Cov(X) M(t)^T + C(t)",
            "propagator": "M(t) = exp(B t)",
            "noise_independent_of_clean_state": True,
            "marginals_alone_define_sde_transitions": False,
        },
        "initial_smoothing": {
            "schedule": "sigma(t) = sigma0 exp(r t)",
            "sigma0": 0.01,
            "r": 0.9,
            "sigmaT": 40.,
            "horizon_policy": "T = log(sigmaT / sigma0) / r",
            "shared_rate_and_horizon_within_comparison": True,
            "target": "X + sigma0 L_Q epsilon",
            "target_is_clean_law": False,
            "terminal_initialization": "exact marginal",
            "terminal_draw": "M(T) X + sigmaT L_Q epsilon",
            "approximate_prior": "separately labeled control only",
            "target_noise_independent_of_sampler_noise": True,
            "clean_law_parameters": "must be recorded before sampling",
        },
        "arms": arms,
        "feasibility_scope": {
            "drift": "B z",
            "diffusion": "real state-independent",
            "initial_law": "smoothed Z_0",
            "G_definition": "dC/dt - B C - C B^T",
            "exponential_G": "2 sigma(t)^2 (r Q - (B Q + Q B^T)/2)",
            "criterion": "G(t) positive semidefinite iff r >= kappa_Q",
            "threshold": "kappa_Q = lambda_max((B Q + Q B^T)/2, Q)",
            "singular_psd_allowed": True,
            "below_threshold_marginals_exist": True,
            "rules_out_all_stochastic_dynamics": False,
            "establishes_physical_validity": False,
        },
    }


def validate(spec):
    expected = expected_specification()
    if not isinstance(spec, dict):
        return {"document": ["specification: expected an object"]}
    results = {}
    for name, contract in expected.items():
        results[name] = (compare(spec[name], contract, name) if name in spec
                         else [f"{name}: missing category"])
    unknown = spec.keys() - expected.keys()
    if unknown:
        results["document"] = [f"{name}: unknown category" for name in sorted(unknown)]
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", nargs="?", type=Path, default=DEFAULT_SPEC)
    args = parser.parse_args(argv)
    try:
        with args.spec.open() as stream:
            spec = json.load(stream)
    except (OSError, ValueError) as error:
        print(f"[FAIL] Cannot read specification {args.spec}: {error}")
        return 1
    results = validate(spec)
    for category, errors in results.items():
        print(f"[{'FAIL' if errors else 'PASS'}] {category}")
        for error in errors:
            print(f"  {error}")
    passed = not any(results.values())
    print(f"Model specification gate: {'PASS' if passed else 'FAIL'}")
    print("Scope: recorded primary conventions and example arm matrices only; "
          "not runner conformance, deterministic implementation, or sampling accuracy.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
