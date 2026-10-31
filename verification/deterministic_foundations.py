"""Validate inventory row 1: coordinate maps and oscillator dynamics.

Run: python verification/deterministic_foundations.py --output results/deterministic_foundations.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.integrate import solve_ivp

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from psdt import Process, hamiltonian_B
from verification.model_specification import DEFAULT_SPEC, validate


def primary_transforms(dimension):
    if dimension not in (2, 4):
        raise ValueError("Primary transforms require dimension 2 or 4")
    pairs = dimension // 2
    cases = [("identity", np.eye(dimension), True, True)]
    for angle in (np.pi / 6, np.pi / 4, np.pi / 2):
        block = np.array([[np.cos(angle), np.sin(angle)],
                          [-np.sin(angle), np.cos(angle)]])
        cases.append((f"rotation-{angle:.8g}", np.kron(np.eye(pairs), block), True, True))
    for scale in (0.5, 2.):
        cases.append((f"uniform-{scale}", scale * np.eye(dimension), False, False))
        cases.append((f"relative-{scale}",
                      np.diag([scale, 1.] * pairs), False, False))
    for scale in (0.5, 1., np.sqrt(2), 2.):
        cases.append((f"squeeze-{scale:.8g}", np.diag([scale, 1 / scale] * pairs),
                      scale == 1, True))
    for shear in (-1., 0.5, 1.):
        cases.append((f"shear-{shear}",
                      np.kron(np.eye(pairs), [[1., shear], [0., 1.]]), False, True))
    if dimension == 4:
        cross = np.eye(4)
        cross[0, 2], cross[3, 1] = 0.3, -0.3
        cases.append(("cross-pair", cross, False, True))
    return cases


def oscillator_flow(frequencies, time):
    """Closed form written independently of the production drift/exponential."""
    frequencies = np.asarray(frequencies, dtype=float)
    if (frequencies.ndim != 1 or not len(frequencies)
            or not np.isfinite(frequencies).all() or np.any(frequencies <= 0)
            or not np.isfinite(time)):
        raise ValueError("Require positive finite frequencies and finite time")
    flow = np.zeros((2 * len(frequencies),) * 2)
    for index, omega in enumerate(frequencies):
        c, s = np.cos(omega * time), np.sin(omega * time)
        flow[2 * index:2 * index + 2, 2 * index:2 * index + 2] = [
            [c, s / omega], [-omega * s, c],
        ]
    return flow


def oscillator_rhs(frequencies, states):
    result = np.empty_like(states)
    result[0::2] = states[1::2]
    result[1::2] = -np.square(frequencies)[:, None] * states[0::2]
    return result


def residual(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    return float(np.linalg.norm(actual - expected)
                 / max(1., np.linalg.norm(actual), np.linalg.norm(expected)))


def check_case(name, A, B, frequencies, orthogonal, symplectic, horizon,
               seed=1729, transformed_drift=None):
    A, B = np.asarray(A, dtype=float), np.asarray(B, dtype=float)
    frequencies = np.asarray(frequencies, dtype=float)
    dimension = 2 * len(frequencies)
    if (A.shape != (dimension, dimension) or B.shape != A.shape
            or not np.isfinite(A).all() or not np.isfinite(B).all()
            or not np.isfinite(horizon) or horizon <= 0):
        raise ValueError("Require matching finite matrices and positive finite horizon")
    oscillator_flow(frequencies, 0.)
    checks = []

    def record(label, error, tolerance, expected=True):
        satisfied = error <= tolerance
        checks.append({
            "name": label, "residual": float(error), "tolerance": float(tolerance),
            "expected_identity": bool(expected),
            "outcome": "supported" if satisfied == expected else "violated",
        })

    singular_values = np.linalg.svd(A, compute_uv=False)
    floor = 64 * dimension * np.finfo(float).eps * singular_values[0]
    invertible = singular_values[-1] > floor
    case = {
        "name": name, "dimension": dimension, "A": A.tolist(), "B": B.tolist(),
        "frequencies": frequencies.tolist(), "singular_values": singular_values.tolist(),
        "determinant": float(np.linalg.det(A)), "checks": checks,
        "claims": {"orthogonal": bool(orthogonal), "symplectic": bool(symplectic)},
    }
    checks.append({
        "name": "invertibility", "smallest_singular_value": float(singular_values[-1]),
        "singular_value_floor": float(floor),
        "outcome": "supported" if invertible else "violated",
    })
    if not invertible:
        case.update({"condition_number": None, "outcome": "violated"})
        return case
    condition = float(singular_values[0] / singular_values[-1])
    case["condition_number"] = condition
    if condition > 1e6:
        checks.append({"name": "primary conditioning budget",
                       "outcome": "numerically unresolved", "limit": 1e6})
        case["outcome"] = "numerically unresolved"
        return case
    tolerance = 512 * dimension * np.finfo(float).eps * max(1., condition**2)
    identity = np.eye(dimension)
    J = np.kron(np.eye(dimension // 2), [[0., 1.], [-1., 0.]])
    K = np.diag(np.column_stack((frequencies**2, np.ones(len(frequencies)))).ravel())
    inverse = np.linalg.solve(A, identity)
    BY = A @ B @ inverse if transformed_drift is None else np.asarray(transformed_drift)
    if BY.shape != B.shape or not np.isfinite(BY).all():
        raise ValueError("Transformed drift must be a matching finite matrix")
    KY = inverse.T @ K @ inverse
    rng = np.random.default_rng(seed)
    states = np.column_stack((identity, rng.standard_normal((dimension, 12))))
    transformed = A @ states
    case["transformed_B"] = BY.tolist()
    record("orthogonality claim", residual(A.T @ A, identity), tolerance, orthogonal)
    record("symplectic claim", residual(A.T @ J @ A, J), tolerance, symplectic)
    record("canonical Poisson brackets", residual(A @ J @ A.T, J), tolerance, symplectic)
    record("state round trip", residual(np.linalg.solve(A, transformed), states), tolerance)
    record("row-array convention", residual(states.T @ A.T, transformed.T), tolerance)
    record("baseline Hamiltonian structure", residual(B, J @ K), tolerance)
    record("baseline drift on basis and states",
           residual(B @ states, oscillator_rhs(frequencies, states)), tolerance)
    h = 1e-5
    base = states[:, -1:]
    jacobian = np.column_stack([
        ((B @ (base + h * e[:, None]) - B @ (base - h * e[:, None])) / (2 * h))[:, 0]
        for e in identity
    ])
    record("finite-difference drift Jacobian", residual(jacobian, B), 1e-9)
    record("transformed drift on states", residual(BY @ transformed, A @ (B @ states)),
           tolerance)
    record("drift inverse round trip", residual(inverse @ BY @ A, B), tolerance)
    record("energy inverse round trip", residual(A.T @ KY @ A, K), tolerance)
    record("general-coordinate Hamiltonian structure",
           residual(BY, (A @ J @ A.T) @ KY), tolerance)
    if symplectic:
        record("canonical Hamiltonian structure", residual(BY, J @ KY), tolerance)
    record("Hamiltonian generator", residual(BY.T @ KY + KY @ BY, np.zeros_like(B)),
           tolerance)
    second = primary_transforms(dimension)[1][1]
    record("state composition", residual(second @ transformed, (second @ A) @ states),
           tolerance)
    composed = second @ A
    record("drift composition",
           residual(second @ BY @ np.linalg.solve(second, identity),
                    composed @ B @ np.linalg.solve(composed, identity)), tolerance)
    times = np.array([0., 0.13, 0.7, horizon])
    reference = solve_ivp(
        lambda t, y: oscillator_rhs(frequencies, y.reshape(dimension, dimension)).ravel(),
        (0., horizon), identity.ravel(), t_eval=times,
        method="DOP853", rtol=2e-12, atol=2e-14,
    )
    numerical = solve_ivp(
        lambda t, y: (BY @ y.reshape(dimension, dimension)).ravel(),
        (0., horizon), identity.ravel(), t_eval=times,
        method="DOP853", rtol=2e-12, atol=2e-14,
    )
    if not reference.success or not numerical.success:
        raise RuntimeError(f"Independent integration failed: {reference.message}; "
                           f"{numerical.message}")
    process, transported = Process(B, 0.9), Process(BY, 0.9)
    for index, time in enumerate(times):
        exact = oscillator_flow(frequencies, time)
        expected_y = A @ exact @ inverse
        prefix = f"t={time:.8g}"
        record(f"{prefix} closed form versus production exponential",
               residual(process.M(time), exact), tolerance)
        record(f"{prefix} closed form versus independent oscillator integration",
               residual(reference.y[:, index].reshape(B.shape), exact), 2e-10)
        record(f"{prefix} propagator similarity",
               residual(transported.M(time), expected_y), tolerance)
        record(f"{prefix} transformed numerical evolution",
               residual(numerical.y[:, index].reshape(B.shape), expected_y), 2e-10)
        record(f"{prefix} energy conservation", residual(expected_y.T @ KY @ expected_y, KY),
               tolerance)
    record("negative-time inverse",
           residual(oscillator_flow(frequencies, -0.7) @ oscillator_flow(frequencies, 0.7),
                    identity), tolerance)
    case["times"] = times.tolist()
    case["outcome"] = ("supported" if all(c["outcome"] == "supported" for c in checks)
                       else "violated")
    return case


def run(spec, seed=1729):
    errors = validate(spec)
    if any(errors.values()):
        raise ValueError(f"Invalid primary specification: {errors}")
    schedule = spec["initial_smoothing"]
    horizon = float(np.log(schedule["sigmaT"] / schedule["sigma0"]) / schedule["r"])
    cases = []
    for dimension in (2, 4):
        frequencies = np.full(dimension // 2, spec["coordinates"]["omega"])
        B = hamiltonian_B(frequencies[0], d=dimension // 2)
        for name, A, orthogonal, symplectic in primary_transforms(dimension):
            cases.append(check_case(name, A, B, frequencies, orthogonal, symplectic,
                                    horizon, seed))
    frequencies = np.array([1.3, 2.7])
    B = np.zeros((4, 4))
    for i, omega in enumerate(frequencies):
        B[2*i:2*i+2, 2*i:2*i+2] = hamiltonian_B(omega, d=1)
    for name, A, orthogonal, symplectic in primary_transforms(4):
        cases.append(check_case(f"distinct-frequency/{name}", A, B, frequencies,
                                orthogonal, symplectic, horizon, seed))
    controls = []
    B = hamiltonian_B(2., d=1)
    for name, A, drift, orthogonal, symplectic, expected_check in [
        ("singular map", np.diag([1., 0.]), None, False, False, "invertibility"),
        ("false orthogonality", np.diag([2., .5]), None, True, True, "orthogonality claim"),
        ("false symplectic claim", 2 * np.eye(2), None, False, True, "symplectic claim"),
        ("wrong transformed drift", np.diag([2., .5]), B, False, True,
         "transformed drift on states"),
        ("wrong oscillator sign", np.eye(2), None, True, True,
         "baseline drift on basis and states"),
    ]:
        candidate = -B if name == "wrong oscillator sign" else B
        result = check_case(name, A, candidate, [2.], orthogonal, symplectic,
                            horizon, seed, transformed_drift=drift)
        detected = any(c["name"] == expected_check and c["outcome"] == "violated"
                       for c in result["checks"])
        controls.append({"name": name, "expected_failing_check": expected_check,
                         "outcome": "supported" if detected else "violated",
                         "evidence": result})
    # In four dimensions volume preservation alone does not imply canonicality.
    false_canonical = check_case("determinant-one noncanonical map",
                                np.diag([2., 1., 1., .5]), hamiltonian_B(2.),
                                [2., 2.], False, True, horizon, seed)
    detected = any(c["name"] == "symplectic claim" and c["outcome"] == "violated"
                   for c in false_canonical["checks"])
    controls.append({"name": false_canonical["name"], "evidence": false_canonical,
                     "outcome": "supported" if detected else "violated"})
    passed = all(c["outcome"] == "supported" for c in cases + controls)
    source_paths = [Path(__file__), Path(__file__).resolve().parents[1] / "psdt.py",
                    Path(__file__).with_name("model_specification.py")]
    return {
        "scope": "Inventory row 1 only: transformation and oscillator dynamics validation.",
        "remaining_limitations": [
            "Does not certify the full deterministic gate (inventory rows 2-8).",
            "No stochastic sampling, threshold classification, trained-score or physical validation.",
            "Extreme transforms beyond condition number 1e6 require a separate precision study.",
        ],
        "seed": seed, "horizon": horizon, "cases": cases, "negative_controls": controls,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "platform": platform.platform()},
        "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in source_paths},
        "specification_sha256": hashlib.sha256(
            json.dumps(spec, sort_keys=True, allow_nan=False).encode()).hexdigest(),
        "acceptance": {
            "matrix_relative_band": "512 * dimension * eps64 * max(1, cond(A)^2)",
            "residual": "Frobenius error / max(1, norm(actual), norm(expected))",
            "finite_difference_band": 1e-9, "integration_band": 2e-10,
            "integration_rtol": 2e-12, "integration_atol": 2e-14,
            "singular_value_floor": "64 * dimension * eps64 * largest singular value",
            "condition_number_limit": 1e6,
        },
        "outcome": "supported" if passed else "violated", "passed": passed,
    }


def analysis_markdown(report):
    cases = report["cases"]
    checks = [check for case in cases for check in case["checks"]]
    lines = [
        "# Transformation and dynamics validation",
        "",
        f"**Outcome: {report['outcome']}.** Inventory row 1 only; "
        "the full deterministic implementation gate remains pending.",
        "",
        f"Executed {len(cases)} cases and {len(checks)} identity/classification checks, "
        f"with {len(report['negative_controls'])} deliberate-error controls. "
        f"Seed: `{report['seed']}`. Primary horizon: `{report['horizon']:.10g}`.",
        "",
        "## Coverage",
        "",
        "| Grid | Cases | Maximum condition number | Outcome |",
        "|---|---:|---:|---|",
    ]
    for label, selected in [
        ("2-D primary", [c for c in cases if c["dimension"] == 2]),
        ("4-D primary", [c for c in cases if c["dimension"] == 4
                        and not c["name"].startswith("distinct-frequency/")]),
        ("4-D distinct frequencies (1.3, 2.7)",
         [c for c in cases if c["name"].startswith("distinct-frequency/")]),
    ]:
        outcome = ("supported" if all(c["outcome"] == "supported" for c in selected)
                   else "not supported")
        conditions = [c["condition_number"] for c in selected
                      if c["condition_number"] is not None]
        maximum = f"{max(conditions):.6g}" if conditions else "not assessed"
        lines.append(f"| {label} | {len(selected)} | {maximum} | {outcome} |")
    lines += [
        "",
        "Both primary grids cover identity, rotations at pi/6, pi/4 and pi/2, "
        "uniform and relative scalings at 0.5 and 2, squeezes at 0.5, 1, sqrt(2) "
        "and 2, and shears at -1, 0.5 and 1. Four-dimensional grids also include "
        "a canonical cross-pair map. Distinct frequencies prevent identical "
        "oscillator blocks from hiding ordering or mixing errors.",
        "",
        "## Numerical analysis",
        "",
        "| Diagnostic | Maximum normalized residual | Maximum fraction of acceptance band |",
        "|---|---:|---:|",
    ]
    groups = {
        "Round trips": lambda n: "round trip" in n,
        "Composition": lambda n: "composition" in n,
        "Hamiltonian structure and generator": lambda n: "Hamiltonian" in n,
        "Finite-difference drift Jacobian": lambda n: "finite-difference" in n,
        "Closed form versus production exponential": lambda n: "production exponential" in n,
        "Closed form versus independent integration":
            lambda n: "independent oscillator integration" in n,
        "Propagator similarity": lambda n: "propagator similarity" in n,
        "Transformed numerical evolution": lambda n: "transformed numerical evolution" in n,
        "Energy conservation": lambda n: "energy conservation" in n,
    }
    for label, select in groups.items():
        selected = [c for c in checks if select(c["name"]) and "residual" in c]
        if selected:
            maximum = max(c["residual"] for c in selected)
            fraction = max(c["residual"] / c["tolerance"] for c in selected)
            lines.append(f"| {label} | {maximum:.6g} | {fraction:.6g} |")
    lines += [
        "",
        "Residuals use the Frobenius error divided by "
        "`max(1, norm(actual), norm(expected))`. Matrix identities use "
        "`512 * dimension * eps64 * max(1, cond(A)^2)`. Finite-difference "
        "Jacobians use a 1e-5 centered step and a 1e-9 band. Independent "
        "DOP853 integrations use rtol=2e-12, atol=2e-14 and a 2e-10 comparison "
        "band. These are declared numerical tolerances, not statistical intervals.",
        "",
        "Closed-form evolution is compared with the production `Process.M` "
        "(SciPy matrix exponential) and an independently written position/velocity "
        "ODE. Transformed numerical dynamics are checked at t=0, 0.13, 0.7 and "
        "the primary horizon. A negative-time inverse check is also included.",
        "",
        "Orthogonality and canonicality are tested against each map's declared "
        "classification: intentional nonorthogonality/noncanonicality is not "
        "a failure. Every invertible map is checked using its transported Poisson "
        "matrix `A J A^T`; only symplectic maps must retain the original `J`.",
        "",
        "## Deliberate-error controls",
        "",
        "| Injected error | Detector outcome |",
        "|---|---|",
    ]
    for control in report["negative_controls"]:
        lines.append(f"| {control['name']} | {control['outcome']} |")
    lines += [
        "",
        "A supported control means the intended error was rejected, not that "
        "the deliberately corrupted model passed. The determinant-one "
        "noncanonical control explicitly tests that volume preservation is "
        "insufficient for a symplectic claim.",
        "",
        "## Interpretation and limits",
        "",
    ]
    lines += [f"- {limitation}" for limitation in report["remaining_limitations"]]
    lines += [
        "",
        "The adjacent JSON contains every matrix, singular value, residual, "
        "acceptance band, failure-control result, environment version, and "
        "source/specification hash. No sampling quality conclusion follows "
        "from these deterministic identities.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args(argv)
    if args.output and args.output.suffix != ".json":
        parser.error("--output must have a .json suffix; analysis uses the adjacent .md file")
    try:
        report = run(json.loads(args.spec.read_text()), args.seed)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            args.output.with_suffix(".md").write_text(analysis_markdown(report))
    except (OSError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        print(f"Transformation and dynamics validation: FAIL: {error}")
        return 1
    for case in report["cases"] + report["negative_controls"]:
        print(f"[{case['outcome']}] {case['name']}"
              + (f" ({case['dimension']}D)" if "dimension" in case else " control"))
    print(f"Transformation and dynamics validation: {'PASS' if report['passed'] else 'FAIL'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
