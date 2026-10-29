"""Verify Section 9.3 forward transitions, not reverse-sampler accuracy.

Run: python verification/forward_feasibility.py --output results/forward_feasibility.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import quad_vec
from scipy.linalg import eigh, expm

if __package__:
    from .model_specification import DEFAULT_SPEC, validate
else:
    from model_specification import DEFAULT_SPEC, validate


def threshold(B, Q):
    """Return the top generalized eigenpair and its relative residual."""
    B, Q = np.asarray(B, dtype=float), np.asarray(Q, dtype=float)
    if (B.ndim != 2 or B.shape[0] != B.shape[1] or Q.shape != B.shape
            or not np.isfinite(B).all() or not np.isfinite(Q).all()
            or not np.allclose(Q, Q.T, rtol=0, atol=1e-14)):
        raise ValueError("B and symmetric Q must be finite square matrices of equal size")
    np.linalg.cholesky(Q)
    H = (B @ Q + Q @ B.T) / 2
    values, vectors = eigh(H, Q)
    kappa, v = float(values[-1]), vectors[:, -1]
    scale = max(np.linalg.norm(H, 2) + abs(kappa) * np.linalg.norm(Q, 2),
                np.finfo(float).tiny)
    residual = np.linalg.norm(H @ v - kappa * Q @ v) / (scale * np.linalg.norm(v))
    return kappa, v, float(residual)


def transition_covariance(B, Q, rate, sigma0, s, t):
    """Return R and a roundoff band based on the operands, before subtraction."""
    if not (np.isfinite([rate, sigma0, s, t]).all() and sigma0 > 0 and 0 <= s < t):
        raise ValueError("Require finite rate, sigma0 > 0, and 0 <= s < t")
    M = expm(B * (t - s))
    Ct = sigma0**2 * np.exp(2 * rate * t) * Q
    propagated = sigma0**2 * np.exp(2 * rate * s) * (M @ Q @ M.T)
    R = Ct - propagated
    scale = np.linalg.norm(Ct, 2) + np.linalg.norm(propagated, 2)
    tolerance = 256 * B.shape[0] * np.finfo(float).eps * scale
    if not np.isfinite(R).all():
        raise ValueError("Nonfinite transition covariance")
    return (R + R.T) / 2, float(tolerance)


def integrated_covariance(B, Q, rate, sigma0, s, t):
    H = (B @ Q + Q @ B.T) / 2

    def integrand(u):
        M = expm(B * (t - u))
        G = 2 * sigma0**2 * np.exp(2 * rate * u) * (rate * Q - H)
        return M @ G @ M.T

    R, error = quad_vec(integrand, s, t, epsabs=1e-15, epsrel=1e-11)
    return (R + R.T) / 2, float(error)


def psd_factor(R, tolerance):
    """Reject negative variance; only neutralize explicitly bounded roundoff."""
    values, vectors = eigh(R)
    if values[0] < -tolerance:
        raise ValueError(f"Negative transition variance: {values[0]:.6g}")
    correction = float(-np.minimum(values, 0).sum())
    factor = vectors * np.sqrt(np.maximum(values, 0))
    return factor, correction


def moment_diagnostics(samples, mean, covariance):
    """Entrywise Gaussian standard errors, including covariance-estimator error."""
    n = len(samples)
    sample_covariance = np.cov(samples, rowvar=False)
    diagonal = np.diag(covariance)
    mean_se = np.sqrt(diagonal / n)
    covariance_se = np.sqrt(
        (np.outer(diagonal, diagonal) + covariance**2) / (n - 1)
    )
    return {
        "mean_max_standard_errors": float(np.max(np.abs(samples.mean(0) - mean) / mean_se)),
        "covariance_max_standard_errors": float(
            np.max(np.abs(sample_covariance - covariance) / covariance_se)
        ),
    }


def simulate(B, Q, rate, sigma0, horizon, samples, rng):
    """Start from an explicitly smoothed Gaussian and apply exact transitions."""
    d = len(B)
    clean_mean = np.linspace(-0.01, 0.01, d)
    clean_covariance = sigma0**2 * np.diag(np.linspace(1, 2, d))
    clean = rng.multivariate_normal(clean_mean, clean_covariance, samples)
    z = clean + sigma0 * (rng.standard_normal((samples, d)) @ np.linalg.cholesky(Q).T)
    times = [0., 0.01 * horizon, 0.1 * horizon, 0.5 * horizon, horizon]
    reports = []
    for s, t in zip(times[:-1], times[1:]):
        R, tolerance = transition_covariance(B, Q, rate, sigma0, s, t)
        factor, correction = psd_factor(R, tolerance)
        noise = rng.standard_normal((samples, d)) @ factor.T
        z = z @ expm(B * (t - s)).T + noise
        M = expm(B * t)
        expected_covariance = (
            M @ clean_covariance @ M.T + sigma0**2 * np.exp(2 * rate * t) * Q
        )
        diagnostics = moment_diagnostics(z, M @ clean_mean, expected_covariance)
        # Innovations separately catch errors hidden by large propagated data variance.
        innovation_mean = np.linalg.norm(noise.mean(0)) / np.sqrt(np.trace(R) / samples)
        innovation_covariance = np.linalg.norm(np.cov(noise, rowvar=False) - R, "fro")
        innovation_se = np.sqrt((np.trace(R)**2 + np.sum(R**2)) / (samples - 1))
        diagnostics.update({
            "s": s, "t": t,
            "roundoff_eigenvalue_correction": correction,
            "innovation_mean_standard_errors": float(innovation_mean),
            "innovation_covariance_standard_errors": float(innovation_covariance / innovation_se),
        })
        diagnostics["passed"] = all(
            diagnostics[key] < 7 for key in [
                "mean_max_standard_errors", "covariance_max_standard_errors",
                "innovation_mean_standard_errors", "innovation_covariance_standard_errors",
            ]
        )
        reports.append(diagnostics)
    return {
        "clean_mean": clean_mean.tolist(),
        "clean_covariance": clean_covariance.tolist(),
        "initial_smoothing_covariance": (sigma0**2 * Q).tolist(),
        "steps": reports,
        "passed": all(step["passed"] for step in reports),
    }


def check_case(B, Q, rate, regime, sigma0, horizon, samples, rng):
    kappa, v, residual = threshold(B, Q)
    intervals = [(0., horizon * 1e-4), (0.3 * horizon, 0.31 * horizon),
                 (0., horizon), (0.5 * horizon, horizon)]
    transitions = []
    for s, t in intervals:
        R, tolerance = transition_covariance(B, Q, rate, sigma0, s, t)
        integral, error = integrated_covariance(B, Q, rate, sigma0, s, t)
        difference = float(np.linalg.norm(R - integral, 2))
        agreement_band = tolerance + 5 * error + 1e-10 * np.linalg.norm(integral, 2)
        minimum = float(eigh(R, eigvals_only=True)[0])
        transitions.append({
            "s": s, "t": t, "minimum_eigenvalue": minimum,
            "covariance": R.tolist(), "quadrature_covariance": integral.tolist(),
            "psd_roundoff_band": tolerance, "quadrature_error_estimate": error,
            "formula_quadrature_difference": difference,
            "formula_quadrature_band": float(agreement_band),
            "passed": bool(difference <= agreement_band
                           and (regime == "below" or minimum >= -tolerance)),
        })
    result = {
        "regime": regime, "rate": rate, "threshold": kappa,
        "threshold_eigenpair_relative_residual": residual,
        "transitions": transitions,
    }
    if regime == "below":
        v = v / np.linalg.norm(v)
        G = 2 * sigma0**2 * (rate * Q - (B @ Q + Q @ B.T) / 2)
        witnesses = []
        for h in [1e-3, 1e-4, 1e-5]:
            R, tolerance = transition_covariance(B, Q, rate, sigma0, 0., h)
            integral, error = integrated_covariance(B, Q, rate, sigma0, 0., h)
            quadratic = float(v @ R @ v)
            integral_quadratic = float(v @ integral @ v)
            witnesses.append({
                "h": h, "quadratic_form": quadratic,
                "quadrature_quadratic_form": integral_quadratic,
                "roundoff_band": tolerance,
                "local_derivative_error": float(abs(quadratic / h - v @ G @ v)),
                "passed": bool(quadratic < -tolerance and integral_quadratic < -5 * error),
            })
        result["witness"] = {
            "direction": v.tolist(), "instantaneous_quadratic_form": float(v @ G @ v),
            "intervals": witnesses,
            "passed": bool(v @ G @ v < 0 and all(w["passed"] for w in witnesses)),
        }
    else:
        result["simulation"] = simulate(B, Q, rate, sigma0, horizon, samples, rng)
    result["passed"] = bool(
        residual < 1e-12 and all(item["passed"] for item in transitions)
        and result.get("witness", result.get("simulation"))["passed"]
    )
    return result


def run(spec, samples=20000, seed=1729):
    errors = validate(spec)
    if any(errors.values()):
        raise ValueError("Invalid model specification: " + "; ".join(
            error for category in errors.values() for error in category))
    if type(samples) is not int or samples < 1000:
        raise ValueError("samples must be an integer >= 1000")
    rng = np.random.default_rng(seed)
    schedule = spec["initial_smoothing"]
    sigma0 = schedule["sigma0"]
    horizon = float(np.log(schedule["sigmaT"] / sigma0) / schedule["r"])
    cases = []
    transport_checks = []
    for dimension in [2, 4]:
        A = np.asarray(spec["arms"]["A"], dtype=float)[:dimension, :dimension]
        baseline = spec["arms"]["baseline"]
        B0 = np.asarray(baseline["B"], dtype=float)[:dimension, :dimension]
        Q0 = np.asarray(baseline["Q"], dtype=float)[:dimension, :dimension]
        kappa0, v0, _ = threshold(B0, Q0)
        transported = spec["arms"]["transported"]
        BY = np.asarray(transported["B"], dtype=float)[:dimension, :dimension]
        QY = np.asarray(transported["Q"], dtype=float)[:dimension, :dimension]
        vY = np.linalg.solve(A.T, v0)
        R0, tol0 = transition_covariance(B0, Q0, kappa0 - 0.3, sigma0, 0., 1e-4)
        RY, tolY = transition_covariance(BY, QY, kappa0 - 0.3, sigma0, 0., 1e-4)
        error = float(np.linalg.norm(RY - A @ R0 @ A.T, 2))
        band = tolY + np.linalg.norm(A, 2)**2 * tol0
        witness_error = float(abs(vY @ RY @ vY - v0 @ R0 @ v0))
        witness_band = np.linalg.norm(vY)**2 * tolY + np.linalg.norm(v0)**2 * tol0
        transport_checks.append({
            "dimension": dimension, "covariance_error": error, "covariance_band": float(band),
            "witness_error": witness_error, "witness_band": float(witness_band),
            "transported_witness": vY.tolist(),
            "passed": bool(error <= band and witness_error <= witness_band
                           and vY @ RY @ vY < -np.linalg.norm(vY)**2 * tolY),
        })
        for name in ["baseline", "transported", "reisotropized"]:
            arm = spec["arms"][name]
            B = np.asarray(arm["B"], dtype=float)[:dimension, :dimension]
            Q = np.asarray(arm["Q"], dtype=float)[:dimension, :dimension]
            kappa, _, _ = threshold(B, Q)
            for regime, rate in [("below", kappa - 0.3), ("at", kappa), ("above", kappa + 0.3)]:
                case = check_case(B, Q, rate, regime, sigma0, horizon, samples, rng)
                case.update({"dimension": dimension, "arm": name, "B": B.tolist(), "Q": Q.tolist()})
                cases.append(case)
    return {
        "scope": "Forward-feasibility evidence for the recorded representative squeeze only; "
                 "does not clear the deterministic prerequisite or the full transform grid.",
        "schedule_policy": "Threshold-centered additive rate sweep on the fixed primary horizon; "
                           "sweep terminal sigma differs from the primary sigmaT.",
        "sigma0": sigma0, "horizon": horizon, "samples": samples, "seed": seed,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "simulation_acceptance": "All reported Gaussian moment errors < 7 standard errors",
        "roundoff_policy": "Classify before factorization; reject eigenvalues below the reported "
                           "operand-scaled band; report any bounded negative-eigenvalue correction.",
        "transport_checks": transport_checks, "cases": cases,
        "passed": all(case["passed"] for case in cases)
                  and all(check["passed"] for check in transport_checks),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--samples", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    try:
        with args.spec.open() as stream:
            spec = json.load(stream)
        report = run(spec, args.samples, args.seed)
        report["specification_path"] = str(args.spec.resolve())
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError, np.linalg.LinAlgError) as error:
        print(f"Forward feasibility: FAIL: {error}")
        return 1
    for case in report["cases"]:
        print(f"[{'PASS' if case['passed'] else 'FAIL'}] "
              f"{case['dimension']}D {case['arm']} {case['regime']}: "
              f"r={case['rate']:.6g}, kappa={case['threshold']:.6g}")
    for check in report["transport_checks"]:
        print(f"[{'PASS' if check['passed'] else 'FAIL'}] "
              f"{check['dimension']}D covariance and negative-witness transport")
    print(f"Forward feasibility: {'PASS' if report['passed'] else 'FAIL'}")
    if args.output:
        print(f"Evidence: {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
