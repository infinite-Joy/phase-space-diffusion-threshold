"""Gaussian reverse-sampling pilot for the Section 9.8 evidence gate.

Run: python verification/gaussian_sampling.py --output results/gaussian_sampling.json
The defaults exercise representative models, not the full transformation grid.
"""
import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, expm

if __package__:
    from .model_specification import DEFAULT_SPEC, validate
    from .forward_feasibility import threshold
else:
    from model_specification import DEFAULT_SPEC, validate
    from forward_feasibility import threshold


MODES = ("ode", "naive", "fixed")
PATH_BAND = 1e-9
MOMENT_BAND = 8.
INTEGRATION_BAND = 0.05


@dataclass
class GaussianModel:
    B: np.ndarray
    Q: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray
    rate: float
    sigma0: float
    horizon: float

    def marginal(self, t):
        M = expm(self.B * t)
        sigma = self.sigma0 * np.exp(self.rate * t)
        return M @ self.mean, M @ self.covariance @ M.T + sigma**2 * self.Q

    def transported(self, A):
        return replace(
            self, B=np.linalg.solve(A.T, (A @ self.B).T).T,
            Q=A @ self.Q @ A.T, mean=A @ self.mean,
            covariance=A @ self.covariance @ A.T,
        )


def diffusion_shapes(model):
    G = 2 * (model.rate * model.Q - (model.B @ model.Q + model.Q @ model.B.T) / 2)
    values, vectors = eigh(G)
    D = (vectors * np.maximum(values, 0)) @ vectors.T
    L = (vectors * np.sqrt(np.maximum(values, 0))) @ vectors.T
    return G, D, L


def moment_oracle(model, mode):
    """Integrate continuous reverse moments independently of sampler coefficients."""
    if mode not in MODES:
        raise ValueError(f"Unknown sampler: {mode}")
    d = len(model.B)
    meanT, covarianceT = model.marginal(model.horizon)

    def rhs(tau, state):
        t = model.horizon - tau
        M = expm(model.B * t)
        sig2 = model.sigma0**2 * np.exp(2 * model.rate * t)
        marginal_mean = M @ model.mean
        marginal_covariance = M @ model.covariance @ M.T + sig2 * model.Q
        precision = np.linalg.solve(marginal_covariance, np.eye(d))
        C = sig2 * model.Q
        G = 2 * model.rate * C - model.B @ C - C @ model.B.T
        if mode == "ode":
            D = np.zeros_like(G)
            W = G / 2
        else:
            values, vectors = eigh((G + G.T) / 2)
            D = (vectors * np.maximum(values, 0)) @ vectors.T
            W = D if mode == "naive" else (G + D) / 2
        drift = -model.B - W @ precision
        offset = W @ precision @ marginal_mean
        mean = state[:d]
        covariance = state[d:].reshape(d, d)
        return np.concatenate([
            drift @ mean + offset,
            (drift @ covariance + covariance @ drift.T + D).ravel(),
        ])

    solution = solve_ivp(
        rhs, (0., model.horizon), np.concatenate([meanT, covarianceT.ravel()]),
        method="DOP853", rtol=2e-11, atol=2e-12,
    )
    if not solution.success:
        raise ValueError(f"Reverse moment integration failed: {solution.message}")
    mean = solution.y[:d, -1]
    covariance = solution.y[d:, -1].reshape(d, d)
    covariance = (covariance + covariance.T) / 2
    np.linalg.cholesky(covariance)
    return mean, covariance


def coefficients(model, steps, diffusion=None):
    """Precompute affine exact-score drifts; the reference ODE never uses these."""
    times = np.linspace(model.horizon, 0., steps + 1)
    M = expm(times[:, None, None] * model.B)
    means = np.einsum("tij,j->ti", M, model.mean)
    sigmas = model.sigma0 * np.exp(model.rate * times)
    covariances = M @ model.covariance @ M.transpose(0, 2, 1)
    covariances += sigmas[:, None, None]**2 * model.Q
    precision = np.linalg.inv(covariances)
    G, D, L = diffusion_shapes(model)
    if diffusion is not None:
        D, L = diffusion
    tables = {}
    for mode in MODES:
        W = G / 2 if mode == "ode" else D if mode == "naive" else (G + D) / 2
        WP = sigmas[:, None, None]**2 * (W @ precision)
        tables[mode] = {
            "K": model.B + WP,
            "b": -np.einsum("tij,tj->ti", WP, means),
            "L": np.zeros_like(L) if mode == "ode" else L,
            "D": np.zeros_like(D) if mode == "ode" else D,
        }
    return times, sigmas, tables


def affine_step(table, sigmas, index, stride, h, mode):
    K, b = table["K"][index], table["b"][index]
    identity = np.eye(len(b))
    if mode == "ode":
        K2, b2 = table["K"][index + stride], table["b"][index + stride]
        F = identity - h * (K + K2) / 2 + h**2 * (K2 @ K) / 2
        c = -h * (b + b2) / 2 + h**2 * (K2 @ b) / 2
    else:
        F, c = identity - h * K, -h * b
    L = sigmas[index] * table["L"]
    return F, c, L


def normalized_error(mean, covariance, reference_mean, reference_covariance):
    root = np.linalg.cholesky(reference_covariance)
    dm = np.linalg.solve(root, mean - reference_mean)
    delta = np.linalg.solve(root, covariance - reference_covariance)
    delta = np.linalg.solve(root, delta.T).T
    return {
        "mean": float(np.linalg.norm(dm)),
        "covariance": float(np.linalg.norm(delta, "fro")),
        "maximum": float(max(np.linalg.norm(dm), np.linalg.norm(delta, "fro"))),
    }


def gaussian_kl(mean, covariance, reference_mean, reference_covariance):
    root = np.linalg.cholesky(reference_covariance)
    whitened = np.linalg.solve(root, covariance)
    whitened = np.linalg.solve(root, whitened.T).T
    values = np.linalg.eigvalsh((whitened + whitened.T) / 2)
    if np.any(values <= 0):
        raise ValueError("Nonpositive covariance in Gaussian KL")
    dm = np.linalg.solve(root, mean - reference_mean)
    # log1p avoids cancellation for covariances close to the reference.
    return float((dm @ dm + np.sum(values - 1 - np.log1p(values - 1))) / 2)


def sample_diagnostics(z, mean, covariance):
    n = len(z)
    root = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(root, (z - mean).T).T
    empirical_mean = z.mean(0)
    empirical_covariance = np.cov(z, rowvar=False)
    covariance_se = np.sqrt((1 + np.eye(z.shape[1])) / (n - 1))
    mean_z = float(np.max(np.abs(whitened.mean(0))) * np.sqrt(n))
    covariance_z = float(np.max(
        np.abs(np.cov(whitened, rowvar=False) - np.eye(z.shape[1])) / covariance_se
    ))
    return {
        "samples": n, "mean": empirical_mean.tolist(),
        "covariance": empirical_covariance.tolist(),
        "mean_max_standard_errors": mean_z,
        "covariance_max_standard_errors": covariance_z,
        "normalized_error": normalized_error(empirical_mean, empirical_covariance, mean, covariance),
        "passed": bool(mean_z < MOMENT_BAND and covariance_z < MOMENT_BAND),
    }


def refinement_passed(errors):
    return bool(
        np.isfinite(errors).all() and errors[-1] < INTEGRATION_BAND
        and errors[-1] <= 0.75 * errors[0] + 1e-7
    )


def coupled_paths(model, transforms, steps, particle_counts, seed):
    """Stream fine Brownian increments; sum them for every coarser level."""
    finest = steps[-1]
    times, sigmas, tables = coefficients(model, finest)
    companions = []
    _, D, L = diffusion_shapes(model)
    for A in transforms:
        other = model.transported(A)
        _, other_sigmas, other_tables = coefficients(
            other, finest, diffusion=(A @ D @ A.T, A @ L)
        )
        companions.append((A, other_sigmas, other_tables))
    terminal_stream, brownian_stream = np.random.SeedSequence(seed).spawn(2)
    terminal_rng = np.random.default_rng(terminal_stream)
    brownian_rng = np.random.default_rng(brownian_stream)
    meanT, covarianceT = model.marginal(model.horizon)
    n, d = particle_counts[-1], len(model.B)
    terminal = meanT + terminal_rng.standard_normal((n, d)) @ np.linalg.cholesky(covarianceT).T
    path_particles = min(n, 32)
    levels = {}
    for count in steps:
        levels[count] = {
            "increment": np.zeros((n, d)),
            "modes": {
                mode: {
                    "z": terminal.copy(), "mean": meanT.copy(), "covariance": covarianceT.copy(),
                    "companions": [terminal[:path_particles] @ A.T for A in transforms],
                    "path_residuals": [0.] * len(transforms),
                } for mode in MODES
            },
        }
    fine_h = model.horizon / finest
    for fine_index in range(finest):
        increment = np.sqrt(fine_h) * brownian_rng.standard_normal((n, d))
        for count, level in levels.items():
            level["increment"] += increment
            stride = finest // count
            if (fine_index + 1) % stride:
                continue
            index = fine_index + 1 - stride
            h = times[index] - times[index + stride]
            for mode, state in level["modes"].items():
                F, c, factor = affine_step(tables[mode], sigmas, index, stride, h, mode)
                state["z"] = state["z"] @ F.T + c + level["increment"] @ factor.T
                state["mean"] = F @ state["mean"] + c
                covariance = F @ state["covariance"] @ F.T + h * (factor @ factor.T)
                state["covariance"] = (covariance + covariance.T) / 2
                for j, (A, other_sigmas, other_tables) in enumerate(companions):
                    FY, cY, LY = affine_step(
                        other_tables[mode], other_sigmas, index, stride, h, mode
                    )
                    y = state["companions"][j]
                    y = y @ FY.T + cY + level["increment"][:path_particles] @ LY.T
                    state["companions"][j] = y
                    pulled_back = np.linalg.solve(A, y.T).T
                    scale = max(1., float(np.max(np.abs(state["z"][:path_particles]))))
                    residual = float(np.max(np.abs(pulled_back - state["z"][:path_particles])) / scale)
                    state["path_residuals"][j] = max(state["path_residuals"][j], residual)
            level["increment"].fill(0)
    reports = []
    for count, level in levels.items():
        for mode, state in level["modes"].items():
            if not np.isfinite(state["z"]).all():
                raise ValueError(f"Nonfinite {mode} samples at K={count}")
            diagnostics = [
                sample_diagnostics(state["z"][:size], state["mean"], state["covariance"])
                for size in particle_counts
            ]
            reports.append({
                "steps": count, "mode": mode,
                "discrete_mean": state["mean"].tolist(),
                "discrete_covariance": state["covariance"].tolist(),
                "particle_refinement": diagnostics,
                "path_max_scaled_residuals": state["path_residuals"],
                "passed": bool(all(item["passed"] for item in diagnostics)
                               and max(state["path_residuals"], default=0.) < PATH_BAND),
            })
    return {
        "seed": seed, "rng_policy": "SeedSequence(seed).spawn(2): exact terminal, Brownian",
        "terminal": sample_diagnostics(terminal, meanT, covarianceT),
        "terminal_mean": meanT.tolist(), "terminal_covariance": covarianceT.tolist(),
        "transport_path_particles": path_particles,
        "levels": reports,
        "passed": bool(all(item["passed"] for item in reports)
                       and sample_diagnostics(terminal, meanT, covarianceT)["passed"]),
    }


def check_model(model, transforms, steps, particle_counts, seeds):
    target_mean, target_covariance = model.marginal(0.)
    references = {}
    oracle_passed = True
    kappa, _, _ = threshold(model.B, model.Q)
    for mode in MODES:
        mean, covariance = moment_oracle(model, mode)
        target_error = normalized_error(mean, covariance, target_mean, target_covariance)
        should_match_target = mode != "naive" or model.rate >= kappa - 1e-12
        passed = not should_match_target or target_error["maximum"] < 1e-6
        oracle_passed &= passed
        references[mode] = {
            "mean": mean.tolist(), "covariance": covariance.tolist(),
            "target_error": target_error,
            "target_kl": gaussian_kl(mean, covariance, target_mean, target_covariance),
            "expected_to_match_target": should_match_target, "passed": bool(passed),
        }
    runs = [coupled_paths(model, transforms, steps, particle_counts, seed) for seed in seeds]
    refinement = {}
    for mode in MODES:
        reference = references[mode]
        levels = [level for level in runs[0]["levels"] if level["mode"] == mode]
        errors = []
        for level in levels:
            error = normalized_error(
                np.asarray(level["discrete_mean"]), np.asarray(level["discrete_covariance"]),
                np.asarray(reference["mean"]), np.asarray(reference["covariance"]),
            )
            errors.append(error["maximum"])
            level["continuous_reference_error"] = error
            level["target_kl"] = gaussian_kl(
                np.asarray(level["discrete_mean"]), np.asarray(level["discrete_covariance"]),
                target_mean, target_covariance,
            )
        refinement[mode] = {
            "steps": steps, "normalized_errors_to_continuous_oracle": errors,
            "passed": refinement_passed(errors),
        }
    # Prefix particle sets are coupled sizes, not independent experimental replicates.
    particle_summary = []
    for mode in MODES:
        for size in particle_counts:
            errors = []
            for run in runs:
                level = next(item for item in run["levels"]
                             if item["mode"] == mode and item["steps"] == steps[-1])
                diagnostic = next(item for item in level["particle_refinement"]
                                  if item["samples"] == size)
                errors.append(diagnostic["normalized_error"]["maximum"])
            particle_summary.append({
                "mode": mode, "samples": size, "seed_errors_to_discrete_law": errors,
                "mean_error": float(np.mean(errors)),
                "seed_standard_error": (float(np.std(errors, ddof=1) / np.sqrt(len(errors)))
                                        if len(errors) > 1 else None),
            })
    paired_particle_differences = []
    for mode in MODES:
        summaries = [item for item in particle_summary if item["mode"] == mode]
        for coarse, fine in zip(summaries[:-1], summaries[1:]):
            differences = np.asarray(fine["seed_errors_to_discrete_law"]) - np.asarray(
                coarse["seed_errors_to_discrete_law"]
            )
            paired_particle_differences.append({
                "mode": mode, "sample_counts": [coarse["samples"], fine["samples"]],
                "seed_error_differences_fine_minus_coarse": differences.tolist(),
                "mean_difference": float(np.mean(differences)),
                "seed_standard_error": (
                    float(np.std(differences, ddof=1) / np.sqrt(len(differences)))
                    if len(differences) > 1 else None
                ),
            })
    return {
        "B": model.B.tolist(), "Q": model.Q.tolist(), "rate": model.rate,
        "threshold": kappa, "horizon": model.horizon, "sigma0": model.sigma0,
        "clean_mean": model.mean.tolist(), "clean_covariance": model.covariance.tolist(),
        "target_mean": target_mean.tolist(), "target_covariance": target_covariance.tolist(),
        "path_transforms": [A.tolist() for A in transforms],
        "continuous_references": references, "runs": runs,
        "step_refinement": refinement, "particle_refinement_summary": particle_summary,
        "paired_particle_differences": paired_particle_differences,
        "passed": bool(oracle_passed and all(run["passed"] for run in runs)
                       and all(item["passed"] for item in refinement.values())),
    }


def run(spec, steps=(1000, 2000, 4000, 8000), particle_counts=(2000, 4000),
        seeds=(0, 1, 2), dimensions=(2, 4), rates=(0.9, 1.5, 1.95)):
    errors = validate(spec)
    if any(errors.values()):
        raise ValueError("Invalid model specification: " + "; ".join(
            error for category in errors.values() for error in category))
    steps, particle_counts = list(steps), list(particle_counts)
    if (len(steps) < 2 or any(type(k) is not int or k < 2 for k in steps)
            or steps != sorted(set(steps)) or any(steps[-1] % k for k in steps)):
        raise ValueError("steps must be increasing distinct integers dividing the finest count")
    if (len(particle_counts) < 2 or any(type(n) is not int or n < 200 for n in particle_counts)
            or particle_counts != sorted(set(particle_counts))):
        raise ValueError("samples must contain at least two increasing integers >= 200")
    if not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    if not dimensions or any(d not in (2, 4) for d in dimensions):
        raise ValueError("dimensions must be selected from 2 and 4")
    if not rates or any(not np.isfinite(r) or r <= 0 for r in rates):
        raise ValueError("rates must be finite and positive")
    schedule = spec["initial_smoothing"]
    cases = []
    for dimension in dimensions:
        B = np.asarray(spec["coordinates"]["B"], dtype=float)[:dimension, :dimension]
        mean = np.tile([0.3, -0.1], dimension // 2)
        covariance = np.kron(
            np.diag(np.arange(1, dimension // 2 + 1)), [[0.7, 0.12], [0.12, 0.4]]
        )
        identity = np.eye(dimension)
        decisive = np.diag(np.tile([np.sqrt(2), 1 / np.sqrt(2)], dimension // 2))
        squeeze = np.asarray(spec["arms"]["A"], dtype=float)[:dimension, :dimension]
        for rate in rates:
            horizon = float(np.log(schedule["sigmaT"] / schedule["sigma0"]) / rate)
            baseline = GaussianModel(B, identity, mean, covariance, rate, schedule["sigma0"], horizon)
            case = check_model(baseline, [identity, decisive, squeeze],
                               steps, particle_counts, seeds)
            case.update({"dimension": dimension, "arm": "baseline_with_fully_transported_Arm_A"})
            cases.append(case)
            for name, A in [("decisive_squeeze", decisive), ("recorded_squeeze", squeeze)]:
                armB = replace(baseline.transported(A), Q=identity)
                case = check_model(armB, [np.linalg.solve(A, identity)],
                                   steps, particle_counts, seeds)
                case.update({"dimension": dimension, "arm": "reisotropized_" + name,
                             "A": A.tolist()})
                cases.append(case)
    return {
        "scope": "Representative Gaussian pilot only; not full transform-grid or prerequisite certification",
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "steps": steps, "particle_counts": particle_counts, "seeds": list(seeds),
        "acceptance": {
            "continuous_exact_target_error": 1e-6, "transport_path_scaled_residual": PATH_BAND,
            "gaussian_moment_standard_errors": MOMENT_BAND,
            "finest_discrete_to_continuous_normalized_error": INTEGRATION_BAND,
            "refinement": "finest error <= 0.75 * coarsest error + 1e-7",
        },
        "methods": {
            "samplers": "Exact-score affine Heun ODE and Euler-Maruyama naive/fixed; "
                        "regression-tested against psdt.sample for isotropic models",
            "reference": "Independent adaptive DOP853 reverse mean/covariance ODE",
            "monte_carlo": "Compare particles to the exact discrete Gaussian law, separating "
                           "Monte Carlo fluctuations from integration bias",
            "coupling": "Same exact terminal draw at all K; sum fine Brownian increments for "
                        "coarse steps; use A L and shared increments for transported paths",
            "Arm_A": "Pure transport, not locally recomputed clipping",
            "Arm_B": "Locally computed spectral clipping in re-isotropized coordinates; "
                     "path check transports that entire sampler back",
            "particle_refinement": "Nested particle prefixes; independent seeds are replicates. "
                                   "No monotonic-error requirement for individual random draws.",
            "rate_sweep": "Fixed absolute rates across arms; each rate has its own shared horizon "
                          "T=log(sigmaT/sigma0)/r and exact sigmaT=40",
        },
        "limitations": [
            "Does not clear deterministic or forward-feasibility prerequisites.",
            "Does not assess mixture sampling, learned scores, or approximate terminal priors.",
            "Representative identity and squeezes only, including condition number 4, "
            "the largest condition number in the listed primary controls.",
            "Below-threshold naive bias is measured, not assumed universally nonzero.",
        ],
        "cases": cases, "passed": all(case["passed"] for case in cases),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", type=int, nargs="+", default=[1000, 2000, 4000, 8000])
    parser.add_argument("--samples", type=int, nargs="+", default=[2000, 4000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--dimensions", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--rates", type=float, nargs="+", default=[0.9, 1.5, 1.95])
    args = parser.parse_args()
    try:
        spec = json.loads(args.spec.read_text())
        report = run(spec, args.steps, args.samples, args.seeds, args.dimensions, args.rates)
        report["specification_path"] = str(args.spec.resolve())
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    except (OSError, ValueError, KeyError, np.linalg.LinAlgError) as error:
        print(f"Gaussian sampling: FAIL: {error}")
        return 1
    for case in report["cases"]:
        worst_path = max(
            residual for run in case["runs"] for level in run["levels"]
            for residual in level["path_max_scaled_residuals"]
        )
        worst_integration = max(
            item["normalized_errors_to_continuous_oracle"][-1]
            for item in case["step_refinement"].values()
        )
        print(f"[{'PASS' if case['passed'] else 'FAIL'}] {case['dimension']}D "
              f"{case['arm']} r={case['rate']:.6g}: "
              f"path={worst_path:.2e}, finest integration={worst_integration:.2e}")
    print(f"Gaussian sampling: {'PASS' if report['passed'] else 'FAIL'}")
    if args.output:
        print(f"Evidence: {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
