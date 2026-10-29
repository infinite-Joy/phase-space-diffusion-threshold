"""Representative exact-score mixture verification for the Section 9.8 gate.

Run with live console and file logs:
  OPENBLAS_NUM_THREADS=1 python verification/mixture_sampling.py \
    --output results/mixture_sampling.json --log-file results/mixture_sampling.log

Smaller primary-rate pilot:
  OPENBLAS_NUM_THREADS=1 python verification/mixture_sampling.py --rates 0.9 \
    --samples 128 256 --output results/mixture_sampling.json \
    --log-file results/mixture_sampling.log

Logs include case/seed starts, integration progress every 10%, and outcomes.
Requires a fixed gmm_exact.fit_gmm cache; never silently fits or changes the target.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from scipy.linalg import expm
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

if __package__:
    from .gaussian_sampling import (
        GaussianModel, MODES, PATH_BAND, diffusion_shapes, gaussian_kl, normalized_error,
    )
    from .forward_feasibility import threshold
    from .model_specification import DEFAULT_SPEC, validate
else:
    from gaussian_sampling import (
        GaussianModel, MODES, PATH_BAND, diffusion_shapes, gaussian_kl, normalized_error,
    )
    from forward_feasibility import threshold
    from model_specification import DEFAULT_SPEC, validate

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("mixture_sampling")
SCORE_BAND = 2e-5
METRIC_MARGINS = {
    "mean": 0.02, "covariance": 0.05, "sliced_w2": 0.05, "mmd2": 0.005,
    "component_mass_max_error": 0.01,
}


@dataclass
class Mixture:
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray

    def __post_init__(self):
        w, m, C = self.weights, self.means, self.covariances
        if (w.ndim != 1 or m.ndim != 2 or C.shape != (len(w), m.shape[1], m.shape[1])
                or len(w) != len(m) or len(w) < 1
                or not all(np.isfinite(a).all() for a in (w, m, C))
                or np.any(w <= 0) or not np.isclose(w.sum(), 1., rtol=0, atol=1e-10)):
            raise ValueError("Invalid mixture weights, means, or covariance shapes")
        if not np.allclose(C, C.transpose(0, 2, 1), rtol=0, atol=1e-12):
            raise ValueError("Mixture covariances must be symmetric")
        np.linalg.cholesky(C)

    def transported(self, A):
        return Mixture(self.weights.copy(), self.means @ A.T,
                       A @ self.covariances @ A.T)

    def moments(self):
        mean = self.weights @ self.means
        delta = self.means - mean
        covariance = np.einsum("k,kij->ij", self.weights, self.covariances)
        covariance += np.einsum("k,ki,kj->ij", self.weights, delta, delta)
        return mean, covariance

    def sample(self, n, rng):
        labels = rng.choice(len(self.weights), n, p=self.weights)
        roots = np.linalg.cholesky(self.covariances)
        z = self.means[labels] + np.einsum(
            "nij,nj->ni", roots[labels], rng.standard_normal((n, self.means.shape[1]))
        )
        return z, labels


def load_mixture(path):
    with np.load(path, allow_pickle=False) as data:
        return Mixture(*(np.asarray(data[key], dtype=float) for key in ("w", "mu", "S")))


def marginal(mixture, model, t):
    M = expm(model.B * t)
    sigma = model.sigma0 * np.exp(model.rate * t)
    return Mixture(mixture.weights, mixture.means @ M.T,
                   M @ mixture.covariances @ M.T + sigma**2 * model.Q)


def draw_marginal(mixture, model, t, n, rng):
    clean, labels = mixture.sample(n, rng)
    sigma = model.sigma0 * np.exp(model.rate * t)
    noise = rng.standard_normal(clean.shape) @ np.linalg.cholesky(model.Q).T
    return clean @ expm(model.B * t).T + sigma * noise, labels


def component_responsibilities(z, means, precision, constants):
    delta = z[:, None] - means
    Pd = np.einsum("kij,nkj->nki", precision, delta)
    logp = constants - np.sum(delta * Pd, axis=-1) / 2
    return np.exp(logp - logsumexp(logp, axis=1, keepdims=True)), Pd


class Score:
    def __init__(self, mixture, model, times):
        M = expm(np.asarray(times)[:, None, None] * model.B)
        self.means = mixture.means @ M.transpose(0, 2, 1)
        sigma = model.sigma0 * np.exp(model.rate * np.asarray(times))
        C = M[:, None] @ mixture.covariances @ M[:, None].transpose(0, 1, 3, 2)
        C += sigma[:, None, None, None]**2 * model.Q
        self.precision = np.linalg.inv(C)
        self.constants = np.log(mixture.weights) - np.linalg.slogdet(C)[1] / 2

    def responsibilities(self, z, index):
        return component_responsibilities(
            z, self.means[index], self.precision[index], self.constants[index]
        )

    def __call__(self, z, index):
        responsibility, Pd = self.responsibilities(z, index)
        return -np.einsum("nk,nki->ni", responsibility, Pd)


def independent_log_density(mixture, model, z, t):
    """Separate scalar-time/component implementation, using SciPy logpdf."""
    M = expm(t * model.B)
    sigma2 = model.sigma0**2 * np.exp(2 * model.rate * t)
    terms = [
        np.log(w) + multivariate_normal.logpdf(
            z, mean=M @ mean, cov=M @ C @ M.T + sigma2 * model.Q
        )
        for w, mean, C in zip(mixture.weights, mixture.means, mixture.covariances)
    ]
    return logsumexp(np.stack(terms, axis=-1), axis=-1)


def score_checks(mixture, model, A, seed=987):
    times = np.array([0., model.horizon / 2, model.horizon])
    score = Score(mixture, model, times)
    other = model.transported(A)
    other_mixture = mixture.transported(A)
    transported = Score(other_mixture, other, times)
    rng = np.random.default_rng(seed)
    records = []
    for index, t in enumerate(times):
        z, _ = draw_marginal(mixture, model, t, 24, rng)
        analytic = score(z, index)
        mean, covariance = marginal(mixture, model, t).moments()
        steps = 1e-5 * np.sqrt(np.diag(covariance))
        numeric = np.empty_like(z)
        for j, step in enumerate(steps):
            offset = np.zeros_like(mean)
            offset[j] = step
            numeric[:, j] = (
                independent_log_density(mixture, model, z + offset, t)
                - independent_log_density(mixture, model, z - offset, t)
            ) / (2 * step)
        finite_difference = float(np.max(np.abs(analytic - numeric)) /
                                  max(1., np.max(np.abs(analytic))))
        y = z @ A.T
        expected = np.linalg.solve(A.T, analytic.T).T
        transport = float(np.max(np.abs(transported(y, index) - expected)) /
                          max(1., np.max(np.abs(expected))))
        density = float(np.max(np.abs(
            independent_log_density(other_mixture, other, y, t)
            - independent_log_density(mixture, model, z, t)
            + np.linalg.slogdet(A)[1]
        )))
        records.append({
            "t": float(t), "finite_difference_scaled_error": finite_difference,
            "score_transport_scaled_error": transport, "log_density_jacobian_error": density,
            "passed": bool(finite_difference < SCORE_BAND and transport < PATH_BAND
                           and density < 1e-8),
        })
    return {"checks": records, "passed": all(item["passed"] for item in records)}


def advance(z, score, model, G, D, L, sigmas, index, stride, h, mode, increment):
    if mode not in MODES:
        raise ValueError(f"Unknown sampler: {mode}")

    def drift(value, i):
        W = G / 2 if mode == "ode" else D if mode == "naive" else (G + D) / 2
        return value @ model.B.T - sigmas[i]**2 * (score(value, i) @ W.T)

    first = drift(z, index)
    if mode == "ode":
        return z - h / 2 * (first + drift(z - h * first, index + stride))
    return z - h * first + sigmas[index] * (increment @ L.T)


def coupled_samples(mixture, model, A, steps, particles, seed):
    """Same terminal draws and summed fine Brownian increments at every level."""
    started = time.monotonic()
    LOGGER.info("Seed %s: preparing scores; particles=%s, grids=%s", seed, particles, steps)
    times = np.linspace(model.horizon, 0., steps[-1] + 1)
    sigmas = model.sigma0 * np.exp(model.rate * times)
    score = Score(mixture, model, times)
    other = model.transported(A)
    other_score = Score(mixture.transported(A), other, times)
    G, D, L = diffusion_shapes(model)
    terminal_stream, brownian_stream = np.random.SeedSequence(seed).spawn(2)
    terminal, labels = draw_marginal(
        mixture, model, model.horizon, particles, np.random.default_rng(terminal_stream)
    )
    rng = np.random.default_rng(brownian_stream)
    subset = min(8, particles)
    levels = {
        k: {
            "increment": np.zeros_like(terminal),
            "states": {mode: terminal.copy() for mode in MODES},
            "transported": {mode: terminal[:subset] @ A.T for mode in MODES},
            "path_errors": {mode: 0. for mode in MODES},
        } for k in steps
    }
    fine_h = model.horizon / steps[-1]
    log_interval = max(1, steps[-1] // 10)
    LOGGER.info("Seed %s: integrating all three samplers and transported paths", seed)
    for fine_index in range(steps[-1]):
        increment = np.sqrt(fine_h) * rng.standard_normal(terminal.shape)
        for k, level in levels.items():
            level["increment"] += increment
            stride = steps[-1] // k
            if (fine_index + 1) % stride:
                continue
            index = fine_index + 1 - stride
            h = times[index] - times[index + stride]
            for mode in MODES:
                z = advance(level["states"][mode], score, model, G, D, L,
                            sigmas, index, stride, h, mode, level["increment"])
                y = advance(level["transported"][mode], other_score, other,
                            A @ G @ A.T, A @ D @ A.T, A @ L,
                            sigmas, index, stride, h, mode, level["increment"][:subset])
                if not np.isfinite(z).all() or not np.isfinite(y).all():
                    raise ValueError(f"Nonfinite {mode} trajectory at K={k}, seed={seed}")
                pulled = np.linalg.solve(A, y.T).T
                residual = float(np.max(np.abs(pulled - z[:subset])) /
                                 max(1., np.max(np.abs(z[:subset]))))
                level["path_errors"][mode] = max(level["path_errors"][mode], residual)
                level["states"][mode], level["transported"][mode] = z, y
            level["increment"].fill(0)
        completed = fine_index + 1
        if completed % log_interval == 0 or completed == steps[-1]:
            LOGGER.info("Seed %s: step %s/%s (%.0f%%), elapsed %.1fs",
                        seed, completed, steps[-1], 100 * completed / steps[-1],
                        time.monotonic() - started)
    counts = np.bincount(labels, minlength=len(mixture.weights))
    expected = particles * mixture.weights
    frequency_score = float(np.max(np.abs(counts - expected) /
                                  np.sqrt(expected * (1 - mixture.weights) + 1)))
    return terminal, counts, frequency_score, levels


def metrics(z, reference, target, projections, bandwidth):
    mean, covariance = target.moments()
    observed_mean, observed_covariance = z.mean(0), np.cov(z, rowvar=False)
    errors = normalized_error(observed_mean, observed_covariance, mean, covariance)
    pz, pr = np.sort(z @ projections, axis=0), np.sort(reference @ projections, axis=0)
    scale = np.sqrt(np.trace(covariance) / len(mean))

    def kernel_sum(x, y, diagonal=False):
        total = 0.
        for start in range(0, len(x), 256):
            block = x[start:start + 256]
            distances = (block**2).sum(1)[:, None] + (y**2).sum(1) - 2 * block @ y.T
            kernel = np.exp(-np.maximum(distances, 0) / (2 * bandwidth**2))
            if diagonal:
                kernel[np.arange(len(block)), start + np.arange(len(block))] = 0
            total += kernel.sum()
        return float(total)

    n, m = len(z), len(reference)
    mmd = (kernel_sum(z, z, True) / (n * (n - 1))
           + kernel_sum(reference, reference, True) / (m * (m - 1))
           - 2 * kernel_sum(z, reference) / (n * m))
    responsibilities, _ = component_responsibilities(
        z, target.means, np.linalg.inv(target.covariances),
        np.log(target.weights) - np.linalg.slogdet(target.covariances)[1] / 2,
    )
    return {
        "mean": errors["mean"], "covariance": errors["covariance"],
        "sliced_w2": float(np.sqrt(np.mean((pz - pr)**2)) / scale),
        "mmd2": float(mmd),
        "gaussian_moment_kl_proxy": gaussian_kl(
            observed_mean, observed_covariance, mean, covariance),
        "variance_ratios": (np.diag(observed_covariance) / np.diag(covariance)).tolist(),
        "component_responsibility_mass": responsibilities.mean(0).tolist(),
        "component_mass_max_error": float(np.max(
            np.abs(responsibilities.mean(0) - target.weights))),
    }


def calibration(target, particle_counts, seed, repeats):
    streams = np.random.SeedSequence(seed).spawn(3)
    rng = np.random.default_rng(streams[0])
    pilot, _ = target.sample(512, rng)
    pairs = pilot[:256] - pilot[256:]
    bandwidth = float(np.sqrt(np.median(np.sum(pairs**2, axis=1))))
    if bandwidth <= 0:
        raise ValueError("Nonpositive target calibration bandwidth")
    projections = rng.standard_normal((target.means.shape[1], 32))
    projections /= np.linalg.norm(projections, axis=0)
    reference, _ = target.sample(particle_counts[-1], np.random.default_rng(streams[1]))
    rng = np.random.default_rng(streams[2])
    records, bands = {}, {}
    for n in particle_counts:
        controls = []
        for _ in range(repeats):
            sample, _ = target.sample(n, rng)
            controls.append(metrics(sample, reference[:n], target, projections, bandwidth))
        bands[n] = {
            key: float(max(item[key] for item in controls) + METRIC_MARGINS[key])
            for key in METRIC_MARGINS
        }
        records[str(n)] = {"target_vs_target": controls, "acceptance_bands": bands[n]}
    return reference, projections, bandwidth, bands, records


def quality_passed(values, bands):
    return bool(all(np.isfinite(values[key]) and values[key] <= band
                    for key, band in bands.items()))


def check_case(mixture, model, A, pullback, steps, particle_counts, seeds, repeats):
    pulled_mixture = mixture.transported(pullback)
    pulled_model = model.transported(pullback)
    target = marginal(pulled_mixture, pulled_model, 0.)
    LOGGER.info("Calibrating target metrics: particle counts=%s, controls per count=%s",
                particle_counts, repeats)
    reference, projections, bandwidth, bands, controls = calibration(
        target, particle_counts, 21991, repeats
    )
    scores = score_checks(mixture, model, A)
    LOGGER.info("Independent score/density checks: %s", "PASS" if scores["passed"] else "FAIL")
    runs = []
    for seed in seeds:
        terminal, counts, frequency_score, levels = coupled_samples(
            mixture, model, A, steps, particle_counts[-1], seed
        )
        terminal_law = marginal(mixture, model, model.horizon)
        terminal_mean, terminal_covariance = terminal_law.moments()
        # This is a generous initialization smoke check, not a Gaussian-law SE claim.
        terminal_mean_score = float(normalized_error(
            terminal.mean(0), np.cov(terminal, rowvar=False),
            terminal_mean, terminal_covariance,
        )["mean"] * np.sqrt(len(terminal)))
        run = {
            "seed": seed, "terminal_component_counts": counts.tolist(),
            "terminal_component_frequency_score": frequency_score,
            "terminal_whitened_mean_sqrt_n": terminal_mean_score,
            "levels": [],
        }
        for k, level in levels.items():
            for mode in MODES:
                z = level["states"][mode] @ pullback.T
                values = []
                for n in particle_counts:
                    observed = metrics(z[:n], reference[:n], target, projections, bandwidth)
                    values.append({"samples": n, "metrics": observed,
                                   "within_target_calibration": quality_passed(observed, bands[n])})
                run["levels"].append({
                    "steps": k, "mode": mode, "particle_refinement": values,
                    "path_max_scaled_residual": level["path_errors"][mode],
                })
        scale = np.sqrt(np.trace(target.moments()[1]))
        run["coupled_step_differences"] = {
            mode: [
                float(np.sqrt(np.mean(np.sum((
                    (levels[a]["states"][mode] - levels[b]["states"][mode]) @ pullback.T
                )**2, axis=1))) / scale)
                for a, b in zip(steps, steps[1:])
            ] for mode in MODES
        }
        run["paired_naive_minus_fixed"] = {
            key: float(
                next(x for x in run["levels"] if x["steps"] == steps[-1]
                     and x["mode"] == "naive")["particle_refinement"][-1]["metrics"][key]
                - next(x for x in run["levels"] if x["steps"] == steps[-1]
                       and x["mode"] == "fixed")["particle_refinement"][-1]["metrics"][key]
            ) for key in METRIC_MARGINS
        }
        run["passed"] = bool(
            frequency_score < 8 and terminal_mean_score < 8
            and all(level["path_max_scaled_residual"] < PATH_BAND for level in run["levels"])
        )
        LOGGER.info("Seed %s: initialization/path checks %s; diagnostics complete",
                    seed, "PASS" if run["passed"] else "FAIL")
        runs.append(run)
    kappa = threshold(model.B, model.Q)[0]
    refinement = {}
    for mode in MODES:
        exact = mode != "naive" or model.rate >= kappa - 1e-12
        averages = {}
        for k in steps:
            items = [
                next(level for level in run["levels"]
                     if level["steps"] == k and level["mode"] == mode)
                ["particle_refinement"][-1]["metrics"] for run in runs
            ]
            averages[str(k)] = {
                key: float(np.mean([item[key] for item in items])) for key in METRIC_MARGINS
            }
        fine_pass = quality_passed(averages[str(steps[-1])], bands[particle_counts[-1]])
        # Allow calibration-scale fluctuations; do not demand monotone random errors.
        converging = all(
            averages[str(steps[-1])][key] <= averages[str(steps[0])][key]
            + bands[particle_counts[-1]][key] for key in METRIC_MARGINS
        )
        refinement[mode] = {
            "expected_exact_target": bool(exact), "seed_mean_metrics": averages,
            "finest_within_target_calibration": fine_pass,
            "no_degradation_beyond_calibration": bool(converging),
            "passed": bool(not exact or (fine_pass and converging)),
        }
    return {
        "B": model.B.tolist(), "Q": model.Q.tolist(), "rate": model.rate,
        "horizon": model.horizon, "threshold": float(kappa),
        "path_transform": A.tolist(), "diagnostic_pullback": pullback.tolist(),
        "target_component_means": target.means.tolist(),
        "target_component_covariances": target.covariances.tolist(),
        "score_validation": scores, "kernel_bandwidth": bandwidth,
        "projections": projections.tolist(), "calibration": controls,
        "runs": runs, "step_refinement": refinement,
        "paired_seed_summary": {
            key: {
                "values": [run["paired_naive_minus_fixed"][key] for run in runs],
                "mean": float(np.mean([run["paired_naive_minus_fixed"][key] for run in runs])),
                "standard_error": float(np.std(
                    [run["paired_naive_minus_fixed"][key] for run in runs], ddof=1
                ) / np.sqrt(len(runs))),
            } for key in METRIC_MARGINS
        },
        "passed": bool(scores["passed"] and all(run["passed"] for run in runs)
                       and all(value["passed"] for value in refinement.values())),
    }


def run(spec, mixture, steps=(1000, 2000, 4000), particle_counts=(512, 1024),
        seeds=(0, 1, 2), rates=(0.9, 1.5, 1.95), calibration_repeats=12):
    errors = validate(spec)
    if any(errors.values()):
        raise ValueError("Invalid model specification: " + "; ".join(
            error for category in errors.values() for error in category))
    steps, particle_counts = list(steps), list(particle_counts)
    if (len(steps) < 2 or steps != sorted(set(steps))
            or any(type(k) is not int or k < 2 or steps[-1] % k for k in steps)):
        raise ValueError("steps must be increasing integers dividing the finest count")
    if (len(particle_counts) < 2 or particle_counts != sorted(set(particle_counts))
            or any(type(n) is not int or n < 100 for n in particle_counts)):
        raise ValueError("samples must contain at least two increasing integers >= 100")
    if (len(seeds) < 2 or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or seed < 0 for seed in seeds)):
        raise ValueError("seeds must contain at least two distinct nonnegative integers")
    if not rates or any(not np.isfinite(r) or r <= 0 for r in rates):
        raise ValueError("rates must be finite and positive")
    if type(calibration_repeats) is not int or calibration_repeats < 4:
        raise ValueError("calibration-repeats must be an integer >= 4")
    B = np.asarray(spec["coordinates"]["B"], dtype=float)
    if mixture.means.shape[1] != len(B):
        raise ValueError("Mixture dimension does not match the model specification")
    identity = np.eye(len(B))
    decisive = np.diag(np.tile([np.sqrt(2), 1 / np.sqrt(2)], len(B) // 2))
    recorded = np.asarray(spec["arms"]["A"], dtype=float)
    mean, covariance = mixture.moments()
    schedule = spec["initial_smoothing"]
    cases = []
    total_cases = 3 * len(rates)
    for rate in rates:
        model = GaussianModel(B, identity, mean, covariance, rate, schedule["sigma0"],
                              float(np.log(schedule["sigmaT"] / schedule["sigma0"]) / rate))
        for name, A, armB in [
            ("baseline_with_transported_Arm_A_recorded_squeeze", recorded, False),
            ("reisotropized_decisive_squeeze", decisive, True),
            ("reisotropized_recorded_squeeze", recorded, True),
        ]:
            LOGGER.info("Case %s/%s: %s r=%g", len(cases) + 1, total_cases, name, rate)
            started = time.monotonic()
            actual = model.transported(A) if armB else model
            actual_mixture = mixture.transported(A) if armB else mixture
            if armB:
                actual.Q = identity
            inverse = np.linalg.solve(A, identity)
            case = check_case(actual_mixture, actual, inverse if armB else A,
                              inverse if armB else identity,
                              steps, particle_counts, seeds, calibration_repeats)
            case["arm"] = name
            cases.append(case)
            LOGGER.info("Case %s/%s: %s r=%g %s, elapsed %.1fs",
                        len(cases), total_cases, name, rate,
                        "PASS" if case["passed"] else "FAIL", time.monotonic() - started)
    return {
        "scope": "Representative exact-score mixture pilot, not full gate certification",
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "steps": steps, "particle_counts": particle_counts, "seeds": list(seeds),
        "rates": list(rates), "calibration_repeats": calibration_repeats,
        "mixture": {"weights": mixture.weights.tolist(), "means": mixture.means.tolist(),
                    "covariances": mixture.covariances.tolist()},
        "acceptance": {
            "score_finite_difference_scaled_error": SCORE_BAND,
            "path_scaled_residual": PATH_BAND, "log_density_jacobian_error": 1e-8,
            "quality": "Seed-mean finest metrics <= maximum independent target-control "
                       "metric + predeclared practical margin; no degradation beyond that band",
            "practical_margins": METRIC_MARGINS,
        },
        "methods": {
            "initialization": "Exact terminal mixture, not moment-matched Gaussian",
            "target": "Clean mixture plus sigma0^2 Q smoothing, pulled back per arm",
            "coupling": "Shared terminal particles and Brownian increments across modes; "
                        "summed fine increments across grids; A L for full transport",
            "metrics": "Common pulled-back coordinates; fixed shared target reference, "
                       "32 projections and independent target-pilot median kernel bandwidth",
            "streams": "Separate terminal/Brownian, calibration, and score-check RNG streams",
            "uncertainty": "Paired metric differences and SE across independent seeds; "
                           "nested particle prefixes are not independent replicates",
        },
        "limitations": [
            "Does not clear the deterministic prerequisites or full transformation grid.",
            "Empirical calibration bands are smoke-test acceptance, not formal equivalence intervals.",
            "Finite-sample negative unbiased MMD squared is allowed.",
            "Gaussian-moment KL is a proxy, not true mixture KL; modes may overlap.",
            "Below-threshold naive target bias is measured, not required to vanish.",
            "Particle refinement is reported, not required to be monotone.",
            "No learned scores, approximate priors, or local re-clipping Arm A intervention.",
        ],
        "cases": cases, "passed": all(case["passed"] for case in cases),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--mixture", type=Path, default=ROOT / "results/gmm_fit.npz")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log-file", type=Path,
                        help="Append timestamped progress to this file as well as stderr")
    parser.add_argument("--steps", type=int, nargs="+", default=[1000, 2000, 4000])
    parser.add_argument("--samples", type=int, nargs="+", default=[512, 1024])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rates", type=float, nargs="+", default=[0.9, 1.5, 1.95])
    parser.add_argument("--calibration-repeats", type=int, default=12)
    args = parser.parse_args()
    try:
        handlers = [logging.StreamHandler(sys.stderr)]
        if args.log_file:
            args.log_file.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                            handlers=handlers, force=True)
        LOGGER.info("Starting mixture sampling: mixture=%s, rates=%s, steps=%s, samples=%s, seeds=%s",
                    args.mixture, args.rates, args.steps, args.samples, args.seeds)
        mixture = load_mixture(args.mixture)
        report = run(json.loads(args.spec.read_text()), mixture, args.steps, args.samples,
                     args.seeds, args.rates, args.calibration_repeats)
        report["mixture_path"] = str(args.mixture.resolve())
        report["mixture_sha256"] = hashlib.sha256(args.mixture.read_bytes()).hexdigest()
        report["specification_path"] = str(args.spec.resolve())
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            LOGGER.info("Evidence written: %s", args.output)
    except (OSError, ValueError, KeyError, np.linalg.LinAlgError) as error:
        LOGGER.error("Mixture sampling: FAIL: %s", error)
        return 1
    except KeyboardInterrupt:
        LOGGER.warning("Mixture sampling interrupted; no completed report written")
        return 130
    LOGGER.info("Mixture sampling: %s", "PASS" if report["passed"] else "FAIL")
    for case in report["cases"]:
        print(f"[{'PASS' if case['passed'] else 'FAIL'}] {case['arm']} r={case['rate']:.6g}")
    print(f"Mixture sampling: {'PASS' if report['passed'] else 'FAIL'}")
    if args.output:
        print(f"Evidence: {args.output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
