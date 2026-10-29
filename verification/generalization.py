"""Section 11 generalization experiments (expensive runs are opt-in).

Create a manifest without executing it:
  python verification/generalization.py --profile full --write-spec /tmp/generalization.json
Run selected cases, preserving per-case evidence:
  OPENBLAS_NUM_THREADS=1 python verification/generalization.py --axis data --resume
"""
import argparse
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import logging
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.integrate import solve_ivp
from scipy.linalg import eigh, expm, solve_triangular
from scipy.special import logsumexp
from scipy.stats import multivariate_normal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psdt import MLP, two_moons
from verification.forward_feasibility import (
    integrated_covariance, threshold, transition_covariance,
)
from verification.gaussian_sampling import (
    GaussianModel, MODES, diffusion_shapes, gaussian_kl, normalized_error,
)
from verification.mixture_sampling import (
    METRIC_MARGINS, Mixture, advance, load_mixture, metrics, quality_passed,
)
from verification.model_specification import DEFAULT_SPEC, validate

LOG = logging.getLogger("generalization")
AXES = ("noise", "data", "prior", "score", "trained", "interaction")
DATA = ("gaussian", "separated", "overlapping", "imbalanced", "fitted_moons", "raw_moons")
PRIORS = ("exact", "moment", "estimated", "noise_only", "shift", "scale_0.8", "scale_1.2")
SCORES = ("exact", "zero", "bias", "scale", "smooth", "independent_bias", "learned")
ARMS = ("baseline", "transported", "reisotropized", "retrained_A")
CHECK_BAND = 2e-5
PATH_BAND = 1e-8


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def noise_shapes():
    raw = {
        "isotropic": np.eye(4),
        "energy": np.diag([0.25, 1., 0.25, 1.]),
        "anisotropic": np.diag([4., 1., 0.25, 1.]),
    }
    L = np.eye(4)
    L[1, 0], L[2, 0], L[3, 1] = 0.5, 0.3, -0.4
    raw["correlated"] = L @ L.T
    shapes = {"isotropic": {"raw": raw["isotropic"].tolist(),
                             "Q": raw["isotropic"].tolist(), "normalization": "none"}}
    for name in ("energy", "anisotropic", "correlated"):
        Q = raw[name]
        for policy, scale in (("trace", 4 / np.trace(Q)),
                              ("det", np.exp(-np.linalg.slogdet(Q)[1] / 4))):
            shapes[f"{name}_{policy}"] = {
                "raw": Q.tolist(), "Q": (scale * Q).tolist(), "normalization": policy,
            }
    return shapes


def transforms(primary):
    cross = np.eye(4)
    cross[0, 2], cross[3, 1] = 0.3, -0.3
    jacobian = cross.copy()
    jacobian[0] *= 1.5
    return {
        "identity": np.eye(4).tolist(),
        "decisive": np.diag([np.sqrt(2), 1 / np.sqrt(2)] * 2).tolist(),
        "cross_pair": cross.tolist(),
        "recorded": primary["arms"]["A"],
        "jacobian": jacobian.tolist(),
    }


def make_spec(profile="smoke", mixture_path=None):
    primary = json.loads(DEFAULT_SPEC.read_text())
    full = profile == "full"
    spec = {
        "version": 1, "profile": profile, "primary_spec": primary,
        "shapes": noise_shapes(), "transforms": transforms(primary),
        "mixture_path": str(mixture_path or ROOT / "results/gmm_fit.npz"),
        "steps": [1000, 2000, 4000, 8000] if full else [128, 256],
        "particles": [2000, 8000] if full else [64, 128],
        "seeds": list(range(5)) if full else [0, 1],
        "training_seeds": [0, 1, 2] if full else [0],
        "calibration_repeats": 12 if full else 4,
        "empirical_points": 2048 if full else 64,
        "estimated_moment_points": 8000 if full else 512,
        "training": {"updates": 12000 if full else 20, "width": 128 if full else 16,
                     "depth": 3 if full else 1, "batch": 512 if full else 32,
                     "learning_rate": 0.002, "validation_size": 512 if full else 32,
                     "validate_every": 1000 if full else 10},
        "acceptance": {"score_residual": CHECK_BAND, "path_residual": PATH_BAND,
                       "metric_margins": METRIC_MARGINS, "gaussian_moment_margin": 0.05},
        "cases": [],
    }

    def add(axis, data="gaussian", shape="isotropic", prior="exact", score="exact",
            amplitude=0., rate=1.95, arm="baseline", transform="decisive", train_seed=None):
        case = dict(axis=axis, data=data, shape=shape, prior=prior, score=score,
                    amplitude=amplitude, rate=rate, arm=arm, transform=transform,
                    training_seed=train_seed)
        case["id"] = f"{axis}-{data}-{digest(case)[:12]}"
        spec["cases"].append(case)

    rates = [0.9, 1.95] if full else [1.95]
    maps = ["identity", "decisive", "cross_pair"] if full else ["decisive"]
    arms = ["baseline", "transported", "reisotropized"]
    for shape in spec["shapes"]:
        Q = np.array(spec["shapes"][shape]["Q"])
        kappa = threshold(np.array(primary["coordinates"]["B"]), Q)[0]
        shape_rates = rates + ([0.9 * kappa, kappa, 1.1 * kappa]
                              if full and kappa > 1e-10 else [])
        for rate in sorted(set(shape_rates)):
            for data in ("gaussian", "fitted_moons"):
                for transform in maps:
                    for arm in arms:
                        add("noise", data=data, shape=shape, rate=rate, arm=arm,
                            transform=transform)
    for data in DATA:
        for rate in rates:
            for transform in maps:
                for arm in arms:
                    add("data", data=data, rate=rate, arm=arm, transform=transform)
    for data in (DATA if full else ("gaussian", "separated")):
        for prior in PRIORS:
            for rate in rates:
                for arm in arms:
                    add("prior", data=data, prior=prior, arm=arm, rate=rate)
    for data in (("gaussian", "separated", "fitted_moons") if full else ("gaussian",)):
        for kind in ("zero", "bias", "scale", "smooth", "independent_bias"):
            for amplitude in ([0.05, 0.1, 0.2] if full else [0.1]):
                for rate in rates:
                    for arm in arms:
                        add("score", data=data, score=kind, amplitude=amplitude, arm=arm, rate=rate)
    for data in DATA:
        for shape in (spec["shapes"] if full else ("isotropic",)):
            for seed in spec["training_seeds"]:
                for arm in (*arms, "retrained_A"):
                    for transform in maps:
                        for rate in rates:
                            add("trained", data=data, shape=shape, score="learned",
                                train_seed=seed, arm=arm, transform=transform, rate=rate)
    for score in ("exact", "learned"):
        for seed in (spec["training_seeds"] if score == "learned" else [None]):
            for prior in ("exact", "moment"):
                for arm in arms:
                    for rate in rates:
                        add("interaction", data="imbalanced", shape="correlated_trace",
                            prior=prior, score=score, train_seed=seed, arm=arm, rate=rate)
    return spec


def validate_spec(spec):
    if spec["version"] != 1 or spec["profile"] not in ("smoke", "full", "custom"):
        raise ValueError("Unsupported manifest version/profile")
    errors = validate(spec["primary_spec"])
    if any(errors.values()):
        raise ValueError(f"Invalid primary specification: {errors}")
    for name in ("steps", "particles", "seeds", "training_seeds"):
        items = spec[name]
        minimum = 2 if name in ("steps", "particles") else 1
        if (len(items) < minimum or items != sorted(set(items))
                or any(type(x) is not int or x < (2 if name in ("steps", "particles") else 0)
                       for x in items)):
            raise ValueError(f"Invalid {name}: require sorted distinct integer values")
    if any(spec["steps"][-1] % k for k in spec["steps"]):
        raise ValueError("All step counts must divide the finest grid")
    if spec["particles"][0] < 8:
        raise ValueError("At least eight particles are required for covariance metrics")
    for key, minimum in (("calibration_repeats", 4), ("empirical_points", 2),
                         ("estimated_moment_points", 5)):
        if type(spec[key]) is not int or spec[key] < minimum:
            raise ValueError(f"Invalid {key}")
    for key in ("updates", "width", "depth", "batch", "validation_size", "validate_every"):
        if type(spec["training"][key]) is not int or spec["training"][key] < 1:
            raise ValueError(f"Invalid training {key}")
    if not np.isfinite(spec["training"]["learning_rate"]) or spec["training"]["learning_rate"] <= 0:
        raise ValueError("Invalid learning rate")
    for name, shape in spec["shapes"].items():
        for key in ("Q", "raw"):
            Q = np.array(shape[key], dtype=float)
            if (Q.shape != (4, 4) or not np.isfinite(Q).all()
                    or not np.allclose(Q, Q.T, atol=1e-13, rtol=0)):
                raise ValueError(f"Invalid shape {name}/{key}")
            np.linalg.cholesky(Q)
        Q, raw = np.array(shape["Q"]), np.array(shape["raw"])
        policy = shape["normalization"]
        expected = {"none": raw, "trace": 4 * raw / np.trace(raw),
                    "det": raw * np.exp(-np.linalg.slogdet(raw)[1] / 4)}
        if policy not in expected or not np.allclose(Q, expected[policy], rtol=1e-12, atol=1e-12):
            raise ValueError(f"Shape {name} does not match its normalization")
    for name, value in spec["transforms"].items():
        A = np.array(value, dtype=float)
        if A.shape != (4, 4) or not np.isfinite(A).all() or np.linalg.cond(A) > 1e6:
            raise ValueError(f"Invalid or ill-conditioned transform {name}")
    acceptance = spec["acceptance"]
    if set(acceptance["metric_margins"]) != set(METRIC_MARGINS):
        raise ValueError("Metric margin keys must match the declared metrics")
    if any(not np.isfinite(x) or x <= 0 for x in (
            acceptance["score_residual"], acceptance["path_residual"],
            acceptance["gaussian_moment_margin"],
            *acceptance["metric_margins"].values())):
        raise ValueError("Acceptance tolerances must be positive and finite")
    ids = set()
    for case in spec["cases"]:
        if (not case["id"] or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
                                  for c in case["id"]) or case["id"] in ids):
            raise ValueError("Case IDs must be unique safe filenames")
        ids.add(case["id"])
        if (case["axis"] not in AXES or case["data"] not in DATA
                or case["shape"] not in spec["shapes"] or case["prior"] not in PRIORS
                or case["score"] not in SCORES or case["arm"] not in ARMS
                or case["transform"] not in spec["transforms"]
                or not np.isfinite(case["rate"]) or case["rate"] <= 0
                or not np.isfinite(case["amplitude"]) or case["amplitude"] < 0):
            raise ValueError(f"Invalid case {case['id']}")
        if case["score"] == "learned" and case["training_seed"] not in spec["training_seeds"]:
            raise ValueError(f"Missing training seed for {case['id']}")
        if case["arm"] == "retrained_A" and case["score"] != "learned":
            raise ValueError("retrained_A requires a learned score")
    if not spec["cases"]:
        raise ValueError("Manifest has no cases")


@dataclass
class Law:
    weights: np.ndarray
    means: np.ndarray
    covariances: np.ndarray
    empirical: bool = False

    def moments(self):
        mean = self.weights @ self.means
        delta = self.means - mean
        return mean, np.einsum("k,kij->ij", self.weights, self.covariances) + (
            delta.T * self.weights) @ delta

    def sample(self, n, rng):
        labels = rng.choice(len(self.weights), n, p=self.weights)
        if self.empirical:
            return self.means[labels].copy(), labels
        roots = np.linalg.cholesky(self.covariances)
        return self.means[labels] + np.einsum(
            "nij,nj->ni", roots[labels], rng.standard_normal((n, 4))), labels

    def transported(self, A):
        return Law(self.weights.copy(), self.means @ A.T,
                   A @ self.covariances @ A.T, self.empirical)

    def record(self):
        return {"weights": self.weights.tolist(), "means": self.means.tolist(),
                "covariances": self.covariances.tolist(), "empirical": self.empirical}


def make_data(name, spec):
    heldout = None
    if name == "fitted_moons":
        mixture = load_mixture(Path(spec["mixture_path"]))
        law = Law(mixture.weights, mixture.means, mixture.covariances)
    elif name == "raw_moons":
        points = two_moons(spec["empirical_points"], np.random.default_rng(401))
        heldout = two_moons(spec["empirical_points"], np.random.default_rng(402))
        law = Law(np.full(len(points), 1 / len(points)), points,
                  np.zeros((len(points), 4, 4)), True)
    elif name == "gaussian":
        L = np.eye(4)
        L[1, 0], L[3, 2] = 0.2, -0.15
        C = L @ np.diag([0.6, 0.15, 0.4, 0.2]) @ L.T
        law = Law(np.ones(1), np.array([[0.3, -0.1, -0.2, 0.1]]), C[None])
    else:
        separation = 0.3 if name == "overlapping" else 2.
        weights = np.array([0.9, 0.1]) if name == "imbalanced" else np.array([0.5, 0.5])
        law = Law(weights, np.array([[separation, 0, 0, 0], [-separation, 0, 0, 0]]),
                  np.repeat((0.1 * np.eye(4))[None], 2, axis=0))
    if law.means.shape[1] != 4:
        raise ValueError("Clean data dimension must be four")
    return law, heldout


def model_for(law, spec, case):
    B = np.array(spec["primary_spec"]["coordinates"]["B"], dtype=float)
    Q = np.array(spec["shapes"][case["shape"]]["Q"], dtype=float)
    schedule = spec["primary_spec"]["initial_smoothing"]
    mean, covariance = law.moments()
    return GaussianModel(B, Q, mean, covariance, case["rate"], schedule["sigma0"],
                         float(np.log(schedule["sigmaT"] / schedule["sigma0"]) / case["rate"]))


def marginal_law(law, model, t):
    M = expm(model.B * t)
    return Mixture(law.weights, law.means @ M.T, M @ law.covariances @ M.T
                   + (model.sigma0 * np.exp(model.rate * t))**2 * model.Q)


def draw(law, model, t, n, rng):
    clean, _ = law.sample(n, rng)
    return clean @ expm(model.B * t).T + (
        model.sigma0 * np.exp(model.rate * t)
        * rng.standard_normal((n, 4)) @ np.linalg.cholesky(model.Q).T)


class Oracle:
    """Chunk over both points and components, including point-mass clean laws."""

    def __init__(self, law, model):
        self.law, self.model = law, model
        self.cache = OrderedDict()

    def parameters(self, t):
        t = float(t)
        if t not in self.cache:
            marginal = marginal_law(self.law, self.model, t)
            C = marginal.covariances
            self.cache[t] = (marginal.means, np.linalg.inv(C),
                             np.log(self.law.weights) - 2 * np.log(2 * np.pi)
                             - np.linalg.slogdet(C)[1] / 2)
            if len(self.cache) > 3:
                self.cache.popitem(last=False)
        return self.cache[t]

    def __call__(self, z, t):
        means, precision, constants = self.parameters(t)
        output = np.empty_like(z)
        for start in range(0, len(z), 128):
            points = z[start:start + 128]
            total_log = np.full(len(points), -np.inf)
            total_score = np.zeros_like(points)
            for first in range(0, len(means), 128):
                sl = slice(first, first + 128)
                delta = points[:, None] - means[sl]
                Pd = np.einsum("kij,nkj->nki", precision[sl], delta)
                logs = constants[sl] - np.sum(delta * Pd, axis=-1) / 2
                block_log = logsumexp(logs, axis=1)
                block_score = -np.einsum(
                    "nk,nki->ni", np.exp(logs - block_log[:, None]), Pd)
                combined = np.logaddexp(total_log, block_log)
                total_score = (np.exp(total_log - combined)[:, None] * total_score
                               + np.exp(block_log - combined)[:, None] * block_score)
                total_log = combined
            output[start:start + len(points)] = total_score
        return output


def independent_density(law, model, z, t):
    marginal = marginal_law(law, model, t)
    result = np.full(len(z), -np.inf)
    for weight, mean, covariance in zip(law.weights, marginal.means, marginal.covariances):
        result = np.logaddexp(result, np.log(weight) + multivariate_normal.logpdf(
            z, mean=mean, cov=covariance))
    return result


def score_checks(law, model, A, band):
    oracle = Oracle(law, model)
    other_law, other_model = law.transported(A), model.transported(A)
    other = Oracle(other_law, other_model)
    density_A = A.copy()
    if abs(np.linalg.slogdet(density_A)[1]) < 1e-8:
        density_A[0] *= 1.5
    density_law, density_model = law.transported(density_A), model.transported(density_A)
    records = []
    for index, t in enumerate([0., model.horizon / 2, model.horizon]):
        z = draw(law, model, t, 6, np.random.default_rng(901 + index))
        analytic = oracle(z, t)
        # At t=0 empirical kernels have width sigma0, not the total data spread.
        steps = 1e-5 * np.sqrt(np.diag(
            (model.sigma0 * np.exp(model.rate * t))**2 * model.Q))
        numeric = np.empty_like(z)
        for j, h in enumerate(steps):
            offset = np.eye(4)[j] * h
            numeric[:, j] = (independent_density(law, model, z + offset, t)
                             - independent_density(law, model, z - offset, t)) / (2 * h)
        expected = np.linalg.solve(A.T, analytic.T).T
        records.append({
            "time": t,
            "finite_difference": float(np.max(np.abs(numeric - analytic))
                                       / max(1., np.max(np.abs(analytic)))),
            "transport": float(np.max(np.abs(other(z @ A.T, t) - expected))
                               / max(1., np.max(np.abs(expected)))),
            "density_jacobian": float(np.max(np.abs(
                independent_density(density_law, density_model, z @ density_A.T, t)
                - independent_density(law, model, z, t) + np.linalg.slogdet(density_A)[1]))),
        })
    return {"records": records, "score_transform": A.tolist(),
            "density_transform": density_A.tolist(), "passed": all(
        r["finite_difference"] < band and r["transport"] < PATH_BAND
        and r["density_jacobian"] < 1e-7 for r in records)}


def geometry_checks(model, A):
    B, Q = model.B, model.Q
    H = (B @ Q + Q @ B.T) / 2
    kappa, _, residual = threshold(B, Q)
    L = np.linalg.cholesky(Q)
    left = solve_triangular(L, H, lower=True)
    whitened = solve_triangular(L, left.T, lower=True).T
    whitening = float(abs(kappa - np.linalg.eigvalsh(whitened)[-1]))
    other = model.transported(A)
    transported = float(abs(kappa - threshold(other.B, other.Q)[0]))
    scaling = float(abs(kappa - threshold(B, 3 * Q)[0]))
    s, t = 0., min(model.horizon, 0.001)
    R, band = transition_covariance(B, Q, model.rate, model.sigma0, s, t)
    integral, error = integrated_covariance(B, Q, model.rate, model.sigma0, s, t)
    quadrature = float(np.linalg.norm(R - integral) / max(np.linalg.norm(R), 1e-15))
    G, D, _ = diffusion_shapes(model)
    values, vectors = eigh(G)
    v = vectors[:, 0]
    eigen_band = 256 * len(B) * np.finfo(float).eps * max(1., np.linalg.norm(G, 2))
    inertia = [int(np.sum(values < -eigen_band)),
               int(np.sum(np.abs(values) <= eigen_band)),
               int(np.sum(values > eigen_band))]
    other_G, other_D, _ = diffusion_shapes(other)
    other_v = np.linalg.solve(A.T, v)
    eps = 1e-6
    C = lambda u: model.sigma0**2 * np.exp(2 * model.rate * u) * Q
    derivative = (C(0.1 + eps) - C(0.1 - eps)) / (2 * eps)
    balance = float(np.linalg.norm(
        derivative - B @ C(0.1) - C(0.1) @ B.T
        - model.sigma0**2 * np.exp(0.2 * model.rate) * G)
        / max(np.linalg.norm(derivative), 1e-15))
    return {
        "threshold": kappa, "eigenpair_residual": residual, "whitening_error": whitening,
        "transport_error": transported, "scaling_error": scaling, "inertia": inertia,
        "diffusion_eigenvalues": values.tolist(), "diffusion_roundoff_band": float(eigen_band),
        "feasibility": "below" if inertia[0] else "boundary_band" if inertia[1] else "above",
        "minimum_diffusion_direction": v.tolist(), "minimum_quadratic": float(v @ G @ v),
        "covariance_balance_error": balance,
        "transported_minimum_direction": other_v.tolist(),
        "transported_minimum_quadratic": float(other_v @ other_G @ other_v),
        "transition_min_eigenvalue": float(eigh(R, eigvals_only=True)[0]),
        "transition_roundoff_band": band, "quadrature_error_estimate": float(error),
        "transition_quadrature_residual": quadrature,
        "local_clipping_congruence_residual": float(np.linalg.norm(other_D - A @ D @ A.T)),
        "diffusion_congruence_residual": float(np.linalg.norm(other_G - A @ G @ A.T)),
        "passed": bool(max(residual, whitening, transported, scaling) < 1e-9
                       and quadrature < 1e-6 and balance < 1e-7
                       and (model.rate < kappa - 1e-10 or eigh(R, eigvals_only=True)[0] >= -band)),
    }


def terminal_moments(law, model, policy, estimated_points):
    mean, covariance = law.moments()
    if policy == "estimated":
        clean, _ = law.sample(estimated_points, np.random.default_rng(1102))
        mean, covariance = clean.mean(0), np.cov(clean, rowvar=False)
    M = expm(model.B * model.horizon)
    noise = (model.sigma0 * np.exp(model.rate * model.horizon))**2 * model.Q
    mean, covariance = M @ mean, M @ covariance @ M.T + noise
    if policy == "noise_only":
        mean, covariance = np.zeros(4), noise
    elif policy == "shift":
        mean = mean + np.array([0.5, 0, 0, 0])
    elif policy.startswith("scale_"):
        covariance = float(policy.split("_")[1]) * covariance
    elif policy not in ("exact", "moment", "estimated"):
        raise ValueError(f"Unknown terminal prior {policy}")
    return mean, covariance


def terminal_prior(law, model, policy, n, seed, estimated_points):
    rng = np.random.default_rng(np.random.SeedSequence([1101, seed]))
    if policy == "exact":
        return draw(law, model, model.horizon, n, rng)
    mean, covariance = terminal_moments(law, model, policy, estimated_points)
    return mean + rng.standard_normal((n, 4)) @ np.linalg.cholesky(covariance).T


def score_error(oracle, model, kind, amplitude):
    rng = np.random.default_rng(1201)
    pilot = draw(oracle.law, model, model.horizon / 2, 256, rng)
    Q = model.Q
    values = oracle(pilot, model.horizon / 2)
    rms = float(np.sqrt(np.mean(np.einsum("ni,ij,nj->n", values, Q, values))))
    direction = np.ones(4)
    direction /= np.sqrt(direction @ Q @ direction)

    def score(z, t):
        exact = oracle(z, t)
        if kind in ("exact", "zero"):
            return exact
        if kind == "scale":
            return (1 + amplitude) * exact
        factor = np.sin(np.pi * t / model.horizon) if kind == "smooth" else 1.
        return exact + amplitude * rms * factor * direction

    return score, {"kind": kind, "amplitude": amplitude, "reference_score_Q_rms": rms,
                   "reference_time": model.horizon / 2, "direction": direction.tolist(),
                   "normalization": "e^T Q e", "calibration_seed": 1201}


class NetworkScore:
    def __init__(self, model, config, seed):
        self.model, self.config = model, config
        self.L = np.linalg.cholesky(model.Q)
        self.inverse = np.linalg.inv(self.L)
        self.net = MLP([9] + [config["width"]] * config["depth"] + [4],
                       np.random.default_rng(seed))

    def features(self, z, t):
        t = np.broadcast_to(np.asarray(t, dtype=float), (len(z),))
        sigma = self.model.sigma0 * np.exp(self.model.rate * t)
        u = t / self.model.horizon
        theta = 2 * np.pi * u
        return np.column_stack([
            (z @ self.inverse.T) / np.sqrt(sigma**2 + 4)[:, None],
            u, np.sin(theta), np.cos(theta), np.sin(2 * theta), np.cos(2 * theta),
        ])

    def __call__(self, z, t):
        t = np.broadcast_to(np.asarray(t, dtype=float), (len(z),))
        sigma = self.model.sigma0 * np.exp(self.model.rate * t)
        return -(self.net.forward(self.features(z, t)) @ self.inverse) / sigma[:, None]

    def save(self, path, metadata):
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"p{i}": p for i, p in enumerate(self.net.params)}
        arrays.update({f"m{i}": m for i, m in enumerate(self.net.m)})
        arrays.update({f"v{i}": v for i, v in enumerate(self.net.v)})
        temporary = path.with_suffix(".tmp.npz")
        np.savez(temporary, **arrays, metadata=json.dumps(metadata), step=self.net.k)
        temporary.replace(path)

    def load(self, path, expected):
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata["identity"] != expected:
                raise ValueError(f"Checkpoint fingerprint mismatch: {path}")
            for prefix, params in (("p", self.net.params), ("m", self.net.m), ("v", self.net.v)):
                for i, p in enumerate(params):
                    value = archive[f"{prefix}{i}"]
                    if value.shape != p.shape or not np.isfinite(value).all():
                        raise ValueError(f"Invalid checkpoint parameter: {path}/{prefix}{i}")
                    p[:] = value
            self.net.k = int(archive["step"])
        return metadata


def training_batch(law, model, size, seed):
    rng = np.random.default_rng(seed)
    clean, _ = law.sample(size, rng)
    times = rng.uniform(0, model.horizon, size)
    M = expm(times[:, None, None] * model.B)
    eps = rng.standard_normal((size, 4))
    z = np.einsum("nij,nj->ni", M, clean) + (
        model.sigma0 * np.exp(model.rate * times))[:, None] * (
            eps @ np.linalg.cholesky(model.Q).T)
    return z, times, eps


def train_network(law, model, spec, seed, directory, code_identity, resume):
    config = spec["training"]
    specification_hash = digest(spec)
    data_hash = digest(law.record())
    identity = digest({"law": law.record(), "B": model.B.tolist(), "Q": model.Q.tolist(),
                       "rate": model.rate, "sigma0": model.sigma0, "horizon": model.horizon,
                       "training": config, "seed": seed, "code": code_identity,
                       "specification": specification_hash})
    network = NetworkScore(model, config, seed)
    best_path = directory / f"{identity}-best.npz"
    last_path = directory / f"{identity}-last.npz"
    validation = training_batch(law, model, config["validation_size"], [1301, seed])
    metadata = {"identity": identity, "best_validation_loss": None, "validation": [],
                "training_seed": seed, "config": config, "normalization": "Cholesky-whitened",
                "normalization_matrix": network.inverse.tolist(), "input_scale_squared_offset": 4.,
                "data_hash": data_hash, "specification_hash": specification_hash,
                "model": {"B": model.B.tolist(), "Q": model.Q.tolist(), "rate": model.rate,
                          "sigma0": model.sigma0, "horizon": model.horizon},
                "loss": "epsilon MSE", "completed": False, "elapsed_seconds": 0.}
    if last_path.exists():
        metadata = network.load(last_path, identity)
        if not best_path.exists():
            raise ValueError(f"Missing best checkpoint for resumed training: {identity}")
        if not resume and not metadata["completed"]:
            raise ValueError(f"Incomplete training exists; use --resume: {identity}")
    started = time.monotonic()
    for step in range(network.net.k, config["updates"]):
        z, t, eps = training_batch(law, model, config["batch"], [1302, seed, step])
        diff = network.net.forward(network.features(z, t)) - eps
        loss = float(np.mean(np.sum(diff**2, axis=1)))
        grads = network.net.backward(2 * diff / len(z))
        if not np.isfinite(loss) or any(not np.isfinite(g).all() for g in grads):
            raise ValueError(f"Nonfinite training loss/gradient at step {step}")
        lr = config["learning_rate"] * 0.5 * (1 + np.cos(np.pi * step / config["updates"]))
        network.net.adam(grads, lr)
        if (step + 1) % config["validate_every"] == 0 or step + 1 == config["updates"]:
            vz, vt, veps = validation
            validation_loss = float(np.mean(np.sum(
                (network.net.forward(network.features(vz, vt)) - veps)**2, axis=1)))
            if not np.isfinite(validation_loss):
                raise ValueError("Nonfinite validation loss")
            metadata["validation"].append({"step": step + 1, "loss": validation_loss})
            metadata["completed"] = step + 1 == config["updates"]
            if (metadata["best_validation_loss"] is None
                    or validation_loss < metadata["best_validation_loss"]):
                metadata["best_validation_loss"] = validation_loss
                network.save(best_path, metadata)
            network.save(last_path, metadata)
            LOG.info("Training %s seed=%s step=%s/%s validation=%.6g",
                     identity[:10], seed, step + 1, config["updates"], validation_loss)
    metadata["elapsed_seconds"] += time.monotonic() - started
    metadata["completed"] = True
    network.save(last_path, metadata)
    best = network.load(best_path, identity)
    metadata["selected_step"] = network.net.k
    metadata["selected_validation_loss"] = best["best_validation_loss"]
    metadata["checkpoint"] = str(best_path)
    metadata["checkpoint_sha256"] = hashlib.sha256(best_path.read_bytes()).hexdigest()
    return network, metadata


def score_evaluation(score, oracle, law, model):
    records = []
    for i, fraction in enumerate((0., 0.1, 0.5, 0.9, 1.)):
        t = fraction * model.horizon
        z = draw(law, model, t, 128, np.random.default_rng(1401 + i))
        error = score(z, t) - oracle(z, t)
        if not np.isfinite(error).all():
            raise ValueError(f"Nonfinite held-out score error at time {t}")
        G, _, _ = diffusion_shapes(model)
        quadratic = np.einsum("ni,ij,nj->n", error, model.Q, error)
        records.append({"time_fraction": fraction, "score_Q_mse": float(np.mean(quadratic)),
                        "signed_score_G_error": float(np.mean(
                            np.einsum("ni,ij,nj->n", error, G, error)
                            * (model.sigma0 * np.exp(model.rate * t))**2))})
    z, t, epsilon = training_batch(law, model, 128, 1409)
    # The conditional score target gives an independent denoising diagnostic.
    sigma = model.sigma0 * np.exp(model.rate * t)
    predicted = np.empty_like(epsilon)
    for i in range(len(z)):
        predicted[i] = -sigma[i] * score(z[i:i + 1], float(t[i]))[0] @ np.linalg.cholesky(model.Q)
    records.append({"heldout_epsilon_mse": float(np.mean(np.sum(
        (predicted - epsilon)**2, axis=1))), "stream_seed": 1409})
    return records


def score_quadratic_checks(score, oracle, law, model, A, band):
    inverse = np.linalg.inv(A)
    G, _, _ = diffusion_shapes(model)
    records = []
    for index, fraction in enumerate((0., 0.5, 1.)):
        t = fraction * model.horizon
        z = draw(law, model, t, 32, np.random.default_rng(1410 + index))
        error = score(z, t) - oracle(z, t)
        transported = error @ inverse
        row = {"time_fraction": fraction}
        for name, matrix in (("Q", model.Q), ("G", G)):
            original = np.einsum("ni,ij,nj->n", error, matrix, error)
            other = np.einsum("ni,ij,nj->n", transported, A @ matrix @ A.T, transported)
            row[name + "_residual"] = float(np.max(np.abs(original - other))
                                           / max(1., np.max(np.abs(original))))
        records.append(row)
    return {"records": records, "signed_G_not_a_loss": True,
            "scope": "Transport of this score's complete error, not independent retraining",
            "passed": all(np.isfinite(row[key]) and row[key] <= band
                          for row in records for key in ("Q_residual", "G_residual"))}


def integrate(score, model, terminal, steps, seed, A, diffusion=None):
    times = np.linspace(model.horizon, 0., steps[-1] + 1)
    sigmas = model.sigma0 * np.exp(model.rate * times)
    G, D, L = diffusion_shapes(model) if diffusion is None else diffusion
    expected_G = 2 * model.rate * model.Q - model.B @ model.Q - model.Q @ model.B.T
    scale = max(1., np.linalg.norm(G), np.linalg.norm(D))
    if (any(np.shape(value) != model.Q.shape or not np.isfinite(value).all()
            for value in (G, D, L))
            or np.linalg.norm(G - expected_G) > PATH_BAND * scale
            or np.linalg.norm(D - D.T) > PATH_BAND * scale
            or np.linalg.norm(L @ L.T - D) > PATH_BAND * scale
            or np.linalg.eigvalsh(D - G)[0] < -PATH_BAND * scale):
        raise ValueError("Inconsistent diffusion, clipping covariance, or Brownian factor")
    other = model.transported(A)
    inverse = np.linalg.inv(A)
    indexed = lambda z, i: score(z, times[i])
    transformed = lambda y, i: score(y @ inverse.T, times[i]) @ inverse
    subset = min(8, len(terminal))
    levels = {
        k: {"states": {mode: terminal.copy() for mode in MODES},
            "transported": {mode: terminal[:subset] @ A.T for mode in MODES},
            "increment": np.zeros_like(terminal), "path_errors": dict.fromkeys(MODES, 0.)}
        for k in steps
    }
    rng = np.random.default_rng(np.random.SeedSequence([1501, seed]))
    for i in range(steps[-1]):
        increment = np.sqrt(model.horizon / steps[-1]) * rng.standard_normal(terminal.shape)
        for k, level in levels.items():
            level["increment"] += increment
            stride = steps[-1] // k
            if (i + 1) % stride:
                continue
            index, h = i + 1 - stride, model.horizon / k
            for mode in MODES:
                z = advance(level["states"][mode], indexed, model, G, D, L, sigmas,
                            index, stride, h, mode, level["increment"])
                y = advance(level["transported"][mode], transformed, other,
                            A @ G @ A.T, A @ D @ A.T, A @ L, sigmas, index, stride,
                            h, mode, level["increment"][:subset])
                if not np.isfinite(z).all() or not np.isfinite(y).all():
                    raise ValueError(f"Nonfinite trajectory at grid={k}, step={index}, mode={mode}")
                residual = float(np.max(np.abs(y @ inverse.T - z[:subset]))
                                 / max(1., np.max(np.abs(z[:subset]))))
                level["path_errors"][mode] = max(level["path_errors"][mode], residual)
                level["states"][mode], level["transported"][mode] = z, y
            level["increment"].fill(0)
        if (i + 1) % max(1, steps[-1] // 10) == 0:
            LOG.info("Sampler seed=%s step=%s/%s", seed, i + 1, steps[-1])
    return levels


def observed_metrics(z, reference, target, projections, bandwidth, empirical):
    if not empirical:
        return metrics(z, reference, target, projections, bandwidth)
    # Component responsibilities for thousands of point kernels are neither a
    # meaningful mode diagnostic nor needed for empirical-law quality.
    mean, covariance = target.moments()
    errors = normalized_error(z.mean(0), np.cov(z, rowvar=False), mean, covariance)
    n = len(z)

    def kernel_sum(x, y, diagonal=False):
        total = 0.
        for first in range(0, len(x), 128):
            block = x[first:first + 128]
            distance = np.maximum(np.sum(block**2, axis=1)[:, None]
                                  + np.sum(y**2, axis=1) - 2 * block @ y.T, 0)
            kernel = np.exp(-distance / (2 * bandwidth**2))
            if diagonal:
                kernel[np.arange(len(block)), first + np.arange(len(block))] = 0
            total += float(kernel.sum())
        return total

    m = len(reference)
    return {
        "mean": errors["mean"], "covariance": errors["covariance"],
        "sliced_w2": float(np.sqrt(np.mean((
            np.sort(z @ projections, axis=0) - np.sort(reference @ projections, axis=0)
        )**2)) / np.sqrt(np.trace(covariance) / 4)),
        "mmd2": (kernel_sum(z, z, True) / (n * (n - 1))
                 + kernel_sum(reference, reference, True) / (m * (m - 1))
                 - 2 * kernel_sum(z, reference) / (n * m)),
        "gaussian_moment_kl_proxy": gaussian_kl(z.mean(0), np.cov(z, rowvar=False),
                                               mean, covariance),
        "variance_ratios": (np.var(z, axis=0, ddof=1) / np.diag(covariance)).tolist(),
    }


def calibrate(target, spec, empirical, stream):
    rng = np.random.default_rng([1601, stream])
    pilot, _ = target.sample(512, rng)
    bandwidth = float(np.sqrt(np.median(np.sum((pilot[:256] - pilot[256:])**2, axis=1))))
    if bandwidth <= 0:
        raise ValueError("Degenerate calibration bandwidth")
    projections = rng.standard_normal((4, 32))
    projections /= np.linalg.norm(projections, axis=0)
    reference, _ = target.sample(spec["particles"][-1], np.random.default_rng([1602, stream]))
    controls = {}
    for n in spec["particles"]:
        raw = [observed_metrics(target.sample(n, np.random.default_rng([1603, stream, j]))[0],
                                reference[:n], target, projections, bandwidth, empirical)
               for j in range(spec["calibration_repeats"])]
        margins = spec["acceptance"]["metric_margins"]
        bands = {key: max(row[key] for row in raw) + margin for key, margin in margins.items()
                 if key in raw[0]}
        controls[str(n)] = {"raw": raw, "bands": bands}
    return reference, projections, bandwidth, controls


def gaussian_reverse_reference(law, model, prior, spec, mode, diffusion=None):
    if len(law.weights) != 1 or law.empirical:
        return None
    mean, C = terminal_moments(law, model, prior, spec["estimated_moment_points"])

    def rhs(tau, state):
        t = model.horizon - tau
        mt, Ct = model.marginal(t)
        cov_noise = (model.sigma0 * np.exp(model.rate * t))**2 * model.Q
        G = 2 * model.rate * cov_noise - model.B @ cov_noise - cov_noise @ model.B.T
        if diffusion is None:
            values, vectors = eigh(G)
            D = (vectors * np.maximum(values, 0)) @ vectors.T
        else:
            G, D = (model.sigma0 * np.exp(model.rate * t))**2 * np.array(diffusion[:2])
        W = G / 2 if mode == "ode" else D if mode == "naive" else (G + D) / 2
        F = -model.B - np.linalg.solve(Ct, W.T).T
        offset = W @ np.linalg.solve(Ct, mt)
        observed_C = state[4:].reshape(4, 4)
        return np.concatenate([F @ state[:4] + offset,
                               (F @ observed_C + observed_C @ F.T
                                + (np.zeros((4, 4)) if mode == "ode" else D)).ravel()])

    solution = solve_ivp(rhs, (0, model.horizon), np.r_[mean, C.ravel()],
                         method="DOP853", rtol=2e-10, atol=2e-11)
    if not solution.success:
        raise ValueError(f"Gaussian reverse reference failed: {solution.message}")
    return {"mean": solution.y[:4, -1].tolist(),
            "covariance": solution.y[4:, -1].reshape(4, 4).tolist()}


def summary(values):
    mean = float(np.mean(values))
    if len(values) < 2:
        return {"values": values, "mean": mean, "standard_error": None,
                "interval_95_normal_descriptive": None}
    se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    return {"values": values, "mean": mean, "standard_error": se,
            "interval_95_normal_descriptive": [mean - 1.96 * se, mean + 1.96 * se]}


def run_case(spec, case, output, code_identity, resume=False):
    started = time.monotonic()
    original, heldout = make_data(case["data"], spec)
    base_model = model_for(original, spec, case)
    A = np.array(spec["transforms"][case["transform"]], dtype=float)
    inverse = np.linalg.inv(A)
    model, law, pullback = base_model, original, np.eye(4)
    if case["arm"] != "baseline":
        model, law, pullback = base_model.transported(A), original.transported(A), inverse
        if case["arm"] == "reisotropized":
            model.Q = np.eye(4)
    oracle = Oracle(law, model)
    training = None
    # Wrapped Arm A shares the exact same fitted network and prior draws as baseline.
    source_law, source_model = (original, base_model) if case["arm"] == "transported" else (law, model)
    source_oracle = Oracle(source_law, source_model)
    if case["score"] == "learned":
        score, training = train_network(source_law, source_model, spec, case["training_seed"],
                                        output / "checkpoints", code_identity, resume)
        error_record = {"kind": "learned"}
    else:
        score, error_record = score_error(source_oracle, source_model,
                                          case["score"], case["amplitude"])
    if case["arm"] == "transported":
        baseline_score = score
        if case["score"] == "independent_bias":
            score, error_record = score_error(oracle, model, "bias", case["amplitude"])
            error_record["intervention"] = "independently normalized transformed-coordinate bias"
        else:
            score = lambda y, t: baseline_score(y @ inverse.T, t) @ inverse
    target = marginal_law(law, model, 0.).transported(pullback)
    terminal_target = marginal_law(law, model, model.horizon).transported(pullback)
    reference, projections, bandwidth, controls = calibrate(target, spec, law.empirical, 0)
    terminal_reference, terminal_P, terminal_bw, terminal_controls = calibrate(
        terminal_target, spec, law.empirical, 1)
    checks = score_checks(law, model, A, spec["acceptance"]["score_residual"])
    quadratics = score_quadratic_checks(
        score, oracle, law, model, A, spec["acceptance"]["path_residual"])
    geometry = geometry_checks(model, A)
    diffusion = None
    if case["arm"] in ("transported", "retrained_A"):
        G, D, L = diffusion_shapes(base_model)
        diffusion = A @ G @ A.T, A @ D @ A.T, A @ L
    prior_law, prior_model = ((original, base_model)
                              if case["arm"] in ("transported", "retrained_A")
                              else (law, model))
    prior_mean, prior_covariance = terminal_moments(
        prior_law, prior_model, case["prior"], spec["estimated_moment_points"])
    if case["arm"] in ("transported", "retrained_A"):
        prior_mean, prior_covariance = A @ prior_mean, A @ prior_covariance @ A.T
    runs = []
    nmax = spec["particles"][-1]
    for seed in spec["seeds"]:
        seed_path = output / "seeds" / f"{case['id']}-{seed}.json"
        seed_fingerprint = digest({
            "spec": spec, "case": case, "seed": seed, "data": original.record(),
            "code": code_identity,
            "checkpoint": training["checkpoint_sha256"] if training else None,
        })
        if seed_path.exists():
            if not resume:
                raise FileExistsError(f"Sampler seed exists; use --resume: {seed_path}")
            stored = json.loads(seed_path.read_text())
            if stored["fingerprint"] != seed_fingerprint:
                raise ValueError(f"Sampler seed fingerprint mismatch: {seed_path}")
            runs.append(stored["run"])
            LOG.info("Reusing sampler seed=%s for %s", seed, case["id"])
            continue
        terminal = terminal_prior(source_law, source_model, case["prior"], nmax, seed,
                                   spec["estimated_moment_points"])
        if case["arm"] == "transported":
            terminal = terminal @ A.T
        # For retrained A the prior intervention must still be a pushforward.
        if case["arm"] == "retrained_A":
            terminal = terminal_prior(original, base_model, case["prior"], nmax, seed,
                                       spec["estimated_moment_points"]) @ A.T
        levels = integrate(score, model, terminal, spec["steps"], seed, A, diffusion)
        record = {"seed": seed, "initial_metrics": observed_metrics(
            terminal @ pullback.T, terminal_reference, terminal_target, terminal_P,
            terminal_bw, law.empirical), "levels": [], "coupled_step_differences": {},
            "endpoint_moments": {}}
        for k, level in levels.items():
            record["endpoint_moments"][str(k)] = {}
            for mode in MODES:
                z = level["states"][mode] @ pullback.T
                record["endpoint_moments"][str(k)][mode] = {
                    "mean": z.mean(0).tolist(), "covariance": np.cov(z, rowvar=False).tolist()}
                values = []
                for n in spec["particles"]:
                    value = observed_metrics(z[:n], reference[:n], target, projections,
                                             bandwidth, law.empirical)
                    values.append({"particles": n, "metrics": value,
                                   "within_target_bands": quality_passed(
                                       value, controls[str(n)]["bands"])})
                record["levels"].append({"steps": k, "mode": mode, "particles": values,
                                         "path_residual": level["path_errors"][mode]})
                if k == spec["steps"][-1]:
                    record.setdefault("sample_preview", {})[mode] = z[:512].tolist()
        for mode in MODES:
            record["coupled_step_differences"][mode] = [
                float(np.sqrt(np.mean(np.sum(((levels[a]["states"][mode]
                                                - levels[b]["states"][mode]) @ pullback.T)**2,
                                              axis=1))) / np.sqrt(np.trace(target.moments()[1])))
                for a, b in zip(spec["steps"], spec["steps"][1:])]
        if heldout is not None:
            heldout_law = Law(np.full(len(heldout), 1 / len(heldout)), heldout,
                              np.zeros((len(heldout), 4, 4)), True)
            heldout_model = model.transported(pullback)
            heldout_target = marginal_law(heldout_law, heldout_model, 0.)
            heldout_reference, _ = heldout_target.sample(nmax, np.random.default_rng([1701, seed]))
            record["heldout_population_proxy"] = {
                mode: observed_metrics(levels[spec["steps"][-1]]["states"][mode] @ pullback.T,
                                       heldout_reference, heldout_target, projections,
                                       bandwidth, True) for mode in MODES}
        write_json(seed_path, {
            "case": case, "seed": seed, "fingerprint": seed_fingerprint,
            "manifest_hash": digest(spec), "data_hash": digest(original.record()),
            "code_identity": code_identity,
            "checkpoint_sha256": training["checkpoint_sha256"] if training else None,
            "run": record,
        })
        runs.append(record)
    refined, paired = {}, {}
    for mode in MODES:
        averages = {}
        for k in spec["steps"]:
            rows = [next(row for row in run["levels"] if row["steps"] == k
                         and row["mode"] == mode)["particles"][-1]["metrics"] for run in runs]
            averages[str(k)] = {key: float(np.mean([row[key] for row in rows]))
                                 for key in controls[str(nmax)]["bands"]}
        bands = controls[str(nmax)]["bands"]
        finest = averages[str(spec["steps"][-1])]
        exact_target = (case["score"] in ("exact", "zero") and case["prior"] == "exact"
                        and (mode != "naive" or model.rate >= geometry["threshold"] - 1e-10))
        within = quality_passed(finest, bands)
        no_degradation = all(finest[key] <= averages[str(spec["steps"][0])][key] + bands[key]
                             for key in bands)
        refined[mode] = {"expected_exact_target": exact_target, "seed_mean_metrics": averages,
                         "within_target_bands": within, "no_degradation": no_degradation,
                         "outcome": "supported" if within and no_degradation else "violated"}
        if case["score"] in ("exact", "zero"):
            # Reference evaluated in the initializer's own coordinates.
            reference_diffusion = diffusion if case["arm"] == "retrained_A" else None
            ref = gaussian_reverse_reference(source_law, source_model, case["prior"], spec,
                                             mode, reference_diffusion)
            if ref is not None:
                reference_pullback = np.eye(4) if case["arm"] == "transported" else pullback
                ref = {"mean": (reference_pullback @ np.array(ref["mean"])).tolist(),
                       "covariance": (reference_pullback @ np.array(ref["covariance"])
                                      @ reference_pullback.T).tolist()}
                refined[mode]["gaussian_reverse_reference"] = ref
                moment_errors = []
                for k in spec["steps"]:
                    # The independently initialized ODE is also the reference
                    # for mismatched Gaussian priors, not the target moments.
                    estimates = [run["endpoint_moments"][str(k)][mode] for run in runs]
                    m = np.mean([r["mean"] for r in estimates], axis=0)
                    C = np.mean([r["covariance"] for r in estimates], axis=0)
                    moment_errors.append({"steps": k, **normalized_error(
                        m, C, np.array(ref["mean"]), np.array(ref["covariance"]))})
                refined[mode]["gaussian_reference_errors"] = moment_errors
                margin = spec["acceptance"]["gaussian_moment_margin"]
                samples = nmax * len(runs)
                limits = {"mean": margin + 8 / np.sqrt(samples),
                          "covariance": margin + 16 / np.sqrt(samples)}
                refined[mode]["gaussian_reference_limits"] = limits
                refined[mode]["gaussian_reference_passed"] = all(
                    moment_errors[-1][key] <= limit
                    and moment_errors[-1][key] <= moment_errors[0][key] + limit
                    for key, limit in limits.items())
    for key in controls[str(nmax)]["bands"]:
        differences = []
        for run in runs:
            fine = {row["mode"]: row["particles"][-1]["metrics"]
                    for row in run["levels"] if row["steps"] == spec["steps"][-1]}
            differences.append(float(fine["naive"][key] - fine["fixed"][key]))
        paired[key] = summary(differences)
    paths_pass = all(row["path_residual"] <= spec["acceptance"]["path_residual"]
                     for run in runs for row in run["levels"])
    required_quality = all(not row["expected_exact_target"] or (
        row["within_target_bands"] and row["no_degradation"]) for row in refined.values())
    required_quality = required_quality and all(
        row.get("gaussian_reference_passed", True) for row in refined.values())
    harness_pass = checks["passed"] and geometry["passed"] and paths_pass and quadratics["passed"]
    return {
        "case": case, "scope": "Optional sensitivity; not core-gate certification",
        "law": original.record(), "data_hash": digest(original.record()),
        "model": {"B": model.B.tolist(), "Q": model.Q.tolist(), "A": A.tolist(),
                  "pullback": pullback.tolist(), "rate": model.rate,
                  "sigma0": model.sigma0, "horizon": model.horizon,
                  "noise_shape": spec["shapes"][case["shape"]]},
        "shape_diagnostics": {"trace": float(np.trace(model.Q)),
                              "determinant": float(np.linalg.det(model.Q)),
                              "logdet": float(np.linalg.slogdet(model.Q)[1]),
                              "eigenvalues": eigh(model.Q, eigvals_only=True).tolist(),
                              "condition_number": float(np.linalg.cond(model.Q))},
        "streams": {
            "generator": "numpy default_rng with SeedSequence integer lists",
            "raw_training_data": 401, "raw_heldout_data": 402,
            "initialization": "[1101, sampler_seed]", "estimated_clean_moments": 1102,
            "score_error_calibration": 1201, "training_validation": "[1301, training_seed]",
            "training_batches": "[1302, training_seed, update]",
            "score_checks": "901 + time_index", "score_evaluation": "1401 + time_index",
            "score_quadratic_checks": "1410 + time_index",
            "heldout_denoising": 1409, "brownian": "[1501, sampler_seed]",
            "metric_pilot_and_projections": "[1601, target_or_terminal_index]",
            "metric_reference": "[1602, target_or_terminal_index]",
            "metric_controls": "[1603, target_or_terminal_index, repeat]",
            "heldout_data_reference": "[1701, sampler_seed]",
        },
        "geometry": geometry, "score_checks": checks, "score_error": error_record,
        "score_quadratic_checks": quadratics,
        "prior_definition": {
            "policy": case["prior"], "mean": prior_mean.tolist(),
            "covariance": prior_covariance.tolist(), "coordinates": "model coordinates",
            "distribution": "exact marginal" if case["prior"] == "exact" else "Gaussian",
            "gaussian_exact_moment_null": len(original.weights) == 1 and not original.empirical
            and case["prior"] in ("exact", "moment"),
            "estimated_moment_points": spec["estimated_moment_points"]
            if case["prior"] == "estimated" else None,
        },
        "sampler_policy": ("fully transported baseline clipping and noise factor"
                           if case["arm"] in ("transported", "retrained_A")
                           else "local spectral clipping"),
        "heldout_score_evaluation": score_evaluation(score, oracle, law, model),
        "outcome_scope": "Expected harness checks and declared exact-target controls; "
                         "approximate-prior and learned-score quality is measured separately",
        "training": training, "calibration": controls, "terminal_calibration": terminal_controls,
        "metric_definition": {"bandwidth": bandwidth, "projections": projections.tolist(),
                              "terminal_bandwidth": terminal_bw,
                              "terminal_projections": terminal_P.tolist(),
                              "coordinates": "pulled back to baseline",
                              "kl": "Gaussian-moment proxy, not distributional KL"},
        "runs": runs, "refinement": refined, "paired_naive_minus_fixed": paired,
        "target_preview": reference[:512].tolist(),
        "harness_passed": bool(harness_pass), "required_quality_passed": bool(required_quality),
        "outcome": "supported" if harness_pass and required_quality else "violated",
        "elapsed_seconds": time.monotonic() - started,
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


def environment():
    revision = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=normal"],
                            capture_output=True, text=True, check=True).stdout.strip()
    sources = [ROOT / "psdt.py", DEFAULT_SPEC, Path(__file__),
               ROOT / "verification/gaussian_sampling.py",
               ROOT / "verification/mixture_sampling.py",
               ROOT / "verification/forward_feasibility.py",
               ROOT / "verification/model_specification.py"]
    return {"revision": revision, "dirty": bool(status), "git_status": status,
            "source_hashes": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in sources},
            "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform()}


def run_manifest(spec, output, axes=None, case_ids=None, resume=False, max_cases=None):
    validate_spec(spec)
    env = environment()
    code_identity = digest({key: env[key] for key in (
        "revision", "source_hashes", "python", "numpy", "scipy", "platform")})
    mixture = Path(spec["mixture_path"])
    input_hash = (hashlib.sha256(mixture.read_bytes()).hexdigest()
                  if mixture.exists() and any(c["data"] == "fitted_moons" for c in spec["cases"])
                  else None)
    run_identity = digest({"spec": spec, "code": code_identity, "mixture": input_hash})
    selected = [case for case in spec["cases"] if (not axes or case["axis"] in axes)
                and (not case_ids or case["id"] in case_ids)]
    if case_ids and set(case_ids) - {case["id"] for case in spec["cases"]}:
        raise ValueError("Requested case ID is not in the manifest")
    if max_cases is not None:
        selected = selected[:max_cases]
    if not selected:
        raise ValueError("No cases selected")
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and digest(json.loads(manifest_path.read_text())) != digest(spec):
        raise ValueError("Output directory belongs to a different manifest")
    write_json(manifest_path, spec)
    records = []
    known = {}
    for case in spec["cases"]:
        path = output / "cases" / f"{case['id']}.json"
        if path.exists():
            previous = json.loads(path.read_text())
            if previous.get("run_identity") == run_identity:
                known[case["id"]] = previous
    failed = False
    for index, case in enumerate(selected):
        LOG.info("Case %s/%s: %s", index + 1, len(selected), case["id"])
        path = output / "cases" / f"{case['id']}.json"
        # Input hashes are verified even when resuming an existing result.
        try:
            law, _ = make_data(case["data"], spec)
            fingerprint = digest({"spec": spec, "case": case, "data": law.record(),
                                  "code": code_identity, "run_identity": run_identity})
            reused = False
            if resume and path.exists():
                result = json.loads(path.read_text())
                if result["fingerprint"] != fingerprint:
                    raise ValueError(f"Result fingerprint mismatch: {case['id']}")
                if result.get("status") == "completed":
                    if result.get("training"):
                        checkpoint = output / "checkpoints" / Path(
                            result["training"]["checkpoint"]).name
                        if (hashlib.sha256(checkpoint.read_bytes()).hexdigest()
                                != result["training"]["checkpoint_sha256"]):
                            raise ValueError(f"Checkpoint changed: {checkpoint}")
                    reused = True
            elif path.exists():
                raise FileExistsError(f"Result exists; use --resume or a new output directory: {path}")
            if not reused:
                result = run_case(spec, case, output, code_identity, resume)
                result.update(status="completed", fingerprint=fingerprint, environment=env,
                              run_identity=run_identity)
                write_json(path, result)
        except (OSError, ValueError, KeyError, FloatingPointError, np.linalg.LinAlgError) as error:
            LOG.error("Case %s failed: %s", case["id"], error)
            result = {"case": case, "status": "failed", "outcome": "not assessed",
                      "error": f"{type(error).__name__}: {error}", "environment": env}
            # Preserve incompatible or immutable prior evidence.
            failure_path = output / "failures" / f"{case['id']}.json"
            write_json(failure_path, result)
            failed = True
        records.append(result)
        known[case["id"]] = result
        if result.get("status") == "completed" and (
                not result["harness_passed"] or not result["required_quality_passed"]):
            failed = True
        coverage = {case["id"]: "not assessed" for case in spec["cases"]}
        for item in known.values():
            coverage[item["case"]["id"]] = item["outcome"]
        write_json(output / "summary.json", {
            "scope": "Selected optional generalization cases; not core-gate certification",
            "profile": spec["profile"], "manifest_hash": digest(spec), "run_identity": run_identity,
            "environment": env, "coverage": coverage,
            "selected": [case["id"] for case in selected],
            "finished": [item["case"]["id"] for item in records],
            "passed": not failed, "full_manifest_completed": len(known) == len(spec["cases"])
            and all(item.get("status") == "completed" for item in known.values()),
            "full_manifest_passed": len(known) == len(spec["cases"]) and all(
                item.get("status") == "completed" and item.get("harness_passed")
                and item.get("required_quality_passed") for item in known.values()),
            "limitations": ["Smoke settings are not main-study acceptance.",
                            "Core prerequisite gates are not certified by this runner.",
                            "Empirical target exactness is relative to frozen training data.",
                            "Calibration bands are not formal equivalence intervals.",
                            "Training seeds, not nested sampler seeds, are training replicates."]})
    return not failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--write-spec", type=Path, help="Write manifest and exit without running")
    parser.add_argument("--mixture", type=Path)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results/coordinate_invariance/generalization")
    parser.add_argument("--axis", choices=AXES, nargs="+")
    parser.add_argument("--case", nargs="+", dest="case_ids")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="List selected cases without execution")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        spec = json.loads(args.spec.read_text()) if args.spec else make_spec(args.profile, args.mixture)
        if args.mixture and args.spec:
            raise ValueError("Set mixture_path in the supplied manifest; --mixture is for generated specs")
        validate_spec(spec)
        if args.max_cases is not None and args.max_cases < 1:
            raise ValueError("--max-cases must be positive")
        if args.write_spec:
            write_json(args.write_spec, spec)
            print(f"Manifest: {args.write_spec} ({len(spec['cases'])} cases; no experiments run)")
            return 0
        if args.dry_run:
            selected = [c for c in spec["cases"] if (not args.axis or c["axis"] in args.axis)
                        and (not args.case_ids or c["id"] in args.case_ids)]
            if args.case_ids and set(args.case_ids) - {c["id"] for c in spec["cases"]}:
                raise ValueError("Requested case ID is not in the manifest")
            if args.max_cases:
                selected = selected[:args.max_cases]
            if not selected:
                raise ValueError("No cases selected")
            for case in selected:
                print(json.dumps(case, sort_keys=True))
            print(f"{len(selected)} selected of {len(spec['cases'])}; no experiments run")
            print(f"Finest grid={spec['steps'][-1]}, max particles={spec['particles'][-1]}, "
                  f"sampler seeds={len(spec['seeds'])}, training seeds={len(spec['training_seeds'])}")
            return 0
        args.output.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(args.output / "progress.log")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        LOG.addHandler(handler)
        passed = run_manifest(spec, args.output, args.axis, args.case_ids, args.resume, args.max_cases)
        LOG.info("Generalization selected run: %s", "PASS" if passed else "FAIL")
        return 0 if passed else 1
    except (OSError, ValueError, KeyError, np.linalg.LinAlgError) as error:
        LOG.error("Generalization: FAIL: %s", error)
        return 1
    except KeyboardInterrupt:
        LOG.warning("Interrupted; completed case evidence and training checkpoints retained")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
