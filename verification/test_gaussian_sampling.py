"""Gaussian pilot regression, coupling, and deliberate-failure checks."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from psdt import Process, sample
from verification.gaussian_sampling import (
    GaussianModel, MODES, affine_step, check_model, coefficients, coupled_paths,
    diffusion_shapes, gaussian_kl, moment_oracle, normalized_error,
    refinement_passed, run, sample_diagnostics,
)
from verification.model_specification import DEFAULT_SPEC


class GaussianSamplingTests(unittest.TestCase):
    def setUp(self):
        self.model = GaussianModel(
            np.array([[0., 1.], [-4., 0.]]), np.eye(2),
            np.array([0.3, -0.1]), np.array([[0.7, 0.12], [0.12, 0.4]]),
            0.9, 0.01, np.log(40 / 0.01) / 0.9,
        )

    def test_independent_oracle_and_expected_clipped_bias(self):
        target = self.model.marginal(0.)
        for mode in MODES:
            with self.subTest(mode=mode):
                reference = moment_oracle(self.model, mode)
                error = normalized_error(*reference, *target)["maximum"]
                if mode == "naive":
                    self.assertGreater(error, 0.01)
                else:
                    self.assertLess(error, 1e-6)
        with self.assertRaisesRegex(ValueError, "Unknown sampler"):
            moment_oracle(self.model, "invalid")

    def test_affine_integrator_matches_production_sampler(self):
        model = self.model
        steps = 128
        proc = Process(model.B, model.rate)
        times, sigmas, tables = coefficients(model, steps)
        meanT, covarianceT = model.marginal(model.horizon)
        terminal = np.random.default_rng(10).multivariate_normal(meanT, covarianceT, 12)

        def score(z, t):
            mean, covariance = model.marginal(t)
            return -np.linalg.solve(covariance, (z - mean).T).T

        for mode in MODES:
            with self.subTest(mode=mode):
                expected = sample(score, proc, terminal.copy(), mode,
                                  np.random.default_rng(11), K=steps)
                rng = np.random.default_rng(11)
                actual = terminal.copy()
                for index in range(steps):
                    h = times[index] - times[index + 1]
                    F, c, L = affine_step(tables[mode], sigmas, index, 1, h, mode)
                    actual = actual @ F.T + c
                    if mode != "ode":
                        actual += np.sqrt(h) * rng.standard_normal(actual.shape) @ L.T
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)

    def test_transport_uses_congruence_and_shared_factor(self):
        A = np.array([[2., 0.3], [0., 0.5]])
        other = self.model.transported(A)
        mean, covariance = self.model.marginal(0.7)
        other_mean, other_covariance = other.marginal(0.7)
        np.testing.assert_allclose(other_mean, A @ mean)
        np.testing.assert_allclose(other_covariance, A @ covariance @ A.T)
        _, D, L = diffusion_shapes(self.model)
        np.testing.assert_allclose((A @ L) @ (A @ L).T, A @ D @ A.T)
        report = coupled_paths(self.model, [A], [64, 128], [200, 400], 0)
        self.assertTrue(report["passed"])
        for level in report["levels"]:
            self.assertLess(max(level["path_max_scaled_residuals"]), 1e-10)

    def test_wrong_brownian_factor_fails_path_equivalence(self):
        original = coefficients

        def wrong(model, steps, diffusion=None):
            if diffusion is not None:
                D, L = diffusion
                # Right-orthogonal factors preserve diffusion covariance, not shared-noise paths.
                diffusion = D, -L
            return original(model, steps, diffusion)

        A = np.diag([2., 0.5])
        with patch("verification.gaussian_sampling.coefficients", side_effect=wrong):
            report = coupled_paths(self.model, [A], [32, 64], [200, 400], 0)
        self.assertFalse(report["passed"])
        failures = [level for level in report["levels"] if level["mode"] != "ode"]
        self.assertTrue(all(max(level["path_max_scaled_residuals"]) > 1e-3
                            for level in failures))

    def test_coarse_paths_use_sums_of_the_fine_brownian_increments(self):
        model = self.model
        report = coupled_paths(model, [np.eye(2)], [32, 64], [200, 400], 3)
        times, sigmas, tables = coefficients(model, 64)
        terminal_stream, brownian_stream = np.random.SeedSequence(3).spawn(2)
        meanT, covarianceT = model.marginal(model.horizon)
        initial = meanT + np.random.default_rng(terminal_stream).standard_normal((400, 2)) @ (
            np.linalg.cholesky(covarianceT).T
        )
        rng = np.random.default_rng(brownian_stream)
        increments = np.sqrt(model.horizon / 64) * rng.standard_normal((64, 400, 2))
        for mode in MODES:
            expected = initial.copy()
            for index in range(0, 64, 2):
                h = times[index] - times[index + 2]
                F, c, L = affine_step(tables[mode], sigmas, index, 2, h, mode)
                expected = expected @ F.T + c + increments[index:index + 2].sum(0) @ L.T
            level = next(item for item in report["levels"]
                         if item["steps"] == 32 and item["mode"] == mode)
            diagnostic = level["particle_refinement"][-1]
            np.testing.assert_allclose(diagnostic["mean"], expected.mean(0))
            np.testing.assert_allclose(diagnostic["covariance"], np.cov(expected, rowvar=False))

    def test_exact_terminal_and_smoothed_target(self):
        mean0, covariance0 = self.model.marginal(0.)
        np.testing.assert_array_equal(mean0, self.model.mean)
        np.testing.assert_allclose(
            covariance0, self.model.covariance + 0.01**2 * self.model.Q
        )
        report = coupled_paths(self.model, [np.eye(2)], [16, 32], [200, 400], 0)
        meanT, covarianceT = self.model.marginal(self.model.horizon)
        np.testing.assert_allclose(report["terminal_mean"], meanT)
        np.testing.assert_allclose(report["terminal_covariance"], covarianceT)
        self.assertTrue(report["terminal"]["passed"])

    def test_statistics_and_refinement_reject_bad_results(self):
        samples = np.random.default_rng(1).standard_normal((2000, 2))
        self.assertTrue(sample_diagnostics(samples, np.zeros(2), np.eye(2))["passed"])
        self.assertFalse(sample_diagnostics(samples + 1, np.zeros(2), np.eye(2))["passed"])
        self.assertTrue(refinement_passed([0.08, 0.04, 0.02]))
        self.assertFalse(refinement_passed([0.01, 0.02, 0.03]))
        self.assertFalse(refinement_passed([0.8, 0.4, 0.2]))
        self.assertAlmostEqual(gaussian_kl(np.zeros(2), np.eye(2), np.zeros(2), np.eye(2)), 0)

    def test_wrong_oracle_is_not_accepted_as_an_exact_target(self):
        mean, covariance = self.model.marginal(0.)
        with patch("verification.gaussian_sampling.moment_oracle",
                   return_value=(mean, 1.2 * covariance)):
            report = check_model(self.model, [np.eye(2)], [32, 64], [200, 400], [0])
        self.assertFalse(report["passed"])
        self.assertFalse(report["continuous_references"]["fixed"]["passed"])

    def test_invalid_configurations(self):
        spec = json.loads(DEFAULT_SPEC.read_text())
        for config in [
            {"steps": [100, 256]}, {"steps": [128]}, {"particle_counts": [200]},
            {"seeds": [0, 0]}, {"dimensions": [3]}, {"rates": [0]},
        ]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                run(spec, **config)
        with self.assertRaisesRegex(ValueError, "Invalid model specification"):
            run({})

    def test_cli_smoke_report_and_errors(self):
        script = Path(__file__).with_name("gaussian_sampling.py").resolve()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.json"
            command = [
                sys.executable, str(script), "--steps", "512", "1024", "2048",
                "--samples", "256", "512", "--seeds", "0", "--dimensions", "2",
                "--rates", "1.95", "--output", str(output),
            ]
            result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(output.read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(len(report["cases"]), 3)
            self.assertAlmostEqual(report["cases"][1]["threshold"], 0, places=12)
            failed = subprocess.run(
                [sys.executable, str(script), "--spec", "missing.json"],
                cwd=directory, capture_output=True, text=True,
            )
            self.assertEqual(failed.returncode, 1)
            self.assertIn("Gaussian sampling: FAIL", failed.stdout)


if __name__ == "__main__":
    unittest.main()
