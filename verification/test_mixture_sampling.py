"""Exact-mixture pilot regression and deliberate-failure controls."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gmm_exact import ExactScore
from psdt import Process, mmd2, sample
from verification.gaussian_sampling import GaussianModel, MODES, diffusion_shapes
from verification.mixture_sampling import (
    Mixture, Score, advance, calibration, check_case, coupled_samples, draw_marginal,
    load_mixture, marginal, metrics, quality_passed, run, score_checks,
)
from verification.model_specification import DEFAULT_SPEC


class MixtureSamplingTests(unittest.TestCase):
    def setUp(self):
        self.mixture = Mixture(
            np.array([0.3, 0.4, 0.3]),
            np.array([[-1., 0.2, -0.5, 0.], [0., -0.1, 1., 0.2], [1., 0., -0.5, -0.2]]),
            np.stack([np.diag([0.6, 0.4, 0.5, 0.3]) * a for a in [1., 1.2, 0.8]]),
        )
        mean, covariance = self.mixture.moments()
        self.model = GaussianModel(
            np.kron(np.eye(2), [[0., 1.], [-4., 0.]]), np.eye(4),
            mean, covariance, 1.95, 0.01, np.log(4000) / 1.95,
        )
        self.A = np.array([[2., 0.1, 0., 0.], [0., 0.5, 0., 0.],
                           [0., 0., 1., 0.3], [0., 0., 0., 1.]])

    def test_score_finite_differences_density_and_transport(self):
        self.assertTrue(score_checks(self.mixture, self.model, self.A)["passed"])
        original = Score.__call__
        with patch.object(Score, "__call__", lambda obj, z, i: -original(obj, z, i)):
            report = score_checks(self.mixture, self.model, self.A)
        self.assertFalse(report["passed"])
        self.assertGreater(report["checks"][0]["finite_difference_scaled_error"], 0.1)

    def test_score_matches_existing_exact_score(self):
        model = self.model
        proc = Process(model.B, model.rate)
        production = ExactScore(
            proc, self.mixture.weights, self.mixture.means, self.mixture.covariances
        )
        times = [0., 0.1, 1., model.horizon]
        score = Score(self.mixture, model, times)
        for index, t in enumerate(times):
            z, _ = draw_marginal(self.mixture, model, t, 32, np.random.default_rng(index))
            np.testing.assert_allclose(score(z, index), production(z, t), atol=1e-12, rtol=1e-11)

    def test_marginals_and_reisotropized_target(self):
        model, mixture = self.model, self.mixture
        target = marginal(mixture, model, 0.)
        np.testing.assert_allclose(target.means, mixture.means)
        np.testing.assert_allclose(
            target.covariances, mixture.covariances + model.sigma0**2 * model.Q
        )
        A = self.A
        armB = model.transported(A)
        armB.Q = np.eye(4)
        inverse = np.linalg.inv(A)
        pulled = marginal(mixture.transported(A), armB, 0.).transported(inverse)
        np.testing.assert_allclose(pulled.means, mixture.means, atol=1e-15)
        np.testing.assert_allclose(
            pulled.covariances,
            mixture.covariances + model.sigma0**2 * inverse @ inverse.T,
        )
        self.assertGreater(np.max(np.abs(pulled.covariances - target.covariances)), 1e-4)
        z, labels = draw_marginal(mixture, model, model.horizon, 50000, np.random.default_rng(5))
        terminal_mean, terminal_covariance = marginal(mixture, model, model.horizon).moments()
        np.testing.assert_allclose(z.mean(0), terminal_mean, atol=0.8)
        np.testing.assert_allclose(np.cov(z, rowvar=False), terminal_covariance, atol=35)
        self.assertEqual(labels.shape, (50000,))

    def test_integrator_matches_production_and_explicit_coarse_noise(self):
        model, mixture = self.model, self.mixture
        steps, n, seed = [32, 64], 100, 3
        terminal, _, _, levels = coupled_samples(mixture, model, self.A, steps, n, seed)
        times = np.linspace(model.horizon, 0., 65)
        sigmas = model.sigma0 * np.exp(model.rate * times)
        score = Score(mixture, model, times)
        G, D, L = diffusion_shapes(model)
        terminal_stream, noise_stream = np.random.SeedSequence(seed).spawn(2)
        expected_terminal, _ = draw_marginal(
            mixture, model, model.horizon, n, np.random.default_rng(terminal_stream)
        )
        np.testing.assert_array_equal(terminal, expected_terminal)
        rng = np.random.default_rng(noise_stream)
        increments = np.sqrt(model.horizon / 64) * rng.standard_normal((64, n, 4))
        proc = Process(model.B, model.rate)
        prod_score = ExactScore(proc, mixture.weights, mixture.means, mixture.covariances)
        for mode in MODES:
            expected = sample(prod_score, proc, terminal.copy(), mode,
                              np.random.default_rng(noise_stream), K=64)
            np.testing.assert_allclose(levels[64]["states"][mode], expected, atol=1e-10)
            coarse = terminal.copy()
            for index in range(0, 64, 2):
                coarse = advance(coarse, score, model, G, D, L, sigmas, index, 2,
                                 times[index] - times[index + 2], mode,
                                 increments[index:index + 2].sum(0))
            np.testing.assert_allclose(levels[32]["states"][mode], coarse)
            self.assertLess(levels[64]["path_errors"][mode], 1e-10)

    def test_wrong_shared_noise_factor_is_detected(self):
        original = advance

        def wrong(z, score, model, G, D, L, *args):
            if model.Q[0, 0] > 2:
                L = -L
            return original(z, score, model, G, D, L, *args)

        with patch("verification.mixture_sampling.advance", side_effect=wrong):
            _, _, _, levels = coupled_samples(
                self.mixture, self.model, self.A, [16, 32], 100, 1
            )
        self.assertLess(levels[32]["path_errors"]["ode"], 1e-10)
        self.assertGreater(levels[32]["path_errors"]["fixed"], 0.01)

    def test_calibration_and_metrics_reject_biased_samples(self):
        target = marginal(self.mixture, self.model, 0.)
        reference, projections, bandwidth, bands, records = calibration(
            target, [200, 400], 22, 4
        )
        z, _ = target.sample(400, np.random.default_rng(99))
        observed = metrics(z, reference, target, projections, bandwidth)
        self.assertTrue(quality_passed(observed, bands[400]))
        self.assertAlmostEqual(observed["mmd2"], mmd2(z, reference, (bandwidth,)), places=12)
        biased = metrics(z + 3, reference, target, projections, bandwidth)
        self.assertFalse(quality_passed(biased, bands[400]))
        collapsed, _ = Mixture(
            np.ones(1), target.means[:1], target.covariances[:1]
        ).sample(400, np.random.default_rng(99))
        self.assertFalse(quality_passed(
            metrics(collapsed, reference, target, projections, bandwidth), bands[400]
        ))
        self.assertEqual(len(records["400"]["target_vs_target"]), 4)
        self.assertIn("component_mass_max_error", bands[400])

    def test_bad_corrected_endpoints_fail_case(self):
        original = coupled_samples

        def biased(*args, **kwargs):
            terminal, counts, frequency, levels = original(*args, **kwargs)
            for level in levels.values():
                level["states"]["fixed"] += 5
            return terminal, counts, frequency, levels

        with patch("verification.mixture_sampling.coupled_samples", side_effect=biased):
            report = check_case(self.mixture, self.model, self.A, np.eye(4),
                                [32, 64], [100, 200], [0, 1], 4)
        self.assertFalse(report["passed"])
        self.assertFalse(report["step_refinement"]["fixed"]["passed"])

    def test_invalid_inputs_and_cache(self):
        with self.assertRaises(ValueError):
            Mixture(np.array([0.5]), np.zeros((1, 4)), np.eye(4)[None])
        with self.assertRaises(np.linalg.LinAlgError):
            Mixture(np.ones(1), np.zeros((1, 4)), -np.eye(4)[None])
        spec = json.loads(DEFAULT_SPEC.read_text())
        for config in [
            {"steps": [64]}, {"steps": [32, 65]}, {"particle_counts": [100]},
            {"seeds": [0]}, {"seeds": [0, 0]}, {"rates": [0]},
            {"calibration_repeats": 1},
        ]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                run(spec, self.mixture, **config)
        with self.assertRaisesRegex(ValueError, "Invalid model specification"):
            run({}, self.mixture)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fit.npz"
            np.savez(path, w=self.mixture.weights, mu=self.mixture.means, S=self.mixture.covariances)
            actual = load_mixture(path)
            np.testing.assert_array_equal(actual.covariances, self.mixture.covariances)
            with self.assertRaises(FileNotFoundError):
                load_mixture(Path(directory) / "missing.npz")

    def test_cli_report_and_missing_cache_error(self):
        script = Path(__file__).with_name("mixture_sampling.py").resolve()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fit.npz"
            output = Path(directory) / "report.json"
            log_file = Path(directory) / "logs" / "pilot.log"
            np.savez(path, w=self.mixture.weights, mu=self.mixture.means, S=self.mixture.covariances)
            command = [
                sys.executable, str(script), "--mixture", str(path),
                "--steps", "512", "1024", "--samples", "128", "256",
                "--seeds", "0", "1", "--rates", "1.95", "--calibration-repeats", "4",
                "--output", str(output), "--log-file", str(log_file),
            ]
            result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads(output.read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(len(report["cases"]), 3)
            self.assertEqual(len(report["mixture_sha256"]), 64)
            self.assertTrue(all(case["score_validation"]["passed"] for case in report["cases"]))
            logs = log_file.read_text()
            for message in ("Starting mixture sampling", "Case 1/3", "Calibrating target metrics",
                            "Independent score/density checks: PASS", "Seed 0: step",
                            "1024/1024 (100%)", "Evidence written", "Mixture sampling: PASS"):
                self.assertIn(message, logs)
                self.assertIn(message, result.stderr)
            failed = subprocess.run(
                [sys.executable, str(script), "--mixture", "missing.npz"],
                cwd=directory, capture_output=True, text=True,
            )
            self.assertEqual(failed.returncode, 1)
            self.assertIn("Mixture sampling: FAIL", failed.stderr)


if __name__ == "__main__":
    unittest.main()
