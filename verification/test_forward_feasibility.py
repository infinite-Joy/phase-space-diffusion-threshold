"""Regression and deliberate-failure checks for forward-feasibility evidence."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from verification.forward_feasibility import (
    check_case, integrated_covariance, psd_factor, run, threshold, transition_covariance,
)
from verification.model_specification import DEFAULT_SPEC


class ForwardFeasibilityTests(unittest.TestCase):
    def setUp(self):
        self.B = np.array([[0., 1.], [-4., 0.]])
        self.Q = np.eye(2)

    def test_threshold_and_correlated_shape(self):
        self.assertAlmostEqual(threshold(self.B, self.Q)[0], 1.5)
        A = np.array([[2., 0.3], [0., 0.5]])
        BY = A @ self.B @ np.linalg.inv(A)
        self.assertAlmostEqual(threshold(BY, A @ A.T)[0], 1.5)

    def test_independent_integral_and_short_time_limit(self):
        for rate in [1.2, 1.5, 1.8]:
            R, tolerance = transition_covariance(self.B, self.Q, rate, 0.01, 0.2, 0.7)
            integral, error = integrated_covariance(self.B, self.Q, rate, 0.01, 0.2, 0.7)
            self.assertLess(np.linalg.norm(R - integral), tolerance + 5 * error)
            h = 1e-6
            R, _ = transition_covariance(self.B, self.Q, rate, 0.01, 0., h)
            G = 2e-4 * (rate * self.Q - (self.B + self.B.T) / 2)
            np.testing.assert_allclose(R / h, G, atol=2e-9)

    def test_long_interval_does_not_replace_local_witness(self):
        R, _ = transition_covariance(self.B, self.Q, 1.2, 0.01, 0., np.pi)
        self.assertGreater(np.linalg.eigvalsh(R)[0], 0)
        case = check_case(self.B, self.Q, 1.2, "below", 0.01, 1., 1000,
                          np.random.default_rng(0))
        self.assertTrue(case["passed"])
        self.assertTrue(all(w["quadratic_form"] < 0 for w in case["witness"]["intervals"]))

    def test_singular_boundary_is_allowed_but_negative_variance_is_not(self):
        B = np.diag([1., 0.])
        R, tolerance = transition_covariance(B, self.Q, 1., 0.01, 0., 0.5)
        factor, correction = psd_factor(R, tolerance)
        np.testing.assert_allclose(factor @ factor.T, R, atol=tolerance)
        self.assertEqual(correction, 0)
        self.assertEqual(np.linalg.matrix_rank(R), 1)
        with self.assertRaisesRegex(ValueError, "Negative transition variance"):
            psd_factor(np.diag([-0.01, 1.]), 1e-12)
        _, correction = psd_factor(np.diag([-1e-16, 1.]), 1e-14)
        self.assertEqual(correction, 1e-16)

    def test_wrong_transition_formula_is_detected(self):
        original = transition_covariance

        def wrong(*args):
            R, tolerance = original(*args)
            return R + 0.01 * np.eye(len(R)), tolerance

        with patch("verification.forward_feasibility.transition_covariance", side_effect=wrong):
            case = check_case(self.B, self.Q, 1.2, "below", 0.01, 1., 1000,
                              np.random.default_rng(0))
        self.assertFalse(case["passed"])
        self.assertFalse(case["witness"]["passed"])

    def test_invalid_inputs_are_rejected(self):
        for Q in [np.zeros((2, 2)), np.array([[1., 1.], [0., 1.]]),
                  np.full((2, 2), np.nan)]:
            with self.assertRaises((ValueError, np.linalg.LinAlgError)):
                threshold(self.B, Q)
        for s, t in [(1., 0.), (0., 0.), (-1., 1.)]:
            with self.assertRaises(ValueError):
                transition_covariance(self.B, self.Q, 1., 0.01, s, t)

    def test_full_report_and_invalid_specification(self):
        spec = json.loads(DEFAULT_SPEC.read_text())
        report = run(spec, samples=1000)
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["cases"]), 18)
        self.assertTrue(all(check["passed"] for check in report["transport_checks"]))
        json.dumps(report, allow_nan=False)
        changed = copy.deepcopy(spec)
        changed["arms"]["transported"]["Q"] = spec["arms"]["baseline"]["Q"]
        with self.assertRaisesRegex(ValueError, "Invalid model specification"):
            run(changed)
        with self.assertRaisesRegex(ValueError, "samples"):
            run(spec, samples=0)

    def test_cli_errors_and_report_from_other_directory(self):
        script = Path(__file__).with_name("forward_feasibility.py").resolve()
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "evidence.json"
            result = subprocess.run(
                [sys.executable, str(script), "--samples", "1000", "--output", str(report_path)],
                cwd=directory, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(json.loads(report_path.read_text())["passed"])
            for args in [["--spec", "missing.json"], ["--samples", "0"]]:
                result = subprocess.run(
                    [sys.executable, str(script), *args], cwd=directory,
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 1)
                self.assertIn("Forward feasibility: FAIL", result.stdout)


if __name__ == "__main__":
    unittest.main()
