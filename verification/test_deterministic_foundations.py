"""Regression tests for the row-1 deterministic evidence and failure controls."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from psdt import hamiltonian_B
from verification.deterministic_foundations import (
    check_case, oscillator_flow, primary_transforms, run,
)
from verification.model_specification import DEFAULT_SPEC


class DeterministicFoundationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(DEFAULT_SPEC.read_text())
        cls.report = run(cls.spec)

    def case(self, A=None, **kwargs):
        return check_case("test", np.eye(2) if A is None else A,
                          kwargs.pop("B", hamiltonian_B(2., d=1)), [2.],
                          kwargs.pop("orthogonal", True),
                          kwargs.pop("symplectic", True), 1., **kwargs)

    def test_full_primary_and_distinct_frequency_grid_passes(self):
        self.assertTrue(self.report["passed"])
        self.assertEqual(len(self.report["cases"]), 47)
        self.assertEqual(sum(c["dimension"] == 2 for c in self.report["cases"]), 15)
        self.assertEqual(sum(c["name"].startswith("distinct-frequency/")
                             for c in self.report["cases"]), 16)
        self.assertTrue(all(c["outcome"] == "supported"
                            for c in self.report["cases"]))
        for case in self.report["cases"]:
            self.assertTrue(all(np.isfinite(case[key])
                                for key in ("condition_number", "determinant")))
        json.dumps(self.report, allow_nan=False)

    def test_primary_families_have_correct_claims(self):
        for dimension in (2, 4):
            J = np.kron(np.eye(dimension // 2), [[0., 1.], [-1., 0.]])
            for name, A, orthogonal, symplectic in primary_transforms(dimension):
                with self.subTest(name=name, dimension=dimension):
                    self.assertEqual(np.allclose(A.T @ A, np.eye(dimension)), orthogonal)
                    self.assertEqual(np.allclose(A.T @ J @ A, J), symplectic)
        with self.assertRaises(ValueError):
            primary_transforms(3)

    def test_negative_controls_detect_the_intended_errors(self):
        self.assertEqual(len(self.report["negative_controls"]), 6)
        for control in self.report["negative_controls"]:
            self.assertEqual(control["outcome"], "supported", control["name"])
            self.assertEqual(control["evidence"]["outcome"], "violated")
        control = self.report["negative_controls"][-1]["evidence"]
        self.assertAlmostEqual(control["determinant"], 1.)

    def test_noncanonical_maps_use_transported_poisson_matrix(self):
        result = self.case(2 * np.eye(2), orthogonal=False, symplectic=False)
        self.assertEqual(result["outcome"], "supported")
        checks = {c["name"]: c for c in result["checks"]}
        self.assertNotIn("canonical Hamiltonian structure", checks)
        self.assertEqual(checks["general-coordinate Hamiltonian structure"]["outcome"],
                         "supported")

    def test_ill_conditioned_map_is_unresolved_not_supported(self):
        result = self.case(np.diag([1e4, 1e-4]), orthogonal=False, symplectic=True)
        self.assertEqual(result["outcome"], "numerically unresolved")

    def test_invalid_inputs_raise_explicit_errors(self):
        for A in (np.eye(3), [[1., 0.], [0., np.nan]], np.zeros((2, 3))):
            with self.subTest(A=A), self.assertRaises(ValueError):
                self.case(A)
        for frequencies, time in (([0.], 1.), ([-1.], 1.), ([np.nan], 1.),
                                  ([2.], np.inf), ([], 1.)):
            with self.subTest(frequencies=frequencies, time=time):
                with self.assertRaises(ValueError):
                    oscillator_flow(frequencies, time)
        with self.assertRaises(ValueError):
            run({})

    def test_production_exponential_error_is_detected(self):
        with patch("verification.deterministic_foundations.Process.M",
                   return_value=np.eye(2)):
            result = self.case()
        self.assertEqual(result["outcome"], "violated")
        self.assertTrue(any("closed form versus production" in c["name"]
                            and c["outcome"] == "violated" for c in result["checks"]))

    def test_production_drift_ordering_error_is_detected(self):
        result = self.case(B=hamiltonian_B(2., d=1).T)
        failures = {c["name"] for c in result["checks"] if c["outcome"] == "violated"}
        self.assertIn("baseline drift on basis and states", failures)
        self.assertIn("baseline Hamiltonian structure", failures)

    def test_wrong_transformed_drift_is_detected(self):
        result = self.case(np.diag([2., .5]), orthogonal=False,
                           transformed_drift=hamiltonian_B(2., d=1))
        self.assertEqual(result["outcome"], "violated")
        self.assertTrue(any(c["name"] == "transformed drift on states"
                            and c["outcome"] == "violated" for c in result["checks"]))

    def test_failed_independent_integrator_is_explicit(self):
        with patch("verification.deterministic_foundations.solve_ivp") as integrator:
            integrator.return_value.success = False
            integrator.return_value.message = "deliberate solver failure"
            with self.assertRaisesRegex(RuntimeError, "Independent integration failed"):
                self.case()

    def test_cli_persists_evidence_and_rejects_invalid_spec(self):
        script = Path(__file__).with_name("deterministic_foundations.py").resolve()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "evidence.json"
            result = subprocess.run(
                [sys.executable, str(script), "--output", str(output)],
                cwd=directory, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue(json.loads(output.read_text())["passed"])
            self.assertIn("full deterministic implementation gate remains pending",
                          output.with_suffix(".md").read_text())
            invalid = Path(directory) / "invalid.json"
            invalid.write_text("{}")
            result = subprocess.run(
                [sys.executable, str(script), "--spec", str(invalid)],
                cwd=directory, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("Invalid primary specification", result.stdout)
            result = subprocess.run(
                [sys.executable, str(script), "--output", str(output.with_suffix(".md"))],
                cwd=directory, capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("--output must have a .json suffix", result.stderr)


if __name__ == "__main__":
    unittest.main()
