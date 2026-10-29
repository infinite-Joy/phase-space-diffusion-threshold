"""Regression checks for the model-specification gate and its failure diagnostics."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from verification.model_specification import DEFAULT_SPEC, validate


class ModelSpecificationTests(unittest.TestCase):
    def setUp(self):
        with DEFAULT_SPEC.open() as stream:
            self.spec = json.load(stream)

    def test_recorded_specification_passes_all_six_categories(self):
        results = validate(self.spec)
        self.assertEqual(len(results), 6)
        self.assertFalse(any(results.values()), results)

    def test_missing_categories_and_fields_fail(self):
        for category in self.spec:
            with self.subTest(category=category):
                changed = copy.deepcopy(self.spec)
                del changed[category]
                self.assertTrue(validate(changed)[category])
                changed = copy.deepcopy(self.spec)
                changed[category] = {}
                self.assertTrue(validate(changed)[category])

    def test_deliberate_specification_errors_are_detected(self):
        changes = [
            ("units", "dimensionless_mass", 2),
            ("units", "reference_mass", 10**1000),
            ("units", "reference_mass", True),
            ("units", "reference_mass", float("nan")),
            ("coordinates", "order", ["x1", "x2", "v1", "v2"]),
            ("coordinates", "row_sample_transform", "Y = Z @ A"),
            ("covariance", "conditional", "total covariance"),
            ("initial_smoothing", "sigma0", 0),
            ("initial_smoothing", "target_is_clean_law", True),
            ("initial_smoothing", "terminal_initialization", "moment-matched Gaussian"),
            ("feasibility_scope", "singular_psd_allowed", False),
            ("feasibility_scope", "rules_out_all_stochastic_dynamics", True),
        ]
        for category, field, value in changes:
            with self.subTest(category=category, field=field):
                changed = copy.deepcopy(self.spec)
                changed[category][field] = value
                errors = validate(changed)[category]
                self.assertTrue(any(f"{category}.{field}:" in error for error in errors))

    def test_inconsistent_arm_matrices_fail(self):
        for arm, field, value in [
            ("transported", "Q", self.spec["arms"]["baseline"]["Q"]),
            ("reisotropized", "pulled_back_Q", self.spec["arms"]["baseline"]["Q"]),
            ("transported", "B", self.spec["coordinates"]["B"]),
            ("baseline", "initial_conditional_covariance", [[0] * 4] * 4),
            ("transported", "noise_factor", [[float("nan")] * 4] * 4),
            ("baseline", "Q", [[1, 0], [0, 1]]),
            ("baseline", "Q", "not a matrix"),
            ("baseline", "Q", [["1", "0", "0", "0"]] * 4),
            ("baseline", "Q", [[True, False, False, False]] * 4),
        ]:
            with self.subTest(arm=arm, field=field):
                changed = copy.deepcopy(self.spec)
                changed["arms"][arm][field] = value
                self.assertTrue(validate(changed)["arms"])

    def test_invalid_structure_fails(self):
        for value in [None, [], "specification"]:
            self.assertTrue(any(validate(value).values()))
        changed = copy.deepcopy(self.spec)
        changed["units"] = None
        self.assertTrue(validate(changed)["units"])
        changed = copy.deepcopy(self.spec)
        changed["unexpected"] = True
        self.assertTrue(validate(changed)["document"])

    def test_cli_exit_codes_and_diagnostics(self):
        script = Path(__file__).with_name("model_specification.py").resolve()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spec.json"
            for contents, code, message in [
                (json.dumps(self.spec), 0, "Model specification gate: PASS"),
                ("{}", 1, "units: missing category"),
                ("{", 1, "Cannot read specification"),
            ]:
                with self.subTest(contents=contents[:20]):
                    path.write_text(contents)
                    result = subprocess.run(
                        [sys.executable, str(script), str(path)],
                        cwd=directory, capture_output=True, text=True, check=False,
                    )
                    self.assertEqual(result.returncode, code, result.stderr)
                    self.assertIn(message, result.stdout)
            path.unlink()
            result = subprocess.run(
                [sys.executable, str(script), str(path)],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("Cannot read specification", result.stdout)


if __name__ == "__main__":
    unittest.main()
