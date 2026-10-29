"""Small deterministic fixtures for the optional experiment runner."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from psdt import Process, ScoreModel, hamiltonian_B
from verification.gaussian_sampling import diffusion_shapes
from verification.generalization import (
    AXES, ARMS, DATA, PRIORS, Law, NetworkScore, Oracle, calibrate, draw,
    gaussian_reverse_reference, geometry_checks, integrate, main, make_data,
    make_spec, marginal_law, model_for, noise_shapes, run_case, run_manifest,
    score_checks, score_error, score_quadratic_checks, terminal_moments,
    train_network, validate_spec,
)


def small_spec():
    spec = make_spec()
    spec.update(profile="custom", steps=[64, 128], particles=[16, 32],
                empirical_points=8, estimated_moment_points=100)
    spec["training"].update(updates=2, width=8, depth=1, batch=8, validation_size=8,
                            validate_every=1)
    return spec


class GeneralizationTests(unittest.TestCase):
    def setUp(self):
        self.spec = small_spec()
        self.case = copy.deepcopy(next(
            c for c in self.spec["cases"] if c["axis"] == "data"
            and c["data"] == "gaussian" and c["arm"] == "baseline"))
        self.law, _ = make_data("gaussian", self.spec)
        self.model = model_for(self.law, self.spec, self.case)
        self.A = np.array(self.spec["transforms"]["cross_pair"])
        self.A[0] *= 1.5

    def test_full_manifest_covers_every_axis_and_policy(self):
        spec = make_spec("full")
        validate_spec(spec)
        self.assertEqual({c["axis"] for c in spec["cases"]}, set(AXES))
        self.assertEqual({c["data"] for c in spec["cases"]}, set(DATA))
        self.assertEqual({c["prior"] for c in spec["cases"]}, set(PRIORS))
        self.assertEqual({c["arm"] for c in spec["cases"]}, set(ARMS))
        self.assertEqual(spec["training_seeds"], [0, 1, 2])
        self.assertTrue(any(c["score"] == "independent_bias" for c in spec["cases"]))
        for name, shape in noise_shapes().items():
            Q = np.array(shape["Q"])
            if name.endswith("_trace"):
                self.assertAlmostEqual(np.trace(Q), 4.)
            if name.endswith("_det"):
                self.assertAlmostEqual(np.linalg.det(Q), 1.)

    def test_invalid_manifests_are_explicit(self):
        edits = [
            lambda s: s.update(steps=[32, 65]),
            lambda s: s.update(particles=[2, 3]),
            lambda s: s.update(seeds=[0, 0]),
            lambda s: s["cases"][0].update(rate=0),
            lambda s: s["cases"][0].update(id="../escape"),
            lambda s: s["cases"][0].update(score="learned", training_seed=None),
            lambda s: s["shapes"]["isotropic"].update(Q=(-np.eye(4)).tolist()),
            lambda s: s["transforms"].update(decisive=np.zeros((4, 4)).tolist()),
            lambda s: s["acceptance"]["metric_margins"].update(mean=-1),
        ]
        for edit in edits:
            spec = copy.deepcopy(self.spec)
            edit(spec)
            with self.subTest(edit=edit), self.assertRaises((ValueError, np.linalg.LinAlgError)):
                validate_spec(spec)

    def test_oracles_transport_and_density_for_all_synthetic_data(self):
        for name in ("gaussian", "separated", "overlapping", "imbalanced", "raw_moons"):
            law, heldout = make_data(name, self.spec)
            model = model_for(law, self.spec, self.case)
            result = score_checks(law, model, self.A, 2e-5)
            self.assertTrue(result["passed"], (name, result))
            if heldout is not None:
                self.assertFalse(np.array_equal(law.means, heldout))
            if name == "raw_moons":
                # Exercise merging across the 128-component chunk boundary.
                points = np.tile(law.means, (20, 1))
                large = Law(np.full(len(points), 1 / len(points)), points,
                            np.zeros((len(points), 4, 4)), True)
                z = draw(law, model, 0.1, 5, np.random.default_rng(3))
                np.testing.assert_allclose(Oracle(large, model)(z, 0.1),
                                           Oracle(law, model)(z, 0.1), atol=1e-10)

    def test_wrong_oracle_score_fails_independent_check(self):
        original = Oracle.__call__
        with patch.object(Oracle, "__call__", lambda obj, z, t: -original(obj, z, t)):
            self.assertFalse(score_checks(self.law, self.model, self.A, 2e-5)["passed"])

    def test_shapes_energy_threshold_and_transport(self):
        for shape in self.spec["shapes"]:
            case = dict(self.case, shape=shape)
            model = model_for(self.law, self.spec, case)
            report = geometry_checks(model, self.A)
            self.assertTrue(report["passed"], report)
            if shape.startswith("energy"):
                self.assertAlmostEqual(report["threshold"], 0.)
        target = marginal_law(self.law, self.model, 0.)
        np.testing.assert_allclose(target.moments()[1],
                                   self.law.moments()[1] + self.model.sigma0**2 * self.model.Q)

    def test_coupled_paths_use_transported_factor_below_threshold(self):
        model = model_for(self.law, self.spec, dict(self.case, rate=0.9))
        score = Oracle(self.law, model)
        terminal = draw(self.law, model, model.horizon, 16, np.random.default_rng(4))
        levels = integrate(score, model, terminal, [32, 64], 5, self.A)
        inverse = np.linalg.inv(self.A)
        other = model.transported(self.A)
        wrapped = lambda y, t: score(y @ inverse.T, t) @ inverse
        G, D, L = diffusion_shapes(model)
        transported = integrate(wrapped, other, terminal @ self.A.T, [32, 64], 5,
                                inverse, (self.A @ G @ self.A.T,
                                          self.A @ D @ self.A.T, self.A @ L))
        for k in levels:
            for mode in ("ode", "naive", "fixed"):
                self.assertLess(levels[k]["path_errors"][mode], 1e-8)
                np.testing.assert_allclose(
                    transported[k]["states"][mode] @ inverse.T,
                    levels[k]["states"][mode], atol=1e-8, rtol=1e-8)
        with self.assertRaisesRegex(ValueError, "Brownian factor"):
            integrate(score, model, terminal, [32, 64], 5, self.A, (G, D, 2 * L))

    def test_network_batched_score_and_checkpoint(self):
        network = NetworkScore(self.model, self.spec["training"], 7)
        z = np.random.default_rng(8).standard_normal((7, 4))
        t = np.linspace(0, self.model.horizon, len(z))
        actual = network(z, t)
        expected = np.vstack([network(z[i:i + 1], t[i]) for i in range(len(z))])
        np.testing.assert_allclose(actual, expected)
        self.assertTrue(np.isfinite(actual).all())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.npz"
            metadata = {"identity": "test"}
            network.save(path, metadata)
            other = NetworkScore(self.model, self.spec["training"], 9)
            other.load(path, "test")
            np.testing.assert_array_equal(network(z, t), other(z, t))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                other.load(path, "different")

    def test_existing_isotropic_score_interface_matches_extension(self):
        process = Process(hamiltonian_B(2.), self.model.rate)
        isotropic = ScoreModel(process, np.random.default_rng(7), width=8, depth=1)
        network = NetworkScore(self.model, self.spec["training"], 7)
        z = np.random.default_rng(8).standard_normal((7, 4))
        t = np.linspace(0., self.model.horizon, len(z))
        expected = np.vstack([isotropic.score(z[i:i + 1], t[i]) for i in range(len(z))])
        np.testing.assert_allclose(isotropic.score(z, t), expected)
        np.testing.assert_allclose(network(z, t), expected, atol=1e-12)
        square = z[:4]
        np.testing.assert_allclose(isotropic.score(square, t[:4]), expected[:4])
        np.testing.assert_allclose(isotropic.score(z, 0.3),
                                   np.vstack([isotropic.score(row[None], 0.3) for row in z]))

    def test_anisotropic_epsilon_prediction_is_covariance_aware(self):
        model = model_for(self.law, self.spec, dict(self.case, shape="correlated_trace"))
        network = NetworkScore(model, self.spec["training"], 7)
        eps = np.random.default_rng(8).standard_normal((7, 4))
        t = np.linspace(0, model.horizon, len(eps))
        sigma = model.sigma0 * np.exp(model.rate * t)
        L = np.linalg.cholesky(model.Q)
        noise = sigma[:, None] * (eps @ L.T)
        expected = np.vstack([-np.linalg.solve(s**2 * model.Q, row)
                              for s, row in zip(sigma, noise)])
        with patch.object(network.net, "forward", return_value=eps):
            np.testing.assert_allclose(network(noise, t), expected, atol=1e-12)

    def test_training_repeatability_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            a, metadata = train_network(self.law, self.model, self.spec, 0,
                                         directory, "code", False)
            z = draw(self.law, self.model, 0.2, 8, np.random.default_rng(2))
            first = a(z, 0.2).copy()
            b, resumed = train_network(self.law, self.model, self.spec, 0,
                                       directory, "code", True)
            np.testing.assert_array_equal(first, b(z, 0.2))
            self.assertEqual(metadata["checkpoint_sha256"], resumed["checkpoint_sha256"])
            self.assertTrue(resumed["completed"])
            self.assertEqual(len(resumed["validation"]), 2)
            self.assertIn("data_hash", resumed)
            self.assertIn("specification_hash", resumed)
            np.testing.assert_allclose(resumed["normalization_matrix"], np.eye(4))
            changed = copy.deepcopy(self.spec)
            changed["particles"] = [16, 64]
            _, different = train_network(self.law, self.model, changed, 0,
                                          directory, "code", False)
            self.assertNotEqual(metadata["identity"], different["identity"])

    def test_calibration_bias_and_gaussian_prior_null(self):
        target = marginal_law(self.law, self.model, 0.)
        reference, P, bw, controls = calibrate(target, self.spec, False, 0)
        from verification.generalization import observed_metrics, terminal_prior
        biased = observed_metrics(reference + 10, reference, target, P, bw, False)
        self.assertGreater(biased["mean"], controls["32"]["bands"]["mean"])
        ref_exact = gaussian_reverse_reference(self.law, self.model, "exact",
                                                self.spec, "ode")
        ref_moment = gaussian_reverse_reference(self.law, self.model, "moment",
                                                 self.spec, "ode")
        np.testing.assert_array_equal(ref_exact["mean"], ref_moment["mean"])
        np.testing.assert_array_equal(ref_exact["covariance"], ref_moment["covariance"])
        for policy in PRIORS:
            prior = terminal_prior(self.law, self.model, policy, 32, 0, 100)
            self.assertEqual(prior.shape, (32, 4))
            self.assertTrue(np.isfinite(prior).all())
        mean, covariance = terminal_moments(self.law, self.model, "exact", 100)
        for policy in ("moment", "shift", "scale_0.8", "scale_1.2"):
            pm, pc = terminal_moments(self.law, self.model, policy, 100)
            np.testing.assert_allclose(pm, mean + (
                np.array([0.5, 0, 0, 0]) if policy == "shift" else 0))
            np.testing.assert_allclose(pc, covariance * (
                float(policy.split("_")[1]) if policy.startswith("scale_") else 1))

    def test_score_error_zero_and_transport(self):
        oracle = Oracle(self.law, self.model)
        z = draw(self.law, self.model, 0.3, 8, np.random.default_rng(12))
        for kind in ("zero", "bias", "scale", "smooth"):
            score, _ = score_error(oracle, self.model, kind, 0.)
            np.testing.assert_array_equal(score(z, 0.3), oracle(z, 0.3))
        model = model_for(self.law, self.spec, dict(self.case, rate=0.9))
        oracle = Oracle(self.law, model)
        score, _ = score_error(oracle, model, "bias", 0.2)
        report = score_quadratic_checks(score, oracle, self.law, model, self.A, 1e-8)
        self.assertTrue(report["passed"])
        self.assertTrue(report["signed_G_not_a_loss"])

    def test_gaussian_reference_respects_transported_clipping(self):
        model = model_for(self.law, self.spec, dict(self.case, rate=0.9))
        other = model.transported(self.A)
        G, D, L = diffusion_shapes(model)
        diffusion = self.A @ G @ self.A.T, self.A @ D @ self.A.T, self.A @ L
        for prior in ("exact", "shift", "scale_0.8"):
            # Compare in one coordinate system so the prior intervention is identical.
            transported = gaussian_reverse_reference(
                self.law.transported(self.A), other, prior, self.spec, "naive", diffusion)
            local = gaussian_reverse_reference(
                self.law.transported(self.A), other, prior, self.spec, "naive")
            self.assertGreater(np.linalg.norm(np.array(transported["covariance"])
                                              - local["covariance"]), 1e-4)
        baseline = gaussian_reverse_reference(self.law, model, "exact", self.spec, "naive")
        exact = gaussian_reverse_reference(self.law.transported(self.A), other, "exact",
                                           self.spec, "naive", diffusion)
        np.testing.assert_allclose(exact["covariance"],
            self.A @ np.array(baseline["covariance"]) @ self.A.T, rtol=1e-7, atol=1e-9)

    def test_run_case_writes_independent_references(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_case(self.spec, self.case, Path(tmp), "code")
        self.assertTrue(result["harness_passed"])
        self.assertTrue(result["score_quadratic_checks"]["passed"])
        self.assertEqual(len(result["runs"]), 2)
        self.assertIn("gaussian_reverse_reference", result["refinement"]["fixed"])
        self.assertIn("endpoint_moments", result["runs"][0])
        json.dumps(result, allow_nan=False)

    def test_completed_sampler_seeds_resume_an_interrupted_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            first = run_case(self.spec, self.case, output, "code")
            seeds = list((output / "seeds").glob("*.json"))
            self.assertEqual(len(seeds), len(self.spec["seeds"]))
            with patch("verification.generalization.integrate",
                       side_effect=AssertionError("Completed seed must not be integrated again")):
                resumed = run_case(self.spec, self.case, output, "code", resume=True)
            self.assertEqual(first["runs"], resumed["runs"])
            changed = copy.deepcopy(self.spec)
            changed["particles"] = [16, 64]
            with self.assertRaisesRegex(ValueError, "Sampler seed fingerprint mismatch"):
                run_case(changed, self.case, output, "code", resume=True)
            with self.assertRaisesRegex(FileExistsError, "Sampler seed exists"):
                run_case(self.spec, self.case, output, "code")

    def test_empirical_and_learned_cases_end_to_end(self):
        for data, score, arm in (("raw_moons", "exact", "reisotropized"),
                                 ("gaussian", "learned", "transported"),
                                 ("gaussian", "learned", "retrained_A")):
            case = dict(self.case, data=data, score=score, arm=arm,
                        training_seed=0 if score == "learned" else None)
            with self.subTest(data=data, score=score, arm=arm), tempfile.TemporaryDirectory() as tmp:
                result = run_case(self.spec, case, Path(tmp), "code")
                self.assertTrue(result["harness_passed"])
                if data == "raw_moons":
                    self.assertIn("heldout_population_proxy", result["runs"][0])
                    self.assertNotIn("component_mass_max_error", result["calibration"]["32"]["bands"])
                if score == "learned":
                    self.assertTrue(Path(result["training"]["checkpoint"]).exists())
                json.dumps(result, allow_nan=False)

    def test_resume_preserves_failed_outcomes_and_detects_changed_checkpoint(self):
        spec = copy.deepcopy(self.spec)
        spec["cases"] = [self.case]
        fake = {"case": self.case, "harness_passed": True, "required_quality_passed": False,
                "outcome": "violated", "training": None}
        with tempfile.TemporaryDirectory() as tmp, patch(
                "verification.generalization.run_case", return_value=fake) as runner:
            directory = Path(tmp)
            self.assertFalse(run_manifest(spec, directory))
            self.assertFalse(run_manifest(spec, directory, resume=True))
            self.assertEqual(runner.call_count, 1)
            report = json.loads((directory / "summary.json").read_text())
            self.assertFalse(report["passed"])
            self.assertEqual(report["coverage"][self.case["id"]], "violated")
            changed = copy.deepcopy(spec)
            changed["particles"] = [16, 64]
            with self.assertRaisesRegex(ValueError, "different manifest"):
                run_manifest(changed, directory, resume=True)

    def test_checkpoint_mutation_and_partial_coverage(self):
        spec = copy.deepcopy(self.spec)
        first = dict(self.case, score="learned", training_seed=0)
        second = dict(first, id="second-case")
        spec["cases"] = [first, second]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            self.assertTrue(run_manifest(spec, output, case_ids=[first["id"]]))
            self.assertTrue(run_manifest(spec, output, case_ids=[second["id"]], resume=True))
            report = json.loads((output / "summary.json").read_text())
            self.assertTrue(report["full_manifest_completed"])
            evidence = json.loads((output / "cases" / f"{first['id']}.json").read_text())
            checkpoint = Path(evidence["training"]["checkpoint"])
            checkpoint.write_bytes(b"damaged fixture")
            self.assertFalse(run_manifest(spec, output, case_ids=[first["id"]], resume=True))
            error = json.loads((output / "failures" / f"{first['id']}.json").read_text())
            self.assertIn("Checkpoint changed", error["error"])

    def test_missing_cache_failure_is_persistent(self):
        spec = copy.deepcopy(self.spec)
        case = dict(self.case, id="missing-cache", data="fitted_moons")
        spec["cases"] = [case]
        spec["mixture_path"] = "/nonexistent/generalization-fixture.npz"
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(run_manifest(spec, Path(tmp)))
            error = json.loads((Path(tmp) / "failures/missing-cache.json").read_text())
            self.assertIn("FileNotFoundError", error["error"])
            report = json.loads((Path(tmp) / "summary.json").read_text())
            self.assertFalse(report["full_manifest_completed"])

    def test_analysis_preserves_nested_training_replicates(self):
        from generalization_analyze import analyze
        spec = copy.deepcopy(self.spec)
        spec["training_seeds"] = [0, 1, 2]
        spec["cases"] = [dict(self.case, id=f"training-{seed}", axis="trained",
                              score="learned", training_seed=seed) for seed in range(3)]
        spec["cases"].append(dict(spec["cases"][0], id="transported-network", arm="transported"))
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "original"
            self.assertTrue(run_manifest(spec, output))
            report, _ = analyze(output)
            group = report["training_groups"][0]
            self.assertEqual(group["training_seeds"], [0, 1, 2])
            statistic = group["between_training_seed_means"]["fixed"]["sliced_w2"]
            self.assertEqual(len(statistic["values"]), 3)
            self.assertEqual(len(group["within_checkpoint_sampler_variability"]), 3)
            self.assertEqual(report["missing_case_ids"], [])
            self.assertTrue(report["arm_comparisons"][0]["endpoint_transport_check"]["passed"])
            relocated = Path(tmp) / "relocated"
            output.rename(relocated)
            moved_report, _ = analyze(relocated)
            self.assertTrue(moved_report["full_manifest_completed"])
            checkpoint = next((relocated / "checkpoints").glob("*-best.npz"))
            checkpoint.write_bytes(b"deliberately corrupted")
            with self.assertRaisesRegex(ValueError, "Checkpoint changed"):
                analyze(relocated)

    def test_cli_execution_resume_and_analysis(self):
        script = Path(__file__).with_name("generalization.py").resolve()
        analyzer = script.parents[1] / "generalization_analyze.py"
        spec = copy.deepcopy(self.spec)
        spec["cases"] = [dict(self.case, score="learned", training_seed=0),
                         dict(self.case, id="shift-control", prior="shift", axis="prior")]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            manifest = directory / "spec.json"
            manifest.write_text(json.dumps(spec))
            output = directory / "evidence"
            command = [sys.executable, str(script), "--spec", str(manifest),
                       "--output", str(output)]
            result = subprocess.run(command, cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run(command + ["--resume"], cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("Sampler seed", result.stderr)
            result = subprocess.run([sys.executable, str(analyzer), "--input", str(output)],
                                    cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            analysis = json.loads((output / "analysis/analysis.json").read_text())
            self.assertTrue(analysis["full_manifest_completed"])
            self.assertTrue((output / "analysis/figures/data-quality.png").exists())
            self.assertTrue((output / "analysis/figures/prior-positions.png").exists())
            self.assertTrue((output / "analysis/figures/data-particle-refinement.png").exists())
            self.assertTrue((output / "analysis/report.md").exists())
            self.assertTrue((output / "analysis/assumption-ledger.json").exists())
            self.assertTrue((output / "analysis/interaction-summary.json").exists())
            self.assertEqual(analysis["study_acceptance"], "not assessed")
            self.assertIn("particle_refinement", analysis["cases"][0])
            self.assertIn("velocity_mean", analysis["cases"][0]["endpoint_diagnostics"]["fixed"])

    def test_cli_manifest_and_dry_run_from_another_directory(self):
        script = Path(__file__).with_name("generalization.py").resolve()
        with tempfile.TemporaryDirectory() as tmp:
            spec_path = Path(tmp) / "spec.json"
            result = subprocess.run([sys.executable, str(script), "--write-spec", str(spec_path)],
                                    cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            validate_spec(json.loads(spec_path.read_text()))
            result = subprocess.run([sys.executable, str(script), "--spec", str(spec_path),
                                     "--dry-run", "--axis", "trained", "--max-cases", "1"],
                                    cwd=tmp, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("1 selected", result.stdout)
            self.assertFalse((Path(tmp) / "results").exists())
            self.assertEqual(main(["--dry-run", "--case", "not-a-case"]), 1)


if __name__ == "__main__":
    unittest.main()
