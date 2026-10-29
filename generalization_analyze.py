"""Analyze completed Section 11 evidence without running experiments.

Run: python generalization_analyze.py --input results/coordinate_invariance/generalization
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from verification.generalization import AXES, digest, make_spec, summary, validate_spec, write_json


def finest(run, mode):
    levels = [row for row in run["levels"] if row["mode"] == mode]
    level = max(levels, key=lambda row: row["steps"])
    return level["particles"][-1]["metrics"]


def comparison_key(case, omitted):
    return digest({key: value for key, value in case.items() if key not in {"id", *omitted}})


def analyze(directory):
    spec = json.loads((directory / "manifest.json").read_text())
    validate_spec(spec)
    run_summary = json.loads((directory / "summary.json").read_text())
    if run_summary["manifest_hash"] != digest(spec):
        raise ValueError("Summary and manifest do not match")
    rows, completed = [], []
    for case in spec["cases"]:
        path = directory / "cases" / f"{case['id']}.json"
        failure = directory / "failures" / f"{case['id']}.json"
        if not path.exists() or run_summary["coverage"].get(case["id"]) == "not assessed":
            message = json.loads(failure.read_text()).get("error") if failure.exists() else None
            rows.append({"case": case, "status": "not assessed", "error": message})
            continue
        record = json.loads(path.read_text())
        if record.get("run_identity") != run_summary["run_identity"]:
            rows.append({"case": case, "status": "not assessed",
                         "error": "Evidence belongs to a different code/input fingerprint"})
            continue
        training = record.get("training")
        if training and hashlib.sha256(Path(training["checkpoint"]).read_bytes()).hexdigest() != (
                training["checkpoint_sha256"]):
            raise ValueError(f"Checkpoint changed: {training['checkpoint']}")
        completed.append(record)
        rows.append({
            "case": case, "status": record["outcome"], "elapsed_seconds": record["elapsed_seconds"],
            "threshold": record["geometry"]["threshold"],
            "sampler_seed_metrics": {
                mode: {key: summary([float(finest(run, mode)[key]) for run in record["runs"]])
                       for key in record["calibration"][str(spec["particles"][-1])]["bands"]}
                for mode in ("ode", "naive", "fixed")},
            "initial_metrics": {
                key: summary([float(run["initial_metrics"][key]) for run in record["runs"]])
                for key in record["terminal_calibration"][str(spec["particles"][-1])]["bands"]},
            "score_error_by_time": record["heldout_score_evaluation"],
            "refinement": record["refinement"],
            "training": record["training"],
            "particle_refinement": {
                mode: {str(n): {
                    key: summary([
                        float(next(level for level in run["levels"]
                                   if level["mode"] == mode and level["steps"] == spec["steps"][-1])
                              ["particles"][index]["metrics"][key])
                        for run in record["runs"]])
                    for key in record["calibration"][str(n)]["bands"]}
                    for index, n in enumerate(spec["particles"])}
                for mode in ("ode", "naive", "fixed")},
            "heldout_population_proxy": {
                mode: {key: summary([float(run["heldout_population_proxy"][mode][key])
                                     for run in record["runs"]])
                       for key in record["calibration"][str(spec["particles"][-1])]["bands"]}
                for mode in ("ode", "naive", "fixed")
            } if "heldout_population_proxy" in record["runs"][0] else None,
            "endpoint_diagnostics": {
                mode: {
                    "velocity_mean": [
                            summary([float(run["endpoint_moments"][str(spec["steps"][-1])]
                                           [mode]["mean"][j]) for run in record["runs"]])
                            for j in (1, 3)],
                    "velocity_variance": [
                        summary([float(run["endpoint_moments"][str(spec["steps"][-1])]
                                       [mode]["covariance"][j][j]) for run in record["runs"]])
                        for j in (1, 3)],
                    "component_responsibility_mass": [
                        finest(run, mode).get("component_responsibility_mass")
                        for run in record["runs"]],
                } for mode in ("ode", "naive", "fixed")},
        })
    comparisons = []
    policies = [
        ("prior", {"prior"}, lambda c: c["prior"] == "exact"),
        ("noise", {"shape"}, lambda c: c["shape"] == "isotropic"),
        ("score", {"score", "amplitude"}, lambda c: c["score"] == "zero"),
        ("trained", {"arm"}, lambda c: c["arm"] == "baseline"),
        ("interaction", {"prior"}, lambda c: c["prior"] == "exact"),
    ]
    for axis, omitted, is_control in policies:
        controls = {comparison_key(row["case"], omitted): row for row in completed
                    if row["case"]["axis"] == axis and is_control(row["case"])}
        for record in completed:
            case = record["case"]
            if case["axis"] != axis or is_control(case):
                continue
            control = controls.get(comparison_key(case, omitted))
            if control is None:
                continue
            control_runs = {run["seed"]: run for run in control["runs"]}
            paired = [(run, control_runs[run["seed"]]) for run in record["runs"]
                      if run["seed"] in control_runs]
            keys = record["calibration"][str(spec["particles"][-1])]["bands"]
            comparisons.append({
                "axis": axis, "case_id": case["id"], "control_id": control["case"]["id"],
                "pairing": "same sampler seed; training seed held fixed",
                "endpoint_effects": {
                    mode: {key: summary([
                        float(finest(run, mode)[key] - finest(reference, mode)[key])
                        for run, reference in paired]) for key in keys}
                    for mode in ("ode", "naive", "fixed")},
                "initial_effects": {key: summary([
                    float(run["initial_metrics"][key] - reference["initial_metrics"][key])
                    for run, reference in paired]) for key in keys},
            })
    arm_comparisons = []
    controls = {comparison_key(row["case"], {"arm"}): row for row in completed
                if row["case"]["arm"] == "baseline"}
    for record in completed:
        case = record["case"]
        if case["arm"] == "baseline":
            continue
        control = controls.get(comparison_key(case, {"arm"}))
        if control is None:
            continue
        control_runs = {run["seed"]: run for run in control["runs"]}
        paired = [(run, control_runs[run["seed"]]) for run in record["runs"]
                  if run["seed"] in control_runs]
        keys = record["calibration"][str(spec["particles"][-1])]["bands"]
        arm_comparisons.append({
            "axis": case["axis"], "case_id": case["id"], "control_id": control["case"]["id"],
            "arm": case["arm"], "score": case["score"],
            "interpretation": ("Transported complete model; independent_bias is a separate intervention"
                               if case["arm"] == "transported" else
                               "Changed noise geometry or independently retrained network"),
            "endpoint_effects": {
                mode: {key: summary([float(finest(run, mode)[key] - finest(reference, mode)[key])
                                     for run, reference in paired]) for key in keys}
                for mode in ("ode", "naive", "fixed")},
        })
    # Average sampler replicates within a checkpoint before estimating
    # between-training variability. Never pool all nested sampler seeds.
    groups = {}
    for row in rows:
        case = row["case"]
        if case["score"] != "learned" or "sampler_seed_metrics" not in row:
            continue
        key = comparison_key(case, {"training_seed"})
        groups.setdefault(key, []).append(row)
    training_groups = []
    for group in groups.values():
        training_groups.append({
            "case_without_training_seed": {k: v for k, v in group[0]["case"].items()
                                           if k not in ("id", "training_seed")},
            "training_seeds": [row["case"]["training_seed"] for row in group],
            "between_training_seed_means": {
                mode: {key: summary([row["sampler_seed_metrics"][mode][key]["mean"]
                                     for row in group])
                       for key in group[0]["sampler_seed_metrics"][mode]}
                for mode in ("ode", "naive", "fixed")},
            "within_checkpoint_sampler_variability": [
                {"training_seed": row["case"]["training_seed"],
                 "metrics": row["sampler_seed_metrics"]} for row in group],
        })
    full_spec = make_spec("full", spec["mixture_path"])
    exposure = (len(spec["seeds"]) * spec["particles"][-1] * sum(spec["steps"]))
    full_exposure = (len(full_spec["seeds"]) * full_spec["particles"][-1]
                     * sum(full_spec["steps"]))
    return {
        "scope": "Optional generalization; descriptive intervals, not equivalence certification",
        "profile": spec["profile"], "manifest_hash": digest(spec),
        "coverage": {axis: {
            status: sum(row["case"]["axis"] == axis and row["status"] == status for row in rows)
            for status in ("supported", "violated", "not assessed", "numerically unresolved")}
            for axis in AXES},
        "full_manifest_completed": len(completed) == len(spec["cases"]),
        "full_manifest_passed": len(completed) == len(spec["cases"]) and all(
            record["harness_passed"] and record["required_quality_passed"] for record in completed),
        "study_acceptance": "not assessed",
        "study_acceptance_reason": "Core prerequisite gates are not certified by these artifacts; "
                                   "smoke/custom budgets are not full-study evidence.",
        "total_case_seconds": sum(row.get("elapsed_seconds", 0) for row in rows),
        "process_peak_rss_kib": max((row["process_peak_rss_kib"] for row in completed), default=0),
        "settings": {key: spec[key] for key in (
            "steps", "particles", "seeds", "training_seeds", "training", "empirical_points")},
        "compute_cost": {
            axis: {"case_seconds": sum(row["elapsed_seconds"] for row in completed
                                       if row["case"]["axis"] == axis),
                   "completed_cases": sum(row["case"]["axis"] == axis for row in completed)}
            for axis in AXES},
        "full_profile_budget": {
            "manifest_cases": len(full_spec["cases"]),
            "sampler_particle_step_ratio_per_case": full_exposure / exposure,
            "calibration_particle_pair_ratio": (
                full_spec["particles"][-1] / spec["particles"][-1])**2
                * full_spec["calibration_repeats"] / spec["calibration_repeats"],
            "empirical_component_ratio": full_spec["empirical_points"] / spec["empirical_points"],
            "limitation": "Operation-count ratios, not wall-clock predictions. Different data, "
                          "batching, training widths/budgets and cache reuse change runtime. "
                          "Time a full-budget case before expanding; these ratios do not "
                          "establish runtime or clear prerequisite gates.",
        },
        "cases": rows, "paired_comparisons": comparisons, "training_groups": training_groups,
        "arm_comparisons": arm_comparisons,
        "missing_case_ids": [row["case"]["id"] for row in rows if row["status"] == "not assessed"],
        "outcome_definition": "Supported means expected implementation checks and declared "
                              "exact-target controls passed, not universal robustness of "
                              "approximate priors or learned scores.",
        "assumption_ledger": [
            {"case_id": record["case"]["id"], "requirement": requirement,
             "classification": category, "outcome": "supported" if passed else "violated",
             "evidence": f"cases/{record['case']['id']}.json",
             "tolerance_or_uncertainty": {
                 "score_residual": spec["acceptance"]["score_residual"],
                 "path_residual": spec["acceptance"]["path_residual"],
                 "geometry": "Condition-aware residual and eigenvalue bands in the case artifact",
                 "quality": "Frozen target calibration bands and independent Gaussian-reference "
                            "limits in the case artifact; seed intervals are descriptive",
             },
             "limitation": limitation}
            for record in completed
            for requirement, category, passed, limitation in (
                ("Fixed-shape covariance transport and generalized threshold", "identity",
                 record["geometry"]["passed"], "Does not establish a physical noise model"),
                ("Exact oracle score, density Jacobian, and score transport", "identity",
                 record["score_checks"]["passed"], "Exact relative to the recorded clean law"),
                ("Fully coupled sampler transport", "identity",
                 all(level["path_residual"] <= spec["acceptance"]["path_residual"]
                     for run in record["runs"] for level in run["levels"]),
                 "Local re-clipping is a different intervention"),
                ("Quadratic score-error transport including signed e^T G e", "identity",
                 record.get("score_quadratic_checks", {}).get("passed", False),
                 "Does not equate independently introduced score errors"),
                ("Declared exact-target and Gaussian-reference controls", "empirical requirement",
                 record["required_quality_passed"],
                 "Does not require approximate-prior or learned-score bias to vanish"),
            )
        ] + [{"case_id": row["case"]["id"], "requirement": "Planned case execution",
              "classification": "empirical requirement", "outcome": "not assessed",
              "evidence": f"failures/{row['case']['id']}.json" if row.get("error") else None,
              "limitation": row.get("error") or "No matching completed evidence"}
             for row in rows if row["status"] == "not assessed"
        ] + [{"requirement": "Physical relevance and outstanding core-gate certification",
              "classification": "model choice", "outcome": "not assessed",
              "limitation": "Optional synthetic experiments cannot establish physical validity"}],
        "limitations": run_summary["limitations"],
    }, completed


def plots(records, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    for axis in AXES:
        cases = [record for record in records if record["case"]["axis"] == axis]
        if not cases:
            continue
        fig, ax = plt.subplots(figsize=(max(6, min(18, len(cases) * 0.4)), 4))
        for mode in ("ode", "naive", "fixed"):
            values = [np.mean([finest(run, mode)["sliced_w2"] for run in row["runs"]])
                      for row in cases]
            ax.plot(range(len(cases)), values, ".-", label=mode)
        ax.set(xlabel="Manifest case index (see analysis.json)", ylabel="Pulled-back sliced W2",
               title=f"{axis}: endpoint quality (different cases are not replicates)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{axis}-quality.png")
        plt.close(fig)
        first = cases[0]
        target = np.array(first["target_preview"])
        samples = first["runs"][0]["sample_preview"]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3))
        for ax, mode in zip(axes, ("ode", "naive", "fixed")):
            z = np.array(samples[mode])
            ax.scatter(target[:, 0], target[:, 2], s=3, alpha=0.3, label="smoothed target")
            ax.scatter(z[:, 0], z[:, 2], s=3, alpha=0.3, label=mode)
            ax.set(title=mode, xlabel="x1", ylabel="x2")
        axes[0].legend()
        fig.suptitle(first["case"]["id"] + " (first sampler seed)")
        fig.tight_layout()
        fig.savefig(output / f"{axis}-positions.png")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        grids = sorted({row["steps"] for row in first["runs"][0]["levels"]})
        for mode in ("ode", "naive", "fixed"):
            errors = [
                np.mean([next(row for row in run["levels"]
                              if row["steps"] == k and row["mode"] == mode)
                         ["particles"][-1]["metrics"]["sliced_w2"] for run in first["runs"]])
                for k in grids]
            ax.plot(grids, errors, "o-", label=mode)
        ax.set(xscale="log", xlabel="Steps", ylabel="Pulled-back sliced W2",
               title=first["case"]["id"] + ": refinement")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{axis}-refinement.png")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        for mode in ("ode", "naive", "fixed"):
            levels = [next(level for level in run["levels"]
                           if level["mode"] == mode
                           and level["steps"] == max(grids)) for run in first["runs"]]
            counts = [row["particles"] for row in levels[0]["particles"]]
            values = [np.mean([level["particles"][i]["metrics"]["sliced_w2"]
                               for level in levels]) for i in range(len(counts))]
            ax.plot(counts, values, "o-", label=mode)
        ax.set(xscale="log", xlabel="Particles", ylabel="Pulled-back sliced W2",
               title=first["case"]["id"] + ": particle refinement")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{axis}-particle-refinement.png")
        plt.close(fig)

    noise = [row for row in records if row["case"]["axis"] == "noise"]
    if noise:
        fig, ax = plt.subplots(figsize=(8, 4))
        for arm in ("baseline", "transported", "reisotropized"):
            selected = [(i, row) for i, row in enumerate(noise) if row["case"]["arm"] == arm]
            if selected:
                ax.scatter([i for i, _ in selected],
                           [row["geometry"]["threshold"] for _, row in selected],
                           s=15, label=arm)
        ax.set(xlabel="Noise-axis case index (see analysis.json)", ylabel="Generalized threshold",
               title="Noise geometry and arm-specific feasibility thresholds")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "noise-thresholds.png")
        plt.close(fig)

    priors = [row for row in records if row["case"]["axis"] in ("prior", "interaction")]
    if priors:
        fig, ax = plt.subplots(figsize=(6, 4))
        for mode in ("ode", "naive", "fixed"):
            ax.scatter([np.mean([r["initial_metrics"]["mmd2"] for r in row["runs"]])
                        for row in priors],
                       [np.mean([finest(r, mode)["mmd2"] for r in row["runs"]])
                        for row in priors], label=mode, s=15)
        ax.set(xlabel="Terminal MMD squared", ylabel="Endpoint MMD squared",
               title="Prior discrepancy: initialization versus endpoint")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "prior-initial-versus-endpoint.png")
        plt.close(fig)

    scored = [row for row in records if row["case"]["score"] not in ("exact", "zero")]
    if scored:
        fig, ax = plt.subplots(figsize=(8, 4))
        for row in scored[:10]:
            bins = [value for value in row["heldout_score_evaluation"] if "time_fraction" in value]
            ax.plot([value["time_fraction"] for value in bins],
                    [value["score_Q_mse"] for value in bins], ".-", label=row["case"]["id"])
        ax.set(xlabel="Time / horizon", ylabel="Held-out e^T Q e", yscale="symlog",
               title="Score error by noise level (first ten scored cases)")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(output / "score-error-by-time.png")
        plt.close(fig)

    trained = [row for row in records if row["training"] is not None]
    if trained:
        fig, ax = plt.subplots(figsize=(8, 4))
        seen = set()
        for row in trained:
            training = row["training"]
            if training["identity"] in seen:
                continue
            seen.add(training["identity"])
            history = training["validation"]
            ax.plot([value["step"] for value in history], [value["loss"] for value in history],
                    label=f"{row['case']['data']}, seed {row['case']['training_seed']}")
            if len(seen) == 10:
                break
        ax.set(xlabel="Training update", ylabel="Validation epsilon MSE",
               title="Validation-selected checkpoints (first ten distinct trainings)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / "training-validation.png")
        plt.close(fig)

        groups = {}
        for row in trained:
            key = comparison_key(row["case"], {"training_seed"})
            groups.setdefault(key, []).append(row)
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, group in enumerate(list(groups.values())[:20]):
            means = [np.mean([finest(run, "fixed")["sliced_w2"] for run in row["runs"]])
                     for row in group]
            ax.scatter([i] * len(means), means, s=15)
        ax.set(xlabel="Training configuration group (see analysis.json)",
               ylabel="Corrected sliced W2, averaged within checkpoint",
               title="Independent training-seed variability (first twenty groups)")
        fig.tight_layout()
        fig.savefig(output / "training-seed-variability.png")
        plt.close(fig)


def write_reports(report, output):
    write_json(output / "assumption-ledger.json", report["assumption_ledger"])
    for axis in AXES:
        write_json(output / f"{axis}-summary.json", {
            "scope": "Interaction appendix" if axis == "interaction" else f"{axis} sensitivity",
            "coverage": report["coverage"][axis],
            "tested_data": sorted({row["case"]["data"] for row in report["cases"]
                                   if row["case"]["axis"] == axis
                                   and row["status"] != "not assessed"}),
            "tested_shapes": sorted({row["case"]["shape"] for row in report["cases"]
                                     if row["case"]["axis"] == axis
                                     and row["status"] != "not assessed"}),
            "tested_priors": sorted({row["case"]["prior"] for row in report["cases"]
                                     if row["case"]["axis"] == axis
                                     and row["status"] != "not assessed"}),
            "cases": [row for row in report["cases"] if row["case"]["axis"] == axis],
            "paired_comparisons": [row for row in report["paired_comparisons"]
                                   if row["axis"] == axis],
            "arm_comparisons": [row for row in report["arm_comparisons"] if row["axis"] == axis],
            "training_groups": [row for row in report["training_groups"]
                                if row["case_without_training_seed"]["axis"] == axis],
        })
    lines = [
        "# Optional generalization analysis", "",
        f"Profile: `{report['profile']}`. These results do not certify core gates.", "",
        "| Axis | Supported | Violated | Not assessed | Numerically unresolved |",
        "|---|---:|---:|---:|---:|",
    ]
    for axis, counts in report["coverage"].items():
        lines.append(f"| {axis} | " + " | ".join(str(counts[key]) for key in (
            "supported", "violated", "not assessed", "numerically unresolved")) + " |")
    lines += [
        "", f"Full manifest completed: **{report['full_manifest_completed']}**. "
        f"Full manifest passed: **{report['full_manifest_passed']}**.", "",
        f"Case execution time: {report['total_case_seconds']:.1f} seconds. "
        f"Process peak RSS: {report['process_peak_rss_kib'] / 1024:.1f} MiB "
        "(process-wide high-water mark, not per-case allocation).", "",
        "## Settings", "", "```json", json.dumps(report["settings"], indent=2), "```", "",
        "## Interpretation", "",
        report["outcome_definition"], "",
        f"There are {len(report['paired_comparisons'])} within-axis paired comparisons and "
        f"{len(report['arm_comparisons'])} baseline-arm comparisons. Effects and descriptive "
        "intervals are in the separate axis summaries. Training uncertainty is nested: sampler "
        "seeds are averaged within checkpoints before comparing independent training seeds.", "",
        "Interaction results are a separate appendix in `interaction-summary.json`, not "
        "isolated one-axis effects. Held-out raw-data quality is a population proxy, not "
        "an exact-target identity. Velocity moments, component responsibility masses, "
        "and particle refinements are recorded per case.", "",
        report["study_acceptance_reason"], "",
        "## Expansion budget", "", "```json",
        json.dumps(report["full_profile_budget"], indent=2), "```", "",
        "## Unmet checks", "",
    ]
    for row in report["cases"]:
        if row["status"] == "violated":
            failed = []
            for mode, refinement in row["refinement"].items():
                if refinement["expected_exact_target"] and not refinement["within_target_bands"]:
                    failed.append(f"{mode}: exact-target bands")
                if refinement["expected_exact_target"] and not refinement["no_degradation"]:
                    failed.append(f"{mode}: step refinement")
                if refinement.get("gaussian_reference_passed") is False:
                    failed.append(f"{mode}: independent Gaussian moments")
            lines.append(f"- `{row['case']['id']}`: {', '.join(failed) or 'implementation checks'}; "
                         f"see `../cases/{row['case']['id']}.json`.")
        elif row["status"] == "not assessed":
            lines.append(f"- `{row['case']['id']}`: {row.get('error') or 'not executed'}.")
    lines += ["", "## Limitations", ""]
    lines.extend("- " + limitation for limitation in report["limitations"])
    lines += ["", "See `assumption-ledger.json` for case-level identity, model-choice, "
              "and empirical-requirement outcomes. No finite grid proves universal robustness.", ""]
    (output / "report.md").write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path,
                        default=Path("results/coordinate_invariance/generalization"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    try:
        report, records = analyze(args.input)
        output = args.output or args.input / "analysis"
        write_json(output / "analysis.json", report)
        write_reports(report, output)
        if not args.no_plots:
            plots(records, output / "figures")
        for axis, counts in report["coverage"].items():
            print(f"{axis}: " + ", ".join(f"{value} {key}" for key, value in counts.items()))
        print(f"Analysis: {output / 'analysis.json'}")
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"Generalization analysis: FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
