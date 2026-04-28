from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "phase5" / "results"
OUT_DIR = RESULTS / "review_experiment_summary"


HELDOUT_DIRS = [
    RESULTS / "heldout_gmg_detector_cantilever_s41_71_guarded_true_residual",
    RESULTS / "heldout_gmg_detector_bridge_s41_59_guarded_true_residual",
]

FULL_LABEL_DIRS = [
    *(RESULTS / f"heldout_full_true_labels_cantilever_s{seed}" for seed in [41, 43, 47, 53, 59, 61, 67, 71]),
    *(RESULTS / f"heldout_full_true_labels_bridge_s{seed}" for seed in [41, 43, 47, 53, 59]),
]

FINE_LADDER_DIR = RESULTS / "fine_ladder_bridge_seed23_41_strict_true_residual"

OPTIMIZED_DIRS = [
    RESULTS / "optimized_density_C64_MF_strict_true_residual",
    RESULTS / "optimized_density_C216_MF_strict_true_residual",
    RESULTS / "optimized_density_C512_MF_strict_true_residual",
    RESULTS / "optimized_density_B500_MF_strict_true_residual",
    RESULTS / "optimized_density_Brk500_MF_strict_true_residual",
    RESULTS / "optimized_density_M500_MF_strict_true_residual",
    RESULTS / "optimized_density_T500_MF_strict_true_residual",
    RESULTS / "optimized_density_Col500_MF_strict_true_residual",
]

FIXED_FLOOR_DIR = RESULTS / "gmg_fixed_floor_controls_strict_true_residual"
FIXED_FLOOR_BASELINE_DIR = RESULTS / "policy_fixed_floor_baselines"
SEVERITY_JUMP_BASELINE_DIR = RESULTS / "policy_severity_jump_baselines"
SENSITIVITY_PERTURBATION_DIR = RESULTS / "heldout_true_keep_sensitivity_perturbation"
SIMP_EXPONENT_SENSITIVITY_DIR = RESULTS / "simp_exponent_sensitivity"
ORIGINAL_FLOOR_SENSITIVITY_DIR = RESULTS / "original_floor_sensitivity"
SIMP_EXPONENT_POLICY_SENSITIVITY_DIR = RESULTS / "simp_exponent_policy_sensitivity"
ORIGINAL_FLOOR_POLICY_SENSITIVITY_DIR = RESULTS / "original_floor_policy_sensitivity"
MECHANISM_ABLATION_DIR = RESULTS / "mechanism_ablation"
SIMP_FLOOR_TRAJECTORY_DIR = RESULTS / "simp_floor_trajectories"
GUARDED_ADAPTIVE_TRAJECTORY_DIR = RESULTS / "guarded_adaptive_trajectories"

PERTURBATION_DIRS = [
    RESULTS / "reduced_floor_perturbation_bridge_3d_s23_41",
    RESULTS / "reduced_floor_perturbation_cantilever_3d_large_s19_41",
]

OPTIMIZED_LABELS = {
    "C64_MF": ("Medium cantilever", "cantilever", "left face fixed; downward tip point load"),
    "C216_MF": ("216k cantilever", "cantilever", "left face fixed; downward tip point load"),
    "C512_MF": ("Large cantilever", "cantilever", "left face fixed; downward tip point load"),
    "B500_MF": ("Large bridge", "bridge", "pinned/roller lower supports; distributed top load"),
    "Brk500_MF": ("Large bracket", "bracket", "top face fixed; oblique point load"),
    "M500_MF": ("Large MBB beam", "MBB", "left edge pin-x and right lower pin-y; top-left point load"),
    "T500_MF": ("Large torsion", "torsion", "left face fixed; opposed end point loads"),
    "Col500_MF": ("Large column", "column", "bottom fixed; distributed top load"),
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def _format_floor(values: pd.Series) -> str:
    floors = sorted({float(value) for value in values})
    return "/".join(f"{floor:.0e}" for floor in floors)


def _with_case_key(df: pd.DataFrame) -> pd.DataFrame:
    keyed = df.copy()
    q_key = (keyed["solid_probability"].astype(float) * 1_000_000).round().astype(int)
    keyed["case_key"] = (
        keyed["preset"].astype(str)
        + "|"
        + keyed["seed"].astype(int).astype(str)
        + "|"
        + q_key.astype(str)
    )
    return keyed


def _load_prospective_dirs(directories: list[Path]) -> pd.DataFrame:
    frames = []
    for directory in directories:
        df = _read_csv(directory / "prospective_summary.csv")
        df["source_dir"] = directory.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _load_fixed_floor_baselines() -> pd.DataFrame:
    frames = []
    for summary_path in sorted(FIXED_FLOOR_BASELINE_DIR.glob("*/fixed_floor_control_summary.csv")):
        df = _read_csv(summary_path)
        df["source_dir"] = summary_path.parent.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No fixed-floor baseline summaries found under {FIXED_FLOOR_BASELINE_DIR}"
        )
    combined = pd.concat(frames, ignore_index=True)
    combined = _with_case_key(combined)
    if combined[["case_key", "rho_min"]].drop_duplicates().shape[0] != len(combined):
        raise ValueError("Fixed-floor baseline cases are not unique by case_key/rho_min")
    return combined


def _load_optional_prospective_root(root: Path) -> pd.DataFrame | None:
    frames = []
    for summary_path in sorted(root.rglob("prospective_summary.csv")):
        df = _read_csv(summary_path)
        df["source_dir"] = summary_path.parent.relative_to(root).as_posix()
        frames.append(df)
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    return _with_case_key(combined)


def _load_optional_sensitivity_perturbation() -> pd.DataFrame | None:
    frames = []
    for summary_path in sorted(SENSITIVITY_PERTURBATION_DIR.glob("*/sensitivity_perturbation_summary.csv")):
        df = _read_csv(summary_path)
        df["source_dir"] = summary_path.parent.name
        frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def _floor_counts(values: pd.Series) -> str:
    counts = values.map(float).value_counts().sort_index()
    return "; ".join(f"{floor:.0e}:{int(count)}" for floor, count in counts.items())


def _policy_row(
    *,
    policy: str,
    times: pd.Series,
    converged: pd.Series,
    iters: pd.Series,
    floors: pd.Series | None,
    notes: str,
) -> dict:
    failures = int((converged.astype(int) == 0).sum())
    floor_counts = ""
    if floors is not None:
        floor_counts = "; ".join(
            f"{float(floor):.0e}:{count}"
            for floor, count in sorted(floors.map(float).value_counts().to_dict().items())
        )
    return {
        "policy": policy,
        "cases": int(len(times)),
        "passes": int(len(times) - failures),
        "failures": failures,
        "convergence_rate": _safe_ratio(int(len(times) - failures), int(len(times))),
        "total_time_s": float(times.sum()),
        "mean_time_s": float(times.mean()),
        "median_time_s": float(times.median()),
        "max_time_s": float(times.max()),
        "mean_final_selected_iters": float(iters.mean()),
        "median_final_selected_iters": float(iters.median()),
        "max_final_selected_iters": int(iters.max()),
        "selected_floor_counts": floor_counts,
        "notes": notes,
    }


def _summarize_policy_baselines(heldout: pd.DataFrame) -> dict:
    full_joined = pd.read_csv(OUT_DIR / "heldout_full_true_labels_joined.csv")
    fixed_baselines = _load_fixed_floor_baselines()
    expected_keys = set(_with_case_key(heldout)["case_key"])
    fixed_keys = set(fixed_baselines["case_key"])
    if fixed_keys != expected_keys:
        missing = sorted(expected_keys - fixed_keys)
        extra = sorted(fixed_keys - expected_keys)
        raise ValueError(
            "Fixed-floor baseline cases do not match held-out set: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    fixed_baselines["total_time_s"] = (
        fixed_baselines["setup_time_s"].astype(float)
        + fixed_baselines["solve_time_s"].astype(float)
    )
    fixed_baselines.to_csv(
        OUT_DIR / "policy_fixed_floor_baseline_combined.csv", index=False
    )

    by_floor = (
        fixed_baselines.groupby("rho_min", dropna=False)
        .agg(
            rows=("rho_min", "size"),
            passes=("solve_converged", "sum"),
            mean_total_time_s=("total_time_s", "mean"),
            median_total_time_s=("total_time_s", "median"),
            max_total_time_s=("total_time_s", "max"),
            total_time_s=("total_time_s", "sum"),
            mean_solve_iters=("solve_iters", "mean"),
            median_solve_iters=("solve_iters", "median"),
            max_solve_iters=("solve_iters", "max"),
            max_final_rel_residual=("solve_final_rel_residual", "max"),
        )
        .reset_index()
        .sort_values("rho_min")
    )
    by_floor["failures"] = by_floor["rows"] - by_floor["passes"]
    by_floor["convergence_rate"] = by_floor["passes"] / by_floor["rows"]
    by_floor.to_csv(OUT_DIR / "policy_fixed_floor_baseline_summary.csv", index=False)

    guarded = heldout.copy()
    comparison_rows = [
        _policy_row(
            policy="guarded_probe_policy",
            times=guarded["recorded_policy_time_s"].astype(float),
            converged=guarded["solve_converged"],
            iters=guarded["solve_iters"].astype(float),
            floors=guarded["recommended_rho_min"],
            notes="Probe policy with recomputed true-residual guard and fallback.",
        ),
        _policy_row(
            policy="original_floor_full_then_fallback",
            times=full_joined["recorded_policy_time_s_full_label"].astype(float),
            converged=full_joined["solve_converged_full_label"],
            iters=full_joined["solve_iters"].astype(float),
            floors=full_joined["recommended_rho_min_full_label"],
            notes="Full 300-iteration original-floor attempt before fallback; used for true labels.",
        ),
    ]
    for floor, group in fixed_baselines.groupby("rho_min", sort=True):
        comparison_rows.append(
            _policy_row(
                policy=f"always_{float(floor):.0e}",
                times=group["total_time_s"].astype(float),
                converged=group["solve_converged"],
                iters=group["solve_iters"].astype(float),
                floors=group["rho_min"],
                notes=(
                    "Fixed-floor solve across all 102 held-out cases; time is "
                    "setup_time_s + solve_time_s from fixed-floor control runs."
                ),
            )
        )

    severity = _load_optional_prospective_root(SEVERITY_JUMP_BASELINE_DIR)
    severity_note = ""
    if severity is not None:
        severity_keys = set(severity["case_key"])
        missing = sorted(expected_keys - severity_keys)
        extra = sorted(severity_keys - expected_keys)
        if extra:
            raise ValueError(
                "Severity-jump baseline contains cases outside held-out set: "
                f"extra={extra[:5]}"
            )
        severity.to_csv(OUT_DIR / "policy_severity_jump_baseline_combined.csv", index=False)
        comparison_rows.append(
            _policy_row(
                policy="probe_severity_jump_1e-02",
                times=severity["recorded_policy_time_s"].astype(float),
                converged=severity["solve_converged"],
                iters=severity["solve_iters"].astype(float),
                floors=severity["recommended_rho_min"],
                notes=(
                    "Guarded probe policy with r50>=0.5 severity rule that skips "
                    "1e-3 and tries 1e-2 first on severe predicted-raise cases."
                ),
            )
        )
        by_jump = (
            severity.groupby("severity_jump_triggered", dropna=False)
            .agg(
                rows=("severity_jump_triggered", "size"),
                passes=("solve_converged", "sum"),
                mean_time_s=("recorded_policy_time_s", "mean"),
                median_time_s=("recorded_policy_time_s", "median"),
                max_time_s=("recorded_policy_time_s", "max"),
                mean_iters=("solve_iters", "mean"),
                max_iters=("solve_iters", "max"),
            )
            .reset_index()
        )
        by_jump.to_csv(OUT_DIR / "policy_severity_jump_baseline_summary.csv", index=False)
        severity_note = (
            f"{len(severity)}/{len(expected_keys)} cases complete; "
            f"severity jumps={int(severity['severity_jump_triggered'].sum())}; "
            f"missing={len(missing)}."
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUT_DIR / "policy_baseline_comparison.csv", index=False)
    return {
        "rows": int(len(fixed_baselines)),
        "passes": int(fixed_baselines["solve_converged"].sum()),
        "failures": int((fixed_baselines["solve_converged"].astype(int) == 0).sum()),
        "floor_summary": "; ".join(
            f"{float(row.rho_min):.0e}:mean {row.mean_total_time_s:.2f}s, "
            f"max {row.max_total_time_s:.2f}s, iters mean {row.mean_solve_iters:.1f}"
            for row in by_floor.itertuples(index=False)
        ),
        "severity_rows": int(len(severity)) if severity is not None else 0,
        "severity_passes": int(severity["solve_converged"].sum()) if severity is not None else 0,
        "severity_failures": (
            int((severity["solve_converged"].astype(int) == 0).sum())
            if severity is not None
            else 0
        ),
        "severity_note": severity_note,
    }


def _full_label_confusion(heldout: pd.DataFrame) -> dict:
    full_labels = _load_prospective_dirs(FULL_LABEL_DIRS)
    heldout_keyed = _with_case_key(heldout)
    full_keyed = _with_case_key(full_labels)
    if heldout_keyed["case_key"].nunique() != len(heldout_keyed):
        raise ValueError("Guarded held-out cases are not unique by preset/seed/solid_probability")
    if full_keyed["case_key"].nunique() != len(full_keyed):
        raise ValueError("Full-label held-out cases are not unique by preset/seed/solid_probability")

    guarded_keys = set(heldout_keyed["case_key"])
    full_keys = set(full_keyed["case_key"])
    if guarded_keys != full_keys:
        missing_full = sorted(guarded_keys - full_keys)
        extra_full = sorted(full_keys - guarded_keys)
        raise ValueError(
            "Full-label cases do not match guarded held-out set: "
            f"missing_full={missing_full[:5]} extra_full={extra_full[:5]}"
        )

    guarded_cols = [
        "case_key",
        "preset",
        "seed",
        "solid_probability",
        "trigger",
        "predicted_raise_floor",
        "recommended_rho_min",
        "attempted_rho_mins",
        "solve_converged",
        "detector_false_keep",
        "recorded_policy_time_s",
    ]
    full_cols = [
        "case_key",
        "post_solve_fallback_triggered",
        "recommended_rho_min",
        "attempted_rho_mins",
        "solve_converged",
        "solver_reported_converged",
        "solve_iters",
        "solve_final_rel_residual",
        "recorded_policy_time_s",
    ]
    joined = heldout_keyed[guarded_cols].merge(
        full_keyed[full_cols],
        on="case_key",
        how="inner",
        suffixes=("_guarded", "_full_label"),
    )
    joined["detector_predicted_raise"] = joined["predicted_raise_floor"].astype(int) == 1
    joined["true_raise_floor"] = joined["post_solve_fallback_triggered"].astype(int) == 1
    joined["true_keep_original_floor"] = ~joined["true_raise_floor"]
    joined["confusion_cell"] = "TN"
    joined.loc[joined["detector_predicted_raise"] & joined["true_raise_floor"], "confusion_cell"] = "TP"
    joined.loc[joined["detector_predicted_raise"] & ~joined["true_raise_floor"], "confusion_cell"] = "FP"
    joined.loc[~joined["detector_predicted_raise"] & joined["true_raise_floor"], "confusion_cell"] = "FN"
    joined.to_csv(OUT_DIR / "heldout_full_true_labels_joined.csv", index=False)

    tp = int((joined["confusion_cell"] == "TP").sum())
    tn = int((joined["confusion_cell"] == "TN").sum())
    fp = int((joined["confusion_cell"] == "FP").sum())
    fn = int((joined["confusion_cell"] == "FN").sum())
    true_raises = int(joined["true_raise_floor"].sum())
    true_keeps = int(joined["true_keep_original_floor"].sum())
    pred_raises = int(joined["detector_predicted_raise"].sum())
    pred_keeps = int((~joined["detector_predicted_raise"]).sum())
    full_label_failures = int((joined["solve_converged_full_label"].astype(int) == 0).sum())

    confusion = {
        "rows": len(joined),
        "passes": int((joined["solve_converged_full_label"].astype(int) == 1).sum()),
        "failures": full_label_failures,
        "true_raises": true_raises,
        "true_keeps": true_keeps,
        "predicted_raises": pred_raises,
        "predicted_keeps": pred_keeps,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "sensitivity": _safe_ratio(tp, tp + fn),
        "specificity": _safe_ratio(tn, tn + fp),
        "false_keep_rate_all": _safe_ratio(fn, len(joined)),
        "false_keep_rate_predicted_keep": _safe_ratio(fn, pred_keeps),
        "false_raise_rate_all": _safe_ratio(fp, len(joined)),
        "false_raise_rate_predicted_raise": _safe_ratio(fp, pred_raises),
    }
    pd.DataFrame([confusion]).to_csv(
        OUT_DIR / "heldout_full_true_label_confusion.csv", index=False
    )
    joined[joined["confusion_cell"] == "FN"].to_csv(
        OUT_DIR / "heldout_full_true_label_false_keeps.csv", index=False
    )
    joined[joined["confusion_cell"] == "FP"].to_csv(
        OUT_DIR / "heldout_full_true_label_false_raises.csv", index=False
    )
    return confusion


def _detector_only_negative_control() -> dict:
    joined = pd.read_csv(OUT_DIR / "heldout_full_true_labels_joined.csv")
    negative = joined.copy()
    negative["detector_only_converged"] = (negative["confusion_cell"] != "FN").astype(int)
    negative["detector_only_failure_reason"] = ""
    negative.loc[
        negative["confusion_cell"] == "FN",
        "detector_only_failure_reason",
    ] = "false keep: original-floor solve would be accepted without final true-residual guard"
    negative.to_csv(OUT_DIR / "detector_only_negative_control.csv", index=False)

    failures = int((negative["detector_only_converged"] == 0).sum())
    pred_keeps = int((negative["detector_predicted_raise"].astype(str).str.lower() != "true").sum())
    return {
        "rows": int(len(negative)),
        "passes": int(len(negative) - failures),
        "failures": failures,
        "predicted_raises": int((negative["detector_predicted_raise"].astype(str).str.lower() == "true").sum()),
        "predicted_keeps": pred_keeps,
        "false_keep_rate_all": _safe_ratio(failures, len(negative)),
        "false_keep_rate_predicted_keep": _safe_ratio(failures, pred_keeps),
    }


def _summarize_sensitivity_perturbation() -> dict | None:
    sensitivity = _load_optional_sensitivity_perturbation()
    if sensitivity is None:
        return None
    sensitivity.to_csv(OUT_DIR / "sensitivity_perturbation_combined.csv", index=False)
    raised = sensitivity[
        sensitivity["rho_min"].astype(float)
        != sensitivity["reference_rho_min"].astype(float)
    ].copy()
    if raised.empty:
        return {
            "rows": int(len(sensitivity)),
            "passes": int(sensitivity["solve_converged"].sum()),
            "failures": int((sensitivity["solve_converged"].astype(int) == 0).sum()),
            "notes": "Only reference-floor sensitivity rows are present.",
        }
    by_floor = (
        raised.groupby("rho_min", dropna=False)
        .agg(
            rows=("rho_min", "size"),
            passes=("solve_converged", "sum"),
            mean_rel_compliance_change=("rel_compliance_change", "mean"),
            mean_abs_rel_compliance_change=("rel_compliance_change", lambda s: s.abs().mean()),
            max_abs_rel_compliance_change=("rel_compliance_change", lambda s: s.abs().max()),
            mean_rel_dc_l2_solid=("rel_dc_l2_solid", "mean"),
            median_rel_dc_l2_solid=("rel_dc_l2_solid", "median"),
            max_rel_dc_l2_solid=("rel_dc_l2_solid", "max"),
            mean_rel_dc_linf_solid=("rel_dc_linf_solid", "mean"),
            max_rel_dc_linf_solid=("rel_dc_linf_solid", "max"),
            min_pearson_dc_solid=("pearson_dc_solid", "min"),
        )
        .reset_index()
        .sort_values("rho_min")
    )
    by_floor["failures"] = by_floor["rows"] - by_floor["passes"]
    by_floor.to_csv(OUT_DIR / "sensitivity_perturbation_summary.csv", index=False)
    return {
        "rows": int(len(sensitivity)),
        "passes": int(sensitivity["solve_converged"].sum()),
        "failures": int((sensitivity["solve_converged"].astype(int) == 0).sum()),
        "notes": "; ".join(
            f"{float(row.rho_min):.0e}:mean |dC/C| {row.mean_abs_rel_compliance_change:.3g}, "
            f"mean solid ||ddc||/||dc|| {row.mean_rel_dc_l2_solid:.3g}, "
            f"max solid {row.max_rel_dc_l2_solid:.3g}"
            for row in by_floor.itertuples(index=False)
        ),
    }


def _summarize_optional_prospective_sweep(
    *,
    root: Path,
    block: str,
    group_cols: list[str],
    out_prefix: str,
) -> dict | None:
    df = _load_optional_prospective_root(root)
    if df is None:
        return None
    df.to_csv(OUT_DIR / f"{out_prefix}_combined.csv", index=False)
    grouped = df.groupby(group_cols, dropna=False)
    summary = (
        grouped.agg(
            rows=("case_key", "size"),
            unique_cases=("case_key", "nunique"),
            passes=("solve_converged", "sum"),
            predicted_raises=("predicted_raise_floor", "sum"),
            detector_false_keeps=("detector_false_keep", "sum"),
            mean_time_s=("recorded_policy_time_s", "mean"),
            median_time_s=("recorded_policy_time_s", "median"),
            max_time_s=("recorded_policy_time_s", "max"),
            mean_iters=("solve_iters", "mean"),
            max_iters=("solve_iters", "max"),
        )
        .reset_index()
    )
    floors = grouped["recommended_rho_min"].apply(_floor_counts).reset_index(name="recommended_floor_counts")
    summary = summary.merge(floors, on=group_cols, how="left")
    summary["failures"] = summary["rows"] - summary["passes"]
    summary.to_csv(OUT_DIR / f"{out_prefix}_summary.csv", index=False)
    note_parts = []
    for row in summary.itertuples(index=False):
        label = ", ".join(f"{col}={getattr(row, col)}" for col in group_cols)
        note_parts.append(
            f"{label}: {int(row.passes)}/{int(row.rows)} pass, "
            f"raises={int(row.predicted_raises)}, false-keeps={int(row.detector_false_keeps)}, "
            f"floors=[{row.recommended_floor_counts}]"
        )
    return {
        "block": block,
        "rows": int(len(df)),
        "passes": int(df["solve_converged"].sum()),
        "failures": int((df["solve_converged"].astype(int) == 0).sum()),
        "predicted_raises": int(df["predicted_raise_floor"].sum()),
        "predicted_keeps": int((df["predicted_raise_floor"].astype(int) == 0).sum()),
        "false_keeps": int(df["detector_false_keep"].sum()),
        "notes": "; ".join(note_parts),
    }


def _summarize_optional_simp_floor_trajectories() -> dict | None:
    frames = []
    for summary_path in sorted(SIMP_FLOOR_TRAJECTORY_DIR.rglob("trajectory_summary.csv")):
        df = _read_csv(summary_path)
        df["source_dir"] = summary_path.parent.relative_to(SIMP_FLOOR_TRAJECTORY_DIR).as_posix()
        frames.append(df)
    if not frames:
        return None
    trajectories = pd.concat(frames, ignore_index=True)
    trajectories.to_csv(OUT_DIR / "simp_floor_trajectory_combined.csv", index=False)
    by_floor = (
        trajectories.groupby(["preset", "rho_min"], dropna=False)
        .agg(
            rows=("run_id", "size"),
            completed=("error", lambda s: int((s.fillna("") == "").sum())),
            mean_wall_time_s=("wall_time_s", "mean"),
            final_compliance=("final_compliance", "mean"),
            best_compliance=("best_compliance", "mean"),
            final_grayness=("final_grayness", "mean"),
            final_solver_iters=("final_solver_iters", "mean"),
        )
        .reset_index()
    )
    by_floor["failures"] = by_floor["rows"] - by_floor["completed"]
    by_floor.to_csv(OUT_DIR / "simp_floor_trajectory_summary.csv", index=False)
    notes = "; ".join(
        f"{row.preset}, rho={float(row.rho_min):.0e}: "
        f"completed={int(row.completed)}/{int(row.rows)}, "
        f"final C={row.final_compliance:.6g}, best C={row.best_compliance:.6g}, "
        f"gray={row.final_grayness:.3g}"
        for row in by_floor.itertuples(index=False)
    )
    return {
        "rows": int(len(trajectories)),
        "passes": int((trajectories["error"].fillna("") == "").sum()),
        "failures": int((trajectories["error"].fillna("") != "").sum()),
        "notes": notes,
    }


def _summarize_optional_guarded_adaptive_trajectories() -> dict | None:
    summary_path = GUARDED_ADAPTIVE_TRAJECTORY_DIR / "trajectory_summary.csv"
    iter_path = GUARDED_ADAPTIVE_TRAJECTORY_DIR / "trajectory_iters.csv"
    if not summary_path.exists() or not iter_path.exists():
        return None

    summary = _read_csv(summary_path)
    iters = _read_csv(iter_path)
    summary.to_csv(OUT_DIR / "guarded_adaptive_trajectory_summary.csv", index=False)
    iters.to_csv(OUT_DIR / "guarded_adaptive_trajectory_iters.csv", index=False)

    ok = summary["error"].fillna("") == ""
    notes = []
    for row in summary.itertuples(index=False):
        notes.append(
            f"{row.preset}: completed={int(row.iters_completed)}/{int(row.iters_requested)}, "
            f"floors=[{row.selected_floor_counts}], fallbacks={int(row.fallback_events)}, "
            f"max true residual={float(row.max_true_rel_residual):.3g}, "
            f"final C={float(row.final_compliance):.6g}, wall={float(row.wall_time_s):.1f}s"
        )
    return {
        "rows": int(len(summary)),
        "passes": int(ok.sum()),
        "failures": int((~ok).sum()),
        "notes": "; ".join(notes),
    }


def _optimized_case_details(optimized: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for density_name, group in optimized.groupby("density_name", sort=False):
        run_dir = ROOT / "experiments" / "paper2" / "runs" / density_name
        meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
        iters = pd.read_csv(run_dir / "iters.csv")
        final_iter = iters.iloc[-1]
        label, geometry, boundary_load = OPTIMIZED_LABELS.get(
            density_name, (density_name, meta.get("preset", ""), "")
        )
        states = sorted(
            {
                "best" if "rho_best" in str(path) else "final"
                for path in group["density_path"]
            }
        )
        rows.append(
            {
                "artifact_id": density_name,
                "reader_label": label,
                "geometry": geometry,
                "preset": meta["preset"],
                "elements": int(meta["n_elem"]),
                "volume_fraction": float(group["density_mean"].iloc[0]),
                "states": "/".join(states),
                "source_optimizer": "OC update with density filter and Heaviside projection",
                "source_iterations": int(meta["n_iter"]),
                "source_compliance": float(meta["compliance"]),
                "source_grayness": float(meta["grayness"]),
                "final_penal": float(final_iter["penal"]),
                "final_filter_radius": float(final_iter["rmin"]),
                "final_projection_beta": float(final_iter["beta"]),
                "final_move_limit": float(final_iter["move"]),
                "boundary_load_summary": boundary_load,
                "selected_floor": _format_floor(group["recommended_rho_min"]),
                "selected_iters": (
                    str(int(group["solve_iters"].iloc[0]))
                    if group["solve_iters"].nunique() == 1
                    else f"{int(group['solve_iters'].min())}-{int(group['solve_iters'].max())}"
                ),
                "policy_decision": "raise" if (group["predicted_raise_floor"].astype(int) == 1).any() else "keep",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    heldout = _load_prospective_dirs(HELDOUT_DIRS)
    pred_keep = heldout["predicted_raise_floor"].astype(int) == 0
    true_fail = heldout["solve_converged"].astype(int) == 0
    detector_false_keep = (
        heldout["detector_false_keep"].astype(int) == 1
        if "detector_false_keep" in heldout.columns
        else pred_keep & true_fail
    )
    rows.append(
        {
            "block": "heldout_gmg_detector",
            "rows": len(heldout),
            "passes": int((~true_fail).sum()),
            "failures": int(true_fail.sum()),
            "predicted_raises": int(heldout["predicted_raise_floor"].sum()),
            "predicted_keeps": int(pred_keep.sum()),
            "false_keeps": int(detector_false_keep.sum()),
            "false_keep_rate_all": _safe_ratio(int(detector_false_keep.sum()), len(heldout)),
            "false_keep_rate_predicted_keep": _safe_ratio(int(detector_false_keep.sum()), int(pred_keep.sum())),
            "notes": "Guarded policy uses recomputed true residual; false_keeps count detector keeps recovered by fallback.",
        }
    )
    heldout[detector_false_keep].to_csv(OUT_DIR / "heldout_false_keeps.csv", index=False)

    full_confusion = _full_label_confusion(heldout)
    rows.append(
        {
            "block": "heldout_full_true_labels",
            "rows": full_confusion["rows"],
            "passes": full_confusion["passes"],
            "failures": full_confusion["failures"],
            "predicted_raises": full_confusion["predicted_raises"],
            "predicted_keeps": full_confusion["predicted_keeps"],
            "false_keeps": full_confusion["false_negative"],
            "false_keep_rate_all": full_confusion["false_keep_rate_all"],
            "false_keep_rate_predicted_keep": full_confusion["false_keep_rate_predicted_keep"],
            "notes": (
                f"True labels: {full_confusion['true_raises']} raises, "
                f"{full_confusion['true_keeps']} keeps; "
                f"TP/TN/FP/FN={full_confusion['true_positive']}/"
                f"{full_confusion['true_negative']}/{full_confusion['false_positive']}/"
                f"{full_confusion['false_negative']}; "
                f"sensitivity={full_confusion['sensitivity']:.6f}; "
                f"specificity={full_confusion['specificity']:.6f}."
            ),
        }
    )
    detector_only = _detector_only_negative_control()
    rows.append(
        {
            "block": "detector_only_negative_control",
            "rows": detector_only["rows"],
            "passes": detector_only["passes"],
            "failures": detector_only["failures"],
            "predicted_raises": detector_only["predicted_raises"],
            "predicted_keeps": detector_only["predicted_keeps"],
            "false_keeps": detector_only["failures"],
            "false_keep_rate_all": detector_only["false_keep_rate_all"],
            "false_keep_rate_predicted_keep": detector_only["false_keep_rate_predicted_keep"],
            "notes": (
                "Derived no-guard control: the four detector false keeps would be "
                "reported as accepted failures without the final true-residual guard."
            ),
        }
    )

    fine = _read_csv(FINE_LADDER_DIR / "prospective_summary.csv")
    ladder_counts = fine["recommended_rho_min"].map(lambda x: f"{float(x):.0e}").value_counts().to_dict()
    rows.append(
        {
            "block": "fine_ladder_sweep",
            "rows": len(fine),
            "passes": int(fine["solve_converged"].sum()),
            "failures": int((fine["solve_converged"].astype(int) == 0).sum()),
            "predicted_raises": int(fine["predicted_raise_floor"].sum()),
            "predicted_keeps": int((fine["predicted_raise_floor"].astype(int) == 0).sum()),
            "false_keeps": 0,
            "false_keep_rate_all": 0.0,
            "false_keep_rate_predicted_keep": 0.0,
            "notes": "; ".join(f"{floor}:{count}" for floor, count in sorted(ladder_counts.items())),
        }
    )

    opt_frames = []
    for directory in OPTIMIZED_DIRS:
        df = _read_csv(directory / "density_detector_summary.csv")
        df["source_dir"] = directory.name
        opt_frames.append(df)
    optimized = pd.concat(opt_frames, ignore_index=True)
    optimized.to_csv(OUT_DIR / "optimized_density_strict_summary.csv", index=False)
    _optimized_case_details(optimized).to_csv(
        OUT_DIR / "optimized_density_case_details.csv", index=False
    )
    opt_fail = optimized["solve_converged"].astype(int) == 0
    opt_raise_counts = optimized["recommended_rho_min"].map(lambda x: f"{float(x):.0e}").value_counts().to_dict()
    rows.append(
        {
            "block": "optimized_density_expansion",
            "rows": len(optimized),
            "passes": int((~opt_fail).sum()),
            "failures": int(opt_fail.sum()),
            "predicted_raises": int(optimized["predicted_raise_floor"].sum()),
            "predicted_keeps": int((optimized["predicted_raise_floor"].astype(int) == 0).sum()),
            "false_keeps": int(((optimized["predicted_raise_floor"].astype(int) == 0) & opt_fail).sum()),
            "false_keep_rate_all": _safe_ratio(
                int(((optimized["predicted_raise_floor"].astype(int) == 0) & opt_fail).sum()), len(optimized)
            ),
            "false_keep_rate_predicted_keep": _safe_ratio(
                int(((optimized["predicted_raise_floor"].astype(int) == 0) & opt_fail).sum()),
                int((optimized["predicted_raise_floor"].astype(int) == 0).sum()),
            ),
            "notes": "; ".join(f"{floor}:{count}" for floor, count in sorted(opt_raise_counts.items())),
        }
    )

    fixed = _read_csv(FIXED_FLOOR_DIR / "fixed_floor_control_summary.csv")
    fixed.to_csv(OUT_DIR / "fixed_floor_controls_strict_summary.csv", index=False)
    rows.append(
        {
            "block": "policy_wall_time_memory",
            "rows": len(fixed),
            "passes": int(fixed["solve_converged"].sum()),
            "failures": int((fixed["solve_converged"].astype(int) == 0).sum()),
            "predicted_raises": "",
            "predicted_keeps": "",
            "false_keeps": "",
            "false_keep_rate_all": "",
            "false_keep_rate_predicted_keep": "",
            "notes": "Strict fixed-floor controls include setup_s, solve_s, gpu memory, and CuPy pool fields.",
        }
    )

    baseline_summary = _summarize_policy_baselines(heldout)
    rows.append(
        {
            "block": "policy_fixed_floor_baselines",
            "rows": baseline_summary["rows"],
            "passes": baseline_summary["passes"],
            "failures": baseline_summary["failures"],
            "predicted_raises": "",
            "predicted_keeps": "",
            "false_keeps": "",
            "false_keep_rate_all": "",
            "false_keep_rate_predicted_keep": "",
            "notes": baseline_summary["floor_summary"],
        }
    )
    if baseline_summary["severity_rows"]:
        rows.append(
            {
                "block": "policy_severity_jump_baseline",
                "rows": baseline_summary["severity_rows"],
                "passes": baseline_summary["severity_passes"],
                "failures": baseline_summary["severity_failures"],
                "predicted_raises": "",
                "predicted_keeps": "",
                "false_keeps": "",
                "false_keep_rate_all": "",
                "false_keep_rate_predicted_keep": "",
                "notes": baseline_summary["severity_note"],
            }
        )

    sensitivity_summary = _summarize_sensitivity_perturbation()
    if sensitivity_summary is not None:
        rows.append(
            {
                "block": "sensitivity_perturbation",
                "rows": sensitivity_summary["rows"],
                "passes": sensitivity_summary["passes"],
                "failures": sensitivity_summary["failures"],
                "predicted_raises": "",
                "predicted_keeps": "",
                "false_keeps": "",
                "false_keep_rate_all": "",
                "false_keep_rate_predicted_keep": "",
                "notes": sensitivity_summary["notes"],
            }
        )

    for optional_summary in [
        _summarize_optional_prospective_sweep(
            root=SIMP_EXPONENT_SENSITIVITY_DIR,
            block="simp_exponent_sensitivity",
            group_cols=["penal"],
            out_prefix="simp_exponent_sensitivity",
        ),
        _summarize_optional_prospective_sweep(
            root=ORIGINAL_FLOOR_SENSITIVITY_DIR,
            block="original_floor_sensitivity",
            group_cols=["baseline_rho_min"],
            out_prefix="original_floor_sensitivity",
        ),
        _summarize_optional_prospective_sweep(
            root=MECHANISM_ABLATION_DIR,
            block="mechanism_ablation",
            group_cols=["stack_variant"],
            out_prefix="mechanism_ablation",
        ),
        _summarize_optional_prospective_sweep(
            root=SIMP_EXPONENT_POLICY_SENSITIVITY_DIR,
            block="simp_exponent_policy_sensitivity",
            group_cols=["penal"],
            out_prefix="simp_exponent_policy_sensitivity",
        ),
        _summarize_optional_prospective_sweep(
            root=ORIGINAL_FLOOR_POLICY_SENSITIVITY_DIR,
            block="original_floor_policy_sensitivity",
            group_cols=["baseline_rho_min"],
            out_prefix="original_floor_policy_sensitivity",
        ),
    ]:
        if optional_summary is None:
            continue
        rows.append(
            {
                "block": optional_summary["block"],
                "rows": optional_summary["rows"],
                "passes": optional_summary["passes"],
                "failures": optional_summary["failures"],
                "predicted_raises": optional_summary["predicted_raises"],
                "predicted_keeps": optional_summary["predicted_keeps"],
                "false_keeps": optional_summary["false_keeps"],
                "false_keep_rate_all": _safe_ratio(
                    optional_summary["false_keeps"], optional_summary["rows"]
                ),
                "false_keep_rate_predicted_keep": _safe_ratio(
                    optional_summary["false_keeps"], optional_summary["predicted_keeps"]
                ),
                "notes": optional_summary["notes"],
            }
        )

    trajectory_summary = _summarize_optional_simp_floor_trajectories()
    if trajectory_summary is not None:
        rows.append(
            {
                "block": "simp_floor_trajectories",
                "rows": trajectory_summary["rows"],
                "passes": trajectory_summary["passes"],
                "failures": trajectory_summary["failures"],
                "predicted_raises": "",
                "predicted_keeps": "",
                "false_keeps": "",
                "false_keep_rate_all": "",
                "false_keep_rate_predicted_keep": "",
                "notes": trajectory_summary["notes"],
            }
        )

    guarded_trajectory_summary = _summarize_optional_guarded_adaptive_trajectories()
    if guarded_trajectory_summary is not None:
        rows.append(
            {
                "block": "guarded_adaptive_trajectories",
                "rows": guarded_trajectory_summary["rows"],
                "passes": guarded_trajectory_summary["passes"],
                "failures": guarded_trajectory_summary["failures"],
                "predicted_raises": "",
                "predicted_keeps": "",
                "false_keeps": "",
                "false_keep_rate_all": "",
                "false_keep_rate_predicted_keep": "",
                "notes": guarded_trajectory_summary["notes"],
            }
        )

    perturbation_frames = []
    for directory in PERTURBATION_DIRS:
        df = _read_csv(directory / "reduced_floor_perturbation_summary.csv")
        df["source_dir"] = directory.name
        perturbation_frames.append(df)
    perturbation = pd.concat(perturbation_frames, ignore_index=True)
    perturbation.to_csv(OUT_DIR / "reduced_floor_perturbation_combined.csv", index=False)
    reference_rows = perturbation[perturbation["rho_min"] == perturbation["reference_rho_min"]]
    clean_ref = reference_rows[reference_rows["reference_direct_rel_residual"] <= 1e-6]
    rows.append(
        {
            "block": "reference_solver_comparison",
            "rows": len(perturbation),
            "passes": len(clean_ref),
            "failures": int(len(reference_rows) - len(clean_ref)),
            "predicted_raises": "",
            "predicted_keeps": "",
            "false_keeps": "",
            "false_keep_rate_all": "",
            "false_keep_rate_predicted_keep": "",
            "notes": (
                f"{len(clean_ref)}/{len(reference_rows)} reference-floor cases have direct residual <= 1e-6; "
                "sparse hard references remain ill-conditioned."
            ),
        }
    )

    out_path = OUT_DIR / "review_experiment_summary.csv"
    fieldnames = [
        "block",
        "rows",
        "passes",
        "failures",
        "predicted_raises",
        "predicted_keeps",
        "false_keeps",
        "false_keep_rate_all",
        "false_keep_rate_predicted_keep",
        "notes",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
