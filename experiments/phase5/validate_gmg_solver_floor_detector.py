from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _case_key(row: dict) -> tuple[int, float]:
    return int(row["seed"]), float(row["solid_probability"])


def _add_summary(rows_by_case: dict[tuple[int, float], dict], path: Path, *, source: str) -> None:
    for row in _read_rows(path):
        if row.get("case_kind") and row["case_kind"] != "baseline":
            continue
        key = _case_key(row)
        if key in rows_by_case:
            continue
        out = dict(row)
        out["baseline_source"] = source
        rows_by_case[key] = out


def _add_history(history_by_case: dict[tuple[int, float], dict[int, float]], path: Path) -> None:
    for row in _read_rows(path):
        if row.get("case_kind") and row["case_kind"] != "baseline":
            continue
        key = _case_key(row)
        history_by_case.setdefault(key, {})[int(row["iter"])] = float(row["rel_residual"])


def _add_rescue(rescue_by_case: dict[tuple[int, float], dict], path: Path, *, source: str) -> None:
    for row in _read_rows(path):
        if abs(float(row["rho_min"]) - 1e-3) > 1e-15:
            continue
        key = _case_key(row)
        out = dict(row)
        out["rescue_source"] = source
        rescue_by_case[key] = out


def _history_value(history: dict[int, float], iteration: int) -> float:
    if iteration in history:
        return history[iteration]
    available = sorted(i for i in history if i <= iteration)
    if not available:
        return float("nan")
    return history[available[-1]]


def _classify(r50: float, r100: float, *, high_residual: float, plateau_residual: float, plateau_ratio: float) -> tuple[bool, str]:
    if r50 >= high_residual:
        return True, "high_r50"
    ratio = r100 / max(r50, 1e-300)
    if r100 >= plateau_residual and ratio >= plateau_ratio:
        return True, "plateau"
    return False, "keep"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--high-residual-threshold", type=float, default=1e-2)
    parser.add_argument("--plateau-residual-threshold", type=float, default=1e-4)
    parser.add_argument("--plateau-ratio-threshold", type=float, default=0.6)
    parser.add_argument(
        "--out-dir",
        default="experiments/phase5/results/gmg_solver_floor_detector",
    )
    args = parser.parse_args()

    baseline_summaries: dict[tuple[int, float], dict] = {}
    baseline_histories: dict[tuple[int, float], dict[int, float]] = {}
    rescue_rows: dict[tuple[int, float], dict] = {}

    _add_summary(
        baseline_summaries,
        Path("experiments/phase5/results/percolation_atlas_seeded_300/atlas_summary.csv"),
        source="percolation_seeded_300",
    )
    _add_summary(
        baseline_summaries,
        Path("experiments/phase5/results/gmg_baseline_probe_seeds7_13_p012/atlas_summary.csv"),
        source="baseline_p012_seeds7_13",
    )
    _add_summary(
        baseline_summaries,
        Path("experiments/phase5/results/gmg_floor_rule_check_seed19/gmg_floor_rule_summary.csv"),
        source="floor_rule_seed19",
    )
    _add_history(
        baseline_histories,
        Path("experiments/phase5/results/percolation_atlas_seeded_300/atlas_history.csv"),
    )
    _add_history(
        baseline_histories,
        Path("experiments/phase5/results/gmg_baseline_probe_seeds7_13_p012/atlas_history.csv"),
    )
    _add_history(
        baseline_histories,
        Path("experiments/phase5/results/gmg_floor_rule_check_seed19/gmg_floor_rule_history.csv"),
    )

    for path, source in [
        (Path("experiments/phase5/results/gmg_high_floor_probe_seed19_rho1e3/atlas_summary.csv"), "seed19_lowp"),
        (Path("experiments/phase5/results/gmg_high_floor_probe_seeds7_13_rho1e3/atlas_summary.csv"), "seeds7_13_lowp"),
        (Path("experiments/phase5/results/gmg_high_floor_probe_seed13_p018_rho1e3/atlas_summary.csv"), "seed13_p018"),
        (Path("experiments/phase5/results/gmg_high_floor_probe_seed13_p020_rho1e3/atlas_summary.csv"), "seed13_p020"),
    ]:
        _add_rescue(rescue_rows, path, source=source)

    prediction_rows = []
    counts = {
        "true_raise": 0,
        "true_keep": 0,
        "false_raise": 0,
        "false_keep": 0,
    }
    rescue_success = 0
    rescue_available = 0
    for key, summary in sorted(baseline_summaries.items(), key=lambda item: (item[0][1], item[0][0])):
        if key not in baseline_histories:
            continue
        seed, probability = key
        history = baseline_histories[key]
        r50 = _history_value(history, 50)
        r100 = _history_value(history, 100)
        predicted_raise, trigger = _classify(
            r50,
            r100,
            high_residual=args.high_residual_threshold,
            plateau_residual=args.plateau_residual_threshold,
            plateau_ratio=args.plateau_ratio_threshold,
        )
        baseline_converged = bool(int(summary["solve_converged"]))
        desired_raise = not baseline_converged
        if predicted_raise and desired_raise:
            bucket = "true_raise"
        elif not predicted_raise and not desired_raise:
            bucket = "true_keep"
        elif predicted_raise and not desired_raise:
            bucket = "false_raise"
        else:
            bucket = "false_keep"
        counts[bucket] += 1

        rescue = rescue_rows.get(key)
        rescue_converged = ""
        rescue_iters = ""
        rescue_final = ""
        rescue_source = ""
        if rescue is not None:
            rescue_available += int(predicted_raise)
            rescue_converged = int(rescue["solve_converged"])
            rescue_iters = rescue["solve_iters"]
            rescue_final = rescue["solve_final_rel_residual"]
            rescue_source = rescue["rescue_source"]
            rescue_success += int(predicted_raise and bool(int(rescue["solve_converged"])))

        prediction_rows.append(
            {
                "seed": seed,
                "solid_probability": probability,
                "solid_fraction": summary["solid_fraction"],
                "n_components": summary["n_components"],
                "largest_component": summary["largest_component"],
                "largest_fraction_of_solid": summary["largest_fraction_of_solid"],
                "support_to_load_patch_components": summary["support_to_load_patch_components"],
                "baseline_converged": int(baseline_converged),
                "baseline_iters": summary["solve_iters"],
                "baseline_final_rel_residual": summary["solve_final_rel_residual"],
                "r50": r50,
                "r100": r100,
                "r100_over_r50": r100 / max(r50, 1e-300),
                "predicted_raise_floor": int(predicted_raise),
                "recommended_rho_min": 1e-3 if predicted_raise else 1e-12,
                "trigger": trigger,
                "desired_raise_floor": int(desired_raise),
                "bucket": bucket,
                "rho_1e3_converged": rescue_converged,
                "rho_1e3_iters": rescue_iters,
                "rho_1e3_final_rel_residual": rescue_final,
                "rescue_source": rescue_source,
                "baseline_source": summary["baseline_source"],
            }
        )

    total = sum(counts.values())
    summary_rows = [
        {
            "high_residual_threshold": args.high_residual_threshold,
            "plateau_residual_threshold": args.plateau_residual_threshold,
            "plateau_ratio_threshold": args.plateau_ratio_threshold,
            **counts,
            "total": total,
            "false_raise_rate": counts["false_raise"] / max(total, 1),
            "false_keep_rate": counts["false_keep"] / max(total, 1),
            "predicted_raise_with_rescue_data": rescue_available,
            "predicted_raise_rescue_successes": rescue_success,
        }
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "gmg_solver_floor_detector_predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)

    summary_path = out_dir / "gmg_solver_floor_detector_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for row in prediction_rows:
        probability = float(row["solid_probability"])
        marker = "o" if row["bucket"] in {"true_raise", "true_keep"} else "X"
        color = "#dc2626" if bool(row["predicted_raise_floor"]) else "#2563eb"
        edge = "#0f172a" if bool(row["desired_raise_floor"]) else "#94a3b8"
        ax.scatter(
            float(row["r50"]),
            float(row["r100_over_r50"]),
            s=80 + 220 * probability,
            marker=marker,
            color=color,
            edgecolor=edge,
            linewidth=1.2,
            alpha=0.9,
        )
        ax.annotate(
            f"{int(row['seed'])}, p={probability:g}",
            (float(row["r50"]), float(row["r100_over_r50"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax.axvline(args.high_residual_threshold, color="#dc2626", linestyle="--", linewidth=1.1)
    ax.axhline(args.plateau_ratio_threshold, color="#b45309", linestyle="--", linewidth=1.1)
    ax.set_xscale("log")
    ax.set_xlabel("baseline residual after 50 FGMRES iterations")
    ax.set_ylabel("residual ratio r100 / r50")
    ax.set_title("100-iteration probe predicts solver-floor escalation")
    ax.text(
        0.02,
        0.04,
        "red: raise to rho_min=1e-3\nblue: keep original floor\nblack edge: baseline failed",
        transform=ax.transAxes,
        fontsize=8,
        va="bottom",
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
    )
    fig_path = out_dir / "fig_gmg_solver_floor_detector.png"
    fig.savefig(fig_path, dpi=240, bbox_inches="tight")

    print(
        f"true_raise={counts['true_raise']}, true_keep={counts['true_keep']}, "
        f"false_raise={counts['false_raise']}, false_keep={counts['false_keep']}, total={total}",
        flush=True,
    )
    print(
        f"Predicted-raise rescue data: {rescue_success}/{rescue_available} converged at rho_min=1e-3",
        flush=True,
    )
    print(f"Wrote {pred_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {fig_path}", flush=True)


if __name__ == "__main__":
    main()
