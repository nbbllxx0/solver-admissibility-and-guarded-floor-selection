from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _max_iter(rows: list[dict]) -> int:
    if not rows:
        return 0
    return max(int(float(row["iter"])) for row in rows)


def _final_residual(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    last = max(rows, key=lambda row: int(float(row["iter"])))
    return float(last["rel_residual"])


def _optional_float(row: dict, key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value == "":
        return default
    return float(value)


def _optional_sum(row: dict, keys: list[str]) -> float:
    values = [_optional_float(row, key) for key in keys]
    if not any(np.isfinite(value) for value in values):
        return float("nan")
    return float(np.nansum(values))


def _nanmean_or_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _nanmax_or_nan(values: list[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmax(arr))


def _attempt_order(summary_row: dict, grouped_attempts: dict[float, list[dict]]) -> list[float]:
    text = summary_row.get("attempted_rho_mins", "")
    out = []
    for part in text.replace(",", ";").split(";"):
        part = part.strip()
        if part:
            out.append(float(part))
    if out:
        return out
    return sorted(grouped_attempts)


def _collect_case(
    *,
    category: str,
    case_id: str,
    summary_row: dict,
    history_rows: list[dict],
    selected_floor: float,
) -> dict:
    probe_rows = [row for row in history_rows if row["phase"] == "probe"]
    attempt_rows = [row for row in history_rows if row["phase"] == "recommended_solve"]
    attempts: dict[float, list[dict]] = defaultdict(list)
    for row in attempt_rows:
        attempts[float(row["rho_min"])].append(row)
    ordered_floors = _attempt_order(summary_row, attempts)
    attempt_iters = {floor: _max_iter(attempts.get(floor, [])) for floor in ordered_floors}
    final_floor = float(selected_floor)
    failed_ladder_iters = sum(iters for floor, iters in attempt_iters.items() if abs(floor - final_floor) > 1e-15)
    selected_solve_iters = int(float(summary_row["solve_iters"]))
    probe_iters = _max_iter(probe_rows)
    probe_final = _final_residual(probe_rows)
    trigger = summary_row.get("trigger", "")
    tol = float(summary_row.get("tol", 1e-6))
    probe_reusable = trigger == "keep" and np.isfinite(probe_final) and probe_final <= tol
    recorded_policy_iters = probe_iters + sum(attempt_iters.values())
    reuse_enabled_policy_iters = probe_iters if probe_reusable else recorded_policy_iters
    probe_time_s = _optional_sum(summary_row, ["probe_setup_time_s", "probe_solve_time_s"])
    selected_time_s = _optional_sum(summary_row, ["selected_setup_time_s", "selected_solve_time_s"])
    failed_ladder_time_s = _optional_sum(summary_row, ["failed_ladder_setup_time_s", "failed_ladder_solve_time_s"])
    recorded_policy_time_s = _optional_float(
        summary_row,
        "recorded_policy_time_s",
        float(np.nansum([probe_time_s, selected_time_s, failed_ladder_time_s]))
        if any(np.isfinite(v) for v in [probe_time_s, selected_time_s, failed_ladder_time_s])
        else float("nan"),
    )
    return {
        "category": category,
        "case_id": case_id,
        "trigger": trigger,
        "selected_floor": final_floor,
        "probe_iters": probe_iters,
        "probe_final_rel_residual": probe_final,
        "probe_reusable": int(probe_reusable),
        "n_floor_attempts": len(ordered_floors),
        "attempted_floors": ";".join(f"{floor:.0e}" for floor in ordered_floors),
        "failed_ladder_iters": failed_ladder_iters,
        "selected_solve_iters": selected_solve_iters,
        "recorded_policy_iters": recorded_policy_iters,
        "reuse_enabled_policy_iters": reuse_enabled_policy_iters,
        "recorded_overhead_iters": recorded_policy_iters - selected_solve_iters,
        "reuse_enabled_overhead_iters": reuse_enabled_policy_iters - selected_solve_iters,
        "recorded_policy_to_selected_ratio": recorded_policy_iters / max(selected_solve_iters, 1),
        "reuse_enabled_policy_to_selected_ratio": reuse_enabled_policy_iters / max(selected_solve_iters, 1),
        "probe_time_s": probe_time_s,
        "selected_solve_time_s": selected_time_s,
        "failed_ladder_time_s": failed_ladder_time_s,
        "recorded_policy_time_s": recorded_policy_time_s,
        "recorded_time_to_selected_ratio": (
            recorded_policy_time_s / selected_time_s
            if np.isfinite(recorded_policy_time_s) and np.isfinite(selected_time_s) and selected_time_s > 0.0
            else float("nan")
        ),
        "selected_cupy_pool_total_mb_after_solve": _optional_float(
            summary_row,
            "selected_cupy_pool_total_mb_after_solve",
        ),
        "solve_converged": int(float(summary_row["solve_converged"])),
        "solve_final_rel_residual": float(summary_row["solve_final_rel_residual"]),
    }


def _prospective_cases(root: Path) -> list[dict]:
    cases = []
    sources = [
        ("cantilever random prospective", root / "gmg_solver_floor_detector_prospective_seed23"),
        ("cantilever random prospective", root / "gmg_solver_floor_detector_prospective_seed31"),
        ("cantilever random prospective", root / "gmg_solver_floor_detector_prospective_highp_controls"),
        ("bridge random geometry transfer", root / "gmg_solver_floor_detector_transfer_bridge_seed23_ladder"),
    ]
    for category, directory in sources:
        summary = _read_rows(directory / "prospective_summary.csv")
        history = _read_rows(directory / "prospective_history.csv")
        for row in summary:
            seed = row["seed"]
            probability = float(row["solid_probability"])
            case_history = [
                h for h in history if h["seed"] == seed and abs(float(h["solid_probability"]) - probability) < 1e-12
            ]
            cases.append(
                _collect_case(
                    category=category,
                    case_id=f"seed {seed} p={probability:g}",
                    summary_row=row,
                    history_rows=case_history,
                    selected_floor=float(row["recommended_rho_min"]),
                )
            )
    return cases


def _density_cases(root: Path) -> list[dict]:
    cases = []
    for directory in sorted(root.glob("gmg_solver_floor_detector_density_*")):
        summary = _read_rows(directory / "density_detector_summary.csv")
        history = _read_rows(directory / "density_detector_history.csv")
        for row in summary:
            name = row["density_name"]
            case_history = [h for h in history if h["density_name"] == name]
            cases.append(
                _collect_case(
                    category="optimized SIMP density",
                    case_id=name,
                    summary_row=row,
                    history_rows=case_history,
                    selected_floor=float(row["recommended_rho_min"]),
                )
            )
    return cases


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="experiments/phase5/results")
    parser.add_argument("--out-dir", default="experiments/phase5/results/gmg_policy_overhead")
    args = parser.parse_args()

    root = Path(args.results_root)
    cases = _prospective_cases(root) + _density_cases(root)
    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "gmg_policy_overhead_cases.csv", cases)

    categories = sorted(set(row["category"] for row in cases))
    summary_rows = []
    for category in categories + ["all"]:
        subset = cases if category == "all" else [row for row in cases if row["category"] == category]
        summary_rows.append(
            {
                "category": category,
                "n_cases": len(subset),
                "n_converged": sum(int(row["solve_converged"]) for row in subset),
                "mean_selected_solve_iters": float(np.mean([row["selected_solve_iters"] for row in subset])),
                "mean_recorded_policy_iters": float(np.mean([row["recorded_policy_iters"] for row in subset])),
                "mean_reuse_enabled_policy_iters": float(
                    np.mean([row["reuse_enabled_policy_iters"] for row in subset])
                ),
                "max_recorded_policy_to_selected_ratio": float(
                    np.max([row["recorded_policy_to_selected_ratio"] for row in subset])
                ),
                "max_reuse_enabled_policy_to_selected_ratio": float(
                    np.max([row["reuse_enabled_policy_to_selected_ratio"] for row in subset])
                ),
                "total_failed_ladder_iters": sum(int(row["failed_ladder_iters"]) for row in subset),
                "probe_reusable_cases": sum(int(row["probe_reusable"]) for row in subset),
                "mean_selected_solve_time_s": _nanmean_or_nan(
                    [row["selected_solve_time_s"] for row in subset]
                ),
                "mean_recorded_policy_time_s": _nanmean_or_nan(
                    [row["recorded_policy_time_s"] for row in subset]
                ),
                "max_recorded_time_to_selected_ratio": _nanmax_or_nan(
                    [row["recorded_time_to_selected_ratio"] for row in subset]
                ),
                "max_selected_cupy_pool_total_mb_after_solve": _nanmax_or_nan(
                    [row["selected_cupy_pool_total_mb_after_solve"] for row in subset]
                ),
            }
        )
    _write_csv(out_dir / "gmg_policy_overhead_summary.csv", summary_rows)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
        }
    )
    x = np.arange(len(cases))
    selected = np.array([row["selected_solve_iters"] for row in cases], dtype=float)
    probe = np.array([row["probe_iters"] for row in cases], dtype=float)
    failed = np.array([row["failed_ladder_iters"] for row in cases], dtype=float)
    labels = [row["case_id"].replace("seed ", "s") for row in cases]

    fig, ax = plt.subplots(figsize=(12.0, 4.8))
    ax.bar(x, selected, label="selected solve", color="#2563eb")
    ax.bar(x, probe, bottom=selected, label="probe", color="#f59e0b")
    ax.bar(x, failed, bottom=selected + probe, label="failed ladder attempts", color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("FGMRES iterations (iteration-equivalent cost)")
    ax.set_title("Recorded probe/ladder overhead for current Phase 5 selected solves")
    ax.legend(frameon=True, framealpha=0.95, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_gmg_policy_overhead.png", dpi=240, bbox_inches="tight")

    print(f"Wrote {out_dir / 'gmg_policy_overhead_cases.csv'}")
    print(f"Wrote {out_dir / 'gmg_policy_overhead_summary.csv'}")
    print(f"Wrote {out_dir / 'fig_gmg_policy_overhead.png'}")


if __name__ == "__main__":
    main()
