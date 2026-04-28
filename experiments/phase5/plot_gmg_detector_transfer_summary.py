from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _optional_float(row: dict, key: str):
    value = row.get(key, "")
    if value == "":
        return ""
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="experiments/phase5/results/gmg_detector_transfer_summary",
    )
    args = parser.parse_args()

    combined = []
    bridge_baseline = {
        float(row["solid_probability"]): row
        for row in _read_rows(Path("experiments/phase5/results/gmg_bridge_baseline_seed23_rho1e12/atlas_summary.csv"))
    }
    for row in _read_rows(Path("experiments/phase5/results/gmg_solver_floor_detector_prospective/prospective_combined.csv")):
        kept_original_floor = row["trigger"] == "keep"
        combined.append(
            {
                "category": "cantilever random prospective",
                "case_id": f"seed {row['seed']} p={float(row['solid_probability']):g}",
                "preset": row["preset"],
                "trigger": row["trigger"],
                "probe_r50": float(row["probe_r50"]),
                "probe_r100": float(row["probe_r100"]),
                "probe_r100_over_r50": float(row["probe_r100_over_r50"]),
                "recommended_rho_min": float(row["recommended_rho_min"]),
                "attempted_rho_mins": row.get("attempted_rho_mins", row["recommended_rho_min"]),
                "n_floor_attempts": row.get("n_floor_attempts", "1"),
                "solve_converged": int(row["solve_converged"]),
                "solve_iters": float(row["solve_iters"]),
                "solve_final_rel_residual": float(row["solve_final_rel_residual"]),
                "probe_solve_time_s": _optional_float(row, "probe_solve_time_s"),
                "selected_solve_time_s": _optional_float(row, "selected_solve_time_s"),
                "recorded_policy_time_s": _optional_float(row, "recorded_policy_time_s"),
                "selected_cupy_pool_total_mb_after_solve": _optional_float(
                    row,
                    "selected_cupy_pool_total_mb_after_solve",
                ),
                "baseline_full_converged": int(row["solve_converged"]) if kept_original_floor else "",
                "baseline_full_final_rel_residual": (
                    float(row["solve_final_rel_residual"]) if kept_original_floor else ""
                ),
            }
        )
    for row in _read_rows(Path("experiments/phase5/results/gmg_solver_floor_detector_transfer_bridge_seed23_ladder/prospective_summary.csv")):
        baseline = bridge_baseline[float(row["solid_probability"])]
        combined.append(
            {
                "category": "bridge random geometry transfer",
                "case_id": f"seed {row['seed']} p={float(row['solid_probability']):g}",
                "preset": row["preset"],
                "trigger": row["trigger"],
                "probe_r50": float(row["probe_r50"]),
                "probe_r100": float(row["probe_r100"]),
                "probe_r100_over_r50": float(row["probe_r100_over_r50"]),
                "recommended_rho_min": float(row["recommended_rho_min"]),
                "attempted_rho_mins": row["attempted_rho_mins"],
                "n_floor_attempts": row["n_floor_attempts"],
                "solve_converged": int(row["solve_converged"]),
                "solve_iters": float(row["solve_iters"]),
                "solve_final_rel_residual": float(row["solve_final_rel_residual"]),
                "probe_solve_time_s": _optional_float(row, "probe_solve_time_s"),
                "selected_solve_time_s": _optional_float(row, "selected_solve_time_s"),
                "recorded_policy_time_s": _optional_float(row, "recorded_policy_time_s"),
                "selected_cupy_pool_total_mb_after_solve": _optional_float(
                    row,
                    "selected_cupy_pool_total_mb_after_solve",
                ),
                "baseline_full_converged": int(baseline["solve_converged"]),
                "baseline_full_final_rel_residual": float(baseline["solve_final_rel_residual"]),
            }
        )
    density_dirs = sorted(Path("experiments/phase5/results").glob("gmg_solver_floor_detector_density_*"))
    for density_dir in density_dirs:
        summary_path = density_dir / "density_detector_summary.csv"
        if not summary_path.exists():
            continue
        for row in _read_rows(summary_path):
            kept_original_floor = row["trigger"] == "keep"
            combined.append(
                {
                    "category": "optimized SIMP density",
                    "case_id": row["density_name"],
                    "preset": row["preset"],
                    "trigger": row["trigger"],
                    "probe_r50": float(row["probe_r50"]),
                    "probe_r100": float(row["probe_r100"]),
                    "probe_r100_over_r50": float(row["probe_r100_over_r50"]),
                    "recommended_rho_min": float(row["recommended_rho_min"]),
                    "attempted_rho_mins": row["attempted_rho_mins"],
                    "n_floor_attempts": row["n_floor_attempts"],
                    "solve_converged": int(row["solve_converged"]),
                    "solve_iters": float(row["solve_iters"]),
                    "solve_final_rel_residual": float(row["solve_final_rel_residual"]),
                    "probe_solve_time_s": _optional_float(row, "probe_solve_time_s"),
                    "selected_solve_time_s": _optional_float(row, "selected_solve_time_s"),
                    "recorded_policy_time_s": _optional_float(row, "recorded_policy_time_s"),
                    "selected_cupy_pool_total_mb_after_solve": _optional_float(
                        row,
                        "selected_cupy_pool_total_mb_after_solve",
                    ),
                    "baseline_full_converged": int(row["solve_converged"]) if kept_original_floor else "",
                    "baseline_full_final_rel_residual": (
                        float(row["solve_final_rel_residual"]) if kept_original_floor else ""
                    ),
                }
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "gmg_detector_transfer_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.2, 4.6))
    colors = {
        "cantilever random prospective": "#2563eb",
        "bridge random geometry transfer": "#dc2626",
        "optimized SIMP density": "#16a34a",
    }
    x = np.arange(len(combined))
    labels = []
    for i, row in enumerate(combined):
        color = colors[row["category"]]
        ax0.scatter(
            i,
            row["recommended_rho_min"],
            s=120,
            color=color,
            marker="o" if row["solve_converged"] else "X",
            edgecolor="#0f172a",
            linewidth=0.9,
        )
        ax1.bar(i, row["solve_iters"], color=color, alpha=0.86)
        labels.append(row["case_id"].replace("seed ", "s"))

    ax0.set_yscale("log")
    ax0.axhline(1e-12, color="#64748b", linestyle="--", linewidth=1.0)
    ax0.axhline(1e-3, color="#f97316", linestyle=":", linewidth=1.0)
    ax0.axhline(1e-2, color="#991b1b", linestyle=":", linewidth=1.0)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax0.set_ylabel("selected solver floor")
    ax0.set_title("Detector-selected floors across transfer tests")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=70, ha="right", fontsize=7)
    ax1.set_ylabel("iterations with selected floor")
    ax1.set_title("All selected transfer solves converged")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markersize=8, label=category)
        for category, color in colors.items()
    ]
    ax1.legend(handles=handles, loc="upper right", fontsize=8, frameon=True, framealpha=0.94)

    fig.tight_layout()
    fig_path = out_dir / "fig_gmg_detector_transfer_summary.png"
    fig.savefig(fig_path, dpi=240, bbox_inches="tight")
    print(f"Wrote {csv_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
