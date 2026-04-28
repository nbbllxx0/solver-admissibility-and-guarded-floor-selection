from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="experiments/phase5/results/gmg_fixed_floor_controls/fixed_floor_control_summary.csv",
    )
    parser.add_argument("--out-dir", default="experiments/phase5/results/gmg_fixed_floor_controls")
    args = parser.parse_args()

    rows = _read_rows(Path(args.summary))
    by_case: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)

    comparison_rows = []
    for case_id, case_rows in sorted(by_case.items()):
        case_rows = sorted(case_rows, key=lambda row: float(row["rho_min"]))
        reference = case_rows[0]
        reference_floor = float(reference["rho_min"])
        reference_compliance = float(reference["compliance"])
        reference_iters = float(reference["solve_iters"])
        reference_solve_time = float(reference["solve_time_s"])
        for row in case_rows:
            compliance = float(row["compliance"])
            solve_iters = float(row["solve_iters"])
            solve_time = float(row["solve_time_s"])
            comparison_rows.append(
                {
                    "case_id": case_id,
                    "case_type": row["case_type"],
                    "preset": row["preset"],
                    "rho_min": float(row["rho_min"]),
                    "reference_floor": reference_floor,
                    "solve_converged": int(float(row["solve_converged"])),
                    "solve_iters": solve_iters,
                    "solve_time_s": solve_time,
                    "setup_time_s": float(row["setup_time_s"]),
                    "final_rel_residual": float(row["solve_final_rel_residual"]),
                    "compliance": compliance,
                    "relative_compliance_change": (compliance - reference_compliance)
                    / max(abs(reference_compliance), 1e-300),
                    "iteration_ratio_vs_reference": solve_iters / max(reference_iters, 1.0),
                    "solve_time_ratio_vs_reference": solve_time / max(reference_solve_time, 1e-300),
                }
            )

    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "fixed_floor_control_comparison.csv", comparison_rows)

    summary_rows = []
    for case_type in sorted(set(row["case_type"] for row in comparison_rows)) + ["all"]:
        subset = comparison_rows if case_type == "all" else [
            row for row in comparison_rows if row["case_type"] == case_type
        ]
        for floor in sorted(set(row["rho_min"] for row in subset)):
            floor_subset = [
                row for row in subset if abs(row["rho_min"] - floor) < 1e-15 and row["rho_min"] != row["reference_floor"]
            ]
            if not floor_subset:
                continue
            summary_rows.append(
                {
                    "case_type": case_type,
                    "rho_min": floor,
                    "n_cases": len(floor_subset),
                    "max_abs_relative_compliance_change": float(
                        np.max([abs(row["relative_compliance_change"]) for row in floor_subset])
                    ),
                    "mean_abs_relative_compliance_change": float(
                        np.mean([abs(row["relative_compliance_change"]) for row in floor_subset])
                    ),
                    "mean_iteration_ratio_vs_reference": float(
                        np.mean([row["iteration_ratio_vs_reference"] for row in floor_subset])
                    ),
                    "mean_solve_time_ratio_vs_reference": float(
                        np.mean([row["solve_time_ratio_vs_reference"] for row in floor_subset])
                    ),
                }
            )
    _write_csv(out_dir / "fixed_floor_control_comparison_summary.csv", summary_rows)

    plot_rows = [row for row in comparison_rows if row["rho_min"] != row["reference_floor"]]
    labels = [f"{row['case_id']}\n{row['rho_min']:.0e}" for row in plot_rows]
    x = np.arange(len(plot_rows))
    rel = np.array([100.0 * row["relative_compliance_change"] for row in plot_rows])
    colors = ["#2563eb" if row["case_type"] == "density" else "#dc2626" for row in plot_rows]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.24,
        }
    )
    fig, ax = plt.subplots(figsize=(11.0, 4.6))
    ax.bar(x, rel, color=colors, alpha=0.88)
    ax.axhline(0.0, color="#0f172a", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
    ax.set_ylabel("compliance change vs lowest tested floor (%)")
    ax.set_title("Always-high-floor controls change the physics even when solves converge")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_fixed_floor_compliance_shift.png", dpi=240, bbox_inches="tight")

    print(f"Wrote {out_dir / 'fixed_floor_control_comparison.csv'}")
    print(f"Wrote {out_dir / 'fixed_floor_control_comparison_summary.csv'}")
    print(f"Wrote {out_dir / 'fig_fixed_floor_compliance_shift.png'}")


if __name__ == "__main__":
    main()
