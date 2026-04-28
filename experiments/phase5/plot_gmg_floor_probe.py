from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _append_rows(rows: list[dict], path: Path, source: str) -> None:
    for row in _read_rows(path):
        out = dict(row)
        out["source"] = source
        if "case_kind" not in out:
            out["case_kind"] = f"fixed_floor_{float(out['rho_min']):.0e}"
        rows.append(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        default="experiments/phase5/results/gmg_floor_probe_summary",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    _append_rows(
        rows,
        Path("experiments/phase5/results/gmg_floor_rule_check_seed19/gmg_floor_rule_summary.csv"),
        "seed19_baseline_and_rule",
    )
    _append_rows(
        rows,
        Path("experiments/phase5/results/gmg_high_floor_probe_seed19_rho1e4/atlas_summary.csv"),
        "seed19_rho1e-4",
    )
    _append_rows(
        rows,
        Path("experiments/phase5/results/gmg_high_floor_probe_seed19_rho1e3/atlas_summary.csv"),
        "seed19_rho1e-3",
    )
    _append_rows(
        rows,
        Path("experiments/phase5/results/gmg_high_floor_probe_seeds7_13_rho1e3/atlas_summary.csv"),
        "seeds7_13_rho1e-3",
    )
    _append_rows(
        rows,
        Path("experiments/phase5/results/gmg_high_floor_probe_seed13_p018_rho1e3/atlas_summary.csv"),
        "seed13_p018_rho1e-3",
    )
    _append_rows(
        rows,
        Path("experiments/phase5/results/gmg_high_floor_probe_seed13_p020_rho1e3/atlas_summary.csv"),
        "seed13_p020_rho1e-3",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / "gmg_floor_probe_combined.csv"
    fieldnames = sorted({key for row in rows for key in row})
    with combined_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    low_p = [0.10, 0.12, 0.15]
    seed19 = [
        row
        for row in rows
        if int(row["seed"]) == 19
        and float(row["solid_probability"]) in low_p
        and float(row["rho_min"]) in {1e-12, 1e-6, 1e-4, 1e-3}
    ]
    high_floor = [
        row
        for row in rows
        if float(row["solid_probability"]) in low_p and abs(float(row["rho_min"]) - 1e-3) < 1e-15
    ]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.8, 4.2))

    colors = {0.10: "#dc2626", 0.12: "#ea580c", 0.15: "#2563eb"}
    for probability in low_p:
        subset = [row for row in seed19 if abs(float(row["solid_probability"]) - probability) < 1e-15]
        subset.sort(key=lambda row: float(row["rho_min"]))
        rho = np.array([float(row["rho_min"]) for row in subset])
        residual = np.array([float(row["solve_final_rel_residual"]) for row in subset])
        ax0.plot(
            rho,
            residual,
            marker="o",
            linewidth=2.2,
            color=colors[probability],
            label=f"p={probability:g}",
        )
    ax0.axhline(1e-6, color="#0f172a", linestyle="--", linewidth=1.1, label="target")
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlabel("rho_min")
    ax0.set_ylabel("final GMG-FGMRES residual at 300 iterations")
    ax0.set_title("Full GMG needs a solver-admissible floor")
    ax0.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.94)

    x_labels = []
    iters = []
    bar_colors = []
    for probability in low_p:
        for seed in (7, 13, 19):
            match = [
                row
                for row in high_floor
                if int(row["seed"]) == seed
                and abs(float(row["solid_probability"]) - probability) < 1e-15
            ][0]
            x_labels.append(f"{probability:g}\nseed {seed}")
            iters.append(float(match["solve_iters"]))
            bar_colors.append(colors[probability])
    x = np.arange(len(x_labels))
    ax1.bar(x, iters, color=bar_colors, alpha=0.88)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels, fontsize=8)
    ax1.set_ylabel("iterations to 1e-6")
    ax1.set_title("rho_min=1e-3 restores all low-p seed cases")
    ax1.set_ylim(0, max(iters) * 1.25)

    fig.tight_layout()
    fig_path = out_dir / "fig_gmg_solver_floor_probe.png"
    fig.savefig(fig_path, dpi=240, bbox_inches="tight")
    print(f"Wrote {combined_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
