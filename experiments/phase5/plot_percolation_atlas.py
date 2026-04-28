from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="experiments/phase5/results/percolation_atlas_seeded_300/atlas_summary.csv",
    )
    parser.add_argument(
        "--aggregate",
        default="experiments/phase5/results/percolation_atlas_seeded_300/atlas_aggregate.csv",
    )
    parser.add_argument(
        "--out",
        default="experiments/phase5/results/percolation_atlas_seeded_300/fig_percolation_transition.png",
    )
    args = parser.parse_args()

    summary_path = Path(args.summary)
    aggregate_path = Path(args.aggregate)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(summary_path)
    agg = _read_rows(aggregate_path)

    probabilities = np.array([float(r["solid_probability"]) for r in rows])
    residuals = np.array([float(r["solve_final_rel_residual"]) for r in rows])
    seeds = np.array([int(r["seed"]) for r in rows])
    converged = np.array([int(r["solve_converged"]) for r in rows], dtype=bool)
    largest = np.array([float(r["largest_fraction_of_solid"]) for r in rows])

    p_agg = np.array([float(r["solid_probability"]) for r in agg])
    mean_res = np.array([float(r["mean_final_rel_residual"]) for r in agg])
    min_res = np.array([float(r["min_final_rel_residual"]) for r in agg])
    max_res = np.array([float(r["max_final_rel_residual"]) for r in agg])
    mean_largest = np.array([float(r["mean_largest_fraction_of_solid"]) for r in agg])
    conv_count = np.array([int(r["converged_count"]) for r in agg])
    n_seeds = np.array([int(r["n_seeds"]) for r in agg])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
        }
    )

    fig, (ax0, ax1) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0], "hspace": 0.08},
    )

    for seed in sorted(set(seeds)):
        mask = seeds == seed
        ax0.scatter(
            probabilities[mask & ~converged],
            residuals[mask & ~converged],
            marker="o",
            s=48,
            alpha=0.75,
            label="stalled runs" if seed == sorted(set(seeds))[0] else None,
            color="#9a3412",
        )
        ax0.scatter(
            probabilities[mask & converged],
            residuals[mask & converged],
            marker="s",
            s=52,
            alpha=0.85,
            label="converged runs" if seed == sorted(set(seeds))[0] else None,
            color="#0f766e",
        )

    ax0.plot(p_agg, mean_res, color="#111827", linewidth=2.2, label="seed mean")
    ax0.fill_between(p_agg, min_res, max_res, color="#64748b", alpha=0.18, label="seed range")
    ax0.axhline(1e-6, color="#334155", linewidth=1.1, linestyle="--", label="solver tolerance")
    ax0.axvspan(0.15, 0.20, color="#f59e0b", alpha=0.13, label="observed transition band")
    ax0.set_yscale("log")
    ax0.set_ylabel("final relative residual")
    ax0.set_title("Ultra-sparse mixed fields exhibit a finite-size solver transition")
    ax0.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.92)

    ax1.plot(p_agg, mean_largest, color="#1d4ed8", marker="o", linewidth=2.0)
    for p, c, n in zip(p_agg, conv_count, n_seeds):
        ax1.text(p, max(mean_largest.max() * 0.16, 0.00045), f"{c}/{n}", ha="center", fontsize=8)
    ax1.set_ylabel("largest solid\ncomponent fraction")
    ax1.set_xlabel("solid probability p")
    ax1.set_ylim(0.0, max(0.006, mean_largest.max() * 1.35))
    ax1.text(
        0.102,
        ax1.get_ylim()[1] * 0.84,
        "numbers: converged seeds / total",
        fontsize=8,
        color="#334155",
    )

    fig.savefig(out_path, dpi=240, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
