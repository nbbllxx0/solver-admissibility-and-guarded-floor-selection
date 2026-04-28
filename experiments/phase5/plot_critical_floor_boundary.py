from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggregate",
        default="experiments/phase5/results/direct_floor_atlas_seeded/direct_floor_critical_aggregate.csv",
    )
    parser.add_argument(
        "--out",
        default="experiments/phase5/results/direct_floor_atlas_seeded/fig_critical_floor_boundary.png",
    )
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.aggregate).open(newline="", encoding="utf-8")))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    p = np.array([float(row["solid_probability"]) for row in rows])
    rho_min = np.array([float(row["min_critical_rho"]) for row in rows])
    rho_max = np.array([float(row["max_critical_rho"]) for row in rows])
    rho_med = np.array([float(row["median_critical_rho"]) for row in rows])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.fill_between(p, rho_min, rho_max, step="mid", color="#38bdf8", alpha=0.24, label="seed range")
    ax.step(p, rho_med, where="mid", color="#0f172a", linewidth=2.4, label="median critical floor")
    ax.scatter(p, rho_med, s=72, color="#0f172a", zorder=3)
    ax.axhline(1e-12, color="#64748b", linestyle="--", linewidth=1.1, label="paper4 floor")
    ax.axvspan(0.15, 0.20, color="#f59e0b", alpha=0.13, label="solver transition band")
    ax.set_yscale("log")
    ax.set_xlabel("solid probability p")
    ax.set_ylabel("minimal rho_min for direct residual <= 1e-6")
    ax.set_title("Admissible void floor decreases sharply across the sparse-field transition")
    ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.94)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
