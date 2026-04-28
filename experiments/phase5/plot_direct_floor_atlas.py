from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="experiments/phase5/results/direct_floor_atlas_seed19/direct_floor_atlas.csv",
    )
    parser.add_argument(
        "--critical",
        default="experiments/phase5/results/direct_floor_atlas_seed19/direct_floor_critical.csv",
    )
    parser.add_argument(
        "--out",
        default="experiments/phase5/results/direct_floor_atlas_seed19/fig_direct_floor_atlas.png",
    )
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.csv).open(newline="", encoding="utf-8")))
    crit_rows = list(csv.DictReader(Path(args.critical).open(newline="", encoding="utf-8")))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    probabilities = sorted({float(row["solid_probability"]) for row in rows})
    floors = sorted({float(row["rho_min"]) for row in rows})
    # Equal-spaced categorical axes make the admissibility matrix readable even
    # when probabilities are tightly clustered near the transition.
    floors_desc = sorted(floors, reverse=True)
    residual_grid = np.full((len(floors_desc), len(probabilities)), np.nan)
    converged_grid = np.zeros((len(floors_desc), len(probabilities)), dtype=bool)
    for row in rows:
        i = floors_desc.index(float(row["rho_min"]))
        j = probabilities.index(float(row["solid_probability"]))
        residual_grid[i, j] = float(row["rel_residual"])
        converged_grid[i, j] = bool(int(row["converged"]))

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    mesh = ax.imshow(
        np.log10(residual_grid),
        cmap="magma_r",
        aspect="auto",
        vmin=-12,
        vmax=-2,
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("log10 direct residual")

    for i, floor in enumerate(floors_desc):
        for j, probability in enumerate(probabilities):
            marker = "✓" if converged_grid[i, j] else "×"
            ax.text(
                j,
                i,
                marker,
                ha="center",
                va="center",
                color="white" if np.log10(residual_grid[i, j]) > -7 else "#111827",
                fontsize=13,
                fontweight="bold",
            )

    for row in crit_rows:
        if row["critical_rho_min"]:
            j = probabilities.index(float(row["solid_probability"]))
            i = floors_desc.index(float(row["critical_rho_min"]))
            ax.scatter([j], [i], s=145, facecolors="none", edgecolors="#38bdf8", linewidths=2.2)

    ax.set_xticks(range(len(probabilities)))
    ax.set_xticklabels([f"{p:.2f}" for p in probabilities])
    ax.set_yticks(range(len(floors_desc)))
    ax.set_yticklabels([f"$10^{{{int(np.log10(floor))}}}$" for floor in floors_desc])
    ax.set_xlabel("solid probability p")
    ax.set_ylabel("void stiffness floor rho_min")
    ax.set_title("Minimal void-floor regularization for direct residual admissibility")
    ax.text(
        0.05,
        0.10,
        "cyan circles: first floor reaching 1e-6 residual",
        color="#0f172a",
        fontsize=8,
        transform=ax.transAxes,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 2},
    )
    fig.savefig(out, dpi=240, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
