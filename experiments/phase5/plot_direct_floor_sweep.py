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
        default="experiments/phase5/results/reduced_reference_cantilever_3d_seed19/direct_floor_sweep.csv",
    )
    parser.add_argument(
        "--out",
        default="experiments/phase5/results/reduced_reference_cantilever_3d_seed19/fig_direct_floor_sweep.png",
    )
    args = parser.parse_args()

    rows = list(csv.DictReader(Path(args.csv).open(newline="", encoding="utf-8")))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.2, 6.0), sharex=True)
    fig.subplots_adjust(hspace=0.08)
    colors = {0.10: "#9a3412", 0.20: "#0f766e"}
    for probability in sorted({float(row["p"]) for row in rows}):
        subset = sorted(
            [row for row in rows if abs(float(row["p"]) - probability) < 1e-12],
            key=lambda row: float(row["rho_min"]),
        )
        floors = np.array([float(row["rho_min"]) for row in subset])
        residuals = np.array([float(row["rel_residual"]) for row in subset])
        xnorms = np.array([float(row["x_norm"]) for row in subset])
        label = f"p={probability:.2f}"
        ax0.plot(floors, residuals, marker="o", linewidth=2.2, color=colors[probability], label=label)
        ax1.plot(floors, xnorms, marker="s", linewidth=2.0, color=colors[probability], label=label)

    ax0.axhline(1e-6, color="#334155", linestyle="--", linewidth=1.1, label="target tolerance")
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax0.set_ylabel("direct solve residual")
    ax1.set_ylabel("solution norm")
    ax1.set_xlabel("void stiffness floor rho_min")
    ax0.set_title("Below-transition residual loss is controlled by the void stiffness floor")
    ax0.legend(loc="lower left", fontsize=8, frameon=True, framealpha=0.94)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
