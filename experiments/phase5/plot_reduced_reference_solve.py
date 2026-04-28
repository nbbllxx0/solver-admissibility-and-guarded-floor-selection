from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_LABELS = {
    "cpu_direct_spsolve": "direct sparse",
    "cpu_cg_jacobi": "CG + Jacobi",
    "cpu_pyamg_plain": "PyAMG plain",
    "cpu_pyamg_rbm": "PyAMG + RBM",
    "gpu_gmg_phase5": "Phase 5 GMG",
}

METHOD_STYLES = {
    "cpu_direct_spsolve": {"color": "#111827", "marker": "o", "linewidth": 2.2},
    "cpu_cg_jacobi": {"color": "#dc2626", "marker": "x", "linewidth": 1.4, "alpha": 0.75},
    "cpu_pyamg_plain": {"color": "#ea580c", "marker": "^", "linewidth": 1.6, "alpha": 0.8},
    "cpu_pyamg_rbm": {"color": "#7c3aed", "marker": "s", "linewidth": 1.8},
    "gpu_gmg_phase5": {"color": "#0f766e", "marker": "D", "linewidth": 2.2},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="experiments/phase5/results/reduced_reference_cantilever_3d_seed19/reduced_reference_combined.csv",
    )
    parser.add_argument(
        "--out",
        default="experiments/phase5/results/reduced_reference_cantilever_3d_seed19/fig_reduced_reference_residuals.png",
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
    fig, ax = plt.subplots(figsize=(7.3, 4.4))
    methods = list(METHOD_LABELS)
    for method in methods:
        method_rows = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: float(row["solid_probability"]),
        )
        p = np.array([float(row["solid_probability"]) for row in method_rows])
        r = np.array([float(row["rel_residual"]) for row in method_rows])
        style = METHOD_STYLES[method]
        ax.plot(p, r, label=METHOD_LABELS[method], **style)
        for row in method_rows:
            if int(row["converged"]):
                ax.scatter(
                    [float(row["solid_probability"])],
                    [float(row["rel_residual"])],
                    s=82,
                    facecolors="none",
                    edgecolors=style["color"],
                    linewidths=1.5,
                )

    ax.axhline(1e-6, color="#334155", linewidth=1.1, linestyle="--", label="target tolerance")
    ax.axvspan(0.15, 0.20, color="#f59e0b", alpha=0.12, label="transition band")
    ax.set_yscale("log")
    ax.set_xlabel("solid probability p")
    ax.set_ylabel("true final relative residual")
    ax.set_title("Reduced reference solve separates transition physics from AMG weakness")
    ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.94)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
