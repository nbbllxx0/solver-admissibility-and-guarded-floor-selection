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
        default="experiments/phase5/results/admissibility_detector_validation/detector_leave_one_seed_summary.csv",
    )
    parser.add_argument(
        "--rules",
        default="experiments/phase5/results/admissibility_detector_validation/detector_rule_thresholds.csv",
    )
    parser.add_argument(
        "--safety-factor",
        type=float,
        default=10.0,
        help="Residual safety factor to plot for the recommended threshold curve.",
    )
    parser.add_argument(
        "--out",
        default="experiments/phase5/results/admissibility_detector_validation/fig_admissibility_detector_validation.png",
    )
    args = parser.parse_args()

    summary_rows = _read_rows(Path(args.summary))
    rule_rows = _read_rows(Path(args.rules))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    median_summary = [row for row in summary_rows if row["mode"] == "median"]
    safety = np.array([float(row["safety_factor"]) for row in median_summary])
    false_admissible = np.array([float(row["false_admissible_rate"]) for row in median_summary])
    false_inadmissible = np.array([float(row["false_inadmissible_rate"]) for row in median_summary])
    order = np.argsort(safety)
    safety = safety[order]
    false_admissible = false_admissible[order]
    false_inadmissible = false_inadmissible[order]

    selected_rules = [
        row for row in rule_rows if abs(float(row["safety_factor"]) - args.safety_factor) < 1e-12
    ]
    p = np.array([float(row["solid_probability"]) for row in selected_rules])
    median_threshold = np.array([float(row["median_seed_threshold"]) for row in selected_rules])
    conservative_threshold = np.array([float(row["conservative_seed_threshold"]) for row in selected_rules])
    min_threshold = np.array([float(row["min_seed_threshold"]) for row in selected_rules])
    order = np.argsort(p)
    p = p[order]
    median_threshold = median_threshold[order]
    conservative_threshold = conservative_threshold[order]
    min_threshold = min_threshold[order]

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

    ax0.plot(
        safety,
        false_admissible,
        marker="o",
        linewidth=2.2,
        color="#dc2626",
        label="unsafe admissible",
    )
    ax0.plot(
        safety,
        false_inadmissible,
        marker="s",
        linewidth=2.2,
        color="#2563eb",
        label="conservative rejection",
    )
    ax0.set_xscale("log")
    ax0.set_xticks(safety)
    ax0.set_xticklabels([f"{v:g}" for v in safety])
    ax0.set_xlabel("training residual safety factor")
    ax0.set_ylabel("leave-one-seed error rate")
    ax0.set_title("Safety margin removes unsafe floor predictions")
    ax0.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.94)

    ax1.fill_between(
        p,
        min_threshold,
        conservative_threshold,
        step="mid",
        color="#f97316",
        alpha=0.18,
        label="seed range",
    )
    ax1.step(
        p,
        median_threshold,
        where="mid",
        color="#0f172a",
        linewidth=2.4,
        label=f"median rule, safety={args.safety_factor:g}",
    )
    ax1.step(
        p,
        conservative_threshold,
        where="mid",
        color="#b45309",
        linewidth=1.8,
        linestyle="--",
        label="conservative rule",
    )
    ax1.scatter(p, median_threshold, s=60, color="#0f172a", zorder=3)
    ax1.axhline(1e-12, color="#64748b", linestyle="--", linewidth=1.1, label="paper4 floor")
    ax1.axvspan(0.15, 0.20, color="#f59e0b", alpha=0.12, label="transition band")
    ax1.set_yscale("log")
    ax1.set_xlabel("solid probability p")
    ax1.set_ylabel("recommended rho_min")
    ax1.set_title("Admissibility floor rule learned from direct solves")
    ax1.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.94)

    fig.tight_layout()
    fig.savefig(out, dpi=240, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
