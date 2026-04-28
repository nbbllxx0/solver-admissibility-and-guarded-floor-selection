from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _append_prospective(rows: list[dict], path: Path, *, source: str) -> None:
    for row in _read_rows(path):
        out = dict(row)
        out["source"] = source
        rows.append(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training",
        default="experiments/phase5/results/gmg_solver_floor_detector/gmg_solver_floor_detector_predictions.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/phase5/results/gmg_solver_floor_detector_prospective",
    )
    args = parser.parse_args()

    prospective_rows: list[dict] = []
    _append_prospective(
        prospective_rows,
        Path("experiments/phase5/results/gmg_solver_floor_detector_prospective_seed23/prospective_summary.csv"),
        source="seed23",
    )
    _append_prospective(
        prospective_rows,
        Path("experiments/phase5/results/gmg_solver_floor_detector_prospective_seed31/prospective_summary.csv"),
        source="seed31",
    )
    _append_prospective(
        prospective_rows,
        Path("experiments/phase5/results/gmg_solver_floor_detector_prospective_highp_controls/prospective_summary.csv"),
        source="highp_controls",
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = out_dir / "prospective_combined.csv"
    fieldnames = sorted({key for row in prospective_rows for key in row})
    with combined_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prospective_rows)

    training_rows = _read_rows(Path(args.training))
    success_count = sum(int(row["solve_converged"]) for row in prospective_rows)
    raise_count = sum(int(row["predicted_raise_floor"]) for row in prospective_rows)
    keep_count = len(prospective_rows) - raise_count

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.8, 4.4))

    for row in training_rows:
        raise_floor = bool(int(row["predicted_raise_floor"]))
        color = "#dc2626" if raise_floor else "#2563eb"
        ax0.scatter(
            float(row["r50"]),
            float(row["r100_over_r50"]),
            s=42,
            marker="o",
            color=color,
            alpha=0.22,
            edgecolor="none",
        )
    for row in prospective_rows:
        raise_floor = bool(int(row["predicted_raise_floor"]))
        color = "#dc2626" if raise_floor else "#2563eb"
        edge = "#166534" if bool(int(row["solve_converged"])) else "#111827"
        ax0.scatter(
            float(row["probe_r50"]),
            float(row["probe_r100_over_r50"]),
            s=125,
            marker="*",
            color=color,
            edgecolor=edge,
            linewidth=1.2,
            alpha=0.94,
        )
        ax0.annotate(
            f"{int(row['seed'])}, p={float(row['solid_probability']):g}",
            (float(row["probe_r50"]), float(row["probe_r100_over_r50"])),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=7,
        )
    ax0.axvline(1e-2, color="#dc2626", linestyle="--", linewidth=1.1)
    ax0.axhline(0.6, color="#b45309", linestyle="--", linewidth=1.1)
    ax0.set_xscale("log")
    ax0.set_xlabel("probe residual r50")
    ax0.set_ylabel("probe ratio r100 / r50")
    ax0.set_title("Prospective cases fall on detector-selected actions")

    labels = []
    iters = []
    colors = []
    for row in sorted(prospective_rows, key=lambda r: (int(r["seed"]), float(r["solid_probability"]))):
        labels.append(f"{int(row['seed'])}\n{float(row['solid_probability']):g}")
        iters.append(float(row["solve_iters"]))
        colors.append("#dc2626" if bool(int(row["predicted_raise_floor"])) else "#2563eb")
    x = np.arange(len(labels))
    ax1.bar(x, iters, color=colors, alpha=0.88)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("iterations with recommended floor")
    ax1.set_title(f"Prospective recommended solves: {success_count}/{len(prospective_rows)} converged")
    ax1.text(
        0.02,
        0.96,
        f"raise: {raise_count}, keep: {keep_count}",
        transform=ax1.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
    )

    fig.tight_layout()
    fig_path = out_dir / "fig_gmg_solver_floor_prospective.png"
    fig.savefig(fig_path, dpi=240, bbox_inches="tight")
    print(f"Wrote {combined_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
