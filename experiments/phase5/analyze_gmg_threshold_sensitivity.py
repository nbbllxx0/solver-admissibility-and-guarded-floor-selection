from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _classify(
    r50: float,
    r100: float,
    *,
    high_residual_threshold: float,
    plateau_residual_threshold: float,
    plateau_ratio_threshold: float,
) -> bool:
    if r50 >= high_residual_threshold:
        return True
    ratio = r100 / max(r50, 1e-300)
    return r100 >= plateau_residual_threshold and ratio >= plateau_ratio_threshold


def _count(rows: list[dict], high: float, plateau_residual: float, plateau_ratio: float) -> dict:
    counts = {
        "true_raise": 0,
        "true_keep": 0,
        "false_raise": 0,
        "false_keep": 0,
    }
    for row in rows:
        predicted = _classify(
            float(row["r50"]),
            float(row["r100"]),
            high_residual_threshold=high,
            plateau_residual_threshold=plateau_residual,
            plateau_ratio_threshold=plateau_ratio,
        )
        desired = bool(int(float(row["desired_raise_floor"])))
        if predicted and desired:
            counts["true_raise"] += 1
        elif not predicted and not desired:
            counts["true_keep"] += 1
        elif predicted and not desired:
            counts["false_raise"] += 1
        else:
            counts["false_keep"] += 1
    total = len(rows)
    counts["total"] = total
    counts["false_raise_rate"] = counts["false_raise"] / max(total, 1)
    counts["false_keep_rate"] = counts["false_keep"] / max(total, 1)
    counts["safe"] = int(counts["false_keep"] == 0)
    return counts


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        default="experiments/phase5/results/gmg_solver_floor_detector/gmg_solver_floor_detector_predictions.csv",
    )
    parser.add_argument("--out-dir", default="experiments/phase5/results/gmg_solver_floor_detector_sensitivity")
    args = parser.parse_args()

    rows = _read_rows(Path(args.predictions))
    high_values = [3e-3, 5e-3, 7.5e-3, 1e-2, 1.5e-2, 2e-2, 3e-2, 5e-2]
    plateau_residual_values = [3e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3]
    plateau_ratio_values = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    grid_rows = []
    for high in high_values:
        for plateau_residual in plateau_residual_values:
            for plateau_ratio in plateau_ratio_values:
                counts = _count(rows, high, plateau_residual, plateau_ratio)
                grid_rows.append(
                    {
                        "labeled_cases": len(rows),
                        "high_residual_threshold": high,
                        "plateau_residual_threshold": plateau_residual,
                        "plateau_ratio_threshold": plateau_ratio,
                        **counts,
                    }
                )

    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "gmg_threshold_sensitivity_grid.csv", grid_rows)

    safe_rows = [row for row in grid_rows if int(row["safe"])]
    safe_rows = sorted(
        safe_rows,
        key=lambda row: (
            int(row["false_raise"]),
            float(row["high_residual_threshold"]),
            float(row["plateau_residual_threshold"]),
            float(row["plateau_ratio_threshold"]),
        ),
    )
    recommended = _count(rows, 1e-2, 1e-4, 0.6)
    summary_rows = [
        {
            "selection": "current_rule",
            "labeled_cases": len(rows),
            "high_residual_threshold": 1e-2,
            "plateau_residual_threshold": 1e-4,
            "plateau_ratio_threshold": 0.6,
            **recommended,
        }
    ]
    for i, row in enumerate(safe_rows[:10], start=1):
        out = dict(row)
        out["selection"] = f"safe_rank_{i}"
        summary_rows.append(out)
    _write_csv(out_dir / "gmg_threshold_sensitivity_summary.csv", summary_rows)

    fixed_plateau_residual = 1e-4
    false_keep = np.zeros((len(plateau_ratio_values), len(high_values)))
    false_raise = np.zeros_like(false_keep)
    for iy, ratio in enumerate(plateau_ratio_values):
        for ix, high in enumerate(high_values):
            counts = _count(rows, high, fixed_plateau_residual, ratio)
            false_keep[iy, ix] = counts["false_keep"]
            false_raise[iy, ix] = counts["false_raise"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True)
    for ax, data, title, cmap in [
        (axes[0], false_keep, "unsafe missed raises", "Reds"),
        (axes[1], false_raise, "conservative false raises", "Blues"),
    ]:
        im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(high_values)))
        ax.set_xticklabels([f"{v:g}" for v in high_values], rotation=40, ha="right")
        ax.set_yticks(range(len(plateau_ratio_values)))
        ax.set_yticklabels([f"{v:g}" for v in plateau_ratio_values])
        ax.set_xlabel("high-r50 threshold")
        ax.set_title(title)
        for iy in range(data.shape[0]):
            for ix in range(data.shape[1]):
                ax.text(ix, iy, f"{int(data[iy, ix])}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    axes[0].set_ylabel("plateau ratio threshold")
    fig.suptitle("GMG detector threshold sensitivity at plateau residual threshold = 1e-4")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_gmg_threshold_sensitivity.png", dpi=240, bbox_inches="tight")

    print(f"Wrote {out_dir / 'gmg_threshold_sensitivity_grid.csv'}")
    print(f"Wrote {out_dir / 'gmg_threshold_sensitivity_summary.csv'}")
    print(f"Wrote {out_dir / 'fig_gmg_threshold_sensitivity.png'}")


if __name__ == "__main__":
    main()
