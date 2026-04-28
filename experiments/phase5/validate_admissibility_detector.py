from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path


def _rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _per_seed_critical_map(atlas_rows: list[dict], *, train_tol: float) -> dict[tuple[int, float], float]:
    grouped: dict[tuple[int, float], list[tuple[float, float]]] = {}
    for row in atlas_rows:
        grouped.setdefault((int(row["seed"]), float(row["solid_probability"])), []).append(
            (float(row["rho_min"]), float(row["rel_residual"]))
        )

    out: dict[tuple[int, float], float] = {}
    for key, vals in grouped.items():
        for rho_min, rel_residual in sorted(vals):
            if rel_residual <= train_tol:
                out[key] = rho_min
                break
        else:
            out[key] = math.inf
    return out


def _critical_map(
    seed_critical: dict[tuple[int, float], float],
    *,
    heldout_seed: int,
    mode: str,
) -> dict[float, float]:
    grouped: dict[float, list[float]] = {}
    for (seed, probability), critical_rho_min in seed_critical.items():
        if seed == heldout_seed:
            continue
        grouped.setdefault(probability, []).append(critical_rho_min)
    out = {}
    for p, vals in grouped.items():
        if mode == "conservative":
            out[p] = max(vals)
        elif mode == "median":
            logs = [math.log10(v) if math.isfinite(v) else math.inf for v in vals]
            out[p] = float(10.0 ** statistics.median(logs))
        else:
            raise ValueError(f"unknown mode {mode!r}")
    return out


def _lookup_threshold(probability: float, critical: dict[float, float]) -> float:
    if probability in critical:
        return critical[probability]
    # Conservative nearest-neighbor extrapolation over the reduced atlas grid.
    nearest = min(critical, key=lambda p: abs(p - probability))
    return critical[nearest]


def _aggregate_rule_rows(
    seed_critical: dict[tuple[int, float], float],
    *,
    safety_factor: float,
    train_tol: float,
) -> list[dict]:
    grouped: dict[float, list[float]] = {}
    for (_, probability), critical_rho_min in seed_critical.items():
        grouped.setdefault(probability, []).append(critical_rho_min)

    rows = []
    for probability, vals in sorted(grouped.items()):
        logs = [math.log10(v) if math.isfinite(v) else math.inf for v in vals]
        median_threshold = float(10.0 ** statistics.median(logs))
        conservative_threshold = max(vals)
        rows.append(
            {
                "safety_factor": safety_factor,
                "train_tol": train_tol,
                "solid_probability": probability,
                "n_seeds": len(vals),
                "min_seed_threshold": min(vals),
                "median_seed_threshold": median_threshold,
                "conservative_seed_threshold": conservative_threshold,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--atlas",
        default="experiments/phase5/results/direct_floor_atlas_seeded/direct_floor_atlas.csv",
    )
    parser.add_argument(
        "--critical",
        default="experiments/phase5/results/direct_floor_atlas_seeded/direct_floor_critical.csv",
    )
    parser.add_argument(
        "--safety-factors",
        default="1,10,100",
        help="Comma-separated residual safety factors. Training threshold is tol / factor.",
    )
    parser.add_argument(
        "--out-dir",
        default="experiments/phase5/results/admissibility_detector_validation",
    )
    args = parser.parse_args()

    atlas_rows = _rows(Path(args.atlas))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds = sorted({int(row["seed"]) for row in atlas_rows})
    test_tol = max(float(row["tol"]) for row in atlas_rows)
    safety_factors = [float(item) for item in args.safety_factors.split(",") if item.strip()]

    prediction_rows = []
    summary_rows = []
    rule_rows = []
    for safety_factor in safety_factors:
        train_tol = test_tol / safety_factor
        seed_critical = _per_seed_critical_map(atlas_rows, train_tol=train_tol)
        rule_rows.extend(
            _aggregate_rule_rows(
                seed_critical,
                safety_factor=safety_factor,
                train_tol=train_tol,
            )
        )
        for mode in ("median", "conservative"):
            counts = {
                "true_admissible": 0,
                "true_inadmissible": 0,
                "false_admissible": 0,
                "false_inadmissible": 0,
            }
            for seed in seeds:
                critical = _critical_map(seed_critical, heldout_seed=seed, mode=mode)
                for row in atlas_rows:
                    if int(row["seed"]) != seed:
                        continue
                    threshold = _lookup_threshold(float(row["solid_probability"]), critical)
                    predicted_admissible = float(row["rho_min"]) >= threshold
                    observed_admissible = bool(int(row["converged"]))
                    if predicted_admissible and observed_admissible:
                        bucket = "true_admissible"
                    elif predicted_admissible and not observed_admissible:
                        bucket = "false_admissible"
                    elif not predicted_admissible and observed_admissible:
                        bucket = "false_inadmissible"
                    else:
                        bucket = "true_inadmissible"
                    counts[bucket] += 1
                    prediction_rows.append(
                        {
                            "mode": mode,
                            "safety_factor": safety_factor,
                            "train_tol": train_tol,
                            "heldout_seed": seed,
                            "solid_probability": row["solid_probability"],
                            "rho_min": row["rho_min"],
                            "threshold": threshold,
                            "observed_admissible": int(observed_admissible),
                            "predicted_admissible": int(predicted_admissible),
                            "bucket": bucket,
                            "rel_residual": row["rel_residual"],
                        }
                    )
            total = sum(counts.values())
            summary_rows.append(
                {
                    "mode": mode,
                    "safety_factor": safety_factor,
                    "train_tol": train_tol,
                    **counts,
                    "total": total,
                    "false_admissible_rate": counts["false_admissible"] / max(total, 1),
                    "false_inadmissible_rate": counts["false_inadmissible"] / max(total, 1),
                }
            )

    pred_path = out_dir / "detector_leave_one_seed_predictions.csv"
    with pred_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "mode",
                "safety_factor",
                "train_tol",
                "heldout_seed",
                "solid_probability",
                "rho_min",
                "threshold",
                "observed_admissible",
                "predicted_admissible",
                "bucket",
                "rel_residual",
            ],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    summary_path = out_dir / "detector_leave_one_seed_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    rule_path = out_dir / "detector_rule_thresholds.csv"
    with rule_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rule_rows[0]))
        writer.writeheader()
        writer.writerows(rule_rows)

    for row in summary_rows:
        print(
            f"{row['mode']} safety={row['safety_factor']}: "
            f"false_admissible={row['false_admissible']}, "
            f"false_inadmissible={row['false_inadmissible']}, total={row['total']}",
            flush=True,
        )
    print(f"Wrote {pred_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {rule_path}", flush=True)


if __name__ == "__main__":
    main()
