"""Render Scientific Reports Fig. S6 at its intended 165 mm print width.

Uses the recorded conditioning/critical-floor CSVs without changing the data.
Requires NumPy and Matplotlib; no GPU dependencies are imported.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments/phase5/results"


def table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [{key: float(value) for key, value in row.items()
                 if key in {"seed", "solid_probability", "rho_min", "kappa",
                            "kappa_times_eps", "rel_residual", "critical_rho_min"}}
                for row in csv.DictReader(stream)]


def result_table(relative):
    path = RESULTS / relative
    if not path.is_file():
        # Release archives store the same CSVs under flat manifest names.
        path = ROOT / "data/analysis_tables" / relative.replace("/", "__")
    return table(path)


# savefig metadata that removes the tool name and date
NO_METADATA = {".pdf": {"Creator": None, "Producer": None, "CreationDate": None}, ".png": {"Software": None}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "figures/generated/figS5_direct_conditioning.pdf")
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    rows = result_table("direct_floor_conditioning/direct_floor_conditioning.csv")
    critical = result_table("direct_floor_atlas_seeded/direct_floor_critical.csv")
    fields = sorted({(r["seed"], r["solid_probability"]) for r in rows})
    eps, tau = np.finfo(float).eps, 1e-6
    x = np.array([r["kappa_times_eps"] for r in rows])
    y = np.array([r["rel_residual"] for r in rows])
    passed = y <= tau
    fit = y > 1e-12
    slope, intercept = np.polyfit(np.log10(x[fit]), np.log10(y[fit]), 1)
    correlation = np.corrcoef(np.log10(x[fit]), np.log10(y[fit]))[0, 1]
    screens = {}
    for field in fields:
        first = min((r for r in rows if (r["seed"], r["solid_probability"]) == field), key=lambda r: r["rho_min"])
        screens[field] = first["rho_min"] * first["kappa"] * eps / tau
    assert (len(rows), len(fields), int(passed.sum()), int((~passed).sum())) == (144, 18, 114, 30)
    assert all(r["critical_rho_min"] <= screens[r["seed"], r["solid_probability"]] for r in critical)
    assert abs(slope - .98) < .005 and abs(correlation - .99) < .005

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.4,
                         "axes.labelsize": 7.4, "axes.titlesize": 8,
                         "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
                         "axes.linewidth": .7, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "mathtext.fontset": "dejavusans"})
    teal, red, blue, gray = "#3B9C8C", "#B64B3A", "#2F6F9F", "#697A82"
    fig, axes = plt.subplots(1, 3, figsize=(165 / 25.4, 78 / 25.4))
    fig.subplots_adjust(left=.085, right=.985, bottom=.22, top=.83, wspace=.73)
    for i, ax in enumerate(axes):
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.tick_params(which="both", length=2.4, width=.6, pad=2)
        ax.grid(alpha=.14, which="major", linewidth=.5)
        ax.text(-.20, 1.10, "abc"[i], transform=ax.transAxes, weight="bold", fontsize=10)

    ax = axes[0]
    for field in fields:
        g = sorted((r for r in rows if (r["seed"], r["solid_probability"]) == field), key=lambda r: r["rho_min"])
        ax.plot([r["rho_min"] for r in g], [r["kappa"] for r in g], color=gray, alpha=.45, linewidth=.7)
    for r in critical:
        point = next(v for v in rows if v["seed"] == r["seed"] and v["solid_probability"] == r["solid_probability"] and v["rho_min"] == r["critical_rho_min"])
        ax.scatter(point["rho_min"], point["kappa"], s=13, c=teal, edgecolors="white", linewidths=.3, zorder=4)
    ax.axhline(1 / eps, color=red, ls="--", lw=.9)
    ax.axhline(tau / eps, color=blue, ls=":", lw=.9)
    ax.text(.04, .90, r"$\kappa=\varepsilon^{-1}$", transform=ax.transAxes, color=red, fontsize=7.4)
    ax.text(.04, .36, r"$\kappa\varepsilon=\tau$", transform=ax.transAxes, color=blue, fontsize=7.4)
    ax.set(xlabel=r"Density floor $\rho_{\min}$", ylabel=r"Condition number $\kappa$", title="Conditioning")
    ax.set_xticks([1e-12, 1e-8, 1e-4]); ax.set_yticks([1e7, 1e11, 1e15])
    ax.set_xlim(3e-13, .03)

    ax = axes[1]
    ax.scatter(x[passed], y[passed], s=12, c=teal, alpha=.8, edgecolors="white", linewidths=.25, label=r"$\eta\leq\tau$: 114")
    ax.scatter(x[~passed], y[~passed], s=12, c=red, alpha=.8, edgecolors="white", linewidths=.25, label=r"$\eta>\tau$: 30")
    ref = np.logspace(-11, 2, 200)
    ax.plot(ref, ref, ls="--", lw=.9, color="#172126", label=r"$\eta=\kappa\varepsilon$")
    ax.plot(ref, 10 ** intercept * ref ** slope, lw=.9, color=gray, label=f"Fit {slope:.2f}, r = {correlation:.2f}")
    ax.axvline(tau, color=blue, ls=":", lw=.8); ax.axhline(tau, color=blue, ls=":", lw=.8)
    ax.set(xlim=(1e-11, 1e2), ylim=(1e-14, .1), xlabel=r"Indicator $\kappa\varepsilon$", ylabel=r"Relative residual $\eta$", title="Attained residual")
    ax.set_xticks([1e-10, 1e-6, 1e-2]); ax.set_yticks([1e-13, 1e-9, 1e-5, 1e-1])
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.52, .005),
               ncol=4, fontsize=7.2, frameon=False, handlelength=1.2,
               handletextpad=.4, columnspacing=1.2)

    ax = axes[2]
    ax.scatter([screens[r["seed"], r["solid_probability"]] for r in critical], [r["critical_rho_min"] for r in critical], s=17, c=teal, edgecolors="white", linewidths=.3, zorder=3)
    ref = np.logspace(-9, -5, 100)
    ax.plot(ref, ref, color="#172126", ls="--", lw=.9)
    ax.text(.04, .94, "Screen =\ncritical floor", transform=ax.transAxes, va="top", fontsize=7.2)
    ax.text(.96, .07, "Conservative for\n18 of 18 fields", transform=ax.transAxes, ha="right", va="bottom", color=teal, fontsize=7.2)
    ax.set(xlim=(1e-9, 1e-5), ylim=(1e-13, 1e-5), xlabel=r"Screen $c(\rho)\varepsilon/\tau$", ylabel="Empirical critical floor", title="Floor screen")
    ax.set_xticks([1e-9, 1e-7, 1e-5]); ax.set_yticks([1e-13, 1e-9, 1e-5])
    for ax in axes:
        ax.minorticks_off()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata=NO_METADATA.get(args.output.suffix))
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.preview, dpi=400, metadata=NO_METADATA.get(args.preview.suffix))
    plt.close(fig)
    print(f"Wrote {args.output}: 144 matrices, 114 passes, 30 failures, all 18 screens conservative.")


if __name__ == "__main__":
    main()
