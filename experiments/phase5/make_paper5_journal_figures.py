from __future__ import annotations

from pathlib import Path
import gc
import io
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

try:
    import pyvista as pv
except ModuleNotFoundError:  # Allows data-only figures to regenerate without the 3D rendering dependency.
    pv = None


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "phase5" / "results"
OUT = Path(os.environ.get("PAPER5_FIGURE_OUT", str(ROOT / "figures" / "generated")))
STRICT_FIXED_FLOOR_RESULTS = RESULTS / "gmg_fixed_floor_controls_strict_true_residual"
STRICT_OPTIMIZED_BRIDGE_RESULTS = RESULTS / "optimized_density_B500_MF_strict_true_residual"
REVIEW_SUMMARY_RESULTS = RESULTS / "review_experiment_summary"
GUARDED_ADAPTIVE_TRAJECTORY_RESULTS = RESULTS / "guarded_adaptive_trajectories"


COLORS = {
    "ink": "#172126",
    "blue": "#2F6F9F",
    "teal": "#3B9C8C",
    "orange": "#C56B2C",
    "red": "#B64B3A",
    "sand": "#E7D8B8",
    "pale": "#F5F0E6",
    "gray": "#697A82",
}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 170,
            "savefig.dpi": 350,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["ink"],
            "ytick.color": COLORS["ink"],
            "text.color": COLORS["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _box(ax, xy, w, h, text, fc, ec=None, fontsize=9, lw=1.2):
    patch = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.06",
        fc=fc,
        ec=ec or COLORS["ink"],
        lw=lw,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )
    return patch


def _arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            lw=1.3,
            color=COLORS["ink"],
            shrinkA=3,
            shrinkB=3,
        )
    )


def _polyline_arrow(ax, points, color=None):
    color = color or COLORS["ink"]
    if len(points) > 2:
        xs, ys = zip(*points[:-1])
        ax.plot(xs, ys, color=color, lw=1.3, solid_capstyle="round")
    ax.add_patch(
        FancyArrowPatch(
            points[-2],
            points[-1],
            arrowstyle="-|>",
            mutation_scale=12,
            lw=1.3,
            color=color,
            shrinkA=0,
            shrinkB=4,
        )
    )


def make_policy_schematic() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 2.45))
    ax.set_xlim(0, 12.55)
    ax.set_ylim(1.62, 5.45)
    ax.axis("off")

    _box(
        ax,
        (0.25, 4.00),
        1.50,
        0.78,
        "Density field\nfloor $10^{-12}$",
        COLORS["pale"],
    )
    _box(
        ax,
        (2.05, 4.00),
        1.55,
        0.78,
        "100-iter probe\nrecord $r_{50}, r_{100}$",
        "#EAF3F1",
    )
    _box(
        ax,
        (3.90, 4.00),
        1.62,
        0.78,
        "Residual rule\nkeep or raise?",
        "#F2E6D8",
    )
    _box(
        ax,
        (6.25, 4.62),
        1.55,
        0.72,
        "Solve at\noriginal floor",
        "#E8F1FA",
    )
    _box(
        ax,
        (6.25, 3.20),
        1.55,
        0.72,
        "Solve from\nfloor ladder",
        "#FBE8E1",
    )
    _box(
        ax,
        (8.70, 3.86),
        1.78,
        1.03,
        "Final guard\ntrue residual\n$\\|f-Ku\\|/\\|f\\|$",
        "#FFFFFF",
        ec=COLORS["ink"],
        lw=1.8,
        fontsize=8.4,
    )
    _box(
        ax,
        (11.35, 4.14),
        0.90,
        0.74,
        "accept",
        COLORS["sand"],
        fontsize=8.4,
    )
    _box(
        ax,
        (4.65, 2.02),
        2.70,
        0.78,
        "Fallback ladder\n$10^{-3}\\rightarrow10^{-2}$;\nelse report failure",
        "#FBE8E1",
        ec=COLORS["red"],
        fontsize=7.9,
    )
    _arrow(ax, (1.75, 4.39), (2.05, 4.39))
    _arrow(ax, (3.60, 4.39), (3.90, 4.39))
    _arrow(ax, (5.52, 4.55), (6.25, 4.98))
    _polyline_arrow(ax, [(5.52, 4.16), (5.72, 3.30), (5.98, 2.80)])
    _arrow(ax, (7.80, 4.98), (8.70, 4.55))
    _arrow(ax, (7.80, 3.56), (8.70, 4.12))
    _arrow(ax, (10.48, 4.42), (11.35, 4.51))
    _polyline_arrow(ax, [(9.55, 3.86), (9.55, 3.00), (7.35, 2.41)], color=COLORS["red"])
    _arrow(ax, (6.00, 2.80), (6.85, 3.20))

    ax.text(5.92, 5.14, "predict keep", fontsize=8.0, ha="center", va="bottom", color=COLORS["gray"])
    ax.text(5.54, 3.26, "predict raise", fontsize=8.0, ha="right", va="center", color=COLORS["gray"])
    ax.text(10.92, 4.62, "pass", fontsize=8.0, ha="center", va="bottom", color=COLORS["gray"])
    ax.text(9.66, 3.32, "guard fail", fontsize=8.0, ha="left", va="center", color=COLORS["red"])

    fig.tight_layout(pad=0.25)
    fig.savefig(OUT / "fig1_solver_admissibility_policy.png", bbox_inches="tight")
    fig.savefig(OUT / "fig1_solver_admissibility_policy.pdf", bbox_inches="tight")
    plt.close(fig)


def _floor_label(value) -> str:
    value = float(value)
    if value <= 1e-11:
        return "$10^{-12}$"
    if np.isclose(value, 1e-3):
        return "$10^{-3}$"
    if np.isclose(value, 1e-2):
        return "$10^{-2}$"
    return f"{value:g}"


def _reader_case_label(case_id: str) -> str:
    labels = {
        "seed 23 p=0.10": "cantilever\nseed 23\nq=0.10",
        "seed 23 p=0.15": "cantilever\nseed 23\nq=0.15",
        "seed 23 p=0.18": "cantilever\nseed 23\nq=0.18",
        "seed 23 p=0.20": "cantilever\nseed 23\nq=0.20",
        "seed 23 p=0.35": "cantilever\nseed 23\nq=0.35",
        "seed 31 p=0.10": "cantilever\nseed 31\nq=0.10",
        "seed 31 p=0.15": "cantilever\nseed 31\nq=0.15",
        "seed 31 p=0.18": "cantilever\nseed 31\nq=0.18",
        "seed 31 p=0.20": "cantilever\nseed 31\nq=0.20",
        "seed 31 p=0.35": "cantilever\nseed 31\nq=0.35",
        "bridge_seed23_p010": "bridge\nseed 23\nq=0.10",
        "bridge_seed23_p020": "bridge\nseed 23\nq=0.20",
        "bridge_seed23_p035": "bridge\nseed 23\nq=0.35",
        "C64_MF": "medium\ncantilever",
        "C512_MF": "large\ncantilever",
        "Brk500_MF": "large\nbracket",
        "B500_MF": "large\nbridge",
        "cant_s23_p035": "random\ncantilever A",
        "cant_s31_p035": "random\ncantilever B",
    }
    return labels.get(case_id, case_id.replace("_", "\n"))


def make_evidence_matrix() -> None:
    heldout = pd.read_csv(REVIEW_SUMMARY_RESULTS / "heldout_full_true_labels_joined.csv")
    false_keeps = pd.read_csv(REVIEW_SUMMARY_RESULTS / "heldout_full_true_label_false_keeps.csv")
    baselines = pd.read_csv(REVIEW_SUMMARY_RESULTS / "policy_baseline_comparison.csv")

    tp = int(((heldout["true_raise_floor"].astype(str) == "True") & (heldout["detector_predicted_raise"].astype(str) == "True")).sum())
    fn = int(((heldout["true_raise_floor"].astype(str) == "True") & (heldout["detector_predicted_raise"].astype(str) != "True")).sum())
    fp = int(((heldout["true_keep_original_floor"].astype(str) == "True") & (heldout["detector_predicted_raise"].astype(str) == "True")).sum())
    tn = int(((heldout["true_keep_original_floor"].astype(str) == "True") & (heldout["detector_predicted_raise"].astype(str) != "True")).sum())

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.6))
    ax = axes[0, 0]
    matrix = np.array([[tp, fn], [fp, tn]], dtype=float)
    im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1, matrix.max()))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["predict\nraise", "predict\nkeep"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["true\nraise", "true\nkeep"])
    ax.set_title("Held-out detector decisions (n=102)")
    for (i, j), value in np.ndenumerate(matrix):
        color = "white" if value > matrix.max() * 0.45 else COLORS["ink"]
        ax.text(j, i, f"{int(value)}", ha="center", va="center", fontsize=13, fontweight="bold", color=color)
    ax.text(
        0.5,
        -0.20,
        "Four predicted keeps are unsafe without the final guard.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=7)
    ax.text(-0.16, 1.06, "a", transform=ax.transAxes, fontsize=13, fontweight="bold")

    ax = axes[0, 1]
    recovery_labels = ["detector-only\nunsafe keeps", "guard\nfallbacks", "final\nfailures"]
    recovery_values = [len(false_keeps), int(false_keeps["post_solve_fallback_triggered"].astype(int).sum()), int((heldout["solve_converged_guarded"].astype(int) == 0).sum())]
    ax.bar(recovery_labels, recovery_values, color=[COLORS["red"], COLORS["blue"], COLORS["teal"]])
    ax.set_ylim(0, max(recovery_values) + 1.3)
    ax.set_ylabel("cases")
    ax.set_title("True-residual guard recovery")
    for i, value in enumerate(recovery_values):
        ax.text(i, value + 0.12, f"{int(value)}", ha="center", va="bottom", fontweight="bold")
    ax.text(-0.16, 1.06, "b", transform=ax.transAxes, fontsize=13, fontweight="bold")

    ax = axes[1, 0]
    selected = heldout["recommended_rho_min_guarded"].astype(float)
    floor_order = [1e-12, 1e-3, 1e-2]
    counts = [int(np.isclose(selected, floor).sum()) for floor in floor_order]
    x = np.arange(len(floor_order))
    ax.bar(x, counts, color=[COLORS["teal"], COLORS["blue"], COLORS["orange"]])
    ax.set_xticks(x)
    ax.set_xticklabels([_floor_label(v) for v in floor_order])
    ax.set_ylabel("held-out cases")
    ax.set_title("Selected floors after\nguarded policy")
    ax.set_ylim(0, max(counts) + 10)
    for i, value in enumerate(counts):
        ax.text(i, value + 1.2, f"{value}", ha="center", va="bottom", fontweight="bold")
    ax.text(-0.16, 1.06, "c", transform=ax.transAxes, fontsize=13, fontweight="bold")

    ax = axes[1, 1]
    policy_order = [
        ("guarded_probe_policy", "guarded"),
        ("original_floor_full_then_fallback", "full\nfallback"),
        ("always_1e-03", "fixed\n$10^{-3}$"),
        ("always_1e-02", "fixed\n$10^{-2}$"),
        ("probe_severity_jump_1e-02", "severity\njump"),
    ]
    rows = baselines.set_index("policy").loc[[key for key, _ in policy_order]]
    x = np.arange(len(rows))
    mean_vals = rows["mean_time_s"].astype(float).to_numpy()
    med_vals = rows["median_time_s"].astype(float).to_numpy()
    colors = [COLORS["blue"], COLORS["gray"], COLORS["teal"], COLORS["orange"], COLORS["sand"]]
    ax.bar(x, mean_vals, color=colors, edgecolor="white", linewidth=0.7, label="mean")
    ax.scatter(x, med_vals, color=COLORS["ink"], marker="D", s=26, zorder=3, label="median")
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in policy_order])
    ax.set_ylabel("wall time per case (s)")
    ax.set_title("Policy timing on the same 102 cases")
    ax.legend(frameon=False, loc="upper right")
    for xpos, val in enumerate(mean_vals):
        ax.text(xpos, val + 3.2, f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.text(-0.16, 1.06, "d", transform=ax.transAxes, fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "fig2_validation_evidence_matrix.png", bbox_inches="tight")
    fig.savefig(OUT / "fig2_validation_evidence_matrix.pdf", bbox_inches="tight")
    plt.close(fig)


def make_residual_phase_map() -> None:
    retro = pd.read_csv(
        RESULTS
        / "gmg_solver_floor_detector"
        / "gmg_solver_floor_detector_predictions.csv"
    )
    retro = retro.rename(
        columns={
            "r50": "probe_r50",
            "r100": "probe_r100",
            "r100_over_r50": "probe_r100_over_r50",
        }
    )
    retro["source_group"] = "retrospective"
    retro["decision"] = retro["bucket"]

    prospective = pd.read_csv(
        RESULTS / "gmg_solver_floor_detector_prospective" / "prospective_combined.csv"
    )
    prospective["source_group"] = "prospective"
    prospective["decision"] = np.where(
        prospective["predicted_raise_floor"].astype(int) == 1,
        "predicted raise",
        "predicted keep",
    )

    transfer = pd.read_csv(
        RESULTS / "gmg_detector_transfer_summary" / "gmg_detector_transfer_summary.csv"
    )
    transfer = transfer.rename(
        columns={
            "category": "source_group",
            "r50": "probe_r50",
            "r100": "probe_r100",
            "r100_over_r50": "probe_r100_over_r50",
        }
    )
    transfer["decision"] = np.where(
        transfer["recommended_rho_min"].astype(float) <= 1e-11,
        "selected keep",
        "selected raise",
    )

    cols = [
        "probe_r50",
        "probe_r100",
        "probe_r100_over_r50",
        "source_group",
        "decision",
        "recommended_rho_min",
        "trigger",
    ]
    data = pd.concat(
        [
            retro[[c for c in cols if c in retro.columns]],
            prospective[[c for c in cols if c in prospective.columns]],
            transfer[[c for c in cols if c in transfer.columns]],
        ],
        ignore_index=True,
        sort=False,
    )
    data["probe_r50"] = pd.to_numeric(data["probe_r50"], errors="coerce")
    data["probe_r100_over_r50"] = pd.to_numeric(data["probe_r100_over_r50"], errors="coerce")
    data = data.dropna(subset=["probe_r50", "probe_r100_over_r50"])
    heldout = pd.read_csv(
        RESULTS
        / "heldout_gmg_detector_cantilever_s41_71_guarded_true_residual"
        / "prospective_summary.csv"
    )
    heldout["probe_r50"] = pd.to_numeric(heldout["probe_r50"], errors="coerce")
    heldout["probe_r100_over_r50"] = pd.to_numeric(heldout["probe_r100_over_r50"], errors="coerce")
    heldout_false_keeps = heldout[heldout["detector_false_keep"].astype(int) == 1].copy()

    fig, ax = plt.subplots(figsize=(7.5, 4.7))
    style = {
        "true_raise": ("o", COLORS["orange"], "retrospective true raise"),
        "true_keep": ("o", COLORS["teal"], "retrospective true keep"),
        "predicted raise": ("^", COLORS["orange"], "prospective raise"),
        "predicted keep": ("^", COLORS["teal"], "prospective keep"),
        "selected raise": ("s", COLORS["red"], "selected raise"),
        "selected keep": ("s", COLORS["blue"], "selected keep"),
    }
    for decision, group in data.groupby("decision"):
        marker, color, label = style.get(decision, ("o", COLORS["gray"], decision))
        ax.scatter(
            group["probe_r50"],
            group["probe_r100_over_r50"],
            s=52,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
            label=label,
        )
    x_min = 8e-6
    xs = np.logspace(np.log10(x_min), np.log10(0.8), 360)
    plateau_boundary = np.maximum(0.6, 1e-4 / xs)
    ax.axvspan(1e-2, 0.8, color=COLORS["orange"], alpha=0.08, label="raise: high $r_{50}$", zorder=0)
    ax.fill_between(
        xs,
        plateau_boundary,
        1.08,
        where=plateau_boundary < 1.08,
        color=COLORS["red"],
        alpha=0.07,
        label="raise: plateau + $r_{100}$",
        zorder=0,
    )
    ax.axvline(1e-2, color=COLORS["ink"], lw=1.0, ls="--")
    ax.axhline(0.6, color=COLORS["ink"], lw=1.0, ls=":")
    if not heldout_false_keeps.empty:
        false_keep_x = heldout_false_keeps["probe_r50"].to_numpy(dtype=float)
        false_keep_y = heldout_false_keeps["probe_r100_over_r50"].to_numpy(dtype=float)
        jitter = np.linspace(-0.18, 0.18, len(false_keep_x))
        y_offsets = np.linspace(0.00, 0.055, len(false_keep_y))
        false_keep_x_plot = false_keep_x * np.exp(jitter)
        false_keep_y_plot = np.clip(false_keep_y + y_offsets, 0.012, 1.04)
        ax.scatter(
            false_keep_x_plot,
            false_keep_y_plot,
            s=130,
            marker="X",
            facecolor="none",
            edgecolor=COLORS["red"],
            linewidth=1.8,
            label="held-out false keeps (n=4; jittered)",
            zorder=5,
        )
        for idx, (xv, yv) in enumerate(zip(false_keep_x_plot, false_keep_y_plot), start=1):
            ax.text(
                xv * 1.10,
                yv + 0.018,
                str(idx),
                color=COLORS["red"],
                fontsize=7,
                fontweight="bold",
                ha="left",
                va="bottom",
                zorder=6,
            )
        ax.text(
            1.1e-5,
            0.17,
            "false keeps\njittered",
            color=COLORS["red"],
            fontsize=7,
            ha="left",
            va="center",
        )
    ax.text(1.08e-2, 0.08, "$r_{50}=10^{-2}$", rotation=90, va="bottom", fontsize=8)
    ax.text(1.5e-4, 0.63, "$r_{100}/r_{50}=0.6$", va="bottom", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(x_min, 0.8)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("$r_{50}$ after baseline probe")
    ax.set_ylabel("$r_{100}/r_{50}$ plateau ratio")
    ax.set_title("Residual-probe decision map")
    handles, labels_seen = ax.get_legend_handles_labels()
    dedup: dict[str, object] = {}
    for handle, label in zip(handles, labels_seen):
        dedup.setdefault(label, handle)
    ax.legend(
        dedup.values(),
        dedup.keys(),
        frameon=False,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig3_residual_probe_phase_map.png", bbox_inches="tight")
    fig.savefig(OUT / "fig3_residual_probe_phase_map.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_history(ax, df: pd.DataFrame, title: str) -> None:
    df = df.copy()
    df["rho_min_float"] = df["rho_min"].astype(float)
    colors = {1e-12: COLORS["teal"], 1e-3: COLORS["blue"], 1e-2: COLORS["orange"]}
    labels = {1e-12: "$10^{-12}$ probe", 1e-3: "$10^{-3}$ attempt", 1e-2: "$10^{-2}$ attempt"}
    for rho, group in df.groupby("rho_min_float"):
        group = group.sort_values("iter")
        ax.plot(
            group["iter"],
            group["rel_residual"],
            color=colors.get(rho, COLORS["gray"]),
            lw=2.0,
            label=labels.get(rho, f"{rho:g}"),
        )
    ax.axhline(1e-6, color=COLORS["ink"], lw=1.0, ls="--", alpha=0.75)
    ax.set_yscale("log")
    ax.set_ylim(3e-7, 2)
    ax.set_xlim(left=0)
    ax.set_title(title)
    ax.set_xlabel("FGMRES iteration")
    ax.grid(axis="y", which="both", alpha=0.12)


def make_rescue_histories() -> None:
    bridge = pd.read_csv(
        RESULTS
        / "gmg_solver_floor_detector_transfer_bridge_seed23_ladder"
        / "prospective_history.csv"
    )
    b500 = pd.read_csv(
        STRICT_OPTIMIZED_BRIDGE_RESULTS / "density_detector_history.csv"
    )
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.8), sharey=True)
    cases = [
        (
            bridge[bridge["solid_probability"].astype(float) == 0.10],
            "Bridge random\n$q=0.10$",
        ),
        (
            bridge[bridge["solid_probability"].astype(float) == 0.20],
            "Bridge random\n$q=0.20$",
        ),
        (b500, "Optimized bridge\ndensity"),
    ]
    for ax, (df, title) in zip(axes, cases):
        _plot_history(ax, df, title)
    axes[0].set_ylabel("relative residual")
    handles, labels_seen = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels_seen,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
    )
    fig.suptitle("Baseline failure and ladder rescue histories", y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "fig4_failure_rescue_histories.png", bbox_inches="tight")
    fig.savefig(OUT / "fig4_failure_rescue_histories.pdf", bbox_inches="tight")
    plt.close(fig)


def make_direct_admissibility_atlas() -> None:
    critical = pd.read_csv(
        RESULTS / "direct_floor_atlas_seeded" / "direct_floor_critical.csv"
    )
    aggregate = pd.read_csv(
        RESULTS
        / "direct_floor_atlas_seeded"
        / "direct_floor_critical_aggregate.csv"
    )
    validation = pd.read_csv(
        RESULTS
        / "admissibility_detector_validation"
        / "detector_leave_one_seed_summary.csv"
    )

    seeds = sorted(critical["seed"].astype(int).unique())
    probs = sorted(critical["solid_probability"].astype(float).unique())
    heat = (
        critical.assign(
            seed=lambda d: d["seed"].astype(int),
            solid_probability=lambda d: d["solid_probability"].astype(float),
            log10_floor=lambda d: np.log10(d["critical_rho_min"].astype(float)),
        )
        .pivot(index="solid_probability", columns="seed", values="log10_floor")
        .reindex(index=probs, columns=seeds)
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.2))
    ax = axes[0]
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="YlOrBr", vmin=-12, vmax=-6)
    ax.set_xticks(np.arange(len(seeds)))
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_yticks(np.arange(len(probs)))
    ax.set_yticklabels([f"{p:.2g}" for p in probs])
    ax.set_xlabel("seed")
    ax.set_ylabel("solid probability")
    ax.set_title("Direct critical floor")
    for iy, prob in enumerate(probs):
        for ix, seed in enumerate(seeds):
            value = heat.loc[prob, seed]
            ax.text(
                ix,
                iy,
                f"{int(value)}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if value > -8.5 else COLORS["ink"],
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("$\\log_{10}(\\rho_{\\min}^{crit})$")
    ax.text(-0.18, 1.08, "a", transform=ax.transAxes, fontsize=13, fontweight="bold")

    ax = axes[1]
    x = np.arange(len(aggregate))
    ax.plot(
        x,
        np.log10(aggregate["min_critical_rho"].astype(float)),
        marker="o",
        color=COLORS["teal"],
        label="min across seeds",
    )
    ax.plot(
        x,
        np.log10(aggregate["median_critical_rho"].astype(float)),
        marker="s",
        color=COLORS["blue"],
        label="median",
    )
    ax.plot(
        x,
        np.log10(aggregate["max_critical_rho"].astype(float)),
        marker="^",
        color=COLORS["orange"],
        label="max",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.2g}" for p in aggregate["solid_probability"].astype(float)])
    ax.set_xlabel("solid probability")
    ax.set_ylabel("$\\log_{10}(\\rho_{\\min}^{crit})$")
    ax.set_title("Critical-floor envelope")
    ax.legend(frameon=False)
    ax.text(-0.18, 1.08, "b", transform=ax.transAxes, fontsize=13, fontweight="bold")

    ax = axes[2]
    val = validation[validation["mode"] == "conservative"].copy()
    val["safety_factor"] = val["safety_factor"].astype(float)
    width = 0.34
    x = np.arange(len(val))
    ax.bar(
        x - width / 2,
        val["false_admissible"].astype(int),
        width,
        color=COLORS["red"],
        label="unsafe false admissible",
    )
    ax.bar(
        x + width / 2,
        val["false_inadmissible"].astype(int),
        width,
        color=COLORS["blue"],
        label="conservative false inadmissible",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"{sf:g}" for sf in val["safety_factor"]])
    ax.set_xlabel("safety factor")
    ax.set_ylabel("leave-one-seed cases")
    ax.set_title("Direct detector tradeoff")
    ax.legend(frameon=False)
    ax.text(-0.18, 1.08, "c", transform=ax.transAxes, fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "fig5_direct_admissibility_atlas.png", bbox_inches="tight")
    fig.savefig(OUT / "fig5_direct_admissibility_atlas.pdf", bbox_inches="tight")
    plt.close(fig)


def make_transfer_outcome_summary() -> None:
    transfer = pd.read_csv(
        RESULTS / "gmg_detector_transfer_summary" / "gmg_detector_transfer_summary.csv"
    ).copy()
    transfer["recommended_rho_min"] = transfer["recommended_rho_min"].astype(float)
    transfer["solve_iters"] = transfer["solve_iters"].astype(float)
    transfer["solve_final_rel_residual"] = transfer["solve_final_rel_residual"].astype(float)
    category_order = [
        "cantilever random prospective",
        "bridge random geometry transfer",
        "optimized SIMP density",
    ]
    category_label = {
        "cantilever random prospective": "prospective\ncantilever",
        "bridge random geometry transfer": "bridge\ntransfer",
        "optimized SIMP density": "optimized\ndensity",
    }
    floor_order = [1e-12, 1e-3, 1e-2]
    floor_colors = {1e-12: COLORS["teal"], 1e-3: COLORS["blue"], 1e-2: COLORS["orange"]}

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    ax = axes[0]
    x_positions = []
    labels = []
    xpos = 0
    for category in category_order:
        subset = transfer[transfer["category"] == category].reset_index(drop=True)
        local_x = xpos + np.arange(len(subset))
        x_positions.extend(local_x)
        labels.extend(subset["case_id"].tolist())
        for floor in floor_order:
            group = subset[np.isclose(subset["recommended_rho_min"], floor)]
            if group.empty:
                continue
            gx = xpos + group.index.to_numpy()
            ax.scatter(
                gx,
                np.full(len(group), np.log10(floor)),
                s=58,
                color=floor_colors[floor],
                edgecolor="white",
                linewidth=0.6,
                label=_floor_label(floor),
            )
        ax.axvline(xpos - 0.5, color="#DDDDDD", lw=0.8)
        ax.text(
            xpos + (len(subset) - 1) / 2,
            -1.35,
            category_label[category],
            ha="center",
            va="top",
            fontsize=8,
        )
        xpos += len(subset) + 1
    ax.set_ylim(-12.6, -1.2)
    ax.set_yticks([-12, -3, -2])
    ax.set_yticklabels(["$10^{-12}$", "$10^{-3}$", "$10^{-2}$"])
    ax.set_ylabel("selected $\\rho_{\\min}$")
    ax.set_xticks([])
    ax.set_title("Selected floor for every transfer case")
    handles, labels_seen = ax.get_legend_handles_labels()
    dedup = dict(zip(labels_seen, handles))
    ax.legend(dedup.values(), dedup.keys(), frameon=False, loc="upper left", title="$\\rho_{\\min}$")
    ax.text(-0.12, 1.06, "a", transform=ax.transAxes, fontsize=13, fontweight="bold")

    ax = axes[1]
    width = 0.26
    x = np.arange(len(category_order))
    grouped = (
        transfer.groupby(["category", "recommended_rho_min"])["solve_iters"]
        .mean()
        .reset_index()
    )
    width = 0.22
    for offset, floor in zip([-width, 0.0, width], floor_order):
        vals = []
        for cat in category_order:
            mask = (grouped["category"] == cat) & np.isclose(
                grouped["recommended_rho_min"].astype(float), floor
            )
            vals.append(float(grouped.loc[mask, "solve_iters"].iloc[0]) if mask.any() else 0.0)
        ax.bar(
            x + offset,
            vals,
            width=width,
            color=floor_colors[floor],
            label=_floor_label(floor),
        )
        for xpos_i, val in zip(x + offset, vals):
            if val > 0:
                ax.text(xpos_i, val + 4.0, f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([category_label[c] for c in category_order])
    ax.set_ylabel("mean selected-solve iterations")
    ax.set_title("Where solver effort concentrates")
    ax.legend(frameon=False, title="$\\rho_{\\min}$")
    ax.text(-0.12, 1.06, "b", transform=ax.transAxes, fontsize=13, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT / "fig6_transfer_outcome_summary.png", bbox_inches="tight")
    fig.savefig(OUT / "fig6_transfer_outcome_summary.pdf", bbox_inches="tight")
    plt.close(fig)


def make_threshold_sensitivity() -> None:
    grid = pd.read_csv(
        RESULTS
        / "gmg_solver_floor_detector_sensitivity"
        / "gmg_threshold_sensitivity_grid.csv"
    )
    current = pd.read_csv(
        RESULTS
        / "gmg_solver_floor_detector_sensitivity"
        / "gmg_threshold_sensitivity_summary.csv"
    )
    current = current[current["selection"] == "current_rule"].iloc[0]
    plateau = float(current["plateau_residual_threshold"])
    subset = grid[np.isclose(grid["plateau_residual_threshold"].astype(float), plateau)].copy()

    high_values = sorted(subset["high_residual_threshold"].astype(float).unique())
    ratio_values = sorted(subset["plateau_ratio_threshold"].astype(float).unique())
    missed = (
        subset.pivot(
            index="plateau_ratio_threshold",
            columns="high_residual_threshold",
            values="false_keep",
        )
        .reindex(index=ratio_values, columns=high_values)
        .astype(float)
    )
    conservative = (
        subset.pivot(
            index="plateau_ratio_threshold",
            columns="high_residual_threshold",
            values="false_raise",
        )
        .reindex(index=ratio_values, columns=high_values)
        .astype(float)
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    specs = [
        (missed, "Unsafe missed raises", COLORS["red"]),
        (conservative, "Conservative false raises", COLORS["blue"]),
    ]
    for ax, (data, title, color) in zip(axes, specs):
        im = ax.imshow(data.to_numpy(), origin="lower", aspect="auto", cmap="Reds" if color == COLORS["red"] else "Blues")
        ax.set_title(title)
        ax.set_xlabel("high-$r_{50}$ threshold")
        ax.set_xticks(np.arange(len(high_values)))
        ax.set_xticklabels([f"{v:g}" for v in high_values], rotation=35, ha="right")
        ax.set_yticks(np.arange(len(ratio_values)))
        ax.set_yticklabels([f"{v:g}" for v in ratio_values])
        ax.set_ylabel("plateau-ratio threshold")
        for iy, ratio in enumerate(ratio_values):
            for ix, high in enumerate(high_values):
                value = int(data.loc[ratio, high])
                ax.text(ix, iy, str(value), ha="center", va="center", fontsize=7, color=COLORS["ink"])
        current_x = high_values.index(float(current["high_residual_threshold"]))
        current_y = ratio_values.index(float(current["plateau_ratio_threshold"]))
        ax.scatter([current_x], [current_y], marker="s", s=90, facecolors="none", edgecolors=COLORS["ink"], linewidths=1.4)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Threshold sensitivity at plateau residual threshold $10^{-4}$", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "fig8_threshold_sensitivity.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig8_threshold_sensitivity.png", bbox_inches="tight")
    plt.close(fig)


def make_policy_overhead() -> None:
    cases = pd.read_csv(RESULTS / "gmg_policy_overhead" / "gmg_policy_overhead_cases.csv")
    cases = cases.copy()
    cases["selected_floor"] = cases["selected_floor"].astype(float)
    cases["case_label"] = cases["case_id"].map(_reader_case_label)

    fig, ax = plt.subplots(figsize=(10.6, 4.9))
    x = np.arange(len(cases))
    selected = cases["selected_solve_iters"].astype(float).to_numpy()
    probe = cases["probe_iters"].astype(float).to_numpy()
    failed = cases["failed_ladder_iters"].astype(float).to_numpy()
    ax.bar(x, selected, color=COLORS["blue"], label="selected solve")
    ax.bar(x, probe, bottom=selected, color=COLORS["sand"], label="baseline probe")
    ax.bar(x, failed, bottom=selected + probe, color=COLORS["red"], label="failed ladder attempts")
    ax.set_ylabel("FGMRES iterations")
    ax.set_title("Iteration-equivalent policy overhead by transfer case")
    ax.set_xticks([])
    ax.set_xlabel("transfer cases ordered by category")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    for boundary in np.where(cases["category"].ne(cases["category"].shift()).to_numpy())[0][1:]:
        ax.axvline(boundary - 0.5, color="#DDDDDD", lw=0.8)
    ymax = max(selected + probe + failed)
    ax.set_ylim(0, ymax * 1.12)
    category_labels = {
        "cantilever random prospective": "prospective cantilever",
        "bridge random geometry transfer": "bridge transfer",
        "optimized SIMP density": "optimized density",
    }
    for category, group in cases.groupby("category", sort=False):
        mid = float(group.index.to_numpy().mean())
        ax.text(
            mid,
            -0.08,
            category_labels.get(category, category),
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8,
        )
    severe = cases[failed > 0].copy()
    for idx, row in severe.iterrows():
        total = float(row["selected_solve_iters"]) + float(row["probe_iters"]) + float(row["failed_ladder_iters"])
        label = str(row["case_label"]).replace("\n", " ")
        ax.text(
            idx,
            total + ymax * 0.025,
            label,
            ha="center",
            va="bottom",
            fontsize=7.2,
            rotation=28,
            color=COLORS["red"],
        )
    fig.tight_layout()
    fig.savefig(OUT / "fig9_policy_overhead.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig9_policy_overhead.png", bbox_inches="tight")
    plt.close(fig)


def make_fixed_floor_controls() -> None:
    controls = pd.read_csv(
        STRICT_FIXED_FLOOR_RESULTS / "fixed_floor_control_comparison.csv"
    )
    sensitivity = pd.read_csv(
        REVIEW_SUMMARY_RESULTS / "sensitivity_perturbation_summary.csv"
    )
    sensitivity_cases = pd.read_csv(
        REVIEW_SUMMARY_RESULTS / "sensitivity_perturbation_combined.csv"
    )
    controls = controls[controls["rho_min"].astype(float) > 1e-11].copy()
    controls["rho_min"] = controls["rho_min"].astype(float)
    controls["percent_change"] = 100.0 * controls["relative_compliance_change"].astype(float).abs()
    controls["case_label"] = controls["case_id"].map(_reader_case_label)

    case_order = [
        "cant_s23_p035",
        "cant_s31_p035",
        "C64_MF",
        "C512_MF",
        "Brk500_MF",
    ]
    controls["case_id"] = pd.Categorical(controls["case_id"], categories=case_order, ordered=True)
    controls = controls.sort_values(["case_id", "rho_min"])

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8.8, 5.9),
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.42},
    )
    ax = axes[0]
    x = np.arange(len(case_order))
    width = 0.34
    colors = {1e-3: COLORS["blue"], 1e-2: COLORS["orange"]}
    labels = {1e-3: "$10^{-3}$", 1e-2: "$10^{-2}$"}
    for offset, floor in [(-width / 2, 1e-3), (width / 2, 1e-2)]:
        vals = []
        for case in case_order:
            row = controls[(controls["case_id"] == case) & np.isclose(controls["rho_min"], floor)]
            vals.append(float(row["percent_change"].iloc[0]) if not row.empty else 0.0)
        ax.bar(x + offset, vals, width, color=colors[floor], label=labels[floor])
    ax.axhline(0.0, color=COLORS["ink"], lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([_reader_case_label(c) for c in case_order], fontsize=8)
    ax.set_ylabel("absolute compliance change (%)")
    ax.set_title("Compliance perturbation in fixed-floor controls")
    ax.set_yscale("log")
    ax.set_ylim(0.006, 35.0)
    ax.set_yticks([1e-2, 1e-1, 1.0, 10.0])
    ax.set_yticklabels(["0.01", "0.1", "1", "10"])
    ax.legend(frameon=False, title="$\\rho_{\\min}$")

    ax = axes[1]
    sensitivity["rho_min"] = sensitivity["rho_min"].astype(float)
    sensitivity = sensitivity.sort_values("rho_min")
    sensitivity_cases["rho_min"] = sensitivity_cases["rho_min"].astype(float)
    sensitivity_cases = sensitivity_cases[sensitivity_cases["rho_min"] > 1e-11].copy()
    distributions = [
        sensitivity_cases[np.isclose(sensitivity_cases["rho_min"], floor)]["rel_dc_l2_solid"].astype(float).to_numpy()
        for floor in [1e-3, 1e-2]
    ]
    parts = ax.boxplot(
        distributions,
        widths=0.48,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COLORS["ink"], "linewidth": 1.2},
        boxprops={"linewidth": 1.0, "color": COLORS["ink"]},
        whiskerprops={"color": COLORS["ink"], "linewidth": 0.9},
        capprops={"color": COLORS["ink"], "linewidth": 0.9},
    )
    for patch, floor in zip(parts["boxes"], [1e-3, 1e-2]):
        patch.set_facecolor(colors[floor])
        patch.set_alpha(0.78)
    rng = np.random.default_rng(7)
    for xpos, values, floor in zip([1, 2], distributions, [1e-3, 1e-2]):
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(
            np.full(len(values), xpos) + jitter,
            values,
            s=14,
            color=colors[floor],
            edgecolor="white",
            linewidth=0.35,
            alpha=0.68,
            zorder=3,
        )
    for xpos, row in zip([1, 2], sensitivity.itertuples(index=False)):
        ax.text(
            xpos,
            1.06,
            f"mean {float(row.mean_rel_dc_l2_solid):.3f}\nmax {float(row.max_rel_dc_l2_solid):.3f}",
            ha="center",
            va="top",
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.5},
        )
    ax.set_xticks([1, 2])
    ax.set_xticklabels([_floor_label(1e-3), _floor_label(1e-2)])
    ax.set_ylabel("relative $\\ell_2$ shift")
    ax.set_title("Solid-element sensitivity perturbation on 24 true-keep states")
    ax.set_ylim(0.0, 1.10)
    fig.tight_layout()
    fig.savefig(OUT / "fig10_fixed_floor_controls.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig10_fixed_floor_controls.png", bbox_inches="tight")
    plt.close(fig)


def _parse_floor_counts(counts: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for item in str(counts).split(";"):
        item = item.strip()
        if not item or ":" not in item:
            continue
        key, value = item.split(":", 1)
        parsed[key.strip()] = int(float(value.strip()))
    return parsed


def _ordered_floor_keys(rows: pd.DataFrame) -> list[str]:
    seen: set[str] = set()
    keys: list[str] = []
    for counts in rows["recommended_floor_counts"]:
        for key in _parse_floor_counts(counts):
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def _plot_floor_stack(ax, rows: pd.DataFrame, label_col: str, title: str) -> None:
    floor_keys = _ordered_floor_keys(rows)
    palette = {
        "1e-12": COLORS["blue"],
        "1e-09": "#7397B8",
        "1e-08": "#86A7A0",
        "1e-06": "#8FBA8C",
        "1e-03": COLORS["orange"],
        "1e-02": COLORS["red"],
    }
    label_map = {
        "1e-12": r"$10^{-12}$",
        "1e-09": r"$10^{-9}$",
        "1e-08": r"$10^{-8}$",
        "1e-06": r"$10^{-6}$",
        "1e-03": r"$10^{-3}$",
        "1e-02": r"$10^{-2}$",
    }
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows), dtype=float)
    for floor in floor_keys:
        values = np.array(
            [_parse_floor_counts(c).get(floor, 0) for c in rows["recommended_floor_counts"]],
            dtype=float,
        )
        ax.bar(
            x,
            values,
            bottom=bottom,
            width=0.66,
            color=palette.get(floor, COLORS["gray"]),
            edgecolor="white",
            linewidth=0.8,
            label=label_map.get(floor, floor),
        )
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in rows[label_col]], rotation=0)
    ax.set_ylim(0, max(13.4, float(bottom.max()) + 1.4))
    ax.set_ylabel("cases")
    ax.set_title(title)
    ax.legend(
        frameon=False,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=min(3, len(floor_keys)),
        title="selected floor",
        title_fontsize=8,
    )
    for i, row in rows.iterrows():
        xpos = list(rows.index).index(i)
        ax.text(
            xpos,
            bottom[xpos] + 0.15,
            f"{int(row['passes'])}/{int(row['rows'])}",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def make_review_extension_evidence() -> None:
    mechanism = pd.read_csv(REVIEW_SUMMARY_RESULTS / "mechanism_ablation_summary.csv")
    simp = pd.read_csv(REVIEW_SUMMARY_RESULTS / "simp_exponent_policy_sensitivity_summary.csv")
    floors = pd.read_csv(REVIEW_SUMMARY_RESULTS / "original_floor_policy_sensitivity_summary.csv")

    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.7))

    ax = axes[0]
    order = [
        "canonical",
        "levels3",
        "jacobi_smoother",
        "w_cycle",
        "tol1em5",
        "tol1em7",
        "no_root_correction",
    ]
    mechanism = mechanism.set_index("stack_variant").loc[order].reset_index()
    y = np.arange(len(mechanism))
    passes = mechanism["passes"].astype(float).to_numpy()
    failures = mechanism["failures"].astype(float).to_numpy()
    labels = [
        "reported stack",
        "3-level hierarchy",
        "Jacobi smoother",
        "W-cycle",
        r"tol $10^{-5}$",
        r"tol $10^{-7}$",
        "without root\ncorrection",
    ]
    ax.barh(y, passes, color=COLORS["teal"], edgecolor="white", label="converged")
    ax.barh(y, failures, left=passes, color=COLORS["red"], edgecolor="white", label="failed")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 6.4)
    ax.set_xlabel("selected solves")
    ax.set_title("Stack-mechanism ablation")
    for yi, p, f in zip(y, passes, failures):
        ax.text(6.12, yi, f"{int(p)}/6", ha="right", va="center", fontsize=8)
    ax.invert_yaxis()

    ax = axes[1]
    simp = simp.copy()
    simp["penal"] = simp["penal"].map(lambda v: f"p={float(v):g}")
    _plot_floor_stack(ax, simp, "penal", "Tuned SIMP-exponent sensitivity")

    ax = axes[2]
    floors = floors.copy()
    floors["baseline_rho_min"] = floors["baseline_rho_min"].map(
        lambda v: {"1e-12": r"$10^{-12}$", "1e-09": r"$10^{-9}$", "1e-08": r"$10^{-8}$", "1e-06": r"$10^{-6}$"}.get(str(v), str(v))
    )
    _plot_floor_stack(ax, floors, "baseline_rho_min", "Tuned original-floor sensitivity")

    for label, ax in zip(["a", "b", "c"], axes.flat):
        ax.text(
            -0.12,
            1.07,
            label,
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="top",
        )

    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.21, top=0.88, wspace=0.42)
    fig.savefig(OUT / "fig11_review_extension_evidence.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig11_review_extension_evidence.png", bbox_inches="tight")
    plt.close(fig)


def _downsample_max(volume: np.ndarray, target: int = 42) -> np.ndarray:
    factors = [max(1, int(np.ceil(s / target))) for s in volume.shape]
    trimmed = tuple((s // f) * f for s, f in zip(volume.shape, factors))
    volume = volume[: trimmed[0], : trimmed[1], : trimmed[2]]
    reshaped = volume.reshape(
        trimmed[0] // factors[0],
        factors[0],
        trimmed[1] // factors[1],
        factors[1],
        trimmed[2] // factors[2],
        factors[2],
    )
    return reshaped.max(axis=(1, 3, 5))


def _crop_white_margin(image: np.ndarray, pad: int = 64) -> np.ndarray:
    mask = np.any(image < 248, axis=2)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return image
    r0 = max(0, rows[0] - pad)
    r1 = min(image.shape[0], rows[-1] + pad + 1)
    c0 = max(0, cols[0] - pad)
    c1 = min(image.shape[1], cols[-1] + pad + 1)
    return image[r0:r1, c0:c1]


def _pad_image(image: np.ndarray, y_frac: float = 0.16, x_frac: float = 0.08) -> np.ndarray:
    height, width = image.shape[:2]
    y_pad = max(1, int(height * y_frac))
    x_pad = max(1, int(width * x_frac))
    white = 1.0 if np.issubdtype(image.dtype, np.floating) else np.iinfo(image.dtype).max
    padded = np.full((height + 2 * y_pad, width + 2 * x_pad, image.shape[2]), white, dtype=image.dtype)
    padded[y_pad : y_pad + height, x_pad : x_pad + width] = image
    return padded


def _render_marching_surface_image(
    density: np.ndarray,
    color: str,
    camera_multipliers: tuple[float, float, float] = (1.75, -1.90, 1.20),
    camera_zoom: float = 1.03,
    parallel_scale_factor: float = 0.42,
    window_size: tuple[int, int] = (1000, 620),
) -> np.ndarray:
    if pv is None:
        raise RuntimeError("pyvista is required to regenerate the 3D topology gallery")
    density = np.asarray(density, dtype=np.float32)
    padded = np.pad(density, 1, mode="constant", constant_values=0)
    verts, faces, _, _ = measure.marching_cubes(
        padded,
        level=0.5,
        spacing=(1.0, 1.0, 1.0),
        step_size=1,
        allow_degenerate=False,
    )
    verts -= 1.0
    faces_pv = np.column_stack([np.full(len(faces), 3), faces]).astype(np.int64).ravel()
    mesh = pv.PolyData(verts, faces_pv).clean()
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")
    try:
        plotter.enable_anti_aliasing("ssaa")
    except Exception:
        plotter.enable_anti_aliasing()
    plotter.add_mesh(
        mesh,
        color=color,
        smooth_shading=True,
        show_edges=False,
        ambient=0.30,
        diffuse=0.76,
        specular=0.14,
        specular_power=18,
    )
    center = np.array(density.shape, dtype=float) / 2.0
    view = np.array(camera_multipliers) * np.array(density.shape, dtype=float)
    camera = (
        center + view,
        center,
        (0.0, 0.0, 1.0),
    )
    plotter.camera_position = camera
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = parallel_scale_factor * max(density.shape)
    plotter.camera.zoom(camera_zoom)
    image = plotter.screenshot(return_img=True)
    plotter.clear()
    plotter.close()
    pv.close_all()
    del plotter
    del mesh
    gc.collect()
    return _crop_white_margin(image)


def _plot_marching_surface_matplotlib(
    ax,
    density: np.ndarray,
    color: str,
    view: tuple[float, float] = (22.0, -58.0),
) -> None:
    density = np.asarray(density, dtype=np.float32)
    display_density = _downsample_max(density, target=90)
    padded = np.pad(display_density, 1, mode="constant", constant_values=0)
    verts, faces, _, _ = measure.marching_cubes(
        padded,
        level=0.5,
        spacing=(1.0, 1.0, 1.0),
        step_size=1,
        allow_degenerate=False,
    )
    verts -= 1.0
    mesh = Poly3DCollection(verts[faces], facecolor=color, edgecolor="none", linewidths=0.0, alpha=1.0)
    mesh.set_antialiased(True)
    ax.add_collection3d(mesh)
    pad = 0.12 * max(display_density.shape)
    ax.set_xlim(-pad, display_density.shape[0] + pad)
    ax.set_ylim(-pad, display_density.shape[1] + pad)
    ax.set_zlim(-pad, display_density.shape[2] + pad)
    ax.set_box_aspect(display_density.shape)
    ax.set_proj_type("ortho")
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_axis_off()


def _render_marching_surface_matplotlib_image(
    density: np.ndarray,
    color: str,
    view: tuple[float, float] = (22.0, -58.0),
) -> np.ndarray:
    fig = plt.figure(figsize=(4.3, 2.6), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    _plot_marching_surface_matplotlib(ax, density, color, view=view)
    ax.set_position([0.0, 0.0, 1.0, 1.0])
    fig.patch.set_facecolor("white")
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", facecolor="white", pad_inches=0.02)
    plt.close(fig)
    buffer.seek(0)
    image = plt.imread(buffer)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return _crop_white_margin(image, pad=24)


def make_3d_topology_gallery() -> None:
    # The manifest records matching SHA256 hashes for rho_best.npy and rho_final.npy
    # for these fixed optimized-density states. The gallery uses rho_final.npy as the
    # canonical stored final-state path while representing the same density arrays.
    cases = [
        (
            "C64_MF",
            ROOT / "experiments" / "paper2" / "runs" / "C64_MF" / "rho_final.npy",
            (80, 40, 20),
            "64k-element cantilever\nkept $10^{-12}$",
            COLORS["teal"],
            (1.75, -1.90, 1.20),
            1.03,
            0.42,
            (20.0, -58.0),
        ),
        (
            "C512_MF",
            ROOT / "experiments" / "paper2" / "runs" / "C512_MF" / "rho_final.npy",
            (160, 80, 40),
            "512k-element cantilever\nkept $10^{-12}$",
            COLORS["blue"],
            (1.75, -1.90, 1.20),
            1.03,
            0.42,
            (20.0, -58.0),
        ),
        (
            "Brk500_MF",
            ROOT / "experiments" / "paper2" / "runs" / "Brk500_MF" / "rho_final.npy",
            (80, 160, 40),
            "512k-element bracket\nkept $10^{-12}$",
            COLORS["teal"],
            (2.30, -0.55, 1.15),
            0.84,
            0.43,
            (20.0, -20.0),
        ),
        (
            "B500_MF",
            ROOT / "experiments" / "paper2" / "runs" / "B500_MF" / "rho_final.npy",
            (210, 70, 35),
            "514.5k-element bridge\nraised to $10^{-2}$",
            COLORS["orange"],
            (1.45, -1.45, 3.20),
            0.76,
            0.42,
            (24.0, -45.0),
        ),
    ]

    fig, axes_grid = plt.subplots(2, 2, figsize=(8.6, 6.2))
    axes = list(axes_grid.flat)

    for i, (
        name,
        path,
        shape,
        title,
        color,
        camera_multipliers,
        camera_zoom,
        parallel_scale_factor,
        fallback_view,
    ) in enumerate(cases, start=1):
        ax = axes[i - 1]
        density = np.load(path).reshape(shape)
        if pv is None:
            image = _render_marching_surface_matplotlib_image(density, color, view=fallback_view)
        else:
            image = _render_marching_surface_image(
                density,
                color,
                camera_multipliers=camera_multipliers,
                camera_zoom=camera_zoom,
                parallel_scale_factor=parallel_scale_factor,
            )
        ax.imshow(_pad_image(image, y_frac=0.095, x_frac=0.04))
        ax.set_axis_off()
        ax.text(
            0.02,
            0.95,
            chr(ord("a") + i - 1),
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="top",
        )
        ax.text(
            0.50,
            0.98,
            title,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )
        ax.text(
            0.50,
            0.055,
            "density $\\geq 0.5$ isosurface\nBC/load: Appendix A",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=7,
            color=COLORS["gray"],
        )
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.04, top=0.97, wspace=0.035, hspace=0.16)
    fig.savefig(OUT / "fig7_3d_topology_transfer_gallery.png", facecolor="white")
    fig.savefig(OUT / "fig7_3d_topology_transfer_gallery.pdf", facecolor="white")
    plt.close(fig)


def relabel_existing_3d_topology_gallery() -> None:
    """Use the preserved rendered gallery when PyVista is unavailable, but replace labels."""
    source = OUT / "fig7_3d_topology_transfer_gallery.png"
    if not source.exists():
        raise RuntimeError("pyvista is unavailable and the existing 3D topology gallery is missing")
    image = plt.imread(source)
    labels = [
        "64k-element cantilever\nkept $10^{-12}$",
        "512k-element cantilever\nkept $10^{-12}$",
        "512k-element bracket\nkept $10^{-12}$",
        "514.5k-element bridge\nraised to $10^{-2}$",
    ]
    letters = ["a", "b", "c", "d"]
    fig, ax = plt.subplots(figsize=(8.6, 5.1))
    ax.imshow(image)
    ax.set_axis_off()
    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)

    # Cover only the old text bands while preserving the rendered surfaces.
    label_boxes = [
        (0.00, 0.02, 0.48, 0.18),
        (0.46, 0.02, 0.50, 0.18),
        (0.00, 0.49, 0.48, 0.18),
        (0.46, 0.49, 0.50, 0.18),
    ]
    for (x, y, w, h) in label_boxes:
        ax.add_patch(
            plt.Rectangle(
                (x * image.shape[1], y * image.shape[0]),
                w * image.shape[1],
                h * image.shape[0],
                facecolor="white",
                edgecolor="none",
                zorder=2,
            )
        )

    label_positions = [
        (0.27, 0.08),
        (0.73, 0.08),
        (0.27, 0.56),
        (0.73, 0.56),
    ]
    letter_positions = [
        (0.09, 0.07),
        (0.55, 0.07),
        (0.09, 0.55),
        (0.55, 0.55),
    ]
    for letter, label, (lx, ly), (tx, ty) in zip(letters, labels, letter_positions, label_positions):
        ax.text(
            lx * image.shape[1],
            ly * image.shape[0],
            letter,
            fontsize=13,
            fontweight="bold",
            ha="left",
            va="top",
            zorder=3,
        )
        ax.text(
            tx * image.shape[1],
            ty * image.shape[0],
            label,
            ha="center",
            va="top",
            fontsize=10,
            zorder=3,
        )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig(OUT / "fig7_3d_topology_transfer_gallery.png", facecolor="white")
    fig.savefig(OUT / "fig7_3d_topology_transfer_gallery.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _set_style()
    make_policy_schematic()
    make_evidence_matrix()
    make_residual_phase_map()
    make_rescue_histories()
    make_direct_admissibility_atlas()
    make_transfer_outcome_summary()
    make_threshold_sensitivity()
    make_policy_overhead()
    make_fixed_floor_controls()
    make_review_extension_evidence()
    make_3d_topology_gallery()
    print(f"Wrote figures to {OUT}")


if __name__ == "__main__":
    main()

