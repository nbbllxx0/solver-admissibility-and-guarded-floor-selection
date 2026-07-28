"""Figure generation for the v3 solver-admissibility manuscript.

The v3 figure set is the v2 set (six main-text figures, four supplementary
figures) plus a problem-setup figure, with the referee-driven corrections
listed in ``UPGRADE_NOTES_v3.md``. All figures are regenerated from the same
tracked CSV artifacts used by the v1/v2 scripts; no numbers are re-derived by
hand. The problem-setup figure is drawn directly from the solver presets in
``src/gpu_fem/presets.py`` and from the documented random-field recipe, so it
cannot drift from the runs.

Semantic colour convention used everywhere in the set:
    teal    -> original floor preserved (keep)
    blue    -> escalated to 1e-3
    orange  -> escalated to 1e-2
    red     -> unguarded / unaccepted / failed
    gray    -> reference or neutral annotation
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "phase5" / "results"
REVIEW = RESULTS / "review_experiment_summary"
OUT = Path(os.environ.get("PAPER5_FIGURE_OUT", str(ROOT / "figures" / "generated")))
# Optional: a directory holding an already generated v1 figure set. When it is
# present the two unchanged supplementary panels are copied from it; otherwise
# they are regenerated from the v1 script.
V1_FIGS = Path(os.environ.get("PAPER5_V1_FIGURES", str(OUT)))

C = {
    "ink": "#172126",
    "keep": "#3B9C8C",      # teal   - original floor preserved
    "r3": "#2F6F9F",        # blue   - escalated to 1e-3
    "r2": "#C56B2C",        # orange - escalated to 1e-2
    "bad": "#B64B3A",       # red    - unguarded / unaccepted
    "sand": "#E7D8B8",
    "pale": "#F5F0E6",
    "gray": "#697A82",
}
FLOOR_COLOR = {1e-12: C["keep"], 1e-3: C["r3"], 1e-2: C["r2"]}


def _style() -> None:
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
            "axes.edgecolor": C["ink"],
            "axes.labelcolor": C["ink"],
            "xtick.color": C["ink"],
            "ytick.color": C["ink"],
            "text.color": C["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _panel(ax, letter, x=-0.02, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12, fontweight="bold",
            ha="left", va="bottom")


def _save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def _box(ax, xy, w, h, text, fc, ec=None, fontsize=8.2, lw=1.2):
    ax.add_patch(
        FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                       fc=fc, ec=ec or C["ink"], lw=lw)
    )
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize)


def _arrow(ax, start, end, color=None, ls="-"):
    ax.add_patch(
        FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, lw=1.2,
                        color=color or C["ink"], shrinkA=2, shrinkB=2, linestyle=ls)
    )


def _stage(ax, x0, x1, y0, h, title, sub, accent, *, emphasis=False, fill="#FFFFFF"):
    """One flowchart stage: light box, coloured accent bar, title and subtitle."""
    lw = 1.6 if emphasis else 1.0
    ec = accent if emphasis else C["ink"]
    ax.add_patch(Rectangle((x0, y0), x1 - x0, h, facecolor=fill, edgecolor=ec,
                           lw=lw, zorder=2))
    ax.add_patch(Rectangle((x0, y0), 1.6, h, facecolor=accent, edgecolor="none", zorder=3))
    cx = (x0 + x1) / 2 + 0.8
    ax.text(cx, y0 + h * 0.66, title, ha="center", va="center", fontsize=8.4,
            fontweight="semibold", zorder=4)
    if sub:
        ax.text(cx, y0 + h * 0.30, sub, ha="center", va="center", fontsize=7.4,
                color=C["gray"], zorder=4, linespacing=1.35)
    return (x0, x1, y0, y0 + h)


def _diamond(ax, cx, cy, hw, hh, text):
    ax.add_patch(Polygon([(cx, cy + hh), (cx + hw, cy), (cx, cy - hh), (cx - hw, cy)],
                         closed=True, facecolor="#FFFFFF", edgecolor=C["ink"], lw=1.0,
                         zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=8.0, zorder=4,
            linespacing=1.3)


def _elbow(ax, pts, color, ls="-"):
    """Orthogonal polyline with a single arrowhead on the final segment."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs[:-1], ys[:-1], color=color, lw=1.2, ls=ls, solid_capstyle="butt", zorder=1)
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, linestyle=ls,
                                shrinkA=0, shrinkB=0), zorder=1)


# --------------------------------------------------------------------------
# Figure 0: problem setup and an example frozen state
# --------------------------------------------------------------------------
def _box_edges(ax, Lx, Ly, Lz, color, lw=0.9):
    pts = np.array([[0, 0, 0], [Lx, 0, 0], [Lx, Ly, 0], [0, Ly, 0],
                    [0, 0, Lz], [Lx, 0, Lz], [Lx, Ly, Lz], [0, Ly, Lz]], dtype=float)
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    for i, j in edges:
        # clip_on=False: with a tight box aspect the projected corners can fall
        # outside the axes rectangle, which visibly amputates the wireframe
        ax.plot(*zip(pts[i], pts[j]), color=color, lw=lw, zorder=1, clip_on=False)


def _domain_axes(ax, Lx, Ly, Lz, title, sub):
    ax.set_box_aspect((Lx, Ly, Lz), zoom=1.0)
    ax.set_axis_off()
    ax.view_init(elev=22, azim=-56)
    ax.set_xlim(-0.02 * Lx, 1.02 * Lx)
    ax.set_ylim(-0.10 * Ly, 1.35 * Ly)
    ax.set_zlim(-0.02 * Lz, 1.02 * Lz)
    ax.set_title(title, fontsize=9.0, pad=-10.0)
    ax.text2D(0.5, 0.04, sub, transform=ax.transAxes, ha="center", va="top",
              fontsize=7.2, color=C["gray"], linespacing=1.4)


def fig1_problem_setup() -> None:
    """Domains, supports and loads for the two families, and one frozen state.

    Geometry, mesh, volume fraction, supports and loads are transcribed from
    ``cantilever_gpu_medium`` and ``bridge_gpu_medium`` in
    ``src/gpu_fem/presets.py``; the density fields are regenerated with the
    recipe used by the experiment drivers.
    """
    fig = plt.figure(figsize=(8.6, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.02], hspace=0.30, wspace=0.14)

    # (a) cantilever, 2.0 x 1.0 x 0.5, 80 x 40 x 20, left face fixed, tip load
    # The mesh axes are (x, y, z); y is the load direction, drawn upward here.
    ax = fig.add_subplot(gs[0, 0], projection="3d")
    Lx, Ly, Lz = 2.0, 1.0, 0.5
    _box_edges(ax, Lx, Ly, Lz, C["ink"])
    yy, zz = np.meshgrid(np.linspace(0, Ly, 2), np.linspace(0, Lz, 2))
    ax.plot_surface(np.zeros_like(yy), yy, zz, color=C["gray"], alpha=0.40,
                    edgecolor="none", zorder=2)
    ax.quiver(Lx, Ly / 2, Lz / 2, 0, -0.42, 0, color=C["bad"], lw=1.8,
              arrow_length_ratio=0.30, zorder=5, clip_on=False)
    ax.text2D(0.995, 0.19, "$f_y=-1$", color=C["bad"], fontsize=7.6, ha="right",
              transform=ax.transAxes, zorder=8)
    ax.text2D(0.03, 0.74, "fixed face", color=C["gray"], fontsize=7.6, ha="left",
              transform=ax.transAxes)
    _domain_axes(ax, Lx, Ly, Lz, "Cantilever",
                 "$2.0\\times1.0\\times0.5$, $80\\times40\\times20 = 64{,}000$ elements\n"
                 "left face fully fixed; unit downward tip load; $V^*=0.30$")
    ax.text2D(0.0, 0.92, "a", transform=ax.transAxes, fontsize=12, fontweight="bold")

    # (b) bridge, 3.0 x 1.0 x 0.5, 90 x 30 x 15, pinned/roller feet, top pressure
    ax = fig.add_subplot(gs[0, 1], projection="3d")
    Lx, Ly, Lz = 3.0, 1.0, 0.5
    _box_edges(ax, Lx, Ly, Lz, C["ink"])
    for xc, marker in [(0.0, "^"), (Lx, "o")]:
        for zc in (0.0, 0.25, 0.5):
            ax.scatter([xc], [0.0], [zc], marker=marker, s=30, color=C["gray"],
                       edgecolor=C["ink"], lw=0.6, zorder=6, clip_on=False)
    for xc in np.linspace(0.20, Lx - 0.20, 7):
        ax.quiver(xc, Ly + 0.26, Lz / 2, 0, -0.26, 0, color=C["bad"], lw=1.1,
                  arrow_length_ratio=0.38, zorder=5, clip_on=False)
    ax.text2D(0.52, 0.82, "distributed top load", color=C["bad"], fontsize=7.6,
              ha="center", transform=ax.transAxes, zorder=8,
              bbox=dict(fc="white", ec="none", pad=0.5))
    ax.text2D(0.02, 0.28, "3 fixed\nnodes ($\\blacktriangle$)", color=C["gray"],
              fontsize=7.4, ha="left", va="center", linespacing=1.25,
              transform=ax.transAxes, zorder=8)
    ax.text2D(1.0, 0.18, "3 roller nodes ($\\bullet$)", color=C["gray"],
              fontsize=7.4, ha="right", va="center",
              transform=ax.transAxes, zorder=8)
    _domain_axes(ax, Lx, Ly, Lz, "Bridge",
                 "$3.0\\times1.0\\times0.5$, $90\\times30\\times15 = 40{,}500$ elements\n"
                 "pinned and roller feet at $y=0$; unit top pressure; $V^*=0.30$")
    ax.text2D(0.0, 0.92, "b", transform=ax.transAxes, fontsize=12, fontweight="bold")

    # (c, d) two frozen Bernoulli states of the held-out cantilever family.
    # Frame and outcome-line colours follow the paper's semantic convention:
    # blue = escalated to 1e-3, red = reported converged but true residual failed.
    nelx, nely, nelz = 80, 40, 20
    seed = 47
    u = np.random.default_rng(seed).random(nelx * nely * nelz).reshape(nelx, nely, nelz)
    cases = (("c", 0, 0.10, C["r3"],
              "inadmissible at $\\rho_0$; escalated to $10^{-3}$"),
             ("d", 1, 0.35, C["bad"],
              "stopping test passed at $\\rho_0$; true residual $1.45\\tau$"))
    for letter, col, q, accent, decision in cases:
        ax = fig.add_subplot(gs[1, col])
        _panel(ax, letter, x=-0.02, y=1.02)
        solid = (u < q)[:, :, nelz // 2]
        ax.imshow(solid.T, origin="lower", cmap="Greys", vmin=0, vmax=1.35,
                  interpolation="nearest", aspect="equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_edgecolor(accent)
            s.set_linewidth(1.6)
        ax.set_title(f"Held-out cantilever state: seed {seed}, $q={q:.2f}$", fontsize=8.8)
        ax.set_xlabel(f"slice solid fraction {solid.mean():.3f}",
                      fontsize=7.4, color=C["gray"], labelpad=3)
        ax.text(0.5, -0.185, decision, transform=ax.transAxes, ha="center",
                va="top", fontsize=7.4, color=accent)
    _save(fig, "fig1_problem_setup")


# --------------------------------------------------------------------------
# Figure 1: policy schematic + the acceptance-guard evidence
# --------------------------------------------------------------------------
def fig2_policy_and_guard() -> None:
    strict = RESULTS / "heldout_gmg_detector_cantilever_s41_71_strict_true_residual"
    summary = pd.read_csv(strict / "prospective_summary.csv")
    history = pd.read_csv(strict / "prospective_history.csv")
    guarded = pd.read_csv(REVIEW / "heldout_false_keeps.csv")

    cases = [(47, 0.35), (59, 0.30), (71, 0.30), (71, 0.35)]
    rows = []
    for seed, q in cases:
        srow = summary[(summary.seed == seed) & (np.isclose(summary.solid_probability, q))].iloc[0]
        hh = history[(history.seed == seed) & (np.isclose(history.solid_probability, q))
                     & (history.phase == "recommended_solve")].sort_values("iter")
        grow = guarded[(guarded.seed == seed) & (np.isclose(guarded.solid_probability, q))].iloc[0]
        rows.append(
            dict(label=f"seed {seed}, $q={q:.2f}$",
                 est=float(hh.rel_residual.iloc[-1]),
                 true=float(srow.solve_final_rel_residual),
                 fallback=float(grow.solve_final_rel_residual))
        )

    fig = plt.figure(figsize=(8.6, 6.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.12, 1.0], hspace=0.30)

    # --- (a) policy flowchart
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 48)
    ax.axis("off")
    _panel(ax, "a", x=0.0, y=0.95)

    R1, R2, BH = 28.0, 4.0, 12.5          # row baselines and box height
    CH_MID, CH_TOP, CH_BOT = 22.5, 45.0, 1.0   # routing channels
    _stage(ax, 13, 34, R1, BH, "frozen density field", "$\\rho_0 = 10^{-12}$", "#C2CBCF")
    _stage(ax, 41, 65, R1, BH, "probe",
           "100 iterations at $\\rho_0$\n$r_{50},\\; r_{100}$", "#C2CBCF")
    _stage(ax, 1, 22, R2, BH, "escalation ladder", "$10^{-3} \\rightarrow 10^{-2}$", C["r2"])
    _stage(ax, 28, 51, R2, BH, "selected solve", "budget $k_{\\max} = 300$", C["r3"])
    _stage(ax, 57, 79, R2, BH, "acceptance guard",
           "$\\|f-Ku\\|/\\|f\\| \\leq \\tau$", C["bad"], emphasis=True)
    _stage(ax, 85, 100, R2, BH, "accept", "floor and $u$", C["keep"], fill=C["keep"] + "22")
    dx, dy, dhw, dhh = 80.0, R1 + BH / 2, 12.5, 7.0
    _diamond(ax, dx, dy, dhw, dhh, "preserve or\nescalate?")

    ym1 = R1 + BH / 2
    ym2 = R2 + BH / 2
    _arrow(ax, (34, ym1), (41, ym1))
    _arrow(ax, (65, ym1), (dx - dhw, ym1))
    _arrow(ax, (22, ym2), (28, ym2))
    _arrow(ax, (51, ym2), (57, ym2))
    _arrow(ax, (79, ym2), (85, ym2), color=C["keep"])

    # keep branch: diamond -> selected solve
    _elbow(ax, [(dx, dy - dhh), (dx, CH_MID), (39.5, CH_MID), (39.5, R2 + BH)], C["keep"])
    ax.text(58, CH_MID + 1.0, "preserve: try $\\rho_0$ first", fontsize=7.8, color=C["keep"],
            ha="center", va="bottom")

    # raise branch: diamond -> ladder
    _elbow(ax, [(dx, dy + dhh), (dx, CH_TOP), (7.5, CH_TOP), (7.5, R2 + BH)], C["r2"])
    ax.text(43, CH_TOP + 1.2, "escalate: enter the ladder", fontsize=7.8, color=C["r2"],
            ha="center", va="bottom")

    # guard failure: guard -> ladder, retry at the next floor
    _elbow(ax, [(68, R2), (68, CH_BOT), (11.5, CH_BOT), (11.5, R2)], C["bad"], ls=(0, (4, 2)))
    ax.text(39.5, CH_BOT - 1.2, "guard fails: retry at the next floor", fontsize=7.8,
            color=C["bad"], ha="center", va="top")
    ax.text(82, ym2 + 1.6, "pass", fontsize=7.8, color=C["keep"], ha="center", va="bottom")

    # terminal exit: the ladder can be exhausted (Algorithm 1, last line).
    # Anchored at the right end of the ladder's top edge so the label sits in
    # the clear band between the rows, right of the orange raise channel (x=7.5)
    # and clear of the keep channel (y=22.5, x>=39.5).
    _elbow(ax, [(18.0, R2 + BH), (18.0, R2 + BH + 4.5)], C["bad"], ls=(0, (4, 2)))
    ax.text(18.0, R2 + BH + 5.1, "ladder exhausted:\nno admissible floor",
            fontsize=7.4, color=C["bad"], ha="center", va="bottom", linespacing=1.3)

    # --- (b) guard evidence
    ax = fig.add_subplot(gs[1, 0])
    _panel(ax, "b", x=-0.09)
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        ax.plot([r["est"], r["true"]], [yi, yi], color=C["gray"], lw=1.1, zorder=1)
        ax.scatter(r["est"], yi, s=46, marker="o", color=C["r3"], zorder=3,
                   label="FGMRES stop estimate" if yi == y[0] else None)
        ax.scatter(r["true"], yi, s=64, marker="X", color=C["bad"], zorder=3,
                   label="recomputed true residual" if yi == y[0] else None)
        ax.scatter(r["fallback"], yi, s=44, marker="s", color=C["keep"], zorder=3,
                   label="true residual after fallback" if yi == y[0] else None)
        factor = r["true"] / 1e-6
        ax.text(r["true"] * 1.35, yi + 0.16,
                f"${factor:.2f}\\tau$" if factor < 10 else f"${factor:.1f}\\tau$",
                fontsize=7.6, color=C["bad"], va="bottom")
    ax.axvline(1e-6, color=C["ink"], ls="--", lw=1.0)
    ax.text(0.92e-6, -0.52, "tolerance $\\tau=10^{-6}$", fontsize=7.8, ha="right",
            va="bottom", color=C["ink"])
    ax.set_xscale("log")
    ax.set_xlim(1.5e-7, 3e-4)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_ylim(-0.75, len(rows) - 0.35)
    ax.set_xlabel("residual measure at the stopping iterate "
                  "(circles: projected estimate; crosses and squares: recomputed)")
    ax.set_title("Reported converged; true residual did not", fontsize=9.6)
    ax.legend(frameon=False, loc="lower right", fontsize=7.6)
    ax.grid(axis="x", which="both", alpha=0.12)
    _save(fig, "fig2_policy_and_guard")


# --------------------------------------------------------------------------
# Figure 2: silent failure inside the optimization loop
# --------------------------------------------------------------------------
def fig3_silent_failure_in_loop() -> None:
    fixed_c = pd.read_csv(RESULTS / "simp_floor_trajectories" / "cantilever" / "trajectory_iters.csv")
    fixed_b = pd.read_csv(RESULTS / "simp_floor_trajectories" / "bridge" / "trajectory_iters.csv")
    guard = pd.read_csv(RESULTS / "guarded_adaptive_trajectories" / "trajectory_iters.csv")

    blocks = [
        ("Cantilever, 64k elements", fixed_c, guard[guard.run_id.str.contains("cantilever")]),
        ("Bridge, 40.5k elements", fixed_b, guard[guard.run_id.str.contains("bridge")]),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2))
    letters = iter("abcd")
    for row, (title, fixed, gd) in enumerate(blocks):
        unguarded = fixed[fixed.rho_min.astype(str).str.startswith("1e-12")].sort_values("iteration")
        gd = gd.sort_values("iteration")
        switch = gd[gd.selected_rho_min.astype(float) > 1e-12]
        switch_iter = int(switch.iteration.min()) if len(switch) else None
        cont_iter = 16

        capped = unguarded[unguarded.cg_iters >= 300]
        ax = axes[row, 0]
        _panel(ax, next(letters))
        # shade exactly the outer iterations whose solve returned at the cap, so a
        # single capped iteration reads as one iteration and not as a range
        for it in capped.iteration.astype(float):
            ax.axvspan(it - 0.5, it + 0.5, color=C["bad"], alpha=0.10, zorder=0, lw=0)
        ax.plot(unguarded.iteration, unguarded.compliance, color=C["bad"], lw=1.8,
                label="unguarded, fixed $10^{-12}$")
        ax.plot(gd.iteration, gd.compliance, color=C["r3"], lw=1.8, label="guarded policy")
        ax.axvline(cont_iter + 0.5, color=C["gray"], ls=":", lw=1.1)
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 2.6)
        ax.text(cont_iter - 1.2, hi * 0.55,
                "continuation\n$p:1.5\\!\\to\\!3.5$, $\\beta:1\\!\\to\\!4$",
                fontsize=7.4, color=C["gray"], va="top", ha="right", linespacing=1.3)
        if len(capped) > 5:
            ax.text(0.98, 0.66, f"{len(capped)} of 40 solves\nat the budget cap",
                    transform=ax.transAxes, fontsize=7.4, color=C["bad"],
                    va="top", ha="right", linespacing=1.3)
            final_c = float(unguarded.compliance.iloc[-1])
            ax.annotate(f"ends at $C={final_c:.3f}$,\nfrom unaccepted solves",
                        xy=(40, final_c), xycoords="data",
                        xytext=(0.60, 0.44), textcoords="axes fraction",
                        fontsize=7.2, color=C["bad"], ha="center", va="bottom",
                        linespacing=1.3,
                        arrowprops=dict(arrowstyle="-|>", color=C["bad"], lw=0.9,
                                        shrinkA=2, shrinkB=3))
        elif len(capped):
            it = int(capped.iteration.iloc[0])
            ax.text(0.98, 0.66, f"1 of 40 solves at the cap\n(outer iteration {it})",
                    transform=ax.transAxes, fontsize=7.4, color=C["bad"],
                    va="top", ha="right", linespacing=1.3)
        ax.set_ylabel("compliance")
        ax.set_xlabel("outer optimization iteration")
        ax.set_title(f"{title}: compliance history")
        ax.legend(frameon=False, loc="upper right", fontsize=7.8)
        ax.grid(axis="y", which="major", alpha=0.12)

        ax = axes[row, 1]
        _panel(ax, next(letters))
        ax.plot(unguarded.iteration, unguarded.cg_iters, color=C["bad"], lw=1.8,
                label="unguarded, fixed $10^{-12}$")
        ax.plot(gd.iteration, gd.cg_iters, color=C["r3"], lw=1.8, label="guarded policy")
        ax.axhline(300, color=C["ink"], ls="--", lw=1.0)
        ax.text(1.0, 305, "iteration budget $k_{\\max}=300$", fontsize=7.6, va="bottom")
        if switch_iter is not None:
            ax.axvspan(switch_iter - 0.5, 40.5, color=C["r3"], alpha=0.08)
            ax.text((switch_iter + 40) / 2, 20, "guarded policy selects $10^{-3}$",
                    fontsize=7.6, color=C["r3"], ha="center")
        ax.axvline(cont_iter + 0.5, color=C["gray"], ls=":", lw=1.1)
        ax.set_ylim(0, 345)
        ax.set_ylabel("FGMRES iterations of the state solve")
        ax.set_xlabel("outer optimization iteration")
        ax.set_title(f"{title}: state-solve cost")
        ax.grid(axis="y", alpha=0.12)

    fig.tight_layout()
    _save(fig, "fig3_silent_failure_in_loop")


# --------------------------------------------------------------------------
# Figure 3: held-out decision map and outcome ledger
# --------------------------------------------------------------------------
def graphical_abstract() -> None:
    """Elsevier graphical abstract: the cantilever silent failure, compressed.

    Reuses the trajectory CSVs of fig3; wide two-panel layout readable at
    5 x 13 cm. Not part of the manuscript figure set.
    """
    fixed_c = pd.read_csv(RESULTS / "simp_floor_trajectories" / "cantilever" / "trajectory_iters.csv")
    guard = pd.read_csv(RESULTS / "guarded_adaptive_trajectories" / "trajectory_iters.csv")
    unguarded = fixed_c[fixed_c.rho_min.astype(str).str.startswith("1e-12")].sort_values("iteration")
    gd = guard[guard.run_id.str.contains("cantilever")].sort_values("iteration")
    capped = unguarded[unguarded.cg_iters >= 300]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65))
    fig.suptitle("A converged solver flag is not a converged solve:\n"
                 "guard SIMP state solves with a recomputed residual",
                 fontsize=9.5, y=1.06, linespacing=1.3)

    ax = axes[0]
    for it in capped.iteration.astype(float):
        ax.axvspan(it - 0.5, it + 0.5, color=C["bad"], alpha=0.10, zorder=0, lw=0)
    ax.plot(unguarded.iteration, unguarded.compliance, color=C["bad"], lw=1.7,
            label="unguarded: silent failure")
    ax.plot(gd.iteration, gd.compliance, color=C["r3"], lw=1.7,
            label="guarded: 40/40 accepted")
    ax.set_yscale("log")
    ax.text(0.97, 0.60, f"{len(capped)} of 40 solves at the cap,\nno error raised",
            transform=ax.transAxes, fontsize=7.2, color=C["bad"], ha="right",
            va="top", linespacing=1.3)
    ax.set_ylabel("compliance")
    ax.set_xlabel("outer optimization iteration")
    ax.legend(frameon=False, fontsize=7.2, loc="upper right")
    ax.grid(axis="y", which="major", alpha=0.12)

    ax = axes[1]
    ax.plot(unguarded.iteration, unguarded.cg_iters, color=C["bad"], lw=1.7)
    ax.plot(gd.iteration, gd.cg_iters, color=C["r3"], lw=1.7)
    ax.axhline(300, color=C["ink"], ls="--", lw=0.9)
    ax.text(1.0, 306, "iteration budget $k_{\\max}=300$", fontsize=7.2, va="bottom")
    ax.set_ylim(0, 348)
    ax.set_ylabel("FGMRES iterations")
    ax.set_xlabel("outer optimization iteration")
    ax.grid(axis="y", alpha=0.12)

    fig.tight_layout()
    _save(fig, "graphical_abstract")


def _heldout_frame() -> pd.DataFrame:
    cant = pd.read_csv(
        RESULTS / "heldout_gmg_detector_cantilever_s41_71_guarded_true_residual" / "prospective_summary.csv")
    bridge = pd.read_csv(
        RESULTS / "heldout_gmg_detector_bridge_s41_59_guarded_true_residual" / "prospective_summary.csv")
    g = pd.concat([cant, bridge], ignore_index=True)
    labels = pd.read_csv(REVIEW / "heldout_full_true_labels_joined.csv")
    labels = labels[["preset", "seed", "solid_probability", "true_keep_original_floor",
                     "true_raise_floor", "confusion_cell", "solve_iters"]]
    labels = labels.rename(columns={"solve_iters": "full_label_iters"})
    return g.merge(labels, on=["preset", "seed", "solid_probability"], how="left")


def fig4_heldout_decision_map() -> None:
    d = _heldout_frame()
    conf = pd.read_csv(REVIEW / "heldout_full_true_label_confusion.csv").iloc[0]

    fig = plt.figure(figsize=(8.4, 4.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.26)

    ax = fig.add_subplot(gs[0, 0])
    _panel(ax, "a", x=-0.04)
    ax.axvspan(1e-2, 1e1, color=C["r2"], alpha=0.07)
    ratios = np.linspace(0.6, 1.25, 80)
    ax.fill_betweenx(ratios, 1e-4 / ratios, 1e-2, color=C["r3"], alpha=0.09)
    ax.plot(1e-4 / ratios, ratios, color=C["r3"], lw=1.0, ls=":")
    ax.axvline(1e-2, color=C["ink"], ls="--", lw=1.0)
    ax.axhline(0.6, color=C["ink"], ls=":", lw=1.0)
    for geom, marker, name in [("cantilever_gpu_medium", "o", "cantilever"),
                               ("bridge_gpu_medium", "s", "bridge")]:
        for is_keep, color, lab in [(1, C["keep"], "reference admissible"), (0, C["r2"], "reference escalation")]:
            m = (d.preset == geom) & (d.true_keep_original_floor == is_keep)
            if not m.any():      # e.g. every bridge state is a reference escalation
                continue
            ax.scatter(d.loc[m, "probe_r50"], d.loc[m, "probe_r100_over_r50"],
                       marker=marker, s=34, facecolor=color, edgecolor="white", lw=0.5,
                       alpha=0.9, zorder=3, label=f"{name}, {lab} ({int(m.sum())})")
    fk = d[d.detector_false_keep == 1]
    ax.scatter(fk.probe_r50, fk.probe_r100_over_r50, marker="o", s=140, facecolor="none",
               edgecolor=C["bad"], lw=1.6, zorder=4, label="missed escalation (guard recovered)")
    ax.set_xscale("log")
    ax.set_xlim(3e-8, 2e0)
    ax.set_ylim(-0.05, 1.30)
    ax.set_xlabel("$r_{50}$ after the baseline probe")
    ax.set_ylabel("$r_{100}/r_{50}$ plateau ratio")
    ax.set_title("Held-out decision map, all 102 cases")
    ax.text(1.5e-2, 1.24, "high-$r_{50}$ escalation", fontsize=7.6, color=C["r2"], va="top")
    ax.text(8e-4, 0.70, "plateau escalation", fontsize=7.2, color=C["r3"], va="top", ha="center")
    ax.annotate("probe converged before iteration 50:\n$r_{100}=r_{50}$, no trigger fires",
                xy=(4e-7, 0.985), xytext=(4.5e-8, 0.50), fontsize=7.0, color=C["gray"],
                arrowprops=dict(arrowstyle="->", color=C["gray"], lw=0.7,
                                shrinkA=1, shrinkB=3))
    ax.legend(frameon=False, fontsize=7.0, loc="upper left", bbox_to_anchor=(0.0, -0.20),
              ncol=2, columnspacing=1.1)
    ax.grid(axis="both", which="major", alpha=0.10)

    ax = fig.add_subplot(gs[0, 1])
    _panel(ax, "b", x=-0.10)
    mat = np.array([[conf.true_positive, conf.false_negative],
                    [conf.false_positive, conf.true_negative]], dtype=float)
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center", fontsize=17,
                    color="white" if mat[i, j] > 0.55 * mat.max() else C["ink"],
                    fontweight="bold")
    # the false-negative cell is the finding: mark it as in panel (a); the red
    # outline plus the footer text carry the message without an arrow that
    # would collide with the panel title
    ax.add_patch(Rectangle((0.5, -0.5), 1.0, 1.0, fill=False, edgecolor=C["bad"],
                           lw=1.8, zorder=5))
    ax.text(1.0, 0.33, "the missed\nescalations", fontsize=6.8, color=C["bad"],
            ha="center", va="center", linespacing=1.25, zorder=6)
    # two-line tick labels: the previous single-line labels extended into
    # panel (a)'s data area
    ax.set_xticks([0, 1], ["predict\nescalate", "predict\npreserve"])
    ax.set_yticks([0, 1], ["reference\nescalation", "reference\nadmissible"])
    ax.set_title("Rule against the 300-iteration reference", fontsize=9.2, pad=8)
    ax.text(0.5, -0.16,
            "sensitivity $74/78=94.9\\%$, specificity $24/24=100\\%$;\n"
            "the guard recovers all four missed escalations, so the\n"
            "guarded policy converges on 102/102 states",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.4)
    for s in ax.spines.values():
        s.set_visible(False)
    _save(fig, "fig4_heldout_decision_map")


# --------------------------------------------------------------------------
# Figure 4: what the policy costs
# --------------------------------------------------------------------------
def fig5_policy_cost() -> None:
    base = pd.read_csv(REVIEW / "policy_baseline_comparison.csv")
    cases = pd.read_csv(RESULTS / "gmg_policy_overhead" / "gmg_policy_overhead_cases.csv")

    order = ["guarded_probe_policy", "probe_severity_jump_1e-02",
             "original_floor_full_then_fallback", "always_1e-03", "always_1e-02"]
    names = {"guarded_probe_policy": "guarded",
             "probe_severity_jump_1e-02": "selective\n$10^{-2}$",
             "original_floor_full_then_fallback": "full $10^{-12}$\nfirst",
             "always_1e-03": "fixed\n$10^{-3}$",
             "always_1e-02": "fixed\n$10^{-2}$"}
    colors = {"guarded_probe_policy": C["keep"], "probe_severity_jump_1e-02": C["keep"],
              "original_floor_full_then_fallback": C["gray"],
              "always_1e-03": C["r3"], "always_1e-02": C["r2"]}
    b = base.set_index("policy").loc[order]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9),
                             gridspec_kw={"width_ratios": [1.0, 1.3], "wspace": 0.26})

    ax = axes[0]
    _panel(ax, "a", x=-0.10)
    x = np.arange(len(order))
    ax.bar(x, b.mean_time_s, color=[colors[p] for p in order], width=0.62)
    ax.errorbar(x, b.median_time_s,
                yerr=[b.median_time_s - b.iqr_low_time_s, b.iqr_high_time_s - b.median_time_s],
                fmt="D", ms=5, color=C["ink"], lw=1.1, capsize=3, label="median and IQR")
    for xi, (p, row) in enumerate(b.iterrows()):
        top = max(row.mean_time_s, row.iqr_high_time_s)
        ax.text(xi, top + 5.0, f"{row.mean_time_s:.1f}", ha="center", fontsize=8.4)
    ax.plot([2.62, 4.38], [52, 52], color=C["ink"], lw=0.9)
    ax.text(3.5, 55, "operator changed", ha="center", va="bottom", fontsize=7.2, color=C["ink"])
    ax.plot([-0.38, 1.38], [70, 70], color=C["keep"], lw=0.9)
    ax.text(0.5, 73, "operator preserved\nwhere admissible", ha="center", va="bottom",
            fontsize=7.2, color=C["keep"])
    ax.set_xticks(x, [names[p] for p in order], fontsize=7.0)
    ax.set_ylabel("wall time per case (s)")
    ax.text(0.99, 0.99, f"axis truncated;\nslowest case {b.max_time_s.max():.1f} s",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
            color=C["gray"], linespacing=1.3)
    ax.set_title("Cost on the same 102 held-out cases")
    ax.legend(frameon=False, fontsize=7.4, loc="upper left", bbox_to_anchor=(0.0, 1.0),
              handletextpad=0.5)
    ax.set_ylim(0, 215)
    ax.grid(axis="y", alpha=0.12)

    ax = axes[1]
    _panel(ax, "b", x=-0.06)
    cases = cases.copy()
    sel = cases["selected_solve_iters"].astype(float).to_numpy()
    probe = cases["probe_iters"].astype(float).to_numpy()
    failed = cases["failed_ladder_iters"].astype(float).to_numpy()
    xi = np.arange(len(cases))
    ax.bar(xi, sel, color=C["r3"], label="selected solve")
    ax.bar(xi, probe, bottom=sel, color=C["sand"], label="baseline probe")
    ax.bar(xi, failed, bottom=sel + probe, color=C["bad"], label="rejected floor attempt")
    ax.set_ylabel("FGMRES iterations")
    ax.set_title("Where the policy overhead is concentrated")
    ax.set_xticks([])
    for boundary in np.where(cases["category"].ne(cases["category"].shift()).to_numpy())[0][1:]:
        ax.axvline(boundary - 0.5, color="#DDDDDD", lw=0.8)
    labels = {"cantilever random prospective": "cantilever random, seeds 23/31",
              "bridge random geometry transfer": "bridge, seed 23",
              "optimized SIMP density": "optimized states"}
    totals = sel + probe + failed
    for gi, (category, group) in enumerate(cases.groupby("category", sort=False)):
        idx = group.index.to_numpy()
        offset = -0.055 if gi % 2 == 0 else -0.125
        ax.plot([idx.min() - 0.4, idx.max() + 0.4], [offset + 0.035] * 2,
                transform=ax.get_xaxis_transform(), color=C["gray"], lw=0.8,
                clip_on=False)
        ax.text(float(idx.mean()), offset, f"{labels.get(category, category)} ({len(idx)})",
                transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=7.2,
                color=C["gray"])
    ax.text(0.03, 0.70, f"all {int(failed.sum())} rejected-attempt iterations\n"
                        "come from 3 severe bridge cases",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.4, color=C["bad"])
    ax.set_ylim(0, 780)
    ax.legend(loc="upper left", fontsize=7.4, frameon=True, framealpha=0.95,
              edgecolor="none", facecolor="white", borderpad=0.4)
    ax.grid(axis="y", alpha=0.12)
    _save(fig, "fig5_policy_cost")


# --------------------------------------------------------------------------
# Figure 5: what a fixed raised floor changes on states that did not need it
# --------------------------------------------------------------------------
def _box_points(ax, groups, colors, ylabel, title, ylim=None):
    data = [g for _, g in groups]
    bp = ax.boxplot(data, widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color=C["ink"], lw=1.3),
                    whiskerprops=dict(color=C["ink"], lw=1.0),
                    capprops=dict(color=C["ink"], lw=1.0),
                    boxprops=dict(lw=1.0))
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.45)
        patch.set_edgecolor(C["ink"])
    rng = np.random.default_rng(7)
    for i, ((_, g), col) in enumerate(zip(groups, colors), start=1):
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(g)), g, s=17, color=col,
                   edgecolor="white", lw=0.4, alpha=0.95, zorder=3)
    ax.set_xticks(range(1, len(groups) + 1), [n for n, _ in groups])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.12)


def fig6_operator_perturbation() -> None:
    pert = pd.read_csv(REVIEW / "sensitivity_perturbation_combined.csv")
    pert = pert[pert.rho_min > 1e-12]
    controls = pd.read_csv(
        RESULTS / "gmg_fixed_floor_controls_strict_true_residual" / "fixed_floor_control_comparison.csv")

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.5))

    ax = axes[0]
    _panel(ax, "a", x=-0.14)
    groups = [("$10^{-3}$", pert[pert.rho_min == 1e-3].rel_compliance_change.abs() * 100),
              ("$10^{-2}$", pert[pert.rho_min == 1e-2].rel_compliance_change.abs() * 100)]
    _box_points(ax, groups, [C["r3"], C["r2"]],
                "$|\\Delta C|/C$ (%)", "Compliance, 24 random admissible states")
    for i, (_, g) in enumerate(groups, start=1):
        ax.text(i, 104, f"mean {g.mean():.1f}%", ha="center", fontsize=8.0, color=C["ink"])
    ax.set_ylim(-4, 116)

    ax = axes[1]
    _panel(ax, "b", x=-0.14)
    groups = [("$10^{-3}$", pert[pert.rho_min == 1e-3].rel_dc_l2_solid),
              ("$10^{-2}$", pert[pert.rho_min == 1e-2].rel_dc_l2_solid)]
    _box_points(ax, groups, [C["r3"], C["r2"]],
                "relative $\\ell_2$ change of $\\partial C/\\partial\\rho$ (solid elements)",
                "Sensitivity, 24 random admissible states")
    for i, (_, g) in enumerate(groups, start=1):
        ax.text(i, 1.045, f"mean {g.mean():.3f}", ha="center", fontsize=8.0)
    ax.set_ylim(-0.04, 1.13)

    ax = axes[2]
    _panel(ax, "c", x=-0.16)
    opt_path = (RESULTS / "optimized_density_sensitivity_perturbation"
                / "optimized_density_sensitivity_perturbation.csv")
    if opt_path.exists():
        # Same question as panel (b), asked of optimized designs instead of
        # severe random states.
        optp = pd.read_csv(opt_path)
        optp = optp[optp.rho_min > 1e-12]
        n_opt = optp.label.nunique()
        groups = [("$10^{-3}$", optp[optp.rho_min == 1e-3].rel_dc_l2_solid),
                  ("$10^{-2}$", optp[optp.rho_min == 1e-2].rel_dc_l2_solid)]
        _box_points(ax, groups, [C["r3"], C["r2"]],
                    "relative $\\ell_2$ change of $\\partial C/\\partial\\rho$ (solid elements)",
                    f"Sensitivity, {n_opt} optimized designs")
        for i, (_, g) in enumerate(groups, start=1):
            # the mean is pulled up by one outlier design, so report both.
            # 1e-3 labels sit above their group; the 1e-2 label hangs below its
            # box, in empty space, so it cannot reach the panel-(b) lines
            if i == 1:
                y, va = min(max(0.02, g.max() * 1.45), 0.16), "bottom"
            else:
                y, va = g.min() * 0.60, "top"
            ax.text(i, y, f"mean {g.mean():.3f}\nmedian {g.median():.3f}",
                    ha="center", va=va, fontsize=7.4, linespacing=1.25,
                    bbox=dict(fc="white", ec="none", pad=0.6), zorder=6)
        # reference lines from panel (b); labels straddle their lines (one above,
        # one below) so neither label crosses the other line
        for mean_val, col, lab, va, yfac in [
            (pert[pert.rho_min == 1e-3].rel_dc_l2_solid.mean(), C["r3"],
             "random-state mean, $10^{-3}$ (panel b)", "top", 0.93),
            (pert[pert.rho_min == 1e-2].rel_dc_l2_solid.mean(), C["r2"],
             "random-state mean, $10^{-2}$ (panel b)", "bottom", 1.08),
        ]:
            ax.axhline(mean_val, color=col, ls="--", lw=1.0, alpha=0.8)
            ax.text(0.56, mean_val * yfac, lab, fontsize=6.6, va=va, ha="left",
                    color=col, zorder=6,
                    bbox=dict(fc="white", ec="none", pad=0.4))
        ax.set_yscale("log")
        ax.set_ylim(max(1e-5, float(optp.rel_dc_l2_solid.min()) * 0.4), 2.0)
        ax.set_xlabel("fixed floor")
    else:
        ax.set_title("Where the perturbation is largest")
        rnd = pert.groupby("rho_min").rel_compliance_change.apply(lambda s: s.abs().mean() * 100)
        # only cases that carry their own original-floor reference row are comparable
        referenced = set(controls.loc[controls.rho_min <= 1e-12, "case_id"])
        controls = controls[controls.case_id.isin(referenced) & (controls.rho_min > 1e-12)]
        opt = controls[controls.case_type == "density"]
        optm = opt.groupby("rho_min").relative_compliance_change.apply(lambda s: s.abs().mean() * 100)
        rndc = controls[controls.case_type == "random"]
        rndcm = rndc.groupby("rho_min").relative_compliance_change.apply(lambda s: s.abs().mean() * 100)
        series = [("random\nkeeps (24)", rnd, 0.0),
                  ("random\ncontrols (2)", rndcm, 1.0),
                  ("optimized\ndesigns (3)", optm, 2.0)]
        w = 0.36
        for xi, (name, s, base) in enumerate(series):
            for k, (rho, col) in enumerate([(1e-3, C["r3"]), (1e-2, C["r2"])]):
                v = float(s.get(rho, np.nan))
                ax.bar(base + (k - 0.5) * w, v, width=w, color=col,
                       label={0: "$10^{-3}$", 1: "$10^{-2}$"}[k] if xi == 0 else None)
                ax.text(base + (k - 0.5) * w, v * 1.18,
                        f"{v:.1f}%" if v >= 1 else f"{v:.2f}%", ha="center", fontsize=7.4)
        ax.set_yscale("log")
        ax.set_ylim(1e-2, 400)
        ax.set_xticks([s[2] for s in series], [s[0] for s in series], fontsize=7.6)
        ax.set_ylabel("mean $|\\Delta C|/C$ (%)")
        ax.set_xlabel("state that did not need a raised floor")
        ax.legend(frameon=False, fontsize=7.6, loc="upper right", title="fixed floor",
                  title_fontsize=7.6)
    ax.grid(axis="y", which="major", alpha=0.12)

    fig.tight_layout()
    _save(fig, "fig6_operator_perturbation")


# --------------------------------------------------------------------------
# Figure 6: rescue histories and the stack-mechanism ablation
# --------------------------------------------------------------------------
def fig7_rescue_and_mechanism() -> None:
    bridge = pd.read_csv(
        RESULTS / "gmg_solver_floor_detector_transfer_bridge_seed23_ladder" / "prospective_history.csv")
    b500 = pd.read_csv(
        RESULTS / "optimized_density_B500_MF_strict_true_residual" / "density_detector_history.csv")
    mech = pd.read_csv(REVIEW / "mechanism_ablation_summary.csv")

    fig, axes = plt.subplots(1, 4, figsize=(8.8, 2.9),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1.05], "wspace": 0.34})
    cases = [(bridge[np.isclose(bridge.solid_probability.astype(float), 0.10)], "Bridge, $q=0.10$"),
             (bridge[np.isclose(bridge.solid_probability.astype(float), 0.20)], "Bridge, $q=0.20$"),
             (b500, "Optimized bridge")]
    letters = iter("abcd")
    for ax, (df, title) in zip(axes[:3], cases):
        _panel(ax, next(letters), x=-0.10)
        df = df.copy()
        df["rho"] = df["rho_min"].astype(float)
        for rho, group in df.groupby("rho"):
            group = group.sort_values("iter")
            ax.plot(group["iter"], group["rel_residual"], lw=1.8,
                    color=FLOOR_COLOR.get(rho, C["gray"]),
                    label={1e-12: "$10^{-12}$ probe", 1e-3: "$10^{-3}$ attempt",
                           1e-2: "$10^{-2}$ attempt"}.get(rho, f"{rho:g}"))
        ax.axhline(1e-6, color=C["ink"], ls="--", lw=1.0)
        ax.set_yscale("log")
        ax.set_ylim(3e-7, 2)
        ax.set_xlim(0, 305)
        ax.set_title(title, fontsize=8.8)
        ax.set_xlabel("FGMRES iteration")
        ax.grid(axis="y", which="major", alpha=0.12)
    axes[0].set_ylabel("relative residual")
    # legend in the empty lower-right region of panel (a): the lower-left
    # position covered the tolerance line and crowded the 1e-2 crossing
    axes[0].legend(fontsize=7.2, loc="center right", bbox_to_anchor=(1.0, 0.32),
                   frameon=False, handlelength=1.6, labelspacing=0.35)

    ax = axes[3]
    _panel(ax, next(letters), x=-0.24)
    labels = {"canonical": "baseline hierarchy", "levels3": "3-level hierarchy",
              "jacobi_smoother": "Jacobi smoother", "w_cycle": "W-cycle",
              "tol1em5": "tolerance $10^{-5}$", "tol1em7": "tolerance $10^{-7}$",
              "no_root_correction": "without fine-level correction"}
    order = ["canonical", "levels3", "jacobi_smoother", "w_cycle", "tol1em5", "tol1em7",
             "no_root_correction"]
    m = mech.set_index("stack_variant").loc[order]
    y = np.arange(len(order))[::-1]
    # neutral gray for "converged": teal means "original floor preserved"
    # elsewhere in the figure set, and in panels (a-c) of this very figure
    ax.barh(y, m.passes, color="#AEB9BF", label="converged")
    ax.barh(y, m.rows - m.passes, left=m.passes, color=C["bad"], label="failed")
    for yi, (p, row) in zip(y, m.iterrows()):
        ax.text(row.rows + 0.12, yi, f"{int(row.passes)}/{int(row.rows)}", va="center", fontsize=8.0)
    ax.set_yticks(y, [labels[p] for p in order])
    ax.yaxis.tick_right()
    ax.legend(frameon=False, fontsize=7.4, loc="lower left", ncol=2, bbox_to_anchor=(0.0, -0.30))
    ax.set_xlim(0, 7.4)
    ax.set_xlabel("selected solves that converge")
    ax.set_title("Solver-stack sensitivity", fontsize=8.8)
    ax.grid(axis="x", alpha=0.12)
    _save(fig, "fig7_rescue_and_mechanism")


# --------------------------------------------------------------------------
# Supplementary figures
# --------------------------------------------------------------------------
def figS5_conditioning() -> None:
    """Conditioning of the reduced direct problem against the floor.

    Produced by ``run_direct_floor_conditioning.py``; CPU only, and used as a
    diagnostic of the operator rather than of the multigrid hierarchy.
    """
    d = pd.read_csv(RESULTS / "direct_floor_conditioning" / "direct_floor_conditioning.csv")
    crit = pd.read_csv(RESULTS / "direct_floor_atlas_seeded" / "direct_floor_critical.csv")
    eps = float(np.finfo(np.float64).eps)
    tau = 1e-6

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), gridspec_kw={"wspace": 0.34})

    ax = axes[0]
    _panel(ax, "a", x=-0.13)
    for (seed, q), g in d.groupby(["seed", "solid_probability"]):
        g = g.sort_values("rho_min")
        ax.plot(g.rho_min, g.kappa, lw=1.0, color=C["gray"], alpha=0.55, zorder=1)
    for _, r in crit.iterrows():
        g = d[(d.seed == r.seed) & np.isclose(d.solid_probability, r.solid_probability)]
        row = g[g.rho_min == r.critical_rho_min]
        if len(row):
            ax.scatter(row.rho_min, row.kappa, s=30, color=C["keep"], zorder=3,
                       edgecolor="white", lw=0.5)
    ax.axhline(1.0 / eps, color=C["bad"], ls="--", lw=1.1)
    ax.axhline(1e-6 / eps, color=C["r3"], ls=":", lw=1.1)
    ax.text(0.02, 0.945, "$\\kappa=\\varepsilon^{-1}$ (FP64 limit)", transform=ax.transAxes,
            fontsize=7.2, color=C["bad"], va="bottom", ha="left")
    ax.text(0.02, 0.415, "$\\kappa\\varepsilon=\\tau$", transform=ax.transAxes,
            fontsize=7.2, color=C["r3"], va="bottom", ha="left")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("density floor $\\rho_{\\min}$")
    ax.set_ylabel("spectral condition number $\\kappa$")
    ax.set_title("Conditioning against the floor, 18 fields", fontsize=9.0)
    ax.scatter([], [], s=30, color=C["keep"], label="empirical critical floor")
    ax.legend(frameon=False, fontsize=7.2, loc="upper right")
    ax.grid(alpha=0.12, which="major")

    ax = axes[1]
    _panel(ax, "b", x=-0.13)
    for conv, color, lab in [(1, C["keep"], "reached $\\tau=10^{-6}$"),
                             (0, C["bad"], "did not reach $\\tau$")]:
        m = d.converged == conv
        ax.scatter(d.loc[m, "kappa_times_eps"], d.loc[m, "rel_residual"], s=20,
                   color=color, edgecolor="white", lw=0.4, alpha=0.9,
                   label=f"{lab} ({int(m.sum())})")
    lim = np.array([1e-11, 1e2])
    ax.plot(lim, lim, color=C["ink"], lw=0.9, ls="--")
    ax.text(2e-3, 6e-3, "$\\eta=\\kappa\\varepsilon$", fontsize=7.2, rotation=34,
            color=C["ink"])
    # regression quoted in the text, drawn on the panel; the 27 solves at the
    # backward-stability floor (~3e-13, spread across all kappa*eps) are
    # excluded from the fit and annotated instead
    floor_mask = d.rel_residual > 1e-12
    lx = np.log10(d.loc[floor_mask, "kappa_times_eps"])
    ly = np.log10(d.loc[floor_mask, "rel_residual"])
    slope, icpt = np.polyfit(lx, ly, 1)
    rr = float(np.corrcoef(lx, ly)[0, 1])
    fx = np.array([1e-10, 1e1])
    ax.plot(fx, 10 ** (slope * np.log10(fx) + icpt), color=C["gray"], lw=1.2, ls="-",
            alpha=0.9, zorder=2,
            label=f"fit: slope {slope:.2f}, $r={rr:.2f}$")
    ax.annotate(f"{int((~floor_mask).sum())} solves at the backward-\n"
                "stability floor $\\approx 3\\times10^{-13}$",
                xy=(3e-4, 8e-13), xytext=(1.5e-6, 3e-11), fontsize=6.8, color=C["gray"],
                linespacing=1.3,
                arrowprops=dict(arrowstyle="->", color=C["gray"], lw=0.7))
    ax.axhline(1e-6, color=C["r3"], ls=":", lw=1.0)
    ax.axvline(1e-6, color=C["r3"], ls=":", lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-11, 1e2)
    ax.set_ylim(1e-14, 1e-1)
    ax.set_xlabel("$\\kappa\\varepsilon$, conditioning indicator")
    ax.set_ylabel("relative residual of the direct solve")
    ax.set_title("Residuals against the indicator", fontsize=9.0)
    ax.legend(frameon=False, fontsize=7.0, loc="upper left")
    ax.grid(alpha=0.12, which="major")

    # (c) the screening rule rho_min >~ c*eps/tau, with c taken at the smallest
    # tested floor, against the empirical critical floor of each field
    ax = axes[2]
    _panel(ax, "c", x=-0.13)
    dmin = d[d.rho_min == d.rho_min.min()].copy()
    dmin["c"] = dmin.kappa * dmin.rho_min
    merged = dmin.merge(crit, on=["seed", "solid_probability"], how="inner")
    screen = merged.c * eps / tau
    ax.scatter(screen, merged.critical_rho_min, s=34, color=C["keep"],
               edgecolor="white", lw=0.5, zorder=3)
    lim2 = np.array([3e-13, 3e-5])
    ax.plot(lim2, lim2, color=C["ink"], lw=0.9, ls="--")
    ax.text(6e-9, 2.4e-9, "screen = critical floor", fontsize=6.8, rotation=38,
            color=C["ink"], va="bottom")
    ax.fill_between(lim2, lim2 * 0 + 1e-14, lim2, color=C["keep"], alpha=0.06, zorder=0)
    ax.text(0.97, 0.06, "screen conservative\non 18 of 18 fields",
            transform=ax.transAxes, fontsize=7.0, color=C["keep"], ha="right",
            va="bottom", linespacing=1.3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-9, 1e-5)
    ax.set_ylim(1e-13, 1e-5)
    ax.set_xlabel("screening floor $c(\\rr)\\,\\varepsilon/\\tau$".replace("\\rr", "\\rho"))
    ax.set_ylabel("empirical critical floor")
    ax.set_title("The screening rule", fontsize=9.0)
    ax.grid(alpha=0.12, which="major")
    _save(fig, "figS5_direct_conditioning")


def figS4_policy_sensitivity() -> None:
    simp = pd.read_csv(REVIEW / "simp_exponent_policy_sensitivity_summary.csv")
    floors = pd.read_csv(REVIEW / "original_floor_policy_sensitivity_summary.csv")

    def _counts(s):
        out = {}
        for part in str(s).split(";"):
            part = part.strip()
            if not part:
                continue
            k, v = part.split(":")
            out[float(k)] = int(v)
        return out

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for letter, ax, (df, keycol, title, xlabel) in zip(
        "ab",
        axes,
        [(simp, "penal", "SIMP exponent sensitivity", "$p_{\\mathrm{SIMP}}$"),
         (floors, "baseline_rho_min", "Original-floor sensitivity", "$\\rho_0$")],
    ):
        _panel(ax, letter, x=-0.12)
        keys = df[keycol].tolist()
        kept = []
        r3 = []
        r2 = []
        for _, row in df.iterrows():
            c = _counts(row["recommended_floor_counts"])
            base = float(row[keycol]) if keycol == "baseline_rho_min" else 1e-12
            kept.append(sum(v for k, v in c.items() if abs(k - base) < 1e-30))
            r3.append(c.get(1e-3, 0))
            r2.append(c.get(1e-2, 0))
        x = np.arange(len(keys))
        ax.bar(x, kept, width=0.62, color=C["keep"], label="kept original floor")
        ax.bar(x, r3, bottom=kept, width=0.62, color=C["r3"], label="escalated to $10^{-3}$")
        ax.bar(x, r2, bottom=np.array(kept) + np.array(r3), width=0.62, color=C["r2"],
               label="escalated to $10^{-2}$")
        for xi, k, a, b in zip(x, kept, r3, r2):
            for val, base in [(k, 0), (a, k), (b, k + a)]:
                if val:
                    ax.text(xi, base + val / 2, str(int(val)), ha="center", va="center",
                            fontsize=7.6, color="white", fontweight="semibold")
        ax.text(0.5, 0.98, "every block accepts 12 of 12 selected solves",
                transform=ax.transAxes, ha="center", va="top", fontsize=7.4, color=C["ink"])
        if keycol == "penal":
            ticks = [f"{float(k):g}" for k in keys]
        else:
            ticks = [f"$10^{{{int(np.log10(float(k)))}}}$" for k in keys]
        ax.set_xticks(x, ticks)
        ax.set_ylim(0, 13.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("cases")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.12)
    axes[0].legend(frameon=False, fontsize=7.4, loc="upper left", bbox_to_anchor=(0.0, -0.18),
                   ncol=3, columnspacing=1.2)
    fig.tight_layout()
    _save(fig, "figS4_policy_sensitivity")


def figS1_direct_atlas() -> None:
    """Reduced direct floor atlas: categorical cells, integer detector counts."""
    crit = pd.read_csv(RESULTS / "direct_floor_atlas_seeded" / "direct_floor_critical.csv")
    det = pd.read_csv(RESULTS / "admissibility_detector_validation"
                      / "detector_leave_one_seed_summary.csv")
    det = det[det["mode"] == "median"].sort_values("safety_factor")

    seeds = sorted(crit.seed.unique())
    probs = sorted(crit.solid_probability.unique())
    grid = np.full((len(probs), len(seeds)), np.nan)
    for r in crit.itertuples(index=False):
        grid[probs.index(r.solid_probability), seeds.index(r.seed)] = np.log10(r.critical_rho_min)

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2),
                            gridspec_kw={"width_ratios": [1.0, 1.05, 1.0], "wspace": 0.58})

    ax = axes[0]
    _panel(ax, "a", x=-0.22)
    im = ax.imshow(grid, cmap="YlOrBr", origin="lower", aspect="auto", vmin=-12.5, vmax=-5.5)
    for i in range(len(probs)):
        for j in range(len(seeds)):
            v = grid[i, j]
            ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=8.0,
                    color="white" if v >= -7.0 else C["ink"])
    ax.set_xticks(range(len(seeds)), [str(s) for s in seeds])
    ax.set_yticks(range(len(probs)), [f"{p:g}" for p in probs])
    ax.set_xlabel("random seed")
    ax.set_ylabel("solid probability $q$")
    ax.set_title("Critical floor, $\\log_{10}\\rho_{\\min}^{\\mathrm{crit}}$")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
    cb.set_ticks([-12, -10, -8, -6])

    ax = axes[1]
    _panel(ax, "b", x=-0.18)
    for name, fn, color, marker in [("minimum", "min", C["keep"], "o"),
                                    ("median", "median", C["gray"], "s"),
                                    ("maximum", "max", C["bad"], "^")]:
        vals = [getattr(np, fn)(grid[i, :]) for i in range(len(probs))]
        ax.plot(range(len(probs)), vals, color=color, marker=marker, ms=5, lw=1.6, label=name)
    ax.set_xticks(range(len(probs)), [f"{p:g}" for p in probs])
    ax.set_xlabel("solid probability $q$")
    ax.set_ylabel("$\\log_{10}\\rho_{\\min}^{\\mathrm{crit}}$")
    ax.set_title("Seed envelope")
    ax.legend(frameon=False, fontsize=7.6)
    ax.grid(axis="y", alpha=0.12)

    ax = axes[2]
    _panel(ax, "c", x=-0.22)
    xs = np.arange(len(det))
    w = 0.38
    ax.bar(xs - w / 2, det.false_admissible, width=w, color=C["bad"],
           label="unsafe: called admissible")
    ax.bar(xs + w / 2, det.false_inadmissible, width=w, color=C["r3"],
           label="conservative: called inadmissible")
    for xi, (u, c) in enumerate(zip(det.false_admissible, det.false_inadmissible)):
        ax.text(xi - w / 2, u + 0.4, str(int(u)), ha="center", fontsize=7.6, color=C["bad"])
        ax.text(xi + w / 2, c + 0.4, str(int(c)), ha="center", fontsize=7.6, color=C["r3"])
    ax.set_xticks(xs, [f"{int(s)}" for s in det.safety_factor])
    ax.set_xlabel("safety factor")
    ax.set_ylabel("wrong calls out of 108")
    ax.set_title("Leave-one-seed detector")
    ax.set_ylim(0, 26)
    ax.legend(frameon=False, fontsize=6.8, loc="upper left",
              bbox_to_anchor=(-0.02, 1.02))
    ax.grid(axis="y", alpha=0.12)
    _save(fig, "figS1_direct_floor_atlas")


def figS2_threshold_sensitivity() -> None:
    """Threshold grid with equal cells and readable annotations."""
    g = pd.read_csv(RESULTS / "gmg_solver_floor_detector_sensitivity"
                    / "gmg_threshold_sensitivity_grid.csv")
    slice_ = g[np.isclose(g.plateau_residual_threshold, 1e-4)]
    highs = sorted(slice_.high_residual_threshold.unique())
    ratios = sorted(slice_.plateau_ratio_threshold.unique())

    def _grid(col):
        m = np.full((len(ratios), len(highs)), np.nan)
        for r in slice_.itertuples(index=False):
            m[ratios.index(r.plateau_ratio_threshold),
              highs.index(r.high_residual_threshold)] = getattr(r, col)
        return m

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.3), gridspec_kw={"wspace": 0.30})
    for letter, ax, col, cmap, title in [
        ("a", axes[0], "false_keep", "Reds", "Missed escalations"),
        ("b", axes[1], "false_raise", "Blues", "Conservative escalations"),
    ]:
        m = _grid(col)
        _panel(ax, letter, x=-0.13)
        vmax = max(1.0, float(np.nanmax(m)))
        ax.imshow(m, cmap=cmap, origin="lower", aspect="auto", vmin=0, vmax=vmax * 1.35)
        for i in range(len(ratios)):
            for j in range(len(highs)):
                v = m[i, j]
                ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=7.6,
                        color="white" if v > 0.62 * vmax else C["ink"])
        ax.set_xticks(range(len(highs)), [f"{h:g}" for h in highs], fontsize=7.2)
        ax.set_yticks(range(len(ratios)), [f"{r:g}" for r in ratios])
        ax.set_xlabel("high-$r_{50}$ threshold")
        ax.set_ylabel("plateau-ratio threshold")
        ax.set_title(title)
        jr = ratios.index(0.6)
        jh = highs.index(0.01)
        ax.add_patch(Rectangle((jh - 0.5, jr - 0.5), 1, 1, facecolor="none",
                               edgecolor=C["ink"], lw=2.0))
        ax.annotate("selected rule", xy=(jh + 0.5, jr + 0.5), xytext=(jh + 1.9, jr + 1.5),
                    fontsize=7.0, color=C["ink"], ha="center",
                    arrowprops=dict(arrowstyle="-", color=C["ink"], lw=0.8))
    _save(fig, "figS2_threshold_sensitivity")


def _load_v1_module():
    """The v1 figure script owns the 3D rendering helpers and the two
    supplementary panels that v2 reuses unchanged."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "v1figs", Path(__file__).with_name("make_paper5_journal_figures.py"))
    v1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1)
    return v1


def figS3_optimized_gallery() -> None:
    v1 = _load_v1_module()

    strict = pd.read_csv(REVIEW / "optimized_density_strict_summary.csv").drop_duplicates("density_name")
    strict = strict.set_index("density_name")

    cases = [
        ("C64_MF", (80, 40, 20), "64k cantilever", (1.75, -1.90, 1.20), 1.03, 0.42),
        ("C216_MF", (120, 60, 30), "216k cantilever", (1.75, -1.90, 1.20), 1.03, 0.42),
        ("C512_MF", (160, 80, 40), "512k cantilever", (1.75, -1.90, 1.20), 1.03, 0.42),
        ("Brk500_MF", (80, 160, 40), "512k bracket", (2.30, -0.55, 1.15), 0.84, 0.43),
        ("M500_MF", (210, 70, 35), "514.5k MBB beam", (1.45, -1.45, 3.20), 0.76, 0.42),
        ("T500_MF", (165, 55, 55), "499k torsion", (1.90, -1.75, 1.30), 0.92, 0.44),
        ("Col500_MF", (50, 200, 50), "500k column", (1.70, -1.75, 0.95), 0.86, 0.42),
        ("B500_MF", (210, 70, 35), "514.5k bridge", (1.45, -1.45, 3.20), 0.76, 0.42),
    ]

    fig, axes_grid = plt.subplots(2, 4, figsize=(8.8, 4.0))
    axes = list(axes_grid.flat)
    for ax, (name, shape, title, cam, zoom, pscale) in zip(axes, cases):
        row = strict.loc[name]
        raised = float(row.recommended_rho_min) > 1e-12
        color = C["r2"] if raised else C["keep"]
        density = np.load(ROOT / "experiments" / "paper2" / "runs" / name / "rho_final.npy").reshape(shape)
        image = v1._render_marching_surface_image(
            density, color, camera_multipliers=cam, camera_zoom=zoom,
            parallel_scale_factor=pscale, window_size=(760, 470))
        ax.imshow(v1._pad_image(image, y_frac=0.03, x_frac=0.02))
        ax.set_axis_off()
        floor = "escalated to $10^{-2}$" if raised else "preserved at $10^{-12}$"
        ax.set_title(f"{title}\n{floor}", fontsize=9.0,
                     color=C["r2"] if raised else C["ink"])
        mant, expo = f"{row.probe_r50:.1e}".split("e")
        gray = float(((density > 0.05) & (density < 0.95)).mean()) * 100.0
        gray_txt = "0" if gray == 0 else (f"{gray:.2f}" if gray >= 0.01 else "$<$0.01")
        ax.text(0.5, 0.055,
                f"$r_{{50}} = {mant}\\times 10^{{{int(expo)}}}$,  "
                f"{int(row.solve_iters)} iterations",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=7.4, color=C["gray"])
        ax.text(0.5, -0.02, f"gray elements {gray_txt}%",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=7.0,
                color=C["bad"] if gray > 0.05 else C["gray"])
    handles = [Line2D([], [], marker="s", ls="", color=C["keep"], label="original floor preserved"),
               Line2D([], [], marker="s", ls="", color=C["r2"], label="escalated to $10^{-2}$")]
    fig.legend(handles=handles, frameon=False, loc="lower center", ncol=2, fontsize=8.4,
               bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.08, top=0.93, wspace=0.01, hspace=0.30)
    _save(fig, "figS3_optimized_density_gallery")


def supplementary_from_v1() -> None:
    """Retained for backwards compatibility: figS1 and figS2 now have their own
    generators (:func:`figS1_direct_atlas`, :func:`figS2_threshold_sensitivity`)."""
    figS1_direct_atlas()
    figS2_threshold_sensitivity()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    fig1_problem_setup()
    fig2_policy_and_guard()
    fig3_silent_failure_in_loop()
    fig4_heldout_decision_map()
    fig5_policy_cost()
    fig6_operator_perturbation()
    fig7_rescue_and_mechanism()
    graphical_abstract()
    figS4_policy_sensitivity()
    figS5_conditioning()
    supplementary_from_v1()
    # The gallery needs PyVista and the stored density fields; when they are not
    # available the previously generated PDF in OUT is left untouched.
    try:
        figS3_optimized_gallery()
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"figS3 skipped ({type(exc).__name__}: {exc}); keeping existing PDF")


if __name__ == "__main__":
    main()
