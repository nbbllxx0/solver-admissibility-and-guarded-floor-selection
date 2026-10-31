"""Main data figures for the Scientific Reports manuscript (Figs. 2-6).

Every figure is built at its printed width (165 mm, the manuscript text width)
with the shared style module ``pubstyle.py``. All numbers are read from the result files under
``experiments/phase5/results``; the asserts fail the build if a file disagrees
with the value the manuscript prints.

Usage:
    PAPER5_FIGURE_OUT=<dir> python make_paper5_srep_figures.py [fig2 fig3 ...]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.patches as mp
import numpy as np
import pandas as pd

import pubstyle as S

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "experiments" / "phase5" / "results"
REVIEW = RESULTS / "review_experiment_summary"
E1 = RESULTS / "e1lite_2026-10-30"
OUT = Path(os.environ.get("PAPER5_FIGURE_OUT", str(ROOT / "figures" / "srep")))

W = 165.0  # mm, manuscript text width; figures are placed at natural size
TOL = 1e-6
KMAX = 300

KEEP = "#009E73"   # original floor kept / solve accepted
R3 = "#0072B2"     # floor 1e-3
R2 = "#E69F00"     # floor 1e-2
F9 = "#CC79A7"     # floor 1e-9
F6 = "#56B4E9"     # floor 1e-6
BAD = S.ACCENT     # unconverged, falsely accepted or unguarded
FLOOR_COLOR = {1e-12: BAD, 1e-9: F9, 1e-6: F6, 1e-3: R3, 1e-2: R2}
FLOOR_TEX = {1e-12: "$10^{-12}$", 1e-9: "$10^{-9}$", 1e-6: "$10^{-6}$", 1e-3: "$10^{-3}$", 1e-2: "$10^{-2}$"}
MATH_PT = 7.2  # labels that carry exponents: scripts at 0.7 x 7.2 = 5.0 pt, the floor


def sci(x: float, digits: int = 2) -> str:
    m, e = f"{x:.{digits}e}".split("e")
    return rf"{m}\times10^{{{int(e)}}}"


def _close(a: float, b: float, rel: float = 5e-3) -> bool:
    return abs(a - b) <= rel * abs(b)


def _save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()  # tick labels get their text only on draw
    S.fix_scripts(fig)
    S.save(fig, str(OUT / name), formats=("pdf", "png"))
    print("wrote", name)


# --------------------------------------------------------------------------
# Data loaders
# --------------------------------------------------------------------------
def e1_runs(case: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    it = pd.read_csv(E1 / case / "trajectory_iters.csv")
    sm = pd.read_csv(E1 / case / "trajectory_summary.csv")
    it["rho_min"] = it.rho_min.astype(float)
    sm["rho_min"] = sm.rho_min.astype(float)
    return it, sm


# --------------------------------------------------------------------------
# Data-figure helpers
# --------------------------------------------------------------------------
def _ax(fig, l, b, w, h):
    return fig.add_axes(S.box_mm(fig, l, b, w, h))


def _letter(ax, letter):
    S.panel_letter(ax, letter, dy_mm=1.5)


def _ptitle(ax, text):
    """Short finding title above a panel, left aligned with the axes."""
    ax.text(0.0, 1.0, text, transform=ax.transAxes, ha="left", va="bottom", fontsize=S.V["body"],
            color=S.INK, fontweight="bold")


def _tol(ax, label=True, x=0.99, ha="right"):
    ax.axhline(TOL, color=S.INK2, ls=(0, (3, 2)), lw=0.6, zorder=1)
    if label:
        ax.text(x, TOL, r"  $\tau=10^{-6}$", transform=ax.get_yaxis_transform(), ha=ha, va="bottom",
                fontsize=S.V["small"], color=S.INK2)


def _ygrid(ax):
    ax.grid(axis="y", which="major", color=S.GRID, lw=0.4)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------
# Fig. 2: frozen states -- the flag, the guard and the probe
# --------------------------------------------------------------------------
def fig2_frozen() -> None:
    from make_paper5_v3_figures import _heldout_frame

    strict = RESULTS / "heldout_gmg_detector_cantilever_s41_71_strict_true_residual"
    summ = pd.read_csv(strict / "prospective_summary.csv")
    hist = pd.read_csv(strict / "prospective_history.csv")
    guarded = pd.read_csv(REVIEW / "heldout_false_keeps.csv")
    cases = [(47, 0.35), (59, 0.30), (71, 0.30), (71, 0.35)]
    rows = []
    for seed, q in cases:
        sel = lambda d: d[(d.seed == seed) & np.isclose(d.solid_probability, q)]  # noqa: E731
        hh = sel(hist)
        hh = hh[hh.phase == "recommended_solve"].sort_values("iter")
        rows.append(dict(label=f"seed {seed}, q = {q:.2f}", est=float(hh.rel_residual.iloc[-1]),
                         true=float(sel(summ).solve_final_rel_residual.iloc[0]),
                         fb=float(sel(guarded).solve_final_rel_residual.iloc[0])))
    assert [round(r["true"] / TOL, 2) for r in rows] == [1.45, 49.53, 1.44, 1.35], rows

    d = _heldout_frame()
    conf = pd.read_csv(REVIEW / "heldout_full_true_label_confusion.csv").iloc[0]
    assert len(d) == 102 and int(d.detector_false_keep.sum()) == 4

    fig = S.figure(W, 70)
    # a: the four false acceptances
    ax = _ax(fig, 24, 18, 38, 42)
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        ax.plot([r["est"], r["true"]], [yi, yi], color=S.MUTED, lw=0.6, zorder=1)
        ax.scatter(r["est"], yi, s=16, facecolor="white", edgecolor=S.INK, lw=0.7, zorder=3)
        ax.scatter(r["true"], yi, s=20, marker="X", color=BAD, lw=0, zorder=4)
        ax.scatter(r["fb"], yi, s=13, marker="s", color=KEEP, lw=0, zorder=3)
        ax.text(r["true"] * 1.45, yi, rf"${r['true'] / TOL:.{2 if r['true'] < 1e-5 else 1}f}\,\tau$",
                va="center", ha="left", fontsize=S.V["small"], color=BAD)
    ax.set_xscale("log")
    ax.set_xlim(2e-7, 1e-3)
    ax.axvline(TOL, color=S.INK2, ls=(0, (3, 2)), lw=0.6, zorder=1)
    ax.text(TOL * 0.85, -0.55, "tolerance", ha="right", va="bottom", fontsize=S.V["small"], color=S.INK2)
    ax.set_yticks(y, [r["label"] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("relative residual at the stopping iterate")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    _ptitle(ax, "Flag set, residual not converged")
    _letter(ax, "a")
    leg = [plt_line("o", "white", S.INK, "stop estimate (flag set)"), plt_line("X", BAD, BAD, "recomputed residual"),
           plt_line("s", KEEP, KEEP, "after fallback to the next floor")]
    fig.legend(handles=leg, loc="lower left", bbox_to_anchor=(0.005, 0.0), ncol=3, fontsize=S.V["small"],
               handletextpad=0.3, columnspacing=1.2, borderaxespad=0.2)

    # b: decision map
    ax = _ax(fig, 78, 18, 52, 42)
    ax.axvspan(1e-2, 2e0, color=R3, alpha=0.07, lw=0)
    ratios = np.linspace(0.6, 1.25, 80)
    ax.fill_betweenx(ratios, 1e-4 / ratios, 1e-2, color=R3, alpha=0.07, lw=0)
    ax.axvline(1e-2, color=S.INK2, ls=(0, (3, 2)), lw=0.6)
    ax.plot(1e-4 / ratios, ratios, color=S.INK2, ls=":", lw=0.6)
    for geom, marker in (("cantilever_gpu_medium", "o"), ("bridge_gpu_medium", "s")):
        for keep, col in ((1, KEEP), (0, R3)):
            m = (d.preset == geom) & (d.true_keep_original_floor == keep)
            if m.any():
                ax.scatter(d.loc[m, "probe_r50"], d.loc[m, "probe_r100_over_r50"], marker=marker, s=12,
                           facecolor=col, edgecolor="white", lw=0.4, zorder=3)
    fk = d[d.detector_false_keep == 1]
    ax.scatter(fk.probe_r50, fk.probe_r100_over_r50, s=46, facecolor="none", edgecolor=BAD, lw=0.9, zorder=4)
    ax.set_xscale("log")
    ax.set_xlim(3e-8, 2e0)
    ax.set_ylim(-0.05, 1.3)
    ax.set_xlabel(r"probe residual $r_{50}$")
    ax.set_ylabel(r"plateau ratio $r_{100}/r_{50}$")
    ax.text(1.4e-2, 1.27, "escalate:\nhigh $r_{50}$", fontsize=S.V["small"], color=R3, va="top")
    ax.text(1.5e-3, 0.66, "escalate:\nplateau", fontsize=S.V["small"], color=R3, va="bottom", ha="center")
    n_keep = int((d.true_keep_original_floor == 1).sum())
    ax.text(8e-8, 0.70, f"admissible: {n_keep}", fontsize=S.V["small"], color=S.INK2, va="center")
    ax.text(8e-8, 0.61, f"needs a raised floor: {len(d) - n_keep}", fontsize=S.V["small"], color=S.INK2,
            va="center")
    ax.scatter([5.5e-8], [0.70], s=10, color=KEEP, zorder=3)
    ax.scatter([5.5e-8], [0.61], s=10, color=R3, zorder=3)
    ax.text(8e-8, 0.52, "circle: cantilever; square: bridge", fontsize=S.V["small"], color=S.INK2, va="center")
    ax.annotate("4 missed:\nguard caught all", xy=(fk.probe_r50.min(), 0.05), xytext=(4e-8, 0.30),
                fontsize=S.V["small"], color=BAD, arrowprops=dict(arrowstyle="-", color=BAD, lw=0.5))
    _ptitle(ax, "Probe triage of 102 held-out states")
    _letter(ax, "b")

    # c: confusion matrix and guard outcome
    ax = _ax(fig, 145, 26, 18, 32)
    mat = np.array([[conf.true_positive, conf.false_negative], [conf.false_positive, conf.true_negative]])
    ax.imshow(np.ones_like(mat, dtype=float), cmap="Greys", vmin=0, vmax=12)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center", fontsize=S.V["label"],
                    fontweight="bold", color=BAD if (i, j) == (0, 1) else S.INK)
    ax.add_patch(mp.Rectangle((0.5, -0.5), 1, 1, fill=False, ec=BAD, lw=0.9))
    ax.set_xticks([0, 1], ["esc.", "keep"])
    ax.set_yticks([0, 1], ["needs\nraise", "admis-\nsible"])
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.set_xlabel("probe predicts", labelpad=2)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.text(0.5, -0.12, f"guard accepted\n{int(conf.passes)} of {int(conf.rows)}", transform=ax.transAxes,
            ha="center", va="top", fontsize=S.V["small"], color=KEEP, fontweight="bold")
    fig.text(134.0 / W, 61.5 / 70.0, "c", fontsize=S.V["label"], fontweight="bold", ha="left", va="bottom")
    _save(fig, "fig2_frozen_states")


# --------------------------------------------------------------------------
# Fig. 3: optimization runs at four floors
# --------------------------------------------------------------------------
def fig3_optimization() -> None:
    from matplotlib.lines import Line2D

    guard = pd.read_csv(RESULTS / "guarded_adaptive_trajectories" / "trajectory_iters.csv")
    floors = [1e-12, 1e-9, 1e-6, 1e-3]
    fig = S.figure(W, 108)
    cases = [("cantilever", "Cantilever, 64,000 elements", 56.0), ("bridge", "Bridge, 40,500 elements", 10.0)]
    letters = iter("abcd")
    for case, name, y0 in cases:
        it, sm = e1_runs(case)
        ref = pd.read_csv(RESULTS / "simp_floor_trajectories" / case / "trajectory_iters.csv")
        ref = ref[ref.rho_min.astype(float) == 1e-12].sort_values("iteration")
        gd = guard[guard.run_id.str.contains(case)].sort_values("iteration")
        smi = sm.set_index("rho_min")

        # residual of every state solve
        ax = _ax(fig, 13, y0, 78, 36)
        for f in floors:
            r = it[it.rho_min == f].sort_values("iteration")
            ax.plot(r.iteration, r.solver_true_rel_residual, color=FLOOR_COLOR[f], lw=0.8, zorder=2)
            cap = r[r.solver_converged_flag == 0]
            ax.scatter(cap.iteration, cap.solver_true_rel_residual, s=9, facecolor="white",
                       edgecolor=FLOOR_COLOR[f], lw=0.6, zorder=3)
            fa = r[(r.solver_converged_flag == 1) & (r.solver_true_rel_residual > TOL)]
            ax.scatter(fa.iteration, fa.solver_true_rel_residual, s=22, marker="X", color=S.INK, lw=0, zorder=4)
        ax.axvline(16.5, color=S.MUTED, ls=":", lw=0.6)
        ax.text(16.0, 0.55, "penalty 1.5 to 3.5", transform=ax.get_xaxis_transform(), rotation=90,
                ha="right", va="center", fontsize=S.V["small"], color=S.MUTED)
        _tol(ax, x=0.0, ha="left")
        ax.set_yscale("log")
        ax.set_xlim(0, 41)
        ax.set_ylabel(r"true residual $\eta$")
        _ygrid(ax)
        _ptitle(ax, name)
        _letter(ax, next(letters))
        for row, (lab, col) in enumerate((("solves at the cap:", "solves_at_cap"),
                                          ("false acceptances:", "false_acceptances"))):
            yy = 0.96 - 0.10 * row
            ax.text(0.015, yy, lab, transform=ax.transAxes, ha="left", va="top", fontsize=S.V["small"],
                    color=S.INK2)
            for k, f in enumerate(floors):
                ax.text(0.270 + 0.036 * k, yy, str(int(smi.loc[f, col])), transform=ax.transAxes, ha="center",
                        va="top", fontsize=S.V["small"], color=FLOOR_COLOR[f], fontweight="bold")
        if case == "bridge":
            ax.set_xlabel("outer optimization iteration")
            ax.annotate(r"flag set, $\eta>\tau$", xy=(20.3, 2.7e-5), xytext=(24, 3.2e-5),
                        fontsize=S.V["small"], color=S.INK, va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.5, color=S.INK))
        else:
            ax.set_xticklabels([])

        # compliance
        ax = _ax(fig, 104, y0, 32, 36)
        for f in floors:
            r = it[it.rho_min == f].sort_values("iteration")
            ax.plot(r.iteration, r.compliance, color=FLOOR_COLOR[f], lw=0.8, zorder=2)
        ax.plot(ref.iteration, ref.compliance, color=BAD, lw=0.8, ls=(0, (2.5, 1.5)), zorder=2)
        ax.plot(gd.iteration, gd.compliance, color=KEEP, lw=1.2, zorder=3)
        ax.axvline(16.5, color=S.MUTED, ls=":", lw=0.6)
        ax.set_yscale("log")
        ax.set_xlim(0, 41)
        ax.set_ylabel("compliance")
        _ygrid(ax)
        _ptitle(ax, "Compliance")
        _letter(ax, next(letters))
        c12 = float(it[(it.rho_min == 1e-12)].sort_values("iteration").compliance.iloc[-1])
        cref = float(ref.compliance.iloc[-1])
        cg = float(gd.compliance.iloc[-1])
        if case == "cantilever":
            assert _close(c12, 2.863) and _close(cref, 0.564) and _close(cg, 2.962), (c12, cref, cg)
            ax.set_xticklabels([])
            S.direct_label(ax, 40.5, cref, f"earlier run {cref:.3f}", color=BAD)
            S.direct_label(ax, 40.5, c12 * 0.85, f"rerun {c12:.3f}", color=BAD)
            S.direct_label(ax, 40.5, cg * 1.35, f"guarded {cg:.3f}", color=KEEP)
        else:
            ax.set_xlabel("outer optimization iteration")
            lo = float(it[it.iteration == 40].compliance.min())
            hi = float(it[it.iteration == 40].compliance.max())
            assert _close(cg, 2.049) and _close(lo, 2.0671, 1e-4) and _close(hi, 2.0735, 1e-4), (cg, lo, hi)
            S.direct_label(ax, 40.5, cg * 1.45, f"fixed {lo:.3f}–{hi:.3f}", color=S.INK2)
            S.direct_label(ax, 40.5, cg * 0.97, f"guarded {cg:.3f}", color=KEEP)

    leg = [Line2D([], [], color=FLOOR_COLOR[f], lw=0.9, label=FLOOR_TEX[f]) for f in floors]
    leg += [Line2D([], [], color=BAD, lw=0.9, ls=(0, (2.5, 1.5)), label=r"earlier $10^{-12}$ run"),
            Line2D([], [], color=KEEP, lw=1.3, label="guarded"),
            Line2D([], [], ls="", marker="o", ms=3.2, mfc="white", mec=S.INK2, mew=0.6, label="stopped at the cap"),
            Line2D([], [], ls="", marker="X", ms=4.2, color=S.INK, mew=0, label="false acceptance")]
    fig.text(0.6 / W, 1 - 2.6 / 108, r"floor $\rho_{\min}$:", fontsize=MATH_PT, color=S.INK2, va="center")
    leg = [h for pair in zip(leg[:4], leg[4:]) for h in pair]  # column-major fill: floors on row 1
    fig.legend(handles=leg, loc="center left", bbox_to_anchor=(19.0 / W, 1 - 5.0 / 108), ncol=4,
               fontsize=MATH_PT, handlelength=1.6, handletextpad=0.4, columnspacing=1.0, borderaxespad=0)
    _save(fig, "fig3_optimization_runs")


# --------------------------------------------------------------------------
# Fig. 4: what a fixed raised floor changes on states that did not need it
# --------------------------------------------------------------------------
def _strip(ax, xs, vals, color, rng, s=10):
    ax.scatter(xs + rng.uniform(-0.16, 0.16, len(vals)), vals, s=s, facecolor=color, edgecolor="white",
               lw=0.3, zorder=3, alpha=0.95)


def fig4_perturbation() -> None:
    pert = pd.read_csv(REVIEW / "sensitivity_perturbation_combined.csv")
    pert = pert[pert.rho_min > 1e-12]
    opt = pd.read_csv(RESULTS / "optimized_density_sensitivity_perturbation"
                      / "optimized_density_sensitivity_perturbation.csv")
    opt = opt[opt.rho_min > 1e-12]
    n_rand, n_opt = pert.case_id.nunique(), opt.label.nunique()
    assert (n_rand, n_opt) == (24, 7), (n_rand, n_opt)

    fig = S.figure(W, 58)
    rng = np.random.default_rng(7)
    groups = [(pert, 1e-3, 0.0), (pert, 1e-2, 1.0), (opt, 1e-3, 2.6), (opt, 1e-2, 3.6)]
    panels = [("rel_compliance_change", 100.0, r"$|\Delta C|/C$ (%)", "Compliance change", (2e-3, 2000)),
              ("rel_dc_l2_solid", 1.0, "relative change of the gradient", "Gradient change, solid elements",
               (5e-5, 20))]
    means = {}
    for pi, (col, scale, ylabel, title, ylim) in enumerate(panels):
        ax = _ax(fig, 15 + 82 * pi, 11, 66, 40)
        for df, f, x in groups:
            v = df.loc[df.rho_min == f, col].abs().to_numpy() * scale
            _strip(ax, np.full(len(v), x), v, FLOOR_COLOR[f], rng)
            m = float(v.mean())
            means[(col, df is opt, f)] = m
            ax.plot([x - 0.3, x + 0.3], [m, m], color=S.INK, lw=0.9, zorder=4)
            txt = (f"{m:.2f}%" if m < 1 else f"{m:.1f}%") if scale == 100.0 else f"{m:.3f}"
            ax.text(x, ylim[1] * 0.45, txt, ha="center", va="top", fontsize=S.V["small"], color=S.INK,
                    fontweight="bold")
        if col == "rel_dc_l2_solid":
            for f in (1e-3, 1e-2):
                ax.plot([2.25, 3.95], [f, f], color=FLOOR_COLOR[f], lw=0.6, ls=(0, (2, 1.5)), zorder=2)
            ax.text(3.95, 1e-3, r" $=\rho_{\min}$", fontsize=S.V["small"], color=S.INK2, va="center")
        ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.set_xlim(-0.6, 4.4)
        ax.axvline(1.8, color=S.GRID, lw=0.6)
        ax.set_xticks([0, 1, 2.6, 3.6], [FLOOR_TEX[g[1]] for g in groups])
        for x, lab in ((0.5, f"{n_rand} random states"), (3.1, f"{n_opt} optimized designs")):
            ax.text(x, -0.20, lab, transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=S.V["small"], color=S.INK2)
        ax.set_ylabel(ylabel)
        _ygrid(ax)
        _ptitle(ax, title)
        _letter(ax, "ab"[pi])
    assert _close(means[("rel_compliance_change", False, 1e-3)], 30.98, 1e-3)
    assert _close(means[("rel_compliance_change", True, 1e-3)], 0.480, 1e-2)
    _save(fig, "fig4_operator_perturbation")


# --------------------------------------------------------------------------
# Fig. 5: cost of the guarded policy
# --------------------------------------------------------------------------
def fig5_cost() -> None:
    base = pd.read_csv(REVIEW / "policy_baseline_comparison.csv").set_index("policy")
    cases = pd.read_csv(RESULTS / "gmg_policy_overhead" / "gmg_policy_overhead_cases.csv")
    order = ["always_1e-02", "always_1e-03", "guarded_probe_policy", "probe_severity_jump_1e-02",
             "original_floor_full_then_fallback"]
    names = {"guarded_probe_policy": "guarded policy",
             "probe_severity_jump_1e-02": r"guarded, $10^{-2}$ first if severe",
             "original_floor_full_then_fallback": r"full $10^{-12}$ attempt first",
             "always_1e-03": r"fixed $10^{-3}$", "always_1e-02": r"fixed $10^{-2}$"}
    colors = {"guarded_probe_policy": KEEP, "probe_severity_jump_1e-02": KEEP,
              "original_floor_full_then_fallback": S.MUTED, "always_1e-03": R3, "always_1e-02": R2}
    keeps = {"guarded_probe_policy": "yes", "probe_severity_jump_1e-02": "yes",
             "original_floor_full_then_fallback": "yes", "always_1e-03": "no", "always_1e-02": "no"}
    b = base.loc[order]
    assert (b.passes == 102).all()
    ratio = b.loc["guarded_probe_policy", "mean_time_s"] / b.loc["always_1e-03", "mean_time_s"]
    assert _close(ratio, 2.45, 1e-2), ratio

    fig = S.figure(W, 62)
    ax = _ax(fig, 40, 14, 40, 38)
    y = np.arange(len(order))[::-1]
    ax.barh(y, b.mean_time_s, color=[colors[p] for p in order], height=0.6, zorder=2)
    ax.errorbar(b.median_time_s, y, xerr=[b.median_time_s - b.iqr_low_time_s, b.iqr_high_time_s - b.median_time_s],
                fmt="o", ms=2.6, mfc="white", mec=S.INK, color=S.INK, lw=0.7, capsize=1.6, zorder=3)
    for yi, (pol, row) in zip(y, b.iterrows()):
        ax.text(1.13, yi, f"{row.mean_time_s:.1f}", transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=S.V["small"], color=S.INK,
                fontweight="bold" if pol == "guarded_probe_policy" else "normal")
        ax.text(1.32, yi, keeps[pol], transform=ax.get_yaxis_transform(), ha="center", va="center",
                fontsize=S.V["small"], color=KEEP if keeps[pol] == "yes" else BAD)
    for x, ha, txt in ((1.13, "right", "mean (s)"), (1.32, "center", "keeps $\\rho_0$")):
        ax.text(x, len(order) - 0.45, txt, transform=ax.get_yaxis_transform(), ha=ha, va="bottom",
                fontsize=S.V["small"], color=S.INK2)
    ax.set_yticks(y, [names[p] for p in order], fontsize=MATH_PT)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 200)
    ax.set_xlabel("wall time per held-out state (s)")
    ax.grid(axis="x", color=S.GRID, lw=0.4)
    ax.set_axisbelow(True)
    _ptitle(ax, "Same 102 states, all accepted")
    S.panel_letter(ax, "a", dx_mm=-27, dy_mm=1.5)

    ax = _ax(fig, 120, 14, 43, 38)
    sel = cases.selected_solve_iters.astype(float).to_numpy()
    probe = cases.probe_iters.astype(float).to_numpy()
    failed = cases.failed_ladder_iters.astype(float).to_numpy()
    assert int(failed.sum()) == 900 and int((failed > 0).sum()) == 3
    xi = np.arange(len(cases))
    ax.bar(xi, sel, color=R3, width=0.75, zorder=2)
    ax.bar(xi, probe, bottom=sel, color="#C9CED3", width=0.75, zorder=2)
    ax.bar(xi, failed, bottom=sel + probe, color=BAD, width=0.75, zorder=2)
    ax.set_xticks([])
    ax.set_xlim(-0.7, len(cases) - 0.3)
    ax.set_ylabel("FGMRES iterations")
    _ygrid(ax)
    _ptitle(ax, "Overhead per state, 19-state cost set")
    _letter(ax, "b")
    short = {"cantilever random prospective": "cantilever", "bridge random geometry transfer": "bridge",
             "optimized SIMP density": "optimized"}
    for cat, g in cases.groupby("category", sort=False):
        idx = g.index.to_numpy()
        ax.plot([idx.min() - 0.35, idx.max() + 0.35], [-0.04] * 2, transform=ax.get_xaxis_transform(),
                color=S.INK2, lw=0.6, clip_on=False)
        ax.text(idx.mean(), -0.07, short[cat], transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=S.V["small"], color=S.INK2)
    top = sel + probe + failed
    for k in np.where(failed > 0)[0]:
        ax.text(k, sel[k] + probe[k] + failed[k] / 2, "300", ha="center", va="center", rotation=90,
                fontsize=S.V["small"], color="white", fontweight="bold")
    ax.set_ylim(0, 800)
    for txt, col, yy in (("rejected floor attempt", BAD, 0.95), ("probe", "#8F979E", 0.86),
                         ("selected solve", R3, 0.77)):
        ax.text(0.03, yy, txt, transform=ax.transAxes, fontsize=S.V["small"], color=col, va="top",
                fontweight="bold")
    _save(fig, "fig5_policy_cost")


# --------------------------------------------------------------------------
# Fig. 6: severe states and the solver component they depend on
# --------------------------------------------------------------------------
def fig6_robustness() -> None:
    bridge = pd.read_csv(RESULTS / "gmg_solver_floor_detector_transfer_bridge_seed23_ladder"
                         / "prospective_history.csv")
    b500 = pd.read_csv(RESULTS / "optimized_density_B500_MF_strict_true_residual" / "density_detector_history.csv")
    mech = pd.read_csv(REVIEW / "mechanism_ablation_summary.csv").set_index("stack_variant")
    cases = [(bridge[np.isclose(bridge.solid_probability.astype(float), 0.10)], "Bridge, q = 0.10"),
             (bridge[np.isclose(bridge.solid_probability.astype(float), 0.20)], "Bridge, q = 0.20"),
             (b500, "Optimized bridge")]
    fig = S.figure(W, 64)
    for i, (df, title) in enumerate(cases):
        ax = _ax(fig, 13 + 31 * i, 13, 24, 38)
        df = df.assign(rho=df.rho_min.astype(float))
        for rho, g in df.groupby("rho"):
            g = g.sort_values("iter")
            ax.plot(g["iter"], g["rel_residual"], color=FLOOR_COLOR[rho], lw=0.9, zorder=2)
            last = g.iloc[-1]
            if rho == 1e-2:
                ax.scatter([last["iter"]], [last["rel_residual"]], s=12, color=KEEP, zorder=4, marker="s", lw=0)
        _tol(ax, label=(i == 0), x=0.0, ha="left")
        ax.set_yscale("log")
        ax.set_ylim(3e-7, 3)
        ax.set_xlim(0, 310)
        ax.set_xticks([0, 100, 200, 300])
        ax.set_xlabel("FGMRES iteration")
        if i == 0:
            ax.set_ylabel(r"residual estimate $\hat{r}_k$")
        else:
            ax.tick_params(axis="y", labelleft=False)
        _ygrid(ax)
        ax.text(0.0, 1.0, title, transform=ax.transAxes, ha="left", va="bottom", fontsize=S.V["small"],
                color=S.INK, fontweight="bold")
        fig.text((13 + 31 * i - 6.5) / W, (13 + 38 + 1.2) / 64, "abc"[i], fontsize=S.V["label"],
                 fontweight="bold", ha="left", va="bottom")
    from matplotlib.lines import Line2D
    leg = [Line2D([], [], color=BAD, lw=0.9, label=r"$10^{-12}$ probe"),
           Line2D([], [], color=R3, lw=0.9, label=r"$10^{-3}$ attempt"),
           Line2D([], [], color=R2, lw=0.9, label=r"$10^{-2}$ attempt"),
           Line2D([], [], ls="", marker="s", ms=3.0, color=KEEP, mew=0, label="accepted")]
    fig.legend(handles=leg, loc="center left", bbox_to_anchor=(11.0 / W, 1 - 3.0 / 64), ncol=4,
               fontsize=MATH_PT, handlelength=1.6, handletextpad=0.4, columnspacing=1.2, borderaxespad=0)

    ax = _ax(fig, 137, 13, 20, 38)
    labels = {"canonical": "reference stack", "levels3": "3 levels", "jacobi_smoother": "Jacobi smoother",
              "w_cycle": "W-cycle", "tol1em5": r"$\tau=10^{-5}$", "tol1em7": r"$\tau=10^{-7}$",
              "no_root_correction": "no fine-level correction"}
    order = ["canonical", "levels3", "jacobi_smoother", "w_cycle", "tol1em5", "tol1em7", "no_root_correction"]
    m = mech.loc[order]
    assert int(m.loc["no_root_correction", "passes"]) == 3 and (m.drop("no_root_correction").passes == 6).all()
    y = np.arange(len(order))[::-1]
    ax.barh(y, m.passes, color=[BAD if o == "no_root_correction" else "#B8C0C6" for o in order], height=0.62,
            zorder=2)
    fail = (m.rows - m.passes).to_numpy()
    ax.barh(y[fail > 0], fail[fail > 0], left=m.passes.to_numpy()[fail > 0], facecolor="white", edgecolor=BAD,
            lw=0.6, height=0.62, zorder=2)
    for yi, (o, row) in zip(y, m.iterrows()):
        ax.text(row.rows + 0.25, yi, f"{int(row.passes)}/{int(row.rows)}", va="center", fontsize=S.V["small"],
                color=BAD if o == "no_root_correction" else S.INK)
    ax.set_yticks(y, [labels[o] for o in order])
    ax.yaxis.tick_left()
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 6)
    ax.set_xticks([0, 3, 6])
    ax.set_xlabel("states solved")
    ax.text(0.0, 1.0, "Six-state solver ablation", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=S.V["small"], color=S.INK, fontweight="bold")
    fig.text(112.0 / W, (13 + 38 + 1.2) / 64, "d", fontsize=S.V["label"], fontweight="bold", ha="left",
             va="bottom")
    _save(fig, "fig6_robustness")


def plt_line(marker, fc, ec, label):
    from matplotlib.lines import Line2D
    return Line2D([], [], ls="none", marker=marker, markersize=4, markerfacecolor=fc, markeredgecolor=ec,
                  markeredgewidth=0.7, label=label)


FIGS = {"fig2": fig2_frozen, "fig3": fig3_optimization,
        "fig4": fig4_perturbation, "fig5": fig5_cost, "fig6": fig6_robustness}


def main(argv: list[str]) -> None:
    S.apply("nature")
    for name in (argv or list(FIGS)):
        FIGS[name]()


if __name__ == "__main__":
    main(sys.argv[1:])
