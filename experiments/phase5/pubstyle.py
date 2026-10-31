"""One style module for every figure in a paper.

Copy this file next to the figure scripts (do not import it from the skill
directory; the paper must rebuild without the skill). Every figure script
then does:

    import pubstyle as S
    S.apply("nature")                      # or "ieee", "elsevier", "generic"
    fig = S.figure("double", height_mm=90)
    ax = fig.add_axes(S.box_mm(fig, 12, 10, 70, 70))
    ...
    S.panel_letter(ax, "a")
    S.save(fig, "figures/fig2_result")     # writes .pdf (+ .svg, .png)
    S.check_text(fig, scale=1.0)           # raises if any text < floor

Rules this module encodes (see references/figure-craft.md):
- figures are built at their printed width, in millimetres;
- one sans-serif family; body 7 pt, ticks/legend 6 pt, floor 5 pt at print;
- panel letters 8 pt bold lowercase, no parentheses;
- neutral greys for everything that is not data; one accent colour;
- categorical colours in fixed order, colour-blind safe (Okabe-Ito / Wong 2011);
- vector output with TrueType (Type 42) fonts, no Type 3.
"""
import math
import re

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as _fm
except ImportError as exc:  # clear message instead of a bare traceback
    raise ImportError("pubstyle.py needs matplotlib in the Python that runs it "
                      "(check `python -c \"import matplotlib\"`; on Windows try `py -0p` "
                      "for another interpreter)") from exc

MM = 1 / 25.4  # inches per millimetre

# Printed widths in mm. Nature: research-figure-guide.nature.com (89 / 183 mm,
# max height 170 mm). IEEE: 3.5 in / 7.16 in. Elsevier: 90 / 190 mm.
VENUES = {
    "nature":   {"single": 89.0,  "double": 183.0, "onehalf": 120.0, "max_h": 170.0, "floor": 5.0, "body": 7.0, "small": 6.0, "label": 8.0},
    "ieee":     {"single": 88.9,  "double": 181.9, "onehalf": 120.0, "max_h": 230.0, "floor": 6.0, "body": 8.0, "small": 7.0, "label": 9.0},
    "elsevier": {"single": 90.0,  "double": 190.0, "onehalf": 140.0, "max_h": 240.0, "floor": 6.0, "body": 8.0, "small": 7.0, "label": 9.0},
    "generic":  {"single": 85.0,  "double": 170.0, "onehalf": 120.0, "max_h": 220.0, "floor": 6.0, "body": 8.0, "small": 7.0, "label": 9.0},
}
V = dict(VENUES["nature"])

# Neutral ink for everything that is not data.
INK, INK2, MUTED, GRID, EDGE, FILL = "#1a1a19", "#4a4945", "#7d7b74", "#e4e2dc", "#c9c7bd", "#f4f3ef"
# Okabe-Ito (Wong 2011, Nat. Methods 8:441), fixed order; never cycle.
CAT = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
ACCENT = "#D55E00"  # the one highlight colour: "changed", "fails", "look here"
# Sequential (single hue, light -> dark) and diverging (two hues, grey middle).
SEQ = ["#deebf7", "#9ecae1", "#4292c6", "#2171b5", "#08306b"]
DIV = ["#2166ac", "#67a9cf", "#f7f7f7", "#ef8a62", "#b2182b"]


def apply(venue="nature", family=("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans")):
    """Set rcParams for a venue. Call once at the top of every figure script."""
    V.clear()
    V.update(VENUES[venue])
    # mathtext "custom" needs ONE installed family: use the first that exists,
    # otherwise matplotlib warns and silently mixes DejaVu maths with sans text.
    installed = []
    for fam in family:
        try:
            _fm.findfont(fam, fallback_to_default=False)
            installed.append(fam)
        except ValueError:
            pass
    math_family = (installed or ["DejaVu Sans"])[0]
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": list(family),
        "font.size": V["body"], "axes.titlesize": V["body"], "axes.labelsize": V["body"],
        "xtick.labelsize": V["small"], "ytick.labelsize": V["small"], "legend.fontsize": V["small"],
        "legend.frameon": False, "legend.handlelength": 1.2, "legend.borderaxespad": 0.2,
        "axes.linewidth": 0.5, "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.0, "ytick.major.size": 2.0, "xtick.direction": "out", "ytick.direction": "out",
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
        "grid.color": GRID, "grid.linewidth": 0.4,
        "lines.linewidth": 1.0, "lines.markersize": 3.5, "patch.linewidth": 0.5,
        "axes.prop_cycle": mpl.cycler(color=CAT),
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "mathtext.fontset": "custom", "mathtext.rm": math_family, "mathtext.it": math_family + ":italic",
        "mathtext.bf": math_family + ":bold",
        "savefig.dpi": 600, "savefig.facecolor": "white", "figure.facecolor": "white",
    })


def figure(width="double", height_mm=80.0):
    """Figure at its printed size. width: 'single' | 'onehalf' | 'double' | mm."""
    w = V[width] if isinstance(width, str) else float(width)
    if height_mm > V["max_h"]:
        raise ValueError(f"height {height_mm} mm exceeds the venue maximum {V['max_h']} mm")
    return plt.figure(figsize=(w * MM, height_mm * MM))


def box_mm(fig, left, bottom, width, height):
    """Axes rectangle in mm from the bottom-left corner (manual, reproducible layout)."""
    W, H = fig.get_size_inches() / MM
    return [left / W, bottom / H, width / W, height / H]


def panel_letter(ax, letter, dx_mm=0.0, dy_mm=1.0):
    """8 pt bold lowercase letter at the top-left of the whole panel.

    Aligned to the panel's outer edge (tick labels and y-label included), not
    the axes spine, so letters line up across panels with different labels.
    """
    fig = ax.figure
    fig.canvas.draw()
    bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    W, H = fig.get_size_inches() / MM
    fig.text(max(bb.x0 + dx_mm / W, 0.0), ax.get_position().y1 + dy_mm / H, letter,
             fontsize=V["label"], fontweight="bold", ha="left", va="bottom", color=INK)


def direct_label(ax, x, y, text, color=INK2, **kw):
    """Label a series at its end instead of using a legend."""
    ax.annotate(text, (x, y), xytext=(3, 0), textcoords="offset points", va="center",
                fontsize=V["small"], color=color, annotation_clip=False, **kw)


def sig(x, n=2):
    """Round to n significant figures, half up, for labels."""
    if x == 0 or not math.isfinite(x):
        return str(x)
    r = float(f"{x:.{n - 1}e}")  # round first: 9.96 -> 10, which then needs 0 decimals, not 1
    d = n - int(math.floor(math.log10(abs(r)))) - 1
    return f"{r:.{max(d, 0)}f}"


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion k/n."""
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def check_text(fig, scale=1.0, floor=None):
    """Raise if any visible text renders below the floor at the printed scale.

    scale: printed width / built width (e.g. 0.92 when LaTeX places a 180 mm
    figure at a 165 mm text width).
    """
    floor = floor or V["floor"]
    fig.canvas.draw()
    bad = []
    for t in fig.findobj(mpl.text.Text):
        s = t.get_text()
        if not (t.get_visible() and s.strip()):
            continue
        pt = t.get_fontsize() * scale
        # mathtext sub/superscripts (e.g. log-axis exponents 10^{-2}) render at
        # ~0.7 of the label size; they are the usual floor violation.
        if "$" in s and re.search(r"[\^_]", s):
            pt *= SCRIPT
        if pt < floor - 1e-6:
            bad.append((round(pt, 2), s[:30]))
    if bad:
        raise ValueError(f"{len(bad)} text items below {floor} pt at print scale "
                         f"(scripts counted at {SCRIPT}x): {bad[:6]}. Use fix_scripts(fig) or plain tick labels.")
    return True


SCRIPT = 0.7  # mathtext script size relative to its label


def fix_scripts(fig, floor=None):
    """Raise labels containing mathtext sub/superscripts so the scripts meet the floor."""
    floor = floor or V["floor"]
    for t in fig.findobj(mpl.text.Text):
        s = t.get_text()
        if t.get_visible() and "$" in s and re.search(r"[\^_]", s) and t.get_fontsize() * SCRIPT < floor:
            t.set_fontsize(floor / SCRIPT)


def log_ticks_plain(ax, axis="x"):
    """Log axis with plain tick labels (0.01, 0.1, 1, 10) instead of 10^n exponents."""
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def check_bounds(fig):
    """Raise if any text or artist spills outside the figure canvas."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    hidden = set()  # tick labels whose tick lies outside the view limits are never drawn
    for ax in fig.axes:
        for axis, (lo, hi) in ((ax.xaxis, sorted(ax.get_xlim())), (ax.yaxis, sorted(ax.get_ylim()))):
            for tk in axis.get_major_ticks() + axis.get_minor_ticks():
                if not (lo - 1e-9 <= tk.get_loc() <= hi + 1e-9):
                    hidden.update({id(tk.label1), id(tk.label2)})
    def outside(bb):
        return bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5

    out = []
    for t in fig.findobj(mpl.text.Text):
        if t.get_visible() and t.get_text().strip() and id(t) not in hidden and outside(t.get_window_extent(r)):
            out.append("text " + repr(t.get_text()[:30]))
    for ax in fig.axes:  # arrows, boxes and other patches drawn in data coordinates
        for p in ax.patches:
            if p.get_visible() and outside(p.get_window_extent(r)):
                out.append(type(p).__name__)
    if out:
        raise ValueError(f"outside the canvas (clipped in print): {out[:6]}")
    return True


# savefig metadata that removes the tool name and date from each format
NO_METADATA = {"pdf": {"Creator": None, "Producer": None, "CreationDate": None},
               "svg": {"Creator": None, "Date": None, "Format": None, "Type": None},
               "png": {"Software": None}}


def save(fig, stem, formats=("pdf", "svg", "png"), check=True, scale=1.0):
    """Write vector (and a preview PNG) without tool or date metadata. Runs the text and
    bounds checks first."""
    if check:
        check_text(fig, scale=scale)
        check_bounds(fig)
    for ext in formats:
        fig.savefig(f"{stem}.{ext}", dpi=300 if ext == "png" else None, metadata=NO_METADATA.get(ext))
        if ext == "svg":  # matplotlib still writes a "Created with" comment
            path = f"{stem}.svg"
            with open(path, encoding="utf-8") as fh:
                text = re.sub(r"<!--.*?-->\s*", "", fh.read(), flags=re.S)
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
