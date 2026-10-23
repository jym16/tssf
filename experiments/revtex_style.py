"""Match matplotlib figures to REVTeX (APS) documents.

Two things make a figure look native in a REVTeX paper:
1. Its width equals the column width (one-column figures) or the full text
   width (two-column / figure* figures), so it is placed at natural size with
   no scaling.
2. Its text and math are set in the same font (Computer Modern) at the same
   physical point size as the surrounding document.

This module provides both. Import it, call `set_revtex_style()` once, and size
every figure with `fig_size(...)`.

Measured directly from revtex4-2 (aps, prl):
    \\columnwidth = 246.0 pt
    \\textwidth   = 510.0 pt
    base font     = 10 pt
    rmdefault     = cmr  (Computer Modern Roman), OT1 encoding
These values are identical across the common APS journals (prl, pra, prb, prd,
rmp, ...) and across revtex4-1 / revtex4-2.
"""

from __future__ import annotations

import shutil
import warnings

import matplotlib as mpl
import numpy as np

# --- REVTeX page geometry -------------------------------------------------
# TeX points -> inches uses 1 in = 72.27 pt (NOT 72). matplotlib figure sizes
# are in inches, so convert here.
_PT_PER_INCH = 72.27

COLUMN_WIDTH_PT = 246.0          # \columnwidth in a two-column APS document
TEXT_WIDTH_PT = 510.0            # \textwidth (spans both columns -> figure*)
BASE_FONT_PT = 10.0              # document base font size
FOOTNOTE_FONT_PT = 8.0           # \footnotesize at a 10 pt body (ticks, legend)

COLUMN_WIDTH_IN = COLUMN_WIDTH_PT / _PT_PER_INCH   # ~3.404 in
TEXT_WIDTH_IN = TEXT_WIDTH_PT / _PT_PER_INCH       # ~7.057 in

GOLDEN_RATIO = (5 ** 0.5 - 1) / 2  # ~0.618, a pleasant default height:width
SQUARE = 1.0                       # equal height and width

# Named aspect ratios accepted as strings by fig_size (height / width).
_ASPECTS = {"golden": GOLDEN_RATIO, "square": SQUARE}


def _resolve_aspect(aspect):
    """Accept a float or a name ('golden'/'square') and return the float."""
    if isinstance(aspect, str):
        try:
            return _ASPECTS[aspect.lower()]
        except KeyError:
            raise ValueError(
                f"aspect must be a number or one of {sorted(_ASPECTS)}; "
                f"got {aspect!r}")
    return aspect


_latex_ok: bool | None = None


def latex_available() -> bool:
    """True if usetex rendering actually works in this environment.

    Checks more than `which latex`: a LaTeX install can be present yet missing
    packages that matplotlib's usetex pipeline needs (e.g. type1ec.sty from
    cm-super). So we attempt a real one-off render and cache the result. This is
    what makes the auto-fallback reliable.
    """
    global _latex_ok
    if _latex_ok is not None:
        return _latex_ok
    if shutil.which("latex") is None:
        _latex_ok = False
        return _latex_ok
    try:
        from matplotlib.texmanager import TexManager
        TexManager().make_dvi(r"$x$ test", 10)
        _latex_ok = True
    except Exception:
        _latex_ok = False
    return _latex_ok


def fig_size(columns: int = 1, aspect: "float | str" = "golden",
             height_in: float | None = None,
             width_fraction: float = 1.0,
             nrows: int = 1, ncols: int = 1,
             panel_aspect: "float | str | None" = None) -> tuple[float, float]:
    """Return a (width, height) tuple in inches for a REVTeX figure.

    columns        1 -> single column width (\\columnwidth, ~3.40 in)
                   2 -> full text width (\\textwidth, ~7.06 in; use with figure*)
    width_fraction Use a fraction of the target width (e.g. 0.8).

    aspect/panel_aspect accept a number (height/width) or the names "golden"
    or "square", so you can switch shape easily, e.g. aspect="square".

    Height is determined in one of three ways, in priority order:
      1. height_in     -> use this height directly (inches).
      2. panel_aspect  -> size for a grid: each of the nrows x ncols panels gets
                          aspect = panel_aspect (height/width), so
                          height = nrows * (width/ncols) * panel_aspect.
                          Use this for multi-panel figures so every panel keeps a
                          sensible shape instead of the whole canvas being golden.
      3. aspect        -> height = width * aspect (whole-figure aspect ratio).
                          The default; golden ratio for a single panel.

    Note: with constrained_layout (enabled by set_revtex_style) the *axes* shrink
    a little to make room for labels, so panels won't be pixel-exactly
    panel_aspect, but this gives the right overall proportions. Tune panel_aspect
    or height_in to taste.
    """
    if columns == 1:
        width = COLUMN_WIDTH_IN
    elif columns == 2:
        width = TEXT_WIDTH_IN
    else:
        raise ValueError("columns must be 1 or 2")
    width *= width_fraction

    if height_in is not None:
        height = height_in
    elif panel_aspect is not None:
        height = nrows * (width / ncols) * _resolve_aspect(panel_aspect)
    else:
        height = width * _resolve_aspect(aspect)
    return (width, height)


def set_revtex_style(use_tex: bool | None = None,
                     font_size: float = BASE_FONT_PT,
                     small_size: float | None = None,
                     preamble: str = "") -> None:
    """Configure matplotlib rcParams to match a REVTeX document.

    use_tex     None  -> use LaTeX rendering if a LaTeX install is found,
                         otherwise fall back to matplotlib's bundled Computer
                         Modern fonts + 'cm' math fontset (no LaTeX needed).
                True  -> force LaTeX rendering (errors later if not installed).
                False -> never use LaTeX; use bundled Computer Modern instead.
    font_size   Base font size in pt. Default 10 = the REVTeX body size, so
                text in the figure matches body text when placed at natural
                size. Axis/label text uses this size.
    small_size  Size for tick labels and legend. Defaults to 8 pt, which is
                REVTeX \\footnotesize for a 10 pt body. (Size ladder at 10 pt
                body: normalsize 10, small 9, footnotesize 8, scriptsize 7.)
                If you change font_size, pass small_size explicitly to match.
    preamble    Extra LaTeX preamble lines (only used when rendering via LaTeX),
                e.g. r"\\usepackage{siunitx}". Append packages your paper uses
                so math in the figure matches the paper exactly.
    """
    if use_tex is None:
        use_tex = latex_available()
    elif use_tex and not latex_available():
        warnings.warn(
            "use_tex=True but no LaTeX executable found; falling back to "
            "matplotlib's bundled Computer Modern fonts.", stacklevel=2)
        use_tex = False

    if small_size is None:
        small_size = FOOTNOTE_FONT_PT

    params = {
        # Font sizes (points). Figure placed at natural size => these are the
        # physical sizes on the printed page.
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": small_size,
        "ytick.labelsize": small_size,
        "legend.fontsize": small_size,
        "figure.titlesize": font_size,
        # Serif body to match Computer Modern Roman.
        "font.family": "serif",
        # Thinner lines/ticks read better at print size.
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.45,
        "ytick.minor.width": 0.45,
        "legend.frameon": False,
        # Shorter legend handles read better in narrow journal columns.
        "legend.handlelength": 1.3,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.3,
        # Fit labels/ticks INSIDE the fixed canvas instead of cropping to them.
        # This is critical: saving with bbox_inches="tight" would change the
        # output width, so LaTeX would rescale the figure and the carefully
        # matched point sizes would no longer match. constrained_layout keeps
        # the saved size exactly equal to fig_size(...).
        "figure.constrained_layout.use": True,
        # Tight, predictable output. Note: NO savefig.bbox="tight" here.
        "figure.dpi": 150,
        "savefig.dpi": 600,
    }

    if use_tex:
        # Render all text through LaTeX with Computer Modern -> exact match.
        params.update({
            "text.usetex": True,
            "font.serif": ["Computer Modern Roman"],
            "text.latex.preamble": preamble,
        })
    else:
        # No LaTeX: use matplotlib's bundled Computer Modern fonts and the
        # 'cm' math fontset so math glyphs also look like Computer Modern.
        params.update({
            "text.usetex": False,
            "font.serif": ["cmr10", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            # Render numeric tick labels via mathtext so they match cmr10
            # (also silences matplotlib's cmr10+formatter warning).
            "axes.formatter.use_mathtext": True,
            # Avoid a known minus-sign glyph warning with cmr10.
            "axes.unicode_minus": False,
        })

    mpl.rcParams.update(params)


def _candidate_corners(pad):
    # (axes-fraction x, y, ha, va) for each corner, inset by `pad`.
    return {
        "upper left":  (pad, 1 - pad, "left", "top"),
        "upper right": (1 - pad, 1 - pad, "right", "top"),
        "lower left":  (pad, pad, "left", "bottom"),
        "lower right": (1 - pad, pad, "right", "bottom"),
    }


def _data_points_display(ax):
    """All plotted vertices (lines + scatter offsets) in display coords."""
    chunks = []
    for line in ax.get_lines():
        xy = line.get_xydata()
        if len(xy):
            chunks.append(ax.transData.transform(xy))
    for col in ax.collections:
        try:
            off = np.asarray(col.get_offsets())
        except Exception:
            off = np.empty((0, 2))
        if len(off):
            chunks.append(ax.transData.transform(off))
    return np.vstack(chunks) if chunks else np.empty((0, 2))


def panel_label(ax, text, prefer="upper left", pad=0.04, margin_pt=2.0,
                **text_kwargs):
    """Add a panel label like "(a)" that avoids overlapping plot contents.

    It places the label at `prefer` (default upper-left), then checks whether
    that spot collides with plotted data, the legend, or earlier labels. If it
    does, it tries the other corners and keeps the emptiest one. This directly
    answers "move it if it overlaps" without you hand-tuning coordinates.

    Call this AFTER plotting the data and creating the legend, so there is
    something to test against. Returns the Text artist.

    ax           the Axes to label.
    text         label string, e.g. "(a)".
    prefer       corner to try first: "upper left"/"upper right"/
                 "lower left"/"lower right".
    pad          inset from the axes edge, in axes fraction.
    margin_pt    breathing room (points) added around the label when testing
                 overlap, so it isn't placed flush against a line.
    text_kwargs  forwarded to ax.text (e.g. fontsize, fontweight, color).
    """
    fig = ax.figure
    fig.canvas.draw()  # ensure transforms/renderer are current
    renderer = fig.canvas.get_renderer()

    corners = _candidate_corners(pad)
    order = [prefer] + [c for c in corners if c != prefer]

    # Create the label once; its size doesn't depend on which corner it's in.
    fx, fy, ha, va = corners[prefer]
    txt = ax.text(fx, fy, text, transform=ax.transAxes, ha=ha, va=va,
                  **text_kwargs)

    bb = txt.get_window_extent(renderer)
    w, h = bb.width, bb.height
    margin_px = margin_pt * fig.dpi / 72.0

    data_pts = _data_points_display(ax)
    legend = ax.get_legend()
    legend_bb = legend.get_window_extent(renderer) if legend else None
    # Other text already on the axes (e.g. previously placed labels).
    other_texts = [t for t in ax.texts
                   if t is not txt and t.get_text().strip()]
    other_bbs = [t.get_window_extent(renderer) for t in other_texts]

    def score(corner):
        fx, fy, ha, va = corners[corner]
        x0, y0 = ax.transAxes.transform((fx, fy))
        left = x0 - (w if ha == "right" else 0)
        bottom = y0 - (h if va == "top" else 0)
        from matplotlib.transforms import Bbox
        box = Bbox.from_bounds(left - margin_px, bottom - margin_px,
                               w + 2 * margin_px, h + 2 * margin_px)
        s = 0.0
        if data_pts.size:
            inside = ((data_pts[:, 0] >= box.x0) & (data_pts[:, 0] <= box.x1) &
                      (data_pts[:, 1] >= box.y0) & (data_pts[:, 1] <= box.y1))
            s += float(inside.sum())
        # Heavily penalise overlapping the legend or another label.
        for other in ([legend_bb] if legend_bb else []) + other_bbs:
            if box.overlaps(other):
                s += 1e6
        return s, (fx, fy, ha, va)

    # Pick the preferred corner if it's clear; otherwise the lowest score.
    best_corner = min(order, key=lambda c: score(c)[0])
    _, (fx, fy, ha, va) = score(best_corner)
    txt.set_position((fx, fy))
    txt.set_ha(ha)
    txt.set_va(va)
    return txt


def savefig(fig, path, **kwargs):
    """Save preserving the figure's exact width (no tight-bbox cropping).

    The figure must keep its `fig_size(...)` width so LaTeX includes it at
    natural size with no scaling. We therefore do NOT pass bbox_inches="tight"
    (which would crop and change the width). Layout fitting is handled by
    constrained_layout, enabled in set_revtex_style(). Override via kwargs if
    you really need to.
    """
    fig.savefig(path, **kwargs)
