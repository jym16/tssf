"""REVTeX-formatted figures of the report (vector PDFs at APS column / text width).

Usage::

    python make_figures.py --data data --out figures            # all figures
    python make_figures.py gallery order --data data            # a subset

Figures ``gallery``, ``thickness``, ``representation`` and ``schematic`` simulate networks on the
fly; ``order`` and ``mechanics`` read the JSON files written by ``run_experiments.py``.
``revtex_style.py`` (same folder) matches REVTeX fonts and widths.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402
from matplotlib.patches import Polygon, Rectangle     # noqa: E402

import revtex_style as rtx                            # noqa: E402
import tssf                                           # noqa: E402
from tssf.geometry import chord                       # noqa: E402

COLOR = {"square": "C0", "triangular": "C1", "kagome": "C2"}
MARKER = {"square": "o", "triangular": "s", "kagome": "^"}
REGISTERED = {"square": 0.499, "triangular": 1 / 3 - 1e-3, "kagome": 1 / 6}   # proven bounds on eps_x
SIN_GAMMA = {"square": 1.0, "triangular": np.sin(np.pi / 3), "kagome": np.sin(np.pi / 3)}  # stub-host angle


def load(data, name):
    with open(os.path.join(data, f"res_{name}.json")) as fh:
        return json.load(fh)


def mean_where(rows, key, **match):
    """Mean of ``key`` (a function of a row) over rows whose fields equal ``match``."""
    return np.mean([key(r) for r in rows if all(r[k] == v for k, v in match.items())])


# ----------------------------------------------------------------------------- network figures
def schematic(args):
    """(a) the snapping map acting on one proposal; (b) stagger of two stubs across a host."""
    fig, axs = plt.subplots(1, 2, figsize=(rtx.COLUMN_WIDTH_IN, 1.75), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax = axs[0]
    cell = np.array([[0.2, 0.15], [2.6, 0.3], [2.85, 1.55], [1.65, 2.45], [0.25, 1.95]])
    for c in (1.0, 2.0):                                         # target lattice lines (h = 1)
        ax.plot([c, c], [-0.05, 2.6], ls=(0, (3, 2)), color="0.6", lw=0.6)
        ax.plot([0, 3], [c, c], ls=(0, (3, 2)), color="0.6", lw=0.6)
    ax.add_patch(Polygon(cell, closed=True, fill=False, ec="k", lw=0.9))
    x, theta, eps = np.array([1.5, 1.3]), np.deg2rad(66), 0.35
    d = 2.0 - x[0]                                               # offset from the vertical line x = 2
    x_snap = x + (1 - eps) * np.array([d, 0.0])
    theta_snap = np.pi / 2 + eps * (theta - np.pi / 2)
    for p, t, col, lw in [(x, theta, "0.5", 0.8), (x_snap, theta_snap, "C0", 1.2)]:
        a, _, b, _ = chord(cell, list(range(len(cell))), p, np.array([np.cos(t), np.sin(t)]))
        ax.plot([a[0], b[0]], [a[1], b[1]], color=col, lw=lw)
    ax.annotate("", xy=(2.0, x[1]), xytext=tuple(x), arrowprops=dict(arrowstyle="<->", lw=0.6, shrinkA=0, shrinkB=0))
    ax.plot(*x, "o", color="0.5", ms=2.5, zorder=5)
    ax.plot(*x_snap, "o", color="C0", ms=2.5, zorder=5)
    ax.text(1.66, x[1] + 0.1, r"$d$", ha="center", fontsize=8)
    ax.text(x[0] - 0.1, x[1] + 0.05, r"$\mathbf{x}$", ha="right", va="bottom", fontsize=8, color="0.35")
    ax.text(x_snap[0] + 0.02, x[1] - 0.12, r"$\mathbf{x}'$", ha="left", va="top", fontsize=8, color="C0")
    ax.legend(handles=[Line2D([], [], color="0.5", lw=0.8, label=r"proposal $(\mathbf{x},\theta)$"),
                       Line2D([], [], color="C0", lw=1.2, label=r"snapped $(\mathbf{x}',\theta')$")],
              fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.02), handlelength=1.3)
    ax.set_xlim(0, 3); ax.set_ylim(-0.05, 2.6); ax.set_aspect("equal"); ax.axis("off")
    ax.text(0.0, 2.55, "(a)", fontsize=9, va="top")
    ax = axs[1]
    ax.add_patch(Rectangle((-0.1, 0.92), 2.0, 0.16, color="0.82", lw=0))   # host strut of finite width
    ax.plot([-0.1, 1.9], [1, 1], color="k", lw=0.9)
    ax.plot([1, 1], [-0.05, 2.05], ls=(0, (3, 2)), color="0.6", lw=0.6)
    xa, xb = 1.14, 0.92
    ax.plot([xa, xa], [1, 2], color="C0", lw=1.2)
    ax.plot([xb, xb], [0, 1], color="C0", lw=1.2)
    ax.annotate("", xy=(xa, 0.62), xytext=(xb, 0.62), arrowprops=dict(arrowstyle="<->", lw=0.6, shrinkA=0, shrinkB=0))
    ax.text(xa + 0.07, 0.62, r"$s$", va="center", fontsize=8)
    ax.text(0.0, 1.14, "host", fontsize=6.5)
    ax.text(-0.1, -0.32, r"$s=\varepsilon_x|d_1-d_2|/\sin\gamma$", fontsize=6.5)
    ax.text(-0.1, -0.62, r"$s<\tau$: X node; $s>\tau$: two T", fontsize=6.5)
    ax.set_xlim(-0.15, 2.0); ax.set_ylim(-0.75, 2.1); ax.set_aspect("equal"); ax.axis("off")
    ax.text(-0.15, 2.05, "(b)", fontsize=9, va="top")
    return fig


def gallery(args):
    """Networks at the four corners of the path for each target lattice (n = 8)."""
    columns = [(1.0, 1.0, r"SR corner, $\varepsilon_\theta=\varepsilon_x=1$"),
               (0.0, 1.0, r"aligned, $\varepsilon_\theta=0$, $\varepsilon_x=1$"),
               (0.0, 0.2, r"registered, $\varepsilon_\theta=0$, $\varepsilon_x=0.2$"),
               (0.0, 0.0, r"lattice, $\varepsilon_\theta=\varepsilon_x=0$")]
    fig, axs = plt.subplots(3, 4, figsize=(rtx.TEXT_WIDTH_IN, 5.55))
    for i, name in enumerate(["square", "triangular", "kagome"]):
        lat = tssf.make_lattice(name, 8)
        for j, (eth, ex, title) in enumerate(columns):
            net = tssf.Network(lat, eth, ex, w_min=lat.default_w_min, seed=10 + i).run()
            ax = tssf.plot_network(net, axs[i, j], linewidth=0.45, frame_linewidth=0.6)
            if i == 0:
                ax.set_title(title, fontsize=8)
            if j == 0:
                ax.text(-0.07, 0.5, name, rotation=90, va="center", ha="center", transform=ax.transAxes, fontsize=9)
            ax.text(0.5, -0.035, f"{net.n_ligaments} ligaments", ha="center", va="top",
                    transform=ax.transAxes, fontsize=7, color="0.35")
    return fig


def thickness(args):
    """SR thickness schedule on SR and on an exact square-lattice geometry."""
    fig, axs = plt.subplots(1, 3, figsize=rtx.fig_size(columns=1, ncols=3, panel_aspect=1.12))
    lat = tssf.square_lattice(16)
    sr = tssf.Network(seed=3).run(T=255)
    grid = tssf.Network(lat, 0.0, 0.0, w_min=lat.default_w_min, seed=3).run()
    for ax, net, lw, alpha, title in [(axs[0], sr, 2.6, 0.5, r"SR, $\alpha=0.5$"),
                                      (axs[1], grid, 2.6, 0.5, r"lattice, $\alpha=0.5$"),
                                      (axs[2], grid, 0.55, 0.0, r"lattice, $\alpha=0$")]:
        tssf.plot_network(net, ax, linewidth=lw, alpha=alpha, frame_linewidth=0.6).set_title(title, fontsize=8)
    return fig


def representation(args):
    """Ligament-graph (history) CCDFs of SR versus an exact 32x32 lattice built in random order."""
    lat = tssf.square_lattice(32)
    grids = [tssf.Network(lat, 0.0, 0.0, w_min=lat.default_w_min, seed=s).run() for s in range(3)]
    T = int(np.mean([g.n_ligaments for g in grids]))
    srs = [tssf.Network(seed=s).run(T=T) for s in range(3)]
    fig, axs = plt.subplots(1, 2, figsize=rtx.fig_size(columns=1, ncols=2, panel_aspect=1.05))

    def ccdf(x):
        x = np.sort(np.asarray(x, float))
        return x, 1 - np.arange(len(x)) / len(x)

    for label, nets, col in [("SR", srs, "0.15"), ("lattice, history", grids, "C0")]:
        lengths = np.concatenate([n.ligaments.length for n in nets])
        degrees = np.concatenate([tssf.struts(n, 1e-6, angle_tol=0.0).degrees for n in nets])  # unmerged
        axs[0].loglog(*ccdf(lengths), color=col, lw=1.0, label=label)
        axs[1].loglog(*ccdf(degrees), color=col, lw=1.0)
    merged = [tssf.struts(g, 1e-6) for g in grids]
    axs[0].axvline(np.mean([m.lengths.mean() for m in merged]), color="C3", lw=1.0, ls="--", label="lattice, struts")
    axs[1].axvline(np.mean([m.degrees.mean() for m in merged]), color="C3", lw=1.0, ls="--")
    xs, ks = np.array([0.05, 0.6]), np.array([6, 60.0])
    axs[0].loglog(xs, 0.01 * (xs / 0.6) ** -2, color="0.5", lw=0.6, ls=":")      # slope -2 guides
    axs[1].loglog(ks, 0.01 * (ks / 60) ** -2, color="0.5", lw=0.6, ls=":")
    axs[0].set(xlabel=r"length $\ell$", ylabel="CCDF")
    axs[1].set(xlabel=r"degree $k$")
    axs[0].legend(fontsize=6.5, frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.12), handlelength=1.4)
    for ax, letter in zip(axs, "ab"):
        ax.text(0.05, 0.04, f"({letter})", transform=ax.transAxes, fontsize=9, va="bottom")
    return fig


# ----------------------------------------------------------------------------- data figures
def order(args):
    """Tests of predictions P1-P4 (orientational / translational order, merging, z, rigidity, integrity)."""
    pos, orient = load(args.data, "positional"), load(args.data, "orient")
    size, rig = load(args.data, "orient_size"), load(args.data, "rigidity")
    fig, axs = plt.subplots(2, 3, figsize=rtx.fig_size(columns=2, nrows=2, ncols=3, panel_aspect=0.78))
    e = np.linspace(0, 1, 200)
    # (a) Psi along eps_theta vs sinc
    ax = axs[0, 0]
    eths = sorted({r["eps_th"] for r in orient})
    for lat, label in [("square", r"$\Psi_4$"), ("triangular", r"$\Psi_6$")]:
        ax.plot(eths, [mean_where(orient, lambda r: r["psi"], lattice=lat, eps_th=x) for x in eths],
                MARKER[lat], color=COLOR[lat], ms=3.5, mfc="none", mew=0.8, label=label)
    ax.plot(e, np.sinc(e), "k-", lw=0.8, label=r"$\mathrm{sinc}(\pi\varepsilon_\theta)$")
    ax.set(xlabel=r"$\varepsilon_\theta$ ($\varepsilon_x=1$, $T=3000$)", ylabel="orientational order")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    # (b) Phi along eps_x vs sinc (filled = registered regime)
    ax = axs[0, 1]
    exs = np.array(sorted({r["eps_x"] for r in pos}))
    for lat in COLOR:
        phi = np.array([mean_where(pos, lambda r: r["phi"], lattice=lat, eps_x=x) for x in exs])
        inside = exs < REGISTERED[lat] + 1e-9
        ax.plot(exs[inside], phi[inside], MARKER[lat], color=COLOR[lat], ms=3.5, label=lat)
        ax.plot(exs[~inside], phi[~inside], MARKER[lat], color=COLOR[lat], ms=3.5, mfc="none", mew=0.8)
    ax.plot(e, np.sinc(e), "k-", lw=0.8)
    ax.set(xlabel=r"$\varepsilon_x$ ($\varepsilon_\theta=0$, jammed)", ylabel=r"translational order $\Phi$")
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    # (c) merge fraction vs u = tau sin(gamma) / (eps_x h), normalised by its eps_x = 0 value
    ax = axs[0, 2]
    u = np.linspace(0, 1.3, 200)
    ax.plot(u, np.where(u >= 1, 1, 1 - (1 - np.minimum(u, 1)) ** 2), "k-", lw=0.8, label=r"$F(u)$")
    for lat in COLOR:
        f0 = mean_where(pos, lambda r: r["tau0.1"]["f"], lattice=lat, eps_x=0.0)
        for t in [0.05, 0.1] + ([] if lat == "kagome" else [0.2]):
            for x in exs[exs > 0]:
                f = mean_where(pos, lambda r: r[f"tau{t}"]["f"], lattice=lat, eps_x=x) / f0
                fill = COLOR[lat] if x < REGISTERED[lat] + 1e-9 else "none"
                ax.plot(t * SIN_GAMMA[lat] / x, f, MARKER[lat], color=COLOR[lat], ms=3, mfc=fill, mew=0.7)
    ax.set(xlim=(0, 1.3), xlabel=r"$u=\tau\sin\gamma/(\varepsilon_x h)$", ylabel=r"merge fraction $f/f_0$")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    # (d) coordination vs merge fraction; (e) floppy modes vs coordination (registered runs only)
    ff = np.linspace(0, 1, 100)
    axs[1, 0].plot(ff, 2 * (3 - ff) / (2 - ff), "k-", lw=0.8, label=r"$2(3-f)/(2-f)$")
    for lat in COLOR:
        taus = [0.05, 0.1] + ([] if lat == "kagome" else [0.2])
        for r in pos:
            if r["lattice"] == lat and r["eps_x"] <= REGISTERED[lat] + 1e-9:
                for t in taus:
                    axs[1, 0].plot(r[f"tau{t}"]["f"], r[f"tau{t}"]["z"], MARKER[lat], color=COLOR[lat], ms=2.6, alpha=0.7)
        for r in rig:
            if r["lattice"] == lat and r["eps_x"] <= REGISTERED[lat] + 1e-9:
                for t in taus:
                    axs[1, 1].plot(r[f"tau{t}"]["z"], r[f"tau{t}"]["floppy_per_node"], MARKER[lat],
                                   color=COLOR[lat], ms=2.8, alpha=0.75)
    axs[1, 0].axhline(4, color="0.5", lw=0.6, ls="--")
    axs[1, 0].text(0.02, 4.1, r"$z=2d$", fontsize=7, color="0.4")
    axs[1, 0].set(xlabel=r"merge fraction $f$", ylabel=r"interior coordination $z$")
    axs[1, 0].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.88))
    zz = np.linspace(3, 4, 50)
    axs[1, 1].plot(zz, 2 - zz / 2, "k-", lw=0.8, label=r"$2-z/2$")     # Maxwell count
    axs[1, 1].axvline(4, color="0.5", lw=0.6, ls="--")
    axs[1, 1].set(xlabel=r"interior coordination $z$", ylabel="floppy modes per node")
    axs[1, 1].legend(fontsize=7, frameon=False, loc="upper right")
    # (f) strut integrity: excess struts per lattice line vs (n-1)(1-f)/2
    ax = axs[1, 2]
    for n, col in zip([8, 16, 24], ["0.75", "0.45", "0.1"]):
        rows = [[mean_where(size, lambda r: r["tau0.1"][k], n=n, eps_th=x) for k in ("f", "merged_n")]
                for x in sorted({r["eps_th"] for r in size})]
        ax.plot([(n - 1) * (1 - f) / 2 for f, _ in rows], [m / (2 * (n - 1)) - 1 for _, m in rows],
                "o", color=col, ms=3, label=f"$n={n}$")
    ax.plot([0, 6], [0, 6], "k-", lw=0.8)
    ax.set(xlabel=r"$(n-1)(1-f)/2$", ylabel="excess struts per line")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    for ax, letter in zip(axs.ravel(), "abcdef"):
        rtx.panel_label(ax, f"({letter})", prefer="upper right" if letter in "abc" else "lower right")
    return fig


def mechanics(args):
    """Floor-corrected frame-model stiffness vs lambda/(eps_x h), and the floor itself."""
    res = load(args.data, "frame")
    fig, axs = plt.subplots(1, 3, figsize=rtx.fig_size(columns=2, ncols=3, panel_aspect=0.8))
    cmap = plt.get_cmap("viridis")
    for ax, lat, exs in [(axs[0], "square", [0.05, 0.1, 0.2, 0.3, 0.45]), (axs[1], "triangular", [0.05, 0.1, 0.2, 0.3])]:
        # lattice reference: exact lattice at lambda/h ~ 0.0126 (stretching-dominated, lambda-independent)
        ref = {k: mean_where(res, lambda r: r["res"][3][k] / r["res"][3]["rho"], lattice=lat, eps_x=0.0) for k in ("Ey", "Ex")}
        for i, ex in enumerate(exs):
            G = []
            for r in (r for r in res if r["lattice"] == lat and r["eps_x"] == ex):
                lam = np.array([d["lam_over_h"] for d in r["res"]])
                for k, floor in (("Ey", r["floor_y"]), ("Ex", r["floor_x"])):
                    G.append((np.array([d[k] / d["rho"] for d in r["res"]]) - floor) / (ref[k] - floor))
            G, col = np.array(G), cmap(i / (len(exs) - 0.5))
            ax.fill_between(lam / ex, G.min(0), G.max(0), color=col, alpha=0.18, lw=0)
            ax.semilogx(lam / ex, np.median(G, 0), "-o", color=col, ms=2, lw=0.9, label=rf"$\varepsilon_x={ex}$")
        ax.set(xlabel=r"$\lambda/(\varepsilon_x h)$", ylim=(-0.03, 1.05))
        ax.text(0.04, 0.96, lat, transform=ax.transAxes, va="top", fontsize=8)
        ax.legend(fontsize=6.5, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.62 if lat == "square" else 0.66))
    axs[0].set_ylabel(r"corrected stiffness $G$")
    ax = axs[2]
    for lat, mk, col in [("sr", "o", "0.2"), ("square", "s", "C0"), ("triangular", "^", "C1")]:
        rows = [r for r in res if r["lattice"] == lat and (lat == "sr" or r["eps_x"] >= 0.2)]   # floor regime
        pts = [(r[f"floor_{a}"], r["res"][0][f"E{a}"] / r["res"][0]["rho"]) for r in rows for a in "yx"]
        ax.plot(*np.array(pts).T, mk, color=col, ms=3, mfc="none", mew=0.7, label="SR" if lat == "sr" else lat)
    ax.plot([0, 0.3], [0, 0.3], "k-", lw=0.7)
    ax.set(xlabel=r"predicted floor $\sum\cos^3\psi/L_{\mathrm{tot}}$", ylabel=r"measured $E/(E_s\bar\rho)$")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    for ax, letter in zip(axs, "abc"):
        rtx.panel_label(ax, f"({letter})", prefer="lower right")
    return fig


FIGURES = dict(schematic=schematic, gallery=gallery, thickness=thickness, representation=representation,
               order=order, mechanics=mechanics)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figures", nargs="*", choices=list(FIGURES), help="default: all")
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rtx.set_revtex_style()
    for name in args.figures or FIGURES:
        rtx.savefig(FIGURES[name](args), os.path.join(args.out, f"fig_{name}.pdf"))
        print("wrote", os.path.join(args.out, f"fig_{name}.pdf"))


if __name__ == "__main__":
    main()
