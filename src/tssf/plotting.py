"""Matplotlib drawing of networks (matplotlib is imported lazily)."""
import numpy as np


def plot_network(net, ax=None, linewidth=0.6, alpha=0.0, color="k", frame_linewidth=0.8):
    """Draw the ligaments of ``net`` in the unit square.

    Line widths are ``linewidth * t**(-alpha)`` points (``t`` = insertion index), i.e. SR's
    thickness schedule; ``alpha = 0`` draws uniform struts. ``frame_linewidth = 0`` hides the
    domain boundary. Returns the axes.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    lig = net.ligaments
    widths = linewidth * np.arange(1, net.n_ligaments + 1) ** (-alpha) if alpha > 0 else linewidth
    ax.add_collection(LineCollection(np.stack([lig.start, lig.end], 1), linewidths=widths,
                                     colors=color, capstyle="butt"))
    if frame_linewidth > 0:
        ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color="k", lw=frame_linewidth)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax
