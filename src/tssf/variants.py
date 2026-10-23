"""Template-free variant: local division rules instead of a target lattice.

The cell to split is chosen with probability proportional to ``A**q`` (``q = inf``: the largest
cell), the point is ``x = c + s (u - c)`` with ``c`` the centroid and ``u`` uniform in the cell, and
the orientation follows a stochastic Errera rule ``p(theta) ~ exp(-kappa l(theta) / sqrt(A))``,
with ``l(theta)`` the chord length (``kappa = inf``: the shortest chord).

``(q, s, kappa) = (1, 1, 0)`` is the SR process. ``(inf, 0, inf)`` produces the exact
``2^k x 2^k`` square grid at ``t = 4^k - 1`` from the unit square (ties broken deterministically:
first angle, lowest cell index). This route cannot reach triangular or kagome order.
"""
import numpy as np

from .geometry import area_centroid, chord, random_point
from .network import Network


class ErreraNetwork(Network):
    """Sequential fragmentation with local (q, s, kappa) division rules.

    ``n_angles`` sets the angular resolution of the chord-length minimisation.
    """

    def __init__(self, q=np.inf, s=0.0, kappa=np.inf, n_angles=720, seed=None):
        super().__init__(lattice=None, seed=seed)
        self.q, self.s, self.kappa = q, s, kappa
        self.angles = np.arange(n_angles) * np.pi / n_angles

    def _propose(self):
        areas = np.array(self.areas)
        if np.isinf(self.q):
            i = int(np.flatnonzero(areas >= areas.max() * (1 - 1e-9))[0])
        else:
            w = areas ** self.q
            i = self.rng.choice(len(areas), p=w / w.sum())
        v, lab = self.cells[i], self.labels[i]
        centre = area_centroid(v)[1]
        x = centre + self.s * (random_point(v, self.rng) - centre)
        lengths = np.array([np.hypot(*(lambda r: r[2] - r[0])(chord(v, lab, x, np.array([np.cos(t), np.sin(t)]))))
                            for t in self.angles])
        if np.isinf(self.kappa):
            j = int(np.flatnonzero(lengths <= lengths.min() * (1 + 1e-9))[0])
        else:
            w = np.exp(-self.kappa * (lengths - lengths.min()) / np.sqrt(areas[i]))
            j = self.rng.choice(len(w), p=w / w.sum())
        return i, x, self.angles[j], -1
