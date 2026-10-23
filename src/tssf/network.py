"""The generative model: scale-rich (SR) sequential fragmentation with template snapping (TSSF).

SR process (Both et al., arXiv:2511.18108), zero-thickness skeleton
    At each step draw a point x uniformly in the void and an angle theta uniformly in [0, pi);
    grow a straight ligament through x in both directions until it hits an existing ligament or
    the boundary of the unit square. Every junction is therefore a T-junction, and the void is
    always a set of convex cells; inserting a ligament splits one cell into two.

Template snapping (this work)
    Before growing, the proposal is pulled toward the nearest line of a target ``LineLattice``:

        theta' = phi_f + eps_theta * (theta - phi_f)          f = nearest director family
        x'     = x - (1 - eps_x) * d * n_f                    d = offset of x from nearest line of f

    eps_theta = eps_x = 1 is exactly SR; eps_theta = eps_x = 0 puts every ligament on a lattice
    line, and the jammed state is then the lattice itself (Prop. 1 of the report).

A split is rejected if either daughter cell is narrower than ``w_min`` (sliver exclusion). The run
stops after ``T`` accepted ligaments, or at jamming (``patience`` consecutive rejections).
Thickness does not enter the dynamics here: ``Network.thickness`` assigns SR's schedule
lambda_t = lambda_0 t^(-alpha) to the ligaments afterwards.
"""
from collections import namedtuple

import numpy as np

from .geometry import area_centroid, chord, clip_halfplane, min_width, random_point, strictly_inside

#: Labels of the four sides of the unit square (bottom, right, top, left), used as ligament hosts.
SIDES = (-1, -2, -3, -4)

#: Array view of all ligaments; ``host_*`` is the ligament index (or side label < 0) an end sits on.
Ligaments = namedtuple("Ligaments", "start end host_start host_end theta family offset length")


class Network:
    """A growing network in the unit square.

    Parameters
    ----------
    lattice : LineLattice or None
        Target lattice for snapping; ``None`` gives the plain SR process.
    eps_theta, eps_x : float in [0, 1]
        Orientational and positional disorder amplitudes (1 = SR, 0 = fully snapped).
    w_min : float
        Sliver cutoff: minimal admissible width of a daughter cell (0 disables it).
    seed : int or None
        Seed for ``numpy.random.default_rng``; equal seeds give identical networks.
    """

    def __init__(self, lattice=None, eps_theta=1.0, eps_x=1.0, w_min=0.0, seed=None):
        self.lattice = lattice
        self.eps_theta, self.eps_x, self.w_min = float(eps_theta), float(eps_x), float(w_min)
        self.rng = np.random.default_rng(seed)
        # void cells (convex polygons), their edge labels, areas and bounding boxes
        self.cells = [np.array([[0, 0], [1, 0], [1, 1], [0, 1]], float)]
        self.labels = [list(SIDES)]
        self.areas = [1.0]
        self.bboxes = [np.array([0, 0, 1, 1.0])]
        self._ligaments = []          # tuples (start, end, host_start, host_end, theta, family, offset)
        self._cache = None
        self.n_proposals = self.n_rejected = 0

    # ------------------------------------------------------------------ proposals
    @property
    def is_sr(self):
        """True when snapping is the identity map (no lattice, or eps_theta = eps_x = 1)."""
        return self.lattice is None or (self.eps_theta == 1.0 and self.eps_x == 1.0)

    def _sr_proposal(self):
        """SR law: a cell chosen with probability proportional to area, a uniform point in it,
        and a uniform axial angle (this is the same as a uniform point in the whole void)."""
        areas = np.array(self.areas)
        i = self.rng.choice(len(areas), p=areas / areas.sum())
        return i, random_point(self.cells[i], self.rng), self.rng.random() * np.pi

    def _propose(self):
        """Return ``(cell index or None, point, angle, family)`` of the next proposal."""
        i, x, theta = self._sr_proposal()
        if self.is_sr:
            return i, x, theta, -1
        lat = self.lattice
        f, dtheta = lat.nearest_family(theta)
        theta = lat.phis[f] + self.eps_theta * dtheta
        x = x - (1.0 - self.eps_x) * lat.offset(x, f) * lat.normals[f]
        if self.eps_x == 1.0:            # point not moved: still in cell i
            return i, x, theta, f
        return self._locate(x), x, theta, f

    def _locate(self, p):
        """Index of the cell containing ``p`` strictly inside, or ``None`` (``p`` on a ligament)."""
        b = np.array(self.bboxes)
        candidates = np.nonzero((b[:, 0] <= p[0]) & (b[:, 2] >= p[0]) & (b[:, 1] <= p[1]) & (b[:, 3] >= p[1]))[0]
        for j in candidates:
            if strictly_inside(self.cells[j], p):
                return j
        return None

    def _admissible(self, left, right):
        """Both daughters are proper polygons of positive area and not thinner than ``w_min``."""
        if len(left) < 3 or len(right) < 3:
            return False
        if area_centroid(left)[0] < 1e-14 or area_centroid(right)[0] < 1e-14:
            return False
        return self.w_min <= 0 or (min_width(left) >= self.w_min and min_width(right) >= self.w_min)

    # ------------------------------------------------------------------ dynamics
    def step(self):
        """Make one proposal; insert the ligament if admissible. Returns True if accepted."""
        self.n_proposals += 1
        i, x, theta, f = self._propose()
        if i is None or not strictly_inside(self.cells[i], x):
            self.n_rejected += 1
            return False
        v, lab = self.cells[i], self.labels[i]
        u = np.array([np.cos(theta), np.sin(theta)])
        new = len(self._ligaments)                       # label of the new ligament
        left, left_lab = clip_halfplane(v, lab, x, u, +1, new)
        right, right_lab = clip_halfplane(v, lab, x, u, -1, new)
        if not self._admissible(left, right):
            self.n_rejected += 1
            return False
        start, h_start, end, h_end = chord(v, lab, x, u)
        offset = self.lattice.offset(x, f) if f >= 0 else np.nan
        self._ligaments.append((start, end, h_start, h_end, theta % np.pi, f, offset))
        # replace the parent cell by the left daughter and append the right one
        self.cells[i], self.labels[i], self.areas[i] = left, left_lab, area_centroid(left)[0]
        self.bboxes[i] = np.r_[left.min(0), left.max(0)]
        self.cells.append(right)
        self.labels.append(right_lab)
        self.areas.append(area_centroid(right)[0])
        self.bboxes.append(np.r_[right.min(0), right.max(0)])
        self._cache = None
        return True

    def run(self, T=None, patience=3000, max_ligaments=200_000, exact_check=None):
        """Grow the network.

        Parameters
        ----------
        T : int or None
            Stop after ``T`` accepted ligaments (SR-faithful mode). If ``None``, run to jamming.
        patience : int
            Jamming criterion: stop after this many consecutive rejections.
        max_ligaments : int
            Safety cap on the number of ligaments (and on 50x as many proposals).
        exact_check : bool or None
            When jamming is declared, verify that no admissible lattice-line split remains and
            continue otherwise (up to 20 times). Default: on for the exact lattice limit
            (``eps_theta = eps_x = 0``), where acceptances become rare near saturation.
        """
        if exact_check is None:
            exact_check = self.lattice is not None and self.eps_theta == 0 and self.eps_x == 0
        streak = checks = 0
        while True:
            streak = 0 if self.step() else streak + 1
            if T is not None and self.n_ligaments >= T:
                break
            if T is None and streak >= patience:
                if exact_check and checks < 20 and not self.is_saturated():
                    checks, streak = checks + 1, 0
                    continue
                break
            if self.n_ligaments >= max_ligaments or self.n_proposals > 50 * max_ligaments:
                break
        return self

    def is_saturated(self):
        """True if no cell can be split admissibly along a line of the target lattice."""
        lat = self.lattice
        for v, lab in zip(self.cells, self.labels):
            for f in range(lat.m):
                s = v @ lat.normals[f]           # line coordinates n_f . x of the cell's vertices
                k_lo = np.ceil((s.min() - lat.offsets[f]) / lat.spacings[f] + 1e-7)
                k_hi = np.floor((s.max() - lat.offsets[f]) / lat.spacings[f] - 1e-7)
                u = np.array([np.cos(lat.phis[f]), np.sin(lat.phis[f])])
                for k in np.arange(k_lo, k_hi + 1):          # lattice lines crossing the interior
                    p = (lat.offsets[f] + k * lat.spacings[f]) * lat.normals[f]
                    left, _ = clip_halfplane(v, lab, p, u, +1, -9)
                    right, _ = clip_halfplane(v, lab, p, u, -1, -9)
                    if self._admissible(left, right):
                        return False
        return True

    # ------------------------------------------------------------------ results
    @property
    def n_ligaments(self):
        return len(self._ligaments)

    @property
    def ligaments(self):
        """All ligaments as a ``Ligaments`` namedtuple of arrays (in insertion order)."""
        if self._cache is None:
            cols = list(zip(*self._ligaments)) if self._ligaments else [[]] * 7
            start, end = np.array(cols[0]).reshape(-1, 2), np.array(cols[1]).reshape(-1, 2)
            self._cache = Ligaments(start, end, np.array(cols[2], int), np.array(cols[3], int),
                                    np.array(cols[4]), np.array(cols[5], int), np.array(cols[6]),
                                    np.hypot(*(end - start).T))
        return self._cache

    @property
    def cell_areas(self):
        return np.array(self.areas)

    def thickness(self, lambda0, alpha):
        """SR thickness schedule lambda_t = lambda0 * t**(-alpha), t = 1, 2, ... insertion index."""
        return lambda0 * np.arange(1, self.n_ligaments + 1, dtype=float) ** (-alpha)

    def __repr__(self):
        target = self.lattice.name if self.lattice is not None else "SR"
        return (f"Network(target={target}, eps_theta={self.eps_theta}, eps_x={self.eps_x}, "
                f"w_min={self.w_min:.4g}, ligaments={self.n_ligaments})")
