"""Network representations and order parameters.

Two representations are provided:

* the *ligament graph* of Both et al. (node = ligament, i.e. one insertion event; edge = T-junction).
  It depends on construction history: a perfect lattice built in random order is heavy-tailed in it;
* the history-free *strut* representation at resolution ``tau``: junctions closer than ``tau``
  form one solid node, and ligament ends meeting anti-parallel at a node are one continuous strut.
  Physically ``tau`` is of the order of the strut thickness.
"""
from collections import namedtuple

import numpy as np
from scipy.spatial import cKDTree

#: End points of the four domain sides, keyed by their labels (see ``network.SIDES``).
_SIDE_SEGMENTS = {-1: ((0, 0), (1, 0)), -2: ((1, 0), (1, 1)), -3: ((1, 1), (0, 1)), -4: ((0, 1), (0, 0))}

#: Result of :func:`struts`.
Struts = namedtuple("Struts", "lengths degrees membership nodes edges cluster")


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[b] = a

    def roots(self):
        return np.array([self.find(i) for i in range(len(self.parent))])


# ---------------------------------------------------------------------------- representations
def ligament_graph(net):
    """Ligament-as-node statistics (Both et al.): lengths and degrees.

    The degree of a ligament counts its own ends that sit on other ligaments plus the ligaments
    that terminate on it.
    """
    lig = net.ligaments
    degree = np.zeros(net.n_ligaments, int)
    for host in (lig.host_start, lig.host_end):
        on_ligament = host >= 0
        degree[on_ligament] += 1
        np.add.at(degree, host[on_ligament], 1)
    return lig.length, degree


def junction_graph(net):
    """Planar graph of the skeleton.

    Nodes are the four domain corners followed by the two end points of every ligament
    (ligament ``i`` owns nodes ``4 + 2i`` and ``5 + 2i``). Edges join consecutive nodes along each
    ligament and each domain side. Returns ``nodes (N, 2)``, ``edges (E, 2)`` and the carrier of
    each edge (ligament index, or side label < 0).
    """
    lig = net.ligaments
    nodes = [np.array(c, float) for c in ((0, 0), (1, 0), (1, 1), (0, 1))]
    on_carrier = {c: [] for c in range(-4, net.n_ligaments)}
    for c, (a, b) in zip((-1, -2, -3, -4), ((0, 1), (1, 2), (2, 3), (3, 0))):
        on_carrier[c] += [a, b]
    for i in range(net.n_ligaments):
        for point, host in ((lig.start[i], lig.host_start[i]), (lig.end[i], lig.host_end[i])):
            on_carrier[i].append(len(nodes))
            on_carrier[int(host)].append(len(nodes))
            nodes.append(point)
    nodes = np.array(nodes)
    edges, carrier = [], []
    for c, ids in on_carrier.items():
        if len(ids) < 2:
            continue
        a, b = (lig.start[c], lig.end[c]) if c >= 0 else (np.array(p, float) for p in _SIDE_SEGMENTS[c])
        u = (b - a) / np.hypot(*(b - a))
        ids = np.array(ids)[np.argsort((nodes[ids] - a) @ u)]          # order along the carrier
        edges += list(zip(ids[:-1], ids[1:]))
        carrier += [c] * (len(ids) - 1)
    return nodes, np.array(edges), np.array(carrier)


def coarse_grain(nodes, edges, tau):
    """Merge all nodes closer than ``tau`` (single linkage) into solid nodes.

    Returns the cluster positions (mean of members), the unique edges between distinct
    clusters, and the node -> cluster map. Keep ``tau`` well below the smallest lattice edge:
    single linkage chains, and clusters percolate for large ``tau``.
    """
    uf = _UnionFind(len(nodes))
    if tau > 0:
        for a, b in cKDTree(nodes).query_pairs(tau):
            uf.union(a, b)
    _, cluster = np.unique(uf.roots(), return_inverse=True)
    pos = np.zeros((cluster.max() + 1, 2))
    np.add.at(pos, cluster, nodes)
    pos /= np.bincount(cluster)[:, None]
    e = cluster[edges]
    e = np.unique(np.sort(e[e[:, 0] != e[:, 1]], 1), axis=0)
    return pos, e, cluster


def struts(net, tau, angle_tol=np.deg2rad(10)):
    """History-free strut representation at resolution ``tau``.

    Two ligament ends that meet at the same solid node with anti-parallel directions (within
    ``angle_tol``) are joined into one strut; candidate pairs are matched greedily, most
    anti-parallel first. Returns a ``Struts`` tuple with strut lengths and degrees (number of
    distinct struts sharing a node), the ligament -> strut map, and the coarse-grained junction
    graph (``nodes``, ``edges``, node -> cluster map ``cluster``).
    """
    lig, n = net.ligaments, net.n_ligaments
    nodes, edges, carrier = junction_graph(net)
    pos, e, cluster = coarse_grain(nodes, edges, max(tau, 1e-9))
    # outward directions of every ligament end, grouped by the solid node the end belongs to
    ends = {}
    for i in range(n):
        d = (lig.end[i] - lig.start[i]) / lig.length[i]
        ends.setdefault(cluster[4 + 2 * i], []).append((i, d))
        ends.setdefault(cluster[5 + 2 * i], []).append((i, -d))
    uf = _UnionFind(n)
    for group in ends.values():
        pairs = sorted((da @ db, a, b) for a, (ia, da) in enumerate(group) for b, (ib, db) in enumerate(group)
                       if a < b and ia != ib and da @ db < -np.cos(angle_tol))
        used = set()
        for _, a, b in pairs:
            if a not in used and b not in used:
                used.update((a, b))
                uf.union(group[a][0], group[b][0])
    _, membership = np.unique(uf.roots(), return_inverse=True)
    lengths = np.bincount(membership, weights=lig.length)
    # strut adjacency: struts touching the same solid node
    struts_at = {}
    for (a, b), c in zip(edges, carrier):
        if c >= 0:
            for x in (a, b):
                struts_at.setdefault(cluster[x], set()).add(membership[c])
    neighbours = [set() for _ in range(len(lengths))]
    for group in struts_at.values():
        for x in group:
            neighbours[x] |= group - {x}
    degrees = np.array([len(s) for s in neighbours])
    return Struts(lengths, degrees, membership, pos, e, cluster)


def merge_fraction(net, s):
    """Fraction of interior ligament ends (ends on another ligament) paired into continuous struts."""
    lig = net.ligaments
    interior_ends = np.sum(lig.host_start >= 0) + np.sum(lig.host_end >= 0)
    return 2 * (net.n_ligaments - len(s.lengths)) / max(interior_ends, 1)


# ---------------------------------------------------------------------------- order parameters
def orientational_order(net, m):
    """Length-weighted ``|<exp(i 2m theta)>|``: tetratic (m = 2) or hexatic (m = 3) order."""
    lig = net.ligaments
    return float(np.abs(np.sum(lig.length * np.exp(2j * m * lig.theta))) / lig.length.sum())


def translational_order(net, lattice):
    """Length-weighted ``|<exp(2 pi i d / h)>|`` with ``d`` the offset of each ligament midpoint
    from the nearest line of its nearest director family (order relative to the template)."""
    lig = net.ligaments
    z = 0j
    for i in range(net.n_ligaments):
        f, _ = lattice.nearest_family(lig.theta[i])
        d = lattice.offset((lig.start[i] + lig.end[i]) / 2, f)
        z += lig.length[i] * np.exp(2j * np.pi * d / lattice.spacings[f])
    return float(abs(z) / lig.length.sum())


def on_boundary(pos, tol=1e-9):
    """Mask of points on the boundary of the unit square."""
    return (pos[:, 0] < tol) | (pos[:, 0] > 1 - tol) | (pos[:, 1] < tol) | (pos[:, 1] > 1 - tol)


def mean_coordination(nodes, edges, interior_only=True):
    """Mean node degree ``z`` (by default over interior nodes) and the degree array used."""
    degree = np.bincount(edges.ravel(), minlength=len(nodes))
    if interior_only:
        degree = degree[~on_boundary(nodes)]
    return float(degree.mean()), degree


def rigidity(nodes, edges, rel_tol=1e-9):
    """Maxwell-Calladine counts of a pin-jointed network.

    Builds the central-force rigidity matrix and returns ``(floppy modes, states of self-stress)``
    with floppy = 2N - rank - 3 (rigid-body modes removed). Dense SVD: fine up to a few thousand
    nodes.
    """
    n, n_e = len(nodes), len(edges)
    if n_e == 0 or n < 2:
        return np.nan, -1
    d = nodes[edges[:, 1]] - nodes[edges[:, 0]]
    d /= np.hypot(d[:, 0], d[:, 1])[:, None]
    R = np.zeros((n_e, 2 * n))
    rows = np.arange(n_e)
    R[rows, 2 * edges[:, 0]], R[rows, 2 * edges[:, 0] + 1] = -d[:, 0], -d[:, 1]
    R[rows, 2 * edges[:, 1]], R[rows, 2 * edges[:, 1] + 1] = d[:, 0], d[:, 1]
    sv = np.linalg.svd(R, compute_uv=False)
    rank = int(np.sum(sv > rel_tol * sv.max()))
    return 2 * n - rank - 3, n_e - rank


# ---------------------------------------------------------------------------- statistics helpers
def tail_exponent(x, xmin):
    """Maximum-likelihood (Hill) exponent of a power-law density tail ``p(x) ~ x^-a`` for x >= xmin."""
    x = np.asarray(x, float)
    x = x[x >= xmin]
    return 1 + len(x) / np.sum(np.log(x / xmin))


def cv(x):
    """Coefficient of variation."""
    x = np.asarray(x, float)
    return float(x.std() / x.mean())
