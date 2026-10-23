"""Convex-polygon primitives used by the fragmentation process.

Cells are convex polygons stored as ``(k, 2)`` vertex arrays in counter-clockwise order, together
with a list of ``k`` integer *edge labels*: ``labels[i]`` identifies what edge ``v[i] -> v[i+1]``
lies on (a ligament index ``>= 0`` or a domain side ``-1..-4``). Labels let us record which
ligament a new ligament terminates on, i.e. the T-junction topology.
"""
import numpy as np

ON_LINE_TOL = 1e-11     # |signed distance| below which a vertex is treated as lying on a cut line
DUPLICATE_TOL = 1e-13   # vertices closer than this are merged after clipping


def area_centroid(v):
    """Signed area (positive for CCW) and centroid of a simple polygon (shoelace formula)."""
    x, y = v[:, 0], v[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area = cross.sum() / 2
    centroid = np.array([((x + xn) * cross).sum(), ((y + yn) * cross).sum()]) / (6 * area)
    return area, centroid


def min_width(v):
    """Minimal width of a convex polygon.

    For a convex polygon the minimal width is attained with one side flush against a supporting
    line, so it is the minimum over sides of the largest vertex distance from that side.
    """
    best = np.inf
    for i in range(len(v)):
        e = v[(i + 1) % len(v)] - v[i]
        length = np.hypot(*e)
        if length < 1e-15:
            continue
        normal = np.array([-e[1], e[0]]) / length
        best = min(best, np.max(np.abs((v - v[i]) @ normal)))
    return best


def random_point(v, rng):
    """Uniform random point in a convex polygon (fan triangulation from ``v[0]``)."""
    a, b, c = v[0], v[1:-1], v[2:]
    tri_area = 0.5 * np.abs((b[:, 0] - a[0]) * (c[:, 1] - a[1]) - (b[:, 1] - a[1]) * (c[:, 0] - a[0]))
    j = rng.choice(len(tri_area), p=tri_area / tri_area.sum())
    r1, r2 = rng.random(2)
    if r1 + r2 > 1:                      # reflect into the triangle
        r1, r2 = 1 - r1, 1 - r2
    return a + r1 * (b[j] - a) + r2 * (c[j] - a)


def strictly_inside(v, p, tol=1e-12):
    """True if ``p`` lies strictly inside the CCW convex polygon ``v`` (not on its boundary)."""
    e = np.roll(v, -1, 0) - v
    w = p - v
    cross = e[:, 0] * w[:, 1] - e[:, 1] * w[:, 0]
    return bool(np.all(cross > tol * np.hypot(e[:, 0], e[:, 1])))


def clip_halfplane(v, labels, p, u, side, cut_label):
    """Part of convex polygon ``v`` on one side of the line through ``p`` with direction ``u``.

    ``side = +1`` keeps the left half-plane (cross(u, x - p) >= 0), ``-1`` the right one.
    Returns the clipped vertices and their edge labels; edges created along the cut get
    ``cut_label``. This is Sutherland-Hodgman clipping with label bookkeeping.
    """
    s = side * (u[0] * (v[:, 1] - p[1]) - u[1] * (v[:, 0] - p[0]))
    s = np.where(np.abs(s) < ON_LINE_TOL, 0.0, s)
    out_v, out_l = [], []
    for i in range(len(v)):
        j = (i + 1) % len(v)
        si, sj = s[i], s[j]
        if si >= 0:                                  # keep v[i]; label of the edge leaving it:
            out_v.append(v[i])
            # the original edge if it stays inside, otherwise the cut (v[i] on the line, next vertex out)
            out_l.append(cut_label if (sj < 0 and si == 0) else labels[i])
        if (si > 0 > sj) or (si < 0 < sj):           # edge crosses the line: add the intersection
            out_v.append(v[i] + si / (si - sj) * (v[j] - v[i]))
            out_l.append(cut_label if si > 0 else labels[i])   # leaving -> cut edge; entering -> original
    # drop (near-)duplicate consecutive vertices, keeping the label of the edge that follows
    kv, kl = [], []
    for x, lab in zip(out_v, out_l):
        if kv and np.hypot(*(x - kv[-1])) < DUPLICATE_TOL:
            kl[-1] = lab
            continue
        kv.append(x)
        kl.append(lab)
    if len(kv) > 1 and np.hypot(*(kv[0] - kv[-1])) < DUPLICATE_TOL:
        kv.pop()
        kl.pop()
    return np.array(kv), kl


def chord(v, labels, p, u):
    """Chord of a convex polygon through interior point ``p`` along direction ``u``.

    Returns ``(P, host_P, Q, host_Q)``: the two endpoints (behind / ahead of ``p``) and the labels
    of the edges they land on, i.e. the ligaments or domain sides the new ligament terminates on.
    """
    t_back, t_ahead, h_back, h_ahead = -np.inf, np.inf, None, None
    for i in range(len(v)):
        a, e = v[i], v[(i + 1) % len(v)] - v[i]
        den = u[0] * e[1] - u[1] * e[0]              # cross(u, e); zero if parallel
        if abs(den) < 1e-15:
            continue
        w = a - p
        t = (w[0] * e[1] - w[1] * e[0]) / den        # position along the chord
        s = (w[0] * u[1] - w[1] * u[0]) / den        # position along the edge, in [0, 1] if hit
        if -1e-9 <= s <= 1 + 1e-9:
            if 0 < t < t_ahead:
                t_ahead, h_ahead = t, labels[i]
            if t_back < t < 0:
                t_back, h_back = t, labels[i]
    return p + t_back * u, h_back, p + t_ahead * u, h_ahead
