"""Linear elastic frame model of a network skeleton dressed with struts of uniform width.

Every segment of the junction graph is a 2D Timoshenko beam of width ``lam`` and unit depth
(area ``A = lam``, second moment ``I = lam^3 / 12``), rigidly connected at the joints. The sample
(the unit square) is compressed uniaxially between rigid platens; lateral faces are free and the
domain sides are not struts. Moduli are in units of the solid modulus ``E_s``.

Slender-beam caveats: segments shorter than their width are outside beam theory, and the relative
density estimate ``lam * total length`` double-counts joint overlaps (~10 % at lam/h = 0.2).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .analysis import coarse_grain, junction_graph


def frame_graph(net, contract):
    """Junction graph restricted to ligaments, with joints closer than ``contract`` merged.

    Returns node positions and edges with unused nodes (e.g. bare domain corners) removed.
    """
    nodes, edges, carrier = junction_graph(net)
    pos, e, _ = coarse_grain(nodes, edges[carrier >= 0], contract)
    used = np.unique(e)
    index = -np.ones(len(pos), int)
    index[used] = np.arange(len(used))
    return pos[used], index[e]


def beam_stiffness(pos, edges, lam, E=1.0, nu=0.3, kappa=5 / 6):
    """Global stiffness matrix (3 dof per node: u, v, rotation) and the total strut length."""
    G = E / (2 * (1 + nu))
    A, I = lam, lam ** 3 / 12
    d = pos[edges[:, 1]] - pos[edges[:, 0]]
    L = np.hypot(d[:, 0], d[:, 1])
    rows, cols, vals = [], [], []
    for k, (l, c, s) in enumerate(zip(L, d[:, 0] / L, d[:, 1] / L)):
        phi = 12 * E * I / (kappa * G * A * l ** 2)      # shear-flexibility parameter
        b = E * I / ((1 + phi) * l ** 3)
        kl = np.zeros((6, 6))                            # local dofs: u1 v1 r1 u2 v2 r2
        kl[np.ix_([0, 3], [0, 3])] = E * A / l * np.array([[1, -1], [-1, 1]])
        kl[np.ix_([1, 2, 4, 5], [1, 2, 4, 5])] = b * np.array([
            [12, 6 * l, -12, 6 * l],
            [6 * l, (4 + phi) * l ** 2, -6 * l, (2 - phi) * l ** 2],
            [-12, -6 * l, 12, -6 * l],
            [6 * l, (2 - phi) * l ** 2, -6 * l, (4 + phi) * l ** 2]])
        rot = np.zeros((6, 6))
        rot[:2, :2] = rot[3:5, 3:5] = [[c, s], [-s, c]]
        rot[2, 2] = rot[5, 5] = 1
        dofs = np.r_[3 * edges[k, 0] + np.arange(3), 3 * edges[k, 1] + np.arange(3)]
        rows.append(np.repeat(dofs, 6))
        cols.append(np.tile(dofs, 6))
        vals.append((rot.T @ kl @ rot).ravel())
    n = 3 * len(pos)
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
    return K, L.sum()


def uniaxial_modulus(net, lam, axis="y", bonded=True, contract_frac=0.1, strain=1e-3):
    """Effective Young's modulus ``E / E_s`` under uniaxial compression, and ``rho = lam * sum(l)``.

    ``axis`` is the loading direction. With ``bonded=True`` the platens also block lateral slip
    (needed for inclined struts to carry load); joints closer than ``contract_frac * lam`` are
    merged first (results are insensitive to this choice in the 0.02-0.3 range).
    """
    pos, e = frame_graph(net, max(contract_frac * lam, 1e-9))
    if axis == "x":
        pos = pos[:, ::-1].copy()                        # mirror so that loading is along y
    K, total_length = beam_stiffness(pos, e, lam)
    bottom = np.nonzero(pos[:, 1] < 1e-9)[0]
    top = np.nonzero(pos[:, 1] > 1 - 1e-9)[0]
    fixed = {3 * i + 1: 0.0 for i in bottom}
    fixed.update({3 * i + 1: -strain for i in top})
    if bonded:
        fixed.update({3 * i: 0.0 for i in np.r_[bottom, top]})
    else:
        fixed[3 * bottom[np.argmin(pos[bottom, 0])]] = 0.0   # only remove rigid lateral translation
    fd = np.array(sorted(fixed))
    fv = np.array([fixed[k] for k in fd])
    free = np.setdiff1d(np.arange(K.shape[0]), fd)
    Kff, Kfd = K[free][:, free], K[free][:, fd]
    reg = 1e-14 * abs(Kff.diagonal()).max()              # guards against dangling mechanisms
    u = np.zeros(K.shape[0])
    u[fd] = fv
    u[free] = spla.spsolve((Kff + reg * sp.identity(len(free))).tocsc(), -Kfd @ fv)
    force = -(K @ u)[3 * top + 1].sum()                  # compressive reaction on the top platen
    return force / strain, lam * total_length


def spanning_floor(net, axis="y"):
    """Affine stiffness floor from ligaments that join the two loaded faces.

    Between bonded platens a straight strut at angle psi to the load axis is strained by
    eps cos^2 psi and adds ``E_s lam cos^3 psi`` to the modulus, so for lam -> 0
    ``E / (E_s rho) >= sum cos^3 psi / L_tot``. Returns ``(floor, number of spanning ligaments)``.
    Collinear stubs closer than the strut width also act as spanning struts; they are not counted.
    """
    lig = net.ligaments
    faces = {-1, -3} if axis == "y" else {-2, -4}
    k = 1 if axis == "y" else 0
    spanning = [i for i in range(net.n_ligaments) if {int(lig.host_start[i]), int(lig.host_end[i])} == faces]
    cos_psi = np.abs((lig.end - lig.start)[spanning, k]) / lig.length[spanning]
    return float(np.sum(cos_psi ** 3) / lig.length.sum()), len(spanning)
