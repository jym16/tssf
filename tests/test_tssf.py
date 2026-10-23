"""Fast physics and consistency checks for tssf (run with ``pytest``; ~1 min)."""
import numpy as np
import pytest

import tssf
from tssf.geometry import area_centroid, clip_halfplane


def test_clip_conserves_area_and_labels():
    """Splitting a convex polygon by a line conserves area; the cut edge gets the new label."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        ang = np.sort(rng.random(7)) * 2 * np.pi
        v = np.c_[np.cos(ang), np.sin(ang)]
        lab = list(range(7))
        p, t = rng.normal(0, 0.2, 2), rng.random() * np.pi
        u = np.array([np.cos(t), np.sin(t)])
        left, ll = clip_halfplane(v, lab, p, u, +1, 99)
        right, rl = clip_halfplane(v, lab, p, u, -1, 99)
        if len(left) >= 3 and len(right) >= 3:
            assert area_centroid(left)[0] + area_centroid(right)[0] == pytest.approx(area_centroid(v)[0])
            assert ll.count(99) == 1 and rl.count(99) == 1


def test_seed_reproducibility():
    a, b = tssf.Network(seed=4).run(T=200), tssf.Network(seed=4).run(T=200)
    assert np.array_equal(a.ligaments.start, b.ligaments.start)
    assert not np.array_equal(a.ligaments.start, tssf.Network(seed=5).run(T=200).ligaments.start)


def test_sr_length_decay():
    """<l_t> ~ l0 t^(-1/2) with l0 ~ 1 (Both et al.)."""
    lengths = tssf.Network(seed=1).run(T=3000).ligaments.length
    t = np.arange(1, len(lengths) + 1)
    assert np.mean(lengths[100:] * np.sqrt(t[100:])) == pytest.approx(1.05, abs=0.15)


def test_sr_has_no_collinear_continuations():
    """In SR, history ligaments are struts almost surely."""
    net = tssf.Network(seed=2).run(T=500)
    assert len(tssf.struts(net, 1e-6).lengths) == net.n_ligaments


@pytest.mark.parametrize("n", [4, 6])
def test_exact_square_lattice(n):
    """eps = 0 + jamming gives the n x n grid; its strut graph is K_(n-1, n-1)."""
    lat = tssf.square_lattice(n)
    net = tssf.Network(lat, 0.0, 0.0, w_min=lat.default_w_min, seed=3).run()
    assert np.allclose(net.cell_areas, 1 / n ** 2)
    s = tssf.struts(net, 1e-6)
    assert len(s.lengths) == 2 * (n - 1)
    assert np.allclose(s.lengths, 1.0) and np.all(s.degrees == n - 1)


@pytest.mark.parametrize("name, z", [("triangular", 6), ("kagome", 4)])
def test_exact_lattice_coordination(name, z):
    """Interior vertices are 6-fold (triangular) or 4-fold (kagome); a few near the clipped
    boundary can miss a strut excluded by w_min, so test the typical degree and the mean."""
    lat = tssf.make_lattice(name, 12)
    net = tssf.Network(lat, 0.0, 0.0, w_min=lat.default_w_min, seed=1).run()
    s = tssf.struts(net, 1e-6)
    zbar, deg = tssf.mean_coordination(s.nodes, s.edges)
    assert np.bincount(deg).argmax() == z and zbar == pytest.approx(z, abs=0.1)


def test_registered_square_identity():
    """In the registered regime every lattice line is one strut once tau >= eps_x h (F(u) = 1),
    and z = 2(3 - f)/(2 - f) holds for pairwise merging."""
    lat = tssf.square_lattice(8)
    net = tssf.Network(lat, 0.0, 0.2, w_min=lat.default_w_min, seed=2).run()
    assert len(tssf.struts(net, 0.2 * lat.h).lengths) == 14
    s = tssf.struts(net, 0.05 * lat.h)
    f = tssf.merge_fraction(net, s)
    assert tssf.mean_coordination(s.nodes, s.edges)[0] == pytest.approx(2 * (3 - f) / (2 - f))


def test_orientational_sinc_law():
    """Psi_4 = sinc(pi eps_theta) along the SR-faithful orientational path (P1)."""
    lat = tssf.square_lattice(12)
    net = tssf.Network(lat, 0.5, 1.0, seed=7).run(T=2000)
    assert tssf.orientational_order(net, 2) == pytest.approx(np.sinc(0.5), abs=0.05)


def test_rigidity_counts():
    tri = np.array([[0, 0], [1, 0], [0.5, 1.0]])
    assert tssf.rigidity(tri, np.array([[0, 1], [1, 2], [2, 0]])) == (0, 0)
    sq = np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]])
    assert tssf.rigidity(sq, np.array([[0, 1], [1, 2], [2, 3], [3, 0]]))[0] == 1


def test_frame_model_square_lattice():
    """Stretching-dominated square lattice: E / (E_s rho) = 1/2 along an axis."""
    lat = tssf.square_lattice(6)
    net = tssf.Network(lat, 0.0, 0.0, w_min=lat.default_w_min, seed=0).run()
    E, rho = tssf.uniaxial_modulus(net, 0.005, axis="y")
    assert E / rho == pytest.approx(0.5, rel=1e-6)


def test_spanning_floor_matches_frame_model():
    """For slender struts the SR stiffness equals the spanning-strut floor (seed with a spanning chord)."""
    net = tssf.Network(seed=15000).run(T=150)
    floor, n_span = tssf.spanning_floor(net, "y")
    E, rho = tssf.uniaxial_modulus(net, 0.000316, axis="y")
    assert n_span > 0 and E / rho == pytest.approx(floor, rel=0.02)


def test_errera_dyadic_grid():
    net = tssf.ErreraNetwork(seed=0).run(T=63)
    assert np.allclose(net.cell_areas, 1 / 64)
