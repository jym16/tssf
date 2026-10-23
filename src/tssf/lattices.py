"""Target line lattices (periodic line arrangements) for template snapping.

A line lattice is a union of ``m`` families of parallel lines. Family ``f`` has direction angle
``phi_f``, unit normal ``n_f = (-sin phi_f, cos phi_f)``, spacing ``h_f`` and offset ``o_f``:

    L_f = { x : n_f . x = o_f + k h_f,  k integer }.

Square (m = 2), triangular and kagome (m = 3) lattices are all of this form; honeycomb and Voronoi
networks are not (their vertices are Y-junctions), so they cannot be reached by straight-chord
fragmentation.
"""
import numpy as np


class LineLattice:
    """A periodic arrangement of ``m`` families of parallel lines.

    Parameters
    ----------
    name : str
    phis : sequence of float
        Direction angle of each family (radians, axial, i.e. defined mod pi).
    spacings, offsets : sequence of float
        Line spacing ``h_f`` and offset ``o_f`` of each family.
    face_width : float
        Smallest width of an unclipped lattice face. Used for the default sliver cutoff
        ``w_min = face_width / 2`` (registered regime, see the report, Prop. 2).
    """

    def __init__(self, name, phis, spacings, offsets, face_width):
        self.name = name
        self.phis = np.asarray(phis, float)
        self.spacings = np.asarray(spacings, float)
        self.offsets = np.asarray(offsets, float)
        self.normals = np.stack([-np.sin(self.phis), np.cos(self.phis)], 1)
        self.face_width = float(face_width)

    @property
    def m(self):
        """Number of line families (2 for square, 3 for triangular / kagome)."""
        return len(self.phis)

    @property
    def h(self):
        """Spacing of family 0 (all families share it for the built-in lattices)."""
        return float(self.spacings[0])

    @property
    def default_w_min(self):
        """Sliver cutoff used throughout the report: half the smallest face width."""
        return 0.5 * self.face_width

    def nearest_family(self, theta):
        """Family whose direction is closest to the axial angle ``theta``, and the signed deviation."""
        best, best_d = 0, 9.0
        for f, phi in enumerate(self.phis):
            d = (theta - phi + np.pi / 2) % np.pi - np.pi / 2   # wrap to [-pi/2, pi/2)
            if abs(d) < abs(best_d):
                best, best_d = f, d
        return best, best_d

    def offset(self, x, f):
        """Signed distance of point ``x`` from the nearest line of family ``f``."""
        s = self.normals[f] @ x
        k = np.round((s - self.offsets[f]) / self.spacings[f])
        return s - (self.offsets[f] + k * self.spacings[f])

    def __repr__(self):
        return f"LineLattice({self.name!r}, m={self.m}, h={self.h:.4g})"


def square_lattice(n):
    """Square lattice with ``n`` cells per side of the unit square (spacing ``h = 1/n``)."""
    h = 1.0 / n
    return LineLattice(f"square{n}", [0.0, np.pi / 2], [h, h], [0.0, 0.0], face_width=h)


def triangular_lattice(n, shift=0.0):
    """Three line families at 0, 60 and 120 degrees with lattice constant ``a = 1/n``.

    All families share the spacing ``h = a*sqrt(3)/2`` and pass through a lattice point at the
    centre of the unit square. ``shift`` (in units of ``h``) displaces family 0:
    ``0`` gives the triangular lattice, ``1/2`` the kagome lattice, and values in between the
    breathing kagome lattice (a continuous path within the lattice class).
    """
    a = 1.0 / n
    h = a * np.sqrt(3) / 2
    phis = np.array([0.0, np.pi / 3, 2 * np.pi / 3])
    normals = np.stack([-np.sin(phis), np.cos(phis)], 1)
    offsets = normals @ np.array([0.5, 0.5])          # concurrent at the square's centre
    offsets[0] += shift * h
    s = shift % 1.0
    if s == 0:
        name, face_width = "triangular", h            # triangle height
    else:
        # smallest triangle has height min(s, 1-s) h  (kagome: h/2)
        name = "kagome" if abs(s - 0.5) < 1e-12 else f"breathing{s:.2f}"
        face_width = min(s, 1 - s) * h
    return LineLattice(f"{name}{n}", phis, [h, h, h], offsets, face_width=face_width)


def kagome_lattice(n):
    """Kagome lattice: the triangular arrangement with one family shifted by half a spacing."""
    return triangular_lattice(n, shift=0.5)


def make_lattice(name, n):
    """Build a lattice by name: ``'square'``, ``'triangular'`` or ``'kagome'``."""
    builders = {"square": square_lattice, "triangular": triangular_lattice, "kagome": kagome_lattice}
    if name not in builders:
        raise ValueError(f"unknown lattice {name!r}; choose from {sorted(builders)}")
    return builders[name](n)
