"""tssf: scale-rich network materials and their continuous interpolation to line lattices.

Template-snapped sequential fragmentation (TSSF) extends the scale-rich (SR) model of
Both et al. (arXiv:2511.18108). Each SR proposal is pulled toward a target lattice with
orientational / positional amplitudes ``eps_theta``, ``eps_x``: 1 gives SR, 0 the exact lattice.

Quick start::

    import tssf
    lat = tssf.square_lattice(12)
    net = tssf.Network(lat, eps_theta=0.0, eps_x=0.2, w_min=lat.default_w_min, seed=1).run()
    s = tssf.struts(net, tau=0.1 * lat.h)
    print(net.n_ligaments, tssf.merge_fraction(net, s), tssf.translational_order(net, lat))
"""
from .analysis import (Struts, coarse_grain, cv, junction_graph, ligament_graph, mean_coordination,
                       merge_fraction, orientational_order, rigidity, struts, tail_exponent,
                       translational_order)
from .lattices import LineLattice, kagome_lattice, make_lattice, square_lattice, triangular_lattice
from .mechanics import beam_stiffness, frame_graph, spanning_floor, uniaxial_modulus
from .network import Ligaments, Network
from .plotting import plot_network
from .variants import ErreraNetwork

__version__ = "0.1.0"

__all__ = [
    "Network", "Ligaments", "ErreraNetwork",
    "LineLattice", "square_lattice", "triangular_lattice", "kagome_lattice", "make_lattice",
    "ligament_graph", "junction_graph", "coarse_grain", "struts", "Struts", "merge_fraction",
    "orientational_order", "translational_order", "mean_coordination", "rigidity", "tail_exponent", "cv",
    "frame_graph", "beam_stiffness", "uniaxial_modulus", "spanning_floor",
    "plot_network",
]
