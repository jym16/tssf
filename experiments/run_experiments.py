"""Regenerate the simulation data behind the report (JSON files read by make_figures.py).

Usage::

    python run_experiments.py positional rigidity orient_size orient frame --out data --workers 2
    python run_experiments.py all

Experiments (seeds are fixed, so results are reproducible bit-for-bit):

positional   eps_theta = 0, eps_x swept, jammed; n = 12 square / triangular / kagome; 3 seeds
rigidity     same path at n = 8 with Maxwell-Calladine counts (dense SVD); 2 seeds
orient_size  square, eps_x = 0, small eps_theta, n = 8, 16, 24 (local vs global registration)
orient       SR-faithful path: eps_x = 1, w_min = 0, T = 3000, eps_theta swept; 4 seeds
frame        Timoshenko frame model along registered paths + SR, with spanning-strut floors
"""
import argparse
import json
import os
import time
from multiprocessing import Pool

import numpy as np

import tssf

LAMBDAS = list(np.round(np.logspace(-2.5, -0.7, 10), 5))       # strut width / h for the frame model


# ----------------------------------------------------------------------------- per-network metrics
def topology_metrics(net, lat, taus, with_rigidity=False):
    """Order parameters and strut statistics of one network at several resolutions tau/h."""
    lengths, degrees = tssf.ligament_graph(net)
    out = dict(n_lig=net.n_ligaments, n_prop=net.n_proposals, hist_len_cv=tssf.cv(lengths),
               hist_deg_cv=tssf.cv(degrees), cell_area_cv=tssf.cv(net.cell_areas),
               psi=tssf.orientational_order(net, lat.m), phi=tssf.translational_order(net, lat))
    for t in taus:
        s = tssf.struts(net, t * lat.h)
        floppy, stress = tssf.rigidity(s.nodes, s.edges) if with_rigidity else (np.nan, -1)
        out[f"tau{t}"] = dict(f=tssf.merge_fraction(net, s), z=tssf.mean_coordination(s.nodes, s.edges)[0],
                              n_nodes=len(s.nodes), floppy_per_node=floppy / len(s.nodes), self_stress=int(stress),
                              merged_n=len(s.lengths), merged_len_cv=tssf.cv(s.lengths),
                              merged_deg_cv=tssf.cv(s.degrees))
    return out


def jammed(name, n, eps_theta, eps_x, seed):
    """Jammed TSSF network with the default sliver cutoff of the target lattice."""
    lat = tssf.make_lattice(name, n)
    return lat, tssf.Network(lat, eps_theta, eps_x, w_min=lat.default_w_min, seed=seed).run()


# ----------------------------------------------------------------------------- jobs (one network each)
def job_topology(args):
    name, n, eps_theta, eps_x, seed, taus, with_rigidity = args
    t0 = time.time()
    lat, net = jammed(name, n, eps_theta, eps_x, seed)
    out = topology_metrics(net, lat, taus, with_rigidity)
    out.update(lattice=name, n=n, eps_x=eps_x, eps_th=eps_theta, seed=seed, secs=time.time() - t0)
    return out


def job_orient(args):
    """Fixed-T SR-faithful run; stores the full ligament statistics for tail fits."""
    name, eps_theta, seed, T = args
    lat = tssf.make_lattice(name, 12)                  # only the directors matter here
    net = tssf.Network(lat, eps_theta, 1.0, w_min=0.0, seed=seed).run(T=T)
    lengths, degrees = tssf.ligament_graph(net)
    t = np.arange(1, net.n_ligaments + 1)
    beta = -np.polyfit(np.log(t[t >= 30]), np.log(lengths[t >= 30]), 1)[0]   # <l_t> ~ t^-beta
    return dict(lattice=name, eps_th=eps_theta, seed=seed, T=T, psi=tssf.orientational_order(net, lat.m),
                L=lengths.tolist(), k=degrees.tolist(), beta=float(beta))


def job_frame(args):
    """Frame-model moduli along both axes for a list of strut widths, plus spanning floors."""
    name, param, seed, lams = args
    if name == "sr":
        net, h = tssf.Network(seed=seed).run(T=int(param)), 0.1
    else:
        lat, net = jammed(name, 10, 0.0, param, seed)
        h = lat.h
    res = []
    for lam in lams:
        Ey, rho = tssf.uniaxial_modulus(net, lam * h, axis="y")
        Ex, _ = tssf.uniaxial_modulus(net, lam * h, axis="x")
        res.append(dict(lam_over_h=lam, rho=rho, Ey=Ey, Ex=Ex))
    (fy, ny), (fx, nx) = tssf.spanning_floor(net, "y"), tssf.spanning_floor(net, "x")
    return dict(lattice=name, eps_x=param, seed=seed, n_lig=net.n_ligaments, Ltot=float(net.ligaments.length.sum()),
                floor_y=fy, n_span_y=ny, floor_x=fx, n_span_x=nx, res=res)


# ----------------------------------------------------------------------------- experiment definitions
LATTICES = ["square", "triangular", "kagome"]
EXPERIMENTS = {
    "positional": (job_topology, [(nm, 12, 0.0, ex, s, [0.05, 0.1, 0.2, 0.4], False) for nm in LATTICES
                                  for ex in [0.0, 0.05, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0] for s in range(3)]),
    "rigidity": (job_topology, [(nm, 8, 0.0, ex, s, [0.05, 0.1, 0.2], True) for nm in LATTICES
                                for ex in [0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 1.0] for s in range(2)]),
    "orient_size": (job_topology, [("square", n, e, 0.0, s, [0.1, 0.2], False) for n in [8, 16, 24]
                                   for e in [0.0, 0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16] for s in range(3)]),
    "orient": (job_orient, [(nm, e, s, 3000) for nm in ["square", "triangular"]
                            for e in np.round(np.linspace(0, 1, 11), 1) for s in range(4)]),
    "frame": (job_frame, [("square", ex, s, LAMBDAS) for ex in [0.0, 0.05, 0.1, 0.2, 0.3, 0.45] for s in range(4)]
              + [("triangular", ex, s, LAMBDAS) for ex in [0.0, 0.05, 0.1, 0.2, 0.3] for s in range(4)]
              + [("sr", T, 100 * T + s, [0.00316, 0.01]) for T in [150, 300, 600, 1200] for s in range(6)]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="+", choices=list(EXPERIMENTS) + ["all"])
    ap.add_argument("--out", default="data", help="output directory for res_<experiment>.json")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name in (list(EXPERIMENTS) if "all" in args.experiments else args.experiments):
        job, jobs = EXPERIMENTS[name]
        t0 = time.time()
        with Pool(args.workers) as pool:
            results = pool.map(job, jobs, chunksize=1)
        with open(os.path.join(args.out, f"res_{name}.json"), "w") as fh:
            json.dump(results, fh)
        print(f"{name}: {len(results)} runs in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
