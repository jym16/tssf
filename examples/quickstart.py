"""A short tour of the tssf package (runs in well under a minute).

    python quickstart.py            # prints a summary and writes quickstart.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402

import tssf                       # noqa: E402

# 1. The scale-rich (SR) process of Both et al.: T straight ligaments, each stopping on contact.
sr = tssf.Network(seed=0).run(T=2000)
lengths, degrees = tssf.ligament_graph(sr)        # ligament-as-node representation
print(f"SR network: {sr.n_ligaments} ligaments, mean degree {degrees.mean():.2f}")
print(f"  tail exponents: P(l) ~ l^-{tssf.tail_exponent(lengths, 0.05):.2f}, "
      f"P(k) ~ k^-{tssf.tail_exponent(degrees - 0.5, 9.5):.2f}  (Both et al.: 3, 3)")

# 2. Template snapping toward a square lattice. eps_theta = 0 aligns ligaments with the lattice
#    directions; eps_x interpolates their positions (eps_x < 1/2 is the 'registered' regime).
lat = tssf.square_lattice(10)
print(f"\nPath to {lat}: w_min = {lat.default_w_min:.3f}")
print(" eps_x  ligaments   Phi    f(tau=0.1h)   z")
nets = {}
for eps_x in [1.0, 0.45, 0.2, 0.0]:
    net = tssf.Network(lat, eps_theta=0.0, eps_x=eps_x, w_min=lat.default_w_min, seed=1).run()
    s = tssf.struts(net, tau=0.1 * lat.h)        # history-free struts at resolution tau
    z, _ = tssf.mean_coordination(s.nodes, s.edges)
    print(f" {eps_x:4.2f}  {net.n_ligaments:6d}   {tssf.translational_order(net, lat):5.3f}"
          f"      {tssf.merge_fraction(net, s):5.3f}      {z:4.2f}")
    nets[eps_x] = net

# 3. History dependence: a perfect lattice looks 'scale-rich' in the ligament graph, not in struts.
grid = nets[0.0]
hist_len, _ = tssf.ligament_graph(grid)
strut = tssf.struts(grid, 1e-6)
print(f"\nExact lattice: ligament-length CV {tssf.cv(hist_len):.2f} (history) vs "
      f"{tssf.cv(strut.lengths):.2f} (struts; {len(strut.lengths)} struts of degree {strut.degrees[0]})")

# 4. Frame-model stiffness of the registered network for a few strut widths, and the floor that
#    straight platen-to-platen struts contribute (subtract it before comparing networks).
net = nets[0.2]
floor, n_span = tssf.spanning_floor(net, axis="y")
print(f"\nRegistered network (eps_x = 0.2): {n_span} spanning struts, floor = {floor:.3f}")
for lam_h in [0.005, 0.02, 0.08]:
    E, rho = tssf.uniaxial_modulus(net, lam_h * lat.h, axis="y")
    print(f"  lambda/h = {lam_h:5.3f}:  rho = {rho:.3f},  E/(E_s rho) = {E / rho:.3f}")

# 5. Pictures: SR with its thickness schedule, and the path to the lattice.
fig, axs = plt.subplots(1, 5, figsize=(12, 2.6))
tssf.plot_network(tssf.Network(seed=0).run(T=300), axs[0], linewidth=3.0, alpha=0.5).set_title("SR, alpha = 0.5")
for ax, eps_x in zip(axs[1:], nets):
    tssf.plot_network(nets[eps_x], ax, linewidth=0.6).set_title(f"eps_theta = 0, eps_x = {eps_x}")
fig.tight_layout()
fig.savefig("quickstart.png", dpi=150)
print("\nwrote quickstart.png")
