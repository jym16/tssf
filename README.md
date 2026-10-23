# tssf — scale-rich network materials, continuously interpolated to line lattices

`tssf` implements the scale-rich (SR) network model of Both *et al.*
([arXiv:2511.18108](https://arxiv.org/abs/2511.18108)) and its extension, **template-snapped
sequential fragmentation (TSSF)**. TSSF moves continuously between SR networks and exact square,
triangular and kagome lattices.

**SR.** In the unit square, repeatedly pick a uniform point and a uniform angle. Grow a straight
ligament both ways until it meets an existing ligament or the boundary.

**TSSF.** Before growing, pull each proposal toward the nearest line of a target lattice:

```
theta' = phi_f + eps_theta * (theta - phi_f)      # orientational amplitude
x'     = x - (1 - eps_x) * d * n_f                # positional amplitude
```

- `eps_theta = eps_x = 1` reproduces SR exactly.
- `eps_theta = eps_x = 0` with jamming gives the lattice itself, for any insertion order.
- A sliver cutoff `w_min` rejects splits that would leave a daughter cell narrower than `w_min`.

## Install

With [uv](https://docs.astral.sh/uv/), from this directory:

```bash
uv sync                     # numpy + scipy
uv sync --extra plot        # + matplotlib for plot_network
uv sync --extra test        # + pytest
uv sync --all-extras        # everything, including jupyter for the examples
```

`uv sync` creates `.venv/` from the pinned `uv.lock` and installs `tssf` in editable mode.
Run commands inside it with `uv run`:

```bash
uv run --extra test pytest                    # test suite
uv run --extra plot python examples/quickstart.py
uv run --extra examples jupyter lab examples/tutorial.ipynb
```

## Quick start

```python
import tssf

# SR network with 1000 ligaments
sr = tssf.Network(seed=0).run(T=1000)
lengths, degrees = tssf.ligament_graph(sr)            # Both et al.'s ligament-as-node graph

# A square-lattice target, registered regime (eps_x < 1/2)
lat = tssf.square_lattice(12)
net = tssf.Network(lat, eps_theta=0.0, eps_x=0.2, w_min=lat.default_w_min, seed=1).run()
s = tssf.struts(net, tau=0.1 * lat.h)                 # history-free struts at resolution tau
print(tssf.merge_fraction(net, s), tssf.translational_order(net, lat))
print(tssf.mean_coordination(s.nodes, s.edges)[0], tssf.rigidity(s.nodes, s.edges))

# Frame-model stiffness with struts of width 0.02 h
E, rho = tssf.uniaxial_modulus(net, 0.02 * lat.h, axis="y")
floor, n_span = tssf.spanning_floor(net, axis="y")
```

`examples/quickstart.py` runs a short tour. `examples/tutorial.ipynb` walks through every part of
the model and reproduces the key checks of the report.

## Package layout

| module | contents |
| --- | --- |
| `network.py` | `Network`: the SR / TSSF process (`run`, `step`, `ligaments`, `thickness`) |
| `lattices.py` | `LineLattice`, `square_lattice`, `triangular_lattice`, `kagome_lattice`, `make_lattice` |
| `analysis.py` | `ligament_graph`, `junction_graph`, `coarse_grain`, `struts`, `merge_fraction`, order parameters, `mean_coordination`, `rigidity`, `tail_exponent` |
| `mechanics.py` | Timoshenko frame model: `uniaxial_modulus`, `spanning_floor`, `beam_stiffness`, `frame_graph` |
| `variants.py` | `ErreraNetwork`: template-free local division rules (exact square grid limit) |
| `geometry.py` | convex-polygon primitives (clipping, chords, widths) |
| `plotting.py` | `plot_network` |

`experiments/` contains the scripts that regenerate all data and REVTeX figures of the report:
`run_experiments.py` and `make_figures.py`. `tests/` holds a pytest suite of fast physics checks.

## Conventions and caveats

- **Geometry.** The domain is the unit square. The networks are zero-thickness skeletons (the
  `lambda_0 -> 0` limit). Thickness enters through `Network.thickness(lambda0, alpha)`, the strut
  resolution `tau`, `w_min`, and the beam width in `mechanics`.
- **Defaults.** `lattice.default_w_min` is half the smallest face width: `h/2` for square and
  triangular, `h/4` for kagome. With it, the proven *registered* regime is `eps_x < 1/2`
  (square), `1/3` (triangular) and `1/6` (kagome).
- **Choosing `tau`.** Single-linkage clustering chains, so keep `tau` well below the smallest
  lattice edge (`tau <= 0.1 h` for kagome).
- **Mechanics.** The frame model is linear and slender-beam. Elements shorter than their width
  lie outside beam theory, and `rho = lam * total length` double-counts joint overlaps.
- **Reproducibility.** Results are reproducible for a given `seed`. The package reproduces the
  original research scripts bit-for-bit.
