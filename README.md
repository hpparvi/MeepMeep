# MeepMeep

**Fast Keplerian orbits for exoplanet modelling.**

MeepMeep computes Keplerian orbit quantities — transit geometry, projected
separations, radial velocities, and phase curves — using 5th-order Taylor
expansions around a set of expansion points distributed along the orbit. This
makes it an order of magnitude faster than per-point Newton-Raphson while
keeping the approximation error well below the photometric noise of current
instruments. Optional analytic gradients with respect to the orbital
parameters make it suitable for gradient-based inference (HMC, optimisers).

All hot paths are [Numba](https://numba.pydata.org)-jitted and can be called
directly from your own `@njit` kernels with no wrapper overhead.

The method is described in
[Parviainen & Korth (2020), MNRAS 499, 3356](https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.3356P/abstract).

## Installation

```bash
pip install meepmeep
```

For a development checkout:

```bash
git clone https://github.com/hpparvi/meepmeep.git
cd meepmeep
pip install -e ".[test]"
```

## Quickstart

```python
import numpy as np
from meepmeep import Orbit

o = Orbit(npt=15, ep_placement="ea")
o.set_pars(tc=0.0, p=3.4, a=8.0, i=1.55, e=0.1, w=0.4)   # times in days, angles in radians
o.set_data(np.linspace(-0.15, 0.15, 500))

x, y, z = o.xyz()                       # sky-frame position (R_star); z > 0 toward observer
sep = np.hypot(x, y)                    # sky-projected separation, the b(t) of transit models
rv  = o.radial_velocity(k=120.0)        # radial velocity in the units of k
```

Bind `tp=...` instead of `tc=...` to anchor the orbit at periastron passage.

### Analytic gradients

```python
o = Orbit(derivatives=True)
o.set_pars(tc=0.0, p=3.4, a=8.0, i=1.55, e=0.1, w=0.4)
o.set_data(times)
x, y, z, dx, dy, dz = o.xyz()           # gradients w.r.t. (tc, p, a, i, e, w, lan), shape (N, 7)
```

## Conventions

- **Units:** times in days, angles in **radians**, lengths in stellar radii
  (`a` is the scaled semi-major axis `a / R_star`).
- **Parameter order** (solvers and gradients): `(tc, p, a, i, e, w, lan)` —
  transit-centre time, period, scaled semi-major axis, inclination,
  eccentricity, argument of periastron, longitude of the ascending node.
  `lan` is optional and defaults to `0.0`.
- **Coordinates:** `x, y` span the sky plane; `z` is the line of sight,
  **positive toward the observer**. Transit occurs at `z > 0`, secondary
  eclipse at `z < 0`; `i = pi/2` is edge-on.

## Public API

| Import | Purpose |
| --- | --- |
| `meepmeep.Orbit` | 3D, multi-expansion-point orbit; any orbital phase |
| `meepmeep.Expansion2D` / `Expansion3D` | single-expansion-point, transit-window evaluators |
| `meepmeep.numba2d` / `meepmeep.numba3d` | low-level `@njit` Taylor primitives |

Everything under `meepmeep.backends/` is implementation detail and may be
restructured without notice; import only from the entry points above. See
[`docs/llms.md`](docs/llms.md) for a complete API cheatsheet.

## Testing

```bash
pip install -e ".[test]"
pytest meepmeep/tests/                                   # full suite
NUMBA_DISABLE_JIT=1 pytest -m "not slow" --cov           # with coverage
```

Coverage must run with the JIT disabled — compiled kernels are invisible to
the tracer otherwise.

## Documentation

Full documentation is built with Sphinx from `docs/source/`:

```bash
cd docs && make html      # output in docs/build/html/
```

## Citing

If MeepMeep contributes to work that leads to a publication, please cite
Parviainen and Korth (2020):

```bibtex
@ARTICLE{2020MNRAS.499.3356P,
       author = {{Parviainen}, H. and {Korth}, J.},
        title = "{Going back to basics: accelerating exoplanet transit modelling using Taylor-series expansion of the orbital motion}",
      journal = {Monthly Notices of the Royal Astronomical Society},
         year = 2020,
        month = dec,
       volume = {499},
       number = {3},
        pages = {3356-3361},
          doi = {10.1093/mnras/staa2953},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.3356P},
}
```

## License

MeepMeep is released under the GNU General Public License v3.0. See
[`LICENSE`](LICENSE) for details.
