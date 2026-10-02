# MeepMeep

**Fast Keplerian orbits for exoplanet modelling.**

MeepMeep computes Keplerian orbit quantities (transit geometry, projected
separations, radial velocities, and phase curves) using 4th-order Taylor
expansions around a set of expansion points distributed along the orbit. This
makes it up to two orders of magnitude faster than per-point Newton-Raphson while
keeping the approximation error well below the photometric noise of current
instruments. Optional analytic gradients with respect to the orbital
parameters make it suitable for gradient-based inference (HMC, optimisers).

All hot paths are [Numba](https://numba.pydata.org)-jitted and can be called
directly from your own `@njit` kernels with no wrapper overhead.

The method is described in
[Parviainen & Korth (2020), MNRAS 499, 3356](https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.3356P/abstract).

## Installation

MeepMeep is available from PyPI

```bash
pip install meepmeep
```

and from conda-forge

```bash
conda install -c conda-forge meepmeep
```

The optional backends have their own pip extras: `pip install "meepmeep[jax]"`
for the experimental JAX backend and `pip install "meepmeep[opencl]"` for running the
OpenCL device functions. Conda has no extras, so there install `jax` or
`pyopencl` from conda-forge alongside MeepMeep. MeepMeep needs Python 3.10 or
newer.

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

## C library

The evaluators are also available as a plain C99 library built from the
same sources as the OpenCL backend, with expansion-point placement and the
orbit-wide solvers included so it stands on its own. It is built and
installed independently of the Python package:

```bash
cmake -S c -B c/build
cmake --build c/build
cmake --install c/build --prefix "$HOME/.local"   # meepmeep.h + libmeepmeep
```

See `docs/source/c_library.rst` and `c/examples/transit.c`.

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

## Generative AI policy

MeepMeep grew out of code written for PyTransit from 2019 onwards and
published in
[Parviainen & Korth (2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.3356P).
That original code was written without AI or LLM assistance. Development
since mid-2026 has been assisted by LLMs, mainly Claude Code with the Opus
and Fable models.

Contributions developed with LLMs are welcome, but each one needs a human
author who oversees the work and takes responsibility for it. Purely agentic
contributions, made without human oversight, are not accepted.

## License

MeepMeep is released under the GNU General Public License v3.0. See
[`LICENSE`](LICENSE) for details.
