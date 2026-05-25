# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MeepMeep is a Python package for fast Keplerian orbit calculations optimized for exoplanet transit modeling. It uses 
Taylor series expansions around knot points to achieve high-performance orbit evaluations.

Reference notebooks live in `notebooks/` and rendered docs source in `doc/`.

## Building and Testing

**Install for development:**
```bash
pip install -e .
```

**Build package:**
```bash
python -m build
```

**Run tests:**
```bash
pytest meepmeep/tests/
```

Tests compare Taylor series approximations against exact Newton-Raphson solutions.

**Test markers** (defined in `pytest.ini`):
- `slow` — long-running tests; deselect with `-m "not slow"`
- `accuracy` — numerical accuracy validation
- `edge_case` — boundary condition tests

## Terminology: projected star-planet separation

When referring to the sky-projected distance between the star and planet centers in transit modeling contexts (the quantity usually denoted `z` or `b(t)` in units of stellar radii), use **"projected separation"** or **"sky-projected separation"** rather than "projected distance" or "projected center distance".

Rationale:
- "Separation" already implies a center-to-center distance between two objects, making qualifiers like "center" redundant.
- This matches the dominant convention in the transit literature (Mandel & Agol 2002, Seager & Mallén-Ornelas 2003, Winn 2010, and most subsequent work).
- "Sky-projected" is preferred over bare "projected" on first use, since it specifies the plane of projection.

On first use in methods sections or docstrings, expand to something like: *"the sky-projected separation between the centers of the star and planet, in units of the stellar radius"*. After that, "projected separation" alone is sufficient.

## Documentation

Sphinx sources live in `doc/source/`; build with:

```bash
cd doc && make html      # output in doc/build/html/
```

Cross-references use Sphinx domain roles (`:class:`, `:mod:`, `:func:`, `:meth:`). The authoritative reference for low-level function naming is `doc/source/naming_conventions.rst` — keep it and the "Adding New Taylor Series Functions" section of this file in sync.

**Citation.** The Taylor-series-around-knot-points method is published in Parviainen & Korth (2020), MNRAS 499, 3356; ADS bibcode `2020MNRAS.499.3356P`. Cite as "Parviainen and Korth (2020)" in docs.

**Voice and audience.** Documentation prose is targeted at PhD-level astrophysicists who are comfortable with programming. Keep it informal, informative, and to the point. Favour short sentences in introductory text. The headline performance claim is "an order of magnitude faster than per-point Newton-Raphson" (the underlying benchmark is up to ~20×; do not quote the specific multiplier in user-facing docs).

## Architecture

### Core Computational Strategy

MeepMeep uses two parallel implementations:

1. **Fast Taylor Series Approximations**: 5th-order Taylor expansions computed at knot points distributed along the orbit
2. **Exact Newton-Raphson Methods**: Reference implementations for validation and error analysis

The Taylor series approach trades exact precision for speed by pre-computing coefficients at knot points and interpolating between them.

### Package Structure

```
meepmeep/
├── __init__.py            # Exports Orbit, eclipse_light_travel_time
├── orbit.py               # Main Orbit class (user-facing API)
├── knot2d.py              # 2D knot point calculations (Knot2D, Knot2DFit)
├── numba2d.py             # Public low-level 2D Taylor API (re-exports of
│                          # backends/numba/taylor/{position2d,position2dd,
│                          # solve2d,solve2dd,util2d}).
├── numba3d.py             # Public low-level 3D Taylor API: single-knot 3D
│                          # Taylor evaluators, multi-knot orbit-spanning
│                          # routines, and dimension-agnostic primitives
│                          # (knots, Newton solvers, orbital-mechanics utils).
├── version.py             # Reads version dynamically via importlib.metadata
├── tests/
│   ├── conftest.py                  # Shared fixtures (orbital params, tolerances)
│   ├── test_numba_newton.py         # Newton-Raphson Kepler solver tests
│   ├── test_numba_orbit3d.py        # 3D orbit / multi-knot evaluator tests
│   ├── test_numba_position2d.py     # 2D position accuracy tests
│   ├── test_numba_solve2d.py        # 2D Taylor coefficient solver tests
│   ├── test_numba_solve3d.py        # 3D Taylor coefficient solver tests
│   ├── test_numba_utils.py          # Orbital-mechanics utility tests
│   ├── test_orbit3d_evaluators.py   # End-to-end orbit3d evaluator tests
│   ├── test_orbit3dd_evaluators.py  # End-to-end orbit3dd (gradient) evaluator tests
│   └── test_orbit_derivatives.py    # Orbit class derivative-mode tests
└── backends/
    ├── numba/             # Primary backend (Numba JIT-compiled)
    │   ├── utils.py       # Orbital mechanics utilities (anomaly conversions, etc.)
    │   ├── knots.py       # Knot placement strategies (mm, ea, ta)
    │   ├── newton/
    │   │   └── newton.py  # Exact Kepler equation solvers (Newton-Raphson)
    │   └── taylor/        # Taylor series expansion modules
    │       ├── position2d.py    # 2D position evaluation (pos, sep, pos_and_sep)
    │       ├── position2dd.py   # 2D position parameter derivatives (pos_d, sep_d)
    │       ├── solve2d.py       # 2D Taylor coefficient computation (solve2d)
    │       ├── solve2dd.py      # 2D derivative coefficient computation (solve2d_d)
    │       ├── util2d.py        # 2D utilities (contact points, bounding box)
    │       ├── position3d.py    # 3D position evaluation (pos, sep, pos_and_sep, pz)
    │       ├── position3dd.py   # 3D position parameter derivatives (pos_d, sep_d, pz_d)
    │       ├── solve3d.py       # 3D Taylor coefficient computation (solve3d)
    │       ├── solve3dd.py      # 3D derivative coefficient computation (solve3d_d)
    │       ├── velocity3d.py    # 3D velocity evaluation (vel_c, zvel, rv)
    │       ├── velocity3dd.py   # 3D velocity parameter derivatives
    │       ├── util3d.py        # 3D utilities (contact points, bounding box)
    │       ├── orbit3d/         # Multi-knot orbit-spanning evaluators, one
    │       │                    # module per quantity (position, separation,
    │       │                    # velocity, radial_velocity, true_anomaly,
    │       │                    # phase_angle, lambert, ev_signal, ...). Shared
    │       │                    # helpers in _common.py; __init__.py re-exports
    │       │                    # the full surface.
    │       └── orbit3dd/        # Multi-knot orbit-spanning gradient evaluators,
    │                            # mirroring orbit3d/ one module per quantity
    │                            # (the *_od / *_osd / *_ovd families).
    └── jax/               # JAX backend (automatic differentiation)
        ├── ea.py          # Eccentric anomaly with custom JVP for AD
        └── ts2d/
            └── positiond.py  # 2D position with JAX AD support
```

The package version is resolved dynamically: `pyproject.toml` declares
`dynamic = ["version"]` (via `setuptools_scm`), and `meepmeep/version.py`
reads it back at runtime through `importlib.metadata`.

### Public low-level API

`meepmeep.numba2d` and `meepmeep.numba3d` are the canonical public
entry points for the low-level Numba primitives. User `@njit` kernels
and direct (non-jitted) callers should import from these modules
rather than reaching into `meepmeep.backends.numba.*` directly. The
layout under `backends/numba/` is implementation detail and may be
restructured without notice; the aggregator modules are the
stability contract.

Each aggregator re-exports source names verbatim (no aliases, no
renames) and declares an explicit `__all__`. When adding a new public
function in `backends/numba/`, add it to the corresponding
aggregator's `__all__` as well.

`numba2d` covers the 2D Taylor surface only. `numba3d` covers the 3D
Taylor surface, the multi-knot orbit-spanning routines from
`orbit3d`/`orbit3dd`, and the dimension-agnostic primitives (knots,
Newton solvers, orbital-mechanics utilities). 2D users who need
dimension-agnostic primitives import them from
`meepmeep.backends.numba.{knots,newton.newton,utils}`.

### Key Concepts

**Knot Points**: The orbit is divided into segments with knots placed according to one of three strategies:
- `'mm'`: Mean motion (uniform time distribution)
- `'ea'`: Eccentric anomaly (default, better for eccentric orbits)
- `'ta'`: True anomaly

**Taylor Expansion**: At each knot point, position is expanded as a 5th-order Taylor series in time. The `_coeffs` 
array stores position, velocity, acceleration, jerk, and snap at each knot.

**Time-to-Knot Table** (`pktable`): Maps normalized time to the appropriate knot segment for fast lookups during evaluation. Used by `knot_ix` in `backends/numba/taylor/orbit3d/_common.py` to dispatch a time to its knot.

**Coefficient matrices**: `solve2d` returns a `(2, 5)` matrix, `solve3d` returns a `(3, 5)` matrix. Rows are spatial 
dimensions (x, y or x, y, z), columns are Taylor order (position through snap, pre-scaled by factorial).

**Derivative coefficient matrices**: `solve2d_d` and `solve3d_d` return an additional `(6, D, 5)` matrix containing 
partial derivatives of the Taylor coefficients w.r.t. the 6 orbital parameters (phase, p, a, i, e, w).

### Orbital Parameters

Standard Keplerian elements used throughout:
- `t0`: Time of inferior conjunction (transit center)
- `p`: Orbital period [days]
- `a`: Scaled semi-major axis [R_star]
- `i`: Inclination [radians]
- `e`: Eccentricity
- `w`: Argument of periastron [radians]

### Performance Considerations

- All performance-critical functions use `@njit` (Numba JIT compilation)
- `fastmath=True` is used where numerical stability permits
- Default `npt=15` knots balances accuracy and memory

### Coordinate Systems

- **X-axis**: Points right along the projected sky plane
- **Y-axis**: Points up along the projected sky plane
- **Z-axis**: Points toward the observer (negative = away from observer)

### Adding New Taylor Series Functions

The `taylor/` module follows a consistent pattern. For each quantity there are two function variants:

1. **Centered (`X_c`)**: Takes time `t` already relative to the expansion point, plus coefficient matrix `c`
2. **Direct (`X`)**: Takes absolute time `t`, reference time `t0`, period `p`, and `c`. Handles epoch-folding internally.

For parameter derivatives, the centered variant gets a `_cd` suffix and the direct variant a `_d` suffix (e.g., `pos_cd`, `pos_d`). These take an additional `dc` array of shape `(6, D, 5)`.

To add a new quantity:
1. Implement the centered version using Horner-scheme evaluation of the coefficient polynomial
2. Implement the direct version that epoch-folds and delegates to the centered version
3. If derivatives are needed, add `_d` variants in the corresponding `*d.py` module
4. Decorate with `@njit(fastmath=True)`
5. If the new function is intended for public use, add its name to the corresponding aggregator's `__all__` and its `from ... import ...` block (`meepmeep/numba2d.py` for 2D quantities, `meepmeep/numba3d.py` for 3D quantities and multi-knot routines).

For multi-knot evaluation (arrays of times with knot lookup), add a new per-quantity module under `orbit3d/` containing a pair of private kernels — `_X_os` (scalar input time) and `_X_ov` (vector of times) — together with a public `X_o` dispatcher that uses `numba.extending.overload` to route between them at compile time / call time, then re-export all three from `orbit3d/__init__.py`. Gradient counterparts go in the mirrored module under `orbit3dd/` as `_X_osd` / `_X_ovd` plus an `X_od` dispatcher, re-exported from `orbit3dd/__init__.py`. Shared helpers (`_is_1d_array`, `knot_ix`, `solve3d_orbit`) live in `orbit3d/_common.py` (`solve3d_orbit_d` and `_is_1d_array` in `orbit3dd/_common.py`). The public dispatcher is what `meepmeep/numba3d.py` re-exports and what callers use; the underscored kernels stay internal. Multi-knot kernels look up the relevant knot via `pktable`/`knot_ix` and delegate to the single-knot evaluators in `position3d`/`velocity3d` (or their gradient variants in `position3dd`/`velocity3dd`).

Note on Numba `cache=True` callers: after introducing or modifying a dispatcher, purge stale `__pycache__/*.nbi` / `*.nbc` files so Numba recompiles against the new overload registration.

### Code Style

- **Docstrings follow the NumPy style** (Parameters / Returns / Notes / Examples sections, with `name : type` parameter headers). See `backends/numba/utils.py`, `backends/numba/taylor/position3d.py`, and `backends/numba/taylor/orbit3d/position.py` for the established convention.
- Never use Unicode characters in docstrings or variable names.
- Function naming in `taylor/` modules:
  - `pos_c`, `pos`: position (centered, direct)
  - `sep_c`, `sep`: sky-projected separation
  - `pos_and_sep_c`, `pos_and_sep`: position and projected separation, returned jointly
  - `pz_c`, `pz`: line-of-sight (z) coordinate only
  - `vel_c`: velocity vector (centered)
  - `zvel_c`, `zvel`: line-of-sight velocity component
  - `rv_c`, `rv`: radial velocity
  - `_c` suffix: centered (time argument is relative to the knot)
  - `_d` suffix: direct evaluator with parameter derivatives
  - `_cd` suffix: centered evaluator with parameter derivatives
  - Dimensionality (2D vs 3D) is encoded by the module name (`position2d` vs `position3d`), not by the function name.
- In the `orbit3d/` / `orbit3dd/` packages (multi-knot dispatchers):
  - `_o`: public overloaded dispatcher; accepts scalar time OR 1-D float64 array (e.g. `pos_o`, `zvel_o`, `rv_o`)
  - `_od`: same, gradient-returning (in the `orbit3dd/` package; e.g. `pos_od`, `rv_od`)
  - `_os` / `_ov` (private, leading underscore): the underlying scalar and vector kernels the dispatcher routes to; only call directly when contributing
  - `_osd` / `_ovd` (private, leading underscore): same, gradient-returning kernels
- The authoritative reference for the full naming scheme is
  `doc/source/naming_conventions.rst`. Update both that file and this
  section if the conventions change.
- All Taylor polynomial evaluations use Horner's method for numerical stability
