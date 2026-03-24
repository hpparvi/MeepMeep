# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MeepMeep is a Python package for fast Keplerian orbit calculations optimized for exoplanet transit modeling. It uses 
Taylor series expansions around knot points to achieve high-performance orbit evaluations.

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

## Architecture

### Core Computational Strategy

MeepMeep uses two parallel implementations:

1. **Fast Taylor Series Approximations**: 5th-order Taylor expansions computed at knot points distributed along the orbit
2. **Exact Newton-Raphson Methods**: Reference implementations for validation and error analysis

The Taylor series approach trades exact precision for speed by pre-computing coefficients at knot points and interpolating between them.

### Package Structure

```
meepmeep/
├── __init__.py            # Exports Orbit, position, eclipse_light_travel_time
├── orbit.py               # Main Orbit class (user-facing API)
├── knot2d.py              # 2D knot point calculations (Knot2D, Knot2DFit)
├── tsorbit.py             # Compatibility module (re-exports bounding_box)
├── version.py             # Package version (0.8.0)
├── xy/
│   └── __init__.py        # Compatibility shim redirecting to backends.numba.taylor
├── tests/
│   ├── conftest.py        # Shared fixtures (orbital params, tolerances)
│   └── test_ts2d_position.py  # 2D Taylor series accuracy tests
└── backends/
    ├── numba/             # Primary backend (Numba JIT-compiled)
    │   ├── utils.py       # Orbital mechanics utilities (anomaly conversions, etc.)
    │   ├── knots.py       # Knot placement strategies (mm, ea, ta)
    │   ├── tsorbit.py     # Taylor series orbit object wrapper
    │   ├── newton/
    │   │   └── newton.py  # Exact Kepler equation solvers (Newton-Raphson)
    │   └── taylor/        # Taylor series expansion modules
    │       ├── position2d.py   # 2D position evaluation (p2d, d2d, pd2d)
    │       ├── position2dd.py  # 2D position parameter derivatives (p2d_d, d2d_d)
    │       ├── solve2d.py      # 2D Taylor coefficient computation (solve2d)
    │       ├── solve2dd.py     # 2D derivative coefficient computation (solve2d_d)
    │       ├── util2d.py       # 2D utilities (contact points, bounding box)
    │       ├── position3d.py   # 3D position evaluation (p3d, d3d, z3d, pd3d)
    │       ├── position3dd.py  # 3D position parameter derivatives (p3d_d, d3d_d, z3d_d)
    │       ├── solve3d.py      # 3D Taylor coefficient computation (solve3d)
    │       ├── solve3dd.py     # 3D derivative coefficient computation (solve3d_d)
    │       ├── velocity3d.py   # 3D velocity evaluation (v3dc, vz3d)
    │       ├── util3d.py       # 3D utilities (contact points, bounding box)
    │       └── extended3d.py   # Multi-knot 3D orbit (phase curves, RV, light travel time)
    └── jax/               # JAX backend (automatic differentiation)
        ├── ea.py          # Eccentric anomaly with custom JVP for AD
        └── ts2d/
            └── positiond.py  # 2D position with JAX AD support
```

### Key Concepts

**Knot Points**: The orbit is divided into segments with knots placed according to one of three strategies:
- `'mm'`: Mean motion (uniform time distribution)
- `'ea'`: Eccentric anomaly (default, better for eccentric orbits)
- `'ta'`: True anomaly

**Taylor Expansion**: At each knot point, position is expanded as a 5th-order Taylor series in time. The `_coeffs` 
array stores position, velocity, acceleration, jerk, and snap at each knot.

**Time-to-Knot Table** (`_tptable`): Maps normalized time to the appropriate knot segment for fast lookups during evaluation.

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

1. **Centered (`Xc`)**: Takes time `t` already relative to the expansion point, plus coefficient matrix `c`
2. **Direct (`X`)**: Takes absolute time `t`, reference time `t0`, period `p`, and `c`. Handles epoch-folding internally.

For parameter derivatives, add a `_d` suffix (e.g., `p3dc_d`, `p3d_d`). These take an additional `dc` array of shape `(6, D, 5)`.

To add a new quantity:
1. Implement the centered version using Horner-scheme evaluation of the coefficient polynomial
2. Implement the direct version that epoch-folds and delegates to the centered version
3. If derivatives are needed, add `_d` variants in the corresponding `*d.py` module
4. Decorate with `@njit(fastmath=True)`

For multi-knot evaluation (arrays of times with knot lookup), add functions to `extended3d.py` following the `_o5s` (scalar) / `_o5v` (vector) naming convention.

### Code Style

- Function naming in `taylor/` modules:
  - `p3dc`, `p3d`: position (centered, direct)
  - `d3dc`, `d3d`: projected distance
  - `z3dc`, `z3d`: z-coordinate only
  - `v3dc`: velocity (centered)
  - `_d` suffix: includes parameter derivatives
  - `2d`/`3d` infix: dimensionality
- In `extended3d.py` (multi-knot functions):
  - `_o5s`: 5th-order Taylor, scalar (single time)
  - `_o5v`: 5th-order Taylor, vector (array of times)
- All Taylor polynomial evaluations use Horner's method for numerical stability
