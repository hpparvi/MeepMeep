# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MeepMeep is a Python package for fast Keplerian orbit calculations optimized for exoplanet transit modeling. It uses 
Taylor series expansions around knot points to achieve high-performance orbit evaluations.

Reference notebooks live in `notebooks/` and rendered docs source in `docs/`.

## Project status: major refactor in progress

The whole `meepmeep` package is undergoing a major refactor to improve clarity and
usability. **Breaking API changes are acceptable when justified by clarity or
usability** — do not preserve backward compatibility for its own sake; choose the
cleaner design and update every call site. The only stability contract is the public
aggregator surface (`meepmeep.numba2d` / `meepmeep.numba3d`, via their `__all__`);
everything under `backends/numba/` is implementation detail and may be restructured freely.

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

When finite-difference-testing parameter derivatives, sample a narrow near-transit window rather than the full
orbit: perturbing timing/period shifts the periastron anchor and can remap a sampled time across a knot boundary,
giving O(1) FD error at isolated points. For exactness, prefer parity against the `*_od` routines.

**Relocating/renaming backend modules** (common during the refactor): use `git mv` to
preserve history; when a module changes package depth, adjust the relative-import dot
count (escaping imports lose/gain one dot per level crossed, intra-subtree imports are
unchanged); purge stale Numba caches
(`find meepmeep/backends/numba -name '*.nbi' -o -name '*.nbc' | xargs rm`); then verify
with `pytest -m "not slow"` — the Newton-Raphson parity tests catch any behavioural drift.

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

Sphinx sources live in `docs/source/`; build with:

```bash
cd docs && make html      # output in docs/build/html/
```

Cross-references use Sphinx domain roles (`:class:`, `:mod:`, `:func:`, `:meth:`). The authoritative reference for low-level function naming is `docs/source/naming_conventions.rst` — keep it and the "Adding New Taylor Series Functions" section of this file in sync. When renaming or removing public functions, also sweep the narrative docs (`taylor_overview.rst`, `derivatives.rst`, `orbit_overview.rst`): they embed function names *and* runnable quickstart examples that drift silently and are not caught by the build.

The build carries a residual backlog of ~110 Sphinx cross-reference warnings (the low-level `backends/numba/` modules are not autodoc'd, so `:func:`/`:mod:` links into them stay unresolved). These are known, not regressions — only chase *new* warnings beyond the baseline. `docs/source/api/generated/` is gitignored autosummary output, not tracked source — stale stubs there (e.g. for a removed function) are build artifacts, not doc bugs to fix.

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
├── knot2d.py              # High-level single-knot 2D wrapper (Knot2D)
├── numba2d.py             # Public low-level 2D Taylor API (re-exports of
│                          # backends/numba/{point2d,point2dd}).
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
    │   ├── point2d/         # Single-knot 2D evaluators (no derivatives),
    │   │                    # one module per quantity plus solve/util:
    │   │                    # position.py (pos), separation.py
    │   │                    # (sep), solve.py (solve2d), util.py (contact
    │   │                    # points, bounding box, durations, find_z_min).
    │   │                    # __init__.py re-exports the surface.
    │   ├── point2dd/        # Single-knot 2D parameter-derivative evaluators
    │   │                    # mirroring point2d/: position.py (pos_cd, pos_d),
    │   │                    # separation.py (sep_cd, sep_d), solve.py
    │   │                    # (solve2d_d). pos_cd/pos_d/sep_cd/sep_d are
    │   │                    # scalar-or-array @overload dispatchers (like
    │   │                    # orbit3d's *_o) over private _s/_v kernels.
    │   ├── point3d/         # Single-knot 3D evaluators (no derivatives),
    │   │                    # one module per quantity plus solve/util:
    │   │                    # position.py (pos), zposition.py
    │   │                    # (zpos), separation.py (sep), velocity.py
    │   │                    # (vel_c), zvelocity.py (zvel), radial_velocity.py
    │   │                    # (rv), solve.py (solve3d), util.py (contact
    │   │                    # points, bounding box, durations, find_z_min).
    │   │                    # __init__.py re-exports the surface.
    │   ├── point3dd/        # Single-knot 3D parameter-derivative evaluators
    │   │                    # mirroring point3d/: position.py (pos_cd, pos_d),
    │   │                    # zposition.py (zpos_cd, zpos_d), separation.py
    │   │                    # (sep_cd, sep_d), velocity.py (vel_cd),
    │   │                    # zvelocity.py (zvel_cd, zvel_d),
    │   │                    # radial_velocity.py (rv_cd, rv_d),
    │   │                    # solve.py (solve3d_d). The evaluators are
    │   │                    # scalar-or-array @overload dispatchers (like
    │   │                    # point2dd) over private _s/_v kernels.
    │   ├── orbit3d/         # Multi-knot orbit-spanning evaluators, one
    │   │                    # module per quantity (position, separation,
    │   │                    # velocity, radial_velocity, true_anomaly,
    │   │                    # phase_angle, lambert, ev_signal, ...). Shared
    │   │                    # helpers in _common.py; __init__.py re-exports
    │   │                    # the full surface.
    │   └── orbit3dd/        # Multi-knot orbit-spanning gradient evaluators,
    │                        # mirroring orbit3d/ one module per quantity
    │                        # (the *_od / *_osd / *_ovd families).
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

**Knot Points**: A knot is a point along the orbit that serves as the *center* of a local 5th-order
Taylor expansion (not a spline-style segment *boundary* — those are the `change_times` returned by
`create_knots`). The orbit is divided into segments with knots placed according to one of three strategies:
- `'mm'`: Mean motion (uniform time distribution)
- `'ea'`: Eccentric anomaly (default, better for eccentric orbits)
- `'ta'`: True anomaly

**Taylor Expansion**: At each knot point, position is expanded as a 5th-order Taylor series in time. The `_coeffs` 
array stores position, velocity, acceleration, jerk, and snap at each knot.

**Time-to-Knot Table** (`pktable`): Maps normalized time to the appropriate knot segment for fast lookups during evaluation. Used by `knot_ix` in `backends/numba/orbit3d/_common.py` to dispatch a time to its knot.

**Coefficient matrices**: `solve2d` returns a `(2, 5)` matrix, `solve3d` returns a `(3, 5)` matrix. Rows are spatial 
dimensions (x, y or x, y, z), columns are Taylor order (position through snap, pre-scaled by factorial).

**Derivative coefficient matrices**: `solve2d_d` and `solve3d_d` return an additional `(7, D, 5)` matrix containing 
partial derivatives of the Taylor coefficients w.r.t. the 7 orbital parameters in the order (tc, p, a, i, e, w, lan).
Slot 0 is the partial w.r.t. the transit-centre time `tc` (with e/w/p taken at constant `tc`). `Orbit` returns
gradients in this transit-centre basis by default, or in the periastron basis (tp, p, a, i, e, w, lan) when bound via
`set_pars(tp=...)` — applied by `tc_to_tp_gradient` (in `backends/numba/utils.py`, re-exported from `numba3d`).
The seventh parameter, `lan` (longitude of the ascending node), is an optional argument defaulting to 0.0; it is a 
constant rotation of the sky-plane (x, y) about the line of sight (in 3D, the line-of-sight z is unaffected).

### Orbital Parameters

Standard Keplerian elements used throughout:
- `t0`: Time of inferior conjunction (transit center)
- `p`: Orbital period [days]
- `a`: Scaled semi-major axis [R_star]
- `i`: Inclination [radians]
- `e`: Eccentricity
- `w`: Argument of periastron [radians]
- `lan`: Longitude of the ascending node [radians] (optional, defaults to 0.0) — a sky-plane rotation about the line of sight

### Performance Considerations

- All performance-critical functions use `@njit` (Numba JIT compilation)
- `fastmath=True` is used where numerical stability permits
- Default `npt=15` knots balances accuracy and memory

### Coordinate Systems

- **X-axis**: Points right along the projected sky plane
- **Y-axis**: Points up along the projected sky plane
- **Z-axis**: Points toward the observer (negative = away from observer)

### Adding New Taylor Series Functions

The Taylor backend modules follow a consistent pattern. For each quantity there are two function variants:

1. **Centered (`X_c`)**: Takes time `t` already relative to the expansion point, plus coefficient matrix `c`
2. **Direct (`X`)**: Takes absolute time `t`, reference time `t0`, period `p`, and `c`. Handles epoch-folding internally.

For parameter derivatives, the centered variant gets a `_cd` suffix and the direct variant a `_d` suffix (e.g., `pos_cd`, `pos_d`). These take an additional `dc` array of shape `(7, D, 5)`.

To add a new quantity:
1. Implement the centered version using Horner-scheme evaluation of the coefficient polynomial
2. Implement the direct version that epoch-folds and delegates to the centered version
3. If derivatives are needed, add `_d`/`_cd` variants in the corresponding derivative module (`point2dd/<quantity>.py` for single-knot 2D, a `*dd.py` module such as `position3dd.py` for 3D)
4. Decorate with `@njit(fastmath=True)`
5. If the new function is intended for public use, add its name to the corresponding aggregator's `__all__` and its `from ... import ...` block (`meepmeep/numba2d.py` for 2D quantities, `meepmeep/numba3d.py` for 3D quantities and multi-knot routines).

The single-knot evaluators are organised into per-dimension packages:
`point2d/`/`point2dd/` for 2D and `point3d/`/`point3dd/` for 3D, where the
plain package holds the non-derivative evaluators and the `dd` package the
parameter-derivative ones. Each has one module per physical quantity
(`position.py`, `separation.py`, and for 3D also `zposition.py`,
`velocity.py`, `zvelocity.py`, `radial_velocity.py`) plus a `solve.py`
module and, in the non-derivative package, a `util.py` for transit
geometry. Each package's `__init__.py` re-exports its surface, mirroring
the `orbit3d`/`orbit3dd` layout. The non-derivative single-knot evaluators
(`point2d`/`point3d`) are scalar-or-array `@overload` dispatchers over
private `_X_c_s`/`_X_c_v` (centered) and `_X_s`/`_X_v` (direct) kernels —
the array path is an explicit loop over the scalar kernel, NOT NumPy
broadcasting (broadcasting inside `@njit` materialises a full-array
temporary per Horner step and measured 4-40x slower). A new quantity adds
`X_c`/`X` to a `point{2,3}d/<quantity>.py` module and, if needed,
`X_cd`/`X_d` to the matching `point{2,3}dd/<quantity>.py`.

The **derivative** evaluators (`point2dd` and `point3dd`) cannot broadcast (the
gradient allocation is `(7,)` for a scalar but `(N, 7)` for an array), so each
public name (`pos_cd`, `pos_d`, `sep_cd`, `sep_d`, and in 3D also `zpos_*`,
`vel_cd`, `zvel_*`, `rv_*`) is a scalar-or-array `numba.extending.overload`
dispatcher — same pattern as `orbit3d`'s `*_o` — each a plain-Python fallback
plus an `@overload` routing to private `_X_..._s` (scalar) and `_X_..._v`
(vector) kernels. `_is_1d_array` lives in each package's `_common.py`
(`point2dd/_common.py`, `point3dd/_common.py`). This is the array path `Knot2D`
uses in 2D, so there is no separate `_dv` variant. The single-knot `_v` kernels
(many times, one coefficient matrix) are distinct from the `orbit3dd` `_X_ovd`
vector kernels, which still exist because *multi-knot* dispatch (per-time knot
lookup via `pktable`) is a separate concern.

**Write-into kernels (`_w` / `_ow`).** The gradient arithmetic itself lives in
`inline='always'` *write-into* kernels that evaluate the Horner polynomials
directly into caller-provided `(7,)` buffers and return the value(s):
`_X_cd_w` in `point2dd`/`point3dd` (e.g. `_pos_cd_w`, `_sep_cd_w`,
`_rv_cd_w` with its hoistable `_rv_scale` helper), and `_X_ow` in `orbit3dd`
(e.g. `_pos_ow`, `_zpos_ow`, `_cos_alpha_ow`; these also fold the epoch and
look up the knot). The `_s`/`_osd` kernels allocate fresh gradient arrays and
delegate; the `_v`/`_ovd` kernels pass preallocated output rows (or hoisted
scratch buffers for intermediate gradients). **Keep the hot vector loops
allocation-free**: never call a scalar gradient kernel (which allocates its
outputs) inside a per-sample loop — call the `_w`/`_ow` kernel with reused
buffers instead. Note that because the `_w` kernels are inlined into both the
scalar and vector callers, `fastmath` contraction can differ between the two
contexts by an ulp; scalar-vs-vector parity tests need a tiny `atol` (~1e-14
relative to signal scale), not `atol=0`.

**Parallel twins (`_ovp` / `_ovdp`).** Every multi-knot vector kernel has a
`prange` twin living in the same quantity module, directly after the serial
vector kernel, compiled with `parallel=True` but otherwise mirroring the
serial body (same write-into kernels, same hoisted invariants). Gradient twins that need intermediate
scratch hoist one buffer per thread (`zeros((get_num_threads(), 7))`, indexed
with `get_thread_id()`) — a single shared buffer would be a data race under
`prange`. The public dispatchers always route to the serial kernels; the
twins are opt-in via `Orbit(parallel=True)`, which uses them only above
`Orbit._PARALLEL_NMIN_GRAD` (1e4) / `_PARALLEL_NMIN_VALUE` (5e4) samples —
below those sizes the parallel-region launch overhead makes them slower.
Mirror the serial body exactly (including fastmath flags): kernels compiled
without fastmath (e.g. `true_anomaly`) must not route positions through a
fastmath path, or near-singular gradients drift beyond parity tolerances.

For multi-knot evaluation (arrays of times with knot lookup), add a new per-quantity module under `orbit3d/` containing a pair of private kernels — `_X_os` (scalar input time) and `_X_ov` (vector of times) — together with a public `X_o` dispatcher that uses `numba.extending.overload` to route between them at compile time / call time, then re-export all three from `orbit3d/__init__.py`. Gradient counterparts go in the mirrored module under `orbit3dd/` as `_X_osd` / `_X_ovd` plus an `X_od` dispatcher, re-exported from `orbit3dd/__init__.py`. Shared helpers (`_is_1d_array`, `knot_ix`, `solve3d_orbit`) live in `orbit3d/_common.py` (`solve3d_orbit_d` and `_is_1d_array` in `orbit3dd/_common.py`). The public dispatcher is what `meepmeep/numba3d.py` re-exports and what callers use; the underscored kernels stay internal. Multi-knot kernels look up the relevant knot via `pktable`/`knot_ix` and delegate to the single-knot evaluators in the `point3d` package (or their gradient variants in `point3dd`).

Note on Numba `cache=True` callers: after introducing or modifying a dispatcher, purge stale `__pycache__/*.nbi` / `*.nbc` files so Numba recompiles against the new overload registration.

### Code Style

- **Docstrings follow the NumPy style** (Parameters / Returns / Notes / Examples sections, with `name : type` parameter headers). See `backends/numba/utils.py`, `backends/numba/point3d/position.py`, and `backends/numba/orbit3d/position.py` for the established convention.
- Never use Unicode characters in docstrings or variable names.
- Function naming in the Taylor backend modules:
  - `pos_c`, `pos`: position (centered, direct)
  - `sep_c`, `sep`: sky-projected separation
  - `zpos_c`, `zpos`: line-of-sight (z) coordinate only
  - `vel_c`: velocity vector (centered)
  - `zvel_c`, `zvel`: line-of-sight velocity component
  - `rv_c`, `rv`: radial velocity
  - `_c` suffix: centered (time argument is relative to the knot)
  - `_d` suffix: direct evaluator with parameter derivatives
  - `_cd` suffix: centered evaluator with parameter derivatives
  - In `point2dd` and `point3dd`, every derivative evaluator (`pos_cd`/`pos_d`/`sep_cd`/`sep_d`, plus in 3D `zpos_*`/`vel_cd`/`zvel_*`/`rv_*`) is a scalar-or-array `@overload` dispatcher (a scalar time gives a `(7,)` gradient; a 1-D time array of length `N` gives a leading-`N` axis, e.g. `sep_d` returns `d` shape `(N,)` and `dd` shape `(N, 7)`). Internally each routes to private `_s` (scalar) and `_v` (vector) kernels, e.g. `_pos_cd_s`/`_pos_cd_v`. This is the array path used by `Knot2D`; there is no `_dv` variant.
  - Dimensionality (2D vs 3D) is encoded by the module or package name (the `point2d`/`point2dd` packages vs `position3d`), not by the function name.
- In the `orbit3d/` / `orbit3dd/` packages (multi-knot dispatchers):
  - `_o`: public overloaded dispatcher; accepts scalar time OR 1-D float64 array (e.g. `pos_o`, `zvel_o`, `rv_o`)
  - `_od`: same, gradient-returning (in the `orbit3dd/` package; e.g. `pos_od`, `rv_od`)
  - `_os` / `_ov` (private, leading underscore): the underlying scalar and vector kernels the dispatcher routes to; only call directly when contributing
  - `_osd` / `_ovd` (private, leading underscore): same, gradient-returning kernels
- The authoritative reference for the full naming scheme is
  `docs/source/naming_conventions.rst`. Update both that file and this
  section if the conventions change.
- All Taylor polynomial evaluations use Horner's method for numerical stability
