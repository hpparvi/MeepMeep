# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MeepMeep is a Python package for fast Keplerian orbit calculations optimized for exoplanet transit modeling. It uses 
Taylor series expansions around expansion points to achieve high-performance orbit evaluations.

The evaluators exist in three forms built from two source trees: the Numba backend (the
reference, and what the Python API uses), and a set of `.cl` files under
`backends/opencl/` that compile both as OpenCL C device functions and, through the
CMake project in `c/`, as a standalone C99 library (`libmeepmeep`). The C library is not
a Python extension; the wheel stays pure Python.

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

Tests compare Taylor series approximations against exact Newton-Raphson solutions. The
OpenCL suites (`test_opencl_*.py`) skip without `pyopencl` and an OpenCL platform; the C
library suite (`test_c_library.py`) compiles `c/` with the host compiler into a temp dir and
skips without one, except for the header drift check, which always runs.

**Build the C library** (independent of `pip`; see the "C library" section below):
```bash
cmake -S c -B c/build -DMEEPMEEP_BUILD_EXAMPLES=ON && cmake --build c/build
python c/tools/generate_header.py     # after any signature change in a .cl file
```

**Measure coverage** (config in `.coveragerc`):
```bash
NUMBA_DISABLE_JIT=1 pytest -m "not slow" --cov
```
Coverage must run with the JIT disabled — compiled kernels are invisible to the tracer, so a
naive `pytest --cov` reports near-zero kernel coverage. `.coveragerc` excludes the `@overload`
registration bodies (they execute only during Numba type resolution and can never be traced).

When finite-difference-testing parameter derivatives, sample a narrow near-transit window rather than the full
orbit: perturbing timing/period shifts the periastron anchor and can remap a sampled time across an expansion point boundary,
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

Cross-references use Sphinx domain roles (`:class:`, `:mod:`, `:func:`, `:meth:`). The authoritative reference for low-level function naming is `docs/source/naming_conventions.rst` — keep it and the "Adding New Taylor Series Functions" section of this file in sync. When renaming or removing public functions, also sweep the narrative docs (`taylor_overview.rst`, `derivatives.rst`, `orbit_overview.rst`, and `c_library.rst` for anything ported to the `.cl` sources): they embed function names *and* runnable quickstart examples that drift silently and are not caught by the build. The C API is documented in `c/include/meepmeep.h` itself (the generated prototype block carries each function's `.cl` comment) plus `c_library.rst`; there is no autodoc for it.

The build carries a residual backlog of ~110 Sphinx cross-reference warnings (the low-level `backends/numba/` modules are not autodoc'd, so `:func:`/`:mod:` links into them stay unresolved). These are known, not regressions — only chase *new* warnings beyond the baseline. `docs/source/api/generated/` is gitignored autosummary output, not tracked source — stale stubs there (e.g. for a removed function) are build artifacts, not doc bugs to fix.

**LLM-facing docs.** `docs/llms.md` is the consumer-facing API cheatsheet for LLM agents.
The Claude Code skill at `.claude/skills/meepmeep/` bundles a snapshot of it as
`reference.md` — when revising `docs/llms.md`, re-copy it there (keep the sync header).

**Citation.** The Taylor-series-around-expansion-points method is published in Parviainen & Korth (2020), MNRAS 499, 3356; ADS bibcode `2020MNRAS.499.3356P`. Cite as "Parviainen and Korth (2020)" in docs.

**Voice and audience.** Documentation prose is targeted at PhD-level astrophysicists who are comfortable with programming. Keep it informal, informative, and to the point. Favour short sentences in introductory text. The headline performance claim is "up to two orders of magnitude faster than per-point Newton-Raphson".

## Architecture

### Core Computational Strategy

MeepMeep uses two parallel implementations:

1. **Fast Taylor Series Approximations**: 5th-order Taylor expansions computed at expansion points distributed along the orbit
2. **Exact Newton-Raphson Methods**: Reference implementations for validation and error analysis

The Taylor series approach trades exact precision for speed by pre-computing coefficients at expansion points and interpolating between them.

### Package Structure

```
meepmeep/
├── __init__.py            # Exports Orbit, eclipse_light_travel_time
├── orbit.py               # Main Orbit class (user-facing API)
├── expansion2d.py              # High-level single-expansion-point 2D wrapper (Expansion2D)
├── numba2d.py             # Public low-level 2D Taylor API (re-exports of
│                          # backends/numba/{point2d,point2dd}).
├── numba3d.py             # Public low-level 3D Taylor API: single-expansion-point 3D
│                          # Taylor evaluators, multi-expansion-point orbit-spanning
│                          # routines, and dimension-agnostic primitives
│                          # (expansion points, Newton solvers, orbital-mechanics utils).
├── version.py             # Reads version dynamically via importlib.metadata
├── tests/                 # ~25 test modules; conftest.py holds the shared
│                          # fixtures (orbital params, tolerances). Most are
│                          # named after the module under test
│                          # (test_numba_solve3d.py, test_expansion2d.py, ...);
│                          # cross-cutting suites cover the dispatchers,
│                          # parallel kernels, scalar/vector parity,
│                          # contact points / durations / find_z_min
│                          # (test_contact_points.py), and tc/tp anchoring
│                          # and gradients.
└── backends/
    ├── numba/             # Primary backend (Numba JIT-compiled)
    │   ├── utils.py       # Orbital mechanics utilities (anomaly conversions, etc.)
    │   ├── expansion_points.py       # Expansion point placement strategies (mm, ea, ta)
    │   ├── newton/
    │   │   └── newton.py  # Exact Kepler equation solvers (Newton-Raphson)
    │   ├── point2d/         # Single-expansion-point 2D evaluators (no derivatives),
    │   │                    # one module per quantity plus solve/util:
    │   │                    # position.py (pos), separation.py
    │   │                    # (sep), solve.py (solve2d), util.py (contact
    │   │                    # points, bounding box, durations, find_z_min).
    │   │                    # __init__.py re-exports the surface.
    │   ├── point2dd/        # Single-expansion-point 2D parameter-derivative evaluators
    │   │                    # mirroring point2d/: position.py (pos_cd, pos_d),
    │   │                    # separation.py (sep_cd, sep_d), solve.py
    │   │                    # (solve2d_d). pos_cd/pos_d/sep_cd/sep_d are
    │   │                    # scalar-or-array @overload dispatchers (like
    │   │                    # orbit3d's *_o) over private _s scalar and
    │   │                    # public _v/_vp vector kernels.
    │   ├── point3d/         # Single-expansion-point 3D evaluators (no derivatives),
    │   │                    # one module per quantity plus solve/util:
    │   │                    # position.py (pos), zposition.py
    │   │                    # (zpos), separation.py (sep), velocity.py
    │   │                    # (vel_c, vel), zvelocity.py (zvel), radial_velocity.py
    │   │                    # (rv), cos_phase_angle.py (cos_alpha),
    │   │                    # lambert.py (lambert_phase_curve),
    │   │                    # ev_signal.py (ev_signal),
    │   │                    # emission.py (emission_phase_curve),
    │   │                    # solve.py (solve3d), util.py (contact
    │   │                    # points, bounding box, durations, find_z_min).
    │   │                    # __init__.py re-exports the surface.
    │   ├── point3dd/        # Single-expansion-point 3D parameter-derivative evaluators
    │   │                    # mirroring point3d/: position.py (pos_cd, pos_d),
    │   │                    # zposition.py (zpos_cd, zpos_d), separation.py
    │   │                    # (sep_cd, sep_d), velocity.py (vel_cd, vel_d),
    │   │                    # zvelocity.py (zvel_cd, zvel_d),
    │   │                    # radial_velocity.py (rv_cd, rv_d),
    │   │                    # cos_phase_angle.py (cos_alpha_cd, cos_alpha_d),
    │   │                    # lambert.py (lambert_phase_curve_cd, lambert_phase_curve_d),
    │   │                    # ev_signal.py (ev_signal_cd, ev_signal_d),
    │   │                    # emission.py (emission_phase_curve_cd, emission_phase_curve_d),
    │   │                    # solve.py (solve3d_d). The evaluators are
    │   │                    # scalar-or-array @overload dispatchers (like
    │   │                    # point2dd) over private _s scalar and
    │   │                    # public _v/_vp vector kernels.
    │   ├── orbit3d/         # Multi-expansion-point orbit-spanning evaluators, one
    │   │                    # module per quantity (position, separation,
    │   │                    # velocity, radial_velocity, true_anomaly,
    │   │                    # cos_phase_angle, lambert, ev_signal, emission, ...). Shared
    │   │                    # helpers in _common.py; __init__.py re-exports
    │   │                    # the full surface.
    │   └── orbit3dd/        # Multi-expansion-point orbit-spanning gradient evaluators,
    │                        # mirroring orbit3d/ one module per quantity
    │                        # (the *_od / *_osd / *_ovd families).
    ├── jax/               # JAX backend (automatic differentiation)
    │   ├── ea.py          # Eccentric anomaly with custom JVP for AD
    │   └── ts2d/
    │       └── positiond.py  # 2D position with JAX AD support
    └── opencl/            # OpenCL backend: device functions, plus opt-in solve kernels.
        │                  # Every .cl file except solve_kernels.cl also compiles as
        │                  # C99 and is the source of the C library (see c/ below).
        ├── source.py      # SOURCE_FILES, read_kernel_source, read_full_source,
        │                  # build_options (stdlib-only; imports without pyopencl)
        ├── common.cl      # dual-target macros (MM_GLOBAL, MM_INLINE, REAL),
        │                  # constants, Horner + utils helpers, mm_mod_two_pi,
        │                  # ea_from_ma (MM_EA_TOL)
        ├── solve2d.cl     # 2D coefficient solvers (solve2d, solve2d_d)
        ├── solve3d.cl     # 3D coefficient solvers (solve3d, solve3d_d)
        ├── solve_kernels.cl  # the ONLY __kernel file: solve{2,3}d[_d]_batch
        ├── point2d.cl     # 2D values ("2"-suffixed names: pos_c2, sep2, ...)
        ├── point2dd.cl    # 2D gradients (pos_cd2, sep_d2, ...)
        ├── point3d.cl     # 3D values ("3"-suffixed names: pos_c3, sep3, ...)
        ├── point3dd.cl    # 3D gradients (pos_cd3, sep_d3, ...)
        ├── orbit3d.cl     # multi-expansion-point values (+ ep_lookup/ep_ix)
        └── orbit3dd.cl    # multi-expansion-point gradients
c/                         # C library (libmeepmeep), built with CMake independently
│                          # of the Python package; not a Python extension
├── CMakeLists.txt
├── include/meepmeep.h     # public API; the prototype block is GENERATED
├── tools/generate_header.py  # regenerates that block from the .cl sources
├── src/meepmeep.c         # unity build of the shared .cl sources as C99
├── src/expansion_points.c # create_expansion_points + brentq transcription
├── src/orbit.c            # solve3d_orbit(_d), gradient basis transforms
└── examples/transit.c
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
Taylor surface, the multi-expansion-point orbit-spanning routines from
`orbit3d`/`orbit3dd`, and the dimension-agnostic primitives (expansion points,
Newton solvers, orbital-mechanics utilities). 2D users who need
dimension-agnostic primitives import them from
`meepmeep.backends.numba.{expansion_points,newton.newton,utils}`.

### Key Concepts

**Expansion point Points**: An expansion point is a point along the orbit that serves as the *center* of a local 5th-order
Taylor expansion (not a spline-style segment *boundary* — those are the `change_times` returned by
`create_expansion_points`). The orbit is divided into segments with expansion points placed according to one of three strategies:
- `'mm'`: Mean motion (uniform time distribution)
- `'ea'`: Eccentric anomaly (default, better for eccentric orbits)
- `'ta'`: True anomaly

**Taylor Expansion**: At each expansion point, position is expanded as a 5th-order Taylor series in time. The `_coeffs` 
array stores position, velocity, acceleration, jerk, and snap at each expansion point.

**Time-to-Expansion point Table** (`ep_table`): Maps normalized time to the appropriate expansion point segment for fast lookups during evaluation. Used by `ep_ix` in `backends/numba/orbit3d/_common.py` to dispatch a time to its expansion point.

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
- `tc`: Time of inferior conjunction (transit center) [days]. `Orbit.set_pars` binds exactly one
  of `tc` or `tp` (time of periastron passage); `Expansion2D` takes `tc` directly.
- `p`: Orbital period [days]
- `a`: Scaled semi-major axis [R_star]
- `i`: Inclination [radians]
- `e`: Eccentricity
- `w`: Argument of periastron [radians]
- `lan`: Longitude of the ascending node [radians] (optional, defaults to 0.0) — a sky-plane rotation about the line of sight

### Performance Considerations

- All performance-critical functions use `@njit` (Numba JIT compilation)
- `fastmath=True` is used where numerical stability permits
- Default `npt=15` expansion points balances accuracy and memory

### Coordinate Systems

- **X-axis**: Points right along the projected sky plane
- **Y-axis**: Points up along the projected sky plane
- **Z-axis**: Points toward the observer (negative = away from observer)

### OpenCL Backend

`meepmeep/backends/opencl/` ships the Numba backend's *evaluation* surface,
and the Taylor coefficient solvers, as OpenCL C **device functions**.
Downstream packages (e.g. pt3/PyTransit) prepend `read_kernel_source(...)`
output to their own kernel code; context/queue/program management is
deliberately left to them. The same `.cl` files are also the C library's
sources (see "C library" below), so every edit here lands in both targets:
follow the "Dual target" bullet, and run the C suite as well as the OpenCL one.

`solve_kernels.cl` is the **only** file containing `__kernel` entry points,
and it is opt-in: request it by name for launchable batched solvers, or omit
it and call the `solve2d`/`solve3d` device functions from your own kernel.
Keeping it a separate file is what preserves "device functions only" for the
rest of the backend. Solve and evaluation are separate kernel launches;
fusing them was considered and deferred, because reading coefficients from
`__local` would need an address-space variant of every evaluator (OpenCL C
1.2 has no generic address space, and NVIDIA accepting one anyway is a
portability trap its own device query does not advertise).

Key conventions (documented in each `.cl` header and `source.py`):

- **Names mirror numba**, with three C-imposed deviations: only the
  scalar evaluators exist (the NDRange supplies the loop, so `*_v`/`*_vp`/
  `*_ov*` have no counterpart; each function takes the public dispatcher's
  name — `_pos_cd_w`/`pos_cd` → `pos_cd3`, `_pos_os` → `pos_o`); the
  single-expansion-point functions take a trailing dimension digit
  (`pos_c2`/`pos_c3`, `sep_cd2`/`sep_cd3`, ...) because OpenCL C has one
  flat namespace where numba disambiguates 2D/3D by package — the
  multi-expansion-point `_o`/`_od` evaluators and the dimension-agnostic
  helpers (`lambert_kernel(_d)`, `rv_scale`, `rv_cd_w`, `ep_lookup`,
  `ep_ix`, `ltt_transit_z_and_d`) are unsuffixed; and optional arguments
  (`te`, `lan`, `timing_is_tc`) are mandatory (pass `(REAL)0.0` / 0/1).
- **Gradients keep the full 7-parameter `(tc, p, a, i, e, w, lan)` order**
  (pt3's 6-row truncation is applied host-side by pt3, not here). Extra
  physical inputs append after the orbital block: `rv_od` is 8 wide (+k),
  lambert and ev_signal 9, emission 10. Every slot is written on every path;
  `true_anomaly_od` zeroes its buffer on entry because its early returns
  rely on numba's `zeros(7)` (an uninitialised read is UB in C).
- **Precision**: `REAL` is set by `build_options(precision)`
  (`-DREAL=double -DUSE_FP64` / `-DREAL=float`); every FP literal is
  `(REAL)`-cast. **Never add `-cl-fast-relaxed-math`** — the tests pin
  ~1e-12 agreement with numba, and the numba kernels that deliberately drop
  fastmath (`true_anomaly`, `rv`) rely on strict math. In fp32 builds,
  absolute times must be host-shifted by a float64 reference epoch (a
  float32 ulp at BJD ~2.4e6 is ~0.25 d).
- **Device-safety guards absent from numba**: `ep_lookup` tests `isnan`
  before the int cast (`(int)floor(NAN)` is INT_MIN on NVIDIA and an OOB
  `__global` read kills the shared context) and clamps the bucket index.
  `ep_table` must be uploaded as int32 (`astype(np.int32)`); the numba
  backend builds it int64.
- **Solvers on device**: `solve2d`/`solve2d_d`/`solve3d`/`solve3d_d` write
  their matrices through `MM_GLOBAL` output pointers (C cannot return them),
  spell the inclination `inc` (`i` is the loop index), and take `lan` and
  `from_periastron` as mandatory arguments. They keep the numba names
  unsuffixed — unlike the evaluators, those already carry the dimension.
  Worth it only for *batches* of parameter sets: a solve kernel is
  launch-bound at these sizes, so one expansion costs about what a thousand
  do. `MM_EA_TOL` makes the Kepler tolerance precision-aware, because
  numba's literal `1e-13` is unreachable in fp32 and would pin every work
  item at the 50-iteration cap.
- **Still excluded from the device**: `newton/`, `expansion_points.py`
  (scipy), and the `util.py` contact-point bisection. Expansion-point
  placement must come from the host for OpenCL; the C library carries its
  own C port of it (`c/src/expansion_points.c`), which is not device code.
- **Dual target.** Every `.cl` file except `solve_kernels.cl` also compiles
  as plain C99: the functions are written against `MM_GLOBAL` (`__global` /
  empty), `MM_INLINE` (`inline` / empty) and `REAL` (`-DREAL=` / `double`),
  defined at the top of `common.cl` by `__OPENCL_VERSION__`. Keep the shared
  bodies free of OpenCL-only builtins (`clamp`, `get_global_id`, address-space
  qualifiers, ...); only `solve_kernels.cl` may use them. Spell new
  signatures with the macros, and after any signature change run
  `python c/tools/generate_header.py` (the C test suite fails while the
  header is stale).
- **`.cl` files ship as package data** via `[tool.setuptools.package-data]`
  (explicit globs are honoured despite `include-package-data = false`);
  `pyopencl` is the optional `opencl` dependency group, and nothing in the
  shipped Python imports it.
- **Tests** (`meepmeep/tests/test_opencl_*.py`) guard with
  `pytest.importorskip("pyopencl")` plus a platform check, build test-only
  `__kernel` wrappers around the device functions (shared plumbing in
  `tests/opencl_utils.py`), and compare against the numba dispatchers at
  `rtol=1e-12` over multi-epoch times — the period chain term
  `d[1] += epoch*d[0]` is unobservable at epoch 0, so single-epoch parity
  tests cannot catch its omission. Mutation-test new `.cl` functions (flip
  a sign, verify red) rather than trusting green tests.
- Each `.cl` function is written to be diffable side-by-side against its
  numba `_s`/`_w` kernel — that correspondence is the primary correctness
  argument. When a numba kernel changes, port the change and keep the
  structure aligned.

### C library

`c/` builds the shared `.cl` sources as a C99 library, `libmeepmeep`, with
CMake (`cmake -S c -B c/build && cmake --build c/build`). It lives in this
repository so the numba, OpenCL and C targets version and test together, but
it is deliberately *not* a Python extension: the wheel stays pure Python and
nothing in `meepmeep/` imports it. Precision is fixed to double.

- `c/src/meepmeep.c` includes the public header first and then the `.cl`
  files in `SOURCE_FILES` order, so a `.cl` signature that drifts from its
  prototype is a compile error. The prototype block of
  `c/include/meepmeep.h` is generated (`python c/tools/generate_header.py`);
  the hand-written part above the markers holds the constants, status codes
  and the C-only functions.
- C-only functions (`c/src/expansion_points.c`, `c/src/orbit.c`) are the
  host-side pieces a self-contained library needs and mirror their numba
  twins by name: `create_expansion_points` (returns an `mm_status` code;
  the `'ea'`/`'ta'` roots come from a transcription of scipy's `brentq`
  with its default tolerances, so the placements agree to ~1e-12 in phase),
  `solve3d_orbit`, `solve3d_orbit_d`, and the gradient basis transforms,
  which work *in place* where numba returns a copy. When a numba twin
  changes, port the change here as well.
- Symbols are unprefixed (they are the OpenCL names); only the C-only
  enumerations carry `MM_`.
- Tests: `meepmeep/tests/test_c_library.py` always checks the header is
  current and, when a C compiler is on `PATH`, builds the library with
  `cc -shared` into a temp dir and compares the C-only functions plus a
  sample of the shared ones against numba through `ctypes`. The full
  evaluator surface is covered by the OpenCL parity suites, which exercise
  the same source text. Gradient comparisons need a signal-scaled `atol`:
  analytically-zero slots (the `lan` slot of a separation gradient) come out
  as +-1e-13 roundoff whose sign depends on fastmath contraction.
- Never build with `-ffast-math`, for the same reason as
  `-cl-fast-relaxed-math` above.

### Adding New Taylor Series Functions

The Taylor backend modules follow a consistent pattern. For each quantity there are two function variants:

1. **Centered (`X_c`)**: Takes time `t` already relative to the expansion point, plus coefficient matrix `c`
2. **Direct (`X`)**: Takes absolute time `t`, the transit-centre time `tc`, period `p`, `c`, and a trailing
   optional expansion-point offset `te=0.0` (the same value passed to the solver). Handles epoch-folding around the
   expansion point (at `tc + te` on the observation time axis) internally.

For parameter derivatives, the centered variant gets a `_cd` suffix and the direct variant a `_d` suffix (e.g., `pos_cd`, `pos_d`). These take an additional `dc` array of shape `(7, D, 5)`.

To add a new quantity:
1. Implement the centered version using Horner-scheme evaluation of the coefficient polynomial
2. Implement the direct version that epoch-folds and delegates to the centered version
3. If derivatives are needed, add `_d`/`_cd` variants in the corresponding derivative module (`point2dd/<quantity>.py` for single-expansion-point 2D, `point3dd/<quantity>.py` for 3D)
4. Decorate with `@njit(fastmath=True)`
5. If the new function is intended for public use, add its name — and the names of its public vector/parallel kernels (`X_v`/`X_vp`, or `X_ov`/`X_ovp`/`X_ovd`/`X_ovdp` for multi-expansion-point) — to the corresponding aggregator's `__all__` and its `from ... import ...` block (`meepmeep/numba2d.py` for 2D quantities, `meepmeep/numba3d.py` for 3D quantities and multi-expansion-point routines). The scalar (`_X_s`/`_X_os`/`_X_osd`), write-into (`_X_..._w`/`_X_ow`), and dual-decoration body (`_X_v_body`) kernels stay private and are not exported.
6. If the quantity belongs in the OpenCL/C surface too, port the scalar kernel to the matching `.cl` file using `MM_INLINE`/`MM_GLOBAL`/`REAL` (no OpenCL-only builtins), give it a "Port of `meepmeep.numbaXd.X`" comment (the generated C header reuses it), regenerate the header with `python c/tools/generate_header.py`, add an OpenCL parity test with a test-only `__kernel` wrapper, and mutation-test it. The C suite then covers it through the same source text.

The single-expansion-point evaluators are organised into per-dimension packages:
`point2d/`/`point2dd/` for 2D and `point3d/`/`point3dd/` for 3D, where the
plain package holds the non-derivative evaluators and the `dd` package the
parameter-derivative ones. Each has one module per physical quantity
(`position.py`, `separation.py`, and for 3D also `zposition.py`,
`velocity.py`, `zvelocity.py`, `radial_velocity.py`, `cos_phase_angle.py`,
`lambert.py`, `ev_signal.py`) plus a `solve.py`
module and, in the non-derivative package, a `util.py` for transit
geometry. Each package's `__init__.py` re-exports its surface, mirroring
the `orbit3d`/`orbit3dd` layout. The non-derivative single-expansion-point evaluators
(`point2d`/`point3d`) are scalar-or-array `@overload` dispatchers over
private scalar `_X_c_s`/`_X_s` and **public** vector `X_c_v`/`X_v` (centered/direct)
kernels — the array path is an explicit loop over the scalar kernel, NOT NumPy
broadcasting (broadcasting inside `@njit` materialises a full-array
temporary per Horner step and measured 4-40x slower). A new quantity adds
`X_c`/`X` to a `point{2,3}d/<quantity>.py` module and, if needed,
`X_cd`/`X_d` to the matching `point{2,3}dd/<quantity>.py`.

The **derivative** evaluators (`point2dd` and `point3dd`) cannot broadcast (the
gradient allocation is `(7,)` for a scalar but `(N, 7)` for an array), so each
public name (`pos_cd`, `pos_d`, `sep_cd`, `sep_d`, and in 3D also `zpos_*`,
`vel_cd`, `zvel_*`, `rv_*`) is a scalar-or-array `numba.extending.overload`
dispatcher — same pattern as `orbit3d`'s `*_o` — each a plain-Python fallback
plus an `@overload` routing to a private `_X_..._s` (scalar) kernel and a
**public** `X_..._v` (vector) kernel. `_is_1d_array` lives in each package's
`_common.py` (`point2dd/_common.py`, `point3dd/_common.py`). This is the array
path `Expansion2D` uses in 2D, so there is no separate `_dv` variant. The
single-expansion-point `X_v` kernels (many times, one coefficient matrix) are
distinct from the `orbit3dd` `X_ovd` vector kernels, which still exist because
*multi-expansion-point* dispatch (per-time expansion point lookup via `ep_table`) is a
separate concern.

**Public vs private kernels.** As of the vector-kernel promotion, the vector
(`X_v`/`X_c_v`/`X_cd_v`/`X_d_v`) and parallel (`X_vp`/...) kernels in
`point{2,3}d{,d}`, and the multi-expansion-point vector/parallel kernels
(`X_ov`/`X_ovp`/`X_ovd`/`X_ovdp`) in `orbit3d`/`orbit3dd`, are **public** —
exported through `numba2d`/`numba3d` `__all__`. The scalar kernels
(`_X_s`/`_X_c_s`/`_X_os`/`_X_osd`), the write-into kernels (`_X_..._w`/`_X_ow`),
the dual-decoration bodies (`_X_v_body`), and helpers (`_rv_scale`,
`_lambert_kernel`, `_is_1d_array`) keep their leading underscore and stay
private.

**Write-into kernels (`_w` / `_ow`).** The gradient arithmetic itself lives in
`inline='always'` *write-into* kernels that evaluate the Horner polynomials
directly into caller-provided `(7,)` buffers and return the value(s):
`_X_cd_w` in `point2dd`/`point3dd` (e.g. `_pos_cd_w`, `_sep_cd_w`,
`_rv_cd_w` with its hoistable `_rv_scale` helper), and `_X_ow` in `orbit3dd`
(e.g. `_pos_ow`, `_zpos_ow`, `_cos_alpha_ow`; these also fold the epoch and
look up the expansion point). The `_s`/`_osd` kernels allocate fresh gradient arrays and
delegate; the public `X_v`/`X_ovd` kernels pass preallocated output rows (or hoisted
scratch buffers for intermediate gradients). **Keep the hot vector loops
allocation-free**: never call a scalar gradient kernel (which allocates its
outputs) inside a per-sample loop — call the `_w`/`_ow` kernel with reused
buffers instead. Note that because the `_w` kernels are inlined into both the
scalar and vector callers, `fastmath` contraction can differ between the two
contexts by an ulp; scalar-vs-vector parity tests need a tiny `atol` (~1e-14
relative to signal scale), not `atol=0`.

**Don't read back a row a `_w` kernel just wrote.** numba 0.61 miscompiles
the serial vector loop of `dX[j] = ...; dX[j, 1] += epoch * dX[j, 0]` in
some kernels: LLVM vectorises the loop from about eight samples and the
period chain term is lost (scalar path, `_vp` twins and
`NUMBA_DISABLE_JIT=1` all stay correct). It hit `zpos_d_v`/`zvel_d_v`, whose
write kernels `_zpos_cd_w`/`_zvel_cd_w` now take the `epoch` and write the
term from a register (centered callers pass `0.0`). The other 52 read-back
sites compile correctly today; `test_numba_vector_epoch_parity.py` runs
every gradient vector kernel on a multi-epoch grid of 24 samples against
the scalar path to catch a recurrence. New kernels should write chain terms
from locals instead of reading the row back, and their tests need
multi-epoch grids of at least 16 samples: epoch-0 or few-sample grids
cannot see a missing chain term.

**Parallel twins (`X_vp` / `X_ovp` / `X_ovdp`, public).** Every vector kernel —
single-expansion-point (`X_v` -> `X_vp`) and multi-expansion-point (`X_ov` -> `X_ovp`,
`X_ovd` -> `X_ovdp`) — has a `prange` twin living in the same quantity module,
directly after its serial counterpart. Both the serial and parallel kernels are
public (exported via `numba2d`/`numba3d`). Two construction patterns, chosen by
whether the loop needs intermediate scratch:

- **Scratch-free loops** (write only into per-sample output elements/rows):
  *dual decoration* — one shared (private) body written with `prange`, compiled
  twice (`X_v = njit(fastmath=True)(_X_v_body)`;
  `X_vp = njit(fastmath=True, parallel=True)(_X_v_body)`). `prange`
  compiles as a plain `range` without `parallel=True`, so the serial kernel
  is unchanged and the math exists once. All single-expansion-point kernels except the
  rv gradients qualify.
- **Scratch-using loops** (reuse an intermediate-gradient buffer across
  samples): *explicit twins* — the serial kernel keeps its single hoisted
  buffer (cheapest), and the hand-written twin hoists one buffer per thread
  (`zeros((get_num_threads(), 7))`, indexed with `get_thread_id()`); a
  shared buffer would be a data race under `prange`, and putting per-thread
  indexing in a shared body costs the serial path ~5%. This covers the
  single-expansion-point rv gradient kernels and the derived multi-expansion-point gradient
  kernels.

The scalar-or-array dispatchers route to the serial kernels; callers reach the
parallel twins either directly (they are public) or via the
`Orbit(parallel=True)` (multi-expansion-point) and `Expansion2D(parallel=True)`
(single-expansion-point 2D) opt-ins, which use them only above the classes'
`_PARALLEL_NMIN_GRAD` / `_PARALLEL_NMIN_VALUE` thresholds (1e4 / 5e4 for
`Orbit`, 1e4 / 1e5 for `Expansion2D`) — below those sizes the parallel-region
launch overhead makes them slower.
Explicit twins must mirror the serial body exactly (including fastmath
flags): kernels compiled without fastmath (e.g. `true_anomaly`) must not
route positions through a fastmath path, or near-singular gradients drift
beyond parity tolerances.

For multi-expansion-point evaluation (arrays of times with expansion point lookup), add a new per-quantity module under `orbit3d/` containing a private scalar kernel `_X_os` (scalar input time) and a public vector kernel `X_ov` (vector of times), together with a public `X_o` dispatcher that uses `numba.extending.overload` to route between them at compile time / call time, then re-export all three (plus the parallel twin `X_ovp`) from `orbit3d/__init__.py`. Gradient counterparts go in the mirrored module under `orbit3dd/` as private `_X_osd` and public `X_ovd` (plus parallel `X_ovdp`) plus an `X_od` dispatcher, re-exported from `orbit3dd/__init__.py`. Shared helpers (`_is_1d_array`, `ep_ix`, `solve3d_orbit`) live in `orbit3d/_common.py` (`solve3d_orbit_d` and `_is_1d_array` in `orbit3dd/_common.py`). The dispatchers AND the vector/parallel kernels are what `meepmeep/numba3d.py` re-exports; the scalar (`_X_os`/`_X_osd`) and write-into kernels stay internal. Multi-expansion-point kernels look up the relevant expansion point via `ep_table`/`ep_ix` and delegate to the single-expansion-point evaluators in the `point3d` package (or their gradient variants in `point3dd`).

Note on Numba `cache=True` callers: after introducing or modifying a dispatcher, purge stale `__pycache__/*.nbi` / `*.nbc` files so Numba recompiles against the new overload registration.

### Code Style

- **Docstrings follow the NumPy style** (Parameters / Returns / Notes / Examples sections, with `name : type` parameter headers). See `backends/numba/utils.py`, `backends/numba/point3d/position.py`, and `backends/numba/orbit3d/position.py` for the established convention.
- Never use Unicode characters in docstrings or variable names.
- Function naming in the Taylor backend modules:
  - `pos_c`, `pos`: position (centered, direct)
  - `sep_c`, `sep`: sky-projected separation
  - `zpos_c`, `zpos`: line-of-sight (z) coordinate only
  - `vel_c`, `vel`: velocity vector (centered, direct)
  - `zvel_c`, `zvel`: line-of-sight velocity component
  - `rv_c`, `rv`: radial velocity
  - `cos_alpha_c`, `cos_alpha`: cosine of the orbital phase angle (star-planet-observer)
  - `_c` suffix: centered (time argument is relative to the expansion point)
  - `_d` suffix: direct evaluator with parameter derivatives
  - `_cd` suffix: centered evaluator with parameter derivatives
  - In `point2dd` and `point3dd`, every derivative evaluator (`pos_cd`/`pos_d`/`sep_cd`/`sep_d`, plus in 3D `zpos_*`/`vel_cd`/`zvel_*`/`rv_*`) is a scalar-or-array `@overload` dispatcher (a scalar time gives a `(7,)` gradient; a 1-D time array of length `N` gives a leading-`N` axis, e.g. `sep_d` returns `d` shape `(N,)` and `dd` shape `(N, 7)`). Internally each routes to a private `_s` (scalar) kernel and a public `X_v` (vector) kernel, e.g. `_pos_cd_s`/`pos_cd_v`. This is the array path used by `Expansion2D`; there is no `_dv` variant.
  - `_v` / `_vp` suffix (public): vector and parallel-vector kernels (e.g. `sep_v`/`sep_vp`, `sep_d_v`/`sep_d_vp`); call directly to skip the dispatcher's scalar-or-array type check. The `_vp` twin multi-threads the sample loop.
  - Dimensionality (2D vs 3D) is encoded by the package name (`point2d`/`point2dd` vs `point3d`/`point3dd`), not by the function name.
- In the `orbit3d/` / `orbit3dd/` packages (multi-expansion-point dispatchers):
  - `_o`: public overloaded dispatcher; accepts scalar time OR 1-D float64 array (e.g. `pos_o`, `zvel_o`, `rv_o`)
  - `_od`: same, gradient-returning (in the `orbit3dd/` package; e.g. `pos_od`, `rv_od`)
  - `_ov` / `_ovp` (public): the vector and parallel-vector value kernels the `_o` dispatcher routes to (e.g. `pos_ov`/`pos_ovp`)
  - `_ovd` / `_ovdp` (public): the vector and parallel-vector gradient kernels the `_od` dispatcher routes to
  - `_os` / `_osd` (private, leading underscore): the scalar value/gradient kernels; only call directly when contributing
- The authoritative reference for the full naming scheme is
  `docs/source/naming_conventions.rst`. Update both that file and this
  section if the conventions change.
- All Taylor polynomial evaluations use Horner's method for numerical stability
