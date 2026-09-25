# Changelog

All notable changes to MeepMeep are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- JAX backend (`meepmeep.jax2d`, `meepmeep.jax3d`, implemented in
  `meepmeep/backends/jax/`): the numba value surface ported to JAX with the
  same names and argument order. It has the coefficient solvers, the
  single-expansion-point `X_c`/`X` evaluators, the multi-expansion-point `X_o`
  evaluators, contact points, durations and `find_z_min`, expansion-point
  placement, and the Newton-Raphson references. Everything traces (`jit`,
  `vmap`, `grad`). Gradients come from autodiff, and the test suites pin them
  to the numba `_d`/`_od` kernels at round-off. There are no gradient, vector
  or basis-transform variants. New: `JaxOrbit`, an immutable pytree counterpart
  of `Orbit` built with `from_tc`/`from_tp`, with the gradient basis set by the
  constructor. `create_expansion_points` places the `'ea'`/`'ta'` grids in
  closed form, so it runs inside `jit` with a traced eccentricity. The contact
  points and durations are differentiable through implicit-function JVPs.
  Requires `jax_enable_x64`; install with the new `jax` extra.

- C library (`c/`): the evaluators, coefficient solvers and multi-expansion-
  point routines as a plain C99 library, `libmeepmeep`, built with CMake
  independently of the Python package. It is a second compile target of the
  `.cl` sources shared with the OpenCL backend, plus C ports of the pieces a
  self-contained library needs: `create_expansion_points` (with a
  transcription of scipy's `brentq` for the anomaly-uniform strategies),
  `solve3d_orbit`, `solve3d_orbit_d`, and the in-place gradient basis
  transforms `tc_to_tp_gradient`, `tp_to_tc_gradient` and
  `tp_to_tc_gradient_orbit`. Precision is fixed to double. The public
  header's prototype block is generated from the shared sources
  (`c/tools/generate_header.py`) and `tests/test_c_library.py` guards it
  against drift and checks the library against numba through `ctypes`.

### Changed
- The OpenCL sources are now written against `MM_GLOBAL`, `MM_INLINE` and
  `REAL` macros defined at the top of `common.cl`, so the same files compile
  as OpenCL C and as C99. On the device they expand to `__global` and
  `inline` as before; kernels built from `read_kernel_source` are unaffected.
  The one OpenCL-only builtin in the shared code (`clamp` in `ep_lookup`)
  was replaced by explicit branches.

### Fixed
- Whole-orbit evaluation lost accuracy at high eccentricity because the
  time-to-expansion-point table was too coarse: its 200 bins were wider than
  the expansion-point regions near periastron, so parts of a bin were
  evaluated from an expansion point far away on the local orbital timescale,
  and the table-building loop advanced by at most one expansion point per
  bin. At e = 0.9 the error stalled at ~0.1 R_star whatever `npt` was; at
  e = 0.7 it was 5-25x larger than the Taylor truncation error.
  `create_expansion_points` (numba, JAX and C) now maps each bin to the region
  containing its centre, and its default `tres=None` sizes the table at eight
  bins per narrowest region of the placement for `max(e, 0.9)` (at least 200,
  at most 2**20); `expansion_table_size` (numba, JAX, and C, where it sizes
  the caller's `ep_table`) returns that size. With it the error converges
  with `npt` again (e = 0.9, npt = 35: 2e-4 R_star). The larger tables cost
  nothing measurable in evaluation. `JaxOrbit` defaults to the same size.
- `zpos_d` and `zvel_d` (and so `Expansion3D` in derivative mode) returned a
  wrong period derivative for arrays of eight or more times spanning several
  epochs: numba 0.61 miscompiled their serial vector kernels and dropped the
  period-folding chain term. The scalar path and the `_vp` twins were
  correct. `_zpos_cd_w`/`_zvel_cd_w` now add the term themselves.
- `solve3d_orbit_d` (numba and C) copied slot 0 into the periodic-image slot
  without the extra `1 * dcf[0]` its phase of 1 adds to the period row, so
  every orbit gradient (`*_od`, `Orbit(derivatives=True)`) had a period
  derivative off by one timing row for times just before periastron.
- `true_anomaly_od` ignored the basis of `dcoeffs` on its circular fast path
  and always returned periastron-basis gradients. It takes a trailing
  `timing_is_tc` (default True, as in `light_travel_time_od`); `Orbit` passes
  its basis. The OpenCL/C `true_anomaly_od` takes it as a mandatory
  argument before the output buffer.
- `eccentricity_vector` ignored the longitude of the ascending node, so
  `true_anomaly_o`/`_od` and `Orbit.true_anomaly` were wrong (by up to the
  node angle) whenever `lan != 0`. It takes an optional `lan`, and `Orbit`
  passes it.

### Removed
- `Orbit(derivatives=True).true_anomaly()` returned wrong `w` and `lan`
  gradients for eccentric orbits (the `w` slot off by O(1), a non-zero `lan`
  slot where the true anomaly does not depend on the node). `true_anomaly_od`
  held the eccentricity vector constant, but the vector turns with `w` and
  `lan`. **Breaking:** `true_anomaly_od` and its `_ovd`/`_ovdp` kernels (numba,
  OpenCL and C) take the Jacobian of the eccentricity vector, `dev` of shape
  `(3, 7)`, after `w`; pass zeros for the old constant-vector behaviour. The
  new `eccentricity_vector_d` (numba, OpenCL and C) returns the vector and its
  Jacobian, and `numba3d` now exports it together with `eccentricity_vector`.
- The JAX prototype modules `meepmeep.backends.jax.ea` and
  `meepmeep.backends.jax.ts2d` (`solve_xy_p5`, `solve_xy_p5_d`, `xy_t15_d`,
  `pd_t15_d`), superseded by the JAX backend above.

## [1.1.0] - 2026-09-08

### Changed
- The transit-centre (slot 0) row of the derivative tensors returned by
  `solve2d_d`, `solve3d_d`, `solve3d_orbit_d` and the JAX `solve_xy_p5_d`
  is now the derivative of the truncated Taylor polynomial the evaluators
  compute, `dc[0, :, n] = -(n + 1) c[:, n + 1]` with a zero fourth-order
  entry, instead of the derivative of the exact orbit obtained by
  propagating `dM/dtc = -n` through Kepler's equation. The two differ by
  the fifth-order term the expansion drops, `x^(5) tau^4 / 4!`: about
  1e-5 relative at `tau = 0.07` d for a hot-Jupiter orbit, growing as
  `tau^4`. The gradient is now the gradient of the value the evaluators
  return.
- The orbit-spanning solver `solve3d_orbit_d` now anchors the point solver at
  periastron (new `from_periastron` argument of `solve2d_d` / `solve3d_d`),
  where its expansion points actually sit, and returns the periastron basis
  `(tp, p, a, i, e, w, lan)` natively. `Orbit` applies the new
  `tp_to_tc_gradient` when bound with `tc` instead of `tc_to_tp_gradient`
  when bound with `tp`. 

### Added
- OpenCL backend (`meepmeep.backends.opencl`) shipping the evaluation
  surface of the Numba backend as OpenCL C *device functions* (no
  `__kernel` entry points): packages using MeepMeep prepend the source
  returned by `read_kernel_source` to their own kernels. Covers the 2D and
  3D single-expansion-point evaluators and the multi-expansion-point
  orbit-spanning evaluators, in both value-only and value-plus-gradient
  forms with the full seven-parameter `(tc, p, a, i, e, w, lan)`
  convention. The `solve*` coefficient solvers, Newton reference solvers,
  expansion-point placement, and contact-point bisection remain host-side
  (Numba/JAX). Function names mirror the Numba backend, with the
  single-expansion-point functions carrying a trailing dimension digit
  (`pos_c2`/`pos_c3`, `sep_cd2`/`sep_cd3`, ...) because OpenCL C has a
  single flat namespace where Numba disambiguates 2D/3D by package.
  Install the optional dependency with `pip install meepmeep[opencl]`.

## [1.0.0] - 2026-06-18

First stable release. The orbit backend was reorganised into a clear,
documented public surface, and the approximation is validated against
Newton-Raphson references across a broad parameter range.

### Added
- High-level `Orbit` class (3D, multi-expansion-point) covering transit
  geometry, line-of-sight position and velocity, radial velocity, phase
  angle, projected and 3D star-planet separations, light-travel time, and
  Lambert / emission / ellipsoidal-variation phase curves.
- High-level single-expansion-point `Expansion2D` and `Expansion3D` classes
  for fast transit-window evaluation.
- Optional analytic gradients (`derivatives=True`) with respect to
  `(tc, p, a, i, e, w, lan)`, plus the extra physical parameters of each
  method; supported in either the transit-centre (`tc`) or periastron (`tp`)
  timing basis.
- Public low-level Numba API via `meepmeep.numba2d` and `meepmeep.numba3d`,
  callable from user `@njit` kernels; serial and parallel (`prange`) vector
  kernels for every quantity.
- Consumer-facing API cheatsheet at `docs/llms.md`.

### Changed
- Reorganised the Numba backend into per-quantity modules under
  `backends/numba/{point2d,point2dd,point3d,point3dd,orbit3d,orbit3dd}`.
  `meepmeep.numba2d` / `meepmeep.numba3d` are now the stability contract;
  everything under `meepmeep.backends/` is implementation detail.
- Standardised on the term "projected separation" for the sky-projected
  star-planet separation throughout the API and documentation.

### Fixed
- Corrected the package discovery configuration so built wheels and sdists
  ship the complete Numba backend (and no longer ship the test suite).
- Fixed the `Orbit._cos_phase_error` diagnostic to compare against the exact
  phase-angle cosine rather than the true anomaly.
