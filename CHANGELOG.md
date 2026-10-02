# Changelog

All notable changes to MeepMeep are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2026-10-02

### Added
- Experimental JAX backend (`meepmeep.jax2d`, `meepmeep.jax3d`), whose API
  may still change: the numba value surface with the same names and argument
  order, fully traceable (`jit`, `vmap`, `grad`), with gradients from
  autodiff. Includes `JaxOrbit`, a pytree counterpart of `Orbit`. Computes in
  float32 or float64, following the input dtypes. Install with
  `pip install meepmeep[jax]`.
- C library (`c/`): `libmeepmeep`, the OpenCL sources compiled as C99 with
  CMake, plus C ports of expansion-point placement, the orbit solvers and the
  gradient basis transforms.

### Changed
- `numba3d` exports `mean_anomaly_at_transit`, and `numba3d`/`jax3d` export
  `eclipse_time_offset`.
- The OpenCL sources also compile as C99. Kernels built from
  `read_kernel_source` are unaffected.

### Fixed
- Whole-orbit evaluation lost accuracy at high eccentricity (~0.1 R_star at
  e = 0.9) because the time-to-expansion-point table was too coarse. The
  default table now resolves periastron; `expansion_table_size` returns its
  size.
- `zpos_d` and `zvel_d` returned a wrong period derivative for multi-epoch
  arrays of eight or more times (a numba 0.61 miscompilation).
- `solve3d_orbit_d` returned a wrong period derivative for times just before
  periastron.
- `true_anomaly_od` ignored the gradient basis on its circular path. It takes
  a `timing_is_tc` flag (mandatory in OpenCL/C).
- `eccentricity_vector` ignored `lan`, so the true anomaly was wrong whenever
  `lan != 0`.
- `Orbit(derivatives=True).true_anomaly()` returned wrong `w` and `lan`
  gradients for eccentric orbits. **Breaking:** `true_anomaly_od` (numba,
  OpenCL and C) takes the eccentricity-vector Jacobian `dev` from the new
  `eccentricity_vector_d`; pass zeros for the old behaviour.

### Removed
- The JAX prototype modules `meepmeep.backends.jax.ea` and
  `meepmeep.backends.jax.ts2d`, superseded by the JAX backend.

## [1.1.0] - 2026-09-08

### Added
- OpenCL backend (`meepmeep.backends.opencl`): the evaluation surface of the
  numba backend as OpenCL C device functions. Install with
  `pip install meepmeep[opencl]`.

### Changed
- The transit-centre row of the derivative tensors from `solve2d_d`,
  `solve3d_d` and `solve3d_orbit_d` is now the derivative of the evaluated
  Taylor polynomial rather than of the exact orbit, so gradients match the
  returned values.
- `solve3d_orbit_d` returns gradients in the periastron basis
  `(tp, p, a, i, e, w, lan)`. `Orbit` converts them with the new
  `tp_to_tc_gradient` when bound with `tc`.

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
- API cheatsheet for LLMs at `docs/llms.md`.

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
