# Changelog

All notable changes to MeepMeep are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
