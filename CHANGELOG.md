# Changelog

All notable changes to MeepMeep are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  return, which is what gradient-based optimisers and samplers need, and
  it agrees with finite differences of any `_d` evaluator with respect to
  its `tc` argument to round-off at any time. The period column of the
  absolute-time evaluators inherits the change through the period-folding
  term. The OpenCL device functions consume the host-solved tensors and
  follow automatically. See "Step 9 — the transit-centre row" in the
  derivatives documentation.
- The orbit-spanning solver `solve3d_orbit_d` now anchors the point solver at
  periastron (new `from_periastron` argument of `solve2d_d` / `solve3d_d`),
  where its expansion points actually sit, and returns the periastron basis
  `(tp, p, a, i, e, w, lan)` natively. `Orbit` applies the new
  `tp_to_tc_gradient` when bound with `tc` instead of `tc_to_tp_gradient`
  when bound with `tp`. Previously the shape rows were differentiated at fixed
  expansion time from the transit centre while the expansion points moved
  with the periastron time, which left the `e`, `w` and `p` columns of an
  `Orbit` gradient off by the same fifth-order term in both bases (about
  1e-4 within 0.04 d of transit for a hot-Jupiter orbit). Finite differences
  of any `Orbit` quantity, in either basis, now reproduce the analytic
  gradient to round-off. `tc_to_tp_gradient` is kept for transit-centre
  anchored blocks from the single-expansion-point solvers, for which it is
  exact.

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
