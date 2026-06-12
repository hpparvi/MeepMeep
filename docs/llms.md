# MeepMeep: API cheatsheet for LLM agents

MeepMeep computes Keplerian orbit quantities for exoplanet modelling
(transit geometry, radial velocities, phase curves) using 5th-order Taylor
expansions around knot points, roughly an order of magnitude faster than
per-point Newton-Raphson. Optional analytic gradients with respect to the
orbital parameters make it suitable for gradient-based fitting (HMC,
optimisers). All hot paths are Numba-jitted and callable from user `@njit`
code. Method: Parviainen & Korth (2020), MNRAS 499, 3356.

This document is the consumer-facing reference. It prioritises the
conventions and pitfalls that are not obvious from any single docstring.

## Stability contract and imports

Import ONLY from these entry points; everything under
`meepmeep.backends/` is implementation detail and may be restructured
without notice:

```python
from meepmeep import Orbit, eclipse_light_travel_time  # high-level
from meepmeep.knot2d import Knot2D                     # high-level, single knot, 2D
import meepmeep.numba2d as mm2                         # low-level 2D primitives
import meepmeep.numba3d as mm3                         # low-level 3D + multi-knot + utils
```

The low-level functions are plain `@njit` functions / Numba overload
dispatchers: they can be called from Python or from inside user `@njit`
kernels with no wrapper overhead. This composability is the main reason
to use the low-level API.

## Conventions (apply everywhere)

- Units: times in days, angles in RADIANS, lengths in stellar radii
  (a is the scaled semi-major axis a/R_star).
- Orbital parameter order, used by all solvers and gradients:
  `(tc, p, a, i, e, w, lan)` = (transit centre time, period, scaled
  semi-major axis, inclination, eccentricity, argument of periastron,
  longitude of ascending node). `lan` is optional, defaults to 0.0, and
  is a constant sky-plane rotation about the line of sight.
- Coordinates: x, y span the sky plane; z is the line of sight,
  POSITIVE TOWARD the observer. Transit happens at z > 0, secondary
  eclipse at z < 0. `i = pi/2` is edge-on.
- "Projected separation" = sqrt(x^2 + y^2) in stellar radii; this is the
  z/b(t) quantity transit light-curve models consume. It is always
  non-negative and does NOT encode the transit/eclipse branch - use the
  sign of the z coordinate (`zpos*`) for that.
- Scalar-or-array dispatch: every evaluator accepts a scalar time or a
  1-D float64 array and returns matching shapes (scalar in -> scalar
  out). Inside `@njit` the dispatch happens at compile time. Pass float64
  arrays, not lists.
- Eccentricity range 0 <= e < 1. For e < 1e-5 the eccentricity vector
  used by `true_anomaly_o/_od` degenerates to a sentinel and the true
  anomaly equals the mean anomaly.

## Gradients

Constructing with `derivatives=True` (or using `*_d` / `*_cd` / `*_od`
low-level variants) returns analytic gradients alongside values:

- Gradient axis order is `(tc, p, a, i, e, w, lan)`; shape `(7,)` for a
  scalar time, `(N, 7)` for an array of N times.
- Methods with extra physical inputs append their derivatives AFTER the
  orbital block, in argument order:
  - `radial_velocity(k)` -> gradient length 8: `(..., k)`
  - `lambert_phase_curve(k, ag)` -> length 9: `(..., ag, k)`
  - `lambert_and_emission(k, ag, fr_night, fr_day)` -> length 12:
    `(..., ag, fr_night, fr_day, emi_offset, k)`
  - `ellipsoidal_variation(alpha, mass_ratio)` -> length 10:
    `(..., alpha, mass_ratio, inc)`
  - `light_travel_time(rstar)` -> length 7 only (no rstar derivative).
- TIMING BASIS: gradients follow the timing parameter you bind. Bind
  `set_pars(tc=...)` and slot 0 is d/dtc with shape parameters taken at
  constant tc (the default). Bind `set_pars(tp=...)` and the gradient is
  returned in the periastron basis `(tp, p, a, i, e, w, lan)`. The
  conversion utility is `numba3d.tc_to_tp_gradient`.

## High-level API: Orbit (3D, multi-knot, any orbital phase)

```python
o = Orbit(npt=15, knot_placement="ea", derivatives=False, parallel=False)
o.set_pars(tc=0.0, p=3.4, a=8.0, i=1.55, e=0.1, w=0.4)   # or tp=... instead of tc
o.set_data(times)                                          # 1-D float64 array
x, y, z = o.xyz()              # add , dx, dy, dz when derivatives=True
```

- `set_pars` is KEYWORD-ONLY and takes exactly one of `tc` / `tp`.
  It recomputes the Taylor coefficients; call it once per parameter set
  (it is the per-likelihood-call hot path and is cheap).
- Methods (each returns per-time arrays; with `derivatives=True` the
  gradients listed above are appended to the return tuple):
  `xyz(times=None)`, `vxyz()`, `star_planet_distance(times=None)` (3D
  distance, NOT projected), `cos_phase()`, `phase()`, `theta()`,
  `mean_anomaly()`, `true_anomaly(exact=False)`,
  `radial_velocity(k)` (k = RV semi-amplitude, output in k's units),
  `lambert_phase_curve(k, ag, times=None)` (k = radius ratio),
  `lambert_and_emission(k, ag, fr_night, fr_day, times=None)` (returns
  reflected and emitted flux ratios separately),
  `ellipsoidal_variation(alpha, mass_ratio, times=None)`,
  `light_travel_time(rstar)` (rstar in solar radii, result in days,
  zero-referenced at primary transit), `plot()`.
- `npt` is the knot count; raise it (e.g. 25) for high-e orbits if more
  accuracy is needed. The knot grid auto-adapts to the bound
  eccentricity. Expected accuracy regime at npt=15: ~1e-4 R_star
  worst-case position error over a full period.
- `parallel=True` routes large-array evaluations to multi-threaded
  kernels (identical results). Worth it for N >= ~1e4 (gradients) /
  ~5e4 (values); 3-8x on a 16-core machine. LEAVE IT OFF when the
  application already parallelises at process level (e.g. one process
  per MCMC chain) - nested thread pools oversubscribe the machine.

## High-level API: Knot2D (single knot, near-transit, 2D)

Cheapest path for pure transit modelling where only the sky plane near
the transit matters:

```python
k2 = Knot2D(tc=0.0, p=3.4, a=8.0, i=1.55, e=0.1, w=0.4,
            lan=0.0, tk=0.0, derivatives=False, parallel=False)
k2.set_data(times)
z = k2.projected_separation     # property; (z, dz) when derivatives=True
x, y = k2.position              # property
k2.duration(k, kind=14)         # kind in {14, 23, 12, 34} (total, full,
                                # ingress, egress) [days]
k2.contact_point(k, point)      # point in 1..4; ABSOLUTE time
k2.bounding_box(k)              # (T1, T4) absolute contact times
k2.min_separation(guess=0.0)    # (t_min, z_min); guess is an offset in
                                # days from the knot, t_min is ABSOLUTE
```

`set_pars(...)` (keyword-only) rebinds the orbital elements; the knot
offset `tk` (expansion point at `tc + tk`) is a construction-time
constant that `set_pars` keeps. Accuracy degrades away
from the transit; do not use Knot2D for full-orbit quantities.
`Knot2D(..., parallel=True)` multi-threads the position/separation
properties for large grids (>= ~1e4 points in derivative mode, ~1e5 in
value mode; identical results) - same caveat as Orbit's `parallel` flag
about process-level parallelism.

## Low-level API

2D (`meepmeep.numba2d`): `solve2d(tk, p, a, i, e, w, lan=0.0)` returns a
(2, 5) Taylor coefficient matrix for a knot at time `tk` RELATIVE TO THE
TRANSIT CENTRE (tk=0 expands at transit). Evaluate with
`pos(time, tc, p, c, tk=0.0)` / `sep(time, tc, p, c, tk=0.0)`: absolute
times, epoch-folded; `tc` is the transit-centre time on the same axis as
`time`, and the optional trailing `tk` is the same knot offset given to
`solve2d`. The `_c` variants take knot-centred times and are fastest.
`solve2d_d` additionally returns a (7, 2, 5) derivative tensor consumed
by `pos_d` / `sep_d` / `pos_cd` / `sep_cd`. Transit geometry helpers
(`t14`, `t23`, `t12`, `t34`, `t1`, `t4`, `bounding_box`,
`find_contact_point`, `find_z_min`) take the radius ratio `k` and the
coefficient matrix and work in knot-centred time (offsets from the
expansion point, unlike the `Knot2D` methods, which return absolute
times).

3D (`meepmeep.numba3d`): the same single-knot families with a (3, 5)
matrix from `solve3d` (adds `zpos*`, `vel_c`, `zvel*`, `rv*`), plus the
multi-knot orbit-spanning evaluators. Multi-knot workflow:

```python
import numpy as np
from meepmeep.numba3d import create_knots, solve3d_orbit, pos_o, sep_o
from meepmeep.backends.numba.utils import mean_anomaly_at_transit  # sanctioned deep import

knot_times, _, dt, pktable = create_knots(npt, max(e, 0.2), "ea")
coeffs = solve3d_orbit(knot_times, p, a, i, e, w, lan, npt=npt)
# Multi-knot evaluators anchor at the PERIASTRON time tpa, not tc:
tpa = tc - mean_anomaly_at_transit(e, w) / (2.0 * np.pi) * p
x, y, z = pos_o(times, tpa, p, dt, pktable, knot_times, coeffs)
```

Gradient counterparts: `solve3d_orbit_d` -> `(coeffs, dcoeffs)`, then
`pos_od(times, tpa, p, dt, pktable, knot_times, coeffs, dcoeffs)` etc.
Available `_o`/`_od` quantities: `pos`, `zpos`, `sep`, `vel`, `zvel`,
`rv`, `true_anomaly`, `cos_alpha` (phase-angle cosine), `cos_v_p_angle`
(angle to a fixed vector), `star_planet_distance`, `lambert_phase_curve`,
`lambert_and_emission`, `ev_signal`, `light_travel_time`.

## Pitfalls (the things agents get wrong)

1. `k` MEANS TWO THINGS: planet-to-star radius ratio in transit-geometry
   and Lambert functions, RV semi-amplitude in `rv*` /
   `radial_velocity`. Check the docstring of the specific function.
2. tc vs tpa anchoring: high-level `Orbit` accepts either `tc` or `tp`;
   the low-level multi-knot `*_o`/`*_od` evaluators ALWAYS take the
   periastron time `tpa`, while the single-knot direct evaluators take
   the transit centre `tc` (plus an optional knot offset `tk`). Mixing
   these up gives phase-shifted orbits, not errors.
3. The gradient basis silently follows the bound timing parameter
   (tc vs tp). Finite-difference checks must perturb the same basis.
4. Angles are radians everywhere. Inclination near pi/2, not near 90.
5. `sep`/`projected_separation` is unsigned; transit vs eclipse must be
   resolved with the z coordinate.
6. Pass float64 1-D arrays as times inside `@njit`; other dtypes or 2-D
   arrays will not match the overload and fail to compile.
7. `set_pars` validates nothing; e >= 1 or nonphysical inputs produce
   garbage, not exceptions.
8. Time anchor of the geometry helpers: the low-level `t14`/`t1`/
   `find_contact_point`/`find_z_min` family returns knot-centred offsets;
   the `Knot2D` methods wrapping them return ABSOLUTE times.

## Validation and testing

The exact Newton-Raphson reference solvers live in
`meepmeep.backends.numba.newton.newton` (this is the one sanctioned
deep import): `xyz_newton_v(times, tc, p, a, i, e, w)`,
`ta_newton_v(times, tc, p, e, w)`, `rv_newton_v(times, k, tc, p, e, w)`.
These anchor at the TRANSIT centre and are the ground truth the package
itself is tested against - prefer them as oracles in downstream tests
(typical agreement: ~1e-3 absolute over a full period at npt=15,
much better near transit). When finite-difference-testing gradients,
sample a narrow near-transit window; timing/period perturbations can
remap a time across a knot boundary and give O(1) FD errors at isolated
points.

`eclipse_light_travel_time(p, a, i, e, w, rstar)` (top-level export)
returns the transit-to-eclipse light-travel delay in days.
