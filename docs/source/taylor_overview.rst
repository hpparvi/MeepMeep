.. _taylor_overview:

Taylor-series backend overview
==============================

This page is for users who want to drop below the
:class:`~meepmeep.orbit.Orbit` class — to compose custom evaluators,
work with the per-knot Taylor coefficients directly, or differentiate
orbit-derived quantities through their own chain rule. It documents
the low-level backend that :class:`~meepmeep.orbit.Orbit` uses under
the hood.

Mechanically, the backend approximates each Keplerian orbit as a set
of 5th-order Taylor expansions of the planet's trajectory, anchored at
one or more *knot* points along the orbit. The Taylor coefficients are
computed analytically; from them, positions, sky-projected
separations, the line-of-sight coordinate, velocities, contact points
and durations all reduce to fast Horner-scheme polynomial evaluations.

.. admonition:: What is a knot?

   A **knot** is a point along the orbit that serves as the *center* of
   a local 5th-order Taylor expansion of the planet's trajectory in
   time. Each knot carries its own ``(D, 5)`` coefficient matrix, and
   the expansion is most accurate close to the knot, degrading as you
   move away from it.

   The term is borrowed from spline interpolation, but with one
   important difference: a spline knot marks where two neighbouring
   polynomial pieces *join* (a segment boundary), whereas a MeepMeep
   knot sits at the *center* of its segment. The boundaries between the
   regions of validity of adjacent knots are a separate set of times
   (the ``change_times`` returned by
   :func:`~meepmeep.backends.numba.knots.create_knots`), encoded in the
   time-to-knot table used for dispatch. So: knots are expansion
   centers, not segment edges.

.. note::

   The backend documented here is the **numba** implementation,
   currently the only complete one. A JAX backend is partially
   implemented and slated for future development.

.. contents::
   :local:
   :depth: 2


.. _taylor_two_modes:

Two ways to use the Taylor backend
----------------------------------

The backend is designed for two usage modes.
Both modes share the same coefficient-matrix layout and the same family
of evaluators; they differ only in *how many* knots you build and how
the evaluators are dispatched.

**Single-knot evaluation.** Build one set of Taylor coefficients at a
chosen phase (typically the transit center for a transit model, or the
secondary-eclipse center for an eclipse model) and evaluate positions
or projected distance in the time window where the series is
accurate. This is the natural mode for transit and eclipse light-curve
codes, for contact-point and duration calculations via
:mod:`~meepmeep.backends.numba.taylor.point2d.util`, and for inspecting orbit
geometry at a fixed phase. See :ref:`taylor_single_knot`.

**Multi-knot orbit-spanning evaluation.** Distribute :math:`N` knots
around the full orbit, precompute coefficients at each, and dispatch
arbitrary input times to the appropriate knot via a precomputed
time-to-knot table. This is the natural mode for whole-orbit
quantities — radial velocity curves, phase curves, ellipsoidal
variation, light travel time. See :ref:`taylor_multi_knot`.

Coordinate system
-----------------

All sky-plane and 3D positions are expressed in an observer-centred
frame in which the star is at the origin and the planet position is
measured in units of the stellar radius :math:`R_\star`:

* **X-axis** — points to the right along the projected sky plane.
* **Y-axis** — points upward along the projected sky plane.
* **Z-axis** — points toward the observer. Positive :math:`z` is in
  front of the star (transit hemisphere); negative :math:`z` is behind
  (eclipse).

The "projected" or "sky-plane" distance used throughout the code is the
Euclidean norm

.. math::

   d = \sqrt{x^2 + y^2},

i.e. the center-to-center separation between planet and star projected
onto the plane of the sky. It is always non-negative and is the
quantity that transit light-curve models consume directly. The sign of
the line-of-sight depth (transit vs. eclipse) lives in :math:`z`, not
in :math:`d`.

The 2D evaluators in
:mod:`~meepmeep.backends.numba.taylor.point2d.position` compute only
:math:`(x, y)` and :math:`d`, which is sufficient for transit
modelling. The 3D evaluators in
:mod:`~meepmeep.backends.numba.taylor.point3d.position` additionally compute
:math:`z`, which is needed for eclipses, light travel time, phase
curves, and radial velocities.


Orbital parameters and the coefficient matrix
---------------------------------------------

The Keplerian parameter set used by every low-level function is:

============  ====================================================
Symbol        Meaning
============  ====================================================
``tc``        Time of inferior conjunction (transit center).
``p``         Orbital period [days].
``a``         Scaled semi-major axis :math:`a/R_\star`.
``i``         Inclination [radians]. :math:`i = \pi/2` is edge-on.
``e``         Eccentricity, :math:`0 \le e < 1`.
``w``         Argument of periastron [radians].
``lan``       Longitude of the ascending node [radians], optional
              (defaults to 0). A constant rotation of the sky-plane
              :math:`(x, y)` about the line of sight.
============  ====================================================

When parameter derivatives are returned, the canonical ordering is
``(tc, p, a, i, e, w, lan)``. The leading axis of every ``dc`` derivative
tensor follows this order.

Both :func:`~meepmeep.backends.numba.taylor.point2d.solve.solve2d` and
:func:`~meepmeep.backends.numba.taylor.point3d.solve.solve3d` return a
**coefficient matrix** ``c`` of shape ``(D, 5)``:

* :math:`D = 2` for the 2D (sky-plane) solver, :math:`D = 3` for 3D.
* Rows index the spatial dimensions: ``c[0]`` is the x-direction Taylor
  series, ``c[1]`` is y, and (in 3D) ``c[2]`` is z.
* Columns index the Taylor order, from position (column 0) through snap
  (column 4): position, velocity, acceleration, jerk, snap.

The columns are **pre-scaled by the factorial of the Taylor order**,
i.e. ``c[d, k]`` already contains
:math:`\partial^k x_d / \partial t^k\,/\,k!` evaluated at the knot. With
this normalisation the polynomial at time :math:`t` (measured relative
to the knot) is simply

.. math::

   x_d(t) = \sum_{k=0}^{4} c[d, k]\, t^k

which the evaluators compute via Horner's scheme:

.. code-block:: python

   px = c[0, 0] + t*(c[0, 1] + t*(c[0, 2] + t*(c[0, 3] + t*c[0, 4])))

Pre-scaling and Horner evaluation together yield an inner loop with no
factorial divisions and only four fused multiply-adds per spatial
dimension.


.. _taylor_single_knot:

Single-knot evaluation
----------------------

Pick a knot time :math:`t_k` of interest, measured in days relative to
the transit centre (for example, the transit centre itself, :math:`t_k = 0`).
A single call to
:func:`~meepmeep.backends.numba.taylor.point2d.solve.solve2d` (or
:func:`~meepmeep.backends.numba.taylor.point3d.solve.solve3d` for 3D) builds
the ``(D, 5)`` coefficient matrix at that knot:

.. code-block:: python

   from meepmeep.backends.numba.taylor.point2d import solve2d

   c = solve2d(tk, p, a, i, e, w)   # shape (2, 5)

The Taylor expansion is accurate inside a window around the knot whose
size depends on the orbit (more eccentric orbits have shorter windows
near periastron). For transit and eclipse modelling, one knot placed at
the event center is normally enough to cover ingress, totality, and
egress with high accuracy.

**Centered vs. direct evaluators.** Every evaluator ships in two
variants and the choice belongs entirely in this single-knot mode:

* **Centered** variants (``c`` suffix) accept a time that is already
  relative to the knot, :math:`t = t_\mathrm{obs} - t_\mathrm{knot}`,
  and skip any epoch arithmetic. They are the fastest path and the
  natural choice when the observation times have already been folded
  around the event.
  Examples: :func:`~meepmeep.backends.numba.taylor.point2d.position.pos_c`,
  :func:`~meepmeep.backends.numba.taylor.point2d.separation.sep_c`,
  :func:`~meepmeep.backends.numba.taylor.point2d.position.pos_and_sep_c`,
  :func:`~meepmeep.backends.numba.taylor.point3d.position.pos_c`,
  :func:`~meepmeep.backends.numba.taylor.point3d.separation.sep_c`,
  :func:`~meepmeep.backends.numba.taylor.point3d.zposition.zpos_c`.

* **Direct** variants (no ``c`` suffix) accept an absolute time
  together with the knot time ``tk`` and ``p`` and epoch-fold internally via
  ``epoch = floor((t - tk + p/2) / p)`` so the residual lies in
  :math:`[-p/2,\, p/2)`. Use them when callers prefer to hand in raw
  observation times.
  Examples: :func:`~meepmeep.backends.numba.taylor.point2d.position.pos`,
  :func:`~meepmeep.backends.numba.taylor.point2d.separation.sep`,
  :func:`~meepmeep.backends.numba.taylor.point3d.position.pos`,
  :func:`~meepmeep.backends.numba.taylor.point3d.separation.sep`,
  :func:`~meepmeep.backends.numba.taylor.point3d.zposition.zpos`.

**Geometric helpers.** The
:mod:`~meepmeep.backends.numba.taylor.point2d.util` and
:mod:`~meepmeep.backends.numba.taylor.point3d.util` modules supply analytic
helpers that operate directly on a single ``c``:

* ``t14`` / ``t23`` — full (first-to-fourth contact) and total
  (second-to-third contact) durations.
* ``t12`` / ``t34`` — ingress and egress durations.
* ``t1`` / ``t4`` — first and fourth contact times.
* ``find_contact_point`` — generic contact-point solver.
* ``find_z_min`` — time of minimum projected separation.
* ``bounding_box`` — axis-aligned bounding box of the orbit segment
  spanned by the knot.

Because they only need one coefficient matrix, they slot naturally
into single-knot pipelines such as transit duration calculators.

**Single-knot quickstart.** A minimal transit-window evaluation:

.. code-block:: python

   import numpy as np
   from meepmeep.backends.numba.taylor.point2d import solve2d
   from meepmeep.backends.numba.taylor.point2d import sep_c

   # Orbital parameters (tc is the transit-centre time)
   tc, p, a, i, e, w = 0.0, 3.0, 8.5, np.radians(89.0), 0.1, np.radians(90.0)

   # One knot at the transit centre (tk = 0, so the knot sits at tc)
   c = solve2d(0.0, p, a, i, e, w)

   # Centered evaluation: t is measured from the knot
   dt = np.linspace(-0.05, 0.05, 1001)
   d = sep_c(dt, c)

If you would rather hand in absolute times, swap ``sep_c(dt, c)`` for
``sep(t, tc, p, c)`` and pass ``t = tc + dt`` (the knot is at the transit
centre, so its ``tk`` equals ``tc`` here). The result is identical;
only the epoch-folding is now done by the evaluator.

**Single-knot gradients.** For analytic derivatives with respect to the
seven orbital parameters, replace
:func:`~meepmeep.backends.numba.taylor.point2d.solve.solve2d` with
:func:`~meepmeep.backends.numba.taylor.point2dd.solve.solve2d_d` (or
:func:`~meepmeep.backends.numba.taylor.point3dd.solve.solve3d_d`). The solver
returns both ``c`` and a ``(7, D, 5)`` derivative tensor ``dc``; feed
them to the matching centered-with-derivatives evaluators
(:func:`~meepmeep.backends.numba.taylor.point2dd.position.pos_cd`,
:func:`~meepmeep.backends.numba.taylor.point2dd.separation.sep_cd`,
:func:`~meepmeep.backends.numba.taylor.point3dd.position.pos_cd`,
:func:`~meepmeep.backends.numba.taylor.point3dd.separation.sep_cd`,
:func:`~meepmeep.backends.numba.taylor.point3dd.zposition.zpos_cd`, and so on).
See :ref:`taylor_derivatives` for the gradient conventions.


.. _taylor_multi_knot:

Multi-knot orbit-spanning evaluation
------------------------------------

A single 5th-order Taylor series is only accurate in a small
neighbourhood of its expansion point. To evaluate the orbit at *any*
phase — for whole-orbit observables such as RV curves and phase
curves — MeepMeep distributes :math:`N` knots along one orbital period
and stores a separate coefficient matrix at each. Lookups from an
input time to the relevant knot are done by a precomputed time-to-knot
table.

**Knot placement strategies** are selectable via the ``quantity``
keyword of :func:`~meepmeep.backends.numba.knots.create_knots`:

* ``'mm'`` — uniform in mean motion (uniform in time).
* ``'ea'`` — uniform in eccentric anomaly (default; preferred for
  moderate to high eccentricity).
* ``'ta'`` — uniform in true anomaly.

The eccentric-anomaly placement clusters knots near periastron, where
the orbital motion is fast and the validity window of each Taylor
series is shortest.

**Per-orbit coefficient assembly.**
:func:`~meepmeep.backends.numba.taylor.orbit3d.solve3d_orbit` calls
:func:`~meepmeep.backends.numba.taylor.point3d.solve.solve3d` once per knot
and returns an ``(N, 3, 5)`` array. The function expects the *last*
knot time to be the periodic image of the first (i.e. one period
later); when this is true it copies the first knot's coefficients into
the last slot instead of recomputing them.
:func:`~meepmeep.backends.numba.knots.create_knots` produces compliant
input automatically; if you hand-roll the knot grid you must enforce
this contract yourself.

**Time-to-knot dispatch.** Each multi-knot evaluator carries a
``pktable`` argument — a precomputed table that maps the position
within one folded period to a knot index in :math:`O(1)`. The dispatch
helper is
:func:`~meepmeep.backends.numba.taylor.orbit3d.knot_ix`.

**Dispatcher suffix convention.** Whole-orbit functions in
:mod:`~meepmeep.backends.numba.taylor.orbit3d` use a different
suffix family from the single-knot evaluators:

* ``*_os`` — orbit-spanning, **s**\ calar input time.
* ``*_ov`` — orbit-spanning, **v**\ ector of input times.

Each dispatcher looks up the relevant knot via ``pktable`` and
delegates to the corresponding centered evaluator in the
:mod:`~meepmeep.backends.numba.taylor.point3d` package. Beyond raw
positions and velocities, the module
provides higher-level whole-orbit outputs:

* :func:`~meepmeep.backends.numba.taylor.orbit3d.true_anomaly_ov`,
  :func:`~meepmeep.backends.numba.taylor.orbit3d.cos_alpha_ov` —
  phase-angle quantities.
* :func:`~meepmeep.backends.numba.taylor.orbit3d.star_planet_distance_ov`
  — 3D separation.
* :func:`~meepmeep.backends.numba.taylor.orbit3d.lambert_phase_curve_ov`,
  :func:`~meepmeep.backends.numba.taylor.orbit3d.lambert_and_emission_ov`
  — reflected-light and combined reflection + thermal phase curves.
* :func:`~meepmeep.backends.numba.taylor.orbit3d.rv_ov` — radial
  velocity.
* :func:`~meepmeep.backends.numba.taylor.orbit3d.ev_signal_ov` —
  ellipsoidal-variation signal.
* :func:`~meepmeep.backends.numba.taylor.orbit3d.light_travel_time_os`
  / :func:`~meepmeep.backends.numba.taylor.orbit3d.light_travel_time_ov`
  — light travel time corrections.

**Multi-knot quickstart.** Build the per-orbit structures once and
evaluate at an array of times:

.. code-block:: python

   import numpy as np
   from meepmeep.backends.numba.knots import create_knots
   from meepmeep.backends.numba.taylor.orbit3d import solve3d_orbit, pos_ov
   from meepmeep.backends.numba.utils import mean_anomaly_at_transit

   # Orbital parameters (tc is the transit-center time, the high-level convention)
   tc, p, a, i, e, w = 0.0, 3.0, 8.5, np.radians(89.0), 0.1, np.radians(90.0)

   # The orbit3d dispatchers anchor their knot grid at periastron, so
   # convert the user-facing transit-center time to the periastron-anchor time.
   tpa = tc - mean_anomaly_at_transit(e, w) / (2.0 * np.pi) * p

   # Knot grid (eccentric-anomaly placement) and the time-to-knot table
   npt = 15
   knot_times, points, pktable, dt = create_knots(npt, e, quantity='ea')

   # Pre-compute Taylor coefficients at every knot
   coeffs = solve3d_orbit(knot_times, p, a, i, e, w, npt=npt)

   # Evaluate the orbit at a grid of times
   times = np.linspace(0.0, p, 2001)
   xs, ys, zs = pos_ov(times, tpa, p, dt, pktable, points, coeffs)


.. _taylor_derivatives:

Parameter derivatives
---------------------

The ``d``-suffixed packages (the ``point2dd`` and ``point3dd`` single-knot
packages and the ``orbit3dd`` multi-knot package) extend
the backend with analytic partial derivatives of every output with
respect to the seven orbital parameters. The same machinery is available
in both usage modes. See :ref:`derivatives` for the full derivation,
with equations for the Kepler implicit-differentiation step, the
orbital-plane derivative chain, and the chain rules used by every
evaluator and dispatcher.

For **single-knot gradients**, swap
:func:`~meepmeep.backends.numba.taylor.point2d.solve.solve2d` /
:func:`~meepmeep.backends.numba.taylor.point3d.solve.solve3d` for their
``_d`` counterparts. They return a coefficient matrix ``c`` *and* a
**derivative tensor** ``dc`` of shape ``(7, D, 5)``:

* Axis 0 — orbital parameter index in the canonical order
  ``(tc, p, a, i, e, w, lan)``.
* Axes 1, 2 — spatial dimension and Taylor order, matching ``c``.

Every gradient-returning evaluator
(e.g. :func:`~meepmeep.backends.numba.taylor.point3dd.position.pos_d`,
:func:`~meepmeep.backends.numba.taylor.point3dd.separation.sep_d`,
:func:`~meepmeep.backends.numba.taylor.point3dd.velocity.vel_cd`) accepts
both ``c`` and ``dc`` and returns the value alongside a length-7
gradient vector. The naming convention is ``_d`` for direct (absolute-
time) variants and ``_cd`` for centered ones.

For **multi-knot gradients**, the assembly side becomes
:func:`~meepmeep.backends.numba.taylor.orbit3dd.solve3d_orbit_d`, which
returns ``(N, 3, 5)`` coefficients and an
``(N, 7, 3, 5)`` derivative tensor. The dispatcher counterparts in
:mod:`~meepmeep.backends.numba.taylor.orbit3dd` share the names of the
non-derivative dispatchers with an added ``_d`` suffix
(``pos_ovd``, ``zvel_osd``, ``rv_ovd``, and so on).

Distance derivatives are reduced from the position gradients via the
chain rule

.. math::

   \frac{\partial d}{\partial \theta} =
       \frac{p_x\, \partial p_x/\partial\theta + p_y\, \partial p_y/\partial\theta}{d},

which is well-behaved as long as :math:`d > 0` — the regime of
interest for transit modelling.
