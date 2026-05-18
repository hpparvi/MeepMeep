.. _taylor_overview:

Taylor-series backend overview
==============================

The numba backend evaluates Keplerian orbits via piecewise 5th-order Taylor
expansions around a set of *knots* distributed along one orbital period.
Once the coefficients at each knot are precomputed, evaluating the orbit at
any time reduces to (i) looking up the appropriate knot, (ii) shifting the
time to be relative to that knot, and (iii) evaluating a small polynomial
with Horner's scheme.

This page introduces the coordinate system, the orbital parameter set used
throughout the package, the structure of the Taylor coefficient arrays, the
two-function (centered / direct) evaluator pattern, and the parameter
derivative machinery. A short worked example at the end ties the pieces
together.

.. contents::
   :local:
   :depth: 2


Coordinate system
-----------------

All sky-plane and 3D positions are expressed in an observer-centred frame in
which the star is at the origin and the planet position is measured in units
of the stellar radius :math:`R_\star`:

* **X-axis** — points to the right along the projected sky plane.
* **Y-axis** — points upward along the projected sky plane.
* **Z-axis** — points toward the observer. Positive :math:`z` is in front of
  the star (transit hemisphere); negative :math:`z` is behind (eclipse).

The "projected" or "sky-plane" distance used throughout the code is the
Euclidean norm

.. math::

   d = \sqrt{x^2 + y^2},

i.e. the center-to-center separation between planet and star projected onto
the plane of the sky. It is always non-negative and is the quantity that
transit light-curve models consume directly. The sign of the line-of-sight
depth (transit vs. eclipse) lives in :math:`z`, not in :math:`d`.

The 2D evaluators in :mod:`~meepmeep.backends.numba.taylor.position2d`
compute only :math:`(x, y)` and :math:`d`, which is sufficient for transit
modelling. The 3D evaluators in
:mod:`~meepmeep.backends.numba.taylor.position3d` additionally compute
:math:`z`, which is needed for eclipses, light travel time, phase curves,
and radial velocities.


Orbital parameters
------------------

The Keplerian parameter set used by every low-level function is:

============  ====================================================
Symbol        Meaning
============  ====================================================
``t0``        Time of inferior conjunction (transit center).
``p``         Orbital period [days].
``a``         Scaled semi-major axis :math:`a/R_\star`.
``i``         Inclination [radians]. :math:`i = \pi/2` is edge-on.
``e``         Eccentricity, :math:`0 \le e < 1`.
``w``         Argument of periastron [radians].
============  ====================================================

When parameter derivatives are returned, the canonical ordering is
``(t0, p, a, i, e, w)``. The leading axis of every ``dc`` derivative tensor
follows this order.


The Taylor coefficient matrix
-----------------------------

Both :func:`~meepmeep.backends.numba.taylor.solve2d.solve2d` and
:func:`~meepmeep.backends.numba.taylor.solve3d.solve3d` return a *coefficient
matrix* ``c`` of shape ``(D, 5)``:

* :math:`D = 2` for the 2D (sky-plane) solver, :math:`D = 3` for 3D.
* Rows index the spatial dimensions: ``c[0]`` is the x-direction Taylor
  series, ``c[1]`` is y, and (in 3D) ``c[2]`` is z.
* Columns index the Taylor order, from position (column 0) through snap
  (column 4): position, velocity, acceleration, jerk, snap.

The columns are **pre-scaled by the factorial of the Taylor order**, i.e.
``c[d, k]`` already contains :math:`\partial^k x_d / \partial t^k\,/\,k!`
evaluated at the knot. With this normalisation the polynomial at time
:math:`t` (measured relative to the knot) is simply

.. math::

   x_d(t) = \sum_{k=0}^{4} c[d, k]\, t^k

which the evaluators compute via Horner's scheme:

.. code-block:: python

   px = c[0, 0] + t*(c[0, 1] + t*(c[0, 2] + t*(c[0, 3] + t*c[0, 4])))

Pre-scaling and Horner evaluation together yield an inner loop with no
factorial divisions and only four fused multiply-adds per spatial
dimension.


Knots and the time-to-knot table
--------------------------------

A single 5th-order Taylor series is only accurate in a small neighbourhood
of its expansion point. To evaluate the orbit at *any* phase, MeepMeep
distributes :math:`N` knots along one orbital period and stores a separate
coefficient matrix at each knot. Three knot-placement strategies are
available, selectable via the ``quantity`` keyword of
:func:`~meepmeep.backends.numba.knots.create_knots`:

* ``'mm'`` — uniform in mean motion (uniform in time).
* ``'ea'`` — uniform in eccentric anomaly (default; preferred for moderate
  to high eccentricity).
* ``'ta'`` — uniform in true anomaly.

The eccentric-anomaly placement clusters knots near periastron, where the
orbital motion is fast and the Taylor series shortest-lived.

Per-orbit coefficients are precomputed in one call to
:func:`~meepmeep.backends.numba.taylor.orbit3d.solve3d_orbit`, which returns
an ``(N, 3, 5)`` array. The function expects the *last* knot time to be the
periodic image of the first (i.e. one period later); when this is true it
copies the first knot's coefficients into the last slot instead of
recomputing them. ``create_knots`` produces compliant input automatically;
if you hand-roll the knot grid, you need to enforce this contract yourself.

Lookups from a time to a knot index are accelerated by a precomputed
**time-to-knot table** (the ``pktable`` argument carried by every
``*_o5s`` / ``*_o5v`` function). The dispatch is performed by
:func:`~meepmeep.backends.numba.taylor.orbit3d.knot_ix`, which epoch-folds
the absolute time into one period and indexes the table.


Centered vs. direct evaluators
------------------------------

Every quantity that the backend can evaluate ships in two variants:

**Direct** — accepts an absolute observation time. The function
epoch-folds the time around the expansion point using
``epoch = floor((t - t0 + p/2) / p)`` so that the residual lies in
:math:`[-p/2,\, p/2)`, then evaluates the polynomial.
Examples: :func:`~meepmeep.backends.numba.taylor.position2d.p2d`,
:func:`~meepmeep.backends.numba.taylor.position3d.p3d`,
:func:`~meepmeep.backends.numba.taylor.position3d.d3d`.

**Centered** — accepts a time that is already relative to the knot. No
epoch folding is performed; this is the fastest path. Use it whenever the
knot index and centered time are already known (e.g. inside multi-knot
dispatch loops). Centered variants carry a ``c`` suffix on the function
name: ``p2dc``, ``p3dc``, ``d3dc``, ``z3dc``, ``pd2d_c``, etc.

Both variants share the same coefficient matrix ``c`` and produce identical
results when fed equivalent times.


Parameter derivatives
---------------------

Modules suffixed with a ``d`` (``solve2dd``, ``position2dd``,
``solve3dd``, ``position3dd``, ``velocity3dd``, ``orbit3dd``) extend the
backend with analytic partial derivatives of every output with respect to
the six orbital parameters.

The solvers
:func:`~meepmeep.backends.numba.taylor.solve2dd.solve2d_d` and
:func:`~meepmeep.backends.numba.taylor.solve3dd.solve3d_d` additionally
return a **derivative tensor** ``dc`` of shape ``(6, D, 5)``:

* Axis 0 — orbital parameter index in the canonical order
  ``(t0, p, a, i, e, w)``.
* Axes 1, 2 — spatial dimension and Taylor order, matching ``c``.

Every evaluator with a ``_d`` suffix (e.g.
:func:`~meepmeep.backends.numba.taylor.position3dd.p3d_d`,
:func:`~meepmeep.backends.numba.taylor.position3dd.d3d_d`,
:func:`~meepmeep.backends.numba.taylor.velocity3dd.v3dc_d`) accepts both
``c`` and ``dc`` and returns the value alongside a length-6 gradient
vector. Distance derivatives are reduced from the position gradients via
the chain rule

.. math::

   \frac{\partial d}{\partial \theta} =
       \frac{p_x\, \partial p_x/\partial\theta + p_y\, \partial p_y/\partial\theta}{d},

which is well-behaved as long as :math:`d > 0` — the regime of interest
for transit modelling.


Multi-knot dispatchers
----------------------

The :mod:`~meepmeep.backends.numba.taylor.orbit3d` module ties the
single-knot evaluators together so that arbitrary time arrays can be
evaluated across a full orbit:

* ``*_o5s`` — 5th-order Taylor, scalar (single time).
* ``*_o5v`` — 5th-order Taylor, vector (array of times).

Each dispatcher looks up the relevant knot via ``pktable`` and delegates
to the corresponding centered evaluator in :mod:`position3d` /
:mod:`velocity3d`. Derivative-aware counterparts live in
:mod:`~meepmeep.backends.numba.taylor.orbit3dd` and carry the same names
with a ``_d`` suffix.

Beyond raw positions and velocities, :mod:`orbit3d` exposes higher-level
quantities built on the same dispatch machinery:

* True anomaly, phase angle, and cosine of the planet-star phase angle.
* Star-planet 3D separation.
* Lambertian phase curves and combined Lambert + thermal emission models.
* Radial velocity (``rv_o5v``) and ellipsoidal-variation signal
  (``ev_signal_o5v``).
* Light travel time corrections (``light_travel_time_o5s/v``).


Utility functions
-----------------

The :mod:`~meepmeep.backends.numba.taylor.util2d` and
:mod:`~meepmeep.backends.numba.taylor.util3d` modules supply geometric
helpers that operate directly on a coefficient matrix ``c``:

* ``t14`` / ``t23`` — full and total transit durations.
* ``t12`` / ``t34`` — ingress and egress durations.
* ``t1`` / ``t4`` — first and fourth contact times.
* ``find_contact_point`` — generic contact-point solver.
* ``find_z_min`` — time of minimum projected separation around a knot.
* ``bounding_box`` — axis-aligned bounding box of the orbit segment.


Minimal usage example
---------------------

The snippet below builds the per-orbit coefficient and time-to-knot
structures and then evaluates the 3D position at an array of times:

.. code-block:: python

   import numpy as np
   from meepmeep.backends.numba.knots import create_knots
   from meepmeep.backends.numba.taylor.orbit3d import solve3d_orbit, xyz_o5v

   # Orbital parameters
   t0, p, a, i, e, w = 0.0, 3.0, 8.5, np.radians(89.0), 0.1, np.radians(90.0)

   # Knot grid (eccentric-anomaly placement) and the time-to-knot table
   npt = 15
   knot_times, points, pktable, dt = create_knots(npt, e, quantity='ea')

   # Pre-compute Taylor coefficients at every knot
   coeffs = solve3d_orbit(knot_times, p, a, i, e, w, npt)

   # Evaluate the orbit at a grid of times
   times = np.linspace(-0.1, 0.1, 2001)
   xs, ys, zs = xyz_o5v(times, t0, p, dt, pktable, points, coeffs)

To compute analytic gradients with respect to ``(t0, p, a, i, e, w)``,
swap in :func:`~meepmeep.backends.numba.taylor.orbit3dd.solve3d_orbit_d`
and :func:`~meepmeep.backends.numba.taylor.orbit3dd.xyz_o5v_d`; the latter
returns both the positions and a ``(6, ...)`` gradient array.
