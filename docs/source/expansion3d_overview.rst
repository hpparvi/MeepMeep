.. _expansion3d_overview:

Expansion3D class overview
==========================

:class:`~meepmeep.expansion3d.Expansion3D` is the 3D counterpart of
:class:`~meepmeep.expansion2d.Expansion2D`. It builds a *single*
5th-order Taylor expansion of the planet's trajectory at one chosen phase
— typically the transit or eclipse centre — but unlike the 2D class it
keeps the full three-dimensional motion, so it exposes the line-of-sight
:math:`z` coordinate, the velocity vector, the radial velocity, the phase
angle, and the reflected-light, thermal-emission, and
ellipsoidal-variation observables, all from the same ``(3, 5)``
coefficient matrix.

Think of it as the middle ground between the two existing entry points.
:class:`~meepmeep.expansion2d.Expansion2D` is single-phase and 2D;
:class:`~meepmeep.orbit.Orbit` is whole-orbit and 3D.
``Expansion3D`` is single-phase and 3D: it gives the richer 3D and
photometric quantities of ``Orbit`` within the time window a single
expansion covers, at the lower setup cost of a single Taylor solve and
the simpler plain-``tc`` time anchor of ``Expansion2D``.

The trade is the same as for ``Expansion2D``: a single expansion point is
accurate only in the time window the series covers (a transit, an
eclipse, a fixed-phase snapshot). For quantities that must be correct at
*every* phase — a full-orbit RV curve or phase curve — reach for
:class:`~meepmeep.orbit.Orbit`, which stitches a grid of expansion points
across the whole period.

With ``derivatives=True`` every quantity also returns analytic gradients
w.r.t. the orbital parameters (plus any method-specific extras), which is
what gradient-based optimisers and HMC samplers want.

.. contents::
   :local:
   :depth: 2


Quickstart
----------

.. code-block:: python

   import numpy as np
   from meepmeep.expansion3d import Expansion3D

   o = Expansion3D(tc=0.0, p=3.0, a=8.5, i=np.radians(89.0),
                   e=0.1, w=np.radians(90.0))
   o.set_data(np.linspace(-0.05, 0.05, 1001))

   x, y, z = o.position()             # 3D sky-frame position per time
   d = o.projected_separation()       # sky-projected separation
   vx, vy, vz = o.velocity()          # velocity vector

As with :class:`~meepmeep.expansion2d.Expansion2D`, the observables are
**methods**: calling them evaluates the bound time grid.

For analytic gradients, switch to derivative mode at construction time
and unpack the extra arrays:

.. code-block:: python

   from meepmeep.expansion3d import Expansion3D

   o = Expansion3D(tc=0.0, p=3.0, a=8.5, i=np.radians(89.0),
                   e=0.1, w=np.radians(90.0), derivatives=True)
   o.set_data(np.linspace(-0.05, 0.05, 1001))

   x, y, z, dx, dy, dz = o.position()      # values plus (N, 7) gradients
   d, dd = o.projected_separation()        # value plus (N, 7) gradient

The trailing gradient axis is the orbital block ``(tc, p, a, i, e, w, lan)``;
observables with extra physical inputs append their derivatives after that
block (see :ref:`expansion3d_derivative_mode` below).


Construction
------------

:class:`~meepmeep.expansion3d.Expansion3D` takes the same arguments as
:class:`~meepmeep.expansion2d.Expansion2D` — the six required orbital
elements plus ``lan``, ``te``, ``derivatives``, and ``parallel``:

=====================    ============================================================
Argument                 Meaning
=====================    ============================================================
``tc, p, a, i, e, w``    The orbital elements, forwarded straight to :meth:`~meepmeep.expansion3d.Expansion3D.set_pars`.
``lan``                  Longitude of the ascending node [rad], a constant rotation
                         of the sky plane about the line of sight (default 0.0). The
                         line-of-sight ``z`` is unaffected by it.
``te``                   Expansion-point time [days], measured relative to ``tc``.
                         ``te = 0`` (the default) expands the series at the transit
                         centre; use a non-zero offset to centre the expansion at,
                         e.g., the secondary eclipse. Fixed for the lifetime of the
                         instance — rebinding via
                         :meth:`~meepmeep.expansion3d.Expansion3D.set_pars` reuses it.
``derivatives``          If ``True``, every observable also returns analytic
                         parameter derivatives.
``parallel``             If ``True``, large time grids are routed to multi-threaded
                         kernels (see `Parallel evaluation`_).
=====================    ============================================================

:meth:`~meepmeep.expansion3d.Expansion3D.set_pars` accepts the orbital
elements as keyword-only arguments and re-solves the ``(3, 5)``
coefficient matrix (and, in derivative mode, the ``(7, 3, 5)`` derivative
tensor) in place. The time anchor is always ``tc`` (time of inferior
conjunction); there is no ``tp`` alternative, because a single expansion
point carries no periastron-anchored grid to convert to.

:meth:`~meepmeep.expansion3d.Expansion3D.set_data` binds the 1-D array of
absolute observation times evaluated by the observable methods; the
evaluators epoch-fold around the expansion point internally. Rebind the
grid as often as you like without recomputing the Taylor coefficients.


Observables
-----------

The class exposes a set of grid-evaluated quantities plus the same four
transit-geometry queries as :class:`~meepmeep.expansion2d.Expansion2D`.
Per-method detail (parameters, return shapes, units, edge cases) lives in
the docstrings, surfaced on the API page; this section is a tour.

**Geometry (methods, no extra inputs).** All are evaluated at the bound
times and, in units of the stellar radius:

- :meth:`~meepmeep.expansion3d.Expansion3D.position` — the 3D sky-frame
  :math:`(x, y, z)` position. ``x``, ``y`` span the sky plane; ``z`` is
  the line of sight, positive toward the observer (transit at
  :math:`z > 0`, secondary eclipse at :math:`z < 0`).
- :meth:`~meepmeep.expansion3d.Expansion3D.z_position` — the
  line-of-sight :math:`z` coordinate alone.
- :meth:`~meepmeep.expansion3d.Expansion3D.projected_separation` — the
  sky-projected separation :math:`d = \sqrt{x^2 + y^2}`, the quantity a
  transit light-curve model consumes directly.
- :meth:`~meepmeep.expansion3d.Expansion3D.velocity` — the
  :math:`(v_x, v_y, v_z)` velocity vector [R_star/day].
- :meth:`~meepmeep.expansion3d.Expansion3D.z_velocity` — the
  line-of-sight velocity component alone.
- :meth:`~meepmeep.expansion3d.Expansion3D.cos_phase` — the cosine of the
  orbital phase angle, :math:`\cos\alpha = -z/r`, in :math:`[-1, 1]`.

**Observables with physical inputs (methods).** These take a few extra
arguments beyond the bound orbit:

- :meth:`~meepmeep.expansion3d.Expansion3D.radial_velocity` ``(k)`` —
  stellar radial velocity, with ``k`` the RV semi-amplitude; the output
  inherits ``k``'s units.
- :meth:`~meepmeep.expansion3d.Expansion3D.lambert_phase_curve`
  ``(ag, k)`` — reflected-light phase curve for geometric albedo ``ag``
  and radius ratio ``k``.
- :meth:`~meepmeep.expansion3d.Expansion3D.ellipsoidal_variation`
  ``(alpha, mass_ratio)`` — ellipsoidal-variation signal; the orbital
  inclination is taken from the bound parameters automatically.
- :meth:`~meepmeep.expansion3d.Expansion3D.emission_phase_curve`
  ``(k, fratio, offset)`` — thermal-emission phase curve from a simple
  cosine model.

**Transit geometry (methods).** Identical to
:class:`~meepmeep.expansion2d.Expansion2D`: these take a planet-to-star
radius ratio ``k`` and operate on the coefficient matrix directly, so
they return absolute times in days —
:meth:`~meepmeep.expansion3d.Expansion3D.duration` (``kind`` in
``{14, 23, 12, 34}``),
:meth:`~meepmeep.expansion3d.Expansion3D.contact_point`,
:meth:`~meepmeep.expansion3d.Expansion3D.bounding_box`, and
:meth:`~meepmeep.expansion3d.Expansion3D.min_separation`.


.. _expansion3d_derivative_mode:

Derivative mode
---------------

Construct with ``derivatives=True`` to switch every grid observable into
gradient-returning form. The geometry methods append a single gradient
per value (e.g.
:meth:`~meepmeep.expansion3d.Expansion3D.projected_separation` returns
``(d, dd)`` with ``dd`` of shape ``(N, 7)``;
:meth:`~meepmeep.expansion3d.Expansion3D.position` returns
``(xs, ys, zs, dxs, dys, dzs)``). The trailing axis is always the orbital
block ``(tc, p, a, i, e, w, lan)``.

The observables with physical inputs append their extra derivatives
*after* the orbital block, in argument order:

- The pure-geometry methods (``position``, ``z_position``,
  ``projected_separation``, ``velocity``, ``z_velocity``, ``cos_phase``)
  and ``radial_velocity(k)`` return gradients of width 7, the orbital
  block ``(tc, p, a, i, e, w, lan)``.
- ``lambert_phase_curve(ag, k)`` returns width 9: ``(..., ag, k)``.
- ``ellipsoidal_variation(alpha, mass_ratio)`` returns width 9:
  ``(..., alpha, mass_ratio)``.
- ``emission_phase_curve(k, fratio, offset)`` returns width 10:
  ``(..., k, fratio, offset)``.

.. note::

   The single-expansion-point ``radial_velocity`` gradient has width 7,
   not 8: ``k`` enters as a pure linear scale, so it carries no separate
   derivative column here. (The whole-orbit
   :meth:`~meepmeep.orbit.Orbit.radial_velocity` does append a ``k``
   column, giving width 8 — a deliberate difference between the two
   entry points.)

For the gradient math (Kepler implicit-differentiation step,
orbital-plane derivative chain, evaluator-level chain rules) see
:doc:`derivatives`.


Parallel evaluation
-------------------

Passing ``parallel=True`` at construction routes sufficiently large time
grids to multi-threaded ``prange`` kernel twins, with the same
size-dependent thresholds as :class:`~meepmeep.expansion2d.Expansion2D`:
derivative-mode grids switch over at ``_PARALLEL_NMIN_GRAD`` (10 000
samples) and value-only grids at ``_PARALLEL_NMIN_VALUE`` (100 000
samples). Smaller grids always take the serial kernels; the results are
identical either way. (The value-only radial velocity has no parallel
kernel and always runs serially.)

Leave ``parallel`` off when the surrounding application already
parallelises at the process level (e.g. one process per MCMC chain),
where nested thread pools would oversubscribe the machine.


Relationship to the other classes
----------------------------------

All three high-level classes share the same construct-once,
update-in-a-loop workflow and the same ``set_pars`` / ``set_data``
rhythm, so moving between them is mechanical. The differences are scope:

==========================  ====================  ====================  ==============================
Aspect                      ``Expansion2D``       ``Expansion3D``       ``Orbit``
==========================  ====================  ====================  ==============================
Expansion points            One (single phase)    One (single phase)    Grid of ``npt`` (whole orbit)
Dimensionality              2D sky plane          3D                    3D
Quantities                  position,             position, velocity,   position, velocity,
                            separation,           separation, phase,    separation, phase, RV,
                            transit geometry      RV, phase curves,     phase curves, LTT
                                                  transit geometry
Time anchor                 ``tc`` only           ``tc`` only           ``tc`` or ``tp``
Accurate window             Near the              Near the              The full orbit
                            expansion point       expansion point
==========================  ====================  ====================  ==============================

Use ``Expansion2D`` for pure transit- or eclipse-light-curve work where
the 2D geometry is all the model consumes; use ``Expansion3D`` when you
need the line-of-sight coordinate or the dynamical and photometric
quantities *within* a single event window; use
:class:`~meepmeep.orbit.Orbit` whenever a quantity must be correct across
the whole orbit. All three rest on the same Taylor backend; see
:ref:`taylor_single_ep` for the single-expansion-point math underneath
this class and :ref:`taylor_two_modes` for the two backend modes in
general.
