.. _expansion2d_overview:

Expansion2D class overview
==========================

:class:`~meepmeep.expansion2d.Expansion2D` is the lightweight front
door for transit geometry. Where :class:`~meepmeep.orbit.Orbit` builds a
grid of expansion points spanning the whole orbit in 3D,
``Expansion2D`` builds a *single* 5th-order Taylor expansion of the
sky-plane trajectory at one chosen phase — typically the transit or eclipse
centre. That is exactly enough for a transit light-curve model, which
only ever needs the planet's position and its sky-projected separation
from the star in the narrow time window around conjunction.

The trade is scope for simplicity. A single expansion point is accurate
only in the time window the series covers (a transit, an eclipse, a
fixed-phase snapshot), and the class exposes only 2D quantities: the
sky-plane :math:`(x, y)` position and the projected separation. There is
no :math:`z`, no velocity, no radial velocity, and no phase curve — for
any of those, reach for :class:`~meepmeep.orbit.Orbit`. In return the
setup is a single ``(2, 5)`` Taylor solve, the time anchor is plain
``tc`` (no periastron bookkeeping), and contact points and durations
fall straight out of the same coefficient matrix.

With ``derivatives=True`` the position and separation also return
analytic gradients w.r.t. the seven orbital parameters, which is what
gradient-based optimisers and HMC samplers want.

.. contents::
   :local:
   :depth: 2


Quickstart
----------

.. code-block:: python

   import numpy as np
   from meepmeep.expansion2d import Expansion2D

   o = Expansion2D(tc=0.0, p=3.0, a=8.5, i=np.radians(89.0),
                   e=0.1, w=np.radians(90.0))
   o.set_data(np.linspace(-0.05, 0.05, 1001))

   x, y = o.position()                # sky-plane position per time
   d = o.projected_separation()       # sky-projected separation

Note that ``position`` and ``projected_separation`` are **methods**:
calling them evaluates the bound time grid.

For analytic gradients in addition to values, switch to derivative mode
at construction time and unpack the extra arrays:

.. code-block:: python

   from meepmeep.expansion2d import Expansion2D

   o = Expansion2D(tc=0.0, p=3.0, a=8.5, i=np.radians(89.0),
                   e=0.1, w=np.radians(90.0), derivatives=True)
   o.set_data(np.linspace(-0.05, 0.05, 1001))

   x, y, dx, dy = o.position()        # values plus (N, 7) gradients
   d, dd = o.projected_separation()   # value plus (N, 7) gradient

The trailing gradient axis is always the orbital block
``(tc, p, a, i, e, w, lan)``; see :ref:`expansion2d_derivative_mode`
below.


Typical modelling loop
----------------------

Like :class:`~meepmeep.orbit.Orbit`, the same instance is reused across
many parameter updates without rebuilding internal state. A
transit-fitting loop typically looks like this:

.. code-block:: python

   import numpy as np
   from meepmeep.expansion2d import Expansion2D

   # One-time setup: the observation times.
   o = Expansion2D(tc=0.0, p=3.0, a=8.5, i=np.radians(89.0),
                   e=0.1, w=np.radians(90.0))
   o.set_data(times)                                  # observation times

   def log_likelihood(theta):
       tc, p, a, i, e, w = theta
       o.set_pars(tc=tc, p=p, a=a, i=i, e=e, w=w)     # update the orbit
       z = o.projected_separation()                   # read out separation
       flux = transit_model(z, k, ldc)                # your limb-darkened model
       return -0.5 * np.sum(((flux_obs - flux) / flux_err)**2)

   # ... drop log_likelihood into your sampler / optimiser ...

Only the very first call after construction pays the numba JIT-compile
cost; every subsequent iteration is one small Taylor solve followed by a
Horner-scheme evaluation of the requested quantity.


Construction
------------

:class:`~meepmeep.expansion2d.Expansion2D` is constructed with the six
required orbital elements plus four optional arguments:

=====================    ============================================================
Argument                 Meaning
=====================    ============================================================
``tc, p, a, i, e, w``    The orbital elements, forwarded straight to :meth:`~meepmeep.expansion2d.Expansion2D.set_pars`.
``lan``                  Longitude of the ascending node [rad], a constant rotation
                         of the sky plane about the line of sight (default 0.0).
``te``                   Expansion-point time [days], measured relative to ``tc``.
                         ``te = 0`` (the default) expands the series at the transit
                         centre; use a non-zero offset to centre the expansion at,
                         e.g., the secondary eclipse. The expansion-point time is
                         fixed for the lifetime of the instance — rebinding via
                         :meth:`~meepmeep.expansion2d.Expansion2D.set_pars` reuses it.
``derivatives``          If ``True``, the position and separation also return
                         analytic parameter derivatives.
``parallel``             If ``True``, large time grids are routed to multi-threaded
                         kernels (see `Parallel evaluation`_).
=====================    ============================================================

Binding orbital elements
------------------------

:meth:`~meepmeep.expansion2d.Expansion2D.set_pars` accepts the orbital
elements as keyword-only arguments and re-solves the ``(2, 5)``
coefficient matrix in place. The time anchor is always ``tc`` (time of
inferior conjunction) — there is no ``tp`` alternative, because a single
expansion point carries no periastron-anchored grid to convert to.

.. code-block:: python

   o.set_pars(tc=0.0, p=3.0, a=8.5, i=np.radians(89.0), e=0.1, w=np.radians(90.0))

The expansion-point time ``te`` is a construction-time constant and is
reused on every call; the constructor itself simply forwards its orbital
arguments to this method.


Binding times
-------------

:meth:`~meepmeep.expansion2d.Expansion2D.set_data` binds a 1-D time
array to the instance. The bound grid is the one evaluated by the
:meth:`~meepmeep.expansion2d.Expansion2D.position` and
:meth:`~meepmeep.expansion2d.Expansion2D.projected_separation`
methods. Both expect absolute observation times in days; the
evaluators epoch-fold around the expansion point internally.

You can rebind the grid as often as you like without recomputing the
Taylor coefficients — :meth:`~meepmeep.expansion2d.Expansion2D.set_data`
only stores the array.


Observables
-----------

The class exposes two grid-evaluated quantities and four transit-geometry
queries. Per-method detail (parameters, return shapes, units, edge cases)
lives in the docstrings, surfaced on the API page; this section is a tour.

**Grid quantities (methods).**
:meth:`~meepmeep.expansion2d.Expansion2D.position` returns the sky-plane
:math:`(x, y)` position at the bound times, in units of the stellar
radius. :meth:`~meepmeep.expansion2d.Expansion2D.projected_separation`
returns the sky-projected separation between the centers of the star and
planet, :math:`d = \sqrt{x^2 + y^2}`, in the same units — the quantity a
transit light-curve model consumes directly.

**Transit geometry (methods).** These take a planet-to-star radius ratio
``k`` and operate on the coefficient matrix directly, so they return
absolute times in days:

- :meth:`~meepmeep.expansion2d.Expansion2D.duration` returns the transit
  duration. ``kind`` selects which: 14 (total, first-to-fourth contact;
  the default), 23 (full), 12 (ingress), or 34 (egress).
- :meth:`~meepmeep.expansion2d.Expansion2D.contact_point` returns the
  absolute time of contact point 1, 2, 3, or 4.
- :meth:`~meepmeep.expansion2d.Expansion2D.bounding_box` returns the
  first- and fourth-contact times ``(T1, T4)`` bracketing the transit.
- :meth:`~meepmeep.expansion2d.Expansion2D.min_separation` locates the
  minimum projected separation near the expansion point, returning
  ``(t_min, z_min)`` — useful for the impact parameter and the time of
  minimum projected separation.


.. _expansion2d_derivative_mode:

Derivative mode
---------------

Construct with ``derivatives=True`` to switch both grid quantities into
gradient-returning form:

- :meth:`~meepmeep.expansion2d.Expansion2D.position` returns
  ``(xs, ys, dxs, dys)`` instead of ``(xs, ys)``, with each ``d*s`` of
  shape ``(N, 7)``.
- :meth:`~meepmeep.expansion2d.Expansion2D.projected_separation` returns
  ``(d, dd)`` instead of ``d``, with ``dd`` of shape ``(N, 7)``.

The trailing axis of every gradient is the orbital block
``(tc, p, a, i, e, w, lan)``, in that order. There are no method-specific
extras here — unlike the photometry and RV observables of
:class:`~meepmeep.orbit.Orbit`, the 2D position and separation depend
only on the orbital elements.

For the gradient math (Kepler implicit-differentiation step,
orbital-plane derivative chain, evaluator-level chain rules) see
:doc:`derivatives`.


Parallel evaluation
-------------------

Passing ``parallel=True`` at construction routes sufficiently large time
grids to multi-threaded ``prange`` kernel twins. The threshold is size
dependent: derivative-mode grids switch over at
``_PARALLEL_NMIN_GRAD`` (10 000 samples) and value-only grids at the
higher ``_PARALLEL_NMIN_VALUE`` (100 000 samples), because the value
kernels run at only a few nanoseconds per sample and so have to amortise
the parallel-region launch over more work. Smaller grids always take the
serial kernels; the results are identical either way.

Leave ``parallel`` off when the surrounding application already
parallelises at the process level (e.g. one process per MCMC chain),
where nested thread pools would oversubscribe the machine.


Relationship to the Orbit class
-------------------------------

:class:`~meepmeep.expansion2d.Expansion2D` and
:class:`~meepmeep.orbit.Orbit` share the same construct-once,
update-in-a-loop workflow and the same ``set_pars`` / ``set_data``
rhythm, so moving between them is mechanical. The differences are scope:

==========================  ==============================  ==============================
Aspect                      ``Expansion2D``                 ``Orbit``
==========================  ==============================  ==============================
Expansion points            One (single phase)              Grid of ``npt`` (whole orbit)
Dimensionality              2D sky plane                    3D
Quantities                  position, separation,           position, velocity, separation,
                            transit geometry                phase, RV, phase curves, LTT
Time anchor                 ``tc`` only                     ``tc`` or ``tp``
Accurate window             Near the expansion point        The full orbit
==========================  ==============================  ==============================

Use ``Expansion2D`` for transit- or eclipse-only light-curve work, where
a single expansion covers the event and the 2D geometry is all the model
consumes. Use :class:`~meepmeep.orbit.Orbit` whenever you need
whole-orbit quantities or the line-of-sight coordinate. Both rest on the
same Taylor backend; see :ref:`taylor_single_ep` for the single-expansion-point
math underneath this class and :ref:`taylor_two_modes` for the two
backend modes in general.
