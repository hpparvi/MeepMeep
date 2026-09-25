.. _orbit_overview:

Orbit class overview
====================

:class:`~meepmeep.orbit.Orbit` is the front door to MeepMeep. It is
built for the workflow that dominates exoplanet light-curve and
radial-velocity work: instantiate the class once, bind the time array
of your observations with :meth:`~meepmeep.orbit.Orbit.set_data`, and
then — inside a modelling or fitting loop —
just update the orbital parameters and read out the quantities you
need. The expensive per-orbit setup happens once; each iteration costs
one small Taylor solve plus the requested observable.

The available observables are planet position, velocity, 3D star-planet
distance, phase angle, radial velocity, reflected-light and thermal
phase curves, ellipsoidal variation, and light travel time. The class has no
sky-projected separation method; at any orbital phase that comes from the
low-level :func:`~meepmeep.numba3d.sep_o` / :func:`~meepmeep.numba3d.sep_od`
(near an event, :class:`~meepmeep.expansion2d.Expansion2D` provides it). With
``derivatives=True`` every observable except
:meth:`~meepmeep.orbit.Orbit.mean_anomaly` (a closed-form helper) also
returns analytic gradients w.r.t. the seven orbital parameters and any
method-specific extras, which is what gradient-based optimisers and HMC
samplers want.

.. contents::
   :local:
   :depth: 2


Quickstart
----------

.. code-block:: python

   import numpy as np
   from meepmeep import Orbit

   o = Orbit(npt=15)
   o.set_pars(tc=0.0, p=3.0, a=8.5, i=np.radians(89.0),
              e=0.1, w=np.radians(90.0))
   o.set_data(np.linspace(-0.05, 0.05, 1001))

   x, y, z = o.xyz()                       # planet position
   d = o.star_planet_distance()            # 3D separation

For analytic gradients in addition to values, switch to derivative mode
at construction time and unpack the extra arrays:

.. code-block:: python

   from meepmeep import Orbit

   o = Orbit(npt=15, derivatives=True)
   o.set_pars(tc=0.0, p=3.0, a=8.5, i=np.radians(89.0),
              e=0.1, w=np.radians(90.0))
   o.set_data(np.linspace(-0.05, 0.05, 1001))

   x, y, z, dx, dy, dz = o.xyz()           # values plus (N, 7) gradients

The shape contract is the same for every observable — see
:ref:`orbit_derivative_mode` below.


Typical modelling loop
----------------------

The class is designed so that the same instance can be reused across
many parameter updates without rebuilding internal state from scratch.
A transit-fitting or RV-fitting loop typically looks like this:

.. code-block:: python

   import numpy as np
   from meepmeep import Orbit

   # Your data: times, rv_obs, rv_err (1-D float64 arrays).
   # One-time setup: the instance and the observation times.
   o = Orbit(npt=15)
   o.set_data(times)                                  # observation times

   def log_likelihood(theta):
       tc, p, a, i, e, w, k = theta
       o.set_pars(tc=tc, p=p, a=a, i=i, e=e, w=w)     # update the orbit
       rv_model = o.radial_velocity(k=k)              # read out observable
       return -0.5 * np.sum(((rv_obs - rv_model) / rv_err)**2)

   # ... drop log_likelihood into your sampler / optimiser ...

Each numba kernel is compiled on its first use in a process (per
observable, and separately for value, gradient and parallel mode), whichever
instance makes the call; after that, every iteration is a fast Taylor solve
followed by a Horner-scheme evaluation per requested observable.


Construction
------------

:class:`~meepmeep.orbit.Orbit` is constructed with four optional
arguments:

==================  ============================================================
Argument            Meaning
==================  ============================================================
``npt``             Number of expansion points used by the multi-expansion-point Taylor expansion
                    (default 15). Includes the periodic-image slot. Raise this
                    when the orbit is eccentric enough that 15 expansion points no longer
                    cover periastron with adequate per-expansion-point accuracy.
``ep_placement``    Expansion-point placement strategy: ``'mm'`` (uniform in mean motion),
                    ``'ea'`` (uniform in eccentric anomaly; default and
                    preferred for eccentric orbits), or ``'ta'`` (uniform in
                    true anomaly). See :ref:`taylor_multi_ep` for the
                    placement strategies' implications.
``derivatives``     If ``True``, every observable also returns analytic
                    parameter derivatives. Costs roughly 3–14× the value-only
                    runtime, depending on the observable: the scalar ones
                    (distance, phase angle, RV, phase curves) sit at 3–6×, the
                    vector-valued ``xyz`` / ``vxyz`` near the top (serial
                    kernels, 1e5 times).
``parallel``        If ``True``, time arrays of at least 1e4 samples (gradient
                    mode) or 5e4 (value mode) run on multi-threaded
                    ``prange`` kernels; smaller arrays use the serial kernels,
                    which are faster there. Results are identical. Default
                    ``False``: leave it off when the application already
                    parallelises at a higher level (e.g. one process per MCMC
                    chain), where nested thread pools oversubscribe the machine.
==================  ============================================================

The expansion-point grid depends on the eccentricity for the ``'ea'`` and
``'ta'`` placements. The constructor builds it for ``e = 0.2``, and
:meth:`~meepmeep.orbit.Orbit.set_pars` rebuilds it for ``max(e, 0.2)``
whenever that differs by more than 0.05 from the eccentricity the current
grid was built for (a scipy root solve, ~0.3 ms); otherwise the grid is
reused. Every eccentricity below 0.2 shares one grid, and the ``'mm'``
grid never changes. A rebuild moves the expansion points, so the Taylor
approximation, and with it the returned value, jumps by the truncation
error; keep this in mind when finite-differencing in ``e``.


Binding orbital elements
------------------------

:meth:`~meepmeep.orbit.Orbit.set_pars` accepts the orbital elements as
keyword-only arguments. The time anchor is specified by exactly one of
``tc`` (transit center) or ``tp`` (periastron passage); the two are
related by

.. math::

   t_p \;=\; t_c \;-\; \frac{M_\mathrm{tr}(e, w)}{2\pi}\,p,

where :math:`M_\mathrm{tr}` is the mean anomaly at transit. Whichever
you pass, the other is derived and stored, so both anchors are
available afterwards regardless of which form you used.

.. code-block:: python

   # Same orbit, two equivalent calls:
   o.set_pars(tc=0.0,        p=3.0, a=8.5, i=np.radians(89.0), e=0.1, w=np.radians(30.0))
   # ... or ...
   o.set_pars(tp=-0.4204009, p=3.0, a=8.5, i=np.radians(89.0), e=0.1, w=np.radians(30.0))

   # After either call (to the 7 digits tp is given to):
   o._tc   # 0.0
   o._tp   # -0.4204009...

(With ``w = 90`` degrees the transit falls on periastron, :math:`M_\mathrm{tr} = 0`,
and the two anchors coincide.)

Passing both ``tc`` and ``tp``, or neither, raises ``TypeError`` — the
call site is always explicit about which convention it uses. The
remaining elements (``p, a, i, e, w``, and the optional ``lan``, the
longitude of the ascending node, default 0.0) are keyword-only too; this
keeps the call sites self-documenting and matches the rest of the package.
``lan`` is not remembered between calls: omitting it resets it to 0.


Binding times
-------------

:meth:`~meepmeep.orbit.Orbit.set_data` binds a 1-D time array to the
instance. The bound grid is used as the default ``times`` argument by
every evaluator that exposes a ``times=None`` parameter
(:meth:`~meepmeep.orbit.Orbit.xyz`,
:meth:`~meepmeep.orbit.Orbit.star_planet_distance`,
:meth:`~meepmeep.orbit.Orbit.lambert_phase_curve`,
:meth:`~meepmeep.orbit.Orbit.emission_phase_curve`,
:meth:`~meepmeep.orbit.Orbit.ellipsoidal_variation`), and used
unconditionally by methods that do not expose ``times`` at all
(:meth:`~meepmeep.orbit.Orbit.mean_anomaly`,
:meth:`~meepmeep.orbit.Orbit.true_anomaly`,
:meth:`~meepmeep.orbit.Orbit.vxyz`,
:meth:`~meepmeep.orbit.Orbit.cos_phase`,
:meth:`~meepmeep.orbit.Orbit.phase`,
:meth:`~meepmeep.orbit.Orbit.theta`,
:meth:`~meepmeep.orbit.Orbit.radial_velocity`,
:meth:`~meepmeep.orbit.Orbit.light_travel_time`).

You can rebind the grid as often as you like without recomputing the
Taylor coefficients — :meth:`~meepmeep.orbit.Orbit.set_data` only
stores the array.

Inputs, units and conventions
-----------------------------

**Call order and input types.** Call
:meth:`~meepmeep.orbit.Orbit.set_pars` and
:meth:`~meepmeep.orbit.Orbit.set_data` before the first observable; the
class does not guard against a missing call, and the resulting error is
an unhelpful ``AttributeError`` or numba ``TypingError``. Times are a 1-D
NumPy array or a scalar, which gives scalar outputs. Python lists are not
converted and fail inside numba with a ``TypingError``; wrap them in
``np.asarray``. The bound array is the public attribute ``times``, and
``npt`` holds the number of expansion points. The two time anchors are
stored as ``_tc`` and ``_tp`` (see `Convention bridge`_).

**Coordinates.** ``x`` points right and ``y`` up in the sky plane, and
``z`` along the line of sight, positive toward the observer: the transit
happens at ``z > 0`` and the secondary eclipse at ``z < 0``. The inclination
is the angle between the orbit normal and the line of sight, so
``i = pi/2`` is edge-on. Angles
are in radians and times in days.

**Units of the outputs.**

====================================================  ==========================================
Method                                                Unit
====================================================  ==========================================
:meth:`~meepmeep.orbit.Orbit.xyz`,                    stellar radii
:meth:`~meepmeep.orbit.Orbit.star_planet_distance`
:meth:`~meepmeep.orbit.Orbit.vxyz`                    stellar radii per day
:meth:`~meepmeep.orbit.Orbit.radial_velocity`         the units of ``k``
:meth:`~meepmeep.orbit.Orbit.light_travel_time`       days (``rstar`` in solar radii)
:meth:`~meepmeep.orbit.Orbit.lambert_phase_curve`,    planet-to-star flux ratio
:meth:`~meepmeep.orbit.Orbit.emission_phase_curve`
:meth:`~meepmeep.orbit.Orbit.ellipsoidal_variation`   relative flux variation
====================================================  ==========================================

**Physical inputs.** The phase-curve and variation methods take:

- :meth:`~meepmeep.orbit.Orbit.lambert_phase_curve` ``(k, ag)``: the
  radius ratio :math:`R_p/R_\star` and the geometric albedo. The gradient
  appends them in the order ``(ag, k)``, and
  :meth:`meepmeep.expansion3d.Expansion3D.lambert_phase_curve` takes them
  as ``(ag, k)`` too. Pass them by keyword when switching between the two
  classes.
- :meth:`~meepmeep.orbit.Orbit.emission_phase_curve`
  ``(k, fratio, offset)``: the radius ratio, the dayside-to-nightside flux
  ratio (the peak-to-peak swing is :math:`k^2 f_\mathrm{ratio}`), and the
  hotspot offset in radians.
- :meth:`~meepmeep.orbit.Orbit.ellipsoidal_variation`
  ``(alpha, mass_ratio)``: the gravity-darkening coefficient and the
  planet-to-star mass ratio :math:`M_p/M_\star`; the inclination comes
  from the bound parameters.
- :meth:`~meepmeep.orbit.Orbit.light_travel_time` ``(rstar)``: the stellar
  radius in solar radii, which sets the light-crossing scale; its
  derivative is not returned.


Observables
-----------

The class groups its observables into a small set of conceptual
families. Per-method detail (parameters, return shapes, units, edge
cases) lives in the docstrings, surfaced on the API page; this section
is a tour.

**Geometry.** :meth:`~meepmeep.orbit.Orbit.xyz` returns the planet
position in the sky frame; :meth:`~meepmeep.orbit.Orbit.vxyz` returns
the velocity; :meth:`~meepmeep.orbit.Orbit.star_planet_distance`
returns the 3D separation
:math:`r = \sqrt{x^2 + y^2 + z^2}`.
:meth:`~meepmeep.orbit.Orbit.cos_phase` returns
:math:`\cos\alpha = -z/r` (the cosine of the star-planet-observer
phase angle, with the planet at the vertex), with :meth:`~meepmeep.orbit.Orbit.phase` and
:meth:`~meepmeep.orbit.Orbit.theta` returning :math:`\alpha` and
:math:`\pi-\alpha` respectively. The latter two clamp slightly inside
the :math:`\arccos` domain in derivative mode so the gradient stays
finite at the conjunction extrema; see the
:meth:`~meepmeep.orbit.Orbit.phase` docstring for the caveat.

**Photometry.** :meth:`~meepmeep.orbit.Orbit.lambert_phase_curve`
evaluates the reflected-light phase curve assuming Lambertian
scattering. :meth:`~meepmeep.orbit.Orbit.emission_phase_curve`
evaluates a simple cosine thermal-emission phase curve with a hotspot
offset (amplitude :math:`k^2 f_\mathrm{ratio}`).
:meth:`~meepmeep.orbit.Orbit.ellipsoidal_variation` evaluates the
stellar ellipsoidal-distortion signal of Lillo-Box et al. (2014).

**Spectroscopy.** :meth:`~meepmeep.orbit.Orbit.radial_velocity` scales
the line-of-sight planet velocity by the closed-form factor
:math:`K / [(2\pi/p)\,(a\sin i)/\sqrt{1-e^2}]` so the result is the
stellar RV signal in m/s.

**Timing.** :meth:`~meepmeep.orbit.Orbit.mean_anomaly` and
:meth:`~meepmeep.orbit.Orbit.true_anomaly` return the corresponding
anomalies at the bound times. ``true_anomaly(exact=True)`` switches to
the Newton-Raphson reference solver for validation runs; the exact
path does not provide parameter derivatives, so it raises
``NotImplementedError`` if combined with ``derivatives=True``.
:meth:`~meepmeep.orbit.Orbit.light_travel_time` returns the LTT
correction referenced to primary transit, with the ``rstar`` derivative
intentionally omitted from the gradient.

**Diagnostics.** :meth:`~meepmeep.orbit.Orbit.plot` renders a
three-panel sky-frame plot of the orbit, with an optional
``show_exact=True`` overlay against the Newton-Raphson reference. The
private helpers ``_xyz_error`` and ``_cos_phase_error`` are available
for quick sanity checks against the same reference.


.. _orbit_derivative_mode:

Derivative mode
---------------

Construct with ``derivatives=True`` to switch every observable into
gradient-returning form. The shape contract is uniform:

- Multi-coordinate value returns are extended with one gradient array
  per coordinate. For example :meth:`~meepmeep.orbit.Orbit.xyz` returns
  ``(xs, ys, zs, dxs, dys, dzs)`` instead of ``(xs, ys, zs)``, with
  each ``d*s`` of shape ``(N, ndp)``.

- Single-value returns become two-tuples ``(value, dvalue)`` with
  ``dvalue`` of shape ``(N, ndp)``.

The trailing axis ``ndp`` is the number of differentiable parameters.
The first seven slots are always the orbital block
``(tc, p, a, i, e, w, lan)`` — with ``tp`` replacing ``tc`` when the orbit
is bound by periastron time (see `Convention bridge`_ below); some methods
append physical extras:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Trailing extras (in order)
   * - :meth:`~meepmeep.orbit.Orbit.radial_velocity`
     - ``k``
   * - :meth:`~meepmeep.orbit.Orbit.lambert_phase_curve`
     - ``ag``, ``k``
   * - :meth:`~meepmeep.orbit.Orbit.emission_phase_curve`
     - ``k``, ``fratio``, ``offset``
   * - :meth:`~meepmeep.orbit.Orbit.ellipsoidal_variation`
     - ``alpha``, ``mass_ratio`` (the signal's direct dependence on the
       inclination is added into the ``i`` slot)

The ``rstar`` derivative of
:meth:`~meepmeep.orbit.Orbit.light_travel_time` is intentionally not
returned — only the 7 orbital derivatives.

For the gradient math (Kepler implicit-differentiation step,
orbital-plane derivative chain, evaluator-level chain rules) see
:doc:`derivatives`.


Convention bridge
-----------------

This class accepts either ``tc`` (transit-center time) or ``tp``
(periastron-passage time) via :meth:`~meepmeep.orbit.Orbit.set_pars`.
The underlying Taylor backend anchors its expansion-point grid at periastron, so
:meth:`~meepmeep.orbit.Orbit.set_pars` converts once and stores both
values internally: ``self._tc`` and ``self._tp``. Every Taylor-backend
dispatcher call uses ``self._tp``; ``self._tc`` is used only by
:meth:`~meepmeep.orbit.Orbit.mean_anomaly`, the Newton-Raphson diagnostic
paths, and :meth:`~meepmeep.orbit.Orbit.plot` with ``show_exact=True``.

If you ever drop down to the Taylor backend directly, note that its
multi-expansion-point dispatchers
(:func:`~meepmeep.backends.numba.orbit3d.pos_o` and friends)
take a ``tpa`` argument that is the periastron-anchored time — not the
transit-center time. See :ref:`taylor_overview` for the low-level
convention and :ref:`taylor_two_modes` for when to drop down at all.
