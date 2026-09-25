.. _api.lowlevel:

Low-level Taylor backend API
============================

The reference catalogue for the low-level Taylor-backend routines. Use
this page when you have dropped below the
:class:`~meepmeep.orbit.Orbit` class and need to know the exact
function names, signatures, and per-function semantics. See
:doc:`../taylor_overview` for the conceptual model that ties these
routines together and :doc:`../naming_conventions` for the suffix
grammar that explains how the names are constructed.

Every function listed here operates on NumPy arrays and can be called
both from plain Python and from inside a user :func:`numba.njit` kernel.
The solvers, the geometric utilities and ``rv``/``rv_c`` are jitted
functions. The evaluators (``pos``, ``sep_d``, ``pos_o``, ...) are plain
Python dispatchers registered with :func:`numba.extending.overload`: in
Python they pick the scalar or vector kernel at call time, inside
``@njit`` at compile time. :func:`~meepmeep.numba3d.create_expansion_points`
is plain Python (it uses scipy) and is not callable from ``@njit``.
Per-function detail (parameters, shapes, units,
mathematical notes) lives in the source docstrings and is rendered
below via ``autosummary``.


Coefficient solvers
-------------------

These functions take Keplerian orbital elements and return Taylor
coefficient matrices that the position and velocity evaluators consume.
Variants suffixed with ``_d`` additionally return the ``(7, D, 5)``
parameter-derivative tensor.

.. currentmodule:: meepmeep.numba2d

.. autosummary::
   :toctree: generated

   solve2d
   solve2d_d

.. currentmodule:: meepmeep.numba3d

.. autosummary::
   :toctree: generated

   solve3d
   solve3d_d


Two-dimensional position and separation
---------------------------------------

Single-expansion-point evaluators for the sky-plane :math:`(x, y)` position and the
projected separation :math:`d = \sqrt{x^2 + y^2}`. Each
function operates on one ``(2, 5)`` coefficient matrix from
:func:`~meepmeep.numba2d.solve2d` and is
sufficient for transit light-curve modelling. There is no 2D whole-orbit
evaluator; the whole-orbit dispatchers in
:ref:`api.lowlevel.orbit_dispatchers` batch the 3D evaluators below.

.. currentmodule:: meepmeep.numba2d

.. autosummary::
   :toctree: generated

   pos
   pos_c
   sep
   sep_c

Parameter-derivative variants:

.. currentmodule:: meepmeep.numba2d

.. autosummary::
   :toctree: generated

   pos_d
   pos_cd
   sep_d
   sep_cd


Three-dimensional position and separation
-----------------------------------------

Single-expansion-point evaluators that additionally return the line-of-sight
coordinate :math:`z`. Each function operates on one ``(3, 5)``
coefficient matrix from
:func:`~meepmeep.numba3d.solve3d`. Needed for
eclipses, light travel time, phase curves, and radial velocities. The
whole-orbit dispatchers that batch these calls across a expansion-point grid live
in :ref:`api.lowlevel.orbit_dispatchers`.

.. currentmodule:: meepmeep.numba3d

.. autosummary::
   :toctree: generated

   pos
   pos_c
   sep
   sep_c
   zpos
   zpos_c

Parameter-derivative variants:

.. currentmodule:: meepmeep.numba3d

.. autosummary::
   :toctree: generated

   pos_d
   pos_cd
   sep_d
   sep_cd
   zpos_d
   zpos_cd


Velocities
----------

Velocity-vector and line-of-sight velocity evaluators built on the same
coefficient matrices used by the position evaluators.

.. currentmodule:: meepmeep.numba3d

.. autosummary::
   :toctree: generated

   vel_c
   vel
   zvel_c
   zvel
   rv_c
   rv

Parameter-derivative variants:

.. currentmodule:: meepmeep.numba3d

.. autosummary::
   :toctree: generated

   vel_cd
   vel_d
   zvel_cd
   zvel_d
   rv_cd
   rv_d


Phase angle, reflected light, ellipsoidal variation, and emission
-----------------------------------------------------------------

Single-expansion-point cosine-of-phase-angle, Lambertian reflected-light
phase-curve, ellipsoidal-variation, and cosine thermal-emission evaluators,
built on the same ``(3, 5)`` coefficient matrices. The Lambert evaluators form
the flux :math:`(k/r)^2 A_g\, f(\alpha)` from the phase-angle cosine, using the
instantaneous star-planet distance :math:`r = \sqrt{x^2+y^2+z^2}` for the
inverse-square illumination (exact for eccentric orbits); the
ellipsoidal-variation signal scales as :math:`1/r^3`; the emission model is
:math:`k^2 f_\mathrm{ratio}\,(1 + \cos\delta\,c_z + \sin\delta\,s)/2` with a
signed in-plane component :math:`s` (from the orbital normal) and a hotspot
offset :math:`\delta`. The whole-orbit dispatchers in
:ref:`api.lowlevel.orbit_dispatchers` delegate to these.

.. currentmodule:: meepmeep.numba3d

.. autosummary::
   :toctree: generated

   cos_alpha_c
   cos_alpha
   lambert_phase_curve_c
   lambert_phase_curve
   ev_signal_c
   ev_signal
   emission_phase_curve_c
   emission_phase_curve

Parameter-derivative variants:

.. autosummary::
   :toctree: generated

   cos_alpha_cd
   cos_alpha_d
   lambert_phase_curve_cd
   lambert_phase_curve_d
   ev_signal_cd
   ev_signal_d
   emission_phase_curve_cd
   emission_phase_curve_d


Geometric utilities
-------------------

Contact points, durations, bounding boxes, and the minimum projected
separation, found numerically from a coefficient matrix (bisection for the
contact points, golden-section search for the minimum).

.. currentmodule:: meepmeep.numba2d

.. autosummary::
   :toctree: generated

   find_contact_point
   find_z_min
   bounding_box
   t1
   t4
   t12
   t14
   t23
   t34

The 3D module :mod:`meepmeep.numba3d` exposes the same set of helpers,
operating on ``(3, 5)`` coefficient matrices:

.. currentmodule:: meepmeep.numba3d

.. autosummary::
   :toctree: generated

   find_contact_point
   find_z_min
   bounding_box
   t1
   t4
   t12
   t14
   t23
   t34

None of these signal failure. For a geometry where the planet does not
transit, or only grazes the star, the contact-point search still returns
a number, and the durations built from it are meaningless (``t23`` of a
grazing transit can even come out longer than ``t14``). Check the impact
parameter first, for example with :func:`find_z_min`.


Vector and parallel kernels
---------------------------

Every evaluator on this page also has public vector kernels that skip the
scalar-or-array check: ``X_v`` / ``X_vp`` for the single-expansion-point
evaluators (serial / multi-threaded), and ``X_ov`` / ``X_ovp`` and
``X_ovd`` / ``X_ovdp`` for the whole-orbit value and gradient dispatchers.
They take the same arguments as the dispatcher, with a 1-D array of times
and every optional argument (such as ``te``) passed explicitly, and they are
not listed individually. :doc:`../naming_conventions` explains the suffixes
and when the parallel twins pay off.


.. _api.lowlevel.orbit_dispatchers:

Whole-orbit dispatchers (multi-expansion-point)
-----------------------------------------------

Whole-orbit evaluators that use a precomputed time-to-expansion-point table
(``ep_table``) to dispatch each input time to the appropriate expansion point and
delegate to the centered single-expansion-point evaluators above. Each ``_o``
name is a single overloaded dispatcher that accepts either a scalar time or a
1-D float64 array of times, and ``_od`` is its gradient-returning
counterpart. Most take the time first, ``X_o(t, tpa, p, dt, ep_table,
ep_times, coeffs)``; four put extra inputs around it:
``ev_signal_o(alpha, mass_ratio, inc, t, ...)``,
``cos_v_p_angle_o(v, t, ...)``,
``true_anomaly_o(t, tpa, p, ex, ey, ez, w, ...)`` and
``light_travel_time_o(t, tpa, p, e, w, rstar, ...)``. The orbit-setup
routines listed first are ordinary jitted functions: :func:`~meepmeep.numba3d.solve3d_orbit` builds the coefficient
stack and :func:`~meepmeep.numba3d.ep_ix` maps one scalar time to its
expansion point.

.. currentmodule:: meepmeep.numba3d

Orbit setup:

.. autosummary::
   :toctree: generated

   solve3d_orbit
   ep_ix

Positions, separation and distance:

.. autosummary::
   :toctree: generated

   pos_o
   zpos_o
   sep_o
   star_planet_distance_o

Velocities and radial velocity:

.. autosummary::
   :toctree: generated

   vel_o
   zvel_o
   rv_o

Phase angles, Lambert curves, ellipsoidal variation, and emission:

.. autosummary::
   :toctree: generated

   true_anomaly_o
   cos_alpha_o
   cos_v_p_angle_o
   lambert_phase_curve_o
   ev_signal_o
   emission_phase_curve_o

Light travel time:

.. autosummary::
   :toctree: generated

   light_travel_time_o


Whole-orbit dispatchers with parameter derivatives
--------------------------------------------------

Gradient-returning counterparts of the orbit dispatchers above. Each ``_od``
dispatcher accepts an additional ``dcoeffs`` tensor of shape
``(N, 7, 3, 5)`` (expansion point, parameter, dimension, Taylor order) produced by
:func:`~meepmeep.numba3d.solve3d_orbit_d`, which is listed first.

.. currentmodule:: meepmeep.numba3d

.. autosummary::
   :toctree: generated

   solve3d_orbit_d
   pos_od
   zpos_od
   sep_od
   vel_od
   zvel_od
   cos_alpha_od
   cos_v_p_angle_od
   true_anomaly_od
   star_planet_distance_od
   lambert_phase_curve_od
   ev_signal_od
   emission_phase_curve_od
   rv_od
   light_travel_time_od

Orbit geometry helpers: the mean anomaly at transit, which converts the
transit-centre time to the periastron anchor ``tpa`` the dispatchers take
(``tpa = tc - M_tr p / 2 pi``); the time of the secondary eclipse relative to
the transit, the ``te`` of an eclipse expansion; and the eccentricity vector
and its Jacobian, which :func:`~meepmeep.numba3d.true_anomaly_od` takes.

.. autosummary::
   :toctree: generated

   mean_anomaly_at_transit
   eclipse_time_offset
   eccentricity_vector
   eccentricity_vector_d

The solvers return gradients in the transit-centre basis
``(tc, p, a, i, e, w, lan)`` (single expansion point) or in the periastron
basis ``(tp, p, a, i, e, w, lan)`` (:func:`~meepmeep.numba3d.solve3d_orbit_d`).
These convert between the two:

.. autosummary::
   :toctree: generated

   tc_to_tp_gradient
   tp_to_tc_gradient
   tp_to_tc_gradient_orbit


Expansion point grid construction
---------------------------------

The expansion-point grid and the time-to-expansion-point table consumed by the multi-expansion-point
dispatchers are built once per orbit by
:func:`~meepmeep.numba3d.create_expansion_points`.

.. currentmodule:: meepmeep.numba3d

.. autosummary::
   :toctree: generated

   create_expansion_points
   expansion_table_size

The two anomaly helpers below are not part of the aggregator surface
but remain available at their source path.

.. currentmodule:: meepmeep.backends.numba.expansion_points

.. autosummary::
   :toctree: generated

   eccentric_anomaly
   true_anomaly
