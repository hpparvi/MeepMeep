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

Every function listed here is :func:`numba.njit`-compiled and operates
on NumPy arrays. Per-function detail (parameters, shapes, units,
mathematical notes) lives in the source docstrings and is rendered
below via ``autosummary``.


Coefficient solvers
-------------------

These functions take Keplerian orbital elements and return Taylor
coefficient matrices that the position and velocity evaluators consume.
Variants suffixed with ``_d`` additionally return the ``(6, D, 5)``
parameter-derivative tensor.

.. currentmodule:: meepmeep.backends.numba.taylor

.. autosummary::
   :toctree: generated

   solve2d.solve2d
   solve2dd.solve2d_d
   solve3d.solve3d
   solve3dd.solve3d_d


Two-dimensional position and distance
-------------------------------------

Single-knot evaluators for the sky-plane :math:`(x, y)` position and the
projected planet-star distance :math:`d = \sqrt{x^2 + y^2}`. Each
function operates on one ``(2, 5)`` coefficient matrix from
:func:`~meepmeep.backends.numba.taylor.solve2d.solve2d` and is
sufficient for transit light-curve modelling. The whole-orbit
dispatchers that batch these calls across a knot grid live in
:ref:`api.lowlevel.orbit_dispatchers`.

.. currentmodule:: meepmeep.backends.numba.taylor.position2d

.. autosummary::
   :toctree: generated

   pos
   pos_c
   sep
   sep_c
   pos_and_sep
   pos_and_sep_c

Parameter-derivative variants:

.. currentmodule:: meepmeep.backends.numba.taylor.position2dd

.. autosummary::
   :toctree: generated

   pos_d
   pos_cd
   sep_d
   sep_cd


Three-dimensional position and distance
---------------------------------------

Single-knot evaluators that additionally return the line-of-sight
coordinate :math:`z`. Each function operates on one ``(3, 5)``
coefficient matrix from
:func:`~meepmeep.backends.numba.taylor.solve3d.solve3d`. Needed for
eclipses, light travel time, phase curves, and radial velocities. The
whole-orbit dispatchers that batch these calls across a knot grid live
in :ref:`api.lowlevel.orbit_dispatchers`.

.. currentmodule:: meepmeep.backends.numba.taylor.position3d

.. autosummary::
   :toctree: generated

   pos
   pos_c
   sep
   sep_c
   pos_and_sep
   pos_and_sep_c
   pz
   pz_c

Parameter-derivative variants:

.. currentmodule:: meepmeep.backends.numba.taylor.position3dd

.. autosummary::
   :toctree: generated

   pos_d
   pos_cd
   sep_d
   sep_cd
   pz_d
   pz_cd


Velocities
----------

Velocity-vector and line-of-sight velocity evaluators built on the same
coefficient matrices used by the position evaluators.

.. currentmodule:: meepmeep.backends.numba.taylor.velocity3d

.. autosummary::
   :toctree: generated

   vel_c
   zvel_c
   zvel
   rv_c
   rv

Parameter-derivative variants:

.. currentmodule:: meepmeep.backends.numba.taylor.velocity3dd

.. autosummary::
   :toctree: generated

   vel_cd
   zvel_cd
   zvel_d
   rv_cd
   rv_d


Geometric utilities
-------------------

Contact points, durations, and bounding boxes derived analytically from a
coefficient matrix.

.. currentmodule:: meepmeep.backends.numba.taylor.util2d

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

The 3D module :mod:`meepmeep.backends.numba.taylor.util3d` exposes the
same set of helpers operating on ``(3, 5)`` coefficient matrices.


.. _api.lowlevel.orbit_dispatchers:

Whole-orbit dispatchers (multi-knot)
------------------------------------

Whole-orbit evaluators that use a precomputed time-to-knot table
(``pktable``) to dispatch each input time to the appropriate knot and
delegate to the centered single-knot evaluators above. The ``_o5s``
suffix denotes a scalar-time variant, ``_o5v`` an array-of-times
variant.

.. currentmodule:: meepmeep.backends.numba.taylor.orbit3d

Orbit setup:

.. autosummary::
   :toctree: generated

   solve3d_orbit
   knot_ix

Positions and distances:

.. autosummary::
   :toctree: generated

   pos_os
   pos_ov
   zpos_os
   zpos_ov
   sep_os
   star_planet_distance_ov

Velocities and radial velocity:

.. autosummary::
   :toctree: generated

   vel_os
   vel_ov
   zvel_os
   zvel_ov
   rv_ov

Phase angles, Lambert curves, and ellipsoidal variation:

.. autosummary::
   :toctree: generated

   true_anomaly_ov
   cos_alpha_os
   cos_alpha_ov
   cos_v_p_angle_ov
   lambert_phase_curve_os
   lambert_phase_curve_ov
   lambert_and_emission_ov
   ev_signal_ov

Light travel time:

.. autosummary::
   :toctree: generated

   light_travel_time_os
   light_travel_time_ov


Whole-orbit dispatchers with parameter derivatives
--------------------------------------------------

Gradient-returning counterparts of the orbit dispatchers above. Every
function accepts an additional ``dcoeffs`` tensor of shape
``(N, 6, D, 5)`` (knot, parameter, dimension, Taylor order) produced by
:func:`~meepmeep.backends.numba.taylor.orbit3dd.solve3d_orbit_d`.

.. currentmodule:: meepmeep.backends.numba.taylor.orbit3dd

.. autosummary::
   :toctree: generated

   solve3d_orbit_d
   pos_osd
   pos_ovd
   zpos_osd
   zpos_ovd
   sep_osd
   vel_osd
   vel_ovd
   zvel_osd
   zvel_ovd
   cos_alpha_osd
   cos_alpha_ovd
   cos_v_p_angle_ovd
   true_anomaly_ovd
   star_planet_distance_ovd
   lambert_phase_curve_osd
   lambert_phase_curve_ovd
   lambert_and_emission_ovd
   ev_signal_ovd
   rv_ovd
   light_travel_time_osd
   light_travel_time_ovd


Knot grid construction
----------------------

The knot grid and the time-to-knot table consumed by the multi-knot
dispatchers are built once per orbit by
:func:`~meepmeep.backends.numba.knots.create_knots`.

.. currentmodule:: meepmeep.backends.numba.knots

.. autosummary::
   :toctree: generated

   create_knots
   eccentric_anomaly
   true_anomaly
