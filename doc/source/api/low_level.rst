.. _api.lowlevel:

Low-level Taylor backend API
============================

This page catalogues the numba-compiled Taylor-series routines that live
under :mod:`meepmeep.backends.numba.taylor`. See :doc:`../taylor_overview`
for the conceptual model and :doc:`../naming_conventions` for the function
naming scheme.

All functions listed here are :func:`numba.njit`-compiled and operate on
NumPy arrays. Per-function detail (parameters, shapes, units, mathematical
notes) is documented in the source docstrings and rendered below via
``autosummary``.


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

   p2d
   p2dc
   d2d
   d2dc
   pd2d_c

Parameter-derivative variants:

.. currentmodule:: meepmeep.backends.numba.taylor.position2dd

.. autosummary::
   :toctree: generated

   p2d_d
   p2dc_d
   d2d_d
   d2dc_d


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

   p3d
   p3dc
   d3d
   d3dc
   pd3d
   pd3dc
   z3d
   z3dc

Parameter-derivative variants:

.. currentmodule:: meepmeep.backends.numba.taylor.position3dd

.. autosummary::
   :toctree: generated

   p3d_d
   p3dc_d
   d3d_d
   d3dc_d
   z3d_d
   z3dc_d


Velocities
----------

Velocity-vector and line-of-sight velocity evaluators built on the same
coefficient matrices used by the position evaluators.

.. currentmodule:: meepmeep.backends.numba.taylor.velocity3d

.. autosummary::
   :toctree: generated

   v3dc
   vzc
   vz
   rvc
   rv

Parameter-derivative variants:

.. currentmodule:: meepmeep.backends.numba.taylor.velocity3dd

.. autosummary::
   :toctree: generated

   v3dc_d
   vzc_d
   vz_d
   rvc_d
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

   xyz_o5s
   xyz_o5v
   z_o5s
   z_o5v
   pd_o5s
   star_planet_distance_o5v

Velocities and radial velocity:

.. autosummary::
   :toctree: generated

   vxyz_o5s
   vxyz_o5v
   vz_o5s
   vz_o5v
   rv_o5v

Phase angles, Lambert curves, and ellipsoidal variation:

.. autosummary::
   :toctree: generated

   true_anomaly_o5v
   cos_alpha_o5s
   cos_alpha_o5v
   cos_v_p_angle_o5v
   lambert_phase_curve_o5s
   lambert_phase_curve_o5v
   lambert_and_emission_o5v
   ev_signal_o5v

Light travel time:

.. autosummary::
   :toctree: generated

   light_travel_time_o5s
   light_travel_time_o5v


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
   xyz_o5s_d
   xyz_o5v_d
   z_o5s_d
   z_o5v_d
   pd_o5s_d
   vxyz_o5s_d
   vxyz_o5v_d
   vz_o5s_d
   vz_o5v_d
   cos_alpha_o5s_d
   cos_alpha_o5v_d
   cos_v_p_angle_o5v_d
   true_anomaly_o5v_d
   star_planet_distance_o5v_d
   lambert_phase_curve_o5s_d
   lambert_phase_curve_o5v_d
   lambert_and_emission_o5v_d
   ev_signal_o5v_d
   rv_o5v_d
   light_travel_time_o5s_d
   light_travel_time_o5v_d


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
