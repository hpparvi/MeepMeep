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


Two-dimensional position and distance
-------------------------------------

Single-expansion-point evaluators for the sky-plane :math:`(x, y)` position and the
projected planet-star distance :math:`d = \sqrt{x^2 + y^2}`. Each
function operates on one ``(2, 5)`` coefficient matrix from
:func:`~meepmeep.numba2d.solve2d` and is
sufficient for transit light-curve modelling. The whole-orbit
dispatchers that batch these calls across a expansion-point grid live in
:ref:`api.lowlevel.orbit_dispatchers`.

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


Three-dimensional position and distance
---------------------------------------

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
   zvel_c
   zvel
   rv_c
   rv

Parameter-derivative variants:

.. currentmodule:: meepmeep.numba3d

.. autosummary::
   :toctree: generated

   vel_cd
   zvel_cd
   zvel_d
   rv_cd
   rv_d


Phase angle and reflected light
-------------------------------

Single-expansion-point cosine-of-phase-angle and Lambertian reflected-light
phase-curve evaluators, built on the same ``(3, 5)`` coefficient matrices.
The Lambert evaluators form the flux :math:`(k/r)^2 A_g\, f(\alpha)` from the
phase-angle cosine, using the instantaneous star-planet distance
:math:`r = \sqrt{x^2+y^2+z^2}` for the inverse-square illumination (exact for
eccentric orbits); the whole-orbit dispatchers in
:ref:`api.lowlevel.orbit_dispatchers` delegate to these.

.. currentmodule:: meepmeep.numba3d

.. autosummary::
   :toctree: generated

   cos_alpha_c
   cos_alpha
   lambert_phase_curve_c
   lambert_phase_curve

Parameter-derivative variants:

.. autosummary::
   :toctree: generated

   cos_alpha_cd
   cos_alpha_d
   lambert_phase_curve_cd
   lambert_phase_curve_d


Geometric utilities
-------------------

Contact points, durations, and bounding boxes derived analytically from a
coefficient matrix.

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

The 3D module :mod:`meepmeep.numba3d` also exposes the
same set of helpers operating on ``(3, 5)`` coefficient matrices.


.. _api.lowlevel.orbit_dispatchers:

Whole-orbit dispatchers (multi-expansion-point)
-----------------------------------------------

Whole-orbit evaluators that use a precomputed time-to-expansion-point table
(``ep_table``) to dispatch each input time to the appropriate expansion point and
delegate to the centered single-expansion-point evaluators above. Each name is a
single overloaded dispatcher that accepts either a scalar time or a
1-D float64 array of times; the ``_o`` suffix denotes the forward
dispatcher and ``_od`` its gradient-returning counterpart.

.. currentmodule:: meepmeep.numba3d

Orbit setup:

.. autosummary::
   :toctree: generated

   solve3d_orbit
   ep_ix

Positions and distances:

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

Phase angles, Lambert curves, and ellipsoidal variation:

.. autosummary::
   :toctree: generated

   true_anomaly_o
   cos_alpha_o
   cos_v_p_angle_o
   lambert_phase_curve_o
   ev_signal_o

Light travel time:

.. autosummary::
   :toctree: generated

   light_travel_time_o


Whole-orbit dispatchers with parameter derivatives
--------------------------------------------------

Gradient-returning counterparts of the orbit dispatchers above. Every
function accepts an additional ``dcoeffs`` tensor of shape
``(N, 7, D, 5)`` (expansion point, parameter, dimension, Taylor order) produced by
:func:`~meepmeep.numba3d.solve3d_orbit_d`.

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
   rv_od
   light_travel_time_od


Expansion point grid construction
---------------------------------

The expansion-point grid and the time-to-expansion-point table consumed by the multi-expansion-point
dispatchers are built once per orbit by
:func:`~meepmeep.numba3d.create_expansion_points`.

.. currentmodule:: meepmeep.numba3d

.. autosummary::
   :toctree: generated

   create_expansion_points

The two anomaly helpers below are not part of the aggregator surface
but remain available at their source path.

.. currentmodule:: meepmeep.backends.numba.expansion_points

.. autosummary::
   :toctree: generated

   eccentric_anomaly
   true_anomaly
