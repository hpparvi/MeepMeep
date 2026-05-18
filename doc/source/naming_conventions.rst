.. _naming_conventions:

Function naming conventions
===========================

The low-level Taylor backend uses a compact, regular naming scheme so that
related routines stay close together alphabetically and the role of each
function is decodable at a glance. This page is the decoder.

.. contents::
   :local:
   :depth: 2


Quantity prefix
---------------

The leading letters of a function name identify the quantity being
evaluated:

===========  ====================================================
Prefix       Quantity
===========  ====================================================
``p``        Position (vector of coordinates).
``d``        Projected planet-star distance, :math:`\sqrt{x^2+y^2}`.
``pd``       Position and projected distance, returned jointly.
``z``        Line-of-sight (z) coordinate only.
``v``        Velocity vector (3D modules only).
``vz``       Line-of-sight velocity component.
``rv``       Radial velocity (line-of-sight velocity, scaled).
===========  ====================================================

Examples: :func:`~meepmeep.backends.numba.taylor.position2d.p2d` returns
the (x, y) position; :func:`~meepmeep.backends.numba.taylor.position3d.d3d`
returns the projected distance from a 3D coefficient set;
:func:`~meepmeep.backends.numba.taylor.velocity3d.vz` returns the
line-of-sight velocity.


Dimensionality infix
--------------------

The digit between the prefix and any suffixes tags the spatial dimension
of the evaluator:

============  ==================================================
Infix         Meaning
============  ==================================================
``2d``        Two-dimensional (sky-plane :math:`x, y`).
``3d``        Three-dimensional (full :math:`x, y, z`).
============  ==================================================

The 2D evaluators are roughly 30 percent cheaper per call and are
sufficient for transit modelling; switch to 3D whenever the
line-of-sight :math:`z` is needed (eclipses, light travel time,
phase curves, radial velocities).


Centered vs. direct suffix
--------------------------

A trailing ``c`` marks a function as the **centered** variant: it takes a
time argument already shifted to lie close to a knot and skips the
epoch-folding step. Functions without the ``c`` are **direct** variants
that accept an absolute time together with ``t0`` and ``p``.

==========================  ==============================================
Suffix                      Meaning
==========================  ==============================================
*(none)*                    Direct: accepts absolute time, epoch-folds.
``c``                       Centered: accepts time relative to the knot.
==========================  ==============================================

Examples: :func:`~meepmeep.backends.numba.taylor.position3d.p3d` is the
direct variant, :func:`~meepmeep.backends.numba.taylor.position3d.p3dc`
the centered one. Both share the same coefficient matrix.

The 2D module follows the same rule — ``p2d`` / ``p2dc``, ``d2d`` /
``d2dc`` — with one combined evaluator named ``pd2d_c`` (centered)
spelled with an underscore to keep the ``pd`` prefix visually distinct
from the ``2d`` infix.


Parameter-derivative suffix
---------------------------

A trailing ``_d`` marks a function that returns *both* the quantity and
its partial derivatives with respect to the six orbital parameters
``(t0, p, a, i, e, w)``:

==========================  ==============================================
Suffix                      Meaning
==========================  ==============================================
``_d``                      Returns gradient w.r.t. orbital parameters.
==========================  ==============================================

The ``_d`` variants accept an additional argument ``dc`` — a
``(6, D, 5)`` parameter-derivative tensor produced by
:func:`~meepmeep.backends.numba.taylor.solve2dd.solve2d_d` or
:func:`~meepmeep.backends.numba.taylor.solve3dd.solve3d_d`. They live in
modules whose filename also ends in a ``d``: ``position2dd.py``,
``position3dd.py``, ``velocity3dd.py``, ``solve2dd.py``, ``solve3dd.py``,
``orbit3dd.py``.

The suffixes compose. A function ending in ``c_d`` is the centered
gradient-returning variant — for example
:func:`~meepmeep.backends.numba.taylor.position2dd.p2dc_d` and
:func:`~meepmeep.backends.numba.taylor.position3dd.d3dc_d`.


Multi-knot dispatcher suffix
----------------------------

Functions that span a whole orbit (i.e. look up the appropriate knot via
``pktable`` and delegate to a centered evaluator) live in
:mod:`~meepmeep.backends.numba.taylor.orbit3d` and use a different
suffix family that encodes the polynomial order and the input
cardinality:

============  ==============================================
Suffix        Meaning
============  ==============================================
``_o5s``      5th-order Taylor, **s**\ calar input time.
``_o5v``      5th-order Taylor, **v**\ ector of input times.
============  ==============================================

Examples: :func:`~meepmeep.backends.numba.taylor.orbit3d.xyz_o5s` /
:func:`~meepmeep.backends.numba.taylor.orbit3d.xyz_o5v`,
:func:`~meepmeep.backends.numba.taylor.orbit3d.vz_o5v`. The
gradient-returning counterparts in
:mod:`~meepmeep.backends.numba.taylor.orbit3dd` carry an extra
``_d``: ``xyz_o5v_d``, ``vz_o5s_d``, ``rv_o5v_d``, and so on.


Module naming
-------------

The same suffix rules apply at the module level:

================  ========================================================
Module suffix     Contents
================  ========================================================
``solve*``        Coefficient solvers (build ``c`` from orbital elements).
``position*``     Position / distance evaluators.
``velocity*``     Velocity / line-of-sight velocity evaluators.
``util*``         Geometric helpers (contact points, bounding box, durations).
``orbit*``        Multi-knot dispatchers spanning a full orbit.
*name*\ ``d``     Same module, parameter-derivative variants.
================  ========================================================

So ``position3dd.py`` is read as "3D position evaluators, with derivatives",
and ``orbit3dd.py`` as "orbit-spanning 3D dispatchers, with derivatives".
