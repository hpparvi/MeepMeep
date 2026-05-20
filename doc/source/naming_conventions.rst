.. _naming_conventions:

Function naming conventions
===========================

The low-level Taylor backend uses a compact, regular naming scheme so that
related routines stay close together alphabetically and the role of each
function is decodable at a glance. This page is the decoder.

.. contents::
   :local:
   :depth: 2


Quantity stem
-------------

The leading word of a function name identifies the quantity being
evaluated:

================  ====================================================
Stem              Quantity
================  ====================================================
``pos``           Position (vector of coordinates).
``sep``           Sky-projected planet-star separation, :math:`\sqrt{x^2+y^2}`.
``pos_and_sep``   Position and projected separation, returned jointly.
``pz``            Line-of-sight (:math:`z`) coordinate only.
``vel``           Velocity vector (3D modules only).
``zvel``          Line-of-sight velocity component.
``rv``            Radial velocity (line-of-sight velocity, scaled).
================  ====================================================

Examples: :func:`~meepmeep.backends.numba.taylor.position2d.pos` returns
the (x, y) position; :func:`~meepmeep.backends.numba.taylor.position3d.sep`
returns the projected separation from a 3D coefficient set;
:func:`~meepmeep.backends.numba.taylor.velocity3d.zvel` returns the
line-of-sight velocity.


Dimensionality lives in the module, not the function
-----------------------------------------------------

The spatial dimensionality of an evaluator is encoded by the module name
(``position2d`` vs ``position3d``, ``util2d`` vs ``util3d``) rather than
by the function name. Both ``meepmeep.backends.numba.taylor.position2d``
and ``meepmeep.backends.numba.taylor.position3d`` therefore expose a
function called ``pos``; the 3D module additionally exposes ``pz``,
``pos_and_sep``, etc.

The 2D evaluators are roughly 30 percent cheaper per call and are
sufficient for transit modelling; switch to 3D whenever the
line-of-sight :math:`z` is needed (eclipses, light travel time,
phase curves, radial velocities).


Centered vs. direct suffix
--------------------------

A trailing ``_c`` marks a function as the **centered** variant: it takes a
time argument already shifted to lie close to a knot and skips the
epoch-folding step. Functions without the ``_c`` are **direct** variants
that accept an absolute time together with ``t0`` and ``p``.

==========================  ==============================================
Suffix                      Meaning
==========================  ==============================================
*(none)*                    Direct: accepts absolute time, epoch-folds.
``_c``                      Centered: accepts time relative to the knot.
==========================  ==============================================

Examples: :func:`~meepmeep.backends.numba.taylor.position3d.pos` is the
direct variant, :func:`~meepmeep.backends.numba.taylor.position3d.pos_c`
the centered one. Both share the same coefficient matrix.

The centered evaluators are the shared workhorses for both usage modes
introduced in :ref:`taylor_two_modes`. Single-knot callers reach them
either directly (when observation times are already folded around the
knot) or via the direct variants (when the evaluator should epoch-fold
on the caller's behalf). Multi-knot dispatchers always reach them
through a ``pktable`` lookup that yields a knot index and a
centered time.

The 2D module follows the same rule — ``pos`` / ``pos_c``, ``sep`` /
``sep_c``, ``pos_and_sep`` / ``pos_and_sep_c``.


Parameter-derivative suffix
---------------------------

The gradient-returning variants live in the ``*dd``-suffixed modules
(``position2dd.py``, ``position3dd.py``, ``velocity3dd.py``,
``solve2dd.py``, ``solve3dd.py``, ``orbit3dd.py``) and return *both* the
quantity and its partial derivatives with respect to the six orbital
parameters ``(t0, p, a, i, e, w)``. The suffix encodes whether the
underlying evaluator is centered or direct:

==========================  ==============================================
Suffix                      Meaning
==========================  ==============================================
``_d``                      Direct evaluator returning a gradient
                            w.r.t. orbital parameters.
``_cd``                     Centered evaluator returning a gradient
                            w.r.t. orbital parameters.
==========================  ==============================================

These functions accept an additional argument ``dc`` — a ``(6, D, 5)``
parameter-derivative tensor produced by
:func:`~meepmeep.backends.numba.taylor.solve2dd.solve2d_d` or
:func:`~meepmeep.backends.numba.taylor.solve3dd.solve3d_d`.

Examples: :func:`~meepmeep.backends.numba.taylor.position2dd.pos_cd` and
:func:`~meepmeep.backends.numba.taylor.position3dd.sep_cd` are the
centered gradient-returning variants;
:func:`~meepmeep.backends.numba.taylor.position3dd.pos_d` is the direct
counterpart.


Multi-knot dispatcher suffix
----------------------------

When the workflow needs a whole-orbit dispatcher — for example to
evaluate a phase curve or an RV time series across an arbitrary range
of times — the functions in
:mod:`~meepmeep.backends.numba.taylor.orbit3d` look up the appropriate
knot via ``pktable`` and delegate to a centered evaluator. These
dispatchers use their own suffix family that encodes the input
cardinality:

============  ==============================================
Suffix        Meaning
============  ==============================================
``_os``       Orbit-spanning, **s**\ calar input time.
``_ov``       Orbit-spanning, **v**\ ector of input times.
============  ==============================================

Examples: :func:`~meepmeep.backends.numba.taylor.orbit3d.pos_os` /
:func:`~meepmeep.backends.numba.taylor.orbit3d.pos_ov`,
:func:`~meepmeep.backends.numba.taylor.orbit3d.zvel_ov`. The
gradient-returning counterparts live in
:mod:`~meepmeep.backends.numba.taylor.orbit3dd` and append ``d`` to
the suffix, mirroring the position/velocity ``_c`` → ``_cd``
convention: ``_osd`` and ``_ovd`` (e.g. ``pos_ovd``, ``zvel_osd``,
``rv_ovd``).


Module naming
-------------

The same suffix rules apply at the module level:

================  ========================================================
Module suffix     Contents
================  ========================================================
``solve*``        Coefficient solvers (build ``c`` from orbital elements).
``position*``     Position / separation evaluators.
``velocity*``     Velocity / line-of-sight velocity evaluators.
``util*``         Geometric helpers (contact points, bounding box, durations).
``orbit*``        Multi-knot dispatchers spanning a full orbit.
*name*\ ``d``     Same module, parameter-derivative variants.
================  ========================================================

So ``position3dd.py`` is read as "3D position evaluators, with derivatives",
and ``orbit3dd.py`` as "orbit-spanning 3D dispatchers, with derivatives".
