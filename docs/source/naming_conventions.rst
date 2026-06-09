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
``zpos``          Line-of-sight (:math:`z`) coordinate only.
``vel``           Velocity vector (3D modules only).
``zvel``          Line-of-sight velocity component.
``rv``            Radial velocity (line-of-sight velocity, scaled).
================  ====================================================

Examples: :func:`~meepmeep.backends.numba.point2d.position.pos` returns
the (x, y) position; :func:`~meepmeep.backends.numba.point3d.separation.sep`
returns the projected separation from a 3D coefficient set;
:func:`~meepmeep.backends.numba.point3d.zvelocity.zvel` returns the
line-of-sight velocity.


Dimensionality lives in the module, not the function
-----------------------------------------------------

The spatial dimensionality of an evaluator is encoded by the package name
(the ``point2d``/``point2dd`` packages vs the ``point3d``/``point3dd``
packages, ``point2d.util`` vs ``point3d.util``) rather than by the
function name. Both ``meepmeep.backends.numba.point2d.position`` and
``meepmeep.backends.numba.point3d.position`` therefore expose a function
called ``pos``; the 3D package additionally exposes ``zpos``, etc.

The 2D evaluators are roughly 30 percent cheaper per call and are
sufficient for transit modelling; switch to 3D whenever the
line-of-sight :math:`z` is needed (eclipses, light travel time,
phase curves, radial velocities).


Centered vs. direct suffix
--------------------------

A trailing ``_c`` marks a function as the **centered** variant: it takes a
time argument already shifted to lie close to a knot and skips the
epoch-folding step. Functions without the ``_c`` are **direct** variants
that accept an absolute time together with the knot time ``tk`` and ``p``.

.. note::

   Two time names are used consistently throughout the backend. ``tc`` is
   the **transit-centre** time (time of inferior conjunction; the orbital
   element the gradient slot 0 is taken with respect to). ``tk`` is the
   **knot time** — the time at the *center* of a local Taylor expansion
   (a *knot*; see :ref:`taylor_overview`), and hence the ``solve*`` first
   argument and the fold reference of the direct evaluators. ``tk`` need
   not equal ``tc``: it equals ``tc`` only when the knot is placed at the
   transit centre.

==========================  ==============================================
Suffix                      Meaning
==========================  ==============================================
*(none)*                    Direct: accepts absolute time, epoch-folds.
``_c``                      Centered: accepts time relative to the knot.
==========================  ==============================================

Examples: :func:`~meepmeep.backends.numba.point3d.position.pos` is the
direct variant, :func:`~meepmeep.backends.numba.point3d.position.pos_c`
the centered one. Both share the same coefficient matrix.

The centered evaluators are the shared workhorses for both usage modes
introduced in :ref:`taylor_two_modes`. Single-knot callers reach them
either directly (when observation times are already folded around the
knot) or via the direct variants (when the evaluator should epoch-fold
on the caller's behalf). Multi-knot dispatchers always reach them
through a ``pktable`` lookup that yields a knot index and a
centered time.

The 2D module follows the same rule — ``pos`` / ``pos_c``, ``sep`` /
``sep_c``.


Parameter-derivative suffix
---------------------------

The gradient-returning variants live in the ``*dd``-suffixed packages
(the ``point2dd``/``point3dd`` single-knot packages and the ``orbit3dd/``
multi-knot package) and return *both* the
quantity and its partial derivatives with respect to the seven orbital
parameters ``(tc, p, a, i, e, w, lan)``. The suffix encodes whether the
underlying evaluator is centered or direct:

==========================  ==============================================
Suffix                      Meaning
==========================  ==============================================
``_d``                      Direct evaluator returning a gradient
                            w.r.t. orbital parameters.
``_cd``                     Centered evaluator returning a gradient
                            w.r.t. orbital parameters.
==========================  ==============================================

These functions accept an additional argument ``dc`` — a ``(7, D, 5)``
parameter-derivative tensor produced by
:func:`~meepmeep.backends.numba.point2dd.solve.solve2d_d` or
:func:`~meepmeep.backends.numba.point3dd.solve.solve3d_d`.

Examples: :func:`~meepmeep.backends.numba.point2dd.position.pos_cd` and
:func:`~meepmeep.backends.numba.point3dd.separation.sep_cd` are the
centered gradient-returning variants;
:func:`~meepmeep.backends.numba.point3dd.position.pos_d` is the direct
counterpart.

Like their value-only twins (``pos`` / ``sep``), the ``_d`` / ``_cd``
evaluators accept **either** a scalar time **or** a 1-D array of times and
dispatch via ``numba.extending.overload`` at compile time (inside ``@njit``)
or at call time (pure Python) — exactly like the ``_o`` / ``_od`` multi-knot
dispatchers below. A scalar time yields a length-7 gradient; a 1-D array of
length ``N`` yields results with a leading ``N`` axis (e.g. ``sep_d`` returns
``d`` of shape ``(N,)`` and ``dd`` of shape ``(N, 7)``). The array path is the
one used by the high-level ``Knot2D`` properties.

Internally each dispatcher routes to a private kernel with the explicit
``_s`` / ``_v`` (scalar / vector) suffix — e.g. ``_pos_cd_s`` and ``_pos_cd_v``
in the ``point2dd/`` package. Reach for those private kernels only when you need
to avoid the dispatcher's type check (rarely useful) or when contributing to
MeepMeep itself.


Multi-knot dispatcher suffix
----------------------------

When the workflow needs a whole-orbit dispatcher — for example to
evaluate a phase curve or an RV time series across an arbitrary range
of times — the functions in
:mod:`~meepmeep.backends.numba.orbit3d` look up the appropriate
knot via ``pktable`` and delegate to a centered evaluator. The public
surface is a single overloaded entry point per quantity that accepts
either a scalar time or a 1-D array of times and dispatches at compile
time (inside ``@njit``) or at call time (pure Python):

============  ==============================================
Suffix        Meaning
============  ==============================================
``_o``        Orbit-spanning dispatcher. Scalar time → scalar
              result; 1-D float64 array → array result.
``_od``       Same, with gradients w.r.t. orbital parameters.
============  ==============================================

Examples: :func:`~meepmeep.backends.numba.orbit3d.pos_o`,
:func:`~meepmeep.backends.numba.orbit3d.zvel_o`,
:func:`~meepmeep.backends.numba.orbit3dd.pos_od`,
:func:`~meepmeep.backends.numba.orbit3dd.rv_od`.

Internally each dispatcher routes to a private kernel with the explicit
``_os`` / ``_ov`` (scalar / vector) suffix — e.g. ``_pos_os`` and
``_pos_ov`` in the ``orbit3d/`` package, ``_pos_osd`` and ``_pos_ovd`` in
the ``orbit3dd/`` package. Reach for those private kernels only when you need to
avoid the dispatcher's type check (rarely useful) or when contributing
to MeepMeep itself.

.. note::
   When upgrading from an earlier MeepMeep release where the public
   names were ``pos_os`` / ``pos_ov`` / ``pos_osd`` / ``pos_ovd``,
   purge any ``__pycache__/*.nbi`` / ``*.nbc`` files from callers
   compiled with ``cache=True`` so Numba recompiles against the new
   dispatchers.


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

So ``point3dd/position.py`` is read as "3D position evaluators, with derivatives",
and the ``orbit3dd/`` package as "orbit-spanning 3D dispatchers, with derivatives".
