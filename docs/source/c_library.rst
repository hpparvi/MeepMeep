C library
=========

The Taylor-series evaluators are also available as a plain C99 library,
``libmeepmeep``. It is not a Python extension and the Python package never
calls it: it is a second compile target of the sources the OpenCL backend
ships, built and installed on its own with CMake from the ``c/`` directory of
the repository. Use it when a model has to run where Python is not welcome,
or to embed the evaluators in another compiled code base.

Precision is fixed to double.

Building
--------

.. code-block:: bash

   cmake -S c -B c/build
   cmake --build c/build
   cmake --install c/build --prefix "$HOME/.local"

The install places ``meepmeep.h`` under ``include/`` and ``libmeepmeep``
under the platform's library directory (CMake's ``GNUInstallDirs``: ``lib/``
on most systems, ``lib64/`` on RHEL-family Linux); link with
``-lmeepmeep -lm``. ``BUILD_SHARED_LIBS=OFF``
gives a static library and ``MEEPMEEP_BUILD_EXAMPLES=ON`` also builds
``c/examples/transit.c``. Never add ``-ffast-math``: the numba kernels that
are deliberately compiled without fastmath (``true_anomaly``, ``rv``) rely on
strict math, and the parity tests pin agreement with numba at about 1e-12.

What is in it
-------------

Shared with the OpenCL backend (the functions in
``meepmeep/backends/opencl/*.cl``, compiled here as C):

- the coefficient solvers ``solve2d``, ``solve2d_d``, ``solve3d``,
  ``solve3d_d``;
- the single-expansion-point evaluators with their trailing dimension digit
  (``pos_c2``, ``sep3``, ``sep_cd3``, ``rv_d3``, ``lambert_phase_curve3``, ...);
- the multi-expansion-point ``_o`` / ``_od`` evaluators (``sep_o``,
  ``pos_od``, ``rv_od``, ``true_anomaly_od``, ``light_travel_time_o``, ...)
  together with ``ep_ix``;
- the helpers ``mean_anomaly_at_transit``, ``ea_from_ma``, ``taylor5``, ...

Only in the C library (``c/src/``), because a self-contained library needs
the pieces that stay host-side for OpenCL:

- ``create_expansion_points``: expansion-point placement for the ``'mm'``,
  ``'ea'`` and ``'ta'`` strategies (``MM_EP_MM``, ``MM_EP_EA``,
  ``MM_EP_TA``). The anomaly-uniform strategies solve for the phases with a
  transcription of scipy's ``brentq``, so the placement agrees with the
  Python side to the root-finder tolerance. Size the time-to-expansion-point
  table with ``expansion_table_size``, which returns the size the Python side
  uses by default (eight bins per narrowest expansion-point region, at least
  200); a coarser table limits the accuracy at high eccentricity. Both return
  an ``mm_status`` code; ``mm_status_string`` describes it:

  ==============================  ========================================
  Code                            Meaning
  ==============================  ========================================
  ``MM_OK`` (0)                   Success.
  ``MM_ERR_N_EP`` (1)             ``n_ep`` must be odd and at least 3.
  ``MM_ERR_QUANTITY`` (2)         Unknown placement strategy.
  ``MM_ERR_TRES`` (3)             ``tres`` must be positive.
  ``MM_ERR_ECCENTRICITY`` (4)     ``e`` must satisfy ``0 <= e < 1``.
  ``MM_ERR_BRACKET`` (5)          The root bracket lost its sign change.
  ``MM_ERR_NO_CONVERGENCE`` (6)   The root finder hit its iteration cap.
  ==============================  ========================================

- ``solve3d_orbit`` and ``solve3d_orbit_d``: the coefficient stacks for every
  expansion point of one orbit.
- ``tc_to_tp_gradient``, ``tp_to_tc_gradient`` and
  ``tp_to_tc_gradient_orbit``: the gradient basis transforms, working in
  place where numba returns a copy.

Shared with OpenCL but worth knowing in C: ``mean_anomaly_at_transit`` (for
the periastron anchor ``tpa``) and ``eccentricity_vector_d``, whose ``dev``
output is the extra input of ``true_anomaly_od``.

Not included: the Newton-Raphson reference solvers, the contact-point,
duration and ``find_z_min`` helpers, and anything from the :class:`~meepmeep.orbit.Orbit`
class layer.

Example
-------

The full-orbit pipeline, from parameters to the sky-projected separation and
its gradient (``c/examples/transit.c`` is the complete program):

.. code-block:: c

   #include "meepmeep.h"

   int tres;
   expansion_table_size(NPT, e, MM_EP_EA, &tres);   /* the numba default */
   double ep_times[NPT], change_times[NPT - 1], dt;
   int *ep_table = malloc(tres * sizeof(int));
   int status = create_expansion_points(NPT, e, MM_EP_EA, tres,
                                        ep_times, change_times, &dt, ep_table);
   if (status != MM_OK) { /* mm_status_string(status) */ }

   double coeffs[NPT * 15], dcoeffs[NPT * 105];
   solve3d_orbit_d(ep_times, NPT, p, a, inc, e, w, lan, coeffs, dcoeffs);
   tp_to_tc_gradient_orbit(dcoeffs, NPT, p, e, w);   /* (tc, p, a, i, e, w, lan) */

   const double two_pi = 6.28318530717958647693;   /* M_PI is not C99 */
   double tpa = tc - mean_anomaly_at_transit(e, w) / two_pi * p;
   double dz[MM_NPAR];
   double z = sep_od(t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs, dz);

For a single expansion point around the transit, the
:class:`~meepmeep.expansion2d.Expansion2D` path, call ``solve2d`` or
``solve3d`` once and evaluate with the ``2`` / ``3``-suffixed functions.

Conventions
-----------

- Names mirror ``meepmeep.numba2d`` / ``meepmeep.numba3d`` with the OpenCL
  adjustments described in :ref:`naming-opencl`: a trailing dimension digit
  on the single-expansion-point functions, scalar forms only, and no
  optional arguments (pass ``te = 0.0``, ``lan = 0.0``, ``timing_is_tc = 1``,
  and the solvers' ``from_periastron = 0`` explicitly). Symbols are not
  prefixed, and the internal helpers (``taylor5``, ``ep_lookup``,
  ``mm_mod_two_pi``, ``lambert_kernel``, ``rv_scale``, ...) are exported
  too, so avoid those names in code that links the library.
- Coefficient arrays are the C-contiguous flattenings of the numba arrays:
  ``c`` is ``(D, 5)`` with row ``d`` at ``c + 5 * d``; ``dc`` is ``(7, D, 5)``
  with parameter block ``m`` at ``dc + 5 * D * m``; ``coeffs`` is
  ``(npt, 3, 5)`` and ``dcoeffs`` ``(npt, 7, 3, 5)``.
- Gradients use the seven-parameter order ``(tc, p, a, i, e, w, lan)`` and
  are written into caller-provided buffers; functions with extra physical
  inputs append their derivatives (``rv_od`` is 8 wide, the Lambert and
  ellipsoidal-variation functions 9, emission 10).
- ``solve3d_orbit_d`` returns the periastron basis ``(tp, p, a, i, e, w, lan)``;
  apply ``tp_to_tc_gradient_orbit`` for the transit-centre basis. The basis
  transforms work in place. The single-block ``tc_to_tp_gradient`` /
  ``tp_to_tc_gradient`` take an extra ``block`` argument after ``dc``: the
  number of doubles per parameter row, 10 for a 2D ``(7, 2, 5)`` block and
  15 for a 3D ``(7, 3, 5)`` one.
- ``ep_table`` is an ``int`` array.

Keeping the two targets in sync
-------------------------------

The ``.cl`` files are written against three macros defined at the top of
``common.cl``: ``MM_GLOBAL`` (``__global`` on the device, empty in C),
``MM_INLINE`` (``inline`` on the device, empty in C so the functions become
the library's exported symbols) and ``REAL`` (``double`` or ``float`` on the
device, ``double`` in C). ``c/src/meepmeep.c`` includes the files in
dependency order after the public header, so a signature that drifts from
its prototype is a compile error.

The prototype block of ``c/include/meepmeep.h`` is generated from the
``.cl`` sources by ``python c/tools/generate_header.py``. The test module
``meepmeep/tests/test_c_library.py`` fails if the committed header is stale,
and, when a C compiler is on ``PATH``, compiles the library and checks the
C-only functions and a sample of the shared ones against numba through
``ctypes``.
