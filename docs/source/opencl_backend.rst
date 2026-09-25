.. _opencl_backend:

OpenCL backend
==============

The OpenCL backend ships the numba evaluators, and the Taylor coefficient
solvers, as OpenCL C *device functions*. You write the kernel and MeepMeep
provides the functions it calls: prepend the source to your own kernel code,
build the program, and launch it. Context, queue and program management stay
with the caller, so the backend drops into an existing OpenCL code base (such
as PyTransit's) without imposing a runtime.

It needs ``pyopencl`` and an OpenCL platform only to *run* kernels; reading
the source does not:

.. code-block:: bash

   pip install "meepmeep[opencl]"

The same ``.cl`` files also compile as plain C99, which is how the C library
is built (see :doc:`c_library`).


A worked example
----------------

The host builds the expansion-point grid and the coefficients with the numba
backend, as for any multi-expansion-point evaluation, and uploads them; the
kernel evaluates the projected separation and its gradient at one time per
work item.

.. code-block:: python

   import numpy as np
   import pyopencl as cl
   from meepmeep.backends.opencl import read_kernel_source, build_options
   from meepmeep.numba3d import (create_expansion_points, solve3d_orbit_d,
                                 tp_to_tc_gradient_orbit, mean_anomaly_at_transit)

   ctx = cl.create_some_context()
   queue = cl.CommandQueue(ctx)

   tc, p, a, i, e, w = 0.0, 3.0, 8.5, np.radians(88.0), 0.1, np.radians(60.0)
   ep_times, _, dt, ep_table = create_expansion_points(15, max(e, 0.2), "ea")
   coeffs, dcoeffs = solve3d_orbit_d(ep_times, p, a, i, e, w, npt=15)
   tp_to_tc_gradient_orbit(dcoeffs, p, e, w)      # transit-centre basis
   tpa = tc - mean_anomaly_at_transit(e, w) / (2 * np.pi) * p
   times = np.linspace(-0.2, 0.2, 10_000)

   kernel_src = """
   __kernel void separation(__global const REAL *t, const REAL tpa, const REAL p,
                            const REAL dt, __global const int *ep_table,
                            __global const REAL *ep_times, __global const REAL *coeffs,
                            __global const REAL *dcoeffs,
                            __global REAL *z, __global REAL *dz) {
       int j = get_global_id(0);
       REAL g[7];
       z[j] = sep_od(t[j], tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs, g);
       for (int m = 0; m < 7; m++) dz[7 * j + m] = g[m];
   }
   """
   program = cl.Program(ctx, read_kernel_source("orbit3dd.cl") + kernel_src)
   program = program.build(options=build_options("double"))

   mf = cl.mem_flags
   def upload(x):
       return cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                        hostbuf=np.ascontiguousarray(x).ravel())

   z, dz = np.empty(times.size), np.empty((times.size, 7))
   b_z = cl.Buffer(ctx, mf.WRITE_ONLY, z.nbytes)
   b_dz = cl.Buffer(ctx, mf.WRITE_ONLY, dz.nbytes)
   program.separation(queue, (times.size,), None, upload(times), np.float64(tpa),
                      np.float64(p), np.float64(dt), upload(ep_table.astype(np.int32)),
                      upload(ep_times), upload(coeffs), upload(dcoeffs), b_z, b_dz)
   cl.enqueue_copy(queue, z, b_z)
   cl.enqueue_copy(queue, dz, b_dz)

In double precision the result agrees with
:func:`~meepmeep.numba3d.sep_od` to about :math:`10^{-15}`. The numba
function each device function ports is the natural oracle for testing a
kernel.


The source files
----------------

:func:`~meepmeep.backends.opencl.read_kernel_source` concatenates the named
files together with everything they depend on, in a valid order, so the
result always compiles when prepended to kernel code:

.. list-table::
   :header-rows: 1
   :widths: 22 50 28

   * - File
     - Contents
     - Pulls in
   * - ``common.cl``
     - Macros, constants, Horner helpers, the Kepler solver, ``mean_anomaly_at_transit``, ``eccentricity_vector_d``
     - (nothing)
   * - ``solve2d.cl``
     - ``solve2d``, ``solve2d_d``
     - ``common.cl``
   * - ``solve3d.cl``
     - ``solve3d``, ``solve3d_d``
     - ``common.cl``
   * - ``point2d.cl``
     - 2D single-expansion-point values
     - ``common.cl``
   * - ``point2dd.cl``
     - 2D single-expansion-point gradients
     - ``common.cl``, ``point2d.cl``
   * - ``point3d.cl``
     - 3D single-expansion-point values
     - ``common.cl``
   * - ``point3dd.cl``
     - 3D single-expansion-point gradients
     - ``common.cl``, ``point3d.cl``
   * - ``orbit3d.cl``
     - Whole-orbit values, ``ep_ix``
     - ``common.cl``, ``point3d.cl``
   * - ``orbit3dd.cl``
     - Whole-orbit gradients
     - ``orbit3d.cl``, ``point3dd.cl``
   * - ``solve_kernels.cl``
     - Batched ``__kernel`` solvers (opt-in)
     - ``solve2d.cl``, ``solve3d.cl``

:func:`~meepmeep.backends.opencl.read_full_source` returns every file,
including the batched solve kernels; ``SOURCE_FILES`` lists the names.
:func:`~meepmeep.backends.opencl.build_options` returns the preprocessor
options for ``'double'`` (``-DREAL=double -DUSE_FP64``, which also enables
``cl_khr_fp64``) or ``'single'`` (``-DREAL=float``) precision, and raises
``ValueError`` for anything else. Never add ``-cl-fast-relaxed-math``: the
device functions are tested for about :math:`10^{-12}` agreement with numba,
and the kernels that numba compiles without fastmath on purpose
(``true_anomaly``, ``rv``) rely on strict math.

.. autofunction:: meepmeep.backends.opencl.read_kernel_source

.. autofunction:: meepmeep.backends.opencl.read_full_source

.. autofunction:: meepmeep.backends.opencl.build_options


Names and signatures
--------------------

The device functions keep the numba names and argument order, values first
and gradient output buffers last, with a few deviations that C imposes:

- The single-expansion-point functions take a trailing dimension digit
  (``pos_c2`` / ``pos_c3``, ``sep_cd2`` / ``sep_cd3``, ...), because OpenCL C
  has one flat namespace. The whole-orbit ``_o`` / ``_od`` functions and the
  dimension-agnostic helpers (``ep_ix``, ``rv_scale``, ``lambert_kernel``, ...)
  are unsuffixed. See :ref:`naming-opencl`.
- Only the scalar forms exist; the NDRange supplies the loop, so there are no
  ``_v`` / ``_vp`` / ``_ov*`` kernels.
- Optional numba arguments are mandatory: pass ``te``, ``lan``,
  ``timing_is_tc`` and the solvers' ``from_periastron`` explicitly.
- The solvers write their matrices through ``__global`` output pointers and
  spell the inclination ``inc``.
- The comment above each device function with a numba counterpart names it
  ("Port of ``meepmeep.numba3d.sep_cd``"); that function's docstring
  documents the arguments, units, shapes and gradient order.

Gradients use the order ``(tc, p, a, i, e, w, lan)``, followed by any
physical inputs (``rv_od`` is 8 wide, the Lambert and ellipsoidal-variation
functions 9, emission 10). ``true_anomaly_od`` also takes the eccentricity
vector's Jacobian, which ``eccentricity_vector_d`` computes on the device.


What the kernel must follow
---------------------------

- Upload the coefficient arrays as their C-contiguous flattenings
  (``np.ascontiguousarray(c).ravel()``): ``c`` is ``(D, 5)``, ``dc``
  ``(7, D, 5)``, ``coeffs`` ``(npt, 3, 5)`` and ``dcoeffs``
  ``(npt, 7, 3, 5)``.
- Upload ``ep_table`` as int32 (``ep_table.astype(np.int32)``); numba builds
  it as int64.
- In single precision, subtract a float64 reference epoch from the absolute
  times (and ``tc`` / ``tpa``) on the host before casting: a float32 ulp at
  BJD 2.4e6 is about a quarter of a day. The Kepler tolerance adapts to the
  precision (``MM_EA_TOL``, overridable with ``-DMM_EA_TOL=...``).
- The expansion-point placement, the Newton-Raphson references and the
  contact-point searches stay on the host; the device has no counterpart.


Solving on the device
---------------------

The coefficient solvers are device functions too, so a kernel can build its
own coefficients. Solving on the device pays off for *batches* of parameter
sets, such as the walkers of a population sampler, where the coefficients
then never leave the device; for one parameter set per likelihood call the
launch overhead dominates, and solving on the host is as fast.
``solve_kernels.cl`` wraps the solvers as launchable kernels, one work item
per parameter set:

.. code-block:: python

   pars = np.array([[0.0, 3.0, 8.5, 1.54, 0.1, 0.5, 0.0],     # rows:
                    [0.0, 3.2, 9.0, 1.53, 0.2, 1.0, 0.3]])    # (te, p, a, i, e, w, lan)
   solver = cl.Program(ctx, read_kernel_source("solve_kernels.cl"))
   solver = solver.build(options=build_options("double"))
   cf = np.empty((len(pars), 3, 5))
   b_cf = cl.Buffer(ctx, mf.WRITE_ONLY, cf.nbytes)
   solver.solve3d_batch(queue, (len(pars),), None, upload(pars),
                        np.int32(len(pars)), b_cf)
   cl.enqueue_copy(queue, cf, b_cf)

The four kernels are ``solve2d_batch``, ``solve3d_batch``,
``solve2d_d_batch`` and ``solve3d_d_batch``; the gradient versions take a
``from_periastron`` flag after ``nsets`` and a second output buffer. Each
parameter set's output block sits at a fixed stride: 10 or 15 values for the
2D or 3D coefficients, 70 or 105 for their derivatives. The NDRange may be
rounded up to a whole number of work groups; the kernels ignore the surplus
work items.
