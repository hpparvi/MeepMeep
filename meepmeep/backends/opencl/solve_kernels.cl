/*  MeepMeep: fast orbit calculations for exoplanet modelling
 *  Copyright (C) 2022-2026 Hannu Parviainen
 *
 *  Batched __kernel entry points for the Taylor coefficient solvers.
 *  Requires: common.cl, solve2d.cl, solve3d.cl
 *
 *  This is the ONLY shipped file containing __kernel entry points; every
 *  other .cl file is device functions only. Request it explicitly
 *  (`read_kernel_source('solve_kernels.cl')`) to get launchable solvers, or
 *  leave it out and call the solve2d/solve3d device functions from your own
 *  kernel. It is a separate file precisely so that "no __kernel entry points"
 *  remains true of the rest of the backend.
 *
 *  Each kernel maps one work item to one orbital parameter set, which is the
 *  right shape for a population sampler: emcee/DE walkers, or any batch of
 *  candidate parameter vectors evaluated together.
 *
 *  Input layout. `pars` is the C-contiguous (nsets, 7) array of
 *  (te, p, a, i, e, w, lan) rows -- note the leading `te`, the expansion-point
 *  time relative to the anchor, which is per parameter set rather than a
 *  kernel-wide constant so one launch can solve at different expansion points.
 *
 *  Output layout. The C-contiguous flattenings the evaluators already expect,
 *  one block per parameter set:
 *
 *      cf   (nsets, 2, 5)      set s at cf  + 10 * s      (2D)
 *      cf   (nsets, 3, 5)      set s at cf  + 15 * s      (3D)
 *      dcf  (nsets, 7, 2, 5)   set s at dcf + 70 * s      (2D)
 *      dcf  (nsets, 7, 3, 5)   set s at dcf + 105 * s     (3D)
 *
 *  The NDRange may be rounded up to a whole number of work groups; the guard
 *  in each kernel discards the surplus items.
 *
 *  In single precision the caller must keep `te` and `p` in a range where a
 *  float32 has useful resolution. `te` is relative, so unlike the absolute
 *  times the evaluators consume it needs no epoch shift.
 */


/* Solve the 2D Taylor coefficients for a batch of orbital parameter sets.

   One work item per set. Wraps `solve2d`. */
__kernel void solve2d_batch(__global const REAL *pars,
                            const int nsets,
                            __global REAL *cf) {
    const int gid = get_global_id(0);
    if (gid >= nsets) return;
    __global const REAL *q = pars + 7 * gid;
    solve2d(q[0], q[1], q[2], q[3], q[4], q[5], q[6], cf + 10 * gid);
}


/* Solve the 2D Taylor coefficients and their parameter derivatives.

   One work item per set. `from_periastron` is applied to every set in the
   batch. Wraps `solve2d_d`. */
__kernel void solve2d_d_batch(__global const REAL *pars,
                              const int nsets,
                              const int from_periastron,
                              __global REAL *cf,
                              __global REAL *dcf) {
    const int gid = get_global_id(0);
    if (gid >= nsets) return;
    __global const REAL *q = pars + 7 * gid;
    solve2d_d(q[0], q[1], q[2], q[3], q[4], q[5], q[6], from_periastron,
              cf + 10 * gid, dcf + 70 * gid);
}


/* Solve the 3D Taylor coefficients for a batch of orbital parameter sets.

   One work item per set. Wraps `solve3d`. */
__kernel void solve3d_batch(__global const REAL *pars,
                            const int nsets,
                            __global REAL *cf) {
    const int gid = get_global_id(0);
    if (gid >= nsets) return;
    __global const REAL *q = pars + 7 * gid;
    solve3d(q[0], q[1], q[2], q[3], q[4], q[5], q[6], cf + 15 * gid);
}


/* Solve the 3D Taylor coefficients and their parameter derivatives.

   One work item per set. `from_periastron` is applied to every set in the
   batch. Wraps `solve3d_d`. */
__kernel void solve3d_d_batch(__global const REAL *pars,
                              const int nsets,
                              const int from_periastron,
                              __global REAL *cf,
                              __global REAL *dcf) {
    const int gid = get_global_id(0);
    if (gid >= nsets) return;
    __global const REAL *q = pars + 7 * gid;
    solve3d_d(q[0], q[1], q[2], q[3], q[4], q[5], q[6], from_periastron,
              cf + 15 * gid, dcf + 105 * gid);
}
