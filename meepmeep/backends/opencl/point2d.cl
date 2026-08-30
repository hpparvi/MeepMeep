/*  MeepMeep: fast orbit calculations for exoplanet modelling
 *  Copyright (C) 2022-2026 Hannu Parviainen
 *
 *  Single-expansion-point 2D evaluators (values only).
 *  Requires: common.cl
 *
 *  These are device-function ports of `meepmeep.numba2d` {pos_c, pos, sep_c,
 *  sep}. OpenCL C has a single flat namespace, so where the numba backend
 *  encodes dimensionality in the package name (point2d vs point3d), the
 *  single-expansion-point functions carry a trailing dimension digit
 *  instead: `2` here (`pos_c2`, ...) and `3` in point3d.cl (`pos_c3`, ...).
 *
 *  `c` points at the 10 contiguous coefficients of a flattened (2, 5)
 *  solve2d matrix: row 0 is x, row 1 is y. C has no default arguments, so
 *  the `te = 0.0` default of the numba direct evaluators becomes a mandatory
 *  parameter: pass (REAL)0.0 for an expansion point at the transit centre.
 */


/* Planet sky-plane (x, y) position at an expansion-point-centred time.

   Port of `meepmeep.numba2d.pos_c`. */
inline void pos_c2(REAL t, __global const REAL *c, REAL *px, REAL *py) {
    *px = taylor5(t, c);
    *py = taylor5(t, c + 5);
}


/* Planet sky-plane (x, y) position at an absolute time.

   Folds `t` into a single orbital epoch around the expansion point at
   `tc + te` and evaluates the centred polynomial. Port of
   `meepmeep.numba2d.pos`. */
inline void pos2(REAL t, REAL tc, REAL p, __global const REAL *c, REAL te,
                 REAL *px, REAL *py) {
    REAL epoch = floor((t - tc - te + (REAL)0.5 * p) / p);
    pos_c2(t - (tc + te + epoch * p), c, px, py);
}


/* Sky-projected planet-star separation at an expansion-point-centred time.

   Port of `meepmeep.numba2d.sep_c`. */
inline REAL sep_c2(REAL t, __global const REAL *c) {
    REAL px, py;
    pos_c2(t, c, &px, &py);
    return sqrt(px * px + py * py);
}


/* Sky-projected planet-star separation at an absolute time.

   Port of `meepmeep.numba2d.sep`. */
inline REAL sep2(REAL t, REAL tc, REAL p, __global const REAL *c, REAL te) {
    REAL epoch = floor((t - tc - te + (REAL)0.5 * p) / p);
    return sep_c2(t - (tc + te + epoch * p), c);
}
