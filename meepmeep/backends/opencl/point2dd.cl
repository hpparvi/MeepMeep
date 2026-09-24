/*  MeepMeep: fast orbit calculations for exoplanet modelling
 *  Copyright (C) 2022-2026 Hannu Parviainen
 *
 *  Single-expansion-point 2D evaluators with orbital-parameter derivatives.
 *  Requires: common.cl, point2d.cl
 *
 *  Device-function ports of `meepmeep.numba2d` {pos_cd, pos_d, sep_cd,
 *  sep_d}, with the trailing `2` disambiguating them from the 3D functions
 *  (see point2d.cl).
 *
 *  `dc` points at the 70 contiguous coefficients of a flattened (7, 2, 5)
 *  solve2d_d tensor: parameter block m at dc + 10*m, ordered
 *  (tc, p, a, i, e, w, lan). Gradient outputs are private REAL[7] buffers;
 *  every slot is written, so the caller does not need to zero them.
 */


/* Position and its (tc, p, a, i, e, w, lan) derivatives at a centred time.

   Port of `meepmeep.numba2d.pos_cd`. */
MM_INLINE void pos_cd2(REAL t, MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc,
                    REAL *px, REAL *py, REAL *dpx, REAL *dpy) {
    pos_c2(t, c, px, py);
    for (int m = 0; m < MM_NPAR; m++) {
        MM_GLOBAL const REAL *r = dc + 10 * m;
        dpx[m] = taylor5(t, r);
        dpy[m] = taylor5(t, r + 5);
    }
}


/* Position and derivatives at an absolute time.

   Folds the time around the expansion point and adds the period-folding
   chain term: the folded time depends on p via -epoch*p, so the total
   period derivative (slot 1) gains epoch times the timing derivative
   (slot 0). Port of `meepmeep.numba2d.pos_d`. */
MM_INLINE void pos_d2(REAL t, REAL tc, REAL p,
                   MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc, REAL te,
                   REAL *px, REAL *py, REAL *dpx, REAL *dpy) {
    REAL epoch = floor((t - tc - te + (REAL)0.5 * p) / p);
    pos_cd2(t - (tc + te + epoch * p), c, dc, px, py, dpx, dpy);
    dpx[1] += epoch * dpx[0];
    dpy[1] += epoch * dpy[0];
}


/* Separation and its derivatives at a centred time.

   The position gradients are reduced with the chain rule
   dd/dtheta = (px*dpx + py*dpy) / d, singular only at an exact centre
   crossing (d = 0), as in the numba original. Port of
   `meepmeep.numba2d.sep_cd`. */
MM_INLINE REAL sep_cd2(REAL t, MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc,
                    REAL *dd) {
    REAL px, py;
    pos_c2(t, c, &px, &py);
    REAL d = sqrt(px * px + py * py);
    for (int m = 0; m < MM_NPAR; m++) {
        MM_GLOBAL const REAL *r = dc + 10 * m;
        REAL dpx = taylor5(t, r);
        REAL dpy = taylor5(t, r + 5);
        dd[m] = (px * dpx + py * dpy) / d;
    }
    return d;
}


/* Separation and derivatives at an absolute time.

   Port of `meepmeep.numba2d.sep_d`. */
MM_INLINE REAL sep_d2(REAL t, REAL tc, REAL p,
                   MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc, REAL te,
                   REAL *dd) {
    REAL epoch = floor((t - tc - te + (REAL)0.5 * p) / p);
    REAL d = sep_cd2(t - (tc + te + epoch * p), c, dc, dd);
    dd[1] += epoch * dd[0];
    return d;
}
