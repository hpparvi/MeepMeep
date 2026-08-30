/*  MeepMeep: fast orbit calculations for exoplanet modelling
 *  Copyright (C) 2022-2026 Hannu Parviainen
 *
 *  Shared device code for the MeepMeep OpenCL backend.
 *
 *  `REAL` is set to `double` or `float` by a -DREAL= build option, and
 *  USE_FP64 is defined alongside -DREAL=double (see
 *  `meepmeep.backends.opencl.build_options`). Every floating-point literal is
 *  cast to REAL so the single-precision build does not silently promote.
 *
 *  Array layout. All coefficient arrays are C-contiguous flattenings of the
 *  NumPy arrays produced by the host-side solvers:
 *
 *      c       (2, 5)         row d at c + 5*d            (solve2d)
 *      c       (3, 5)         row d at c + 5*d            (solve3d)
 *      dc      (7, 2, 5)      parameter block m at dc + 10*m   (solve2d_d)
 *      dc      (7, 3, 5)      parameter block m at dc + 15*m   (solve3d_d)
 *      coeffs  (npt, 3, 5)    expansion point ix at coeffs + 15*ix
 *      dcoeffs (npt, 7, 3, 5) expansion point ix at dcoeffs + 105*ix
 *
 *  Gradients follow the seven-parameter convention (tc, p, a, i, e, w, lan);
 *  functions with extra physical inputs append their derivatives after the
 *  orbital block in argument order.
 *
 *  Single-precision time origin: a float32 ulp at BJD ~2.4e6 is ~0.25 days,
 *  so every function that consumes absolute times (the epoch-folding direct
 *  and orbit-spanning evaluators) requires the host to subtract a float64
 *  reference epoch from the times (and from tc/tpa) before casting to
 *  float32.
 */

#ifdef USE_FP64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

#define PI_R ((REAL)3.14159265358979323846)
#define TWO_PI_R ((REAL)6.28318530717958647693)
#define HALF_PI_R ((REAL)1.57079632679489661923)

/* Number of orbital parameters carried through every gradient:
   (tc, p, a, i, e, w, lan). */
#define MM_NPAR 7

/* Light travel time across one solar radius in days. Keep in sync with
   `backends.numba.orbit3d.light_travel_time.LTT_DAYS_PER_RSUN`. */
#define LTT_DAYS_PER_RSUN ((REAL)2.685885891543453e-05)


/* Evaluate a 5th-order Taylor polynomial with Horner's scheme.

   `cf` points at five contiguous coefficients ordered [position, velocity,
   acceleration/2, jerk/6, snap/24] (pre-scaled by the factorial, so this is
   a plain polynomial evaluation). One row of a solve2d/solve3d matrix. */
inline REAL taylor5(REAL t, __global const REAL *cf) {
    return cf[0] + t * (cf[1] + t * (cf[2] + t * (cf[3] + t * cf[4])));
}


/* Time derivative of `taylor5` over the same coefficient row.

   Mirrors the Horner form of `backends.numba.point3d.velocity._vel_c_s`. */
inline REAL taylor5_dot(REAL t, __global const REAL *cf) {
    return cf[1] + t * ((REAL)2.0 * cf[2] + t * ((REAL)3.0 * cf[3] + t * (REAL)4.0 * cf[4]));
}


/* Mean anomaly at the moment of primary transit.

   Port of `backends.numba.utils.mean_anomaly_at_transit`. */
inline REAL mean_anomaly_at_transit(REAL ecc, REAL w) {
    REAL m = atan2(sqrt((REAL)1.0 - ecc * ecc) * sin(HALF_PI_R - w),
                   ecc + cos(HALF_PI_R - w));
    m -= ecc * sin(m);
    return m;
}


/* Mean anomaly at transit and its derivatives w.r.t. e and w.

   The value is returned; the derivatives are written into `dm_de` and
   `dm_dw`. Port of
   `backends.numba.utils.mean_anomaly_at_transit_with_derivatives`. */
inline REAL mean_anomaly_at_transit_with_derivatives(REAL ecc, REAL w,
                                                     REAL *dm_de, REAL *dm_dw) {
    REAL sqe2 = sqrt((REAL)1.0 - ecc * ecc);
    REAL cw = cos(w);
    REAL sw = sin(w);
    REAL y_e = sqe2 * cw;
    REAL x_e = ecc + sw;
    REAL e_off = atan2(y_e, x_e);
    REAL se = sin(e_off);
    REAL ce = cos(e_off);
    REAL m_at_transit = e_off - ecc * se;
    REAL denom = x_e * x_e + y_e * y_e;
    REAL de_off_de = (x_e * (-ecc / sqe2) * cw - y_e) / denom;
    REAL de_off_dw = (x_e * (-sqe2 * sw) - y_e * cw) / denom;
    *dm_de = de_off_de * ((REAL)1.0 - ecc * ce) - se;
    *dm_dw = de_off_dw * ((REAL)1.0 - ecc * ce);
    return m_at_transit;
}
