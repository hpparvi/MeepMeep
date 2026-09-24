/*  MeepMeep: fast orbit calculations for exoplanet modelling
 *  Copyright (C) 2022-2026 Hannu Parviainen
 *
 *  Shared code for the MeepMeep OpenCL backend and the C library.
 *
 *  Dual target. Every `.cl` file except `solve_kernels.cl` is written to
 *  compile both as OpenCL C (device functions prepended to a user kernel)
 *  and as plain C99 (the unity build in `c/src/meepmeep.c`). The two
 *  targets differ only through the macros defined at the top of this file:
 *
 *      MM_GLOBAL   `__global` on the device, empty in C
 *      MM_INLINE   `inline` on the device, empty in C (so the functions get
 *                  external linkage and become the C library's symbols)
 *      REAL        the floating-point type; `double` or `float` on the
 *                  device via a -DREAL= build option, always `double` in C
 *
 *  Keep the function bodies free of OpenCL-only builtins (`clamp`,
 *  `get_global_id`, address-space qualifiers, ...); only `solve_kernels.cl`
 *  may use them.
 *
 *  On the device, `REAL` is set to `double` or `float` by a -DREAL= build
 *  option, and USE_FP64 is defined alongside -DREAL=double (see
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

#ifdef __OPENCL_VERSION__
  /* OpenCL C: device functions inlined into the user's kernel. */
  #define MM_GLOBAL __global
  #define MM_INLINE inline
  #ifdef USE_FP64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
  #endif
#else
  /* Plain C99: the C library build. Fixed double precision. */
  #include <math.h>
  #ifndef REAL
    #define REAL double
  #endif
  #ifndef USE_FP64
    #define USE_FP64
  #endif
  #define MM_GLOBAL
  #define MM_INLINE
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
MM_INLINE REAL taylor5(REAL t, MM_GLOBAL const REAL *cf) {
    return cf[0] + t * (cf[1] + t * (cf[2] + t * (cf[3] + t * cf[4])));
}


/* Time derivative of `taylor5` over the same coefficient row.

   Mirrors the Horner form of `backends.numba.point3d.velocity._vel_c_s`. */
MM_INLINE REAL taylor5_dot(REAL t, MM_GLOBAL const REAL *cf) {
    return cf[1] + t * ((REAL)2.0 * cf[2] + t * ((REAL)3.0 * cf[3] + t * (REAL)4.0 * cf[4]));
}


/* Mean anomaly at the moment of primary transit.

   Port of `backends.numba.utils.mean_anomaly_at_transit`. */
MM_INLINE REAL mean_anomaly_at_transit(REAL ecc, REAL w) {
    REAL m = atan2(sqrt((REAL)1.0 - ecc * ecc) * sin(HALF_PI_R - w),
                   ecc + cos(HALF_PI_R - w));
    m -= ecc * sin(m);
    return m;
}


/* Mean anomaly at transit and its derivatives w.r.t. e and w.

   The value is returned; the derivatives are written into `dm_de` and
   `dm_dw`. Port of
   `backends.numba.utils.mean_anomaly_at_transit_with_derivatives`. */
MM_INLINE REAL mean_anomaly_at_transit_with_derivatives(REAL ecc, REAL w,
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


/* Python/NumPy float `%` semantics: the result takes the sign of the divisor,
   so it lands in [0, 2pi). C's fmod takes the sign of the dividend instead.
   The numba solvers wrap the mean anomaly with the NumPy convention.

   The distinction is unobservable through `ea_from_ma` alone, because
   E - e sin(E) = M is strictly monotonic, so root(M + 2pi) = root(M) + 2pi
   exactly and the solvers consume only sin(E) and cos(E). It is kept because
   that stops being true the moment anything reads E itself. */
MM_INLINE REAL mm_mod_two_pi(REAL x) {
    REAL r = fmod(x, TWO_PI_R);
    return (r < (REAL)0.0) ? r + TWO_PI_R : r;
}


/* Kepler-solver convergence tolerance.

   numba uses a literal 1e-13, which float32 cannot reach (eps ~ 1.2e-7): a
   verbatim port would run every fp32 work item to the 50-iteration cap, for
   roughly double the kernel time and no accuracy gain. Override with
   -DMM_EA_TOL=<value> to pin a specific tolerance. */
#ifndef MM_EA_TOL
  #ifdef USE_FP64
    #define MM_EA_TOL ((REAL)1e-13)
  #else
    #define MM_EA_TOL ((REAL)1e-6)
  #endif
#endif


/* Solve Kepler's equation E - e sin(E) = M for the eccentric anomaly.

   Port of `meepmeep.backends.numba.newton.newton.ea_from_ma`. The iteration
   count is data-dependent, so within a warp every lane pays for the slowest
   lane: a batch spanning a range of eccentricities costs more per parameter
   set than one sharing a single eccentricity. */
MM_INLINE REAL ea_from_ma(REAL ma, REAL ecc) {
    REAL ea = (ecc > (REAL)0.8) ? PI_R : ma;
    for (int it = 0; it < 50; ++it) {
        REAL f = ea - ecc * sin(ea) - ma;
        REAL df = (REAL)1.0 - ecc * cos(ea);
        REAL dea = -f / df;
        ea += dea;
        if (fabs(dea) < MM_EA_TOL) break;
    }
    return ea;
}
