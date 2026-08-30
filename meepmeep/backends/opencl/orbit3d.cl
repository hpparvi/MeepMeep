/*  MeepMeep: fast orbit calculations for exoplanet modelling
 *  Copyright (C) 2022-2026 Hannu Parviainen
 *
 *  Multi-expansion-point orbit-spanning evaluators (values only).
 *  Requires: common.cl, point3d.cl
 *
 *  Device-function ports of the `meepmeep.numba3d` *_o evaluators. Common
 *  trailing arguments:
 *
 *      tpa       periastron time (the high-level Orbit API converts tc to
 *                tpa before calling; note the anchor differs from the
 *                transit-centred point3d evaluators)
 *      p         orbital period
 *      dt        ep_table bucket width as a fraction of the period
 *      ep_table  int32 time-to-expansion-point lookup table (the numba
 *                backend builds it as int64; the host must cast)
 *      ep_times  (npt,) normalised expansion-point phases in [0, 1]
 *      coeffs    flattened (npt, 3, 5) solve3d_orbit stack, expansion
 *                point ix at coeffs + 15*ix
 */


/* Expansion-point lookup for an already-folded time tc in [0, p).

   Two guards absent from the numba original, which relies on in-range
   float-to-int behaviour a device cannot: (int)floor(NAN) is INT_MIN on
   NVIDIA, and an out-of-bounds __global read can kill the shared context
   for the whole process, so NaN returns index 0 (the following Horner
   evaluation then yields NaN gracefully, matching numba); and fp rounding
   can put tc exactly at p, so the bucket is clamped to the table length
   tres = 1/dt. */
inline int ep_lookup(REAL tc, REAL p, REAL dt, __global const int *ep_table) {
    if (isnan(tc))
        return 0;
    int nb = (int)((REAL)1.0 / dt + (REAL)0.5);
    int b = clamp((int)floor(tc / (dt * p)), 0, nb - 1);
    return ep_table[b];
}


/* Expansion-point index for an absolute time.

   Port of `backends.numba.orbit3d._common.ep_ix`. */
inline int ep_ix(REAL t, REAL tpa, REAL p, REAL dt, __global const int *ep_table) {
    REAL epoch = floor((t - tpa) / p);
    return ep_lookup(t - tpa - epoch * p, p, dt, ep_table);
}


/* Planet (x, y, z) position at any orbital phase.

   Port of `meepmeep.numba3d.pos_o`. */
inline void pos_o(REAL t, REAL tpa, REAL p, REAL dt,
                  __global const int *ep_table, __global const REAL *ep_times,
                  __global const REAL *coeffs,
                  REAL *px, REAL *py, REAL *pz) {
    REAL epoch = floor((t - tpa) / p);
    REAL tc = t - tpa - epoch * p;
    int ix = ep_lookup(tc, p, dt, ep_table);
    pos_c3(tc - ep_times[ix] * p, coeffs + 15 * ix, px, py, pz);
}


/* Line-of-sight z coordinate at any orbital phase.

   Port of `meepmeep.numba3d.zpos_o`. */
inline REAL zpos_o(REAL t, REAL tpa, REAL p, REAL dt,
                   __global const int *ep_table, __global const REAL *ep_times,
                   __global const REAL *coeffs) {
    REAL epoch = floor((t - tpa) / p);
    REAL tc = t - tpa - epoch * p;
    int ix = ep_lookup(tc, p, dt, ep_table);
    return zpos_c3(tc - ep_times[ix] * p, coeffs + 15 * ix);
}


/* Sky-projected planet-star separation at any orbital phase.

   Port of `meepmeep.numba3d.sep_o`. */
inline REAL sep_o(REAL t, REAL tpa, REAL p, REAL dt,
                  __global const int *ep_table, __global const REAL *ep_times,
                  __global const REAL *coeffs) {
    REAL epoch = floor((t - tpa) / p);
    REAL tc = t - tpa - epoch * p;
    int ix = ep_lookup(tc, p, dt, ep_table);
    return sep_c3(tc - ep_times[ix] * p, coeffs + 15 * ix);
}


/* Planet (vx, vy, vz) velocity at any orbital phase.

   Port of `meepmeep.numba3d.vel_o`. */
inline void vel_o(REAL t, REAL tpa, REAL p, REAL dt,
                  __global const int *ep_table, __global const REAL *ep_times,
                  __global const REAL *coeffs,
                  REAL *vx, REAL *vy, REAL *vz) {
    REAL epoch = floor((t - tpa) / p);
    REAL tc = t - tpa - epoch * p;
    int ix = ep_lookup(tc, p, dt, ep_table);
    vel_c3(tc - ep_times[ix] * p, coeffs + 15 * ix, vx, vy, vz);
}


/* Line-of-sight velocity at any orbital phase.

   Port of `meepmeep.numba3d.zvel_o`. */
inline REAL zvel_o(REAL t, REAL tpa, REAL p, REAL dt,
                   __global const int *ep_table, __global const REAL *ep_times,
                   __global const REAL *coeffs) {
    REAL epoch = floor((t - tpa) / p);
    REAL tc = t - tpa - epoch * p;
    int ix = ep_lookup(tc, p, dt, ep_table);
    return zvel_c3(tc - ep_times[ix] * p, coeffs + 15 * ix);
}


/* Stellar radial velocity at any orbital phase.

   Port of `meepmeep.numba3d.rv_o`. */
inline REAL rv_o(REAL t, REAL k, REAL tpa, REAL p, REAL a, REAL i, REAL e,
                 REAL dt, __global const int *ep_table,
                 __global const REAL *ep_times, __global const REAL *coeffs) {
    REAL scale = k / (TWO_PI_R / p * (a * sin(i)) / sqrt((REAL)1.0 - e * e));
    return zvel_o(t, tpa, p, dt, ep_table, ep_times, coeffs) * scale;
}


/* Cosine of the star-planet-observer phase angle at any orbital phase.

   Port of `meepmeep.numba3d.cos_alpha_o`. */
inline REAL cos_alpha_o(REAL t, REAL tpa, REAL p, REAL dt,
                        __global const int *ep_table, __global const REAL *ep_times,
                        __global const REAL *coeffs) {
    REAL epoch = floor((t - tpa) / p);
    REAL tc = t - tpa - epoch * p;
    int ix = ep_lookup(tc, p, dt, ep_table);
    return cos_alpha_c3(tc - ep_times[ix] * p, coeffs + 15 * ix);
}


/* Cosine of the angle between the planet position and a fixed vector.

   The numba original takes `v` as a 3-array; here it is three scalars
   (vx, vy, vz) so callers holding the vector in any address space can pass
   it without qualifier friction. Port of `meepmeep.numba3d.cos_v_p_angle_o`. */
inline REAL cos_v_p_angle_o(REAL vx, REAL vy, REAL vz,
                            REAL t, REAL tpa, REAL p, REAL dt,
                            __global const int *ep_table, __global const REAL *ep_times,
                            __global const REAL *coeffs) {
    REAL inv_nv = (REAL)1.0 / sqrt(vx * vx + vy * vy + vz * vz);
    REAL x, y, z;
    pos_o(t, tpa, p, dt, ep_table, ep_times, coeffs, &x, &y, &z);
    return (x * vx + y * vy + z * vz) * inv_nv / sqrt(x * x + y * y + z * z);
}


/* True anomaly from the position and the eccentricity vector (ex, ey, ez).

   The numba original is deliberately compiled without fastmath (the
   arccos argument sits near +-1 over much of the orbit); OpenCL strict
   math matches that. The early-return clamps are ported exactly. `w` is
   kept for signature parity with the numba dispatcher. Port of
   `meepmeep.numba3d.true_anomaly_o`. */
inline REAL true_anomaly_o(REAL t, REAL tpa, REAL p,
                           REAL ex, REAL ey, REAL ez, REAL w,
                           REAL dt, __global const int *ep_table,
                           __global const REAL *ep_times, __global const REAL *coeffs) {
    REAL nes = ex * ex + ey * ey + ez * ez;

    if (ex <= (REAL)-0.9999 && nes > (REAL)0.99) {
        /* Circular-orbit fast path: with the periastron anchor tpa, the
           true anomaly equals the mean anomaly folded into [0, 2 pi). */
        REAL tau = t - tpa;
        REAL epoch = floor(tau / p);
        return TWO_PI_R * (tau - epoch * p) / p;
    }

    REAL epoch = floor((t - tpa) / p);
    REAL tc = t - tpa - epoch * p;
    int ix = ep_lookup(tc, p, dt, ep_table);
    REAL x, y, z;
    pos_c3(tc - ep_times[ix] * p, coeffs + 15 * ix, &x, &y, &z);
    REAL edp = (x * ex + y * ey + z * ez) / sqrt((x * x + y * y + z * z) * nes);

    if (edp <= (REAL)-1.0)
        return PI_R;
    else if (edp >= (REAL)1.0)
        return (REAL)0.0;
    else if (tc < (REAL)0.5 * p)
        /* Branch selection from the mean anomaly: f and M share the
           half-plane, and M = 2 pi tc / p is exact here. */
        return acos(edp);
    else
        return TWO_PI_R - acos(edp);
}


/* Lambertian reflected-light phase curve at any orbital phase.

   Port of `meepmeep.numba3d.lambert_phase_curve_o`. */
inline REAL lambert_phase_curve_o(REAL t, REAL ag, REAL k, REAL tpa, REAL p,
                                  REAL dt, __global const int *ep_table,
                                  __global const REAL *ep_times,
                                  __global const REAL *coeffs) {
    REAL epoch = floor((t - tpa) / p);
    REAL tc = t - tpa - epoch * p;
    int ix = ep_lookup(tc, p, dt, ep_table);
    return lambert_phase_curve_c3(tc - ep_times[ix] * p, ag, k, coeffs + 15 * ix);
}


/* Ellipsoidal-variation signal at any orbital phase.

   Port of `meepmeep.numba3d.ev_signal_o`. */
inline REAL ev_signal_o(REAL alpha, REAL mass_ratio, REAL inc,
                        REAL t, REAL tpa, REAL p, REAL dt,
                        __global const int *ep_table, __global const REAL *ep_times,
                        __global const REAL *coeffs) {
    REAL epoch = floor((t - tpa) / p);
    REAL tc = t - tpa - epoch * p;
    int ix = ep_lookup(tc, p, dt, ep_table);
    return ev_signal_c3(tc - ep_times[ix] * p, alpha, mass_ratio, inc, coeffs + 15 * ix);
}


/* Thermal-emission phase curve at any orbital phase.

   Port of `meepmeep.numba3d.emission_phase_curve_o`. */
inline REAL emission_phase_curve_o(REAL t, REAL k, REAL fratio, REAL offset,
                                   REAL tpa, REAL p, REAL dt,
                                   __global const int *ep_table,
                                   __global const REAL *ep_times,
                                   __global const REAL *coeffs) {
    REAL epoch = floor((t - tpa) / p);
    REAL tc = t - tpa - epoch * p;
    int ix = ep_lookup(tc, p, dt, ep_table);
    return emission_phase_curve_c3(tc - ep_times[ix] * p, k, fratio, offset, coeffs + 15 * ix);
}


/* Three-dimensional star-planet distance at any orbital phase.

   Port of `meepmeep.numba3d.star_planet_distance_o`. */
inline REAL star_planet_distance_o(REAL t, REAL tpa, REAL p, REAL dt,
                                   __global const int *ep_table,
                                   __global const REAL *ep_times,
                                   __global const REAL *coeffs) {
    REAL x, y, z;
    pos_o(t, tpa, p, dt, ep_table, ep_times, coeffs, &x, &y, &z);
    return sqrt(x * x + y * y + z * z);
}


/* Light-travel-time correction referenced to the primary transit.

   Positive when the signal from phase t arrives later than a transit-
   referenced clock expects. A kernel evaluating many times may hoist the
   transit-reference term z_tr = zpos_o(tpa + to, ...) host-side or into a
   pre-pass; this scalar port recomputes it per call, matching the numba
   scalar kernel. Port of `meepmeep.numba3d.light_travel_time_o`. */
inline REAL light_travel_time_o(REAL t, REAL tpa, REAL p, REAL e, REAL w,
                                REAL rstar, REAL dt,
                                __global const int *ep_table,
                                __global const REAL *ep_times,
                                __global const REAL *coeffs) {
    REAL to = mean_anomaly_at_transit(e, w) / TWO_PI_R * p;
    REAL z_t = zpos_o(t, tpa, p, dt, ep_table, ep_times, coeffs);
    REAL z_tr = zpos_o(tpa + to, tpa, p, dt, ep_table, ep_times, coeffs);
    return -(z_t - z_tr) * rstar * LTT_DAYS_PER_RSUN;
}
