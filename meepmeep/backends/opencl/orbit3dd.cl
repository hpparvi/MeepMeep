/*  MeepMeep: fast orbit calculations for exoplanet modelling
 *  Copyright (C) 2022-2026 Hannu Parviainen
 *
 *  Multi-expansion-point orbit-spanning gradient evaluators.
 *  Requires: common.cl, point3d.cl, point3dd.cl, orbit3d.cl
 *
 *  Device-function ports of the `meepmeep.numba3d` *_od evaluators. They
 *  add the `dcoeffs` argument - the flattened (npt, 7, 3, 5) stack from
 *  `solve3d_orbit_d`, expansion point ix at dcoeffs + 105*ix - and every
 *  epoch fold carries the period chain term d[1] += epoch * d[0] (the
 *  folded time depends on p via -epoch*p, so the total period derivative
 *  gains epoch times the timing derivative; zero at epoch 0, growing with
 *  the orbit count).
 *
 *  Gradient buffer lengths: 7 for the purely orbital quantities;
 *  rv_od REAL[8] (+k), lambert_phase_curve_od REAL[9] (+ag, +k),
 *  ev_signal_od REAL[9] (+alpha, +mass_ratio), emission_phase_curve_od
 *  REAL[10] (+k, +fratio, +offset). Every slot is written on every path.
 */


/* Position and its (tc, p, a, i, e, w, lan) derivatives at any phase.

   Port of `meepmeep.numba3d.pos_od`. */
MM_INLINE void pos_od(REAL t, REAL tpa, REAL p, REAL dt,
                   MM_GLOBAL const int *ep_table, MM_GLOBAL const REAL *ep_times,
                   MM_GLOBAL const REAL *coeffs, MM_GLOBAL const REAL *dcoeffs,
                   REAL *px, REAL *py, REAL *pz,
                   REAL *dpx, REAL *dpy, REAL *dpz) {
    REAL epoch = floor((t - tpa) / p);
    REAL tc = t - tpa - epoch * p;
    int ix = ep_lookup(tc, p, dt, ep_table);
    pos_cd3(tc - ep_times[ix] * p, coeffs + 15 * ix, dcoeffs + 105 * ix,
           px, py, pz, dpx, dpy, dpz);
    dpx[1] += epoch * dpx[0];
    dpy[1] += epoch * dpy[0];
    dpz[1] += epoch * dpz[0];
}


/* Line-of-sight z and derivatives at any phase.

   Port of `meepmeep.numba3d.zpos_od`. */
MM_INLINE REAL zpos_od(REAL t, REAL tpa, REAL p, REAL dt,
                    MM_GLOBAL const int *ep_table, MM_GLOBAL const REAL *ep_times,
                    MM_GLOBAL const REAL *coeffs, MM_GLOBAL const REAL *dcoeffs,
                    REAL *dz) {
    REAL epoch = floor((t - tpa) / p);
    REAL tc = t - tpa - epoch * p;
    int ix = ep_lookup(tc, p, dt, ep_table);
    REAL z = zpos_cd3(tc - ep_times[ix] * p, coeffs + 15 * ix, dcoeffs + 105 * ix, dz);
    dz[1] += epoch * dz[0];
    return z;
}


/* Sky-projected separation and derivatives at any phase.

   Port of `meepmeep.numba3d.sep_od`. */
MM_INLINE REAL sep_od(REAL t, REAL tpa, REAL p, REAL dt,
                   MM_GLOBAL const int *ep_table, MM_GLOBAL const REAL *ep_times,
                   MM_GLOBAL const REAL *coeffs, MM_GLOBAL const REAL *dcoeffs,
                   REAL *dd) {
    REAL epoch = floor((t - tpa) / p);
    REAL tc = t - tpa - epoch * p;
    int ix = ep_lookup(tc, p, dt, ep_table);
    REAL d = sep_cd3(tc - ep_times[ix] * p, coeffs + 15 * ix, dcoeffs + 105 * ix, dd);
    dd[1] += epoch * dd[0];
    return d;
}


/* Velocity and derivatives at any phase.

   Port of `meepmeep.numba3d.vel_od`. */
MM_INLINE void vel_od(REAL t, REAL tpa, REAL p, REAL dt,
                   MM_GLOBAL const int *ep_table, MM_GLOBAL const REAL *ep_times,
                   MM_GLOBAL const REAL *coeffs, MM_GLOBAL const REAL *dcoeffs,
                   REAL *vx, REAL *vy, REAL *vz,
                   REAL *dvx, REAL *dvy, REAL *dvz) {
    REAL epoch = floor((t - tpa) / p);
    REAL tc = t - tpa - epoch * p;
    int ix = ep_lookup(tc, p, dt, ep_table);
    vel_cd3(tc - ep_times[ix] * p, coeffs + 15 * ix, dcoeffs + 105 * ix,
           vx, vy, vz, dvx, dvy, dvz);
    dvx[1] += epoch * dvx[0];
    dvy[1] += epoch * dvy[0];
    dvz[1] += epoch * dvz[0];
}


/* Line-of-sight velocity and derivatives at any phase.

   Port of `meepmeep.numba3d.zvel_od`. */
MM_INLINE REAL zvel_od(REAL t, REAL tpa, REAL p, REAL dt,
                    MM_GLOBAL const int *ep_table, MM_GLOBAL const REAL *ep_times,
                    MM_GLOBAL const REAL *coeffs, MM_GLOBAL const REAL *dcoeffs,
                    REAL *dvz) {
    REAL epoch = floor((t - tpa) / p);
    REAL tc = t - tpa - epoch * p;
    int ix = ep_lookup(tc, p, dt, ep_table);
    REAL vz = zvel_cd3(tc - ep_times[ix] * p, coeffs + 15 * ix, dcoeffs + 105 * ix, dvz);
    dvz[1] += epoch * dvz[0];
    return vz;
}


/* Stellar radial velocity and derivatives at any phase.

   drv: REAL[8], ordered (tc, p, a, i, e, w, lan, k); slot 7 is
   d(rv)/dk = rv/k, zero when k is zero. Port of `meepmeep.numba3d.rv_od`. */
MM_INLINE REAL rv_od(REAL t, REAL k, REAL tpa, REAL p, REAL a, REAL i, REAL e,
                  REAL dt, MM_GLOBAL const int *ep_table,
                  MM_GLOBAL const REAL *ep_times,
                  MM_GLOBAL const REAL *coeffs, MM_GLOBAL const REAL *dcoeffs,
                  REAL *drv) {
    REAL dsp, dsa, dsi, dse;
    REAL s = rv_scale(k, p, a, i, e, &dsp, &dsa, &dsi, &dse);
    REAL epoch = floor((t - tpa) / p);
    REAL tc = t - tpa - epoch * p;
    int ix = ep_lookup(tc, p, dt, ep_table);
    REAL rv_val = rv_cd_w(tc - ep_times[ix] * p, s, dsp, dsa, dsi, dse,
                          coeffs + 15 * ix, dcoeffs + 105 * ix, drv);
    drv[1] += epoch * drv[0];
    drv[7] = k != (REAL)0.0 ? rv_val / k : (REAL)0.0;
    return rv_val;
}


/* Phase-angle cosine and derivatives at any phase.

   Port of `meepmeep.numba3d.cos_alpha_od`. */
MM_INLINE REAL cos_alpha_od(REAL t, REAL tpa, REAL p, REAL dt,
                         MM_GLOBAL const int *ep_table, MM_GLOBAL const REAL *ep_times,
                         MM_GLOBAL const REAL *coeffs, MM_GLOBAL const REAL *dcoeffs,
                         REAL *dca) {
    REAL epoch = floor((t - tpa) / p);
    REAL tc = t - tpa - epoch * p;
    int ix = ep_lookup(tc, p, dt, ep_table);
    REAL ca = cos_alpha_cd3(tc - ep_times[ix] * p, coeffs + 15 * ix, dcoeffs + 105 * ix, dca);
    dca[1] += epoch * dca[0];
    return ca;
}


/* Angle-to-fixed-vector cosine and derivatives at any phase.

   The numba original takes `v` as a 3-array; here it is three scalars
   (see cos_v_p_angle_o in orbit3d.cl). Port of
   `meepmeep.numba3d.cos_v_p_angle_od`. */
MM_INLINE REAL cos_v_p_angle_od(REAL vx, REAL vy, REAL vz,
                             REAL t, REAL tpa, REAL p, REAL dt,
                             MM_GLOBAL const int *ep_table, MM_GLOBAL const REAL *ep_times,
                             MM_GLOBAL const REAL *coeffs, MM_GLOBAL const REAL *dcoeffs,
                             REAL *dcs) {
    REAL inv_nv = (REAL)1.0 / sqrt(vx * vx + vy * vy + vz * vz);
    REAL x, y, z, dx[7], dy[7], dz[7];
    pos_od(t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs, &x, &y, &z, dx, dy, dz);
    REAL r2 = x * x + y * y + z * z;
    REAL r = sqrt(r2);
    REAL inv_r = (REAL)1.0 / r;
    REAL inv_r3 = inv_r / r2;
    REAL dot = x * vx + y * vy + z * vz;
    REAL cs = dot * inv_nv * inv_r;
    for (int m = 0; m < MM_NPAR; m++) {
        REAL ddot = dx[m] * vx + dy[m] * vy + dz[m] * vz;
        REAL xdotdx = x * dx[m] + y * dy[m] + z * dz[m];
        dcs[m] = inv_nv * (ddot * inv_r - dot * xdotdx * inv_r3);
    }
    return cs;
}


/* True anomaly and derivatives from the position and eccentricity vector.

   Strict math (the numba original deliberately drops fastmath: the acos
   argument sits near +-1 and the 1/sqrt(1 - edp^2) gradient denominator
   is near-singular). The gradient buffer is zeroed on entry because the
   early-return paths (the circular fast path leaves the a, i and lan slots
   zero; the edp clamps leave all slots zero) rely on it - the numba original
   allocates with zeros(7). `timing_is_tc` states the basis of dcoeffs; only
   the circular fast path, which does not read dcoeffs, uses it (see
   `_circular_w` in the numba module). Port of `meepmeep.numba3d.true_anomaly_od`. */
MM_INLINE REAL true_anomaly_od(REAL t, REAL tpa, REAL p,
                            REAL ex, REAL ey, REAL ez, REAL w,
                            REAL dt, MM_GLOBAL const int *ep_table,
                            MM_GLOBAL const REAL *ep_times,
                            MM_GLOBAL const REAL *coeffs, MM_GLOBAL const REAL *dcoeffs,
                            int timing_is_tc, REAL *df) {
    for (int m = 0; m < MM_NPAR; m++)
        df[m] = (REAL)0.0;
    REAL nes = ex * ex + ey * ey + ez * ez;

    if (ex <= (REAL)-0.9999 && nes > (REAL)0.99) {
        /* Circular-orbit fast path: f = 2 pi (t - tpa) / p folded. In the
           transit-centre basis tpa = tc - M_tr(e, w) p / (2 pi) moves with
           p, e and w too; the sentinel stands for e ~ 0, so M_tr is taken
           at e = 0. */
        REAL tau = t - tpa;
        REAL epoch = floor(tau / p);
        REAL tau_red = tau - epoch * p;
        REAL d0 = -TWO_PI_R / p;
        REAL d1 = -TWO_PI_R * tau_red / (p * p) + epoch * d0;
        df[0] = d0;
        if (timing_is_tc) {
            REAL dm_tr_de, dm_tr_dw;
            REAL m_tr = mean_anomaly_at_transit_with_derivatives((REAL)0.0, w, &dm_tr_de, &dm_tr_dw);
            df[1] = d1 - d0 * m_tr / TWO_PI_R;
            df[4] = -d0 * dm_tr_de * p / TWO_PI_R;
            df[5] = -d0 * dm_tr_dw * p / TWO_PI_R;
        } else {
            df[1] = d1;
        }
        return TWO_PI_R * tau_red / p;
    }

    REAL epoch = floor((t - tpa) / p);
    REAL tc = t - tpa - epoch * p;
    int ix = ep_lookup(tc, p, dt, ep_table);
    REAL x, y, z, dx[7], dy[7], dz[7];
    pos_cd3(tc - ep_times[ix] * p, coeffs + 15 * ix, dcoeffs + 105 * ix,
           &x, &y, &z, dx, dy, dz);

    REAL r2 = x * x + y * y + z * z;
    REAL sqrt_r2_nes = sqrt(r2 * nes);
    REAL edp = (x * ex + y * ey + z * ez) / sqrt_r2_nes;

    if (edp <= (REAL)-1.0)
        return PI_R;
    if (edp >= (REAL)1.0)
        return (REAL)0.0;

    /* Branch selection from the mean anomaly: f and M share the
       half-plane, and M = 2 pi tc / p is exact here. */
    REAL sign = tc < (REAL)0.5 * p ? (REAL)1.0 : (REAL)-1.0;
    REAL base = acos(edp);
    REAL f = sign > (REAL)0.0 ? base : TWO_PI_R - base;
    REAL denom = sqrt((REAL)1.0 - edp * edp);
    REAL xdote = x * ex + y * ey + z * ez;
    for (int m = 0; m < MM_NPAR; m++) {
        REAL dxdote = dx[m] * ex + dy[m] * ey + dz[m] * ez;
        REAL xdotdx = x * dx[m] + y * dy[m] + z * dz[m];
        REAL dedp = dxdote / sqrt_r2_nes - xdote * xdotdx / (r2 * sqrt_r2_nes);
        REAL df_m = -dedp / denom;
        df[m] = sign > (REAL)0.0 ? df_m : -df_m;
    }
    df[1] += epoch * df[0];
    return f;
}


/* Lambertian phase curve and derivatives at any phase.

   dflux: REAL[9]. Port of `meepmeep.numba3d.lambert_phase_curve_od`. */
MM_INLINE REAL lambert_phase_curve_od(REAL t, REAL ag, REAL k, REAL tpa, REAL p,
                                   REAL dt, MM_GLOBAL const int *ep_table,
                                   MM_GLOBAL const REAL *ep_times,
                                   MM_GLOBAL const REAL *coeffs,
                                   MM_GLOBAL const REAL *dcoeffs,
                                   REAL *dflux) {
    REAL epoch = floor((t - tpa) / p);
    REAL tc = t - tpa - epoch * p;
    int ix = ep_lookup(tc, p, dt, ep_table);
    REAL flux = lambert_phase_curve_cd3(tc - ep_times[ix] * p, ag, k,
                                       coeffs + 15 * ix, dcoeffs + 105 * ix, dflux);
    dflux[1] += epoch * dflux[0];
    return flux;
}


/* Ellipsoidal-variation signal and derivatives at any phase.

   dout: REAL[9]. Port of `meepmeep.numba3d.ev_signal_od`. */
MM_INLINE REAL ev_signal_od(REAL alpha, REAL mass_ratio, REAL inc,
                         REAL t, REAL tpa, REAL p, REAL dt,
                         MM_GLOBAL const int *ep_table, MM_GLOBAL const REAL *ep_times,
                         MM_GLOBAL const REAL *coeffs, MM_GLOBAL const REAL *dcoeffs,
                         REAL *dout) {
    REAL epoch = floor((t - tpa) / p);
    REAL tc = t - tpa - epoch * p;
    int ix = ep_lookup(tc, p, dt, ep_table);
    REAL out = ev_signal_cd3(tc - ep_times[ix] * p, alpha, mass_ratio, inc,
                            coeffs + 15 * ix, dcoeffs + 105 * ix, dout);
    dout[1] += epoch * dout[0];
    return out;
}


/* Thermal-emission phase curve and derivatives at any phase.

   dout: REAL[10]. Port of `meepmeep.numba3d.emission_phase_curve_od`. */
MM_INLINE REAL emission_phase_curve_od(REAL t, REAL k, REAL fratio, REAL offset,
                                    REAL tpa, REAL p, REAL dt,
                                    MM_GLOBAL const int *ep_table,
                                    MM_GLOBAL const REAL *ep_times,
                                    MM_GLOBAL const REAL *coeffs,
                                    MM_GLOBAL const REAL *dcoeffs,
                                    REAL *dout) {
    REAL epoch = floor((t - tpa) / p);
    REAL tc = t - tpa - epoch * p;
    int ix = ep_lookup(tc, p, dt, ep_table);
    REAL flux = emission_phase_curve_cd3(tc - ep_times[ix] * p, k, fratio, offset,
                                        coeffs + 15 * ix, dcoeffs + 105 * ix, dout);
    dout[1] += epoch * dout[0];
    return flux;
}


/* Three-dimensional star-planet distance and derivatives at any phase.

   Port of `meepmeep.numba3d.star_planet_distance_od`. */
MM_INLINE REAL star_planet_distance_od(REAL t, REAL tpa, REAL p, REAL dt,
                                    MM_GLOBAL const int *ep_table,
                                    MM_GLOBAL const REAL *ep_times,
                                    MM_GLOBAL const REAL *coeffs,
                                    MM_GLOBAL const REAL *dcoeffs,
                                    REAL *dr) {
    REAL x, y, z, dx[7], dy[7], dz[7];
    pos_od(t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs, &x, &y, &z, dx, dy, dz);
    REAL r = sqrt(x * x + y * y + z * z);
    REAL inv_r = (REAL)1.0 / r;
    for (int m = 0; m < MM_NPAR; m++)
        dr[m] = (x * dx[m] + y * dy[m] + z * dz[m]) * inv_r;
    return r;
}


/* Line-of-sight z at the transit event and its total derivative.

   The total derivative of z(t_transit(theta); theta) combines the
   fixed-time gradient with v_z(t_transit) * dt_transit/dtheta, where
   dt_transit/dtheta depends on the bound timing basis: with the transit
   centre bound (timing_is_tc = 1, dcoeffs after tp_to_tc_gradient_orbit)
   only the timing slot is non-zero; with the periastron time bound
   (timing_is_tc = 0, the native solve3d_orbit_d basis) the p, e,
   and w slots join through t_o = M_tr(e, w) p / (2 pi). dz_tr: REAL[7].
   Port of the numba helper `_ltt_transit_z_and_d`. */
MM_INLINE REAL ltt_transit_z_and_d(REAL tpa, REAL p, REAL e, REAL w, REAL dt,
                                MM_GLOBAL const int *ep_table,
                                MM_GLOBAL const REAL *ep_times,
                                MM_GLOBAL const REAL *coeffs,
                                MM_GLOBAL const REAL *dcoeffs,
                                int timing_is_tc, REAL *dz_tr) {
    REAL dm_tr_de, dm_tr_dw;
    REAL m_tr = mean_anomaly_at_transit_with_derivatives(e, w, &dm_tr_de, &dm_tr_dw);
    REAL to = m_tr / TWO_PI_R * p;
    REAL t_transit = tpa + to;

    REAL z_tr = zpos_od(t_transit, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs, dz_tr);
    REAL vz_tr = zvel_o(t_transit, tpa, p, dt, ep_table, ep_times, coeffs);

    /* dt_transit/dtheta in the bound timing basis; adding vz * dttr makes
       the timing slot of the total cancel to zero: z at the transit event
       does not depend on when the transit happens. */
    dz_tr[0] += vz_tr;
    if (!timing_is_tc) {
        dz_tr[1] += vz_tr * m_tr / TWO_PI_R;
        dz_tr[4] += vz_tr * dm_tr_de * p / TWO_PI_R;
        dz_tr[5] += vz_tr * dm_tr_dw * p / TWO_PI_R;
    }
    return z_tr;
}


/* Light-travel-time correction and derivatives at any phase.

   dltt: REAL[7] (no rstar slot, matching numba). `timing_is_tc` selects
   the timing basis of `dcoeffs` (see ltt_transit_z_and_d); the numba
   default is True (pass 1). A kernel evaluating many times may hoist
   ltt_transit_z_and_d host-side or into a pre-pass, as the numba vector
   kernels do - that is why the helper is public. Port of
   `meepmeep.numba3d.light_travel_time_od`. */
MM_INLINE REAL light_travel_time_od(REAL t, REAL tpa, REAL p, REAL e, REAL w,
                                 REAL rstar, REAL dt,
                                 MM_GLOBAL const int *ep_table,
                                 MM_GLOBAL const REAL *ep_times,
                                 MM_GLOBAL const REAL *coeffs,
                                 MM_GLOBAL const REAL *dcoeffs,
                                 int timing_is_tc, REAL *dltt) {
    REAL dz_t[7], dz_tr[7];
    REAL z_t = zpos_od(t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs, dz_t);
    REAL z_tr = ltt_transit_z_and_d(tpa, p, e, w, dt, ep_table, ep_times,
                                    coeffs, dcoeffs, timing_is_tc, dz_tr);
    REAL factor = -rstar * LTT_DAYS_PER_RSUN;
    for (int m = 0; m < MM_NPAR; m++)
        dltt[m] = factor * (dz_t[m] - dz_tr[m]);
    return factor * (z_t - z_tr);
}
