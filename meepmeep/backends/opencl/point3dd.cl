/*  MeepMeep: fast orbit calculations for exoplanet modelling
 *  Copyright (C) 2022-2026 Hannu Parviainen
 *
 *  Single-expansion-point 3D evaluators with orbital-parameter derivatives.
 *  Requires: common.cl, point3d.cl
 *
 *  Device-function ports of the `meepmeep.numba3d` *_cd / *_d evaluators,
 *  with the trailing `3` dimension digit (see point3d.cl; the helpers
 *  `rv_scale`, `rv_cd_w`, and `lambert_kernel_d` are unsuffixed).
 *  `dc` points at the 105 contiguous coefficients of a flattened (7, 3, 5)
 *  solve3d_d tensor: parameter block m at dc + 15*m, ordered
 *  (tc, p, a, i, e, w, lan).
 *
 *  Gradient outputs are private REAL buffers sized 7 for the purely orbital
 *  quantities; quantities with extra physical inputs append those
 *  derivatives after the orbital block in argument order:
 *
 *      lambert_phase_curve_cd3/_d3   REAL[9]   (..., lan, ag, k)
 *      ev_signal_cd3/_d3             REAL[9]   (..., lan, alpha, mass_ratio)
 *      emission_phase_curve_cd3/_d3  REAL[10]  (..., lan, k, fratio, offset)
 *
 *  Every slot is written on every path, so callers do not need to zero the
 *  buffers. The numba write-into kernels take caller-provided scratch for
 *  intermediate gradients; on a device, private arrays are register-backed,
 *  so the scratch is local here and never appears in a signature.
 */


/* Position and its (tc, p, a, i, e, w, lan) derivatives at a centred time.

   dpx, dpy, dpz: REAL[7] output buffers. Port of `meepmeep.numba3d.pos_cd`. */
MM_INLINE void pos_cd3(REAL t, MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc,
                    REAL *px, REAL *py, REAL *pz,
                    REAL *dpx, REAL *dpy, REAL *dpz) {
    pos_c3(t, c, px, py, pz);
    for (int m = 0; m < MM_NPAR; m++) {
        MM_GLOBAL const REAL *r = dc + 15 * m;
        dpx[m] = taylor5(t, r);
        dpy[m] = taylor5(t, r + 5);
        dpz[m] = taylor5(t, r + 10);
    }
}


/* Position and derivatives at an absolute time.

   Adds the period-folding chain term: the folded time depends on p via
   -epoch*p, so the total period derivative (slot 1) gains epoch times the
   timing derivative (slot 0). Port of `meepmeep.numba3d.pos_d`. */
MM_INLINE void pos_d3(REAL t, REAL tc, REAL p,
                   MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc, REAL te,
                   REAL *px, REAL *py, REAL *pz,
                   REAL *dpx, REAL *dpy, REAL *dpz) {
    REAL epoch = floor((t - tc - te + (REAL)0.5 * p) / p);
    pos_cd3(t - (tc + te + epoch * p), c, dc, px, py, pz, dpx, dpy, dpz);
    dpx[1] += epoch * dpx[0];
    dpy[1] += epoch * dpy[0];
    dpz[1] += epoch * dpz[0];
}


/* Line-of-sight z and its derivatives at a centred time.

   dpz: REAL[7]. Port of `meepmeep.numba3d.zpos_cd`. */
MM_INLINE REAL zpos_cd3(REAL t, MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc,
                     REAL *dpz) {
    for (int m = 0; m < MM_NPAR; m++)
        dpz[m] = taylor5(t, dc + 15 * m + 10);
    return taylor5(t, c + 10);
}


/* Line-of-sight z and derivatives at an absolute time.

   Port of `meepmeep.numba3d.zpos_d`. */
MM_INLINE REAL zpos_d3(REAL t, REAL tc, REAL p,
                    MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc, REAL te,
                    REAL *dpz) {
    REAL epoch = floor((t - tc - te + (REAL)0.5 * p) / p);
    REAL pz = zpos_cd3(t - (tc + te + epoch * p), c, dc, dpz);
    dpz[1] += epoch * dpz[0];
    return pz;
}


/* Sky-projected separation and its derivatives at a centred time.

   dd: REAL[7]. Chain rule dd/dtheta = (px*dpx + py*dpy) / d, singular only
   at an exact centre crossing (d = 0), as in the numba original. Port of
   `meepmeep.numba3d.sep_cd`. */
MM_INLINE REAL sep_cd3(REAL t, MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc,
                    REAL *dd) {
    REAL px = taylor5(t, c);
    REAL py = taylor5(t, c + 5);
    REAL d = sqrt(px * px + py * py);
    for (int m = 0; m < MM_NPAR; m++) {
        MM_GLOBAL const REAL *r = dc + 15 * m;
        REAL dpx = taylor5(t, r);
        REAL dpy = taylor5(t, r + 5);
        dd[m] = (px * dpx + py * dpy) / d;
    }
    return d;
}


/* Sky-projected separation and derivatives at an absolute time.

   Port of `meepmeep.numba3d.sep_d`. */
MM_INLINE REAL sep_d3(REAL t, REAL tc, REAL p,
                   MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc, REAL te,
                   REAL *dd) {
    REAL epoch = floor((t - tc - te + (REAL)0.5 * p) / p);
    REAL d = sep_cd3(t - (tc + te + epoch * p), c, dc, dd);
    dd[1] += epoch * dd[0];
    return d;
}


/* Velocity and its derivatives at a centred time.

   dvx, dvy, dvz: REAL[7]. Port of `meepmeep.numba3d.vel_cd`. */
MM_INLINE void vel_cd3(REAL t, MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc,
                    REAL *vx, REAL *vy, REAL *vz,
                    REAL *dvx, REAL *dvy, REAL *dvz) {
    vel_c3(t, c, vx, vy, vz);
    for (int m = 0; m < MM_NPAR; m++) {
        MM_GLOBAL const REAL *r = dc + 15 * m;
        dvx[m] = taylor5_dot(t, r);
        dvy[m] = taylor5_dot(t, r + 5);
        dvz[m] = taylor5_dot(t, r + 10);
    }
}


/* Velocity and derivatives at an absolute time.

   Port of `meepmeep.numba3d.vel_d`. */
MM_INLINE void vel_d3(REAL t, REAL tc, REAL p,
                   MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc, REAL te,
                   REAL *vx, REAL *vy, REAL *vz,
                   REAL *dvx, REAL *dvy, REAL *dvz) {
    REAL epoch = floor((t - tc - te + (REAL)0.5 * p) / p);
    vel_cd3(t - (tc + te + epoch * p), c, dc, vx, vy, vz, dvx, dvy, dvz);
    dvx[1] += epoch * dvx[0];
    dvy[1] += epoch * dvy[0];
    dvz[1] += epoch * dvz[0];
}


/* Line-of-sight velocity and its derivatives at a centred time.

   dvz: REAL[7]. Port of `meepmeep.numba3d.zvel_cd`. */
MM_INLINE REAL zvel_cd3(REAL t, MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc,
                     REAL *dvz) {
    for (int m = 0; m < MM_NPAR; m++)
        dvz[m] = taylor5_dot(t, dc + 15 * m + 10);
    return taylor5_dot(t, c + 10);
}


/* Line-of-sight velocity and derivatives at an absolute time.

   Port of `meepmeep.numba3d.zvel_d`. */
MM_INLINE REAL zvel_d3(REAL t, REAL tc, REAL p,
                    MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc, REAL te,
                    REAL *dvz) {
    REAL epoch = floor((t - tc - te + (REAL)0.5 * p) / p);
    REAL vz = zvel_cd3(t - (tc + te + epoch * p), c, dc, dvz);
    dvz[1] += epoch * dvz[0];
    return vz;
}


/* Radial-velocity scale factor s = k/n and its non-zero derivatives.

   Returns s and writes ds/dp, ds/da, ds/di, ds/de; the derivatives w.r.t.
   tc, w, and lan are identically zero. Hoist this out of per-sample loops:
   it depends only on the orbital parameters. Port of the numba helper
   `_rv_scale`. */
MM_INLINE REAL rv_scale(REAL k, REAL p, REAL a, REAL i, REAL e,
                     REAL *dsp, REAL *dsa, REAL *dsi, REAL *dse) {
    REAL n = TWO_PI_R / p * (a * sin(i)) / sqrt((REAL)1.0 - e * e);
    REAL s = k / n;
    *dsp = s / p;
    *dsa = -s / a;
    *dsi = -s * cos(i) / sin(i);
    *dse = -s * e / ((REAL)1.0 - e * e);
    return s;
}


/* Radial velocity and derivatives at a centred time, hoisted-scale form.

   Takes the precomputed scale factor and its derivatives from `rv_scale`
   so a kernel looping over many samples per work item computes them once.
   drv: REAL[7]. Port of the numba helper `_rv_cd_w`. */
MM_INLINE REAL rv_cd_w(REAL t, REAL s, REAL dsp, REAL dsa, REAL dsi, REAL dse,
                    MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc,
                    REAL *drv) {
    REAL dvz[7];
    REAL vz = zvel_cd3(t, c, dc, dvz);
    for (int m = 0; m < MM_NPAR; m++)
        drv[m] = s * dvz[m];
    drv[1] += vz * dsp;
    drv[2] += vz * dsa;
    drv[3] += vz * dsi;
    drv[4] += vz * dse;
    return s * vz;
}


/* Radial velocity and derivatives at a centred time.

   drv: REAL[7] (the derivative w.r.t. k is not included at the point
   level, matching numba; see rv_od in orbit3dd.cl). Port of
   `meepmeep.numba3d.rv_cd`. */
MM_INLINE REAL rv_cd3(REAL t, REAL k, REAL p, REAL a, REAL i, REAL e,
                   MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc,
                   REAL *drv) {
    REAL dsp, dsa, dsi, dse;
    REAL s = rv_scale(k, p, a, i, e, &dsp, &dsa, &dsi, &dse);
    return rv_cd_w(t, s, dsp, dsa, dsi, dse, c, dc, drv);
}


/* Radial velocity and derivatives at an absolute time.

   Port of `meepmeep.numba3d.rv_d`. */
MM_INLINE REAL rv_d3(REAL t, REAL k, REAL tc, REAL p, REAL a, REAL i, REAL e,
                  MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc, REAL te,
                  REAL *drv) {
    REAL epoch = floor((t - tc - te + (REAL)0.5 * p) / p);
    REAL rv_val = rv_cd3(t - (tc + te + epoch * p), k, p, a, i, e, c, dc, drv);
    drv[1] += epoch * drv[0];
    return rv_val;
}


/* Phase-angle cosine and its derivatives at a centred time.

   dca: REAL[7]. Chain rule
   d(-z/r)/dtheta = -dz/r + z (x dx + y dy + z dz) / r^3. Port of
   `meepmeep.numba3d.cos_alpha_cd`. */
MM_INLINE REAL cos_alpha_cd3(REAL t, MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc,
                          REAL *dca) {
    REAL px, py, pz, dpx[7], dpy[7], dpz[7];
    pos_cd3(t, c, dc, &px, &py, &pz, dpx, dpy, dpz);
    REAL r2 = px * px + py * py + pz * pz;
    REAL r = sqrt(r2);
    REAL ca = -pz / r;
    REAL inv_r = (REAL)1.0 / r;
    REAL inv_r3 = inv_r / r2;
    for (int m = 0; m < MM_NPAR; m++)
        dca[m] = -dpz[m] * inv_r + pz * (px * dpx[m] + py * dpy[m] + pz * dpz[m]) * inv_r3;
    return ca;
}


/* Phase-angle cosine and derivatives at an absolute time.

   Port of `meepmeep.numba3d.cos_alpha_d`. */
MM_INLINE REAL cos_alpha_d3(REAL t, REAL tc, REAL p,
                         MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc, REAL te,
                         REAL *dca) {
    REAL epoch = floor((t - tc - te + (REAL)0.5 * p) / p);
    REAL ca = cos_alpha_cd3(t - (tc + te + epoch * p), c, dc, dca);
    dca[1] += epoch * dca[0];
    return ca;
}


/* Lambertian phase function, phase angle, and d(phase)/d(cos alpha).

   The derivative simplifies to (pi - alpha)/pi: the contributions from
   the sin(alpha) and alpha terms cancel exactly. `cos_alpha` is clamped
   to [-1, 1]. Port of the numba helper `_lambert_kernel_d`. */
MM_INLINE REAL lambert_kernel_d(REAL cos_alpha, REAL *alpha, REAL *dphase_dc) {
    if (cos_alpha > (REAL)1.0)
        cos_alpha = (REAL)1.0;
    else if (cos_alpha < (REAL)-1.0)
        cos_alpha = (REAL)-1.0;
    REAL sin_alpha = sqrt((REAL)1.0 - cos_alpha * cos_alpha);
    *alpha = acos(cos_alpha);
    *dphase_dc = (PI_R - *alpha) / PI_R;
    return (sin_alpha + (PI_R - *alpha) * cos_alpha) / PI_R;
}


/* Lambertian phase curve and its derivatives at a centred time.

   dflux: REAL[9], ordered (tc, p, a, i, e, w, lan, ag, k). The orbital
   block chains through both the phase angle and the 1/r^2 illumination.
   Port of `meepmeep.numba3d.lambert_phase_curve_cd`. */
MM_INLINE REAL lambert_phase_curve_cd3(REAL t, REAL ag, REAL k,
                                    MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc,
                                    REAL *dflux) {
    REAL px, py, pz, alpha, dphase_dc, dpx[7], dpy[7], dpz[7];
    pos_cd3(t, c, dc, &px, &py, &pz, dpx, dpy, dpz);
    REAL r2 = px * px + py * py + pz * pz;
    REAL r = sqrt(r2);
    REAL inv_r = (REAL)1.0 / r;
    REAL inv_r2 = (REAL)1.0 / r2;
    REAL inv_r3 = inv_r * inv_r2;
    REAL inv_r4 = inv_r2 * inv_r2;
    REAL phase = lambert_kernel_d(-pz * inv_r, &alpha, &dphase_dc);
    REAL amp = k * k * ag;
    REAL flux = amp * phase * inv_r2;
    for (int m = 0; m < MM_NPAR; m++) {
        REAL s = px * dpx[m] + py * dpy[m] + pz * dpz[m];      /* = r * dr/dtheta */
        REAL dcosa = -dpz[m] * inv_r + pz * s * inv_r3;
        dflux[m] = amp * (dphase_dc * dcosa * inv_r2 - (REAL)2.0 * phase * s * inv_r4);
    }
    dflux[7] = k * k * phase * inv_r2;               /* d/d(ag) */
    dflux[8] = (REAL)2.0 * k * ag * phase * inv_r2;  /* d/dk */
    return flux;
}


/* Lambertian phase curve and derivatives at an absolute time.

   Port of `meepmeep.numba3d.lambert_phase_curve_d`. */
MM_INLINE REAL lambert_phase_curve_d3(REAL t, REAL ag, REAL k, REAL tc, REAL p,
                                   MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc,
                                   REAL te, REAL *dflux) {
    REAL epoch = floor((t - tc - te + (REAL)0.5 * p) / p);
    REAL flux = lambert_phase_curve_cd3(t - (tc + te + epoch * p), ag, k, c, dc, dflux);
    dflux[1] += epoch * dflux[0];
    return flux;
}


/* Ellipsoidal-variation signal and its derivatives at a centred time.

   dout: REAL[9], ordered (tc, p, a, i, e, w, lan, alpha, mass_ratio).
   Inclination enters both implicitly through the position (orbital chain)
   and explicitly through the sin^2(inc) prefactor; both contributions sum
   into slot 3. Port of `meepmeep.numba3d.ev_signal_cd`. */
MM_INLINE REAL ev_signal_cd3(REAL t, REAL alpha, REAL mass_ratio, REAL inc,
                          MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc,
                          REAL *dout) {
    REAL sin_inc = sin(inc);
    REAL cos_inc = cos(inc);
    REAL sin2_inc = sin_inc * sin_inc;
    REAL pre = -alpha * mass_ratio * sin2_inc;

    REAL px, py, pz, dpx[7], dpy[7], dpz[7];
    pos_cd3(t, c, dc, &px, &py, &pz, dpx, dpy, dpz);
    REAL d2 = px * px + py * py + pz * pz;
    REAL d = sqrt(d2);
    REAL cz = pz / d;
    REAL g = ((REAL)2.0 * cz * cz - (REAL)1.0) / (d2 * d);
    REAL out = pre * g;

    REAL d5 = d2 * d2 * d;
    REAL A = (REAL)2.0 * pz * pz - d2;
    for (int m = 0; m < MM_NPAR; m++) {
        REAL xdotdx = px * dpx[m] + py * dpy[m] + pz * dpz[m];
        REAL dd = xdotdx / d;
        REAL dA = (REAL)-2.0 * (px * dpx[m] + py * dpy[m]) + (REAL)2.0 * pz * dpz[m];
        REAL dg = (dA - (REAL)5.0 * A * dd / d) / d5;
        dout[m] = pre * dg;
    }
    dout[3] += -alpha * mass_ratio * (REAL)2.0 * sin_inc * cos_inc * g;
    dout[7] = -mass_ratio * sin2_inc * g;   /* d/d(alpha) */
    dout[8] = -alpha * sin2_inc * g;        /* d/d(mass_ratio) */
    return out;
}


/* Ellipsoidal-variation signal and derivatives at an absolute time.

   Port of `meepmeep.numba3d.ev_signal_d`. */
MM_INLINE REAL ev_signal_d3(REAL t, REAL alpha, REAL mass_ratio, REAL inc,
                         REAL tc, REAL p,
                         MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc, REAL te,
                         REAL *dout) {
    REAL epoch = floor((t - tc - te + (REAL)0.5 * p) / p);
    REAL out = ev_signal_cd3(t - (tc + te + epoch * p), alpha, mass_ratio, inc, c, dc, dout);
    dout[1] += epoch * dout[0];
    return out;
}


/* Thermal-emission phase curve and its derivatives at a centred time.

   dout: REAL[10], ordered (tc, p, a, i, e, w, lan, k, fratio, offset).
   The orbital block chains through the phase-angle cosine cz = -z/d and
   the signed in-plane component s = -(wx*y - wy*x)/(|w| d) with w = r x v.
   Port of `meepmeep.numba3d.emission_phase_curve_cd`. */
MM_INLINE REAL emission_phase_curve_cd3(REAL t, REAL k, REAL fratio, REAL offset,
                                     MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc,
                                     REAL *dout) {
    REAL x, y, z, vx, vy, vz;
    REAL dpx[7], dpy[7], dpz[7], dvx[7], dvy[7], dvz[7];
    pos_cd3(t, c, dc, &x, &y, &z, dpx, dpy, dpz);
    vel_cd3(t, c, dc, &vx, &vy, &vz, dvx, dvy, dvz);
    REAL d2 = x * x + y * y + z * z;
    REAL d = sqrt(d2);
    REAL wx = y * vz - z * vy;
    REAL wy = z * vx - x * vz;
    REAL wz = x * vy - y * vx;
    REAL el = sqrt(wx * wx + wy * wy + wz * wz);
    REAL cz = -z / d;
    REAL m = wx * y - wy * x;
    REAL ld = el * d;
    REAL s = -m / ld;
    REAL cd = cos(offset);
    REAL sd = sin(offset);
    REAL g = (REAL)0.5 * ((REAL)1.0 + cd * cz + sd * s);
    REAL amp = k * k * fratio;
    REAL flux = amp * g;

    for (int mm = 0; mm < MM_NPAR; mm++) {
        REAL dxk = dpx[mm], dyk = dpy[mm], dzk = dpz[mm];
        REAL dvxk = dvx[mm], dvyk = dvy[mm], dvzk = dvz[mm];
        REAL dd = (x * dxk + y * dyk + z * dzk) / d;
        REAL dcz = -dzk / d + z * dd / d2;
        REAL dwx = dyk * vz + y * dvzk - dzk * vy - z * dvyk;
        REAL dwy = dzk * vx + z * dvxk - dxk * vz - x * dvzk;
        REAL dwz = dxk * vy + x * dvyk - dyk * vx - y * dvxk;
        REAL dl = (wx * dwx + wy * dwy + wz * dwz) / el;
        REAL dm = dwx * y + wx * dyk - dwy * x - wy * dxk;
        REAL ds = -dm / ld + m * (dl * d + el * dd) / (ld * ld);
        dout[mm] = amp * (REAL)0.5 * (cd * dcz + sd * ds);
    }
    dout[7] = (REAL)2.0 * k * fratio * g;              /* d/dk */
    dout[8] = k * k * g;                               /* d/d(fratio) */
    dout[9] = amp * (REAL)0.5 * (-sd * cz + cd * s);   /* d/d(offset) */
    return flux;
}


/* Thermal-emission phase curve and derivatives at an absolute time.

   Port of `meepmeep.numba3d.emission_phase_curve_d`. */
MM_INLINE REAL emission_phase_curve_d3(REAL t, REAL k, REAL fratio, REAL offset,
                                    REAL tc, REAL p,
                                    MM_GLOBAL const REAL *c, MM_GLOBAL const REAL *dc,
                                    REAL te, REAL *dout) {
    REAL epoch = floor((t - tc - te + (REAL)0.5 * p) / p);
    REAL flux = emission_phase_curve_cd3(t - (tc + te + epoch * p), k, fratio, offset, c, dc, dout);
    dout[1] += epoch * dout[0];
    return flux;
}
