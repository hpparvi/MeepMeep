/*  MeepMeep: fast orbit calculations for exoplanet modelling
 *  Copyright (C) 2022-2026 Hannu Parviainen
 *
 *  Single-expansion-point 3D evaluators (values only).
 *  Requires: common.cl
 *
 *  Device-function ports of the `meepmeep.numba3d` single-expansion-point
 *  evaluators. OpenCL C has a single flat namespace, so where the numba
 *  backend encodes dimensionality in the package name (point3d vs point2d),
 *  the functions carry a trailing dimension digit instead: `3` here
 *  (`pos_c3`, ...) and `2` in point2d.cl. The dimension-agnostic helper
 *  `lambert_kernel` is unsuffixed. `c` points at the 15 contiguous
 *  coefficients of a flattened (3, 5) solve3d matrix: row 0 is x, row 1 y,
 *  row 2 z (positive z toward the observer). The direct evaluators take
 *  the numba `te = 0.0` default as a mandatory parameter: pass (REAL)0.0
 *  for an expansion point at the transit centre.
 */


/* Planet (x, y, z) position at an expansion-point-centred time.

   Port of `meepmeep.numba3d.pos_c`. */
MM_INLINE void pos_c3(REAL t, MM_GLOBAL const REAL *c, REAL *px, REAL *py, REAL *pz) {
    *px = taylor5(t, c);
    *py = taylor5(t, c + 5);
    *pz = taylor5(t, c + 10);
}


/* Planet (x, y, z) position at an absolute time.

   Port of `meepmeep.numba3d.pos`. */
MM_INLINE void pos3(REAL t, REAL tc, REAL p, MM_GLOBAL const REAL *c, REAL te,
                 REAL *px, REAL *py, REAL *pz) {
    REAL epoch = floor((t - tc - te + (REAL)0.5 * p) / p);
    pos_c3(t - (tc + te + epoch * p), c, px, py, pz);
}


/* Line-of-sight z coordinate at a centred time.

   Port of `meepmeep.numba3d.zpos_c`. */
MM_INLINE REAL zpos_c3(REAL t, MM_GLOBAL const REAL *c) {
    return taylor5(t, c + 10);
}


/* Line-of-sight z coordinate at an absolute time.

   Port of `meepmeep.numba3d.zpos`. */
MM_INLINE REAL zpos3(REAL t, REAL tc, REAL p, MM_GLOBAL const REAL *c, REAL te) {
    REAL epoch = floor((t - tc - te + (REAL)0.5 * p) / p);
    return zpos_c3(t - (tc + te + epoch * p), c);
}


/* Sky-projected planet-star separation at a centred time.

   Port of `meepmeep.numba3d.sep_c`. */
MM_INLINE REAL sep_c3(REAL t, MM_GLOBAL const REAL *c) {
    REAL px = taylor5(t, c);
    REAL py = taylor5(t, c + 5);
    return sqrt(px * px + py * py);
}


/* Sky-projected planet-star separation at an absolute time.

   Port of `meepmeep.numba3d.sep`. */
MM_INLINE REAL sep3(REAL t, REAL tc, REAL p, MM_GLOBAL const REAL *c, REAL te) {
    REAL epoch = floor((t - tc - te + (REAL)0.5 * p) / p);
    return sep_c3(t - (tc + te + epoch * p), c);
}


/* Planet (vx, vy, vz) velocity at a centred time.

   Port of `meepmeep.numba3d.vel_c`. */
MM_INLINE void vel_c3(REAL t, MM_GLOBAL const REAL *c, REAL *vx, REAL *vy, REAL *vz) {
    *vx = taylor5_dot(t, c);
    *vy = taylor5_dot(t, c + 5);
    *vz = taylor5_dot(t, c + 10);
}


/* Planet (vx, vy, vz) velocity at an absolute time.

   Port of `meepmeep.numba3d.vel`. */
MM_INLINE void vel3(REAL t, REAL tc, REAL p, MM_GLOBAL const REAL *c, REAL te,
                 REAL *vx, REAL *vy, REAL *vz) {
    REAL epoch = floor((t - tc - te + (REAL)0.5 * p) / p);
    vel_c3(t - (tc + te + epoch * p), c, vx, vy, vz);
}


/* Line-of-sight velocity at a centred time.

   Port of `meepmeep.numba3d.zvel_c`. */
MM_INLINE REAL zvel_c3(REAL t, MM_GLOBAL const REAL *c) {
    return taylor5_dot(t, c + 10);
}


/* Line-of-sight velocity at an absolute time.

   Port of `meepmeep.numba3d.zvel`. */
MM_INLINE REAL zvel3(REAL t, REAL tc, REAL p, MM_GLOBAL const REAL *c, REAL te) {
    REAL epoch = floor((t - tc - te + (REAL)0.5 * p) / p);
    return zvel_c3(t - (tc + te + epoch * p), c);
}


/* Stellar radial velocity at a centred time, Perryman (2018) Eq. 2.23.

   `k` is the RV semi-amplitude in physical units, which the output
   inherits. The numba original is compiled without fastmath for RV
   precision; OpenCL strict math (no -cl-fast-relaxed-math) matches that.
   Port of `meepmeep.numba3d.rv_c`. */
MM_INLINE REAL rv_c3(REAL t, REAL k, REAL p, REAL a, REAL i, REAL e,
                  MM_GLOBAL const REAL *c) {
    REAL n = TWO_PI_R / p * (a * sin(i)) / sqrt((REAL)1.0 - e * e);
    return zvel_c3(t, c) / n * k;
}


/* Stellar radial velocity at an absolute time.

   Port of `meepmeep.numba3d.rv`. */
MM_INLINE REAL rv3(REAL t, REAL k, REAL tc, REAL p, REAL a, REAL i, REAL e,
                MM_GLOBAL const REAL *c, REAL te) {
    REAL n = TWO_PI_R / p * (a * sin(i)) / sqrt((REAL)1.0 - e * e);
    return zvel3(t, tc, p, c, te) / n * k;
}


/* Cosine of the star-planet-observer phase angle at a centred time.

   Port of `meepmeep.numba3d.cos_alpha_c`. */
MM_INLINE REAL cos_alpha_c3(REAL t, MM_GLOBAL const REAL *c) {
    REAL px, py, pz;
    pos_c3(t, c, &px, &py, &pz);
    return -pz / sqrt(px * px + py * py + pz * pz);
}


/* Cosine of the phase angle at an absolute time.

   Port of `meepmeep.numba3d.cos_alpha`. */
MM_INLINE REAL cos_alpha3(REAL t, REAL tc, REAL p, MM_GLOBAL const REAL *c, REAL te) {
    REAL epoch = floor((t - tc - te + (REAL)0.5 * p) / p);
    return cos_alpha_c3(t - (tc + te + epoch * p), c);
}


/* Lambertian phase function at a cosine of the phase angle.

   Returns the disk-integrated reflectance
   (sin(alpha) + (pi - alpha) cos(alpha)) / pi and writes the phase angle
   into `alpha` as a by-product. `cos_alpha` is clamped to [-1, 1] so a
   Taylor-rounding overshoot cannot produce a NaN from acos. Port of the
   numba helper `_lambert_kernel`. */
MM_INLINE REAL lambert_kernel(REAL cos_alpha, REAL *alpha) {
    if (cos_alpha > (REAL)1.0)
        cos_alpha = (REAL)1.0;
    else if (cos_alpha < (REAL)-1.0)
        cos_alpha = (REAL)-1.0;
    REAL sin_alpha = sqrt((REAL)1.0 - cos_alpha * cos_alpha);
    *alpha = acos(cos_alpha);
    return (sin_alpha + (PI_R - *alpha) * cos_alpha) / PI_R;
}


/* Lambertian reflected-light phase curve at a centred time.

   `ag` is the geometric albedo and `k` the radius ratio. Port of
   `meepmeep.numba3d.lambert_phase_curve_c`. */
MM_INLINE REAL lambert_phase_curve_c3(REAL t, REAL ag, REAL k,
                                   MM_GLOBAL const REAL *c) {
    REAL px, py, pz, alpha;
    pos_c3(t, c, &px, &py, &pz);
    REAL r2 = px * px + py * py + pz * pz;
    REAL phase = lambert_kernel(-pz / sqrt(r2), &alpha);
    return k * k * ag / r2 * phase;
}


/* Lambertian reflected-light phase curve at an absolute time.

   Port of `meepmeep.numba3d.lambert_phase_curve`. */
MM_INLINE REAL lambert_phase_curve3(REAL t, REAL ag, REAL k, REAL tc, REAL p,
                                 MM_GLOBAL const REAL *c, REAL te) {
    REAL epoch = floor((t - tc - te + (REAL)0.5 * p) / p);
    return lambert_phase_curve_c3(t - (tc + te + epoch * p), ag, k, c);
}


/* Ellipsoidal-variation signal at a centred time (Lillo-Box et al. 2014).

   `alpha` is the EV amplitude coefficient, `mass_ratio` the planet-star
   mass ratio, and `inc` the inclination. Port of
   `meepmeep.numba3d.ev_signal_c`. */
MM_INLINE REAL ev_signal_c3(REAL t, REAL alpha, REAL mass_ratio, REAL inc,
                         MM_GLOBAL const REAL *c) {
    REAL sin_inc = sin(inc);
    REAL pre = -alpha * mass_ratio * sin_inc * sin_inc;
    REAL px, py, pz;
    pos_c3(t, c, &px, &py, &pz);
    REAL d2 = px * px + py * py + pz * pz;
    REAL d = sqrt(d2);
    REAL cz = pz / d;
    return pre * ((REAL)2.0 * cz * cz - (REAL)1.0) / (d2 * d);
}


/* Ellipsoidal-variation signal at an absolute time.

   Port of `meepmeep.numba3d.ev_signal`. */
MM_INLINE REAL ev_signal3(REAL t, REAL alpha, REAL mass_ratio, REAL inc,
                       REAL tc, REAL p, MM_GLOBAL const REAL *c, REAL te) {
    REAL epoch = floor((t - tc - te + (REAL)0.5 * p) / p);
    return ev_signal_c3(t - (tc + te + epoch * p), alpha, mass_ratio, inc, c);
}


/* Thermal-emission phase curve at a centred time.

   `k` is the radius ratio, `fratio` the day-side flux ratio, and `offset`
   the hot-spot offset [radians]. The orbital angular-momentum vector
   (w = r x v) orients the offset in the orbital plane. Port of
   `meepmeep.numba3d.emission_phase_curve_c`. */
MM_INLINE REAL emission_phase_curve_c3(REAL t, REAL k, REAL fratio, REAL offset,
                                    MM_GLOBAL const REAL *c) {
    REAL x, y, z, vx, vy, vz;
    pos_c3(t, c, &x, &y, &z);
    vel_c3(t, c, &vx, &vy, &vz);
    REAL d = sqrt(x * x + y * y + z * z);
    REAL wx = y * vz - z * vy;
    REAL wy = z * vx - x * vz;
    REAL wz = x * vy - y * vx;
    REAL el = sqrt(wx * wx + wy * wy + wz * wz);
    REAL cz = -z / d;
    REAL m = wx * y - wy * x;
    REAL s = -m / (el * d);
    REAL g = (REAL)0.5 * ((REAL)1.0 + cos(offset) * cz + sin(offset) * s);
    return k * k * fratio * g;
}


/* Thermal-emission phase curve at an absolute time.

   Port of `meepmeep.numba3d.emission_phase_curve`. */
MM_INLINE REAL emission_phase_curve3(REAL t, REAL k, REAL fratio, REAL offset,
                                  REAL tc, REAL p, MM_GLOBAL const REAL *c, REAL te) {
    REAL epoch = floor((t - tc - te + (REAL)0.5 * p) / p);
    return emission_phase_curve_c3(t - (tc + te + epoch * p), k, fratio, offset, c);
}
