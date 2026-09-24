/*  MeepMeep: fast orbit calculations for exoplanet modelling
 *  Copyright (C) 2022-2026 Hannu Parviainen
 *
 *  Single-expansion-point 2D Taylor coefficient solvers.
 *  Requires: common.cl
 *
 *  Device-function ports of `meepmeep.numba2d` {solve2d, solve2d_d}. These
 *  produce the coefficient matrices the point2d/point2dd evaluators consume,
 *  so a caller can solve on the device instead of solving on the host and
 *  uploading. Solving one expansion costs about what solving a thousand does
 *  (the launch dominates), so this pays off for batches of parameter sets --
 *  population samplers -- and not for a single set per likelihood call.
 *
 *  Unlike the evaluators these are NOT named with a trailing dimension digit:
 *  the numba names already carry the dimension, so `solve2d` here is
 *  `meepmeep.numba2d.solve2d` verbatim and cannot collide with `solve3d`.
 *
 *  Deviations from the numba twins, all forced by C:
 *
 *  - Results are written through MM_GLOBAL output pointers rather than
 *    returned: `cf` is the flattened (2, 5) matrix (10 REALs) and `dcf` the
 *    flattened (7, 2, 5) derivative tensor (70 REALs), in the array layout
 *    documented in common.cl. Both must be distinct, non-overlapping.
 *  - The inclination argument is spelled `inc`; `i` is the loop index.
 *  - `lan` and `from_periastron` are mandatory, not defaulted; pass
 *    (REAL)0.0 and 0 for the numba defaults.
 *  - The Kepler tolerance is precision-aware (see MM_EA_TOL in common.cl).
 *
 *  Each function is written to be read side by side with its numba twin;
 *  that correspondence is the primary correctness argument. When the numba
 *  solver changes, port the change and keep the structure aligned.
 */


/* Taylor coefficients for the sky-plane (x, y) position around an expansion
   point at time `te` relative to the transit centre.

   Writes the 10 contiguous elements of the flattened (2, 5) matrix into `cf`:
   row 0 is x, row 1 is y. Port of `meepmeep.numba2d.solve2d`. */
MM_INLINE void solve2d(REAL te, REAL p, REAL a, REAL inc, REAL e, REAL w, REAL lan,
                    MM_GLOBAL REAL *cf) {
    /* Constants */
    REAL n = TWO_PI_R / p;
    REAL mu = n * n * a * a * a;   /* [R_star^3 / day^2] */

    REAL sqe2 = sqrt((REAL)1.0 - e * e);
    REAL ci = cos(inc);
    REAL cw = cos(w);
    REAL sw = sin(w);

    /* 1. Mean anomaly and eccentric anomaly */
    REAL offset = mean_anomaly_at_transit(e, w);
    REAL ma = mm_mod_two_pi(TWO_PI_R * (te - (-offset * p / TWO_PI_R)) / p);

    REAL ea = ea_from_ma(ma, e);
    REAL sea = sin(ea);
    REAL cea = cos(ea);

    /* 2. Orbital-plane position and velocity */
    REAL r_val = a * ((REAL)1.0 - e * cea);
    REAL xi = a * (cea - e);
    REAL eta = a * sqe2 * sea;

    REAL ea_dot = n * a / r_val;

    REAL v_xi = -a * sea * ea_dot;
    REAL v_eta = a * sqe2 * cea * ea_dot;

    /* 3. Acceleration, jerk, snap */
    REAL r2 = r_val * r_val;
    REAL v2 = v_xi * v_xi + v_eta * v_eta;
    REAL rv = xi * v_xi + eta * v_eta;

    REAL inv_r3 = (REAL)1.0 / (r2 * r_val);
    REAL inv_r5 = inv_r3 / r2;
    REAL inv_r7 = inv_r5 / r2;

    REAL u = -mu * inv_r3;
    REAL u_dot = (REAL)3.0 * mu * rv * inv_r5;
    REAL u_ddot = (REAL)3.0 * mu * (v2 * inv_r5 - (REAL)5.0 * rv * rv * inv_r7)
                  - (REAL)3.0 * u * u;

    REAL a_xi = u * xi;
    REAL a_eta = u * eta;

    REAL j_xi = u_dot * xi + u * v_xi;
    REAL j_eta = u_dot * eta + u * v_eta;

    REAL s_coeff = u_ddot + u * u;
    REAL s_xi = s_coeff * xi + (REAL)2.0 * u_dot * v_xi;
    REAL s_eta = s_coeff * eta + (REAL)2.0 * u_dot * v_eta;

    /* 4. Rotation to the sky plane
       X = -xi * cw + eta * sw
       Y = (-xi * sw - eta * cw) * ci */
    REAL m00 = -cw;
    REAL m01 = sw;
    REAL m10 = -sw * ci;
    REAL m11 = -cw * ci;

    /* Built in private memory because step 5 rereads every column; a global
       round-trip per column would cost more than the registers do. */
    REAL x[5], y[5];
    x[0] = m00 * xi + m01 * eta;
    y[0] = m10 * xi + m11 * eta;
    x[1] = m00 * v_xi + m01 * v_eta;
    y[1] = m10 * v_xi + m11 * v_eta;
    x[2] = (m00 * a_xi + m01 * a_eta) * (REAL)0.5;
    y[2] = (m10 * a_xi + m11 * a_eta) * (REAL)0.5;
    x[3] = (m00 * j_xi + m01 * j_eta) / (REAL)6.0;
    y[3] = (m10 * j_xi + m11 * j_eta) / (REAL)6.0;
    x[4] = (m00 * s_xi + m01 * s_eta) / (REAL)24.0;
    y[4] = (m10 * s_xi + m11 * s_eta) / (REAL)24.0;

    /* 5. Longitude of the ascending node: a constant rotation of the sky
       plane about the line of sight, so it applies to every Taylor column. */
    REAL cO = cos(lan);
    REAL sO = sin(lan);
    for (int col = 0; col < 5; ++col) {
        cf[col]     = cO * x[col] - sO * y[col];
        cf[5 + col] = sO * x[col] + cO * y[col];
    }
}


/* Taylor coefficients and their derivatives w.r.t. (tc, p, a, i, e, w, lan).

   Writes the flattened (2, 5) matrix into `cf` and the flattened (7, 2, 5)
   derivative tensor into `dcf`. With `from_periastron` set, `te` is measured
   from periastron and the rows are the periastron-basis ones
   (tp, p, a, i, e, w, lan). Port of `meepmeep.numba2d.solve2d_d`. */
MM_INLINE void solve2d_d(REAL te, REAL p, REAL a, REAL inc, REAL e, REAL w, REAL lan,
                      int from_periastron,
                      MM_GLOBAL REAL *cf, MM_GLOBAL REAL *dcf) {
    /* Parameter indices: 0=tc, 1=p, 2=a, 3=i, 4=e, 5=w, 6=lan. The working
       vectors are 6 long; the lan row is built analytically at the end.
       The numba twin holds these as rows of one scratch block for the same
       reason they are plain locals here: separate allocations are the cost. */
    REAL dn[6], dmu[6], dsqe2[6], doffset[6], dma[6], dea[6], dsea[6], dcea[6];
    REAL dr[6], dxi[6], deta[6], dea_dot[6], dv_xi[6], dv_eta[6];
    REAL dv2[6], drv[6], dinv_r3[6], dinv_r5[6], dinv_r7[6];
    REAL du[6], du_dot[6], drv2[6], du_ddot[6];
    REAL da_xi[6], da_eta[6], dj_xi[6], dj_eta[6];
    REAL ds_coeff[6], ds_xi[6], ds_eta[6];
    REAL dm00[6], dm01[6], dm10[6], dm11[6];

    for (int k = 0; k < 6; ++k) {
        dn[k] = dmu[k] = dsqe2[k] = doffset[k] = dma[k] = (REAL)0.0;
        dm00[k] = dm01[k] = dm10[k] = dm11[k] = (REAL)0.0;
    }

    /* Step 1: constants and their derivatives */
    REAL n = TWO_PI_R / p;
    dn[1] = -TWO_PI_R / (p * p);

    REAL mu = n * n * a * a * a;
    dmu[1] = (REAL)2.0 * n * dn[1] * a * a * a;
    dmu[2] = (REAL)3.0 * n * n * a * a;

    REAL sqe2 = sqrt((REAL)1.0 - e * e);
    dsqe2[4] = -e / sqe2;

    REAL ci = cos(inc);
    REAL si = sin(inc);

    REAL cw = cos(w);
    REAL sw = sin(w);

    /* Step 2: mean-anomaly offset and its derivatives */
    REAL offset;
    if (from_periastron) {
        /* te is measured from periastron: M = 2 pi te / p, free of e and w. */
        offset = (REAL)0.0;
    } else {
        REAL dm_de, dm_dw;
        offset = mean_anomaly_at_transit_with_derivatives(e, w, &dm_de, &dm_dw);
        doffset[4] = dm_de;
        doffset[5] = dm_dw;
    }

    /* Step 3: mean anomaly and Kepler's equation */
    REAL ma = mm_mod_two_pi(TWO_PI_R * te / p + offset);

    /* Slot 0 (the timing parameter) is not propagated through Kepler's
       equation: the polynomial depends on it only through its argument, so
       its row is rebuilt from the coefficients in the final step. */
    dma[0] = (REAL)0.0;
    /* At a fixed phase from periastron the mean anomaly is period-independent;
       the expansion point rides with the orbit when p changes. */
    dma[1] = from_periastron ? (REAL)0.0 : -TWO_PI_R * te / (p * p);
    dma[4] = doffset[4];
    dma[5] = doffset[5];

    REAL ea = ea_from_ma(ma, e);
    REAL sea = sin(ea);
    REAL cea = cos(ea);

    /* Implicit differentiation: dE/dq = (dM/dq + sin(E) de/dq) / (1 - e cos(E)) */
    REAL inv_denom = (REAL)1.0 / ((REAL)1.0 - e * cea);
    for (int k = 0; k < 6; ++k) {
        REAL de_k = (k == 4) ? (REAL)1.0 : (REAL)0.0;
        dea[k] = (dma[k] + sea * de_k) * inv_denom;
    }
    for (int k = 0; k < 6; ++k) {
        dsea[k] = cea * dea[k];
        dcea[k] = -sea * dea[k];
    }

    /* Step 4: orbital-plane position and velocity */
    REAL r_val = a * ((REAL)1.0 - e * cea);
    for (int k = 0; k < 6; ++k) {
        REAL da_k = (k == 2) ? (REAL)1.0 : (REAL)0.0;
        REAL de_k = (k == 4) ? (REAL)1.0 : (REAL)0.0;
        dr[k] = da_k * ((REAL)1.0 - e * cea) + a * (-de_k * cea - e * dcea[k]);
    }

    REAL xi = a * (cea - e);
    for (int k = 0; k < 6; ++k) {
        REAL da_k = (k == 2) ? (REAL)1.0 : (REAL)0.0;
        REAL de_k = (k == 4) ? (REAL)1.0 : (REAL)0.0;
        dxi[k] = da_k * (cea - e) + a * (dcea[k] - de_k);
    }

    REAL eta = a * sqe2 * sea;
    for (int k = 0; k < 6; ++k) {
        REAL da_k = (k == 2) ? (REAL)1.0 : (REAL)0.0;
        deta[k] = da_k * sqe2 * sea + a * dsqe2[k] * sea + a * sqe2 * dsea[k];
    }

    REAL ea_dot = n * a / r_val;
    for (int k = 0; k < 6; ++k) {
        REAL da_k = (k == 2) ? (REAL)1.0 : (REAL)0.0;
        dea_dot[k] = (dn[k] * a + n * da_k) / r_val - n * a * dr[k] / (r_val * r_val);
    }

    REAL v_xi = -a * sea * ea_dot;
    for (int k = 0; k < 6; ++k) {
        REAL da_k = (k == 2) ? (REAL)1.0 : (REAL)0.0;
        dv_xi[k] = -(da_k * sea * ea_dot + a * dsea[k] * ea_dot + a * sea * dea_dot[k]);
    }

    REAL v_eta = a * sqe2 * cea * ea_dot;
    for (int k = 0; k < 6; ++k) {
        REAL da_k = (k == 2) ? (REAL)1.0 : (REAL)0.0;
        dv_eta[k] = (da_k * sqe2 * cea * ea_dot + a * dsqe2[k] * cea * ea_dot
                     + a * sqe2 * dcea[k] * ea_dot + a * sqe2 * cea * dea_dot[k]);
    }

    /* Step 5: higher-order derivatives */
    REAL r2 = r_val * r_val;
    REAL v2 = v_xi * v_xi + v_eta * v_eta;
    REAL rv = xi * v_xi + eta * v_eta;

    for (int k = 0; k < 6; ++k) {
        dv2[k] = (REAL)2.0 * v_xi * dv_xi[k] + (REAL)2.0 * v_eta * dv_eta[k];
        drv[k] = dxi[k] * v_xi + xi * dv_xi[k] + deta[k] * v_eta + eta * dv_eta[k];
    }

    REAL inv_r3 = (REAL)1.0 / (r2 * r_val);
    REAL inv_r5 = inv_r3 / r2;
    REAL inv_r7 = inv_r5 / r2;

    /* d(r^-n)/dq = -n r^-n (dr/dq) / r */
    for (int k = 0; k < 6; ++k) {
        dinv_r3[k] = (REAL)-3.0 * inv_r3 * dr[k] / r_val;
        dinv_r5[k] = (REAL)-5.0 * inv_r5 * dr[k] / r_val;
        dinv_r7[k] = (REAL)-7.0 * inv_r7 * dr[k] / r_val;
    }

    REAL u = -mu * inv_r3;
    for (int k = 0; k < 6; ++k) {
        du[k] = -dmu[k] * inv_r3 - mu * dinv_r3[k];
    }

    REAL u_dot = (REAL)3.0 * mu * rv * inv_r5;
    for (int k = 0; k < 6; ++k) {
        du_dot[k] = (REAL)3.0 * (dmu[k] * rv * inv_r5 + mu * drv[k] * inv_r5
                                 + mu * rv * dinv_r5[k]);
    }

    REAL rv2 = rv * rv;
    for (int k = 0; k < 6; ++k) {
        drv2[k] = (REAL)2.0 * rv * drv[k];
    }

    REAL u_ddot = (REAL)3.0 * mu * (v2 * inv_r5 - (REAL)5.0 * rv2 * inv_r7)
                  - (REAL)3.0 * u * u;
    for (int k = 0; k < 6; ++k) {
        du_ddot[k] = ((REAL)3.0 * (dmu[k] * (v2 * inv_r5 - (REAL)5.0 * rv2 * inv_r7)
                                   + mu * (dv2[k] * inv_r5 + v2 * dinv_r5[k]
                                           - (REAL)5.0 * (drv2[k] * inv_r7
                                                          + rv2 * dinv_r7[k])))
                      - (REAL)6.0 * u * du[k]);
    }

    REAL a_xi = u * xi;
    REAL a_eta = u * eta;
    for (int k = 0; k < 6; ++k) {
        da_xi[k] = du[k] * xi + u * dxi[k];
        da_eta[k] = du[k] * eta + u * deta[k];
    }

    REAL j_xi = u_dot * xi + u * v_xi;
    REAL j_eta = u_dot * eta + u * v_eta;
    for (int k = 0; k < 6; ++k) {
        dj_xi[k] = du_dot[k] * xi + u_dot * dxi[k] + du[k] * v_xi + u * dv_xi[k];
        dj_eta[k] = du_dot[k] * eta + u_dot * deta[k] + du[k] * v_eta + u * dv_eta[k];
    }

    REAL s_coeff = u_ddot + u * u;
    for (int k = 0; k < 6; ++k) {
        ds_coeff[k] = du_ddot[k] + (REAL)2.0 * u * du[k];
    }

    REAL s_xi = s_coeff * xi + (REAL)2.0 * u_dot * v_xi;
    REAL s_eta = s_coeff * eta + (REAL)2.0 * u_dot * v_eta;
    for (int k = 0; k < 6; ++k) {
        ds_xi[k] = ds_coeff[k] * xi + s_coeff * dxi[k]
                   + (REAL)2.0 * (du_dot[k] * v_xi + u_dot * dv_xi[k]);
        ds_eta[k] = ds_coeff[k] * eta + s_coeff * deta[k]
                    + (REAL)2.0 * (du_dot[k] * v_eta + u_dot * dv_eta[k]);
    }

    /* Step 6: rotation matrix and its derivatives */
    REAL m00 = -cw;
    REAL m01 = sw;
    REAL m10 = -sw * ci;
    REAL m11 = -cw * ci;

    dm00[5] = sw;
    dm01[5] = cw;
    dm10[3] = sw * si;
    dm10[5] = -cw * ci;
    dm11[3] = cw * si;
    dm11[5] = sw * ci;

    /* Step 7: assemble, in private memory. Steps 8 and 9 reread every entry,
       so the result is stored to global once at the end. */
    REAL c[10];    /* (2, 5)    row*5 + col        */
    REAL d[70];    /* (7, 2, 5) k*10 + row*5 + col */

    /* One Taylor column and its six derivative rows. A macro rather than an
       array of pointers over the (xi, v_xi, a_xi, ...) groups: private pointer
       arrays are address-taken, which defeats the compiler's scalarisation and
       spills the whole working set to local memory. */
    #define MM_FILL_COL2(col, qx, qe, dqx, dqe, s)                                  \
        do {                                                                        \
            c[0 * 5 + (col)] = (m00 * (qx) + m01 * (qe)) * (s);                     \
            c[1 * 5 + (col)] = (m10 * (qx) + m11 * (qe)) * (s);                     \
            for (int k = 0; k < 6; ++k) {                                           \
                d[k * 10 + 0 * 5 + (col)] = (dm00[k] * (qx) + m00 * (dqx)[k]        \
                                             + dm01[k] * (qe) + m01 * (dqe)[k]) * (s); \
                d[k * 10 + 1 * 5 + (col)] = (dm10[k] * (qx) + m10 * (dqx)[k]        \
                                             + dm11[k] * (qe) + m11 * (dqe)[k]) * (s); \
            }                                                                       \
        } while (0)

    MM_FILL_COL2(0, xi,   eta,   dxi,   deta,   (REAL)1.0);
    MM_FILL_COL2(1, v_xi, v_eta, dv_xi, dv_eta, (REAL)1.0);
    MM_FILL_COL2(2, a_xi, a_eta, da_xi, da_eta, (REAL)0.5);
    MM_FILL_COL2(3, j_xi, j_eta, dj_xi, dj_eta, (REAL)1.0 / (REAL)6.0);
    MM_FILL_COL2(4, s_xi, s_eta, ds_xi, ds_eta, (REAL)1.0 / (REAL)24.0);

    #undef MM_FILL_COL2

    /* Step 8: longitude of the ascending node. A constant rotation R(lan) of
       the sky plane, independent of the other six parameters, so the product
       rule collapses: rotate what is there and add R'(lan) . c as row 6. */
    REAL cO = cos(lan);
    REAL sO = sin(lan);
    for (int col = 0; col < 5; ++col) {
        REAL x0 = c[col];
        REAL y0 = c[5 + col];

        d[6 * 10 + 0 * 5 + col] = -sO * x0 - cO * y0;
        d[6 * 10 + 1 * 5 + col] =  cO * x0 - sO * y0;

        c[col]     = cO * x0 - sO * y0;
        c[5 + col] = sO * x0 + cO * y0;

        for (int k = 0; k < 6; ++k) {
            REAL dx0 = d[k * 10 + 0 * 5 + col];
            REAL dy0 = d[k * 10 + 1 * 5 + col];
            d[k * 10 + 0 * 5 + col] = cO * dx0 - sO * dy0;
            d[k * 10 + 1 * 5 + col] = sO * dx0 + cO * dy0;
        }
    }

    /* Step 9: timing row. The evaluators compute P(t - tc), so dP/dtc is the
       derivative of the truncated polynomial: d[0, :, n] = -(n+1) c[:, n+1],
       zero at n = 4. That keeps the gradient consistent with the value the
       evaluators return rather than with the exact orbit. */
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 2; ++row) {
            d[0 * 10 + row * 5 + col] = -(REAL)(col + 1) * c[row * 5 + col + 1];
        }
    }
    d[0 * 10 + 0 * 5 + 4] = (REAL)0.0;
    d[0 * 10 + 1 * 5 + 4] = (REAL)0.0;

    for (int m = 0; m < 10; ++m) cf[m] = c[m];
    for (int m = 0; m < 70; ++m) dcf[m] = d[m];
}
