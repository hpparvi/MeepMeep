/*  MeepMeep: fast orbit calculations for exoplanet modelling
 *  Copyright (C) 2022-2026 Hannu Parviainen
 *
 *  Expansion-point placement. Port of
 *  `meepmeep.backends.numba.expansion_points`, which on the Python side
 *  leans on scipy's brentq for the anomaly-uniform strategies; the Brent
 *  solver below reproduces brentq's algorithm and default tolerances so the
 *  two placements agree to the root-finder tolerance.
 */

#include <math.h>
#include <float.h>

#include "meepmeep.h"

#define MM_TWO_PI 6.28318530717958647693

/* brentq defaults: xtol=2e-12, rtol=4*eps, maxiter=100. */
#define MM_BRENT_XTOL 2e-12
#define MM_BRENT_RTOL (4.0 * DBL_EPSILON)
#define MM_BRENT_MAXITER 100


const char *mm_status_string(int status) {
    switch (status) {
    case MM_OK: return "ok";
    case MM_ERR_N_EP: return "n_ep must be odd and at least 3";
    case MM_ERR_QUANTITY: return "unknown expansion-point placement quantity";
    case MM_ERR_TRES: return "tres must be positive";
    case MM_ERR_ECCENTRICITY: return "eccentricity must satisfy 0 <= e < 1";
    case MM_ERR_BRACKET: return "root bracket does not change sign";
    case MM_ERR_NO_CONVERGENCE: return "root finder did not converge";
    default: return "unknown status";
    }
}


/* Eccentric anomaly at phase t in [0, 1]. Port of
   `expansion_points.eccentric_anomaly`, which evaluates
   ea_newton_s(t, tc=0, p=1, e, w=pi/2): with w = pi/2 the transit mean
   anomaly offset is exactly zero, so the mean anomaly is just 2 pi t. */
static double eccentric_anomaly(double t, double e) {
    return ea_from_ma(mm_mod_two_pi(MM_TWO_PI * t), e);
}


/* True anomaly at phase t, wrapped to [0, 2 pi). Port of
   `expansion_points.true_anomaly` (ta_from_ea followed by the wrap). */
static double true_anomaly(double t, double e) {
    double ea = eccentric_anomaly(t, e);
    double denom = 1.0 - e * cos(ea);
    double f = atan2(sqrt(1.0 - e * e) * sin(ea) / denom, (cos(ea) - e) / denom);
    if (f < 0.0) f += MM_TWO_PI;
    return f;
}


/* The objective whose root is the phase at which the chosen anomaly equals
   the target v. */
static double anomaly_residual(double t, double e, double v, int quantity) {
    return (quantity == MM_EP_EA ? eccentric_anomaly(t, e) : true_anomaly(t, e)) - v;
}


/* Brent's method on anomaly_residual over [xa, xb]. A transcription of
   scipy.optimize.brentq's C implementation, so the placement matches the
   numba backend's to the solver tolerance. */
static int brentq(double e, double v, int quantity, double xa, double xb, double *root) {
    double xpre = xa, xcur = xb, xblk = 0.0;
    double fpre, fcur, fblk = 0.0;
    double spre = 0.0, scur = 0.0, sbis, delta, stry, dpre, dblk;

    fpre = anomaly_residual(xpre, e, v, quantity);
    fcur = anomaly_residual(xcur, e, v, quantity);
    if (fpre == 0.0) { *root = xpre; return MM_OK; }
    if (fcur == 0.0) { *root = xcur; return MM_OK; }
    if ((fpre < 0.0) == (fcur < 0.0)) return MM_ERR_BRACKET;

    for (int it = 0; it < MM_BRENT_MAXITER; ++it) {
        if (fpre != 0.0 && fcur != 0.0 && ((fpre < 0.0) != (fcur < 0.0))) {
            xblk = xpre;
            fblk = fpre;
            spre = scur = xcur - xpre;
        }
        if (fabs(fblk) < fabs(fcur)) {
            xpre = xcur; xcur = xblk; xblk = xpre;
            fpre = fcur; fcur = fblk; fblk = fpre;
        }

        delta = 0.5 * (MM_BRENT_XTOL + MM_BRENT_RTOL * fabs(xcur));
        sbis = 0.5 * (xblk - xcur);
        if (fcur == 0.0 || fabs(sbis) < delta) { *root = xcur; return MM_OK; }

        if (fabs(spre) > delta && fabs(fcur) < fabs(fpre)) {
            if (xpre == xblk) {
                /* secant (interpolation) */
                stry = -fcur * (xcur - xpre) / (fcur - fpre);
            } else {
                /* inverse quadratic (extrapolation) */
                dpre = (fpre - fcur) / (xpre - xcur);
                dblk = (fblk - fcur) / (xblk - xcur);
                stry = -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre));
            }
            double bound = fabs(spre) < 3.0 * fabs(sbis) - delta ? fabs(spre) : 3.0 * fabs(sbis) - delta;
            if (2.0 * fabs(stry) < bound) {
                spre = scur;
                scur = stry;
            } else {
                spre = sbis;
                scur = sbis;
            }
        } else {
            spre = sbis;
            scur = sbis;
        }

        xpre = xcur;
        fpre = fcur;
        if (fabs(scur) > delta) xcur += scur;
        else xcur += (sbis > 0.0 ? delta : -delta);
        fcur = anomaly_residual(xcur, e, v, quantity);
    }
    *root = xcur;
    return MM_ERR_NO_CONVERGENCE;
}


int create_expansion_points(int n_ep, double e, int quantity, int tres,
                            double *ep_times, double *change_times,
                            double *dt, int *ep_table) {
    if (quantity != MM_EP_MM && quantity != MM_EP_EA && quantity != MM_EP_TA)
        return MM_ERR_QUANTITY;
    if (n_ep < 3 || n_ep % 2 != 1)
        return MM_ERR_N_EP;
    if (tres < 1)
        return MM_ERR_TRES;
    if (!(e >= 0.0 && e < 1.0))
        return MM_ERR_ECCENTRICITY;

    const int half = n_ep / 2;

    if (quantity == MM_EP_MM) {
        /* numpy.linspace(0, 1, n_ep): i * step, with the end point pinned. */
        double step = 1.0 / (n_ep - 1);
        for (int i = 0; i < n_ep; ++i) ep_times[i] = i * step;
        ep_times[n_ep - 1] = 1.0;
        for (int i = 0; i < n_ep - 1; ++i)
            change_times[i] = 0.5 * (ep_times[i] + ep_times[i + 1]);
    } else {
        const double ep_sep = MM_TWO_PI / n_ep;
        const double t_hi = 1.0 - 1e-5;
        int status;

        /* Lower half from the root finder, upper half by mirror symmetry
           about the apoastron point pinned at phase 0.5. */
        ep_times[0] = 0.0;
        ep_times[half] = 0.5;
        double t0 = 1e-5;
        for (int i = 1; i < half; ++i) {
            status = brentq(e, i * ep_sep, quantity, t0, t_hi, &ep_times[i]);
            if (status != MM_OK) return status;
            t0 = ep_times[i];
        }
        for (int j = 1; j < half; ++j)
            ep_times[half + j] = 1.0 - ep_times[half - j];
        ep_times[n_ep - 1] = 1.0;

        t0 = 1e-5;
        for (int i = 0; i < half; ++i) {
            status = brentq(e, (i + 0.5) * ep_sep, quantity, t0, t_hi, &change_times[i]);
            if (status != MM_OK) return status;
            t0 = change_times[i];
        }
        for (int j = 0; j < half; ++j)
            change_times[half + j] = 1.0 - change_times[half - 1 - j];
    }

    /* Time-to-expansion-point table, bin i covering phase [i dt, (i+1) dt). */
    *dt = 1.0 / tres;
    int ik = 0;
    for (int i = 0; i < tres; ++i) {
        if (i * *dt > change_times[ik]) ik += 1;
        if (ik >= n_ep - 1) {
            for (int j = i; j < tres; ++j) ep_table[j] = ik;
            break;
        }
        ep_table[i] = ik;
    }
    return MM_OK;
}
