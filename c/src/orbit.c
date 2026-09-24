/*  MeepMeep: fast orbit calculations for exoplanet modelling
 *  Copyright (C) 2022-2026 Hannu Parviainen
 *
 *  Orbit-wide coefficient solvers and gradient basis transforms. Ports of
 *  `meepmeep.backends.numba.orbit3d._common.solve3d_orbit`,
 *  `orbit3dd._common.solve3d_orbit_d`, and the `*_gradient*` helpers in
 *  `backends.numba.utils`. The transforms work in place where numba
 *  returns a copy.
 */

#include <string.h>

#include "meepmeep.h"

#define MM_TWO_PI 6.28318530717958647693


void solve3d_orbit(const double *ep_times, int npt, double p, double a,
                   double inc, double e, double w, double lan, double *coeffs) {
    double to = mean_anomaly_at_transit(e, w) / MM_TWO_PI * p;
    for (int ix = 0; ix < npt - 1; ++ix)
        solve3d(p * ep_times[ix] - to, p, a, inc, e, w, lan, coeffs + 15 * ix);
    memcpy(coeffs + 15 * (npt - 1), coeffs, 15 * sizeof(double));
}


void solve3d_orbit_d(const double *ep_times, int npt, double p, double a,
                     double inc, double e, double w, double lan,
                     double *coeffs, double *dcoeffs) {
    for (int ix = 0; ix < npt - 1; ++ix) {
        double *dcf = dcoeffs + 105 * ix;
        /* Expansion points sit at fixed phases from periastron, so the
           solver is anchored there and the rows are the periastron-basis
           ones. */
        solve3d_d(p * ep_times[ix], p, a, inc, e, w, lan, 1, coeffs + 15 * ix, dcf);
        /* The expansion time p * phase moves with the period, which shifts
           the polynomial argument: at the coefficient level that is phase
           times the timing row, added to the period row. */
        for (int j = 0; j < 15; ++j)
            dcf[15 + j] += ep_times[ix] * dcf[j];
    }
    memcpy(coeffs + 15 * (npt - 1), coeffs, 15 * sizeof(double));
    memcpy(dcoeffs + 105 * (npt - 1), dcoeffs, 105 * sizeof(double));
    /* The periodic image sits at phase ep_times[npt - 1] (= ep_times[0] + 1),
       not at slot 0's phase, so its period row takes the phase-times-timing-row
       term for its own phase. */
    double *dimg = dcoeffs + 105 * (npt - 1);
    for (int j = 0; j < 15; ++j)
        dimg[15 + j] += (ep_times[npt - 1] - ep_times[0]) * dimg[j];
}


/* Shared body of the two single-block transforms: add `sign` times the
   chain-rule multiples of the timing row to the p, e and w rows. */
static void shift_timing_row(double *dc, int block, double p, double e, double w,
                             double sign) {
    double dm_de, dm_dw;
    double m_tr = mean_anomaly_at_transit_with_derivatives(e, w, &dm_de, &dm_dw);
    double c = 1.0 / MM_TWO_PI;
    double fp = sign * (m_tr * c);
    double fe = sign * (dm_de * p * c);
    double fw = sign * (dm_dw * p * c);
    for (int j = 0; j < block; ++j) {
        double timing = dc[j];
        dc[1 * block + j] += timing * fp;
        dc[4 * block + j] += timing * fe;
        dc[5 * block + j] += timing * fw;
    }
}


void tc_to_tp_gradient(double *dc, int block, double p, double e, double w) {
    shift_timing_row(dc, block, p, e, w, 1.0);
}


void tp_to_tc_gradient(double *dc, int block, double p, double e, double w) {
    shift_timing_row(dc, block, p, e, w, -1.0);
}


void tp_to_tc_gradient_orbit(double *dcoeffs, int npt, double p, double e,
                             double w) {
    for (int ix = 0; ix < npt; ++ix)
        shift_timing_row(dcoeffs + 105 * ix, 15, p, e, w, -1.0);
}
