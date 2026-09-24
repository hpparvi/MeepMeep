/*  libmeepmeep example: sky-projected separation and its gradient across a
 *  transit, using the multi-expansion-point orbit machinery end to end.
 *
 *  Build with -DMEEPMEEP_BUILD_EXAMPLES=ON, or by hand:
 *      cc -std=c99 -O2 -I c/include examples/transit.c -L c/build -lmeepmeep -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "meepmeep.h"

#define NPT 15
#define TRES 200

int main(void) {
    /* Orbital parameters: tc, p [d], a [R_star], i [rad], e, w [rad], lan. */
    const double tc = 0.0, p = 3.5, a = 12.0, inc = 1.55, e = 0.15, w = 0.8, lan = 0.0;
    const double two_pi = 6.28318530717958647693;

    double ep_times[NPT], change_times[NPT - 1], dt;
    int ep_table[TRES];
    int status = create_expansion_points(NPT, e, MM_EP_EA, TRES,
                                         ep_times, change_times, &dt, ep_table);
    if (status != MM_OK) {
        fprintf(stderr, "create_expansion_points: %s\n", mm_status_string(status));
        return EXIT_FAILURE;
    }

    double coeffs[NPT * 15], dcoeffs[NPT * 105];
    solve3d_orbit_d(ep_times, NPT, p, a, inc, e, w, lan, coeffs, dcoeffs);
    tp_to_tc_gradient_orbit(dcoeffs, NPT, p, e, w);   /* (tc, p, a, i, e, w, lan) */

    /* The orbit-spanning evaluators anchor on the periastron time. */
    const double tpa = tc - mean_anomaly_at_transit(e, w) / two_pi * p;

    printf("%10s %12s %12s %12s\n", "t [d]", "z [R_star]", "dz/dtc", "dz/da");
    for (int k = 0; k <= 10; ++k) {
        double t = tc - 0.1 + 0.02 * k;
        double dz[MM_NPAR];
        double z = sep_od(t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs, dz);
        printf("%10.4f %12.6f %12.6f %12.6f\n", t, z, dz[0], dz[2]);
    }
    return EXIT_SUCCESS;
}
