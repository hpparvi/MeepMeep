/*  MeepMeep: fast orbit calculations for exoplanet modelling
 *  Copyright (C) 2022-2026 Hannu Parviainen
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

/*  libmeepmeep: the MeepMeep Taylor-series orbit evaluators as a C99 library.
 *
 *  The library is a second compile target of the sources shared with the
 *  OpenCL backend (meepmeep/backends/opencl (the .cl files), built as plain C by
 *  c/src/meepmeep.c), plus the host-side pieces a self-contained library
 *  needs: expansion-point placement, the orbit-wide coefficient solvers, and
 *  the gradient basis transforms (c/src/expansion_points.c, c/src/orbit.c).
 *  Precision is fixed to double.
 *
 *  Pipeline for a full-orbit model:
 *
 *      int tres;
 *      expansion_table_size(NPT, e, MM_EP_EA, &tres);
 *      double ep_times[NPT], change_times[NPT - 1], dt;
 *      int *ep_table = malloc(tres * sizeof(int));
 *      create_expansion_points(NPT, e, MM_EP_EA, tres,
 *                              ep_times, change_times, &dt, ep_table);
 *      double coeffs[NPT * 15];
 *      solve3d_orbit(ep_times, NPT, p, a, inc, e, w, lan, coeffs);
 *      double tpa = tc - mean_anomaly_at_transit(e, w) / (2 pi) * p;
 *      double z = sep_o(t, tpa, p, dt, ep_table, ep_times, coeffs);
 *
 *  For a single expansion point around the transit (the Expansion2D /
 *  Expansion3D path), call solve2d / solve3d once and evaluate with the
 *  "2"/"3"-suffixed single-expansion-point functions (sep3, pos_cd3, ...).
 *
 *  Conventions (see also the header comments of the .cl files):
 *
 *  - Names mirror the numba backend (meepmeep.numba2d / meepmeep.numba3d).
 *    The single-expansion-point functions carry a trailing dimension digit
 *    (pos_c2 / pos_c3, sep_cd2 / sep_cd3, ...); the multi-expansion-point
 *    _o / _od evaluators and the dimension-agnostic helpers are unsuffixed.
 *    Symbols are NOT prefixed, so link the library into a namespace of your
 *    own if these short names clash.
 *  - Only the scalar evaluators exist; loop over your samples.
 *  - Optional numba arguments (te, lan, timing_is_tc) are mandatory here.
 *  - Coefficient arrays are C-contiguous: c (D, 5) row d at c + 5*d;
 *    dc (7, D, 5) parameter block m at dc + 5*D*m; coeffs (npt, 3, 5)
 *    expansion point ix at coeffs + 15*ix; dcoeffs (npt, 7, 3, 5) at
 *    dcoeffs + 105*ix.
 *  - Gradients use the seven-parameter order (tc, p, a, i, e, w, lan) and
 *    are written into caller-provided buffers of MM_NPAR doubles (or more,
 *    for functions with extra physical inputs: rv_od is 8 wide, the Lambert
 *    and ellipsoidal-variation functions 9, emission 10).
 *  - solve3d_orbit_d returns the periastron basis (tp, p, a, i, e, w, lan);
 *    apply tp_to_tc_gradient_orbit for the transit-centre basis.
 *  - Never compile with -ffast-math: the numba twins that are compiled
 *    without fastmath (true_anomaly, rv) rely on strict math, and the parity
 *    tests pin ~1e-12 agreement.
 */

#ifndef MEEPMEEP_H
#define MEEPMEEP_H

#ifdef __cplusplus
extern "C" {
#endif

/* Number of orbital parameters carried through every gradient:
   (tc, p, a, i, e, w, lan). */
#define MM_NPAR 7

/* Status codes returned by the functions that can fail. */
enum mm_status {
    MM_OK = 0,
    MM_ERR_N_EP = 1,           /* n_ep must be odd and at least 3 */
    MM_ERR_QUANTITY = 2,       /* unknown placement strategy */
    MM_ERR_TRES = 3,           /* tres must be positive */
    MM_ERR_ECCENTRICITY = 4,   /* e must satisfy 0 <= e < 1 */
    MM_ERR_BRACKET = 5,        /* root bracket lost its sign change */
    MM_ERR_NO_CONVERGENCE = 6  /* root finder hit its iteration cap */
};

/* Expansion-point placement strategies (the `quantity` argument of
   create_expansion_points; numba's 'mm', 'ea', 'ta'). */
enum mm_ep_quantity {
    MM_EP_MM = 0,   /* uniform in mean motion (time) */
    MM_EP_EA = 1,   /* uniform in eccentric anomaly */
    MM_EP_TA = 2    /* uniform in true anomaly */
};

/* Human-readable description of an mm_status value. */
const char *mm_status_string(int status);


/* Place n_ep expansion points along one orbital period and build the
   time-to-expansion-point table. Port of
   `meepmeep.backends.numba.expansion_points.create_expansion_points`.

   Writes ep_times[n_ep] (expansion-point phases in [0, 1]; the last slot is
   the periodic image of the first), change_times[n_ep - 1] (the phases at
   which dispatch switches to the next expansion point), *dt = 1 / tres, and
   ep_table[tres] (the index of the expansion point whose region contains
   each phase bin's centre). The 'ea' and 'ta' strategies solve for the
   phases with a Brent root finder matching scipy's brentq tolerances (xtol
   2e-12). Size the table with expansion_table_size: a table too coarse to
   resolve the regions near periastron limits the accuracy at high
   eccentricity. Returns MM_OK or an mm_status error; the outputs are
   unspecified on error. */
int create_expansion_points(int n_ep, double e, int quantity, int tres,
                            double *ep_times, double *change_times,
                            double *dt, int *ep_table);

/* Recommended number of time-to-expansion-point table bins for a placement:
   eight bins per narrowest expansion-point region of the placement for
   max(e, 0.9), between 200 and 2^20. Writes it to *tres, the value the
   Python side uses by default. Port of
   `meepmeep.backends.numba.expansion_points.expansion_table_size`. Returns
   MM_OK or an mm_status error. */
int expansion_table_size(int n_ep, double e, int quantity, int *tres);


/* Taylor coefficients at every expansion point of one orbit. Port of
   `meepmeep.numba3d.solve3d_orbit`.

   ep_times has npt entries (from create_expansion_points); coeffs receives
   the npt * 15 doubles of the flattened (npt, 3, 5) coefficient stack. The
   last slot is copied from the first. */
void solve3d_orbit(const double *ep_times, int npt, double p, double a,
                   double inc, double e, double w, double lan, double *coeffs);


/* Taylor and parameter-derivative coefficients at every expansion point.
   Port of `meepmeep.numba3d.solve3d_orbit_d`.

   As solve3d_orbit, plus dcoeffs receives the npt * 105 doubles of the
   flattened (npt, 7, 3, 5) derivative stack in the PERIASTRON basis
   (tp, p, a, i, e, w, lan). The last slot is copied from the first except
   for its period row, which gains the timing row once more because the
   periodic image sits one full phase later. */
void solve3d_orbit_d(const double *ep_times, int npt, double p, double a,
                     double inc, double e, double w, double lan,
                     double *coeffs, double *dcoeffs);


/* Reparametrise one (7, D, 5) gradient block between the transit-centre
   basis (tc, p, a, i, e, w, lan) and the periastron basis
   (tp, p, a, i, e, w, lan). Ports of `meepmeep.numba3d.tc_to_tp_gradient`
   and `tp_to_tc_gradient`, except that they transform IN PLACE.

   `block` is the number of doubles per parameter row: 10 for a 2D block,
   15 for a 3D block. Both transforms read only the timing row (block 0),
   which they never write, so in-place is safe. */
void tc_to_tp_gradient(double *dc, int block, double p, double e, double w);
void tp_to_tc_gradient(double *dc, int block, double p, double e, double w);


/* Reparametrise every expansion point of an orbit gradient stack from the
   periastron basis solve3d_orbit_d returns to the transit-centre basis, in
   place. Port of `meepmeep.numba3d.tp_to_tc_gradient_orbit`. dcoeffs holds
   npt * 105 doubles. */
void tp_to_tc_gradient_orbit(double *dcoeffs, int npt, double p, double e,
                             double w);


/* ======================================================================
 *  Prototypes of the functions shared with the OpenCL backend. Each is
 *  defined in the named .cl file, whose header comment documents the
 *  file's conventions, and its comment names the numba twin that carries
 *  the full docstring (argument semantics, units, gradient ordering).
 * ====================================================================== */

/* BEGIN GENERATED PROTOTYPES -- edit c/tools/generate_header.py, not this block. */

/* ---- common.cl ----------------------------------------------------------- */
/* Evaluate a 5th-order Taylor polynomial with Horner's scheme.

   `cf` points at five contiguous coefficients ordered [position, velocity,
   acceleration/2, jerk/6, snap/24] (pre-scaled by the factorial, so this is
   a plain polynomial evaluation). One row of a solve2d/solve3d matrix. */
double taylor5(double t, const double *cf);

/* Time derivative of `taylor5` over the same coefficient row.

   Mirrors the Horner form of `backends.numba.point3d.velocity._vel_c_s`. */
double taylor5_dot(double t, const double *cf);

/* Mean anomaly at the moment of primary transit.

   Port of `backends.numba.utils.mean_anomaly_at_transit`. */
double mean_anomaly_at_transit(double ecc, double w);

/* Mean anomaly at transit and its derivatives w.r.t. e and w.

   The value is returned; the derivatives are written into `dm_de` and
   `dm_dw`. Port of
   `backends.numba.utils.mean_anomaly_at_transit_with_derivatives`. */
double mean_anomaly_at_transit_with_derivatives(double ecc, double w,
                                                double *dm_de, double *dm_dw);

/* Eccentricity vector `ev` (3 values) and its Jacobian `dev` (3 x 7,
   row-major, columns (tc, p, a, i, e, w, lan)) in the observer's frame; the
   (x, y) rows are rotated by the node `lan`. The circular sentinel
   (e <= 1e-5) gives ev = (-1, 0, 0) and a zero Jacobian. `dev` is the
   eccentricity-vector input of `true_anomaly_od`. Port of
   `meepmeep.numba3d.eccentricity_vector_d`. */
void eccentricity_vector_d(double inc, double e, double w, double lan,
                           double *ev, double *dev);

/* Python/NumPy float `%` semantics: the result takes the sign of the divisor,
   so it lands in [0, 2pi). C's fmod takes the sign of the dividend instead.
   The numba solvers wrap the mean anomaly with the NumPy convention.

   The distinction is unobservable through `ea_from_ma` alone, because
   E - e sin(E) = M is strictly monotonic, so root(M + 2pi) = root(M) + 2pi
   exactly and the solvers consume only sin(E) and cos(E). It is kept because
   that stops being true the moment anything reads E itself. */
double mm_mod_two_pi(double x);

/* Solve Kepler's equation E - e sin(E) = M for the eccentric anomaly.

   Port of `meepmeep.backends.numba.newton.newton.ea_from_ma`. The iteration
   count is data-dependent, so within a warp every lane pays for the slowest
   lane: a batch spanning a range of eccentricities costs more per parameter
   set than one sharing a single eccentricity. */
double ea_from_ma(double ma, double ecc);


/* ---- solve2d.cl ---------------------------------------------------------- */
/* Taylor coefficients for the sky-plane (x, y) position around an expansion
   point at time `te` relative to the transit centre.

   Writes the 10 contiguous elements of the flattened (2, 5) matrix into `cf`:
   row 0 is x, row 1 is y. Port of `meepmeep.numba2d.solve2d`. */
void solve2d(double te, double p, double a, double inc, double e, double w,
             double lan, double *cf);

/* Taylor coefficients and their derivatives w.r.t. (tc, p, a, i, e, w, lan).

   Writes the flattened (2, 5) matrix into `cf` and the flattened (7, 2, 5)
   derivative tensor into `dcf`. With `from_periastron` set, `te` is measured
   from periastron and the rows are the periastron-basis ones
   (tp, p, a, i, e, w, lan). Port of `meepmeep.numba2d.solve2d_d`. */
void solve2d_d(double te, double p, double a, double inc, double e, double w,
               double lan, int from_periastron, double *cf, double *dcf);


/* ---- solve3d.cl ---------------------------------------------------------- */
/* Taylor coefficients for the (x, y, z) position around an expansion point at
   time `te` relative to the transit centre.

   Port of `meepmeep.numba3d.solve3d`. */
void solve3d(double te, double p, double a, double inc, double e, double w,
             double lan, double *cf);

/* Taylor coefficients and their derivatives w.r.t. (tc, p, a, i, e, w, lan).

   Port of `meepmeep.numba3d.solve3d_d`. With `from_periastron` set, `te` is
   measured from periastron and the rows are the periastron-basis ones
   (tp, p, a, i, e, w, lan). */
void solve3d_d(double te, double p, double a, double inc, double e, double w,
               double lan, int from_periastron, double *cf, double *dcf);


/* ---- point2d.cl ---------------------------------------------------------- */
/* Planet sky-plane (x, y) position at an expansion-point-centred time.

   Port of `meepmeep.numba2d.pos_c`. */
void pos_c2(double t, const double *c, double *px, double *py);

/* Planet sky-plane (x, y) position at an absolute time.

   Folds `t` into a single orbital epoch around the expansion point at
   `tc + te` and evaluates the centred polynomial. Port of
   `meepmeep.numba2d.pos`. */
void pos2(double t, double tc, double p, const double *c, double te, double *px,
          double *py);

/* Sky-projected planet-star separation at an expansion-point-centred time.

   Port of `meepmeep.numba2d.sep_c`. */
double sep_c2(double t, const double *c);

/* Sky-projected planet-star separation at an absolute time.

   Port of `meepmeep.numba2d.sep`. */
double sep2(double t, double tc, double p, const double *c, double te);


/* ---- point2dd.cl --------------------------------------------------------- */
/* Position and its (tc, p, a, i, e, w, lan) derivatives at a centred time.

   Port of `meepmeep.numba2d.pos_cd`. */
void pos_cd2(double t, const double *c, const double *dc, double *px,
             double *py, double *dpx, double *dpy);

/* Position and derivatives at an absolute time.

   Folds the time around the expansion point and adds the period-folding
   chain term: the folded time depends on p via -epoch*p, so the total
   period derivative (slot 1) gains epoch times the timing derivative
   (slot 0). Port of `meepmeep.numba2d.pos_d`. */
void pos_d2(double t, double tc, double p, const double *c, const double *dc,
            double te, double *px, double *py, double *dpx, double *dpy);

/* Separation and its derivatives at a centred time.

   The position gradients are reduced with the chain rule
   dd/dtheta = (px*dpx + py*dpy) / d, singular only at an exact centre
   crossing (d = 0), as in the numba original. Port of
   `meepmeep.numba2d.sep_cd`. */
double sep_cd2(double t, const double *c, const double *dc, double *dd);

/* Separation and derivatives at an absolute time.

   Port of `meepmeep.numba2d.sep_d`. */
double sep_d2(double t, double tc, double p, const double *c, const double *dc,
              double te, double *dd);


/* ---- point3d.cl ---------------------------------------------------------- */
/* Planet (x, y, z) position at an expansion-point-centred time.

   Port of `meepmeep.numba3d.pos_c`. */
void pos_c3(double t, const double *c, double *px, double *py, double *pz);

/* Planet (x, y, z) position at an absolute time.

   Port of `meepmeep.numba3d.pos`. */
void pos3(double t, double tc, double p, const double *c, double te, double *px,
          double *py, double *pz);

/* Line-of-sight z coordinate at a centred time.

   Port of `meepmeep.numba3d.zpos_c`. */
double zpos_c3(double t, const double *c);

/* Line-of-sight z coordinate at an absolute time.

   Port of `meepmeep.numba3d.zpos`. */
double zpos3(double t, double tc, double p, const double *c, double te);

/* Sky-projected planet-star separation at a centred time.

   Port of `meepmeep.numba3d.sep_c`. */
double sep_c3(double t, const double *c);

/* Sky-projected planet-star separation at an absolute time.

   Port of `meepmeep.numba3d.sep`. */
double sep3(double t, double tc, double p, const double *c, double te);

/* Planet (vx, vy, vz) velocity at a centred time.

   Port of `meepmeep.numba3d.vel_c`. */
void vel_c3(double t, const double *c, double *vx, double *vy, double *vz);

/* Planet (vx, vy, vz) velocity at an absolute time.

   Port of `meepmeep.numba3d.vel`. */
void vel3(double t, double tc, double p, const double *c, double te, double *vx,
          double *vy, double *vz);

/* Line-of-sight velocity at a centred time.

   Port of `meepmeep.numba3d.zvel_c`. */
double zvel_c3(double t, const double *c);

/* Line-of-sight velocity at an absolute time.

   Port of `meepmeep.numba3d.zvel`. */
double zvel3(double t, double tc, double p, const double *c, double te);

/* Stellar radial velocity at a centred time, Perryman (2018) Eq. 2.23.

   `k` is the RV semi-amplitude in physical units, which the output
   inherits. The numba original is compiled without fastmath for RV
   precision; OpenCL strict math (no -cl-fast-relaxed-math) matches that.
   Port of `meepmeep.numba3d.rv_c`. */
double rv_c3(double t, double k, double p, double a, double i, double e,
             const double *c);

/* Stellar radial velocity at an absolute time.

   Port of `meepmeep.numba3d.rv`. */
double rv3(double t, double k, double tc, double p, double a, double i,
           double e, const double *c, double te);

/* Cosine of the star-planet-observer phase angle at a centred time.

   Port of `meepmeep.numba3d.cos_alpha_c`. */
double cos_alpha_c3(double t, const double *c);

/* Cosine of the phase angle at an absolute time.

   Port of `meepmeep.numba3d.cos_alpha`. */
double cos_alpha3(double t, double tc, double p, const double *c, double te);

/* Lambertian phase function at a cosine of the phase angle.

   Returns the disk-integrated reflectance
   (sin(alpha) + (pi - alpha) cos(alpha)) / pi and writes the phase angle
   into `alpha` as a by-product. `cos_alpha` is clamped to [-1, 1] so a
   Taylor-rounding overshoot cannot produce a NaN from acos. Port of the
   numba helper `_lambert_kernel`. */
double lambert_kernel(double cos_alpha, double *alpha);

/* Lambertian reflected-light phase curve at a centred time.

   `ag` is the geometric albedo and `k` the radius ratio. Port of
   `meepmeep.numba3d.lambert_phase_curve_c`. */
double lambert_phase_curve_c3(double t, double ag, double k, const double *c);

/* Lambertian reflected-light phase curve at an absolute time.

   Port of `meepmeep.numba3d.lambert_phase_curve`. */
double lambert_phase_curve3(double t, double ag, double k, double tc, double p,
                            const double *c, double te);

/* Ellipsoidal-variation signal at a centred time (Lillo-Box et al. 2014).

   `alpha` is the EV amplitude coefficient, `mass_ratio` the planet-star
   mass ratio, and `inc` the inclination. Port of
   `meepmeep.numba3d.ev_signal_c`. */
double ev_signal_c3(double t, double alpha, double mass_ratio, double inc,
                    const double *c);

/* Ellipsoidal-variation signal at an absolute time.

   Port of `meepmeep.numba3d.ev_signal`. */
double ev_signal3(double t, double alpha, double mass_ratio, double inc,
                  double tc, double p, const double *c, double te);

/* Thermal-emission phase curve at a centred time.

   `k` is the radius ratio, `fratio` the day-side flux ratio, and `offset`
   the hot-spot offset [radians]. The orbital angular-momentum vector
   (w = r x v) orients the offset in the orbital plane. Port of
   `meepmeep.numba3d.emission_phase_curve_c`. */
double emission_phase_curve_c3(double t, double k, double fratio, double offset,
                               const double *c);

/* Thermal-emission phase curve at an absolute time.

   Port of `meepmeep.numba3d.emission_phase_curve`. */
double emission_phase_curve3(double t, double k, double fratio, double offset,
                             double tc, double p, const double *c, double te);


/* ---- point3dd.cl --------------------------------------------------------- */
/* Position and its (tc, p, a, i, e, w, lan) derivatives at a centred time.

   dpx, dpy, dpz: REAL[7] output buffers. Port of `meepmeep.numba3d.pos_cd`. */
void pos_cd3(double t, const double *c, const double *dc, double *px,
             double *py, double *pz, double *dpx, double *dpy, double *dpz);

/* Position and derivatives at an absolute time.

   Adds the period-folding chain term: the folded time depends on p via
   -epoch*p, so the total period derivative (slot 1) gains epoch times the
   timing derivative (slot 0). Port of `meepmeep.numba3d.pos_d`. */
void pos_d3(double t, double tc, double p, const double *c, const double *dc,
            double te, double *px, double *py, double *pz, double *dpx,
            double *dpy, double *dpz);

/* Line-of-sight z and its derivatives at a centred time.

   dpz: REAL[7]. Port of `meepmeep.numba3d.zpos_cd`. */
double zpos_cd3(double t, const double *c, const double *dc, double *dpz);

/* Line-of-sight z and derivatives at an absolute time.

   Port of `meepmeep.numba3d.zpos_d`. */
double zpos_d3(double t, double tc, double p, const double *c, const double *dc,
               double te, double *dpz);

/* Sky-projected separation and its derivatives at a centred time.

   dd: REAL[7]. Chain rule dd/dtheta = (px*dpx + py*dpy) / d, singular only
   at an exact centre crossing (d = 0), as in the numba original. Port of
   `meepmeep.numba3d.sep_cd`. */
double sep_cd3(double t, const double *c, const double *dc, double *dd);

/* Sky-projected separation and derivatives at an absolute time.

   Port of `meepmeep.numba3d.sep_d`. */
double sep_d3(double t, double tc, double p, const double *c, const double *dc,
              double te, double *dd);

/* Velocity and its derivatives at a centred time.

   dvx, dvy, dvz: REAL[7]. Port of `meepmeep.numba3d.vel_cd`. */
void vel_cd3(double t, const double *c, const double *dc, double *vx,
             double *vy, double *vz, double *dvx, double *dvy, double *dvz);

/* Velocity and derivatives at an absolute time.

   Port of `meepmeep.numba3d.vel_d`. */
void vel_d3(double t, double tc, double p, const double *c, const double *dc,
            double te, double *vx, double *vy, double *vz, double *dvx,
            double *dvy, double *dvz);

/* Line-of-sight velocity and its derivatives at a centred time.

   dvz: REAL[7]. Port of `meepmeep.numba3d.zvel_cd`. */
double zvel_cd3(double t, const double *c, const double *dc, double *dvz);

/* Line-of-sight velocity and derivatives at an absolute time.

   Port of `meepmeep.numba3d.zvel_d`. */
double zvel_d3(double t, double tc, double p, const double *c, const double *dc,
               double te, double *dvz);

/* Radial-velocity scale factor s = k/n and its non-zero derivatives.

   Returns s and writes ds/dp, ds/da, ds/di, ds/de; the derivatives w.r.t.
   tc, w, and lan are identically zero. Hoist this out of per-sample loops:
   it depends only on the orbital parameters. Port of the numba helper
   `_rv_scale`. */
double rv_scale(double k, double p, double a, double i, double e, double *dsp,
                double *dsa, double *dsi, double *dse);

/* Radial velocity and derivatives at a centred time, hoisted-scale form.

   Takes the precomputed scale factor and its derivatives from `rv_scale`
   so a kernel looping over many samples per work item computes them once.
   drv: REAL[7]. Port of the numba helper `_rv_cd_w`. */
double rv_cd_w(double t, double s, double dsp, double dsa, double dsi,
               double dse, const double *c, const double *dc, double *drv);

/* Radial velocity and derivatives at a centred time.

   drv: REAL[7] (the derivative w.r.t. k is not included at the point
   level, matching numba; see rv_od in orbit3dd.cl). Port of
   `meepmeep.numba3d.rv_cd`. */
double rv_cd3(double t, double k, double p, double a, double i, double e,
              const double *c, const double *dc, double *drv);

/* Radial velocity and derivatives at an absolute time.

   Port of `meepmeep.numba3d.rv_d`. */
double rv_d3(double t, double k, double tc, double p, double a, double i,
             double e, const double *c, const double *dc, double te,
             double *drv);

/* Phase-angle cosine and its derivatives at a centred time.

   dca: REAL[7]. Chain rule
   d(-z/r)/dtheta = -dz/r + z (x dx + y dy + z dz) / r^3. Port of
   `meepmeep.numba3d.cos_alpha_cd`. */
double cos_alpha_cd3(double t, const double *c, const double *dc, double *dca);

/* Phase-angle cosine and derivatives at an absolute time.

   Port of `meepmeep.numba3d.cos_alpha_d`. */
double cos_alpha_d3(double t, double tc, double p, const double *c,
                    const double *dc, double te, double *dca);

/* Lambertian phase function, phase angle, and d(phase)/d(cos alpha).

   The derivative simplifies to (pi - alpha)/pi: the contributions from
   the sin(alpha) and alpha terms cancel exactly. `cos_alpha` is clamped
   to [-1, 1]. Port of the numba helper `_lambert_kernel_d`. */
double lambert_kernel_d(double cos_alpha, double *alpha, double *dphase_dc);

/* Lambertian phase curve and its derivatives at a centred time.

   dflux: REAL[9], ordered (tc, p, a, i, e, w, lan, ag, k). The orbital
   block chains through both the phase angle and the 1/r^2 illumination.
   Port of `meepmeep.numba3d.lambert_phase_curve_cd`. */
double lambert_phase_curve_cd3(double t, double ag, double k, const double *c,
                               const double *dc, double *dflux);

/* Lambertian phase curve and derivatives at an absolute time.

   Port of `meepmeep.numba3d.lambert_phase_curve_d`. */
double lambert_phase_curve_d3(double t, double ag, double k, double tc,
                              double p, const double *c, const double *dc,
                              double te, double *dflux);

/* Ellipsoidal-variation signal and its derivatives at a centred time.

   dout: REAL[9], ordered (tc, p, a, i, e, w, lan, alpha, mass_ratio).
   Inclination enters both implicitly through the position (orbital chain)
   and explicitly through the sin^2(inc) prefactor; both contributions sum
   into slot 3. Port of `meepmeep.numba3d.ev_signal_cd`. */
double ev_signal_cd3(double t, double alpha, double mass_ratio, double inc,
                     const double *c, const double *dc, double *dout);

/* Ellipsoidal-variation signal and derivatives at an absolute time.

   Port of `meepmeep.numba3d.ev_signal_d`. */
double ev_signal_d3(double t, double alpha, double mass_ratio, double inc,
                    double tc, double p, const double *c, const double *dc,
                    double te, double *dout);

/* Thermal-emission phase curve and its derivatives at a centred time.

   dout: REAL[10], ordered (tc, p, a, i, e, w, lan, k, fratio, offset).
   The orbital block chains through the phase-angle cosine cz = -z/d and
   the signed in-plane component s = -(wx*y - wy*x)/(|w| d) with w = r x v.
   Port of `meepmeep.numba3d.emission_phase_curve_cd`. */
double emission_phase_curve_cd3(double t, double k, double fratio,
                                double offset, const double *c,
                                const double *dc, double *dout);

/* Thermal-emission phase curve and derivatives at an absolute time.

   Port of `meepmeep.numba3d.emission_phase_curve_d`. */
double emission_phase_curve_d3(double t, double k, double fratio, double offset,
                               double tc, double p, const double *c,
                               const double *dc, double te, double *dout);


/* ---- orbit3d.cl ---------------------------------------------------------- */
/* Expansion-point lookup for an already-folded time tc in [0, p).

   Two guards absent from the numba original, which relies on in-range
   float-to-int behaviour a device cannot: (int)floor(NAN) is INT_MIN on
   NVIDIA, and an out-of-bounds __global read can kill the shared context
   for the whole process, so NaN returns index 0 (the following Horner
   evaluation then yields NaN gracefully, matching numba); and fp rounding
   can put tc exactly at p, so the bucket is clamped to the table length
   tres = 1/dt. */
int ep_lookup(double tc, double p, double dt, const int *ep_table);

/* Expansion-point index for an absolute time.

   Port of `backends.numba.orbit3d._common.ep_ix`. */
int ep_ix(double t, double tpa, double p, double dt, const int *ep_table);

/* Planet (x, y, z) position at any orbital phase.

   Port of `meepmeep.numba3d.pos_o`. */
void pos_o(double t, double tpa, double p, double dt, const int *ep_table,
           const double *ep_times, const double *coeffs, double *px, double *py,
           double *pz);

/* Line-of-sight z coordinate at any orbital phase.

   Port of `meepmeep.numba3d.zpos_o`. */
double zpos_o(double t, double tpa, double p, double dt, const int *ep_table,
              const double *ep_times, const double *coeffs);

/* Sky-projected planet-star separation at any orbital phase.

   Port of `meepmeep.numba3d.sep_o`. */
double sep_o(double t, double tpa, double p, double dt, const int *ep_table,
             const double *ep_times, const double *coeffs);

/* Planet (vx, vy, vz) velocity at any orbital phase.

   Port of `meepmeep.numba3d.vel_o`. */
void vel_o(double t, double tpa, double p, double dt, const int *ep_table,
           const double *ep_times, const double *coeffs, double *vx, double *vy,
           double *vz);

/* Line-of-sight velocity at any orbital phase.

   Port of `meepmeep.numba3d.zvel_o`. */
double zvel_o(double t, double tpa, double p, double dt, const int *ep_table,
              const double *ep_times, const double *coeffs);

/* Stellar radial velocity at any orbital phase.

   Port of `meepmeep.numba3d.rv_o`. */
double rv_o(double t, double k, double tpa, double p, double a, double i,
            double e, double dt, const int *ep_table, const double *ep_times,
            const double *coeffs);

/* Cosine of the star-planet-observer phase angle at any orbital phase.

   Port of `meepmeep.numba3d.cos_alpha_o`. */
double cos_alpha_o(double t, double tpa, double p, double dt,
                   const int *ep_table, const double *ep_times,
                   const double *coeffs);

/* Cosine of the angle between the planet position and a fixed vector.

   The numba original takes `v` as a 3-array; here it is three scalars
   (vx, vy, vz) so callers holding the vector in any address space can pass
   it without qualifier friction. Port of `meepmeep.numba3d.cos_v_p_angle_o`. */
double cos_v_p_angle_o(double vx, double vy, double vz, double t, double tpa,
                       double p, double dt, const int *ep_table,
                       const double *ep_times, const double *coeffs);

/* True anomaly from the position and the eccentricity vector (ex, ey, ez).

   The numba original is deliberately compiled without fastmath (the
   arccos argument sits near +-1 over much of the orbit); OpenCL strict
   math matches that. The early-return clamps are ported exactly. `w` is
   kept for signature parity with the numba dispatcher. Port of
   `meepmeep.numba3d.true_anomaly_o`. */
double true_anomaly_o(double t, double tpa, double p, double ex, double ey,
                      double ez, double w, double dt, const int *ep_table,
                      const double *ep_times, const double *coeffs);

/* Lambertian reflected-light phase curve at any orbital phase.

   Port of `meepmeep.numba3d.lambert_phase_curve_o`. */
double lambert_phase_curve_o(double t, double ag, double k, double tpa,
                             double p, double dt, const int *ep_table,
                             const double *ep_times, const double *coeffs);

/* Ellipsoidal-variation signal at any orbital phase.

   Port of `meepmeep.numba3d.ev_signal_o`. */
double ev_signal_o(double alpha, double mass_ratio, double inc, double t,
                   double tpa, double p, double dt, const int *ep_table,
                   const double *ep_times, const double *coeffs);

/* Thermal-emission phase curve at any orbital phase.

   Port of `meepmeep.numba3d.emission_phase_curve_o`. */
double emission_phase_curve_o(double t, double k, double fratio, double offset,
                              double tpa, double p, double dt,
                              const int *ep_table, const double *ep_times,
                              const double *coeffs);

/* Three-dimensional star-planet distance at any orbital phase.

   Port of `meepmeep.numba3d.star_planet_distance_o`. */
double star_planet_distance_o(double t, double tpa, double p, double dt,
                              const int *ep_table, const double *ep_times,
                              const double *coeffs);

/* Light-travel-time correction referenced to the primary transit.

   Positive when the signal from phase t arrives later than a transit-
   referenced clock expects. A kernel evaluating many times may hoist the
   transit-reference term z_tr = zpos_o(tpa + to, ...) host-side or into a
   pre-pass; this scalar port recomputes it per call, matching the numba
   scalar kernel. Port of `meepmeep.numba3d.light_travel_time_o`. */
double light_travel_time_o(double t, double tpa, double p, double e, double w,
                           double rstar, double dt, const int *ep_table,
                           const double *ep_times, const double *coeffs);


/* ---- orbit3dd.cl --------------------------------------------------------- */
/* Position and its (tc, p, a, i, e, w, lan) derivatives at any phase.

   Port of `meepmeep.numba3d.pos_od`. */
void pos_od(double t, double tpa, double p, double dt, const int *ep_table,
            const double *ep_times, const double *coeffs, const double *dcoeffs,
            double *px, double *py, double *pz, double *dpx, double *dpy,
            double *dpz);

/* Line-of-sight z and derivatives at any phase.

   Port of `meepmeep.numba3d.zpos_od`. */
double zpos_od(double t, double tpa, double p, double dt, const int *ep_table,
               const double *ep_times, const double *coeffs,
               const double *dcoeffs, double *dz);

/* Sky-projected separation and derivatives at any phase.

   Port of `meepmeep.numba3d.sep_od`. */
double sep_od(double t, double tpa, double p, double dt, const int *ep_table,
              const double *ep_times, const double *coeffs,
              const double *dcoeffs, double *dd);

/* Velocity and derivatives at any phase.

   Port of `meepmeep.numba3d.vel_od`. */
void vel_od(double t, double tpa, double p, double dt, const int *ep_table,
            const double *ep_times, const double *coeffs, const double *dcoeffs,
            double *vx, double *vy, double *vz, double *dvx, double *dvy,
            double *dvz);

/* Line-of-sight velocity and derivatives at any phase.

   Port of `meepmeep.numba3d.zvel_od`. */
double zvel_od(double t, double tpa, double p, double dt, const int *ep_table,
               const double *ep_times, const double *coeffs,
               const double *dcoeffs, double *dvz);

/* Stellar radial velocity and derivatives at any phase.

   drv: REAL[8], ordered (tc, p, a, i, e, w, lan, k); slot 7 is
   d(rv)/dk = rv/k, zero when k is zero. Port of `meepmeep.numba3d.rv_od`. */
double rv_od(double t, double k, double tpa, double p, double a, double i,
             double e, double dt, const int *ep_table, const double *ep_times,
             const double *coeffs, const double *dcoeffs, double *drv);

/* Phase-angle cosine and derivatives at any phase.

   Port of `meepmeep.numba3d.cos_alpha_od`. */
double cos_alpha_od(double t, double tpa, double p, double dt,
                    const int *ep_table, const double *ep_times,
                    const double *coeffs, const double *dcoeffs, double *dca);

/* Angle-to-fixed-vector cosine and derivatives at any phase.

   The numba original takes `v` as a 3-array; here it is three scalars
   (see cos_v_p_angle_o in orbit3d.cl). Port of
   `meepmeep.numba3d.cos_v_p_angle_od`. */
double cos_v_p_angle_od(double vx, double vy, double vz, double t, double tpa,
                        double p, double dt, const int *ep_table,
                        const double *ep_times, const double *coeffs,
                        const double *dcoeffs, double *dcs);

/* True anomaly and derivatives from the position and eccentricity vector.

   Strict math (the numba original deliberately drops fastmath: the acos
   argument sits near +-1 and the 1/sqrt(1 - edp^2) gradient denominator
   is near-singular). The gradient buffer is zeroed on entry because the
   early-return paths (the circular fast path leaves the a, i and lan slots
   zero; the edp clamps leave all slots zero) rely on it - the numba original
   allocates with zeros(7). `dev` (3 x 7, row-major) is the Jacobian of the
   eccentricity vector from `eccentricity_vector_d`; zeros hold the vector
   constant. It lives in private memory like `df`, so a kernel can compute it
   on the device. `timing_is_tc` states the basis of dcoeffs; only
   the circular fast path, which does not read dcoeffs, uses it (see
   `_circular_w` in the numba module). Port of `meepmeep.numba3d.true_anomaly_od`. */
double true_anomaly_od(double t, double tpa, double p, double ex, double ey,
                       double ez, double w, const double *dev, double dt,
                       const int *ep_table, const double *ep_times,
                       const double *coeffs, const double *dcoeffs,
                       int timing_is_tc, double *df);

/* Lambertian phase curve and derivatives at any phase.

   dflux: REAL[9]. Port of `meepmeep.numba3d.lambert_phase_curve_od`. */
double lambert_phase_curve_od(double t, double ag, double k, double tpa,
                              double p, double dt, const int *ep_table,
                              const double *ep_times, const double *coeffs,
                              const double *dcoeffs, double *dflux);

/* Ellipsoidal-variation signal and derivatives at any phase.

   dout: REAL[9]. Port of `meepmeep.numba3d.ev_signal_od`. */
double ev_signal_od(double alpha, double mass_ratio, double inc, double t,
                    double tpa, double p, double dt, const int *ep_table,
                    const double *ep_times, const double *coeffs,
                    const double *dcoeffs, double *dout);

/* Thermal-emission phase curve and derivatives at any phase.

   dout: REAL[10]. Port of `meepmeep.numba3d.emission_phase_curve_od`. */
double emission_phase_curve_od(double t, double k, double fratio, double offset,
                               double tpa, double p, double dt,
                               const int *ep_table, const double *ep_times,
                               const double *coeffs, const double *dcoeffs,
                               double *dout);

/* Three-dimensional star-planet distance and derivatives at any phase.

   Port of `meepmeep.numba3d.star_planet_distance_od`. */
double star_planet_distance_od(double t, double tpa, double p, double dt,
                               const int *ep_table, const double *ep_times,
                               const double *coeffs, const double *dcoeffs,
                               double *dr);

/* Line-of-sight z at the transit event and its total derivative.

   The total derivative of z(t_transit(theta); theta) combines the
   fixed-time gradient with v_z(t_transit) * dt_transit/dtheta, where
   dt_transit/dtheta depends on the bound timing basis: with the transit
   centre bound (timing_is_tc = 1, dcoeffs after tp_to_tc_gradient_orbit)
   only the timing slot is non-zero; with the periastron time bound
   (timing_is_tc = 0, the native solve3d_orbit_d basis) the p, e,
   and w slots join through t_o = M_tr(e, w) p / (2 pi). dz_tr: REAL[7].
   Port of the numba helper `_ltt_transit_z_and_d`. */
double ltt_transit_z_and_d(double tpa, double p, double e, double w, double dt,
                           const int *ep_table, const double *ep_times,
                           const double *coeffs, const double *dcoeffs,
                           int timing_is_tc, double *dz_tr);

/* Light-travel-time correction and derivatives at any phase.

   dltt: REAL[7] (no rstar slot, matching numba). `timing_is_tc` selects
   the timing basis of `dcoeffs` (see ltt_transit_z_and_d); the numba
   default is True (pass 1). A kernel evaluating many times may hoist
   ltt_transit_z_and_d host-side or into a pre-pass, as the numba vector
   kernels do - that is why the helper is public. Port of
   `meepmeep.numba3d.light_travel_time_od`. */
double light_travel_time_od(double t, double tpa, double p, double e, double w,
                            double rstar, double dt, const int *ep_table,
                            const double *ep_times, const double *coeffs,
                            const double *dcoeffs, int timing_is_tc,
                            double *dltt);

/* END GENERATED PROTOTYPES */

#ifdef __cplusplus
}
#endif

#endif /* MEEPMEEP_H */
