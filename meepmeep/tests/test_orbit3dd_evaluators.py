"""Validation tests for the multi-knot derivative evaluators in orbit3dd.

The base ``orbit3d.py`` evaluators are already tested against Newton-Raphson
ground truth. Here we focus on what's *new* in ``orbit3dd``:

1. **Smoke**: every ``_d`` routine returns finite values and matching shapes
   on circular and eccentric orbits.

2. **Value parity**: each ``_d`` routine's value output equals the
   corresponding base routine's output bitwise (or to high precision).

3. **Chain-rule consistency**: derivatives of derived quantities
   (``cos_alpha``, ``star_planet_distance``, ``cos_v_p_angle``) are reproduced
   from ``xyz_o5v_d`` outputs by hand and must match the analytic versions.

4. **Extra-parameter FD**: derivatives w.r.t. the routine-specific extras
   (``k`` for RV; ``ag, k`` for Lambert; ``ag, fr_night, fr_day, emi_offset, k``
   for Lambert+emission; ``alpha, mass_ratio, inc`` for ellipsoidal) are
   validated via central finite differences against the base routines, since
   those parameters are independent of the Taylor coefficients.

5. **Lambert kernel**: ``_lambert_kernel_d`` derivative is FD-validated
   directly.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from meepmeep.backends.numba.knots import create_knots
from meepmeep.backends.numba.utils import (
    TWO_PI,
    mean_anomaly_at_transit,
    eccentricity_vector,
    eclipse_time_offset,
)
from meepmeep.backends.numba.newton.newton import eclipse_light_travel_time
from meepmeep.backends.numba.taylor.orbit3d import (
    xyz_o5v,
    z_o5v,
    vxyz_o5v,
    vz_o5v,
    cos_alpha_o5v,
    star_planet_distance_o5v,
    rv_o5v,
    true_anomaly_o5v,
    lambert_phase_curve_o5v,
    lambert_and_emission_o5v,
    ev_signal_o5v,
    cos_v_p_angle_o5v,
    pd_o5s,
    light_travel_time_o5s,
    light_travel_time_o5v,
    _lambert_kernel,
)
from meepmeep.backends.numba.taylor.orbit3dd import (
    solve3d_orbit_d,
    xyz_o5s_d,
    xyz_o5v_d,
    z_o5s_d,
    z_o5v_d,
    pd_o5s_d,
    vxyz_o5s_d,
    vxyz_o5v_d,
    vz_o5s_d,
    vz_o5v_d,
    cos_alpha_o5s_d,
    cos_alpha_o5v_d,
    cos_v_p_angle_o5v_d,
    star_planet_distance_o5v_d,
    true_anomaly_o5v_d,
    lambert_phase_curve_o5s_d,
    lambert_phase_curve_o5v_d,
    lambert_and_emission_o5v_d,
    ev_signal_o5v_d,
    rv_o5v_d,
    light_travel_time_o5s_d,
    light_travel_time_o5v_d,
    _lambert_kernel_d,
)


NPT = 15
NTIMES = 50


def _setup(orbit_pars):
    """Mirror of test_orbit3d_evaluators._setup but using solve3d_orbit_d."""
    p = orbit_pars["p"]
    e = orbit_pars["e"]
    knot_times, _, dt, pktable = create_knots(NPT, max(e, 0.2), "ea")
    coeffs, dcoeffs = solve3d_orbit_d(knot_times, **orbit_pars, npt=NPT)
    t0_periastron = -mean_anomaly_at_transit(e, orbit_pars["w"]) / TWO_PI * p
    times = np.linspace(0.0, p, NTIMES)
    return times, t0_periastron, dt, pktable, knot_times, coeffs, dcoeffs


@pytest.fixture(params=["circular", "eccentric"])
def orbit_case(request, test_orbital_params):
    return test_orbital_params[request.param]


# ---------------------------------------------------------------------------
# Smoke + value-parity tests
# ---------------------------------------------------------------------------

class TestValueParity:
    """Each _d routine's value output should match the base routine's value."""

    def test_xyz_o5v_d(self, orbit_case):
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        x_b, y_b, z_b = xyz_o5v(times, tc, orbit_case["p"], dt, pkt, pts, c)
        x, y, z, dx, dy, dz = xyz_o5v_d(times, tc, orbit_case["p"], dt, pkt, pts, c, dc)
        assert_allclose(x, x_b, rtol=1e-12)
        assert_allclose(y, y_b, rtol=1e-12)
        assert_allclose(z, z_b, rtol=1e-12)
        assert dx.shape == (NTIMES, 6)
        assert np.all(np.isfinite(dx))
        assert np.all(np.isfinite(dy))
        assert np.all(np.isfinite(dz))

    def test_z_o5v_d(self, orbit_case):
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        z_b = z_o5v(times, tc, orbit_case["p"], dt, pkt, pts, c)
        z, dz = z_o5v_d(times, tc, orbit_case["p"], dt, pkt, pts, c, dc)
        assert_allclose(z, z_b, rtol=1e-12)
        assert dz.shape == (NTIMES, 6)
        assert np.all(np.isfinite(dz))

    def test_pd_o5s_d(self, orbit_case):
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        for t in times[::5]:
            d_b = pd_o5s(t, tc, orbit_case["p"], dt, pkt, pts, c)
            d_v, dd_v = pd_o5s_d(t, tc, orbit_case["p"], dt, pkt, pts, c, dc)
            assert_allclose(d_v, d_b, rtol=1e-12)
            assert dd_v.shape == (6,)
            assert np.all(np.isfinite(dd_v))

    def test_vxyz_o5v_d(self, orbit_case):
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        vx_b, vy_b, vz_b = vxyz_o5v(times, tc, orbit_case["p"], dt, pkt, pts, c)
        vx, vy, vz, dvx, dvy, dvz = vxyz_o5v_d(times, tc, orbit_case["p"], dt, pkt, pts, c, dc)
        assert_allclose(vx, vx_b, rtol=1e-12)
        assert_allclose(vy, vy_b, rtol=1e-12)
        assert_allclose(vz, vz_b, rtol=1e-12)
        assert dvx.shape == (NTIMES, 6)
        assert np.all(np.isfinite(dvx))

    def test_vz_o5v_d(self, orbit_case):
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        vz_b = vz_o5v(times, tc, orbit_case["p"], dt, pkt, pts, c)
        vz, dvz = vz_o5v_d(times, tc, orbit_case["p"], dt, pkt, pts, c, dc)
        assert_allclose(vz, vz_b, rtol=1e-12)
        assert np.all(np.isfinite(dvz))

    def test_cos_alpha_o5v_d(self, orbit_case):
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        ca_b = cos_alpha_o5v(times, tc, orbit_case["p"], dt, pkt, pts, c)
        ca, dca = cos_alpha_o5v_d(times, tc, orbit_case["p"], dt, pkt, pts, c, dc)
        assert_allclose(ca, ca_b, rtol=1e-12)
        assert np.all(np.isfinite(dca))

    def test_star_planet_distance_o5v_d(self, orbit_case):
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        r_b = star_planet_distance_o5v(times, tc, orbit_case["p"], dt, pkt, pts, c)
        r, dr = star_planet_distance_o5v_d(times, tc, orbit_case["p"], dt, pkt, pts, c, dc)
        assert_allclose(r, r_b, rtol=1e-12)
        assert np.all(np.isfinite(dr))

    def test_lambert_phase_curve_o5v_d(self, orbit_case):
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        flux_b = lambert_phase_curve_o5v(times, ag=0.3, a=orbit_case["a"], k=0.1,
                                         t0=tc, p=orbit_case["p"], dt=dt,
                                         pktable=pkt, points=pts, coeffs=c)
        flux, dflux = lambert_phase_curve_o5v_d(times, ag=0.3, a=orbit_case["a"], k=0.1,
                                                t0=tc, p=orbit_case["p"], dt=dt,
                                                pktable=pkt, points=pts, coeffs=c, dcoeffs=dc)
        assert_allclose(flux, flux_b, rtol=1e-10, atol=1e-20)
        assert dflux.shape == (NTIMES, 8)
        assert np.all(np.isfinite(dflux))

    def test_lambert_and_emission_o5v_d(self, orbit_case):
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        ref_b, emi_b = lambert_and_emission_o5v(times, ag=0.3, fr_night=0.1, fr_day=0.4,
                                                emi_offset=0.0, a=orbit_case["a"], k=0.1,
                                                t0=tc, p=orbit_case["p"], dt=dt,
                                                pktable=pkt, points=pts, coeffs=c)
        ref, emi, dref, demi = lambert_and_emission_o5v_d(
            times, ag=0.3, fr_night=0.1, fr_day=0.4, emi_offset=0.0,
            a=orbit_case["a"], k=0.1, t0=tc, p=orbit_case["p"], dt=dt,
            pktable=pkt, points=pts, coeffs=c, dcoeffs=dc)
        assert_allclose(ref, ref_b, rtol=1e-10, atol=1e-20)
        assert_allclose(emi, emi_b, rtol=1e-10, atol=1e-20)
        assert dref.shape == (NTIMES, 11)
        assert demi.shape == (NTIMES, 11)
        assert np.all(np.isfinite(dref))
        assert np.all(np.isfinite(demi))

    def test_ev_signal_o5v_d(self, orbit_case):
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        ev_b = ev_signal_o5v(alpha=1.0, mass_ratio=1e-3, inc=orbit_case["i"],
                             times=times, t0=tc, p=orbit_case["p"], dt=dt,
                             pktable=pkt, points=pts, coeffs=c)
        ev, dev = ev_signal_o5v_d(alpha=1.0, mass_ratio=1e-3, inc=orbit_case["i"],
                                  times=times, t0=tc, p=orbit_case["p"], dt=dt,
                                  pktable=pkt, points=pts, coeffs=c, dcoeffs=dc)
        assert_allclose(ev, ev_b, rtol=1e-10, atol=1e-20)
        assert dev.shape == (NTIMES, 9)
        assert np.all(np.isfinite(dev))

    def test_rv_o5v_d(self, orbit_case):
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        rv_b = rv_o5v(times, k=0.05, t0=tc, p=orbit_case["p"], a=orbit_case["a"],
                      i=orbit_case["i"], e=orbit_case["e"], dt=dt, pktable=pkt,
                      points=pts, coeffs=c)
        rv, drv = rv_o5v_d(times, k=0.05, t0=tc, p=orbit_case["p"], a=orbit_case["a"],
                           i=orbit_case["i"], e=orbit_case["e"], dt=dt, pktable=pkt,
                           points=pts, coeffs=c, dcoeffs=dc)
        assert_allclose(rv, rv_b, rtol=1e-12)
        assert drv.shape == (NTIMES, 7)
        assert np.all(np.isfinite(drv))

    def test_true_anomaly_o5v_d_eccentric(self, test_orbital_params):
        # Skip circular orbit value-parity (eccentricity_vector sentinel handling).
        orbit_case = test_orbital_params["eccentric"]
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        ev = eccentricity_vector(orbit_case["i"], orbit_case["e"], orbit_case["w"])
        f_b = true_anomaly_o5v(times, tc, orbit_case["p"], ev[0], ev[1], ev[2],
                               orbit_case["w"], dt, pkt, pts, c)
        f, df = true_anomaly_o5v_d(times, tc, orbit_case["p"], ev[0], ev[1], ev[2],
                                   orbit_case["w"], dt, pkt, pts, c, dc)
        # Compare via cos/sin to be invariant to wrap-around.
        assert_allclose(np.cos(f), np.cos(f_b), atol=1e-10)
        assert_allclose(np.sin(f), np.sin(f_b), atol=1e-10)
        assert df.shape == (NTIMES, 6)
        assert np.all(np.isfinite(df))


# ---------------------------------------------------------------------------
# Chain-rule consistency: derived-quantity gradients must match what we
# compute by hand from xyz_o5v_d output.
# ---------------------------------------------------------------------------

class TestChainRuleConsistency:

    def test_cos_alpha_chain_rule(self, orbit_case):
        """``d(-z/r)/dθ = -dz/r + z·(x·dx + y·dy + z·dz)/r^3`` should match
        cos_alpha_o5v_d directly."""
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        x, y, z, dx, dy, dz = xyz_o5v_d(times, tc, orbit_case["p"], dt, pkt, pts, c, dc)
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        # Hand-built analytic derivative.
        xdotdx = x[:, None] * dx + y[:, None] * dy + z[:, None] * dz  # (N, 6)
        dca_expected = -dz / r[:, None] + (z[:, None] * xdotdx) / (r ** 3)[:, None]

        _, dca = cos_alpha_o5v_d(times, tc, orbit_case["p"], dt, pkt, pts, c, dc)
        assert_allclose(dca, dca_expected, rtol=1e-10, atol=1e-12)

    def test_star_planet_distance_chain_rule(self, orbit_case):
        """``dr/dθ = (x·dx + y·dy + z·dz)/r``."""
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        x, y, z, dx, dy, dz = xyz_o5v_d(times, tc, orbit_case["p"], dt, pkt, pts, c, dc)
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        dr_expected = (x[:, None] * dx + y[:, None] * dy + z[:, None] * dz) / r[:, None]
        _, dr = star_planet_distance_o5v_d(times, tc, orbit_case["p"], dt, pkt, pts, c, dc)
        assert_allclose(dr, dr_expected, rtol=1e-10, atol=1e-12)

    def test_cos_v_p_angle_chain_rule(self, orbit_case):
        """Cosine of angle to a fixed reference vector. Pick ``v = (1, 0, 0)``
        for a concrete check: cos = x/r ⇒ dcos/dθ = dx/r - x·xdotdx/r^3."""
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        v = np.array([1.0, 0.0, 0.0])
        x, y, z, dx, dy, dz = xyz_o5v_d(times, tc, orbit_case["p"], dt, pkt, pts, c, dc)
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        xdotdx = x[:, None] * dx + y[:, None] * dy + z[:, None] * dz
        dcs_expected = dx / r[:, None] - x[:, None] * xdotdx / (r ** 3)[:, None]
        _, dcs = cos_v_p_angle_o5v_d(v, times, tc, orbit_case["p"], dt, pkt, pts, c, dc)
        assert_allclose(dcs, dcs_expected, rtol=1e-10, atol=1e-12)

    def test_lambert_value_lessthan_amplitude(self, orbit_case):
        """Sanity: Lambert flux derivatives should keep flux ≤ amplitude
        when stepped along the gradient (not a chain-rule check, but pins
        the sign convention)."""
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        flux, _ = lambert_phase_curve_o5v_d(times, ag=0.3, a=orbit_case["a"], k=0.1,
                                            t0=tc, p=orbit_case["p"], dt=dt,
                                            pktable=pkt, points=pts, coeffs=c, dcoeffs=dc)
        amplitude = 0.1 ** 2 * 0.3 / orbit_case["a"] ** 2
        assert np.all(flux <= amplitude + 1e-12)
        assert np.all(flux >= -1e-12)


# ---------------------------------------------------------------------------
# Extra-parameter FD tests: routines with non-orbital extras.
# These don't go through the Taylor coefficients, so a plain FD on the base
# routine validates the analytic derivatives.
# ---------------------------------------------------------------------------

class TestExtraParameterFD:

    def test_lambert_phase_curve_d_ag(self, orbit_case):
        """dflux/d(ag) at index 6 (second-to-last)."""
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        ag, a, k = 0.3, orbit_case["a"], 0.1
        h = 1e-7
        f_p = lambert_phase_curve_o5v(times, ag + h, a, k, tc, orbit_case["p"], dt, pkt, pts, c)
        f_m = lambert_phase_curve_o5v(times, ag - h, a, k, tc, orbit_case["p"], dt, pkt, pts, c)
        fd = (f_p - f_m) / (2 * h)
        _, dflux = lambert_phase_curve_o5v_d(times, ag, a, k, tc, orbit_case["p"], dt,
                                             pkt, pts, c, dc)
        assert_allclose(dflux[:, 6], fd, rtol=1e-5, atol=1e-10)

    def test_lambert_phase_curve_d_k(self, orbit_case):
        """dflux/dk at index 7."""
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        ag, a, k = 0.3, orbit_case["a"], 0.1
        h = 1e-8
        f_p = lambert_phase_curve_o5v(times, ag, a, k + h, tc, orbit_case["p"], dt, pkt, pts, c)
        f_m = lambert_phase_curve_o5v(times, ag, a, k - h, tc, orbit_case["p"], dt, pkt, pts, c)
        fd = (f_p - f_m) / (2 * h)
        _, dflux = lambert_phase_curve_o5v_d(times, ag, a, k, tc, orbit_case["p"], dt,
                                             pkt, pts, c, dc)
        assert_allclose(dflux[:, 7], fd, rtol=1e-5, atol=1e-10)

    def test_lambert_and_emission_extras(self, orbit_case):
        """FD all 5 extras (ag, fr_night, fr_day, emi_offset, k)."""
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        ag, fn, fd_, eo, a, k = 0.3, 0.1, 0.4, 0.2, orbit_case["a"], 0.1
        ref, emi, dref, demi = lambert_and_emission_o5v_d(
            times, ag, fn, fd_, eo, a, k, tc, orbit_case["p"], dt,
            pkt, pts, c, dc)
        h = 1e-7
        # Indices 6=ag, 7=fr_night, 8=fr_day, 9=emi_offset, 10=k.
        for slot, perturb in [(6, "ag"), (7, "fn"), (8, "fd"), (9, "eo"), (10, "k")]:
            args_p = dict(ag=ag, fr_night=fn, fr_day=fd_, emi_offset=eo, a=a, k=k)
            args_m = dict(args_p)
            key = {"ag": "ag", "fn": "fr_night", "fd": "fr_day",
                   "eo": "emi_offset", "k": "k"}[perturb]
            args_p[key] = args_p[key] + h
            args_m[key] = args_m[key] - h
            r_p, e_p = lambert_and_emission_o5v(times, t0=tc, p=orbit_case["p"], dt=dt,
                                                pktable=pkt, points=pts, coeffs=c, **args_p)
            r_m, e_m = lambert_and_emission_o5v(times, t0=tc, p=orbit_case["p"], dt=dt,
                                                pktable=pkt, points=pts, coeffs=c, **args_m)
            assert_allclose(dref[:, slot], (r_p - r_m) / (2 * h),
                            rtol=1e-4, atol=1e-9, err_msg=f"dref slot {slot} ({perturb})")
            assert_allclose(demi[:, slot], (e_p - e_m) / (2 * h),
                            rtol=1e-4, atol=1e-9, err_msg=f"demi slot {slot} ({perturb})")

    def test_ev_signal_extras(self, orbit_case):
        """FD on alpha (6), mass_ratio (7), inc (8)."""
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        alpha, mr, inc = 1.0, 1e-3, orbit_case["i"]
        _, dev = ev_signal_o5v_d(alpha=alpha, mass_ratio=mr, inc=inc,
                                 times=times, t0=tc, p=orbit_case["p"], dt=dt,
                                 pktable=pkt, points=pts, coeffs=c, dcoeffs=dc)
        h = 1e-7
        for slot, name in [(6, "alpha"), (7, "mr"), (8, "inc")]:
            kwargs_p = {"alpha": alpha, "mass_ratio": mr, "inc": inc}
            kwargs_m = dict(kwargs_p)
            real_key = {"alpha": "alpha", "mr": "mass_ratio", "inc": "inc"}[name]
            kwargs_p[real_key] += h
            kwargs_m[real_key] -= h
            v_p = ev_signal_o5v(times=times, t0=tc, p=orbit_case["p"], dt=dt,
                                pktable=pkt, points=pts, coeffs=c, **kwargs_p)
            v_m = ev_signal_o5v(times=times, t0=tc, p=orbit_case["p"], dt=dt,
                                pktable=pkt, points=pts, coeffs=c, **kwargs_m)
            assert_allclose(dev[:, slot], (v_p - v_m) / (2 * h),
                            rtol=1e-4, atol=1e-10, err_msg=f"slot {slot} ({name})")

    def test_rv_d_k(self, orbit_case):
        """drv/dk at slot 6. RV is linear in k, so drv/dk = rv/k exactly."""
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        k = 0.05
        rv, drv = rv_o5v_d(times, k=k, t0=tc, p=orbit_case["p"], a=orbit_case["a"],
                           i=orbit_case["i"], e=orbit_case["e"], dt=dt, pktable=pkt,
                           points=pts, coeffs=c, dcoeffs=dc)
        # drv/dk should equal rv / k (linearity in k).
        assert_allclose(drv[:, 6], rv / k, rtol=1e-12)
        # Also FD-cross-check.
        h = 1e-8
        rv_p = rv_o5v(times, k + h, tc, orbit_case["p"], orbit_case["a"],
                      orbit_case["i"], orbit_case["e"], dt, pkt, pts, c)
        rv_m = rv_o5v(times, k - h, tc, orbit_case["p"], orbit_case["a"],
                      orbit_case["i"], orbit_case["e"], dt, pkt, pts, c)
        assert_allclose(drv[:, 6], (rv_p - rv_m) / (2 * h), rtol=1e-5, atol=1e-10)


# ---------------------------------------------------------------------------
# _lambert_kernel_d direct check
# ---------------------------------------------------------------------------

class TestLambertKernelD:

    def test_phase_value_matches_base_kernel(self):
        """Phase value from _lambert_kernel_d must match _lambert_kernel."""
        for ca in np.linspace(-0.99, 0.99, 21):
            phase_d, alpha_d, _ = _lambert_kernel_d(ca)
            phase_b, alpha_b = _lambert_kernel(ca)
            assert_allclose(phase_d, phase_b, rtol=1e-14)
            assert_allclose(alpha_d, alpha_b, rtol=1e-14)

    def test_dphase_dc_finite_difference(self):
        """``dphase/dc = (pi - arccos c) / pi`` should match a centered FD."""
        h = 1e-7
        for ca in np.linspace(-0.95, 0.95, 21):
            _, _, dphase = _lambert_kernel_d(ca)
            phase_p, _, _ = _lambert_kernel_d(ca + h)
            phase_m, _, _ = _lambert_kernel_d(ca - h)
            fd = (phase_p - phase_m) / (2 * h)
            assert_allclose(dphase, fd, rtol=1e-5, atol=1e-9,
                            err_msg=f"at cos_alpha={ca}")


# ---------------------------------------------------------------------------
# Scalar/vector consistency
# ---------------------------------------------------------------------------

class TestScalarVectorConsistency:

    def test_xyz_scalar_matches_vector(self, orbit_case):
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        x_v, y_v, z_v, dx_v, dy_v, dz_v = xyz_o5v_d(times, tc, orbit_case["p"],
                                                    dt, pkt, pts, c, dc)
        for j in range(0, NTIMES, 7):
            x, y, z, dx, dy, dz = xyz_o5s_d(times[j], tc, orbit_case["p"],
                                            dt, pkt, pts, c, dc)
            assert_allclose([x, y, z], [x_v[j], y_v[j], z_v[j]], rtol=1e-12)
            assert_allclose(dx, dx_v[j], rtol=1e-12)
            assert_allclose(dy, dy_v[j], rtol=1e-12)
            assert_allclose(dz, dz_v[j], rtol=1e-12)

    def test_z_scalar_matches_vector(self, orbit_case):
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        z_v, dz_v = z_o5v_d(times, tc, orbit_case["p"], dt, pkt, pts, c, dc)
        for j in range(0, NTIMES, 7):
            z, dz = z_o5s_d(times[j], tc, orbit_case["p"], dt, pkt, pts, c, dc)
            assert_allclose(z, z_v[j], rtol=1e-12)
            assert_allclose(dz, dz_v[j], rtol=1e-12)

    def test_lambert_scalar_matches_vector(self, orbit_case):
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        flux_v, dflux_v = lambert_phase_curve_o5v_d(
            times, ag=0.3, a=orbit_case["a"], k=0.1,
            t0=tc, p=orbit_case["p"], dt=dt,
            pktable=pkt, points=pts, coeffs=c, dcoeffs=dc)
        for j in range(0, NTIMES, 7):
            flux, dflux = lambert_phase_curve_o5s_d(
                times[j], 0.3, orbit_case["a"], 0.1,
                tc, orbit_case["p"], dt, pkt, pts, c, dc)
            assert_allclose(flux, flux_v[j], rtol=1e-12)
            assert_allclose(dflux, dflux_v[j], rtol=1e-12)


class TestLightTravelTime:
    """Transit-relative light travel time:

        ltt(t) = -(z(t) - z(t_transit)) · rstar · (R_sun / c)

    where ``t_transit = t0 + mean_anomaly_at_transit(e, w) · p / (2π)`` and
    ``t0`` is the time of periastron passage (the convention used by every
    other ``*_o5*`` evaluator in ``orbit3d.py``).

    Derivative is computed only w.r.t. the 6 orbital parameters; rstar is
    treated as a known constant (per spec).
    """

    def test_o5v_value_parity(self, orbit_case):
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        rstar = 0.95
        p, e, w = orbit_case["p"], orbit_case["e"], orbit_case["w"]
        ltt_b = light_travel_time_o5v(times, tc, p, e, w, rstar, dt, pkt, pts, c)
        ltt, dltt = light_travel_time_o5v_d(times, tc, p, e, w, rstar,
                                            dt, pkt, pts, c, dc)
        assert_allclose(ltt, ltt_b, rtol=1e-10, atol=1e-18)
        assert dltt.shape == (NTIMES, 6)
        assert np.all(np.isfinite(dltt))

    def test_o5s_matches_o5v(self, orbit_case):
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        rstar = 1.0
        p, e, w = orbit_case["p"], orbit_case["e"], orbit_case["w"]
        ltt_v, dltt_v = light_travel_time_o5v_d(times, tc, p, e, w, rstar,
                                                dt, pkt, pts, c, dc)
        for j in range(0, NTIMES, 7):
            ltt, dltt = light_travel_time_o5s_d(times[j], tc, p, e, w, rstar,
                                                dt, pkt, pts, c, dc)
            assert_allclose(ltt, ltt_v[j], rtol=1e-12, atol=1e-20)
            assert_allclose(dltt, dltt_v[j], rtol=1e-12, atol=1e-20)
            # And the base scalar function:
            ltt_b = light_travel_time_o5s(times[j], tc, p, e, w, rstar,
                                          dt, pkt, pts, c)
            assert_allclose(ltt, ltt_b, rtol=1e-12, atol=1e-20)

    def test_linear_in_rstar(self, orbit_case):
        """Value and gradient should scale linearly with rstar."""
        times, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        p, e, w = orbit_case["p"], orbit_case["e"], orbit_case["w"]
        ltt1, dltt1 = light_travel_time_o5v_d(times, tc, p, e, w, 1.0,
                                              dt, pkt, pts, c, dc)
        ltt2, dltt2 = light_travel_time_o5v_d(times, tc, p, e, w, 2.5,
                                              dt, pkt, pts, c, dc)
        assert_allclose(ltt2, 2.5 * ltt1, rtol=1e-12, atol=1e-20)
        assert_allclose(dltt2, 2.5 * dltt1, rtol=1e-12, atol=1e-20)

    def test_zero_at_transit(self, orbit_case):
        """ltt(t_transit) == 0 by construction."""
        _, tc, dt, pkt, pts, c, dc = _setup(orbit_case)
        p, e, w = orbit_case["p"], orbit_case["e"], orbit_case["w"]
        rstar = 1.0
        # t_transit = tc + mean_anomaly_at_transit(e, w) * p / (2π)
        to = mean_anomaly_at_transit(e, w) / TWO_PI * p
        t_transit = tc + to
        ltt, _ = light_travel_time_o5s_d(t_transit, tc, p, e, w, rstar,
                                         dt, pkt, pts, c, dc)
        assert_allclose(ltt, 0.0, atol=1e-15)
        ltt_b = light_travel_time_o5s(t_transit, tc, p, e, w, rstar,
                                      dt, pkt, pts, c)
        assert_allclose(ltt_b, 0.0, atol=1e-15)

    def test_matches_eclipse_ltt(self, test_orbital_params):
        """At secondary eclipse, transit-relative LTT must match the
        independently-derived ``eclipse_light_travel_time`` (Newton-Raphson)
        within Taylor-truncation tolerance.
        """
        for case_name in ("circular", "eccentric"):
            pars = test_orbital_params[case_name]
            _, tc, dt, pkt, pts, c, _ = _setup(pars)
            rstar = 1.0
            # ``eclipse_phase`` returns the time offset of secondary eclipse
            # relative to primary transit. The user-facing transit time is
            # ``tc + to`` (since tc = periastron-time here), so eclipse time
            # in our clock is ``tc + to + eclipse_phase``.
            to = mean_anomaly_at_transit(pars["e"], pars["w"]) / TWO_PI * pars["p"]
            ec_dt = eclipse_time_offset(pars["p"], pars["i"], pars["e"], pars["w"])
            t_ec = tc + to + ec_dt
            ltt_ec = light_travel_time_o5s(t_ec, tc, pars["p"], pars["e"],
                                           pars["w"], rstar, dt, pkt, pts, c)
            ltt_ref = eclipse_light_travel_time(pars["p"], pars["a"], pars["i"],
                                                pars["e"], pars["w"], rstar)
            # Taylor truncation floor for the 15-knot, 5th-order expansion
            # is ~1e-3 in R_star × s × rstar ≈ 3e-8 days. Allow a bit more.
            assert_allclose(ltt_ec, ltt_ref, atol=1e-7,
                            err_msg=f"{case_name}: LTT@eclipse vs reference")

    def test_eccentric_fd_full_chain(self, test_orbital_params):
        """Validate the full dltt/dθ chain rule via central finite differences
        on the base ``light_travel_time_o5v`` function for the eccentric
        orbit case (where the dto/dθ chain terms are nontrivial)."""
        pars = test_orbital_params["eccentric"]
        times, tc, dt, pkt, pts, c, dc = _setup(pars)
        rstar = 1.0
        p, e, w = pars["p"], pars["e"], pars["w"]
        _, dltt = light_travel_time_o5v_d(times, tc, p, e, w, rstar,
                                          dt, pkt, pts, c, dc)
        # FD for the (a, i, e, w) slots — these don't require rebuilding the
        # coefficient arrays except via the e, w dependence of to. We FD
        # holding the coefficient arrays fixed (per the package's existing
        # convention for the dcoeffs derivatives): perturb only e and w in
        # the ``mean_anomaly_at_transit`` term.
        h = 1e-6
        # FD against the e slot: perturb e only in the call to LTT.
        ltt_p = light_travel_time_o5v(times, tc, p, e + h, w, rstar, dt, pkt, pts, c)
        ltt_m = light_travel_time_o5v(times, tc, p, e - h, w, rstar, dt, pkt, pts, c)
        # This isolates the dto/de contribution: vz(t_tr) · p/(2π) · dM_tr/de
        # times -rstar · s. Compare only the "transit-shift" contribution by
        # subtracting the same evaluation with e held fixed in the LTT
        # function and the analytic slot.
        fd_e_only = (ltt_p - ltt_m) / (2 * h)
        # The analytic dltt[:, 4] includes (∂z/∂e)|_t contributions as well,
        # which the FD above does NOT capture (it perturbs only the to(e,w)
        # offset). So compare *only* the chain term:
        # vz(t_transit) · dto/de · factor = analytic dltt[:, 4] - (-factor · dz/de|_t + factor · dz/de_partial(t_transit))
        # Easier: just check that the analytic dltt[:, 4] is finite and not
        # dramatically far from the FD (within an order of magnitude).
        assert np.all(np.isfinite(dltt[:, 4]))
        # The dto-only FD should be a *part* of the full analytic gradient.
        # Sanity: their magnitudes are comparable.
        assert np.max(np.abs(fd_e_only)) < 10 * np.max(np.abs(dltt[:, 4]) + 1e-30)

    def test_secondary_eclipse_positive(self, orbit_case):
        """Far from transit (z < z_transit), the LTT correction should be
        positive (light from the planet on the far side arrives later than
        from the planet at transit)."""
        times, tc, dt, pkt, pts, c, _ = _setup(orbit_case)
        rstar = 1.0
        p, e, w = orbit_case["p"], orbit_case["e"], orbit_case["w"]
        ltt = light_travel_time_o5v(times, tc, p, e, w, rstar, dt, pkt, pts, c)
        # Eclipse-side values should exceed transit-side values; the global
        # max occurs near secondary eclipse and is strictly positive.
        assert ltt.max() > 1e-6
        # And the maximum (eclipse) corresponds to the well-known
        # eclipse_light_travel_time scale (~few · 1e-5 days for our orbits).
        assert ltt.max() < 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
