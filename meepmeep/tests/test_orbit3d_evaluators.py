"""Regression tests for the multi-expansion-point Taylor evaluators in orbit3d.

Each test compares the Taylor-series result against the Newton-Raphson
ground truth from ``backends.numba.newton.newton``. These tests guard the
``extended3d.py`` -> ``orbit3d.py`` migration: they exercise the whole
``solve3d_orbit`` -> evaluator path with the new ``(npt, 3, 5)`` coefficient
layout.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from meepmeep.backends.numba.expansion_points import create_expansion_points
from meepmeep.backends.numba.newton.newton import (
    ta_newton_v,
    xyz_newton_v,
    rv_newton_v,
)
from meepmeep.backends.numba.orbit3d import (
    solve3d_orbit,
    pos_o,
    vel_o,
    zvel_o,
    zpos_o,
    cos_alpha_o,
    star_planet_distance_o,
    rv_o,
    true_anomaly_o,
    lambert_phase_curve_o,
    ev_signal_o,
)
from meepmeep.backends.numba.utils import (
    TWO_PI,
    mean_anomaly_at_transit,
    eccentricity_vector,
)
from meepmeep.orbit import Orbit


NPT = 15
NTIMES = 200
# Tolerance for full-period comparisons. With npt=15 expansion points and a 5th-order
# Taylor expansion, the worst-case error away from expansion points is ~1e-4 in
# (R_star) units. Tightening would require more expansion points; this matches the
# package's documented accuracy regime.
TOL = {"rtol": 1e-3, "atol": 1e-3}


def _setup(orbit_pars):
    """Build the (expansion point table, coeffs, t0_periastron) triple expected by the evaluators."""
    p = orbit_pars["p"]
    e = orbit_pars["e"]
    w = orbit_pars["w"]
    ep_times, _, dt, ep_table = create_expansion_points(NPT, max(e, 0.2), "ea")
    coeffs = solve3d_orbit(ep_times, **orbit_pars, npt=NPT)
    t0_periastron = -mean_anomaly_at_transit(e, w) / TWO_PI * p
    times = np.linspace(0.0, p, NTIMES)
    return times, t0_periastron, dt, ep_table, ep_times, coeffs


@pytest.fixture(params=["circular", "eccentric"])
def orbit_case(request, test_orbital_params):
    return test_orbital_params[request.param]


class TestPositionEvaluators:
    def test_xyz_matches_newton(self, orbit_case):
        times, tc, dt, pkt, pts, c = _setup(orbit_case)
        x, y, z = pos_o(times, tc, orbit_case["p"], dt, pkt, pts, c)
        xn, yn, zn = xyz_newton_v(times, 0.0, **orbit_case)
        assert_allclose(x, xn, **TOL)
        assert_allclose(y, yn, **TOL)
        assert_allclose(z, zn, **TOL)

    def test_z_only_matches_xyz(self, orbit_case):
        times, tc, dt, pkt, pts, c = _setup(orbit_case)
        _, _, z_full = pos_o(times, tc, orbit_case["p"], dt, pkt, pts, c)
        z_only = zpos_o(times, tc, orbit_case["p"], dt, pkt, pts, c)
        assert_allclose(z_only, z_full, rtol=1e-12)

    def test_star_planet_distance(self, orbit_case):
        times, tc, dt, pkt, pts, c = _setup(orbit_case)
        d = star_planet_distance_o(times, tc, orbit_case["p"], dt, pkt, pts, c)
        xn, yn, zn = xyz_newton_v(times, 0.0, **orbit_case)
        dn = np.sqrt(xn ** 2 + yn ** 2 + zn ** 2)
        assert_allclose(d, dn, **TOL)


class TestVelocityEvaluators:
    def test_vxyz_finite_difference(self, orbit_case):
        """Velocity from the Taylor expansion should match a centered finite
        difference of the Newton position."""
        times, tc, dt, pkt, pts, c = _setup(orbit_case)
        vx, vy, vz = vel_o(times, tc, orbit_case["p"], dt, pkt, pts, c)

        h = 1e-4
        xp, yp, zp = xyz_newton_v(times + h, 0.0, **orbit_case)
        xm, ym, zm = xyz_newton_v(times - h, 0.0, **orbit_case)
        vx_fd = (xp - xm) / (2 * h)
        vy_fd = (yp - ym) / (2 * h)
        vz_fd = (zp - zm) / (2 * h)
        assert_allclose(vx, vx_fd, rtol=1e-2, atol=2e-2)
        assert_allclose(vy, vy_fd, rtol=1e-2, atol=2e-2)
        assert_allclose(vz, vz_fd, rtol=1e-2, atol=2e-2)

    def test_vz_consistent_with_vxyz(self, orbit_case):
        times, tc, dt, pkt, pts, c = _setup(orbit_case)
        _, _, vz_full = vel_o(times, tc, orbit_case["p"], dt, pkt, pts, c)
        vz_only = zvel_o(times, tc, orbit_case["p"], dt, pkt, pts, c)
        assert_allclose(vz_only, vz_full, rtol=1e-12)


class TestPhaseAngles:
    def test_cos_alpha_matches_xyz(self, orbit_case):
        times, tc, dt, pkt, pts, c = _setup(orbit_case)
        ca = cos_alpha_o(times, tc, orbit_case["p"], dt, pkt, pts, c)
        xn, yn, zn = xyz_newton_v(times, 0.0, **orbit_case)
        ca_ref = -zn / np.sqrt(xn ** 2 + yn ** 2 + zn ** 2)
        assert_allclose(ca, ca_ref, **TOL)

    def test_true_anomaly_matches_newton(self, test_orbital_params):
        # Skip circular orbit: the eccentricity vector has zero magnitude there
        # so the geometric (xyz vs ev) definition of true anomaly is undefined.
        orbit_case = test_orbital_params["eccentric"]
        times, tc, dt, pkt, pts, c = _setup(orbit_case)
        ev = eccentricity_vector(orbit_case["i"], orbit_case["e"], orbit_case["w"])
        f_ts = true_anomaly_o(times, tc, orbit_case["p"], ev[0], ev[1], ev[2],
                               orbit_case["w"], dt, pkt, pts, c)
        f_nr = ta_newton_v(times, 0.0, orbit_case["p"], orbit_case["e"], orbit_case["w"])
        # Compare via cos/sin to be invariant to 2*pi wrap-around discontinuities.
        assert_allclose(np.cos(f_ts), np.cos(f_nr), atol=1e-3)
        assert_allclose(np.sin(f_ts), np.sin(f_nr), atol=1e-3)

    def test_true_anomaly_circular_matches_newton(self, test_orbital_params):
        """The circular-orbit fast path (eccentricity-vector sentinel) must
        return the e -> 0 limit of the geometric definition, which with the
        periastron anchor is simply f = 2*pi*(t - tpa)/p. The Newton solver
        anchored at the transit centre provides the reference."""
        orbit_case = test_orbital_params["circular"]
        times, tpa, dt, pkt, pts, c = _setup(orbit_case)
        ev = eccentricity_vector(orbit_case["i"], orbit_case["e"], orbit_case["w"])
        f_ts = true_anomaly_o(times, tpa, orbit_case["p"], ev[0], ev[1], ev[2],
                              orbit_case["w"], dt, pkt, pts, c)
        f_nr = ta_newton_v(times, 0.0, orbit_case["p"], orbit_case["e"], orbit_case["w"])
        # Both sides are closed-form here, so the tolerance is tight.
        assert_allclose(np.cos(f_ts), np.cos(f_nr), atol=1e-9)
        assert_allclose(np.sin(f_ts), np.sin(f_nr), atol=1e-9)

    def test_true_anomaly_near_circular_branch(self, test_orbital_params):
        """Just above the eccentricity-vector sentinel (e = 2e-5) the r.v
        signal is O(e) and smaller than the Taylor truncation noise, so the
        arccos branch selection must come from an exact quantity (the mean
        anomaly), not from the sign of r.v. The sin comparison is the
        branch-sensitive one; cos is branch-insensitive."""
        pars = dict(test_orbital_params["circular"])
        pars["e"] = 2e-5
        times, tpa, dt, pkt, pts, c = _setup(pars)
        ev = eccentricity_vector(pars["i"], pars["e"], pars["w"])
        f_ts = true_anomaly_o(times, tpa, pars["p"], ev[0], ev[1], ev[2],
                              pars["w"], dt, pkt, pts, c)
        f_nr = ta_newton_v(times, 0.0, pars["p"], pars["e"], pars["w"])
        assert_allclose(np.cos(f_ts), np.cos(f_nr), atol=1e-3)
        assert_allclose(np.sin(f_ts), np.sin(f_nr), atol=1e-3)


class TestRadialVelocity:
    def test_rv_matches_newton(self, orbit_case):
        times, tc, dt, pkt, pts, c = _setup(orbit_case)
        k = 0.05
        rv = rv_o(times, k, tc, orbit_case["p"], orbit_case["a"],
                   orbit_case["i"], orbit_case["e"], dt, pkt, pts, c)
        rv_ref = rv_newton_v(times, k, 0.0, orbit_case["p"],
                             orbit_case["e"], orbit_case["w"])
        assert_allclose(rv, rv_ref, rtol=1e-3, atol=2e-4)


class TestPhotometricSignals:
    def test_lambert_phase_curve_finite_and_bounded(self, orbit_case):
        times, tc, dt, pkt, pts, c = _setup(orbit_case)
        flux = lambert_phase_curve_o(times, ag=0.3, k=0.1,
                                      tpa=tc, p=orbit_case["p"], dt=dt,
                                      ep_table=pkt, ep_times=pts, coeffs=c)
        assert np.all(np.isfinite(flux))
        assert np.all(flux >= 0.0)
        # Flux = (k/r)^2 ag f(alpha) with f <= 1 and r >= a(1-e), so the
        # tightest constant ceiling uses the periastron distance.
        r_min = orbit_case["a"] * (1.0 - orbit_case["e"])
        amplitude = 0.1 ** 2 * 0.3 / r_min ** 2
        assert np.all(flux <= amplitude + 1e-12)

    def test_ev_signal_finite(self, orbit_case):
        times, tc, dt, pkt, pts, c = _setup(orbit_case)
        ev = ev_signal_o(alpha=1.0, mass_ratio=1e-3, inc=orbit_case["i"],
                          t=times, tpa=tc, p=orbit_case["p"], dt=dt,
                          ep_table=pkt, ep_times=pts, coeffs=c)
        assert np.all(np.isfinite(ev))


class TestOrbitClass:
    """End-to-end smoke test: exercise every Orbit method through the new pipeline."""

    @pytest.fixture
    def orbit(self, test_orbital_params):
        pars = test_orbital_params["eccentric"]
        o = Orbit(npt=NPT)
        o.set_pars(tc=0.0, **pars)
        o.set_data(np.linspace(0.0, pars["p"], NTIMES))
        return o, pars

    def test_xyz_matches_newton(self, orbit):
        o, pars = orbit
        x, y, z = o.xyz()
        xn, yn, zn = xyz_newton_v(o.times, 0.0, **pars)
        assert_allclose(x, xn, **TOL)
        assert_allclose(y, yn, **TOL)
        assert_allclose(z, zn, **TOL)

    def test_radial_velocity(self, orbit):
        o, pars = orbit
        rv = o.radial_velocity(k=0.05)
        rv_ref = rv_newton_v(o.times, 0.05, 0.0, pars["p"], pars["e"], pars["w"])
        assert_allclose(rv, rv_ref, rtol=1e-3, atol=2e-4)

    def test_cos_phase_in_range(self, orbit):
        o, _ = orbit
        ca = o.cos_phase()
        assert np.all(np.abs(ca) <= 1.0 + 1e-12)

    def test_lambert_phase_curve(self, orbit):
        o, _ = orbit
        flux = o.lambert_phase_curve(k=0.1, ag=0.3)
        assert np.all(np.isfinite(flux))
        assert np.all(flux >= 0.0)

    def test_ellipsoidal_variation(self, orbit):
        o, _ = orbit
        ev = o.ellipsoidal_variation(alpha=1.0, mass_ratio=1e-3)
        assert np.all(np.isfinite(ev))


class TestContracts:
    """Lock down implicit contracts that other code relies on."""

    def test_solve3d_orbit_periodic_boundary(self, test_orbital_params):
        """``solve3d_orbit`` copies the first expansion point's coefficients to the last
        slot — but that's only correct if ``ep_times[-1]`` is the periodic
        image of ``ep_times[0]``. Pin the contract so a future change to
        ``create_expansion_points`` can't silently break the wrap-around."""
        pars = test_orbital_params["eccentric"]
        ep_times, _, _, _ = create_expansion_points(NPT, pars["e"], "ea")
        # Phase-domain check: ep_times spans exactly one period.
        assert ep_times[0] == 0.0
        assert ep_times[-1] == 1.0
        coeffs = solve3d_orbit(ep_times, **pars, npt=NPT)
        assert_allclose(coeffs[-1], coeffs[0], rtol=1e-12)

    def test_true_anomaly_circular_no_nan(self, test_orbital_params):
        """``true_anomaly_o5v`` must produce finite values for orbits with
        e at or below the ``eccentricity_vector`` sentinel cutoff (1e-5)."""
        pars = dict(test_orbital_params["circular"])
        for e in [0.0, 1e-7, 1e-6]:
            pars["e"] = e
            times, tc, dt, pkt, pts, c = _setup(pars)
            ev = eccentricity_vector(pars["i"], pars["e"], pars["w"])
            f = true_anomaly_o(times, tc, pars["p"], ev[0], ev[1], ev[2],
                                pars["w"], dt, pkt, pts, c)
            assert np.all(np.isfinite(f)), f"NaN in true anomaly for e={e}"
            # For a circular orbit it should equal mean anomaly mod 2*pi.
            assert np.all((f >= 0.0) & (f < 2 * np.pi + 1e-9))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
