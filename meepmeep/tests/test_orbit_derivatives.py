"""Tests for the derivatives=True mode of the Orbit class.

Two-part strategy:

1. **Value parity**: every method's value output in derivative mode equals
   the value returned by the same method in non-derivative mode (modulo
   tiny ``fastmath`` rounding noise).

2. **Underlying-function parity**: the derivative output of each method
   exactly matches a direct call to the corresponding ``*_od`` routine
   in ``meepmeep.backends.numba.taylor.orbit3dd`` using the same
   ``coeffs/dcoeffs`` arrays — this confirms the wrapper does no
   unintended math.

Plus a handful of edge cases (``true_anomaly(exact=True)`` guard, ``phase``
and ``theta`` arccos chain rule, light-travel-time wiring).
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from meepmeep.orbit import Orbit
from meepmeep.backends.numba.utils import eccentricity_vector
from meepmeep.backends.numba.taylor.orbit3dd import (
    pos_od,
    vel_od,
    cos_alpha_od,
    star_planet_distance_od,
    rv_od,
    lambert_phase_curve_od,
    lambert_and_emission_od,
    ev_signal_od,
    true_anomaly_od,
    light_travel_time_od,
)


PARS = dict(tc=0.0, p=5.0, a=15.0, i=1.55, e=0.3, w=0.5)
NTIMES = 60
RTOL_VAL = 1e-8
RTOL_DERIV = 1e-12


def _make_orbit(derivatives):
    o = Orbit(npt=15, derivatives=derivatives)
    o.set_pars(**PARS)
    o.set_data(np.linspace(0.0, PARS["p"], NTIMES))
    return o


@pytest.fixture
def orbit_noderiv():
    return _make_orbit(False)


@pytest.fixture
def orbit_deriv():
    return _make_orbit(True)


@pytest.fixture
def orbit_pair():
    return _make_orbit(False), _make_orbit(True)


# ---------------------------------------------------------------------------
# Value parity: derivative-mode value output matches non-derivative-mode.
# ---------------------------------------------------------------------------

class TestValueParity:

    def test_xyz(self, orbit_pair):
        nd, d = orbit_pair
        x_b, y_b, z_b = nd.xyz()
        x, y, z, *_ = d.xyz()
        assert_allclose(x, x_b, rtol=RTOL_VAL, atol=1e-13)
        assert_allclose(y, y_b, rtol=RTOL_VAL, atol=1e-13)
        assert_allclose(z, z_b, rtol=RTOL_VAL, atol=1e-13)

    def test_vxyz(self, orbit_pair):
        nd, d = orbit_pair
        vx_b, vy_b, vz_b = nd.vxyz()
        vx, vy, vz, *_ = d.vxyz()
        assert_allclose(vx, vx_b, rtol=RTOL_VAL, atol=1e-13)
        assert_allclose(vy, vy_b, rtol=RTOL_VAL, atol=1e-13)
        assert_allclose(vz, vz_b, rtol=RTOL_VAL, atol=1e-13)

    def test_cos_phase(self, orbit_pair):
        nd, d = orbit_pair
        ca_b = nd.cos_phase()
        ca, _ = d.cos_phase()
        assert_allclose(ca, ca_b, rtol=RTOL_VAL, atol=1e-13)

    def test_phase(self, orbit_pair):
        nd, d = orbit_pair
        ph_b = nd.phase()
        ph, _ = d.phase()
        # Value path goes through arccos(ca) directly in non-deriv mode,
        # while deriv mode clamps ca slightly before arccos. The clamp
        # affects only a tiny epsilon near |ca|=1, so values agree at the
        # ~1e-7 level (arccos has infinite slope at the boundary).
        assert_allclose(ph, ph_b, atol=1e-6)

    def test_theta(self, orbit_pair):
        nd, d = orbit_pair
        th_b = nd.theta()
        th, _ = d.theta()
        assert_allclose(th, th_b, atol=1e-6)

    def test_star_planet_distance(self, orbit_pair):
        nd, d = orbit_pair
        r_b = nd.star_planet_distance()
        r, _ = d.star_planet_distance()
        assert_allclose(r, r_b, rtol=RTOL_VAL, atol=1e-13)

    def test_radial_velocity(self, orbit_pair):
        nd, d = orbit_pair
        rv_b = nd.radial_velocity(k=0.05)
        rv, _ = d.radial_velocity(k=0.05)
        assert_allclose(rv, rv_b, rtol=RTOL_VAL, atol=1e-13)

    def test_lambert_phase_curve(self, orbit_pair):
        nd, d = orbit_pair
        flux_b = nd.lambert_phase_curve(k=0.1, ag=0.3)
        flux, _ = d.lambert_phase_curve(k=0.1, ag=0.3)
        assert_allclose(flux, flux_b, rtol=RTOL_VAL, atol=1e-18)

    def test_lambert_and_emission(self, orbit_pair):
        nd, d = orbit_pair
        ref_b, emi_b = nd.lambert_and_emission(k=0.1, ag=0.3, fr_night=0.1, fr_day=0.4)
        ref, emi, _, _ = d.lambert_and_emission(k=0.1, ag=0.3, fr_night=0.1, fr_day=0.4)
        assert_allclose(ref, ref_b, rtol=RTOL_VAL, atol=1e-18)
        assert_allclose(emi, emi_b, rtol=RTOL_VAL, atol=1e-18)

    def test_ellipsoidal_variation(self, orbit_pair):
        nd, d = orbit_pair
        ev_b = nd.ellipsoidal_variation(alpha=1.0, mass_ratio=1e-3)
        ev, _ = d.ellipsoidal_variation(alpha=1.0, mass_ratio=1e-3)
        assert_allclose(ev, ev_b, rtol=RTOL_VAL, atol=1e-18)

    def test_true_anomaly(self, orbit_pair):
        nd, d = orbit_pair
        f_b = nd.true_anomaly()
        f, _ = d.true_anomaly()
        # cos/sin to be invariant to 2π wrap-around discontinuities.
        assert_allclose(np.cos(f), np.cos(f_b), atol=1e-10)
        assert_allclose(np.sin(f), np.sin(f_b), atol=1e-10)

    def test_light_travel_time(self, orbit_pair):
        nd, d = orbit_pair
        ltt_b = nd.light_travel_time(rstar=1.0)
        ltt, _ = d.light_travel_time(rstar=1.0)
        assert_allclose(ltt, ltt_b, rtol=RTOL_VAL, atol=1e-18)


# ---------------------------------------------------------------------------
# Underlying-function parity: derivative output of Orbit methods matches
# direct calls to the orbit3dd routines.
# ---------------------------------------------------------------------------

class TestUnderlyingParity:

    def _dispatch_args(self, o):
        return (o._tp, o._p, o._dt, o._tptable, o._points, o._coeffs, o._dcoeffs)

    def test_xyz(self, orbit_deriv):
        o = orbit_deriv
        x, y, z, dx, dy, dz = o.xyz()
        x_r, y_r, z_r, dx_r, dy_r, dz_r = pos_od(o.times, *self._dispatch_args(o))
        assert_allclose(dx, dx_r, rtol=RTOL_DERIV)
        assert_allclose(dy, dy_r, rtol=RTOL_DERIV)
        assert_allclose(dz, dz_r, rtol=RTOL_DERIV)

    def test_vxyz(self, orbit_deriv):
        o = orbit_deriv
        _, _, _, dvx, dvy, dvz = o.vxyz()
        _, _, _, dvx_r, dvy_r, dvz_r = vel_od(o.times, *self._dispatch_args(o))
        assert_allclose(dvx, dvx_r, rtol=RTOL_DERIV)
        assert_allclose(dvy, dvy_r, rtol=RTOL_DERIV)
        assert_allclose(dvz, dvz_r, rtol=RTOL_DERIV)

    def test_cos_phase(self, orbit_deriv):
        o = orbit_deriv
        _, dca = o.cos_phase()
        _, dca_r = cos_alpha_od(o.times, *self._dispatch_args(o))
        assert_allclose(dca, dca_r, rtol=RTOL_DERIV)

    def test_star_planet_distance(self, orbit_deriv):
        o = orbit_deriv
        _, dr = o.star_planet_distance()
        _, dr_r = star_planet_distance_od(o.times, *self._dispatch_args(o))
        assert_allclose(dr, dr_r, rtol=RTOL_DERIV)

    def test_radial_velocity(self, orbit_deriv):
        o = orbit_deriv
        _, drv = o.radial_velocity(k=0.05)
        _, drv_r = rv_od(o.times, 0.05, o._tp, o._p, o._a, o._i, o._e,
                          o._dt, o._tptable, o._points, o._coeffs, o._dcoeffs)
        assert_allclose(drv, drv_r, rtol=RTOL_DERIV)

    def test_lambert_phase_curve(self, orbit_deriv):
        o = orbit_deriv
        _, dflux = o.lambert_phase_curve(k=0.1, ag=0.3)
        _, dflux_r = lambert_phase_curve_od(
            o.times, 0.3, o._a, 0.1, o._tp, o._p,
            o._dt, o._tptable, o._points, o._coeffs, o._dcoeffs,
        )
        assert_allclose(dflux, dflux_r, rtol=RTOL_DERIV)

    def test_lambert_and_emission(self, orbit_deriv):
        o = orbit_deriv
        _, _, dref, demi = o.lambert_and_emission(k=0.1, ag=0.3,
                                                  fr_night=0.1, fr_day=0.4)
        _, _, dref_r, demi_r = lambert_and_emission_od(
            o.times, 0.3, 0.1, 0.4, 0.0, o._a, 0.1, o._tp, o._p,
            o._dt, o._tptable, o._points, o._coeffs, o._dcoeffs,
        )
        assert_allclose(dref, dref_r, rtol=RTOL_DERIV)
        assert_allclose(demi, demi_r, rtol=RTOL_DERIV)

    def test_ellipsoidal_variation(self, orbit_deriv):
        o = orbit_deriv
        _, dev = o.ellipsoidal_variation(alpha=1.0, mass_ratio=1e-3)
        _, dev_r = ev_signal_od(
            1.0, 1e-3, o._i, o.times, o._tp, o._p,
            o._dt, o._tptable, o._points, o._coeffs, o._dcoeffs,
        )
        assert_allclose(dev, dev_r, rtol=RTOL_DERIV)

    def test_true_anomaly(self, orbit_deriv):
        o = orbit_deriv
        ev = eccentricity_vector(o._i, o._e, o._w)
        _, df = o.true_anomaly()
        _, df_r = true_anomaly_od(
            o.times, o._tp, o._p, ev[0], ev[1], ev[2], o._w,
            o._dt, o._tptable, o._points, o._coeffs, o._dcoeffs,
        )
        assert_allclose(df, df_r, rtol=RTOL_DERIV)

    def test_light_travel_time(self, orbit_deriv):
        o = orbit_deriv
        rstar = 1.0
        _, dltt = o.light_travel_time(rstar=rstar)
        _, dltt_r = light_travel_time_od(
            o.times, o._tp, o._p, o._e, o._w, rstar,
            o._dt, o._tptable, o._points, o._coeffs, o._dcoeffs,
        )
        assert_allclose(dltt, dltt_r, rtol=RTOL_DERIV)


# ---------------------------------------------------------------------------
# Shape and chain-rule checks
# ---------------------------------------------------------------------------

class TestShapesAndChainRule:

    def test_xyz_shapes(self, orbit_deriv):
        x, y, z, dx, dy, dz = orbit_deriv.xyz()
        assert x.shape == (NTIMES,)
        assert dx.shape == (NTIMES, 7)
        assert dy.shape == (NTIMES, 7)
        assert dz.shape == (NTIMES, 7)

    def test_radial_velocity_shape(self, orbit_deriv):
        rv, drv = orbit_deriv.radial_velocity(k=0.05)
        assert rv.shape == (NTIMES,)
        assert drv.shape == (NTIMES, 8)  # 6 orbital + lan + k

    def test_lambert_phase_curve_shape(self, orbit_deriv):
        flux, dflux = orbit_deriv.lambert_phase_curve(k=0.1, ag=0.3)
        assert flux.shape == (NTIMES,)
        assert dflux.shape == (NTIMES, 9)  # 6 orbital + lan + ag + k

    def test_lambert_and_emission_shape(self, orbit_deriv):
        ref, emi, dref, demi = orbit_deriv.lambert_and_emission(
            k=0.1, ag=0.3, fr_night=0.1, fr_day=0.4)
        assert ref.shape == (NTIMES,)
        assert emi.shape == (NTIMES,)
        assert dref.shape == (NTIMES, 12)  # 6 orbital + lan + ag,fr_night,fr_day,emi_offset,k
        assert demi.shape == (NTIMES, 12)

    def test_ellipsoidal_variation_shape(self, orbit_deriv):
        ev, dev = orbit_deriv.ellipsoidal_variation(alpha=1.0, mass_ratio=1e-3)
        assert ev.shape == (NTIMES,)
        assert dev.shape == (NTIMES, 10)  # 6 orbital + lan + alpha,mass_ratio,inc

    def test_phase_chain_rule_against_cos_phase(self, orbit_deriv):
        """``dphase/dθ = -dcα/dθ / sqrt(1 - cα²)`` (clamped at boundary)."""
        ca, dca = orbit_deriv.cos_phase()
        ph, dph = orbit_deriv.phase()
        ca_c = np.clip(ca, -1.0 + 1e-15, 1.0 - 1e-15)
        expected = -dca / np.sqrt(1.0 - ca_c ** 2)[:, None]
        # The clamp guarantees the derivative is finite, but at the very
        # extrema the analytic gradient blows up. Tolerance is generous.
        finite_mask = np.abs(ca) < 0.999
        assert_allclose(dph[finite_mask], expected[finite_mask],
                        rtol=1e-10, atol=1e-12)

    def test_theta_chain_rule_against_cos_phase(self, orbit_deriv):
        ca, dca = orbit_deriv.cos_phase()
        th, dth = orbit_deriv.theta()
        ca_c = np.clip(ca, -1.0 + 1e-15, 1.0 - 1e-15)
        expected = dca / np.sqrt(1.0 - ca_c ** 2)[:, None]
        finite_mask = np.abs(ca) < 0.999
        assert_allclose(dth[finite_mask], expected[finite_mask],
                        rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_true_anomaly_exact_raises_in_deriv_mode(self, orbit_deriv):
        with pytest.raises(NotImplementedError, match="exact=True"):
            orbit_deriv.true_anomaly(exact=True)

    def test_true_anomaly_exact_works_in_noderiv_mode(self, orbit_noderiv):
        # Should not raise; should return an (N,) array.
        f = orbit_noderiv.true_anomaly(exact=True)
        assert f.shape == (NTIMES,)
        assert np.all(np.isfinite(f))

    def test_noderiv_mode_unchanged_return_shape(self, orbit_noderiv):
        """Regression guard: in derivatives=False mode, every method's
        return shape must match the pre-existing API exactly."""
        o = orbit_noderiv
        x, y, z = o.xyz()
        assert x.shape == (NTIMES,) and y.shape == (NTIMES,) and z.shape == (NTIMES,)
        vx, vy, vz = o.vxyz()
        assert vx.shape == (NTIMES,)
        assert o.cos_phase().shape == (NTIMES,)
        assert o.phase().shape == (NTIMES,)
        assert o.theta().shape == (NTIMES,)
        assert o.star_planet_distance().shape == (NTIMES,)
        assert o.radial_velocity(k=0.05).shape == (NTIMES,)
        assert o.lambert_phase_curve(k=0.1, ag=0.3).shape == (NTIMES,)
        ref, emi = o.lambert_and_emission(k=0.1, ag=0.3, fr_night=0.1, fr_day=0.4)
        assert ref.shape == (NTIMES,) and emi.shape == (NTIMES,)
        assert o.ellipsoidal_variation(alpha=1.0, mass_ratio=1e-3).shape == (NTIMES,)
        assert o.true_anomaly().shape == (NTIMES,)
        assert o.light_travel_time(rstar=1.0).shape == (NTIMES,)

    def test_set_pars_populates_dcoeffs_only_in_deriv_mode(self):
        nd = Orbit(npt=15)
        nd.set_pars(**PARS)
        assert nd._dcoeffs is None
        d = Orbit(npt=15, derivatives=True)
        d.set_pars(**PARS)
        assert d._dcoeffs is not None
        assert d._dcoeffs.shape == (15, 7, 3, 5)


class TestLongitudeOfAscendingNode:
    """Threading of the longitude of the ascending node (lan) through Orbit.

    `lan` is a constant rotation of the sky-plane (x, y) about the line of
    sight; it leaves the line-of-sight z coordinate (and every quantity that
    depends only on it) invariant, and it enters the gradient as column 6.
    """

    def _orbit(self, lan, derivatives=True):
        o = Orbit(npt=15, derivatives=derivatives)
        o.set_pars(**{**PARS, "lan": lan})
        o.set_data(np.linspace(0.0, PARS["p"], NTIMES))
        return o

    def test_lan_default_zero_value_parity(self):
        """Omitting lan reproduces lan=0.0 exactly (value path)."""
        x0, y0, z0 = _make_orbit(False).xyz()
        ol = self._orbit(0.0, derivatives=False)
        x, y, z = ol.xyz()
        assert_allclose(x, x0, rtol=0, atol=0)
        assert_allclose(y, y0, rtol=0, atol=0)
        assert_allclose(z, z0, rtol=0, atol=0)

    def test_quarter_turn_rotates_xy_leaves_z(self):
        """lan=pi/2 maps (x, y) -> (-y, x) and leaves z unchanged."""
        o0 = self._orbit(0.0)
        ol = self._orbit(np.pi / 2)
        x0, y0, z0 = o0.xyz()[:3]
        x, y, z = ol.xyz()[:3]
        assert_allclose(x, -y0, rtol=1e-9, atol=1e-9)
        assert_allclose(y, x0, rtol=1e-9, atol=1e-9)
        assert_allclose(z, z0, rtol=1e-12, atol=1e-12)

    def test_xyz_gradient_has_lan_column(self):
        """xyz gradients are (N, 7); column 6 is the lan derivative."""
        o = self._orbit(0.7)
        _, _, _, dx, dy, dz = o.xyz()
        assert dx.shape == (NTIMES, 7)
        # At the evaluated lan, d(x,y)/dlan is non-trivial but dz/dlan == 0.
        assert np.any(np.abs(dx[:, 6]) > 1e-6)
        assert np.any(np.abs(dy[:, 6]) > 1e-6)
        assert_allclose(dz[:, 6], 0.0, atol=1e-12)

    def test_lan_derivative_vs_finite_difference(self):
        """xyz column-6 gradient matches a central difference in lan."""
        lan, h = 0.7, 1e-6
        o = self._orbit(lan)
        x, y, z, dx, dy, dz = o.xyz()
        xp, yp, zp = self._orbit(lan + h).xyz()[:3]
        xm, ym, zm = self._orbit(lan - h).xyz()[:3]
        assert_allclose(dx[:, 6], (xp - xm) / (2 * h), rtol=1e-5, atol=1e-7)
        assert_allclose(dy[:, 6], (yp - ym) / (2 * h), rtol=1e-5, atol=1e-7)
        assert_allclose(dz[:, 6], (zp - zm) / (2 * h), atol=1e-9)

    def test_rotation_invariant_quantities_have_zero_lan_derivative(self):
        """Separation, z-position, and radial velocity are invariant under lan,
        so their lan-column gradient is zero and their values are lan-independent."""
        o0 = self._orbit(0.0)
        ol = self._orbit(0.9)
        r_l, dr = ol.star_planet_distance()
        r_0, _ = o0.star_planet_distance()
        rv_l, drv = ol.radial_velocity(k=0.05)
        rv_0, _ = o0.radial_velocity(k=0.05)
        # Values unchanged under the rotation.
        assert_allclose(r_l, r_0, rtol=1e-10, atol=1e-12)
        assert_allclose(rv_l, rv_0, rtol=1e-10, atol=1e-12)
        # lan-column gradients ~ 0.
        assert_allclose(dr[:, 6], 0.0, atol=1e-10)
        assert_allclose(drv[:, 6], 0.0, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
