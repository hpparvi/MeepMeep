"""Test suite for meepmeep.backends.numba.taylor.solve3d module.

Tests validate the solve3d function which computes (3, 5) Taylor coefficient
matrices for 3D Keplerian position using analytic derivatives.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

from meepmeep.backends.numba.taylor.solve3d import solve3d
from meepmeep.backends.numba.taylor.solve3dd import solve3d_d
from meepmeep.backends.numba.taylor.position3d import pos_c
from meepmeep.backends.numba.taylor.position3dd import pos_d, sep_d
from meepmeep.backends.numba.taylor.velocity3d import vel_c
from meepmeep.backends.numba.newton.newton import xyz_newton_v


@pytest.fixture
def circular_orbit():
    return {"p": 3.0, "a": 10.0, "i": 1.5, "e": 0.0, "w": 0.0}


@pytest.fixture
def eccentric_orbit():
    return {"p": 5.0, "a": 15.0, "i": 1.55, "e": 0.3, "w": 0.5}


@pytest.fixture
def high_e_orbit():
    return {"p": 7.0, "a": 20.0, "i": 1.4, "e": 0.7, "w": 1.2}


class TestSolve3dOutput:
    """Test basic output properties."""

    def test_output_shape(self, circular_orbit):
        cf = solve3d(0.0, **circular_orbit)
        assert cf.shape == (3, 5)

    @pytest.mark.parametrize("orbit_fixture", ["circular_orbit", "eccentric_orbit", "high_e_orbit"])
    def test_coefficients_finite(self, orbit_fixture, request):
        orbit = request.getfixturevalue(orbit_fixture)
        cf = solve3d(0.0, **orbit)
        assert np.all(np.isfinite(cf))

    def test_different_phases(self, eccentric_orbit):
        """Coefficients should be finite at phases spanning the full orbit."""
        p = eccentric_orbit["p"]
        for phase in np.linspace(0, p, 20, endpoint=False):
            cf = solve3d(phase, **eccentric_orbit)
            assert np.all(np.isfinite(cf)), f"Non-finite coefficients at phase={phase}"


class TestSolve3dPhysics:
    """Test physical correctness of the coefficients."""

    def test_circular_orbit_at_transit(self, circular_orbit):
        """At transit center for circular orbit, x ≈ 0 and y < 0."""
        cf = solve3d(0.0, **circular_orbit)
        assert abs(cf[0, 0]) < 0.1, "X position should be near zero at transit"
        assert cf[1, 0] < 0, "Y position should be negative at transit"

    def test_velocity_direction_at_transit(self, circular_orbit):
        """X-velocity should be positive (planet crossing toward observer) at transit."""
        cf = solve3d(0.0, **circular_orbit)
        assert cf[0, 1] > 0, "X velocity should be positive during transit"
        assert abs(cf[1, 1]) < 1e-6, "Y velocity should be near zero at transit for circular orbit"

    def test_edge_on_y_zero(self):
        """For edge-on orbit (i = π/2), y-position at transit should be zero."""
        params = {"p": 3.0, "a": 10.0, "i": np.pi / 2, "e": 0.0, "w": 0.0}
        cf = solve3d(0.0, **params)
        assert_allclose(cf[1, 0], 0.0, atol=1e-10,
                        err_msg="Y position should be zero for edge-on orbit at transit")

    def test_coefficient_prescaling(self, circular_orbit):
        """Verify pre-scaling: the Taylor polynomial evaluated via Horner's method
        should match the velocity obtained by analytically differentiating it."""
        cf = solve3d(0.0, **circular_orbit)

        # The velocity from the derivative of the Horner polynomial at t=0
        # should equal cf[:, 1] (the pre-scaled 1st derivative coefficient).
        # v3dc evaluates this derivative analytically.
        vx, vy, vz = vel_c(0.0, cf)
        assert_allclose(vx, cf[0, 1], rtol=1e-12)
        assert_allclose(vy, cf[1, 1], rtol=1e-12)
        assert_allclose(vz, cf[2, 1], rtol=1e-12)

    def test_position_radius_consistent(self, eccentric_orbit):
        """Position magnitude should be consistent with the orbital semi-major axis."""
        cf = solve3d(0.0, **eccentric_orbit)
        r = np.sqrt(cf[0, 0]**2 + cf[1, 0]**2 + cf[2, 0]**2)
        a = eccentric_orbit["a"]
        e = eccentric_orbit["e"]
        # r should be between a(1-e) and a(1+e)
        assert a * (1 - e) <= r <= a * (1 + e), \
            f"Position radius {r} outside orbital bounds [{a*(1-e)}, {a*(1+e)}]"


class TestSolve3dAccuracy:
    """Test accuracy against Newton-Raphson ground truth."""

    def test_position_vs_newton(self, eccentric_orbit):
        """Taylor series evaluated near the expansion point should match Newton."""
        cf = solve3d(0.0, **eccentric_orbit)
        times = np.linspace(-0.02, 0.02, 10)

        for t in times:
            x_ts, y_ts, z_ts = pos_c(t, cf)
            x_nr, y_nr, z_nr = xyz_newton_v(
                np.array([t]), 0.0, eccentric_orbit["p"],
                eccentric_orbit["a"], eccentric_orbit["i"],
                eccentric_orbit["e"], eccentric_orbit["w"]
            )
            assert_allclose(x_ts, x_nr[0], rtol=1e-6, atol=1e-8)
            assert_allclose(y_ts, y_nr[0], rtol=1e-6, atol=1e-8)
            assert_allclose(z_ts, z_nr[0], rtol=1e-6, atol=1e-8)

    @pytest.mark.parametrize("e", [0.0, 0.1, 0.3, 0.5, 0.7])
    def test_accuracy_various_eccentricities(self, e):
        """Accuracy near expansion point across eccentricities."""
        params = {"p": 5.0, "a": 15.0, "i": 1.5, "e": e, "w": 0.5}
        cf = solve3d(0.0, **params)

        times = np.linspace(-0.01, 0.01, 10)
        rtol = 1e-5 * (1 + e)

        for t in times:
            x_ts, y_ts, z_ts = pos_c(t, cf)
            x_nr, y_nr, z_nr = xyz_newton_v(
                np.array([t]), 0.0, params["p"],
                params["a"], params["i"], params["e"], params["w"]
            )
            assert_allclose(x_ts, x_nr[0], rtol=rtol, atol=1e-8)
            assert_allclose(y_ts, y_nr[0], rtol=rtol, atol=1e-8)
            assert_allclose(z_ts, z_nr[0], rtol=rtol, atol=1e-8)

    def test_velocity_vs_newton_finite_diff(self, eccentric_orbit):
        """Taylor velocity should match finite differences of Newton positions."""
        cf = solve3d(0.0, **eccentric_orbit)
        p = eccentric_orbit["p"]
        a = eccentric_orbit["a"]
        i = eccentric_orbit["i"]
        e = eccentric_orbit["e"]
        w = eccentric_orbit["w"]

        h = 1e-5
        t = 0.005
        x1, y1, z1 = xyz_newton_v(np.array([t - h]), 0.0, p, a, i, e, w)
        x2, y2, z2 = xyz_newton_v(np.array([t + h]), 0.0, p, a, i, e, w)

        vx_fd = (x2[0] - x1[0]) / (2 * h)
        vy_fd = (y2[0] - y1[0]) / (2 * h)
        vz_fd = (z2[0] - z1[0]) / (2 * h)

        vx_ts, vy_ts, vz_ts = vel_c(t, cf)

        assert_allclose(vx_ts, vx_fd, rtol=1e-4, atol=1e-6)
        assert_allclose(vy_ts, vy_fd, rtol=1e-4, atol=1e-6)
        assert_allclose(vz_ts, vz_fd, rtol=1e-4, atol=1e-6)

    def test_degradation_far_from_expansion(self, eccentric_orbit):
        """Error should increase with distance from expansion point."""
        cf = solve3d(0.0, **eccentric_orbit)
        p = eccentric_orbit["p"]
        a = eccentric_orbit["a"]
        i = eccentric_orbit["i"]
        e = eccentric_orbit["e"]
        w = eccentric_orbit["w"]

        def position_error(t):
            x_ts, y_ts, z_ts = pos_c(t, cf)
            x_nr, y_nr, z_nr = xyz_newton_v(np.array([t]), 0.0, p, a, i, e, w)
            return np.sqrt((x_ts - x_nr[0])**2 + (y_ts - y_nr[0])**2 + (z_ts - z_nr[0])**2)

        err_near = position_error(0.01)
        err_far = position_error(0.3)
        assert err_far > err_near, "Error should increase away from expansion point"


class TestSolve3dEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_eccentricity(self):
        params = {"p": 3.0, "a": 10.0, "i": 1.5, "e": 0.0, "w": 0.0}
        cf = solve3d(0.0, **params)
        assert cf.shape == (3, 5)
        assert np.all(np.isfinite(cf))

    def test_high_eccentricity(self):
        params = {"p": 5.0, "a": 15.0, "i": 1.5, "e": 0.9, "w": 0.5}
        cf = solve3d(0.0, **params)
        assert cf.shape == (3, 5)
        assert np.all(np.isfinite(cf))

    def test_short_period(self):
        params = {"p": 0.5, "a": 5.0, "i": 1.5, "e": 0.0, "w": 0.0}
        cf = solve3d(0.0, **params)
        assert np.all(np.isfinite(cf))

    def test_long_period(self):
        params = {"p": 100.0, "a": 50.0, "i": 1.5, "e": 0.2, "w": 0.5}
        cf = solve3d(0.0, **params)
        assert np.all(np.isfinite(cf))


class TestSolve3dLan:
    """Test the longitude of the ascending node (lan) rotation in solve3d.

    In 3D the rotation is about the line of sight, so x and y rotate while
    the line-of-sight z coordinate is unchanged.
    """

    def test_lan_zero_identity(self, eccentric_orbit):
        """lan=0.0 must reproduce the call without lan exactly."""
        cf = solve3d(0.0, **eccentric_orbit)
        cf_lan0 = solve3d(0.0, **eccentric_orbit, lan=0.0)
        assert_allclose(cf_lan0, cf, rtol=0, atol=0)

    def test_lan_default_is_zero(self, eccentric_orbit):
        """The default value of lan is 0.0 (no rotation)."""
        cf = solve3d(0.0, **eccentric_orbit)
        cf_default = solve3d(0.0, eccentric_orbit["p"], eccentric_orbit["a"],
                             eccentric_orbit["i"], eccentric_orbit["e"], eccentric_orbit["w"])
        assert_allclose(cf_default, cf, rtol=0, atol=0)

    def test_quarter_turn(self, eccentric_orbit):
        """lan=pi/2 maps each column (x, y) -> (-y, x) and leaves z unchanged."""
        cf = solve3d(0.0, **eccentric_orbit)
        rot = solve3d(0.0, **eccentric_orbit, lan=np.pi / 2)
        assert_allclose(rot[0], -cf[1], rtol=1e-12, atol=1e-12)
        assert_allclose(rot[1], cf[0], rtol=1e-12, atol=1e-12)
        assert_allclose(rot[2], cf[2], rtol=1e-12, atol=1e-12)

    def test_z_invariant(self, eccentric_orbit):
        """The line-of-sight z row is invariant under any lan."""
        cf = solve3d(0.0, **eccentric_orbit)
        for lan in (0.3, 1.1, 2.7):
            rot = solve3d(0.0, **eccentric_orbit, lan=lan)
            assert_allclose(rot[2], cf[2], rtol=1e-12, atol=1e-12)

    def test_xy_norm_invariant(self, eccentric_orbit):
        """A rotation about the line of sight preserves the (x, y) norm."""
        cf = solve3d(0.0, **eccentric_orbit)
        rot = solve3d(0.0, **eccentric_orbit, lan=1.1)
        assert_allclose(np.hypot(rot[0], rot[1]), np.hypot(cf[0], cf[1]), rtol=1e-12, atol=1e-12)


class TestSolve3dDLan:
    """Test the lan parameter and its derivative row in solve3d_d."""

    def test_gradient_shape(self, eccentric_orbit):
        """The derivative tensor gains a 7th row for lan: (7, 3, 5)."""
        cf, dcf = solve3d_d(0.0, **eccentric_orbit)
        assert cf.shape == (3, 5)
        assert dcf.shape == (7, 3, 5)

    def test_cf_matches_solve3d(self, eccentric_orbit):
        """The cf returned by solve3d_d matches solve3d for non-zero lan."""
        lan = 0.6
        cf_d, _ = solve3d_d(0.0, **eccentric_orbit, lan=lan)
        cf = solve3d(0.0, **eccentric_orbit, lan=lan)
        assert_allclose(cf_d, cf, rtol=1e-12, atol=1e-12)

    def test_lan_derivative_z_row_zero(self, eccentric_orbit):
        """The z row of the lan-derivative is zero (z is lan-independent)."""
        _, dcf = solve3d_d(0.0, **eccentric_orbit, lan=0.4)
        assert_allclose(dcf[6, 2], 0.0, atol=1e-14)

    def test_lan_derivative_at_zero(self, eccentric_orbit):
        """At lan=0, the lan-derivative row equals (-y, x, 0) per column."""
        cf, dcf = solve3d_d(0.0, **eccentric_orbit, lan=0.0)
        assert_allclose(dcf[6, 0], -cf[1], rtol=1e-12, atol=1e-12)
        assert_allclose(dcf[6, 1], cf[0], rtol=1e-12, atol=1e-12)
        assert_allclose(dcf[6, 2], 0.0, atol=1e-14)

    @pytest.mark.parametrize("lan", [0.0, 0.4, 1.3, 2.5])
    def test_lan_derivative_vs_finite_diff(self, eccentric_orbit, lan):
        """The analytic lan-derivative row matches a central finite difference."""
        h = 1e-6
        cf_p = solve3d(0.0, **eccentric_orbit, lan=lan + h)
        cf_m = solve3d(0.0, **eccentric_orbit, lan=lan - h)
        fd = (cf_p - cf_m) / (2 * h)
        _, dcf = solve3d_d(0.0, **eccentric_orbit, lan=lan)
        assert_allclose(dcf[6], fd, rtol=1e-6, atol=1e-8)

    def test_existing_rows_are_rotated(self, eccentric_orbit):
        """For non-zero lan, derivative rows 0..5 equal R(lan) applied (in x, y) to
        the lan=0 derivative rows; the z row is unchanged."""
        lan = 0.9
        _, dcf0 = solve3d_d(0.0, **eccentric_orbit, lan=0.0)
        _, dcf = solve3d_d(0.0, **eccentric_orbit, lan=lan)
        cO, sO = np.cos(lan), np.sin(lan)
        for k in range(6):
            assert_allclose(dcf[k, 0], cO * dcf0[k, 0] - sO * dcf0[k, 1], rtol=1e-12, atol=1e-12)
            assert_allclose(dcf[k, 1], sO * dcf0[k, 0] + cO * dcf0[k, 1], rtol=1e-12, atol=1e-12)
            assert_allclose(dcf[k, 2], dcf0[k, 2], rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("kidx", [0, 1, 2, 3, 4, 5])
    def test_kepler_derivatives_vs_finite_diff(self, eccentric_orbit, kidx):
        """With a non-zero lan, the first six derivative rows still match finite
        differences w.r.t. their parameters (tc, p, a, i, e, w)."""
        lan = 0.8
        params = [0.0, eccentric_orbit["p"], eccentric_orbit["a"],
                  eccentric_orbit["i"], eccentric_orbit["e"], eccentric_orbit["w"]]
        h = 1e-6
        pp = list(params); pp[kidx] += h
        pm = list(params); pm[kidx] -= h
        cf_p = solve3d(pp[0], pp[1], pp[2], pp[3], pp[4], pp[5], lan=lan)
        cf_m = solve3d(pm[0], pm[1], pm[2], pm[3], pm[4], pm[5], lan=lan)
        fd = (cf_p - cf_m) / (2 * h)
        _, dcf = solve3d_d(params[0], params[1], params[2], params[3],
                           params[4], params[5], lan=lan)
        # Slot 0 is the derivative w.r.t. the transit-centre time tc. The orbit
        # depends on (t_obs - tc), so d/dtc = -d/dtk; the finite difference above
        # perturbs the solver's knot-time argument tk (= d/dtk), so the
        # slot-0 reference is negated.
        expected = -fd if kidx == 0 else fd
        assert_allclose(dcf[kidx], expected, rtol=1e-5, atol=1e-7)


class TestT0Derivative:
    """Slot 0 of the derivative tensor is the partial w.r.t. the transit-center
    time tc."""

    @pytest.mark.parametrize("orbit_fixture", ["circular_orbit", "eccentric_orbit", "high_e_orbit"])
    def test_slot0_equals_negative_next_coefficient(self, orbit_fixture, request):
        """Exact, truncation-free check of the tc convention. Because the Taylor
        coefficient c[:, n] = P^(n)(0)/n! and d/dtc = -d/dtk, the (n+1)-th
        coefficient is the tk-derivative of the n-th, so the tc-derivative obeys
        dc[0, :, n] = -(n+1) * c[:, n+1] for n = 0..3 (to machine precision).
        With the old +TWO_PI/p sign this identity would be off by a sign."""
        orbit = request.getfixturevalue(orbit_fixture)
        cf, dcf = solve3d_d(0.0, **orbit)
        for n in range(4):
            assert_allclose(dcf[0, :, n], -(n + 1) * cf[:, n + 1], rtol=1e-7, atol=1e-9)

    @pytest.mark.parametrize("time", [0.005, 0.01, 0.02, -0.02])
    def test_sep_d_and_pos_d_dtc_finite_difference(self, eccentric_orbit, time):
        """End-to-end: slot 0 of sep_d/pos_d matches a central difference of the
        value w.r.t. the tc argument. Times are kept near the knot so the Taylor
        truncation (the value model's own accuracy limit) stays below tolerance."""
        c, dc = solve3d_d(0.0, **eccentric_orbit)
        p = eccentric_orbit["p"]
        tc, h = 0.0, 1e-6

        _, dd = sep_d(time, tc, p, c, dc)
        dp, _ = sep_d(time, tc + h, p, c, dc)
        dm, _ = sep_d(time, tc - h, p, c, dc)
        assert_allclose(dd[0], (dp - dm) / (2 * h), rtol=1e-5, atol=1e-6)

        _, _, _, dpx, dpy, dpz = pos_d(time, tc, p, c, dc)
        pxp, pyp, pzp, _, _, _ = pos_d(time, tc + h, p, c, dc)
        pxm, pym, pzm, _, _, _ = pos_d(time, tc - h, p, c, dc)
        assert_allclose(dpx[0], (pxp - pxm) / (2 * h), rtol=1e-5, atol=1e-6)
        assert_allclose(dpy[0], (pyp - pym) / (2 * h), rtol=1e-5, atol=1e-6)
        assert_allclose(dpz[0], (pzp - pzm) / (2 * h), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
