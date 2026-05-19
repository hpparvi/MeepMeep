"""Test suite for meepmeep.backends.numba.taylor.solve3d module.

Tests validate the solve3d function which computes (3, 5) Taylor coefficient
matrices for 3D Keplerian position using analytic derivatives.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

from meepmeep.backends.numba.taylor.solve3d import solve3d
from meepmeep.backends.numba.taylor.position3d import pos_c
from meepmeep.backends.numba.taylor.velocity3d import v3dc
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
        vx, vy, vz = v3dc(0.0, cf)
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

        vx_ts, vy_ts, vz_ts = v3dc(t, cf)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
