"""Test suite for meepmeep.backends.numba.taylor.position2d module.

This test suite validates the Taylor series approximation functions against
the exact Newton-Raphson methods as ground truth.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

from meepmeep.backends.numba.taylor.position2d import pos, pos_c, pos_and_sep, sep_c
from meepmeep.backends.numba.taylor.util2d import find_contact_point, bounding_box
from meepmeep.backends.numba.taylor.solve2d import solve2d
from meepmeep.backends.numba.newton.newton import xy_newton_v, z_newton_v


# Test fixtures: orbital parameter sets
@pytest.fixture
def circular_orbit():
    """Circular orbit parameters."""
    return {"p": 3.0,  # period [days]
        "a": 10.0,  # semi-major axis [R_star]
        "i": 1.5,  # inclination [rad] (~86 degrees)
        "e": 0.0,  # eccentricity (circular)
        "w": 0.0,  # argument of periastron [rad]
    }


@pytest.fixture
def eccentric_orbit():
    """Moderately eccentric orbit parameters."""
    return {"p": 5.0, "a": 15.0, "i": 1.55,  # ~89 degrees
        "e": 0.3,  # moderate eccentricity
        "w": 0.5, }


@pytest.fixture
def high_eccentricity_orbit():
    """Highly eccentric orbit parameters."""
    return {"p": 7.0, "a": 20.0, "i": 1.4, "e": 0.7,  # high eccentricity
        "w": 1.2, }


@pytest.fixture
def edge_on_orbit():
    """Edge-on orbit (i = 90 degrees)."""
    return {"p": 2.5, "a": 8.0, "i": np.pi / 2,  # 90 degrees
        "e": 0.1, "w": 0.3, }


class TestSolveXYP5S:
    """Test the solve_xy_p5 function (Taylor expansion coefficients)."""

    def test_output_shape(self, circular_orbit):
        """Test that output has correct shape (2x5)."""
        coeffs = solve2d(0.0, **circular_orbit)
        assert coeffs.shape == (2, 5), f"Expected shape (2, 5), got {coeffs.shape}"

    def test_circular_orbit_at_transit(self, circular_orbit):
        """Test Taylor expansion at transit center for circular orbit."""
        phase = 0.0  # transit center
        coeffs = solve2d(phase, **circular_orbit)

        # At transit center for circular orbit, x should be small, y should be close to -a
        # Position coefficients (column 0)
        assert abs(coeffs[0, 0]) < 0.1, "X position should be near zero at transit"
        assert coeffs[1, 0] < 0, "Y position should be negative"

        # Velocity should be positive in x-direction during transit
        assert coeffs[0, 1] > 0, "X velocity should be positive during transit"

    def test_coefficients_finite(self, eccentric_orbit):
        """Test that all coefficients are finite."""
        coeffs = solve2d(0.0, **eccentric_orbit)
        assert np.all(np.isfinite(coeffs)), "All coefficients should be finite"

    def test_different_phases(self, circular_orbit):
        """Test expansion at different orbital phases."""
        phases = np.linspace(0, circular_orbit["p"], 10)
        for phase in phases:
            coeffs = solve2d(phase, **circular_orbit)
            assert coeffs.shape == (2, 5)
            assert np.all(np.isfinite(coeffs))

    def test_symmetry_circular_orbit(self, circular_orbit):
        """Test symmetry properties for circular orbit."""
        # For circular orbit centered at transit, y-velocity should be zero
        coeffs = solve2d(0.0, **circular_orbit)

        # At transit center for circular orbit, y-velocity should be very small
        assert abs(coeffs[1, 1]) < 1e-6, "Y velocity should be near zero at transit center"


class TestPosition2dFunctions:
    """Test the position2d family of functions (Taylor series evaluation)."""

    def test_p2d_scalar(self, circular_orbit):
        """Test scalar version of p2d."""
        # Get coefficients at transit center
        coeffs = solve2d(0.0, **circular_orbit)

        # Evaluate at a small time offset
        t = 0.01
        x, y = pos(t, 0.0, circular_orbit["p"], coeffs)

        assert isinstance(x, (float, np.floating))
        assert isinstance(y, (float, np.floating))
        assert np.isfinite(x)
        assert np.isfinite(y)

    def test_p2d_accuracy_vs_newton(self, circular_orbit):
        """Test accuracy of Taylor approximation vs Newton-Raphson."""
        # Get coefficients at transit center
        coeffs = solve2d(0.0, **circular_orbit)

        # Test at small time offsets where Taylor series should be very accurate
        times = np.linspace(-0.02, 0.02, 10)

        for t in times:
            # Taylor approximation
            x_taylor, y_taylor = pos(t, 0.0, circular_orbit["p"], coeffs)

            # Newton-Raphson ground truth
            x_newton, y_newton = xy_newton_v(np.array([t]), 0.0, circular_orbit["p"], circular_orbit["a"],
                circular_orbit["i"], circular_orbit["e"], circular_orbit["w"])

            # Should be very accurate near expansion point
            assert_allclose(x_taylor, x_newton[0], rtol=1e-6, atol=1e-8)
            assert_allclose(y_taylor, y_newton[0], rtol=1e-6, atol=1e-8)

    def test_p2dc(self, circular_orbit):
        """Test p2dc (centered version)."""
        coeffs = solve2d(0.0, **circular_orbit)
        t = 0.01

        x, y = pos_c(t, coeffs)

        assert np.isfinite(x)
        assert np.isfinite(y)

    def test_pd2d(self, circular_orbit):
        """Test pd2d returns position and distance."""
        coeffs = solve2d(0.0, **circular_orbit)
        t = 0.01

        x, y, d = pos_and_sep(t, coeffs)

        # Distance should match sqrt(x^2 + y^2)
        expected_d = np.sqrt(x ** 2 + y ** 2)
        assert_allclose(d, expected_d, rtol=1e-12)


class TestProjectedDistance:
    """Test projected distance calculation functions."""

    def test_d2dc(self, circular_orbit):
        """Test d2dc (centered version) for scalar input."""
        coeffs = solve2d(0.0, **circular_orbit)
        t = 0.01

        d = sep_c(t, coeffs)

        assert isinstance(d, (float, np.floating))
        assert d > 0
        assert np.isfinite(d)

    def test_d2dc_matches_xy(self, eccentric_orbit):
        """Test that d2dc matches distance from p2dc."""
        coeffs = solve2d(0.0, **eccentric_orbit)
        t = 0.02

        x, y = pos_c(t, coeffs)
        d_from_xy = np.sqrt(x ** 2 + y ** 2)

        d_direct = sep_c(t, coeffs)

        assert_allclose(d_direct, d_from_xy, rtol=1e-12)

    def test_d2dc_accuracy_vs_newton(self, circular_orbit):
        """Test projected distance accuracy vs Newton method."""
        coeffs = solve2d(0.0, **circular_orbit)
        times = np.array([0.0, 0.01, 0.02])

        # Taylor approximation using centered version
        d_taylor = np.array([sep_c(t, coeffs) for t in times])

        # Newton-Raphson ground truth (times relative to tc=0)
        d_newton = z_newton_v(times, 0.0, circular_orbit["p"], circular_orbit["a"], circular_orbit["i"],
            circular_orbit["e"], circular_orbit["w"])

        # Should be accurate near expansion point
        assert_allclose(d_taylor, d_newton, rtol=1e-5, atol=1e-8)


class TestContactPoints:
    """Test contact point and bounding box calculations."""

    def test_find_contact_point_returns_finite(self, circular_orbit):
        """Test that find_contact_point returns finite values."""
        coeffs = solve2d(0.0, **circular_orbit)
        k = 0.1  # planet-to-star radius ratio

        for point in [1, 2, 3, 4]:
            t = find_contact_point(k, point, coeffs)
            assert np.isfinite(t), f"Contact point {point} should be finite"

    def test_contact_points_order(self, circular_orbit):
        """Test that contact points are in correct time order."""
        coeffs = solve2d(0.0, **circular_orbit)
        k = 0.1

        t1 = find_contact_point(k, 1, coeffs)
        t2 = find_contact_point(k, 2, coeffs)
        t3 = find_contact_point(k, 3, coeffs)
        t4 = find_contact_point(k, 4, coeffs)

        # Contact points should be ordered: t1 < t2 < t3 < t4
        assert t1 < t2, "First contact should be before second contact"
        assert t2 < 0, "Second contact should be before center"
        assert 0 < t3, "Third contact should be after center"
        assert t3 < t4, "Third contact should be before fourth contact"

    def test_contact_points_symmetry_circular(self, circular_orbit):
        """Test contact point symmetry for circular orbit."""
        coeffs = solve2d(0.0, **circular_orbit)
        k = 0.1

        t1 = find_contact_point(k, 1, coeffs)
        t4 = find_contact_point(k, 4, coeffs)

        # For circular orbit centered at 0, should be symmetric
        assert_allclose(abs(t1), abs(t4), rtol=1e-5)

    def test_bounding_box(self, circular_orbit):
        """Test bounding box calculation."""
        coeffs = solve2d(0.0, **circular_orbit)
        k = 0.1

        t1, t4 = bounding_box(k, coeffs)

        assert t1 < 0, "T1 should be before center"
        assert t4 > 0, "T4 should be after center"
        assert np.isfinite(t1)
        assert np.isfinite(t4)

    def test_larger_planet_wider_box(self, circular_orbit):
        """Test that larger planet gives wider bounding box."""
        coeffs = solve2d(0.0, **circular_orbit)

        k_small = 0.05
        k_large = 0.15

        t1_small, t4_small = bounding_box(k_small, coeffs)
        t1_large, t4_large = bounding_box(k_large, coeffs)

        width_small = t4_small - t1_small
        width_large = t4_large - t1_large

        assert width_large > width_small, "Larger planet should have wider transit"


class TestAccuracyVsNewton:
    """Comprehensive accuracy tests comparing Taylor series to Newton-Raphson."""

    @pytest.mark.parametrize("e", [0.0, 0.1, 0.3, 0.5])
    def test_accuracy_various_eccentricities(self, e):
        """Test accuracy across various eccentricities."""
        params = {"p": 3.0, "a": 10.0, "i": 1.5, "e": e, "w": 0.5, }

        coeffs = solve2d(0.0, **params)

        # Test near expansion point
        times = np.linspace(-0.01, 0.01, 20)

        for t in times:
            x_taylor, y_taylor = pos(t, 0.0, params["p"], coeffs)
            x_newton, y_newton = xy_newton_v(np.array([t]), 0.0, params["p"], params["a"], params["i"], params["e"],
                params["w"])

            # Relative tolerance increases with eccentricity
            rtol = 1e-5 * (1 + e)
            assert_allclose(x_taylor, x_newton[0], rtol=rtol, atol=1e-8)
            assert_allclose(y_taylor, y_newton[0], rtol=rtol, atol=1e-8)

    def test_degradation_far_from_expansion(self, eccentric_orbit):
        """Test that accuracy degrades away from expansion point."""
        coeffs = solve2d(0.0, **eccentric_orbit)

        # Near expansion point
        t_near = 0.01
        x_near, y_near = pos(t_near, 0.0, eccentric_orbit["p"], coeffs)
        x_newton_near, y_newton_near = xy_newton_v(np.array([t_near]), 0.0, eccentric_orbit["p"], eccentric_orbit["a"],
            eccentric_orbit["i"], eccentric_orbit["e"], eccentric_orbit["w"])
        error_near = np.sqrt((x_near - x_newton_near[0]) ** 2 + (y_near - y_newton_near[0]) ** 2)

        # Far from expansion point
        t_far = 0.3
        x_far, y_far = pos(t_far, 0.0, eccentric_orbit["p"], coeffs)
        x_newton_far, y_newton_far = xy_newton_v(np.array([t_far]), 0.0, eccentric_orbit["p"], eccentric_orbit["a"],
            eccentric_orbit["i"], eccentric_orbit["e"], eccentric_orbit["w"])
        error_far = np.sqrt((x_far - x_newton_far[0]) ** 2 + (y_far - y_newton_far[0]) ** 2)

        # Error should increase with distance from expansion point
        assert error_far > error_near, "Error should increase away from expansion point"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_eccentricity(self):
        """Test with perfectly circular orbit (e=0)."""
        params = {"p": 3.0, "a": 10.0, "i": 1.5, "e": 0.0, "w": 0.0}
        coeffs = solve2d(0.0, **params)

        assert coeffs.shape == (2, 5)
        assert np.all(np.isfinite(coeffs))

    def test_high_eccentricity(self):
        """Test with high eccentricity (e=0.9)."""
        params = {"p": 5.0, "a": 15.0, "i": 1.5, "e": 0.9, "w": 0.5}
        coeffs = solve2d(0.0, **params)

        assert coeffs.shape == (2, 5)
        assert np.all(np.isfinite(coeffs))

    def test_edge_on_inclination(self):
        """Test with edge-on orbit (i = π/2)."""
        params = {"p": 3.0, "a": 10.0, "i": np.pi / 2, "e": 0.1, "w": 0.3}
        coeffs = solve2d(0.0, **params)

        assert coeffs.shape == (2, 5)
        assert np.all(np.isfinite(coeffs))

        # At edge-on, y should be zero
        assert_allclose(coeffs[1, 0], 0.0, atol=1e-10)

    def test_very_short_period(self):
        """Test with ultra-short period orbit."""
        params = {"p": 0.5, "a": 5.0, "i": 1.5, "e": 0.0, "w": 0.0}
        coeffs = solve2d(0.0, **params)

        assert coeffs.shape == (2, 5)
        assert np.all(np.isfinite(coeffs))

    def test_very_long_period(self):
        """Test with long period orbit."""
        params = {"p": 100.0, "a": 50.0, "i": 1.5, "e": 0.2, "w": 0.5}
        coeffs = solve2d(0.0, **params)

        assert coeffs.shape == (2, 5)
        assert np.all(np.isfinite(coeffs))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
