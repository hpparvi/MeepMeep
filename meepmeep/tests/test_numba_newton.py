"""Test suite for meepmeep.backends.numba.newton.newton module.

Tests validate the Newton-Raphson Kepler equation solvers and derived
orbital quantities (position, projected distance, radial velocity,
eclipse light travel time) using mathematical identities and physical
invariants.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

from meepmeep.backends.numba.newton.newton import (
    ea_from_ma,
    ea_newton_s, ea_newton_v,
    ta_newton_s, ta_newton_v,
    xy_newton_v, xyz_newton_v,
    z_newton_s, z_newton_v,
    rv_newton_v,
    eclipse_light_travel_time,
)


@pytest.fixture
def circular_orbit():
    return {"p": 3.0, "a": 10.0, "i": 1.5, "e": 0.0, "w": 0.0}


@pytest.fixture
def eccentric_orbit():
    return {"p": 5.0, "a": 15.0, "i": 1.55, "e": 0.3, "w": 0.5}


@pytest.fixture
def high_e_orbit():
    return {"p": 7.0, "a": 20.0, "i": 1.4, "e": 0.7, "w": 1.2}


@pytest.fixture
def edge_on_orbit():
    return {"p": 2.5, "a": 8.0, "i": np.pi / 2, "e": 0.1, "w": 0.3}


# ============================================================
#  ea_from_ma: Kepler's equation solver
# ============================================================

class TestEaFromMa:
    """Test the core Kepler equation solver E - e*sin(E) = M."""

    def test_circular_orbit_identity(self):
        """For e=0, E should equal M exactly."""
        for ma in np.linspace(0, 2 * np.pi, 20):
            ea = ea_from_ma(ma, 0.0)
            assert_allclose(ea, ma, atol=1e-12)

    @pytest.mark.parametrize("ecc", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9])
    def test_kepler_equation_satisfied(self, ecc):
        """The solution must satisfy E - e*sin(E) = M."""
        for ma in np.linspace(0, 2 * np.pi, 30, endpoint=False):
            ea = ea_from_ma(ma, ecc)
            residual = ea - ecc * np.sin(ea) - ma
            assert abs(residual) < 1e-12, (
                f"Kepler residual {residual:.2e} for M={ma:.4f}, e={ecc}"
            )

    def test_high_eccentricity_convergence(self):
        """Should converge even for very high eccentricity."""
        for ma in np.linspace(0.1, 2 * np.pi - 0.1, 15):
            ea = ea_from_ma(ma, 0.99)
            residual = ea - 0.99 * np.sin(ea) - ma
            assert abs(residual) < 1e-10

    @pytest.mark.parametrize("ma", [0.0, np.pi, 2 * np.pi])
    def test_boundary_mean_anomalies(self, ma):
        """Boundary M values should produce valid results."""
        ea = ea_from_ma(ma, 0.3)
        assert np.isfinite(ea)
        residual = ea - 0.3 * np.sin(ea) - ma
        assert abs(residual) < 1e-12


# ============================================================
#  ea_newton_s / ea_newton_v: Eccentric anomaly from time
# ============================================================

class TestEaNewton:
    """Test scalar and vector eccentric anomaly computation."""

    def test_scalar_vector_consistency(self, eccentric_orbit):
        """Scalar and vector versions must agree element-wise."""
        p = eccentric_orbit["p"]
        e = eccentric_orbit["e"]
        w = eccentric_orbit["w"]
        times = np.linspace(-0.5 * p, 0.5 * p, 30)

        ea_vec = ea_newton_v(times, 0.0, p, e, w)
        for j, t in enumerate(times):
            ea_s = ea_newton_s(t, 0.0, p, e, w)
            assert_allclose(ea_s, ea_vec[j], atol=1e-10,
                            err_msg=f"Mismatch at t={t:.4f}")

    def test_periodicity(self, eccentric_orbit):
        """Eccentric anomaly should repeat after one period."""
        p = eccentric_orbit["p"]
        e = eccentric_orbit["e"]
        w = eccentric_orbit["w"]
        times = np.linspace(0.0, 0.4 * p, 10)

        ea1 = ea_newton_v(times, 0.0, p, e, w)
        ea2 = ea_newton_v(times + p, 0.0, p, e, w)
        # EA wraps mod 2π so compare via sin/cos
        assert_allclose(np.sin(ea1), np.sin(ea2), atol=1e-8)
        assert_allclose(np.cos(ea1), np.cos(ea2), atol=1e-8)

    def test_circular_orbit_linear(self, circular_orbit):
        """For e=0, eccentric anomaly should increase linearly with time."""
        p = circular_orbit["p"]
        times = np.linspace(0.0, p, 50, endpoint=False)
        ea = ea_newton_v(times, 0.0, p, 0.0, 0.0)
        # Differences should be nearly constant
        diffs = np.diff(ea)
        # Handle wrapping
        diffs = np.mod(diffs + np.pi, 2 * np.pi) - np.pi
        assert_allclose(diffs, diffs[0], atol=1e-6)


# ============================================================
#  ta_newton_s / ta_newton_v: True anomaly
# ============================================================

class TestTaNewton:
    """Test true anomaly computation."""

    def test_scalar_vector_consistency(self, eccentric_orbit):
        """Scalar and vector versions must agree."""
        p = eccentric_orbit["p"]
        e = eccentric_orbit["e"]
        w = eccentric_orbit["w"]
        times = np.linspace(-0.3 * p, 0.3 * p, 20)

        ta_vec = ta_newton_v(times, 0.0, p, e, w)
        for j, t in enumerate(times):
            ta_s = ta_newton_s(t, 0.0, p, e, w)
            assert_allclose(ta_s, ta_vec[j], atol=1e-10)

    def test_circular_orbit_ta_equals_ea(self, circular_orbit):
        """For e=0, true anomaly equals eccentric anomaly."""
        p = circular_orbit["p"]
        times = np.linspace(0.0, p, 30, endpoint=False)
        ea = ea_newton_v(times, 0.0, p, 0.0, 0.0)
        ta = ta_newton_v(times, 0.0, p, 0.0, 0.0)
        assert_allclose(np.sin(ta), np.sin(ea), atol=1e-8)
        assert_allclose(np.cos(ta), np.cos(ea), atol=1e-8)

    def test_periodicity(self, eccentric_orbit):
        """True anomaly should repeat after one period."""
        p = eccentric_orbit["p"]
        e = eccentric_orbit["e"]
        w = eccentric_orbit["w"]
        times = np.linspace(0.1, 0.4 * p, 10)

        ta1 = ta_newton_v(times, 0.0, p, e, w)
        ta2 = ta_newton_v(times + p, 0.0, p, e, w)
        assert_allclose(np.sin(ta1), np.sin(ta2), atol=1e-8)
        assert_allclose(np.cos(ta1), np.cos(ta2), atol=1e-8)

    def test_range(self, eccentric_orbit):
        """True anomaly should be in [-π, π]."""
        p = eccentric_orbit["p"]
        times = np.linspace(0, p, 100, endpoint=False)
        ta = ta_newton_v(times, 0.0, p,
                         eccentric_orbit["e"], eccentric_orbit["w"])
        assert np.all(ta >= -np.pi) and np.all(ta <= np.pi)


# ============================================================
#  xy_newton_v: 2D position
# ============================================================

class TestXyNewton:
    """Test 2D sky-projected position."""

    def test_circular_orbit_at_transit(self, circular_orbit):
        """At transit center, x ≈ 0 and y ≈ -a*cos(i)."""
        x, y = xy_newton_v(np.array([0.0]), 0.0,
                           circular_orbit["p"], circular_orbit["a"],
                           circular_orbit["i"], circular_orbit["e"],
                           circular_orbit["w"])
        assert abs(x[0]) < 0.1, "x should be near zero at transit"
        expected_y = -circular_orbit["a"] * np.cos(circular_orbit["i"])
        assert_allclose(y[0], expected_y, rtol=1e-4)

    def test_periodicity(self, eccentric_orbit):
        """Position should repeat after one period."""
        p = eccentric_orbit["p"]
        times = np.linspace(0.1, 0.4 * p, 15)
        x1, y1 = xy_newton_v(times, 0.0, p,
                              eccentric_orbit["a"], eccentric_orbit["i"],
                              eccentric_orbit["e"], eccentric_orbit["w"])
        x2, y2 = xy_newton_v(times + p, 0.0, p,
                              eccentric_orbit["a"], eccentric_orbit["i"],
                              eccentric_orbit["e"], eccentric_orbit["w"])
        assert_allclose(x1, x2, atol=1e-8)
        assert_allclose(y1, y2, atol=1e-8)

    def test_radius_bounds(self, eccentric_orbit):
        """Orbital radius should be within a*(1-e) and a*(1+e)."""
        p = eccentric_orbit["p"]
        a = eccentric_orbit["a"]
        e = eccentric_orbit["e"]
        times = np.linspace(0, p, 100, endpoint=False)
        x, y = xy_newton_v(times, 0.0, p, a,
                           eccentric_orbit["i"], e, eccentric_orbit["w"])
        # x,y are sky-projected so r_sky ≤ r_3d ≤ a*(1+e)
        r_sky = np.sqrt(x**2 + y**2)
        assert np.all(r_sky <= a * (1 + e) + 1e-8)

    def test_xy_matches_xyz(self, eccentric_orbit):
        """x,y from xy_newton_v should match xyz_newton_v."""
        p = eccentric_orbit["p"]
        times = np.linspace(-0.2 * p, 0.2 * p, 20)
        x2, y2 = xy_newton_v(times, 0.0, p,
                              eccentric_orbit["a"], eccentric_orbit["i"],
                              eccentric_orbit["e"], eccentric_orbit["w"])
        x3, y3, z3 = xyz_newton_v(times, 0.0, p,
                                   eccentric_orbit["a"], eccentric_orbit["i"],
                                   eccentric_orbit["e"], eccentric_orbit["w"])
        assert_allclose(x2, x3, atol=1e-12)
        assert_allclose(y2, y3, atol=1e-12)


# ============================================================
#  xyz_newton_v: 3D position
# ============================================================

class TestXyzNewton:
    """Test 3D position."""

    def test_edge_on_y_zero_at_transit(self, edge_on_orbit):
        """For i=π/2, y should be zero at transit."""
        x, y, z = xyz_newton_v(np.array([0.0]), 0.0,
                                edge_on_orbit["p"], edge_on_orbit["a"],
                                edge_on_orbit["i"], edge_on_orbit["e"],
                                edge_on_orbit["w"])
        assert_allclose(y[0], 0.0, atol=1e-8,
                        err_msg="y should be zero for edge-on at transit")

    def test_z_positive_at_transit(self, circular_orbit):
        """At transit, planet is in front of star so z > 0."""
        x, y, z = xyz_newton_v(np.array([0.0]), 0.0,
                                circular_orbit["p"], circular_orbit["a"],
                                circular_orbit["i"], circular_orbit["e"],
                                circular_orbit["w"])
        assert z[0] > 0, "z should be positive at transit (planet in front)"

    def test_3d_radius_bounds(self, eccentric_orbit):
        """3D radius must be within a*(1-e) and a*(1+e)."""
        p = eccentric_orbit["p"]
        a = eccentric_orbit["a"]
        e = eccentric_orbit["e"]
        times = np.linspace(0, p, 100, endpoint=False)
        x, y, z = xyz_newton_v(times, 0.0, p, a,
                                eccentric_orbit["i"], e,
                                eccentric_orbit["w"])
        r = np.sqrt(x**2 + y**2 + z**2)
        assert np.all(r >= a * (1 - e) - 1e-8)
        assert np.all(r <= a * (1 + e) + 1e-8)

    def test_all_finite(self, high_e_orbit):
        """All coordinates should be finite across the orbit."""
        p = high_e_orbit["p"]
        times = np.linspace(0, p, 50, endpoint=False)
        x, y, z = xyz_newton_v(times, 0.0, p,
                                high_e_orbit["a"], high_e_orbit["i"],
                                high_e_orbit["e"], high_e_orbit["w"])
        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(y))
        assert np.all(np.isfinite(z))


# ============================================================
#  z_newton_s / z_newton_v: Projected distance
# ============================================================

class TestZNewton:
    """Test projected distance computation."""

    def test_scalar_vector_magnitude_consistency(self, eccentric_orbit):
        """Scalar and vector versions must agree in magnitude.

        Note: z_from_ta_s uses copysign while z_from_ta_v uses sign, which
        can produce different signs far from transit. We compare magnitudes.
        """
        p = eccentric_orbit["p"]
        times = np.linspace(-0.2 * p, 0.2 * p, 20)
        z_vec = z_newton_v(times, 0.0, p,
                           eccentric_orbit["a"], eccentric_orbit["i"],
                           eccentric_orbit["e"], eccentric_orbit["w"])
        for j, t in enumerate(times):
            z_s = z_newton_s(t, 0.0, p,
                             eccentric_orbit["a"], eccentric_orbit["i"],
                             eccentric_orbit["e"], eccentric_orbit["w"])
            assert_allclose(np.abs(z_s), np.abs(z_vec[j]), atol=1e-8,
                            err_msg=f"Magnitude mismatch at t={t:.4f}")

    def test_minimum_near_transit(self, circular_orbit):
        """Projected distance should be smallest near transit center."""
        p = circular_orbit["p"]
        times = np.linspace(-0.1 * p, 0.1 * p, 50)
        z = z_newton_v(times, 0.0, p,
                       circular_orbit["a"], circular_orbit["i"],
                       circular_orbit["e"], circular_orbit["w"])
        # z at transit should be the minimum (positive) value
        idx_center = len(times) // 2
        assert np.abs(z[idx_center]) <= np.abs(z[0]) + 1e-8

    def test_positive_at_transit(self, circular_orbit):
        """z should be positive at transit (planet in front of star)."""
        z = z_newton_s(0.0, 0.0, circular_orbit["p"],
                       circular_orbit["a"], circular_orbit["i"],
                       circular_orbit["e"], circular_orbit["w"])
        assert z > 0, "z should be positive at transit"


# ============================================================
#  rv_newton_v: Radial velocity
# ============================================================

class TestRvNewton:
    """Test radial velocity computation."""

    def test_circular_orbit_amplitude(self):
        """For a circular orbit, RV amplitude should equal K."""
        k = 50.0  # m/s semi-amplitude
        p, e, w = 3.0, 0.0, 0.0
        times = np.linspace(0, p, 200, endpoint=False)
        rv = rv_newton_v(times, k, 0.0, p, e, w)
        assert_allclose(np.max(np.abs(rv)), k, rtol=1e-2,
                        err_msg="RV amplitude should equal K for circular orbit")

    def test_periodicity(self, eccentric_orbit):
        """RV should repeat after one period."""
        k = 30.0
        p = eccentric_orbit["p"]
        e = eccentric_orbit["e"]
        w = eccentric_orbit["w"]
        times = np.linspace(0.1, 0.4 * p, 15)
        rv1 = rv_newton_v(times, k, 0.0, p, e, w)
        rv2 = rv_newton_v(times + p, k, 0.0, p, e, w)
        assert_allclose(rv1, rv2, atol=1e-8)

    def test_zero_k_gives_zero_rv(self, eccentric_orbit):
        """K=0 should give zero RV everywhere."""
        p = eccentric_orbit["p"]
        times = np.linspace(0, p, 30)
        rv = rv_newton_v(times, 0.0, 0.0, p,
                         eccentric_orbit["e"], eccentric_orbit["w"])
        assert_allclose(rv, 0.0, atol=1e-15)

    def test_all_finite(self, high_e_orbit):
        """RV should be finite across the full orbit."""
        p = high_e_orbit["p"]
        times = np.linspace(0, p, 50, endpoint=False)
        rv = rv_newton_v(times, 100.0, 0.0, p,
                         high_e_orbit["e"], high_e_orbit["w"])
        assert np.all(np.isfinite(rv))


# ============================================================
#  eclipse_light_travel_time
# ============================================================

class TestEclipseLightTravelTime:
    """Test eclipse light travel time difference."""

    def test_circular_orbit_near_zero(self):
        """For e=0, w=0, light travel time should be very small."""
        ltt = eclipse_light_travel_time(p=3.0, a=10.0, i=np.pi / 2,
                                        e=0.0, w=0.0, rstar=1.0)
        # Should be close to zero for symmetric circular orbit
        assert abs(ltt) < 1e-3, "LTT should be near zero for circular orbit"

    def test_finite_for_eccentric(self):
        """Should return a finite value for eccentric orbits."""
        ltt = eclipse_light_travel_time(p=5.0, a=15.0, i=1.5,
                                        e=0.3, w=0.5, rstar=1.0)
        assert np.isfinite(ltt)

    def test_physical_scale(self):
        """Result should be small (seconds to minutes in day units)."""
        ltt = eclipse_light_travel_time(p=3.0, a=10.0, i=1.5,
                                        e=0.3, w=0.5, rstar=1.0)
        # Light travel time across ~20 R_sun is ~10 seconds ≈ 1e-4 days
        assert abs(ltt) < 0.01, "LTT should be much less than a day"

    def test_scales_with_rstar(self):
        """LTT should scale linearly with stellar radius."""
        ltt1 = eclipse_light_travel_time(p=5.0, a=15.0, i=1.5,
                                         e=0.3, w=0.5, rstar=1.0)
        ltt2 = eclipse_light_travel_time(p=5.0, a=15.0, i=1.5,
                                         e=0.3, w=0.5, rstar=2.0)
        assert_allclose(ltt2, 2.0 * ltt1, rtol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
