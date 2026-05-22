"""Test suite for meepmeep.backends.numba.utils module.

Covers the orbital-mechanics helpers: eccentricity vector, eclipse timing,
distance factor, impact parameter, mean/true anomaly conversions, and
their analytical derivatives. Tests rely on mathematical identities
(round-trips, limits, periodicity) and finite-difference cross-checks
for the *_with_derivatives variants.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

from meepmeep.backends.numba.utils import (
    eccentricity_vector,
    eclipse_time_offset,
    transit_distance_factor,
    i_from_baew,
    as_from_rhop,
    ta_from_ea,
    mean_anomaly_at_transit,
    mean_anomaly_at_transit_with_derivatives,
    mean_anomaly,
    mean_anomaly_with_derivatives,
    z_from_ta,
    impact_parameter,
    impact_parameter_ec,
    d_from_pkaiews,
)
from meepmeep.backends.numba.newton.newton import ta_newton_s, z_newton_s


TWO_PI = 2.0 * np.pi
HALF_PI = 0.5 * np.pi


# ============================================================
#  eccentricity_vector
# ============================================================

class TestEccentricityVector:
    """Eccentricity vector in the observer's frame."""

    def test_circular_orbit_returns_reference(self):
        """For e < 1e-5 the function should fall back to [-1, 0, 0]."""
        vec = eccentricity_vector(1.5, 0.0, 0.0)
        assert_allclose(vec, np.array([-1.0, 0.0, 0.0]))

    def test_tiny_eccentricity_still_circular(self):
        """e below the 1e-5 threshold should hit the circular branch."""
        vec = eccentricity_vector(1.5, 1e-7, 0.4)
        assert_allclose(vec, np.array([-1.0, 0.0, 0.0]))

    @pytest.mark.parametrize("e", [0.05, 0.3, 0.7, 0.95])
    @pytest.mark.parametrize("w", [0.0, 0.7, np.pi, 3.0])
    @pytest.mark.parametrize("i", [0.5, HALF_PI, 1.4])
    def test_magnitude_equals_eccentricity(self, e, w, i):
        """|e_vec| must equal e: a rotation preserves length."""
        vec = eccentricity_vector(i, e, w)
        assert_allclose(np.linalg.norm(vec), e, rtol=1e-12)

    def test_components_match_formula(self):
        """Direct check against the documented formula."""
        e, w, i = 0.4, 0.6, 1.2
        vec = eccentricity_vector(i, e, w)
        expected = np.array([
            -e * np.cos(w),
            -e * np.sin(w) * np.cos(i),
            e * np.sin(w) * np.sin(i),
        ])
        assert_allclose(vec, expected, rtol=1e-12)

    def test_edge_on_kills_y_component(self):
        """At i = pi/2 (edge-on), ey vanishes since cos(i) = 0."""
        vec = eccentricity_vector(HALF_PI, 0.5, 0.7)
        assert_allclose(vec[1], 0.0, atol=1e-15)


# ============================================================
#  eclipse_time_offset
# ============================================================

class TestEclipseTimeOffset:
    """Time between primary transit and secondary eclipse."""

    @pytest.mark.parametrize("p", [1.0, 3.5, 10.0])
    @pytest.mark.parametrize("w", [0.0, 0.5, 1.7])
    def test_circular_orbit_offset_is_half_period(self, p, w):
        """For e=0 the secondary eclipse is exactly half a period later."""
        offset = eclipse_time_offset(p, 1.5, 0.0, w)
        assert_allclose(offset, 0.5 * p, rtol=1e-12)

    @pytest.mark.parametrize("e", [0.0, 0.2, 0.5, 0.8])
    @pytest.mark.parametrize("w", [0.0, 0.5, np.pi, 2.0])
    def test_offset_in_bounds(self, e, w):
        """Offset must lie within [0, p] for any valid orbit."""
        p = 4.0
        offset = eclipse_time_offset(p, 1.5, e, w)
        assert 0.0 <= offset <= p

    def test_eccentric_offset_asymmetric(self):
        """For eccentric orbits the offset deviates from p/2."""
        p = 3.0
        offset = eclipse_time_offset(p, 1.5, 0.5, 0.3)
        assert abs(offset - 0.5 * p) > 1e-3


# ============================================================
#  transit_distance_factor
# ============================================================

class TestTransitDistanceFactor:
    """Dimensionless r_tr / a factor at primary transit."""

    @pytest.mark.parametrize("w", [0.0, 0.5, np.pi, 4.2])
    def test_circular_factor_is_unity(self, w):
        assert_allclose(transit_distance_factor(0.0, w), 1.0, rtol=1e-12)

    def test_periastron_at_transit(self):
        """w = pi/2 puts periastron at transit: factor = 1 - e."""
        e = 0.4
        assert_allclose(transit_distance_factor(e, HALF_PI), 1.0 - e, rtol=1e-12)

    def test_apoastron_at_transit(self):
        """w = 3pi/2 puts apoastron at transit: factor = 1 + e."""
        e = 0.4
        assert_allclose(transit_distance_factor(e, 1.5 * np.pi), 1.0 + e, rtol=1e-12)

    @pytest.mark.parametrize("e", [0.1, 0.3, 0.7])
    @pytest.mark.parametrize("w", [0.0, 0.5, np.pi, 4.2])
    def test_factor_is_positive(self, e, w):
        """Distance ratio is physically positive for bound orbits."""
        assert transit_distance_factor(e, w) > 0.0


# ============================================================
#  i_from_baew  (inverse of the impact parameter formula)
# ============================================================

class TestIFromBaew:
    """Inversion: given impact parameter, recover inclination."""

    @pytest.mark.parametrize("b", [0.0, 0.2, 0.5, 0.9])
    def test_circular_inverse(self, b):
        """For e=0 the inverse reduces to arccos(b/a)."""
        a = 10.0
        i = i_from_baew(b, a, 0.0, 0.0)
        assert_allclose(i, np.arccos(b / a), rtol=1e-12)

    @pytest.mark.parametrize("e", [0.0, 0.2, 0.5])
    @pytest.mark.parametrize("w", [0.0, 0.5, 1.7])
    @pytest.mark.parametrize("i_in", [0.5, 1.2, 1.45])
    def test_roundtrip_with_impact_parameter_ec(self, e, w, i_in):
        """impact_parameter_ec then i_from_baew must give back i."""
        a = 12.0
        b = impact_parameter_ec(a, i_in, e, w, 1.0)
        i_out = i_from_baew(b, a, e, w)
        assert_allclose(i_out, i_in, rtol=1e-12)

    def test_b_equals_a_factor_gives_edge_grazer(self):
        """b at the geometric limit produces i = 0."""
        a, e, w = 10.0, 0.3, 0.7
        b = a * transit_distance_factor(e, w)
        i = i_from_baew(b, a, e, w)
        assert_allclose(i, 0.0, atol=1e-12)


# ============================================================
#  as_from_rhop  (Kepler's third law -> a/R_star)
# ============================================================

class TestAsFromRhop:
    """Scaled semi-major axis from stellar density."""

    def test_solar_earth_value(self):
        """Sun (rho ~ 1.408 g/cm^3) + 1-year period yields ~215 R_sun."""
        rho_sun = 1.408
        period_year = 365.25
        a_over_rs = as_from_rhop(rho_sun, period_year)
        assert_allclose(a_over_rs, 215.0, rtol=2e-2)

    @pytest.mark.parametrize("period", [0.5, 3.0, 10.0, 100.0])
    def test_period_scaling(self, period):
        """a/R_star scales as period^(2/3) at fixed density."""
        rho = 1.0
        a1 = as_from_rhop(rho, 1.0)
        ap = as_from_rhop(rho, period)
        assert_allclose(ap / a1, period ** (2.0 / 3.0), rtol=1e-12)

    @pytest.mark.parametrize("rho", [0.5, 1.0, 2.5, 10.0])
    def test_density_scaling(self, rho):
        """a/R_star scales as rho^(1/3) at fixed period."""
        period = 3.0
        a1 = as_from_rhop(1.0, period)
        ar = as_from_rhop(rho, period)
        assert_allclose(ar / a1, rho ** (1.0 / 3.0), rtol=1e-12)


# ============================================================
#  ta_from_ea
# ============================================================

class TestTaFromEa:
    """Eccentric anomaly -> true anomaly conversion."""

    @pytest.mark.parametrize("ea", np.linspace(-np.pi + 1e-3, np.pi - 1e-3, 11))
    def test_circular_identity(self, ea):
        """For e=0 the true anomaly equals the eccentric anomaly."""
        assert_allclose(ta_from_ea(ea, 0.0), ea, atol=1e-12)

    @pytest.mark.parametrize("ecc", [0.0, 0.2, 0.5, 0.9])
    def test_fixed_points(self, ecc):
        """E = 0 and E = pi map to f = 0 and f = pi respectively."""
        assert_allclose(ta_from_ea(0.0, ecc), 0.0, atol=1e-12)
        assert_allclose(ta_from_ea(np.pi, ecc), np.pi, atol=1e-12)

    @pytest.mark.parametrize("ecc", [0.2, 0.5, 0.8])
    def test_kepler_geometric_relation(self, ecc):
        """cos(f) = (cos(E) - e) / (1 - e*cos(E)) must hold."""
        for ea in np.linspace(-np.pi + 0.1, np.pi - 0.1, 12):
            f = ta_from_ea(ea, ecc)
            expected_cos = (np.cos(ea) - ecc) / (1.0 - ecc * np.cos(ea))
            assert_allclose(np.cos(f), expected_cos, atol=1e-12)


# ============================================================
#  mean_anomaly_at_transit
# ============================================================

class TestMeanAnomalyAtTransit:
    """Mean anomaly at the moment of primary transit."""

    def test_circular_w_pihalf_is_zero(self):
        """For e=0 and w=pi/2 (transit at f=0) the mean anomaly is zero."""
        m = mean_anomaly_at_transit(0.0, HALF_PI)
        assert_allclose(m, 0.0, atol=1e-12)

    @pytest.mark.parametrize("w", [0.0, 0.4, 1.0, 2.5])
    def test_circular_equals_pihalf_minus_w(self, w):
        """For e=0 the result is pi/2 - w wrapped to (-pi, pi]."""
        m = mean_anomaly_at_transit(0.0, w)
        expected = np.arctan2(np.sin(HALF_PI - w), np.cos(HALF_PI - w))
        assert_allclose(m, expected, atol=1e-12)

    @pytest.mark.parametrize("e", [0.0, 0.2, 0.5])
    @pytest.mark.parametrize("w", [0.0, 0.6, 2.0])
    def test_kepler_consistency_with_newton(self, e, w):
        """ta_newton_s at t0 must return f = pi/2 - w (the transit anomaly)."""
        t0, p = 0.0, 3.0
        f = ta_newton_s(t0, t0, p, e, w)
        expected_f = np.arctan2(np.sin(HALF_PI - w), np.cos(HALF_PI - w))
        assert_allclose(f, expected_f, atol=1e-7)


# ============================================================
#  mean_anomaly_at_transit_with_derivatives
# ============================================================

class TestMeanAnomalyAtTransitWithDerivatives:
    """Value + derivatives variant must agree with the scalar version and FD."""

    @pytest.mark.parametrize("e", [0.0, 0.1, 0.4, 0.7])
    @pytest.mark.parametrize("w", [0.0, 0.5, 1.5, 2.8])
    def test_value_matches_scalar_version(self, e, w):
        m, _, _ = mean_anomaly_at_transit_with_derivatives(e, w)
        m_ref = mean_anomaly_at_transit(e, w)
        assert_allclose(m, m_ref, rtol=1e-12)

    @pytest.mark.parametrize("e", [0.1, 0.4, 0.7])
    @pytest.mark.parametrize("w", [0.3, 1.2, 2.5])
    def test_de_matches_finite_difference(self, e, w):
        h = 1e-6
        _, dm_de, _ = mean_anomaly_at_transit_with_derivatives(e, w)
        fd = (mean_anomaly_at_transit(e + h, w) - mean_anomaly_at_transit(e - h, w)) / (2 * h)
        assert_allclose(dm_de, fd, atol=1e-7)

    @pytest.mark.parametrize("e", [0.1, 0.4, 0.7])
    @pytest.mark.parametrize("w", [0.3, 1.2, 2.5])
    def test_dw_matches_finite_difference(self, e, w):
        h = 1e-6
        _, _, dm_dw = mean_anomaly_at_transit_with_derivatives(e, w)
        fd = (mean_anomaly_at_transit(e, w + h) - mean_anomaly_at_transit(e, w - h)) / (2 * h)
        assert_allclose(dm_dw, fd, atol=1e-7)


# ============================================================
#  mean_anomaly
# ============================================================

class TestMeanAnomaly:
    """Wrapped mean anomaly as a function of time."""

    def test_in_bounds(self):
        """Output must lie in [0, 2*pi)."""
        for t in np.linspace(-10, 10, 51):
            m = mean_anomaly(t, 0.0, 3.0, 0.3, 0.7)
            assert 0.0 <= m < TWO_PI

    @pytest.mark.parametrize("e", [0.0, 0.2, 0.6])
    @pytest.mark.parametrize("w", [0.0, 0.5, 2.1])
    def test_at_t0_equals_transit_anomaly_mod_2pi(self, e, w):
        """At t = t0 the mean anomaly equals mean_anomaly_at_transit mod 2pi."""
        t0, p = 0.0, 3.0
        m = mean_anomaly(t0, t0, p, e, w)
        m_tr = mean_anomaly_at_transit(e, w) % TWO_PI
        assert_allclose(m, m_tr, atol=1e-12)

    @pytest.mark.parametrize("e", [0.0, 0.3, 0.7])
    def test_periodicity(self, e):
        """m(t) and m(t + p) must agree."""
        t0, p, w = 0.0, 3.0, 0.4
        for t in np.linspace(0.0, p, 8):
            m1 = mean_anomaly(t, t0, p, e, w)
            m2 = mean_anomaly(t + p, t0, p, e, w)
            assert_allclose(m1, m2, atol=1e-12)


# ============================================================
#  mean_anomaly_with_derivatives
# ============================================================

class TestMeanAnomalyWithDerivatives:
    """Value + derivatives variant: cross-check value and all four derivatives."""

    @pytest.mark.parametrize("e", [0.0, 0.2, 0.5])
    @pytest.mark.parametrize("w", [0.3, 1.5])
    def test_value_matches_wrapped(self, e, w):
        """Unwrapped m mod 2pi must equal mean_anomaly."""
        t, t0, p = 1.0, 0.0, 3.0
        m, *_ = mean_anomaly_with_derivatives(t, t0, p, e, w)
        m_ref = mean_anomaly(t, t0, p, e, w)
        assert_allclose(m % TWO_PI, m_ref, atol=1e-12)

    def test_dt0_matches_finite_difference(self):
        t, t0, p, e, w = 1.0, 0.0, 3.0, 0.3, 0.7
        h = 1e-6
        _, dm_dt0, _, _, _ = mean_anomaly_with_derivatives(t, t0, p, e, w)
        m_plus, *_ = mean_anomaly_with_derivatives(t, t0 + h, p, e, w)
        m_minus, *_ = mean_anomaly_with_derivatives(t, t0 - h, p, e, w)
        fd = (m_plus - m_minus) / (2 * h)
        assert_allclose(dm_dt0, fd, atol=1e-7)

    def test_dp_matches_finite_difference(self):
        t, t0, p, e, w = 1.0, 0.0, 3.0, 0.3, 0.7
        h = 1e-6
        _, _, dm_dp, _, _ = mean_anomaly_with_derivatives(t, t0, p, e, w)
        m_plus, *_ = mean_anomaly_with_derivatives(t, t0, p + h, e, w)
        m_minus, *_ = mean_anomaly_with_derivatives(t, t0, p - h, e, w)
        fd = (m_plus - m_minus) / (2 * h)
        assert_allclose(dm_dp, fd, atol=1e-7)

    def test_de_matches_finite_difference(self):
        t, t0, p, e, w = 1.0, 0.0, 3.0, 0.3, 0.7
        h = 1e-6
        _, _, _, dm_de, _ = mean_anomaly_with_derivatives(t, t0, p, e, w)
        m_plus, *_ = mean_anomaly_with_derivatives(t, t0, p, e + h, w)
        m_minus, *_ = mean_anomaly_with_derivatives(t, t0, p, e - h, w)
        fd = (m_plus - m_minus) / (2 * h)
        assert_allclose(dm_de, fd, atol=1e-7)

    def test_dw_matches_finite_difference(self):
        t, t0, p, e, w = 1.0, 0.0, 3.0, 0.3, 0.7
        h = 1e-6
        _, _, _, _, dm_dw = mean_anomaly_with_derivatives(t, t0, p, e, w)
        m_plus, *_ = mean_anomaly_with_derivatives(t, t0, p, e, w + h)
        m_minus, *_ = mean_anomaly_with_derivatives(t, t0, p, e, w - h)
        fd = (m_plus - m_minus) / (2 * h)
        assert_allclose(dm_dw, fd, atol=1e-7)


# ============================================================
#  z_from_ta
# ============================================================

class TestZFromTa:
    """Sky-projected separation as a function of true anomaly."""

    @pytest.mark.parametrize("e", [0.0, 0.2, 0.5])
    @pytest.mark.parametrize("w", [0.0, 0.5, 2.1])
    def test_at_transit_matches_impact_parameter(self, e, w):
        """At f = pi/2 - w (transit), z equals impact_parameter_ec for tr_sign=+1."""
        a, i = 10.0, 1.45
        f_tr = HALF_PI - w
        z = z_from_ta(f_tr, a, i, e, w)
        b = impact_parameter_ec(a, i, e, w, 1.0)
        assert_allclose(z, b, rtol=1e-12)

    @pytest.mark.parametrize("e", [0.0, 0.3])
    @pytest.mark.parametrize("w", [0.0, 1.2])
    def test_at_eclipse_z_is_negative(self, e, w):
        """At f = -pi/2 - w (eclipse) the projected distance carries a negative sign."""
        a, i = 10.0, 1.45
        f_ec = -HALF_PI - w
        z = z_from_ta(f_ec, a, i, e, w)
        assert z < 0.0

    def test_edge_on_central_transit_is_zero(self):
        """Edge-on (i=pi/2) at transit gives z=0."""
        z = z_from_ta(HALF_PI, 10.0, HALF_PI, 0.0, 0.0)
        assert_allclose(z, 0.0, atol=1e-12)

    def test_matches_z_newton_at_transit(self):
        """z_from_ta(transit anomaly) must agree with z_newton_s at t=t0."""
        t0, p, a, i, e, w = 0.0, 3.0, 10.0, 1.45, 0.2, 0.4
        f = ta_newton_s(t0, t0, p, e, w)
        z_a = z_from_ta(f, a, i, e, w)
        z_b = z_newton_s(t0, t0, p, a, i, e, w)
        assert_allclose(z_a, z_b, rtol=1e-7)


# ============================================================
#  impact_parameter & impact_parameter_ec
# ============================================================

class TestImpactParameter:
    """Circular and eccentric impact parameter formulas."""

    def test_edge_on_circular_is_zero(self):
        assert_allclose(impact_parameter(10.0, HALF_PI), 0.0, atol=1e-15)

    @pytest.mark.parametrize("a", [5.0, 10.0, 20.0])
    @pytest.mark.parametrize("i", [0.5, 1.0, 1.4])
    def test_circular_formula(self, a, i):
        assert_allclose(impact_parameter(a, i), a * np.cos(i), rtol=1e-12)

    @pytest.mark.parametrize("a", [5.0, 10.0])
    @pytest.mark.parametrize("i", [0.5, 1.45])
    @pytest.mark.parametrize("w", [0.0, 1.2, 3.5])
    def test_eccentric_reduces_to_circular(self, a, i, w):
        """For e=0 impact_parameter_ec must equal impact_parameter."""
        b_ec = impact_parameter_ec(a, i, 0.0, w, 1.0)
        b_circ = impact_parameter(a, i)
        assert_allclose(b_ec, b_circ, rtol=1e-12)

    def test_transit_vs_eclipse_asymmetry(self):
        """Primary (tr_sign=1) and secondary (tr_sign=-1) impact parameters differ for e!=0, w!=0."""
        a, i, e, w = 10.0, 1.45, 0.4, 0.6
        b_tr = impact_parameter_ec(a, i, e, w, 1.0)
        b_ec = impact_parameter_ec(a, i, e, w, -1.0)
        assert abs(b_tr - b_ec) > 1e-3


# ============================================================
#  d_from_pkaiews  (transit duration)
# ============================================================

class TestDFromPkaiews:
    """Analytical transit and eclipse duration."""

    def test_t14_greater_than_t23(self):
        """T14 must exceed T23 by twice the half-ingress duration.

        Note: i=1.52 keeps the impact parameter inside (1-k), so the
        sqrt arguments stay real (avoids grazing).
        """
        p, k, a, i, e, w = 3.0, 0.1, 10.0, 1.52, 0.0, 0.0
        t14 = d_from_pkaiews(p, k, a, i, e, w, 1.0, kind=14)
        t23 = d_from_pkaiews(p, k, a, i, e, w, 1.0, kind=23)
        assert t14 > t23

    def test_central_circular_known_value(self):
        """Closed-form check: edge-on, e=0, b=0 -> T14 = p/pi * arcsin((1+k)/a)."""
        p, k, a, i, e, w = 3.0, 0.1, 10.0, HALF_PI, 0.0, 0.0
        t14 = d_from_pkaiews(p, k, a, i, e, w, 1.0, kind=14)
        expected = p / np.pi * np.arcsin((1.0 + k) / a)
        assert_allclose(t14, expected, rtol=1e-12)

    def test_central_circular_t23(self):
        """At b=0, T23 = p/pi * arcsin((1-k)/a)."""
        p, k, a, i, e, w = 3.0, 0.1, 10.0, HALF_PI, 0.0, 0.0
        t23 = d_from_pkaiews(p, k, a, i, e, w, 1.0, kind=23)
        expected = p / np.pi * np.arcsin((1.0 - k) / a)
        assert_allclose(t23, expected, rtol=1e-12)

    @pytest.mark.parametrize("p", [1.0, 3.0, 10.0])
    def test_duration_scales_with_period(self, p):
        """Duration scales linearly with period at fixed geometry."""
        k, a, i, e, w = 0.1, 10.0, HALF_PI, 0.0, 0.0
        t_p = d_from_pkaiews(p, k, a, i, e, w, 1.0, kind=14)
        t_1 = d_from_pkaiews(1.0, k, a, i, e, w, 1.0, kind=14)
        assert_allclose(t_p / t_1, p, rtol=1e-12)

    def test_transit_eclipse_asymmetry(self):
        """For eccentric orbits transit and eclipse durations differ."""
        p, k, a, i, e, w = 3.0, 0.1, 10.0, 1.52, 0.4, 0.6
        d_tr = d_from_pkaiews(p, k, a, i, e, w, 1.0, kind=14)
        d_ec = d_from_pkaiews(p, k, a, i, e, w, -1.0, kind=14)
        assert abs(d_tr - d_ec) > 1e-4
