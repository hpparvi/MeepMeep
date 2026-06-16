"""Test suite for the high-level Expansion2D wrapper (meepmeep.expansion2d).

Expansion2D is a thin convenience layer over the single-expansion-point 2D Taylor
evaluators with an Orbit-style API: orbital elements are bound via
``set_pars`` (the constructor forwards to it), the observation times via
``set_data``, and ``position`` / ``projected_separation`` are properties
evaluated over the bound time grid. These tests check (a) that the
properties forward to the low-level functions faithfully, (b) that values
match the exact Newton-Raphson reference near the expansion point, (c) that derivative
mode matches finite differences, and (d) that the contact-point / duration
helpers behave sensibly and return absolute times.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

from meepmeep.expansion2d import Expansion2D
from meepmeep.backends.numba.point2d import solve2d, sep
from meepmeep.backends.numba.newton.newton import xy_newton_v, z_newton_v


@pytest.fixture
def circular_orbit():
    """Circular orbit parameters."""
    return {"p": 3.0, "a": 10.0, "i": 1.5, "e": 0.0, "w": 0.0}


@pytest.fixture
def eccentric_orbit():
    """Moderately eccentric orbit parameters."""
    return {"p": 5.0, "a": 15.0, "i": 1.55, "e": 0.3, "w": 0.5}


class TestConstruction:
    """Construction and coefficient storage."""

    def test_import_and_build(self, circular_orbit):
        k = Expansion2D(te=0.0, tc=0.0, **circular_orbit)
        assert k._coeffs.shape == (2, 5)
        assert k._dcoeffs is None

    def test_derivative_mode_stores_dcoeffs(self, circular_orbit):
        k = Expansion2D(te=0.0, tc=0.0, derivatives=True, **circular_orbit)
        assert k._coeffs.shape == (2, 5)
        assert k._dcoeffs.shape == (7, 2, 5)

    def test_constructor_uses_set_pars(self, eccentric_orbit):
        """Rebinding via set_pars must equal constructing afresh."""
        k = Expansion2D(te=0.0, tc=0.0, p=1.0, a=5.0, i=1.4, e=0.0, w=0.0)
        k.set_pars(tc=10.0, **eccentric_orbit)
        fresh = Expansion2D(te=0.0, tc=10.0, **eccentric_orbit)
        assert_allclose(k._coeffs, fresh._coeffs, rtol=1e-12)
        assert k._ep_time == fresh._ep_time


class TestForwarding:
    """The properties should forward to the low-level API unchanged."""

    def test_separation_matches_low_level(self, eccentric_orbit):
        te, tc = 0.0, 1234.5
        k = Expansion2D(te=te, tc=tc, **eccentric_orbit)
        c = solve2d(te, **eccentric_orbit)
        times = tc + np.linspace(-0.02, 0.02, 7)
        k.set_data(times)
        expected = np.array([sep(t - tc, te, eccentric_orbit["p"], c) for t in times])
        assert_allclose(k.projected_separation, expected, rtol=1e-12)

    def test_absolute_time_offset(self, circular_orbit):
        """Evaluating at tc+dt with tc set must equal dt with tc=0."""
        dt = np.array([0.013])
        k0 = Expansion2D(te=0.0, tc=0.0, **circular_orbit)
        kt = Expansion2D(te=0.0, tc=500.0, **circular_orbit)
        k0.set_data(dt)
        kt.set_data(500.0 + dt)
        assert_allclose(k0.projected_separation, kt.projected_separation, rtol=1e-12)


class TestAccuracyVsNewton:
    """Values match the exact Newton-Raphson reference near the expansion point."""

    @pytest.mark.accuracy
    def test_position_vs_newton(self, circular_orbit):
        k = Expansion2D(te=0.0, tc=0.0, **circular_orbit)
        times = np.linspace(-0.02, 0.02, 11)
        k.set_data(times)
        xs, ys = k.position
        xn, yn = xy_newton_v(times, 0.0, **circular_orbit)
        assert_allclose(xs, xn, rtol=1e-6, atol=1e-8)
        assert_allclose(ys, yn, rtol=1e-6, atol=1e-8)

    @pytest.mark.accuracy
    def test_separation_vs_newton(self, eccentric_orbit):
        k = Expansion2D(te=0.0, tc=0.0, **eccentric_orbit)
        times = np.linspace(-0.02, 0.02, 11)
        k.set_data(times)
        d_ep = k.projected_separation
        d_newton = z_newton_v(times, 0.0, **eccentric_orbit)
        assert_allclose(d_ep, d_newton, rtol=1e-5, atol=1e-8)


class TestDerivatives:
    """Analytic derivatives match central finite differences."""

    @pytest.mark.accuracy
    def test_separation_derivatives(self, eccentric_orbit):
        # Order matches the low-level dc tensor: (tc, p, a, i, e, w, lan).
        # tc is the transit centre; coefficients do not depend on it, so
        # perturbing tc purely shifts the evaluation time (d/dtc with the
        # correct sign). A single near-expansion point sample avoids remapping across a
        # expansion point boundary.
        base = dict(te=0.0, tc=0.0, lan=0.0, **eccentric_orbit)
        keys = ["tc", "p", "a", "i", "e", "w", "lan"]
        times = np.array([0.015])
        eps = 1e-6

        k = Expansion2D(derivatives=True, **base)
        k.set_data(times)
        _, dd = k.projected_separation

        fd = np.zeros(7)
        for j, key in enumerate(keys):
            hi, lo = dict(base), dict(base)
            hi[key] += eps
            lo[key] -= eps
            khi, klo = Expansion2D(**hi), Expansion2D(**lo)
            khi.set_data(times)
            klo.set_data(times)
            fd[j] = (khi.projected_separation[0] - klo.projected_separation[0]) / (2 * eps)

        assert_allclose(dd[0], fd, rtol=1e-4, atol=1e-6)

    def test_property_matches_scalar_loop(self, eccentric_orbit):
        """Vectorized property equals looping the scalar low-level evaluators."""
        from meepmeep.backends.numba.point2dd import solve2d_d, sep_d, pos_d

        te, tc = 0.0, 0.0
        times = np.linspace(-0.02, 0.02, 9)
        c, dc = solve2d_d(te, **eccentric_orbit)

        k = Expansion2D(te=te, tc=tc, derivatives=True, **eccentric_orbit)
        k.set_data(times)

        # Separation property vs scalar sep_d loop.
        d, dd = k.projected_separation
        for n, t in enumerate(times):
            d_n, dd_n = sep_d(t - tc, te, eccentric_orbit["p"], c, dc)
            assert_allclose(d[n], d_n, rtol=1e-12)
            assert_allclose(dd[n], dd_n, rtol=1e-12)

        # Position property vs scalar pos_d loop.
        xs, ys, dxs, dys = k.position
        for n, t in enumerate(times):
            x_n, y_n, dx_n, dy_n = pos_d(t - tc, te, eccentric_orbit["p"], c, dc)
            assert_allclose(xs[n], x_n, rtol=1e-12)
            assert_allclose(ys[n], y_n, rtol=1e-12)
            assert_allclose(dxs[n], dx_n, rtol=1e-12)
            assert_allclose(dys[n], dy_n, rtol=1e-12)


class TestContactPointsAndDurations:
    """Contact-point and duration helpers."""

    def test_duration_ordering(self, circular_orbit):
        k = Expansion2D(te=0.0, tc=0.0, **circular_orbit)
        rr = 0.1
        assert k.duration(rr, 14) > k.duration(rr, 23) > 0.0

    def test_duration_default_is_t14(self, circular_orbit):
        k = Expansion2D(te=0.0, tc=0.0, **circular_orbit)
        rr = 0.1
        assert k.duration(rr) == k.duration(rr, 14)

    def test_duration_invalid_kind_raises(self, circular_orbit):
        k = Expansion2D(te=0.0, tc=0.0, **circular_orbit)
        with pytest.raises(ValueError):
            k.duration(0.1, 99)

    def test_ingress_egress_sum(self, circular_orbit):
        k = Expansion2D(te=0.0, tc=0.0, **circular_orbit)
        rr = 0.1
        # T12 + T23 + T34 should equal T14.
        total = k.duration(rr, 12) + k.duration(rr, 23) + k.duration(rr, 34)
        assert_allclose(total, k.duration(rr, 14), rtol=1e-4)

    def test_contacts_absolute_and_ordered(self, circular_orbit):
        tc = 100.0
        k = Expansion2D(te=0.0, tc=tc, **circular_orbit)
        rr = 0.1
        t1a = k.contact_point(rr, 1)
        t4a = k.contact_point(rr, 4)
        # Absolute times sit around the transit centre tc, and t1 < t4.
        assert t1a < tc < t4a
        bt1, bt4 = k.bounding_box(rr)
        assert_allclose((bt1, bt4), (t1a, t4a), rtol=1e-10)

    def test_min_separation_is_impact_parameter(self, circular_orbit):
        """For a central expansion point, the minimum separation is the impact parameter."""
        tc = 50.0
        k = Expansion2D(te=0.0, tc=tc, **circular_orbit)
        t_min, z_min = k.min_separation()
        b = circular_orbit["a"] * np.cos(circular_orbit["i"])
        assert_allclose(z_min, abs(b), rtol=1e-4, atol=1e-4)
        assert_allclose(t_min, tc, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
