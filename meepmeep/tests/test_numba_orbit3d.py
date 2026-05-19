"""Test suite for meepmeep.backends.numba.taylor.orbit3d module.

Tests validate solve3d_orbit (Taylor coefficient computation at knot points)
and knot_ix (time-to-knot index mapping) against solve3d and Newton-Raphson
ground truth.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

from meepmeep.backends.numba.taylor.orbit3d import solve3d_orbit, knot_ix
from meepmeep.backends.numba.taylor.solve3d import solve3d
from meepmeep.backends.numba.taylor.position3d import pos_c
from meepmeep.backends.numba.knots import create_knots
from meepmeep.backends.numba.newton.newton import xyz_newton_v
from meepmeep.backends.numba.utils import mean_anomaly_at_transit, TWO_PI


NPT = 15


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
def knots_ea():
    """Create knots using eccentric anomaly placement with default npt."""
    knot_times, change_times, dt, pktable = create_knots(NPT, 0.2, "ea")
    return knot_times, change_times, dt, pktable


class TestSolve3dOrbit:
    """Test the solve3d_orbit function."""

    def test_output_shape(self, circular_orbit, knots_ea):
        knot_times = knots_ea[0]
        coeffs = solve3d_orbit(knot_times, **circular_orbit, npt=NPT)
        assert coeffs.shape == (NPT, 3, 5)

    @pytest.mark.parametrize("npt", [5, 15, 25])
    def test_output_shape_various_npt(self, circular_orbit, npt):
        knot_times, _, _, _ = create_knots(npt, 0.2, "ea")
        coeffs = solve3d_orbit(knot_times, **circular_orbit, npt=npt)
        assert coeffs.shape == (npt, 3, 5)

    def test_coefficients_finite_circular(self, circular_orbit, knots_ea):
        coeffs = solve3d_orbit(knots_ea[0], **circular_orbit, npt=NPT)
        assert np.all(np.isfinite(coeffs))

    def test_coefficients_finite_eccentric(self, eccentric_orbit, knots_ea):
        coeffs = solve3d_orbit(knots_ea[0], **eccentric_orbit, npt=NPT)
        assert np.all(np.isfinite(coeffs))

    def test_coefficients_finite_high_e(self, high_e_orbit):
        knot_times, _, _, _ = create_knots(NPT, 0.7, "ea")
        coeffs = solve3d_orbit(knot_times, **high_e_orbit, npt=NPT)
        assert np.all(np.isfinite(coeffs))

    def test_periodic_boundary(self, circular_orbit, knots_ea):
        """Last row should equal first row (periodic orbit)."""
        coeffs = solve3d_orbit(knots_ea[0], **circular_orbit, npt=NPT)
        assert_allclose(coeffs[-1], coeffs[0])

    def test_matches_solve3d(self, eccentric_orbit, knots_ea):
        """Each row should match solve3d called with the same phase, raveled."""
        knot_times = knots_ea[0]
        coeffs = solve3d_orbit(knot_times, **eccentric_orbit, npt=NPT)

        p = eccentric_orbit["p"]
        e = eccentric_orbit["e"]
        w = eccentric_orbit["w"]
        t_offset = mean_anomaly_at_transit(e, w) / TWO_PI * p

        for ix in range(NPT - 1):
            phase = p * knot_times[ix] - t_offset
            expected = solve3d(phase, **eccentric_orbit)
            assert_allclose(coeffs[ix], expected, rtol=1e-12,
                            err_msg=f"Mismatch at knot {ix}")

    def test_accuracy_vs_newton(self, eccentric_orbit, knots_ea):
        """Taylor series evaluated from stored coefficients should match Newton ground truth."""
        knot_times = knots_ea[0]
        coeffs = solve3d_orbit(knot_times, **eccentric_orbit, npt=NPT)

        p = eccentric_orbit["p"]
        e = eccentric_orbit["e"]
        w = eccentric_orbit["w"]
        t_offset = mean_anomaly_at_transit(e, w) / TWO_PI * p
        t0 = -t_offset

        for ix in range(NPT - 1):
            c = coeffs[ix].reshape(3, 5)
            # Evaluate at small offsets from the knot expansion point
            for dt in [-0.01, 0.0, 0.01]:
                t_knot = t0 + p * knot_times[ix]
                t_eval = t_knot + dt

                x_ts, y_ts, z_ts = pos_c(dt, c)
                x_nr, y_nr, z_nr = xyz_newton_v(
                    np.array([t_eval]), 0.0, p,
                    eccentric_orbit["a"], eccentric_orbit["i"], e, w
                )

                assert_allclose(x_ts, x_nr[0], rtol=1e-5, atol=1e-8,
                                err_msg=f"X mismatch at knot {ix}, dt={dt}")
                assert_allclose(y_ts, y_nr[0], rtol=1e-5, atol=1e-8,
                                err_msg=f"Y mismatch at knot {ix}, dt={dt}")
                assert_allclose(z_ts, z_nr[0], rtol=1e-5, atol=1e-8,
                                err_msg=f"Z mismatch at knot {ix}, dt={dt}")


class TestKnotIx:
    """Test the knot_ix function."""

    def test_returns_valid_index(self, knots_ea):
        knot_times, _, dt, pktable = knots_ea
        t0 = 0.0
        p = 3.0
        times = np.linspace(t0 + 0.001, t0 + p - 0.001, 100)
        for t in times:
            ix = knot_ix(t, t0, p, dt, pktable)
            assert 0 <= ix < NPT, f"Index {ix} out of range for t={t}"

    def test_epoch_folding(self, knots_ea):
        """Times offset by full periods should return the same knot index."""
        _, _, dt, pktable = knots_ea
        t0 = 0.0
        p = 3.0
        t_base = t0 + 0.5 * p
        ix_base = knot_ix(t_base, t0, p, dt, pktable)
        for n_epoch in [1, 2, 5, -1, -3]:
            ix = knot_ix(t_base + n_epoch * p, t0, p, dt, pktable)
            assert ix == ix_base, f"Epoch folding failed for n={n_epoch}"

    def test_monotonic_within_orbit(self, knots_ea):
        """Knot indices should be non-decreasing within one period."""
        _, _, dt, pktable = knots_ea
        t0 = 0.0
        p = 3.0
        times = np.linspace(t0 + 0.001, t0 + p - 0.001, 200)
        indices = [knot_ix(t, t0, p, dt, pktable) for t in times]
        for j in range(1, len(indices)):
            assert indices[j] >= indices[j - 1], \
                f"Non-monotonic at t={times[j]}: {indices[j-1]} -> {indices[j]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
