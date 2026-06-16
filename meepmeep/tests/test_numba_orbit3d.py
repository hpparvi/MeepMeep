"""Test suite for meepmeep.backends.numba.orbit3d module.

Tests validate solve3d_orbit (Taylor coefficient computation at expansion points)
and ep_ix (time-to-expansion-point index mapping) against solve3d and Newton-Raphson
ground truth.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

from meepmeep.backends.numba.orbit3d import solve3d_orbit, ep_ix
from meepmeep.backends.numba.point3d import solve3d, pos_c
from meepmeep.backends.numba.expansion_points import create_expansion_points
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
def ep_grid_ea():
    """Create expansion points using eccentric anomaly placement with default npt."""
    ep_times, change_times, dt, ep_table = create_expansion_points(NPT, 0.2, "ea")
    return ep_times, change_times, dt, ep_table


class TestSolve3dOrbit:
    """Test the solve3d_orbit function."""

    def test_output_shape(self, circular_orbit, ep_grid_ea):
        ep_times = ep_grid_ea[0]
        coeffs = solve3d_orbit(ep_times, **circular_orbit, npt=NPT)
        assert coeffs.shape == (NPT, 3, 5)

    @pytest.mark.parametrize("npt", [5, 15, 25])
    def test_output_shape_various_npt(self, circular_orbit, npt):
        ep_times, _, _, _ = create_expansion_points(npt, 0.2, "ea")
        coeffs = solve3d_orbit(ep_times, **circular_orbit, npt=npt)
        assert coeffs.shape == (npt, 3, 5)

    def test_coefficients_finite_circular(self, circular_orbit, ep_grid_ea):
        coeffs = solve3d_orbit(ep_grid_ea[0], **circular_orbit, npt=NPT)
        assert np.all(np.isfinite(coeffs))

    def test_coefficients_finite_eccentric(self, eccentric_orbit, ep_grid_ea):
        coeffs = solve3d_orbit(ep_grid_ea[0], **eccentric_orbit, npt=NPT)
        assert np.all(np.isfinite(coeffs))

    def test_coefficients_finite_high_e(self, high_e_orbit):
        ep_times, _, _, _ = create_expansion_points(NPT, 0.7, "ea")
        coeffs = solve3d_orbit(ep_times, **high_e_orbit, npt=NPT)
        assert np.all(np.isfinite(coeffs))

    def test_periodic_boundary(self, circular_orbit, ep_grid_ea):
        """Last row should equal first row (periodic orbit)."""
        coeffs = solve3d_orbit(ep_grid_ea[0], **circular_orbit, npt=NPT)
        assert_allclose(coeffs[-1], coeffs[0])

    def test_matches_solve3d(self, eccentric_orbit, ep_grid_ea):
        """Each row should match solve3d called with the same phase, raveled."""
        ep_times = ep_grid_ea[0]
        coeffs = solve3d_orbit(ep_times, **eccentric_orbit, npt=NPT)

        p = eccentric_orbit["p"]
        e = eccentric_orbit["e"]
        w = eccentric_orbit["w"]
        t_offset = mean_anomaly_at_transit(e, w) / TWO_PI * p

        for ix in range(NPT - 1):
            phase = p * ep_times[ix] - t_offset
            expected = solve3d(phase, **eccentric_orbit)
            assert_allclose(coeffs[ix], expected, rtol=1e-12,
                            err_msg=f"Mismatch at expansion point {ix}")

    def test_accuracy_vs_newton(self, eccentric_orbit, ep_grid_ea):
        """Taylor series evaluated from stored coefficients should match Newton ground truth."""
        ep_times = ep_grid_ea[0]
        coeffs = solve3d_orbit(ep_times, **eccentric_orbit, npt=NPT)

        p = eccentric_orbit["p"]
        e = eccentric_orbit["e"]
        w = eccentric_orbit["w"]
        t_offset = mean_anomaly_at_transit(e, w) / TWO_PI * p
        t0 = -t_offset

        for ix in range(NPT - 1):
            c = coeffs[ix].reshape(3, 5)
            # Evaluate at small offsets from the expansion point expansion point
            for dt in [-0.01, 0.0, 0.01]:
                t_ep = t0 + p * ep_times[ix]
                t_eval = t_ep + dt

                x_ts, y_ts, z_ts = pos_c(dt, c)
                x_nr, y_nr, z_nr = xyz_newton_v(
                    np.array([t_eval]), 0.0, p,
                    eccentric_orbit["a"], eccentric_orbit["i"], e, w
                )

                assert_allclose(x_ts, x_nr[0], rtol=1e-5, atol=1e-8,
                                err_msg=f"X mismatch at expansion point {ix}, dt={dt}")
                assert_allclose(y_ts, y_nr[0], rtol=1e-5, atol=1e-8,
                                err_msg=f"Y mismatch at expansion point {ix}, dt={dt}")
                assert_allclose(z_ts, z_nr[0], rtol=1e-5, atol=1e-8,
                                err_msg=f"Z mismatch at expansion point {ix}, dt={dt}")


class TestEpIx:
    """Test the ep_ix function."""

    def test_returns_valid_index(self, ep_grid_ea):
        ep_times, _, dt, ep_table = ep_grid_ea
        t0 = 0.0
        p = 3.0
        times = np.linspace(t0 + 0.001, t0 + p - 0.001, 100)
        for t in times:
            ix = ep_ix(t, t0, p, dt, ep_table)
            assert 0 <= ix < NPT, f"Index {ix} out of range for t={t}"

    def test_epoch_folding(self, ep_grid_ea):
        """Times offset by full periods should return the same expansion-point index."""
        _, _, dt, ep_table = ep_grid_ea
        t0 = 0.0
        p = 3.0
        t_base = t0 + 0.5 * p
        ix_base = ep_ix(t_base, t0, p, dt, ep_table)
        for n_epoch in [1, 2, 5, -1, -3]:
            ix = ep_ix(t_base + n_epoch * p, t0, p, dt, ep_table)
            assert ix == ix_base, f"Epoch folding failed for n={n_epoch}"

    def test_monotonic_within_orbit(self, ep_grid_ea):
        """Expansion point indices should be non-decreasing within one period."""
        _, _, dt, ep_table = ep_grid_ea
        t0 = 0.0
        p = 3.0
        times = np.linspace(t0 + 0.001, t0 + p - 0.001, 200)
        indices = [ep_ix(t, t0, p, dt, ep_table) for t in times]
        for j in range(1, len(indices)):
            assert indices[j] >= indices[j - 1], \
                f"Non-monotonic at t={times[j]}: {indices[j-1]} -> {indices[j]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
