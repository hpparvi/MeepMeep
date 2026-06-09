"""Tests for the knot-placement strategies in ``backends.numba.knots``.

The multi-knot machinery (``solve3d_orbit`` and the orbit3d/orbit3dd
dispatchers) relies on a structural contract shared by every placement
strategy:

- ``knot_times`` has ``n_knots`` entries; the first is 0.0 and the last is
  1.0, the periodic image of the first (``solve3d_orbit`` copies the first
  knot's coefficients into the last slot instead of recomputing them).
- ``change_times`` holds the ``n_knots - 1`` boundaries between adjacent
  knots' regions of validity, each lying strictly between its two knots.
- ``tktable`` maps every folded-time bin to the knot whose region contains
  it.

These tests pin that contract for all three strategies and add an
end-to-end Taylor-vs-Newton accuracy check on an ``'mm'`` grid, which
guards against a grid that is internally inconsistent but structurally
plausible.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from meepmeep.backends.numba.knots import create_knots
from meepmeep.backends.numba.newton.newton import xyz_newton_v
from meepmeep.backends.numba.orbit3d import solve3d_orbit, pos_o
from meepmeep.backends.numba.utils import TWO_PI, mean_anomaly_at_transit


STRATEGIES = ["mm", "ea", "ta"]


@pytest.fixture(params=STRATEGIES)
def strategy(request):
    return request.param


@pytest.fixture(params=[5, 15])
def n_knots(request):
    return request.param


class TestKnotGridContract:
    """Structural contract shared by every placement strategy."""

    def test_knot_count(self, strategy, n_knots):
        knot_times, _, _, _ = create_knots(n_knots, 0.3, strategy)
        assert len(knot_times) == n_knots

    def test_periodic_image(self, strategy, n_knots):
        """First knot at 0, last entry is its periodic image at exactly 1."""
        knot_times, _, _, _ = create_knots(n_knots, 0.3, strategy)
        assert knot_times[0] == 0.0
        assert knot_times[-1] == 1.0

    def test_knots_strictly_increasing(self, strategy, n_knots):
        knot_times, _, _, _ = create_knots(n_knots, 0.3, strategy)
        assert np.all(np.diff(knot_times) > 0.0)

    def test_midpoint_knot(self, strategy, n_knots):
        """One knot lands exactly at the orbit midpoint (n_knots is odd)."""
        knot_times, _, _, _ = create_knots(n_knots, 0.3, strategy)
        assert knot_times[n_knots // 2] == pytest.approx(0.5)

    def test_change_times_bracket_knots(self, strategy, n_knots):
        """Each region boundary lies strictly between its two knots."""
        knot_times, change_times, _, _ = create_knots(n_knots, 0.3, strategy)
        assert len(change_times) == n_knots - 1
        assert np.all(knot_times[:-1] < change_times)
        assert np.all(change_times < knot_times[1:])

    def test_tktable_consistent_with_change_times(self, strategy, n_knots):
        """Every folded-time bin maps to the knot whose region contains it."""
        knot_times, change_times, dt, tktable = create_knots(n_knots, 0.3, strategy)
        boundaries = np.concatenate(([0.0], change_times, [1.0]))
        for i, ik in enumerate(tktable):
            assert 0 <= ik <= n_knots - 1
            t = i * dt
            assert boundaries[ik] <= t <= boundaries[ik + 1], (
                f"Bin {i} (t={t:.4f}) assigned to knot {ik} with region "
                f"[{boundaries[ik]:.4f}, {boundaries[ik + 1]:.4f}]"
            )


class TestMeanMotionPlacement:
    """Geometry specific to the 'mm' (uniform in time) strategy."""

    def test_uniform_spacing(self, n_knots):
        knot_times, _, _, _ = create_knots(n_knots, 0.3, "mm")
        assert_allclose(np.diff(knot_times), 1.0 / (n_knots - 1))

    def test_change_times_at_midpoints(self, n_knots):
        knot_times, change_times, _, _ = create_knots(n_knots, 0.3, "mm")
        assert_allclose(change_times, 0.5 * (knot_times[:-1] + knot_times[1:]))


@pytest.mark.accuracy
class TestMeanMotionEndToEnd:
    """A full solve3d_orbit -> pos_o pass on an 'mm' grid vs Newton-Raphson.

    This catches grid bugs the structural tests can't, e.g. a real knot
    being overwritten by the periodic-image copy in ``solve3d_orbit``.
    """

    @pytest.mark.parametrize("case", ["circular", "eccentric"])
    def test_xyz_matches_newton(self, case, test_orbital_params):
        pars = test_orbital_params[case]
        npt = 15
        p, e, w = pars["p"], pars["e"], pars["w"]

        knot_times, _, dt, pktable = create_knots(npt, e, "mm")
        coeffs = solve3d_orbit(knot_times, **pars, npt=npt)
        tpa = -mean_anomaly_at_transit(e, w) / TWO_PI * p

        times = np.linspace(0.0, p, 200)
        x, y, z = pos_o(times, tpa, p, dt, pktable, knot_times, coeffs)
        xn, yn, zn = xyz_newton_v(times, 0.0, **pars)
        # Looser than the 'ea' comparisons in test_orbit3d_evaluators.py:
        # uniform-in-time knots don't cluster near periastron, so the
        # truncation error for e = 0.3 reaches a few 1e-3. The grid bugs
        # this test guards against produce O(1)-O(10) errors.
        assert_allclose(x, xn, rtol=1e-2, atol=1e-2)
        assert_allclose(y, yn, rtol=1e-2, atol=1e-2)
        assert_allclose(z, zn, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
