"""Tests for the expansion point-placement strategies in ``backends.numba.expansion_points``.

The multi-expansion-point machinery (``solve3d_orbit`` and the orbit3d/orbit3dd
dispatchers) relies on a structural contract shared by every placement
strategy:

- ``ep_times`` has ``n_ep`` entries; the first is 0.0 and the last is
  1.0, the periodic image of the first (``solve3d_orbit`` copies the first
  expansion point's coefficients into the last slot instead of recomputing them).
- ``change_times`` holds the ``n_ep - 1`` boundaries between adjacent
  expansion points' regions of validity, each lying strictly between its two expansion points.
- ``ep_table`` maps every folded-time bin to the expansion point whose region contains
  it.

These tests pin that contract for all three strategies and add an
end-to-end Taylor-vs-Newton accuracy check on an ``'mm'`` grid, which
guards against a grid that is internally inconsistent but structurally
plausible.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from meepmeep.backends.numba.expansion_points import create_expansion_points, expansion_table_size
from meepmeep.backends.numba.newton.newton import xyz_newton_v
from meepmeep.backends.numba.orbit3d import solve3d_orbit, pos_o
from meepmeep.backends.numba.utils import TWO_PI, mean_anomaly_at_transit


STRATEGIES = ["mm", "ea", "ta"]


@pytest.fixture(params=STRATEGIES)
def strategy(request):
    return request.param


@pytest.fixture(params=[5, 15])
def n_ep(request):
    return request.param


class TestEpGridContract:
    """Structural contract shared by every placement strategy."""

    def test_ep_count(self, strategy, n_ep):
        ep_times, _, _, _ = create_expansion_points(n_ep, 0.3, strategy)
        assert len(ep_times) == n_ep

    def test_periodic_image(self, strategy, n_ep):
        """First expansion point at 0, last entry is its periodic image at exactly 1."""
        ep_times, _, _, _ = create_expansion_points(n_ep, 0.3, strategy)
        assert ep_times[0] == 0.0
        assert ep_times[-1] == 1.0

    def test_eps_strictly_increasing(self, strategy, n_ep):
        ep_times, _, _, _ = create_expansion_points(n_ep, 0.3, strategy)
        assert np.all(np.diff(ep_times) > 0.0)

    def test_midpoint_ep(self, strategy, n_ep):
        """One expansion point lands exactly at the orbit midpoint (n_ep is odd)."""
        ep_times, _, _, _ = create_expansion_points(n_ep, 0.3, strategy)
        assert ep_times[n_ep // 2] == pytest.approx(0.5)

    def test_change_times_bracket_knots(self, strategy, n_ep):
        """Each region boundary lies strictly between its two expansion points."""
        ep_times, change_times, _, _ = create_expansion_points(n_ep, 0.3, strategy)
        assert len(change_times) == n_ep - 1
        assert np.all(ep_times[:-1] < change_times)
        assert np.all(change_times < ep_times[1:])




class TestTimeTable:
    """The time-to-expansion-point table.

    Each bin maps to the expansion point whose region contains the bin's centre,
    and the default resolution keeps the bins much narrower than the narrowest
    region. A coarse table evaluates part of a bin from the neighbouring
    expansion point, which near periastron at high eccentricity is far away on
    the local orbital timescale (0.14 R_star error at e = 0.9 with 200 bins).
    """

    @pytest.mark.parametrize("tres", [None, 200])
    @pytest.mark.parametrize("e", [0.3, 0.7, 0.9])
    @pytest.mark.parametrize("n", [5, 15, 35])
    def test_each_bin_maps_to_the_region_of_its_centre(self, strategy, n, e, tres):
        ep_times, change_times, dt, ep_table = create_expansion_points(n, e, strategy, tres)
        boundaries = np.concatenate(([0.0], change_times, [1.0]))
        centres = (np.arange(ep_table.size) + 0.5) * dt
        assert np.all((0 <= ep_table) & (ep_table <= n - 1))
        inside = (boundaries[ep_table] <= centres) & (centres <= boundaries[ep_table + 1])
        assert inside.all(), f"bins {np.flatnonzero(~inside)[:5]} map outside their centre's region"

    @pytest.mark.parametrize("e", [0.0, 0.5, 0.9, 0.97])
    @pytest.mark.parametrize("n", [15, 35])
    def test_default_resolution_resolves_the_narrowest_region(self, strategy, n, e):
        _, change_times, dt, ep_table = create_expansion_points(n, e, strategy)
        widths = np.diff(np.concatenate(([0.0], change_times, [1.0])))
        assert dt <= widths.min() / 8
        assert ep_table.size >= 200
        assert dt == 1.0 / ep_table.size

    @pytest.mark.parametrize("n", [15, 35])
    def test_default_resolution_does_not_depend_on_e_below_0_9(self, strategy, n):
        """Tables are sized for max(e, 0.9), so the JAX backend (static table shape)
        can match numba bin for bin whether or not e is traced."""
        sizes = {create_expansion_points(n, e, strategy)[3].size for e in (0.0, 0.3, 0.6, 0.9)}
        assert len(sizes) == 1

    def test_size_is_capped(self):
        assert expansion_table_size(35, 0.9999, "ta") == 2 ** 20

    def test_explicit_resolution_is_honoured(self, strategy):
        _, _, dt, ep_table = create_expansion_points(15, 0.3, strategy, 200)
        assert ep_table.size == 200 and dt == 1.0 / 200


class TestHighEccentricityAccuracy:
    """Whole-orbit accuracy keeps improving with the expansion-point count at high e.

    With a table that cannot resolve the expansion-point regions near periastron,
    the error stalled at ~0.1 R_star for e = 0.9 whatever the count.
    """

    P, A, I, W = 3.0, 8.5, np.radians(88.0), np.radians(60.0)

    def _max_error(self, e, npt):
        ep_times, _, dt, ep_table = create_expansion_points(npt, e, "ea")
        coeffs = solve3d_orbit(ep_times, self.P, self.A, self.I, e, self.W, npt=npt)
        tpa = -mean_anomaly_at_transit(e, self.W) / TWO_PI * self.P
        # Several epochs, densely sampled so the periastron passages are resolved.
        t = np.linspace(-2.0 * self.P, 3.0 * self.P, 250_001)
        x, y, z = pos_o(t, tpa, self.P, dt, ep_table, ep_times, coeffs)
        rx, ry, rz = xyz_newton_v(t, 0.0, self.P, self.A, self.I, e, self.W)
        return np.max(np.sqrt((x - rx) ** 2 + (y - ry) ** 2 + (z - rz) ** 2))

    @pytest.mark.parametrize("e, npt, bound", [(0.8, 35, 3e-4), (0.9, 35, 1e-3)])
    def test_error_bound(self, e, npt, bound):
        assert self._max_error(e, npt) < bound

    def test_error_converges_with_npt_at_e_0_9(self):
        assert self._max_error(0.9, 35) < self._max_error(0.9, 15) / 20

class TestMeanMotionPlacement:
    """Geometry specific to the 'mm' (uniform in time) strategy."""

    def test_uniform_spacing(self, n_ep):
        ep_times, _, _, _ = create_expansion_points(n_ep, 0.3, "mm")
        assert_allclose(np.diff(ep_times), 1.0 / (n_ep - 1))

    def test_change_times_at_midpoints(self, n_ep):
        ep_times, change_times, _, _ = create_expansion_points(n_ep, 0.3, "mm")
        assert_allclose(change_times, 0.5 * (ep_times[:-1] + ep_times[1:]))


@pytest.mark.accuracy
class TestMeanMotionEndToEnd:
    """A full solve3d_orbit -> pos_o pass on an 'mm' grid vs Newton-Raphson.

    This catches grid bugs the structural tests can't, e.g. a real expansion point
    being overwritten by the periodic-image copy in ``solve3d_orbit``.
    """

    @pytest.mark.parametrize("case", ["circular", "eccentric"])
    def test_xyz_matches_newton(self, case, test_orbital_params):
        pars = test_orbital_params[case]
        npt = 15
        p, e, w = pars["p"], pars["e"], pars["w"]

        ep_times, _, dt, ep_table = create_expansion_points(npt, e, "mm")
        coeffs = solve3d_orbit(ep_times, **pars, npt=npt)
        tpa = -mean_anomaly_at_transit(e, w) / TWO_PI * p

        times = np.linspace(0.0, p, 200)
        x, y, z = pos_o(times, tpa, p, dt, ep_table, ep_times, coeffs)
        xn, yn, zn = xyz_newton_v(times, 0.0, **pars)
        # Looser than the 'ea' comparisons in test_orbit3d_evaluators.py:
        # uniform-in-time expansion points don't cluster near periastron, so the
        # truncation error for e = 0.3 reaches a few 1e-3. The grid bugs
        # this test guards against produce O(1)-O(10) errors.
        assert_allclose(x, xn, rtol=1e-2, atol=1e-2)
        assert_allclose(y, yn, rtol=1e-2, atol=1e-2)
        assert_allclose(z, zn, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
