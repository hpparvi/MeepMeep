"""Tests for the eccentricity-adaptive expansion-point grid in the Orbit class.

The ``'ea'`` / ``'ta'`` expansion point-placement strategies cluster expansion points near
periastron, which only helps if the grid is built for (roughly) the
eccentricity actually bound via ``set_pars``. The grid is therefore
rebuilt inside ``set_pars`` whenever the bound eccentricity drifts more
than ``Orbit._EP_GRID_E_TOL`` from the eccentricity the current grid
was built for. Rebuilds are deliberately rare: ``create_expansion_points`` runs
scipy root solves in Python, and ``set_pars`` is the per-likelihood-call
hot path in fitting applications.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from meepmeep.backends.numba.expansion_points import create_expansion_points
from meepmeep.backends.numba.newton.newton import xyz_newton_v
from meepmeep.orbit import Orbit


HIGH_E_PARS = {"p": 7.0, "a": 20.0, "i": 1.4, "e": 0.7, "w": 1.2}


class TestEpGridAdaptation:
    def test_grid_adapts_to_high_e(self):
        """After binding a high eccentricity, the expansion-point grid must match the
        one create_expansion_points builds for that eccentricity."""
        o = Orbit(npt=15)
        o.set_pars(tc=0.0, **HIGH_E_PARS)
        expected, _, _, _ = create_expansion_points(15, HIGH_E_PARS["e"], "ea")
        assert_allclose(o._ep_times, expected)

    def test_grid_floor_at_low_e(self):
        """Below the floor eccentricity the grid stays at the floor: a
        near-circular grid is essentially uniform anyway and the floor
        avoids rebuilding for every small-e fit."""
        o = Orbit(npt=15)
        pts_init = o._ep_times
        o.set_pars(tc=0.0, p=3.0, a=10.0, i=1.5, e=0.0, w=0.0)
        assert o._ep_times is pts_init

    def test_no_rebuild_for_small_e_change(self):
        """Eccentricity jitter below the tolerance (the MCMC case) must not
        trigger a rebuild."""
        o = Orbit(npt=15)
        o.set_pars(tc=0.0, p=5.0, a=15.0, i=1.55, e=0.30, w=0.5)
        pts = o._ep_times
        o.set_pars(tc=0.0, p=5.0, a=15.0, i=1.55, e=0.32, w=0.5)
        assert o._ep_times is pts

    def test_rebuild_beyond_tolerance(self):
        """A change beyond the tolerance rebuilds the grid for the new e."""
        o = Orbit(npt=15)
        o.set_pars(tc=0.0, p=5.0, a=15.0, i=1.55, e=0.30, w=0.5)
        o.set_pars(tc=0.0, p=5.0, a=15.0, i=1.55, e=0.50, w=0.5)
        expected, _, _, _ = create_expansion_points(15, 0.50, "ea")
        assert_allclose(o._ep_times, expected)

    def test_mm_grid_never_rebuilt(self):
        """The 'mm' grid is eccentricity-independent, so binding any e must
        keep the construction-time grid object."""
        o = Orbit(npt=15, ep_placement="mm")
        pts_init = o._ep_times
        o.set_pars(tc=0.0, **HIGH_E_PARS)
        assert o._ep_times is pts_init


@pytest.mark.accuracy
class TestHighEccentricityAccuracy:
    def test_xyz_accuracy_high_e(self):
        """End-to-end: with the adapted grid the high-e position error stays
        in the documented accuracy regime. With the construction-time
        e = 0.2 grid the error is ~0.7 stellar radii."""
        o = Orbit(npt=15)
        o.set_pars(tc=0.0, **HIGH_E_PARS)
        times = np.linspace(0.0, HIGH_E_PARS["p"], 500)
        o.set_data(times)
        x, y, z = o.xyz()
        xn, yn, zn = xyz_newton_v(times, 0.0, **HIGH_E_PARS)
        err = np.sqrt((x - xn) ** 2 + (y - yn) ** 2 + (z - zn) ** 2)
        assert np.max(err) < 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
