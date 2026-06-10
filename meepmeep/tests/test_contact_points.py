"""Contact-point and transit-duration tests for the point2d/point3d util modules.

The sky-projected separation is invariant under a rotation of the sky plane
about the line of sight (the ``lan`` parameter), so every contact-point time
and transit duration must be invariant too. The ``lan = pi/2`` and
``lan = pi`` cases specifically guard the bracketing logic in
``find_contact_point``: a rotated transit chord makes the x-velocity at the
expansion point near zero (bracket blows up) or negative (bracket lands on
the wrong side of the transit in time), so the bracket must be derived from
the sky-plane speed, not the x-velocity alone.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

from meepmeep.backends.numba import point2d, point3d


K = 0.1
# Bisection terminates on a 1e-6 bracket; two independent runs can differ
# by twice that.
ATOL = 5e-6
# 0.3 is a generic rotation; pi/2 and 3pi/2 put the transit chord along the
# y-axis (vx ~ 0); pi flips the sign of vx.
LANS = [0.3, 0.5 * np.pi, np.pi, 1.5 * np.pi]

CASES = {
    "2d": (point2d.solve2d, point2d),
    "3d": (point3d.solve3d, point3d),
}


@pytest.fixture(params=["2d", "3d"])
def backend(request):
    """(solve, util module) pair for the 2D and 3D single-knot packages."""
    return CASES[request.param]


@pytest.fixture(params=["circular", "eccentric"])
def orbit_case(request, test_orbital_params):
    return test_orbital_params[request.param]


class TestLanRotationInvariance:
    @pytest.mark.parametrize("lan", LANS)
    @pytest.mark.parametrize("point", [1, 2, 3, 4])
    def test_contact_points(self, backend, orbit_case, lan, point):
        solve, util = backend
        c_ref = solve(0.0, **orbit_case, lan=0.0)
        c_rot = solve(0.0, **orbit_case, lan=lan)
        t_ref = util.find_contact_point(K, point, c_ref)
        t_rot = util.find_contact_point(K, point, c_rot)
        assert np.isfinite(t_rot)
        assert_allclose(t_rot, t_ref, atol=ATOL)

    @pytest.mark.parametrize("lan", LANS)
    def test_durations(self, backend, orbit_case, lan):
        solve, util = backend
        c_ref = solve(0.0, **orbit_case, lan=0.0)
        c_rot = solve(0.0, **orbit_case, lan=lan)
        assert_allclose(util.t14(K, c_rot), util.t14(K, c_ref), atol=ATOL)
        assert_allclose(util.t23(K, c_rot), util.t23(K, c_ref), atol=ATOL)


class TestContactPointGeometry:
    """Geometric sanity independent of the lan = 0 reference."""

    @pytest.mark.parametrize("lan", [0.0] + LANS)
    def test_separation_at_contact(self, backend, orbit_case, lan):
        """At the contact time the separation equals its target value."""
        solve, util = backend
        c = solve(0.0, **orbit_case, lan=lan)
        for point, target in ((1, 1.0 + K), (2, 1.0 - K), (3, 1.0 - K), (4, 1.0 + K)):
            t = util.find_contact_point(K, point, c)
            assert_allclose(util.sep_c(t, c), target, atol=1e-4)

    @pytest.mark.parametrize("lan", [0.0] + LANS)
    def test_contact_point_ordering(self, backend, orbit_case, lan):
        solve, util = backend
        c = solve(0.0, **orbit_case, lan=lan)
        ts = [util.find_contact_point(K, point, c) for point in (1, 2, 3, 4)]
        assert ts[0] < ts[1] < 0.0 < ts[2] < ts[3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
