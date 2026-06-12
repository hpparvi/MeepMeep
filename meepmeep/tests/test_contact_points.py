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

    def test_limb_crossing_point(self, backend, orbit_case):
        """Contact "point" 12 is the ingress limb crossing (separation = 1).

        It falls between the first and second contacts, where the planet
        center crosses the stellar limb.
        """
        solve, util = backend
        c = solve(0.0, **orbit_case)
        t_limb = util.find_contact_point(K, 12, c)
        t1 = util.find_contact_point(K, 1, c)
        t2 = util.find_contact_point(K, 2, c)
        assert t1 < t_limb < t2
        assert_allclose(util.sep_c(t_limb, c), 1.0, atol=1e-4)


class TestDurationHelpers:
    """The duration helpers are thin compositions of ``find_contact_point``.

    ``find_contact_point`` itself is validated geometrically above, so these
    tests pin the composition: which contact each helper uses and the sign of
    each difference. Identical inputs make the underlying bisection runs
    deterministic, hence the near-exact tolerances.
    """

    def test_single_contact_helpers(self, backend, orbit_case):
        solve, util = backend
        c = solve(0.0, **orbit_case)
        assert_allclose(util.t1(K, c), util.find_contact_point(K, 1, c), atol=1e-12)
        assert_allclose(util.t4(K, c), util.find_contact_point(K, 4, c), atol=1e-12)

    def test_bounding_box(self, backend, orbit_case):
        solve, util = backend
        c = solve(0.0, **orbit_case)
        tb1, tb4 = util.bounding_box(K, c)
        assert_allclose(tb1, util.find_contact_point(K, 1, c), atol=1e-12)
        assert_allclose(tb4, util.find_contact_point(K, 4, c), atol=1e-12)
        assert tb1 < 0.0 < tb4

    def test_partial_durations(self, backend, orbit_case):
        solve, util = backend
        c = solve(0.0, **orbit_case)
        ts = {point: util.find_contact_point(K, point, c) for point in (1, 2, 3, 4)}
        assert_allclose(util.t12(K, c), ts[2] - ts[1], atol=1e-12)
        assert_allclose(util.t34(K, c), ts[4] - ts[3], atol=1e-12)
        assert util.t12(K, c) > 0.0
        assert util.t34(K, c) > 0.0

    def test_duration_composition(self, backend, orbit_case):
        """T14 = T12 + T23 + T34: the partial durations tile the transit."""
        solve, util = backend
        c = solve(0.0, **orbit_case)
        total = util.t12(K, c) + util.t23(K, c) + util.t34(K, c)
        assert_allclose(util.t14(K, c), total, atol=1e-12)


class TestFindZMin:
    @pytest.mark.parametrize("lan", [0.0, 0.3])
    def test_minimum_beats_grid_scan(self, backend, orbit_case, lan):
        """The golden-section minimum is at least as deep as a dense grid scan.

        The search window is the fixed ``tc +/- 0.01`` interval used by
        ``find_z_min``; the returned value must be consistent with ``sep_c``
        at the returned time.
        """
        solve, util = backend
        c = solve(0.0, **orbit_case, lan=lan)
        t_min, z_min = util.find_z_min(0.0, c)
        assert abs(t_min) < 0.01
        assert_allclose(util.sep_c(t_min, c), z_min, atol=1e-12)
        tg = np.linspace(-0.01, 0.01, 2001)
        assert z_min <= util.sep_c(tg, c).min() + 1e-10

    def test_circular_impact_parameter(self, backend, test_orbital_params):
        """For a circular orbit the minimum separation is exactly a*cos(i)."""
        solve, util = backend
        pars = test_orbital_params["circular"]
        c = solve(0.0, **pars)
        _, z_min = util.find_z_min(0.0, c)
        assert_allclose(z_min, pars["a"] * np.cos(pars["i"]), rtol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
