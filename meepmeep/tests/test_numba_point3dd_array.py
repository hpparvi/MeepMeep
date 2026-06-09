"""Scalar-or-array dispatch tests for the meepmeep.backends.numba.point3dd evaluators.

Each single-knot 3D derivative evaluator (pos_cd/pos_d, sep_cd/sep_d,
zpos_cd/zpos_d, vel_cd, zvel_cd/zvel_d, rv_cd/rv_d) now accepts EITHER a scalar
time OR a 1-D array of times via numba @overload, mirroring the 2D point2dd
evaluators. This suite checks that the array path equals the per-element scalar
loop and returns the (N,) / (N, 7) layout, and that the same function name still
serves a scalar time with a (7,) gradient.
"""
import pytest
import numpy as np
from numpy.testing import assert_allclose

from meepmeep.backends.numba.point3dd import (
    solve3d_d,
    pos_cd, pos_d, sep_cd, sep_d, zpos_cd, zpos_d,
    vel_cd, zvel_cd, zvel_d, rv_cd, rv_d,
)


@pytest.fixture
def eccentric_orbit():
    return {"p": 5.0, "a": 15.0, "i": 1.55, "e": 0.3, "w": 0.5}


@pytest.fixture
def setup(eccentric_orbit):
    """(c, dc, times, p) for an eccentric orbit with a non-zero lan."""
    c, dc = solve3d_d(0.0, **eccentric_orbit, lan=0.7)
    times = np.linspace(-0.02, 0.02, 9)
    return c, dc, times, eccentric_orbit["p"]


# --- helpers ---------------------------------------------------------------

def _check_vector_quantities(values, n):
    """Each value array is (n,)."""
    for v in values:
        assert v.shape == (n,)


def _check_gradients(grads, n):
    """Each gradient array is (n, 7)."""
    for g in grads:
        assert g.shape == (n, 7)


# --- position --------------------------------------------------------------

class TestPos:
    def test_pos_cd_array_matches_scalar_loop(self, setup):
        c, dc, times, _ = setup
        xs, ys, zs, dxs, dys, dzs = pos_cd(times, c, dc)
        _check_vector_quantities((xs, ys, zs), times.size)
        _check_gradients((dxs, dys, dzs), times.size)
        for n, t in enumerate(times):
            x, y, z, dx, dy, dz = pos_cd(t, c, dc)
            assert_allclose([xs[n], ys[n], zs[n]], [x, y, z], rtol=1e-12)
            assert_allclose(dxs[n], dx, rtol=1e-12)
            assert_allclose(dys[n], dy, rtol=1e-12)
            assert_allclose(dzs[n], dz, rtol=1e-12)

    def test_pos_d_array_matches_scalar_loop(self, setup):
        c, dc, times, p = setup
        xs, ys, zs, dxs, dys, dzs = pos_d(times, 0.0, p, c, dc)
        _check_vector_quantities((xs, ys, zs), times.size)
        _check_gradients((dxs, dys, dzs), times.size)
        for n, t in enumerate(times):
            x, y, z, dx, dy, dz = pos_d(t, 0.0, p, c, dc)
            assert_allclose([xs[n], ys[n], zs[n]], [x, y, z], rtol=1e-12)
            assert_allclose(dxs[n], dx, rtol=1e-12)

    def test_pos_cd_scalar_gradient_shape(self, setup):
        c, dc, _, _ = setup
        x, y, z, dx, dy, dz = pos_cd(0.003, c, dc)
        assert np.isscalar(x) and np.isscalar(y) and np.isscalar(z)
        assert dx.shape == dy.shape == dz.shape == (7,)


# --- separation ------------------------------------------------------------

class TestSep:
    def test_sep_cd_array_matches_scalar_loop(self, setup):
        c, dc, times, _ = setup
        d, dd = sep_cd(times, c, dc)
        assert d.shape == (times.size,)
        assert dd.shape == (times.size, 7)
        for n, t in enumerate(times):
            d_n, dd_n = sep_cd(t, c, dc)
            assert_allclose(d[n], d_n, rtol=1e-12)
            assert_allclose(dd[n], dd_n, rtol=1e-12)

    def test_sep_d_array_matches_scalar_loop(self, setup):
        c, dc, times, p = setup
        d, dd = sep_d(times, 0.0, p, c, dc)
        assert d.shape == (times.size,)
        assert dd.shape == (times.size, 7)
        for n, t in enumerate(times):
            d_n, dd_n = sep_d(t, 0.0, p, c, dc)
            assert_allclose(d[n], d_n, rtol=1e-12)
            assert_allclose(dd[n], dd_n, rtol=1e-12)

    def test_sep_d_scalar_gradient_shape(self, setup):
        c, dc, _, p = setup
        d, dd = sep_d(0.003, 0.0, p, c, dc)
        assert np.isscalar(d)
        assert dd.shape == (7,)


# --- z position ------------------------------------------------------------

class TestZpos:
    def test_zpos_cd_array_matches_scalar_loop(self, setup):
        c, dc, times, _ = setup
        pz, dpz = zpos_cd(times, c, dc)
        assert pz.shape == (times.size,)
        assert dpz.shape == (times.size, 7)
        for n, t in enumerate(times):
            pz_n, dpz_n = zpos_cd(t, c, dc)
            assert_allclose(pz[n], pz_n, rtol=1e-12)
            assert_allclose(dpz[n], dpz_n, rtol=1e-12)

    def test_zpos_d_array_matches_scalar_loop(self, setup):
        c, dc, times, p = setup
        pz, dpz = zpos_d(times, 0.0, p, c, dc)
        assert pz.shape == (times.size,)
        assert dpz.shape == (times.size, 7)
        for n, t in enumerate(times):
            pz_n, dpz_n = zpos_d(t, 0.0, p, c, dc)
            assert_allclose(pz[n], pz_n, rtol=1e-12)
            assert_allclose(dpz[n], dpz_n, rtol=1e-12)


# --- velocity (cd-only) ----------------------------------------------------

class TestVel:
    def test_vel_cd_array_matches_scalar_loop(self, setup):
        c, dc, times, _ = setup
        vx, vy, vz, dvx, dvy, dvz = vel_cd(times, c, dc)
        _check_vector_quantities((vx, vy, vz), times.size)
        _check_gradients((dvx, dvy, dvz), times.size)
        for n, t in enumerate(times):
            vx_n, vy_n, vz_n, dvx_n, dvy_n, dvz_n = vel_cd(t, c, dc)
            assert_allclose([vx[n], vy[n], vz[n]], [vx_n, vy_n, vz_n], rtol=1e-12)
            assert_allclose(dvx[n], dvx_n, rtol=1e-12)
            assert_allclose(dvz[n], dvz_n, rtol=1e-12)


# --- z velocity ------------------------------------------------------------

class TestZvel:
    def test_zvel_cd_array_matches_scalar_loop(self, setup):
        c, dc, times, _ = setup
        vz, dvz = zvel_cd(times, c, dc)
        assert vz.shape == (times.size,)
        assert dvz.shape == (times.size, 7)
        for n, t in enumerate(times):
            vz_n, dvz_n = zvel_cd(t, c, dc)
            assert_allclose(vz[n], vz_n, rtol=1e-12)
            assert_allclose(dvz[n], dvz_n, rtol=1e-12)

    def test_zvel_d_array_matches_scalar_loop(self, setup):
        c, dc, times, p = setup
        vz, dvz = zvel_d(times, 0.0, p, c, dc)
        assert vz.shape == (times.size,)
        assert dvz.shape == (times.size, 7)
        for n, t in enumerate(times):
            vz_n, dvz_n = zvel_d(t, 0.0, p, c, dc)
            assert_allclose(vz[n], vz_n, rtol=1e-12)
            assert_allclose(dvz[n], dvz_n, rtol=1e-12)


# --- radial velocity (extra params) ----------------------------------------

class TestRv:
    K = 50.0  # RV semi-amplitude

    def test_rv_cd_array_matches_scalar_loop(self, setup, eccentric_orbit):
        c, dc, times, p = setup
        a, i, e = eccentric_orbit["a"], eccentric_orbit["i"], eccentric_orbit["e"]
        rv, drv = rv_cd(times, self.K, p, a, i, e, c, dc)
        assert rv.shape == (times.size,)
        assert drv.shape == (times.size, 7)
        for n, t in enumerate(times):
            rv_n, drv_n = rv_cd(t, self.K, p, a, i, e, c, dc)
            assert_allclose(rv[n], rv_n, rtol=1e-12)
            assert_allclose(drv[n], drv_n, rtol=1e-12)

    def test_rv_d_array_matches_scalar_loop(self, setup, eccentric_orbit):
        c, dc, times, p = setup
        a, i, e = eccentric_orbit["a"], eccentric_orbit["i"], eccentric_orbit["e"]
        rv, drv = rv_d(times, self.K, 0.0, p, a, i, e, c, dc)
        assert rv.shape == (times.size,)
        assert drv.shape == (times.size, 7)
        for n, t in enumerate(times):
            rv_n, drv_n = rv_d(t, self.K, 0.0, p, a, i, e, c, dc)
            assert_allclose(rv[n], rv_n, rtol=1e-12)
            assert_allclose(drv[n], drv_n, rtol=1e-12)

    def test_rv_cd_scalar_gradient_shape(self, setup, eccentric_orbit):
        c, dc, _, p = setup
        a, i, e = eccentric_orbit["a"], eccentric_orbit["i"], eccentric_orbit["e"]
        rv, drv = rv_cd(0.003, self.K, p, a, i, e, c, dc)
        assert np.isscalar(rv)
        assert drv.shape == (7,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
