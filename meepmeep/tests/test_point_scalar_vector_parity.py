"""Scalar/array parity for the single-knot value-only evaluators.

The ``point2d``/``point3d`` value evaluators accept a scalar time or a 1-D
array of times. These tests pin that the array path returns exactly what a
loop over scalar calls returns, both from pure Python and from inside an
``@njit`` caller (where the dispatch happens at compile time). They guard
the implementation of the array path - originally NumPy broadcasting, now
explicit loop kernels - against behavioural drift.
"""
import numpy as np
import pytest
from numba import njit
from numpy.testing import assert_allclose

from meepmeep.backends.numba import point2d, point3d
from meepmeep.backends.numba.point2d import solve2d
from meepmeep.backends.numba.point3d import solve3d

PARS = {"p": 5.0, "a": 15.0, "i": 1.55, "e": 0.3, "w": 0.5}
P = PARS["p"]
K_RV = 50.0
TIMES = np.linspace(-0.4, 0.4, 257)
TOL = {"rtol": 1e-12, "atol": 1e-14}


@pytest.fixture(scope="module")
def c2():
    return solve2d(0.0, **PARS)


@pytest.fixture(scope="module")
def c3():
    return solve3d(0.0, **PARS)


def _check(fn, args):
    """Array call must equal a loop of scalar calls (multi-output aware)."""
    res_v = fn(TIMES, *args)
    multi = isinstance(res_v, tuple)
    for j in (0, len(TIMES) // 2, len(TIMES) - 1):
        res_s = fn(float(TIMES[j]), *args)
        if multi:
            for u, v in zip(res_v, res_s):
                assert_allclose(u[j], v, **TOL)
        else:
            assert_allclose(res_v[j], res_s, **TOL)


class TestPoint2dValueParity:
    def test_pos_c(self, c2):
        _check(point2d.pos_c, (c2,))

    def test_pos(self, c2):
        _check(point2d.pos, (0.0, P, c2))

    def test_sep_c(self, c2):
        _check(point2d.sep_c, (c2,))

    def test_sep(self, c2):
        _check(point2d.sep, (0.0, P, c2))


class TestPoint3dValueParity:
    def test_pos_c(self, c3):
        _check(point3d.pos_c, (c3,))

    def test_pos(self, c3):
        _check(point3d.pos, (0.0, P, c3))

    def test_zpos_c(self, c3):
        _check(point3d.zpos_c, (c3,))

    def test_zpos(self, c3):
        _check(point3d.zpos, (0.0, P, c3))

    def test_sep_c(self, c3):
        _check(point3d.sep_c, (c3,))

    def test_sep(self, c3):
        _check(point3d.sep, (0.0, P, c3))

    def test_vel_c(self, c3):
        _check(point3d.vel_c, (c3,))

    def test_zvel_c(self, c3):
        _check(point3d.zvel_c, (c3,))

    def test_zvel(self, c3):
        _check(point3d.zvel, (0.0, P, c3))


class TestRvValueParity:
    """rv/rv_c take extra parameters before the coefficient matrix."""

    def test_rv_c(self, c3):
        a, i, e = PARS["a"], PARS["i"], PARS["e"]
        rv_v = point3d.rv_c(TIMES, K_RV, P, a, i, e, c3)
        for j in (0, len(TIMES) // 2, len(TIMES) - 1):
            assert_allclose(rv_v[j], point3d.rv_c(float(TIMES[j]), K_RV, P, a, i, e, c3), **TOL)

    def test_rv(self, c3):
        a, i, e = PARS["a"], PARS["i"], PARS["e"]
        rv_v = point3d.rv(TIMES, K_RV, 0.0, P, a, i, e, c3)
        for j in (0, len(TIMES) // 2, len(TIMES) - 1):
            assert_allclose(rv_v[j], point3d.rv(float(TIMES[j]), K_RV, 0.0, P, a, i, e, c3), **TOL)


class TestNjitContext:
    """The same dispatch must work inside @njit callers, scalar and array."""

    def test_pos3d_njit(self, c3):
        from meepmeep.backends.numba.point3d import pos as pos3

        @njit
        def runner(t, tk, p, c):
            return pos3(t, tk, p, c)

        xs, ys, zs = runner(TIMES, 0.0, P, c3)
        for j in (0, len(TIMES) - 1):
            x, y, z = runner(float(TIMES[j]), 0.0, P, c3)
            assert_allclose((xs[j], ys[j], zs[j]), (x, y, z), **TOL)

    def test_sep2d_njit(self, c2):
        from meepmeep.backends.numba.point2d import sep as sep2

        @njit
        def runner(t, tk, p, c):
            return sep2(t, tk, p, c)

        d = runner(TIMES, 0.0, P, c2)
        for j in (0, len(TIMES) - 1):
            assert_allclose(d[j], runner(float(TIMES[j]), 0.0, P, c2), **TOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
