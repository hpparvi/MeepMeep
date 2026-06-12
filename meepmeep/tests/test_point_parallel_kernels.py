"""Serial/parallel parity for the single-knot prange kernel twins.

Every single-knot vector kernel (``_X_v`` / ``_X_cd_v`` / ``_X_d_v``) has a
parallel twin with a trailing ``p`` (``_X_vp`` etc.) compiled with
``parallel=True``. The scratch-free kernels are dual-decorated from one
shared ``prange`` body (prange compiles as a plain range in the serial
compilation), so serial and parallel results come from identical source;
the rv gradient kernels need per-thread scratch and are explicit twins.
These tests pin that the parallel kernels return what the serial kernels
return.

Tolerances: fastmath contraction can differ by an ulp between the serial
and parallel compilations, so comparisons use a tiny atol instead of
exact equality.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from meepmeep.backends.numba.point2d import (
    solve2d,
    _pos_c_v, _pos_c_vp, _pos_v, _pos_vp,
    _sep_c_v, _sep_c_vp, _sep_v, _sep_vp,
)
from meepmeep.backends.numba.point2dd import (
    solve2d_d,
    _pos_cd_v, _pos_cd_vp, _pos_d_v, _pos_d_vp,
    _sep_cd_v, _sep_cd_vp, _sep_d_v, _sep_d_vp,
)
from meepmeep.backends.numba.point3d import (
    solve3d,
    _pos_c_v as _pos_c_v3, _pos_c_vp as _pos_c_vp3,
    _pos_v as _pos_v3, _pos_vp as _pos_vp3,
    _zpos_c_v, _zpos_c_vp, _zpos_v, _zpos_vp,
    _sep_c_v as _sep_c_v3, _sep_c_vp as _sep_c_vp3,
    _sep_v as _sep_v3, _sep_vp as _sep_vp3,
    _vel_c_v, _vel_c_vp,
    _zvel_c_v, _zvel_c_vp, _zvel_v, _zvel_vp,
)
from meepmeep.backends.numba.point3dd import (
    solve3d_d,
    _pos_cd_v as _pos_cd_v3, _pos_cd_vp as _pos_cd_vp3,
    _pos_d_v as _pos_d_v3, _pos_d_vp as _pos_d_vp3,
    _zpos_cd_v, _zpos_cd_vp, _zpos_d_v, _zpos_d_vp,
    _sep_cd_v as _sep_cd_v3, _sep_cd_vp as _sep_cd_vp3,
    _sep_d_v as _sep_d_v3, _sep_d_vp as _sep_d_vp3,
    _vel_cd_v, _vel_cd_vp,
    _zvel_cd_v, _zvel_cd_vp, _zvel_d_v, _zvel_d_vp,
    _rv_cd_v, _rv_cd_vp, _rv_d_v, _rv_d_vp,
)

PARS = dict(p=3.4, a=8.0, i=1.54, e=0.1, w=0.4)
P = PARS["p"]
TC = 1.5
TK = 0.01
K_RV = 50.0
T_CEN = np.linspace(-0.06, 0.08, 2001)          # knot-centred times
T_ABS = TC + TK + np.linspace(-0.06, 0.08, 2001)  # absolute times near the knot
TOL = {"rtol": 1e-12, "atol": 1e-14}


@pytest.fixture(scope="module")
def c2():
    return solve2d(TK, **PARS)


@pytest.fixture(scope="module")
def cdc2():
    return solve2d_d(TK, **PARS)


@pytest.fixture(scope="module")
def c3():
    return solve3d(TK, **PARS)


@pytest.fixture(scope="module")
def cdc3():
    return solve3d_d(TK, **PARS)


def _compare(serial, par, args):
    res_s = serial(*args)
    res_p = par(*args)
    if not isinstance(res_s, tuple):
        res_s, res_p = (res_s,), (res_p,)
    for u, v in zip(res_s, res_p):
        assert_allclose(v, u, **TOL)


class TestPoint2dValue:
    def test_pos_c(self, c2):
        _compare(_pos_c_v, _pos_c_vp, (T_CEN, c2))

    def test_pos(self, c2):
        _compare(_pos_v, _pos_vp, (T_ABS, TC, P, c2, TK))

    def test_sep_c(self, c2):
        _compare(_sep_c_v, _sep_c_vp, (T_CEN, c2))

    def test_sep(self, c2):
        _compare(_sep_v, _sep_vp, (T_ABS, TC, P, c2, TK))


class TestPoint2dGradient:
    def test_pos_cd(self, cdc2):
        _compare(_pos_cd_v, _pos_cd_vp, (T_CEN, *cdc2))

    def test_pos_d(self, cdc2):
        c, dc = cdc2
        _compare(_pos_d_v, _pos_d_vp, (T_ABS, TC, P, c, dc, TK))

    def test_sep_cd(self, cdc2):
        _compare(_sep_cd_v, _sep_cd_vp, (T_CEN, *cdc2))

    def test_sep_d(self, cdc2):
        c, dc = cdc2
        _compare(_sep_d_v, _sep_d_vp, (T_ABS, TC, P, c, dc, TK))


class TestPoint3dValue:
    def test_pos_c(self, c3):
        _compare(_pos_c_v3, _pos_c_vp3, (T_CEN, c3))

    def test_pos(self, c3):
        _compare(_pos_v3, _pos_vp3, (T_ABS, TC, P, c3, TK))

    def test_zpos_c(self, c3):
        _compare(_zpos_c_v, _zpos_c_vp, (T_CEN, c3))

    def test_zpos(self, c3):
        _compare(_zpos_v, _zpos_vp, (T_ABS, TC, P, c3, TK))

    def test_sep_c(self, c3):
        _compare(_sep_c_v3, _sep_c_vp3, (T_CEN, c3))

    def test_sep(self, c3):
        _compare(_sep_v3, _sep_vp3, (T_ABS, TC, P, c3, TK))

    def test_vel_c(self, c3):
        _compare(_vel_c_v, _vel_c_vp, (T_CEN, c3))

    def test_zvel_c(self, c3):
        _compare(_zvel_c_v, _zvel_c_vp, (T_CEN, c3))

    def test_zvel(self, c3):
        _compare(_zvel_v, _zvel_vp, (T_ABS, TC, P, c3, TK))


class TestPoint3dGradient:
    def test_pos_cd(self, cdc3):
        _compare(_pos_cd_v3, _pos_cd_vp3, (T_CEN, *cdc3))

    def test_pos_d(self, cdc3):
        c, dc = cdc3
        _compare(_pos_d_v3, _pos_d_vp3, (T_ABS, TC, P, c, dc, TK))

    def test_zpos_cd(self, cdc3):
        _compare(_zpos_cd_v, _zpos_cd_vp, (T_CEN, *cdc3))

    def test_zpos_d(self, cdc3):
        c, dc = cdc3
        _compare(_zpos_d_v, _zpos_d_vp, (T_ABS, TC, P, c, dc, TK))

    def test_sep_cd(self, cdc3):
        _compare(_sep_cd_v3, _sep_cd_vp3, (T_CEN, *cdc3))

    def test_sep_d(self, cdc3):
        c, dc = cdc3
        _compare(_sep_d_v3, _sep_d_vp3, (T_ABS, TC, P, c, dc, TK))

    def test_vel_cd(self, cdc3):
        _compare(_vel_cd_v, _vel_cd_vp, (T_CEN, *cdc3))

    def test_zvel_cd(self, cdc3):
        _compare(_zvel_cd_v, _zvel_cd_vp, (T_CEN, *cdc3))

    def test_zvel_d(self, cdc3):
        c, dc = cdc3
        _compare(_zvel_d_v, _zvel_d_vp, (T_ABS, TC, P, c, dc, TK))


class TestRvExplicitTwins:
    """rv gradients use per-thread z-velocity scratch in the parallel twin."""

    def test_rv_cd(self, cdc3):
        c, dc = cdc3
        a, i, e = PARS["a"], PARS["i"], PARS["e"]
        _compare(_rv_cd_v, _rv_cd_vp, (T_CEN, K_RV, P, a, i, e, c, dc))

    def test_rv_d(self, cdc3):
        c, dc = cdc3
        a, i, e = PARS["a"], PARS["i"], PARS["e"]
        _compare(_rv_d_v, _rv_d_vp, (T_ABS, K_RV, TC, P, a, i, e, c, dc, TK))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
