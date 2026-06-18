"""Serial/parallel parity for the single-expansion-point prange kernel twins.

Every single-expansion-point vector kernel (``_X_v`` / ``_X_cd_v`` / ``_X_d_v``) has a
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

from meepmeep.expansion2d import Expansion2D
from meepmeep.backends.numba.point2d import (
    solve2d,
    pos_c_v, pos_c_vp, pos_v, pos_vp,
    sep_c_v, sep_c_vp, sep_v, sep_vp,
)
from meepmeep.backends.numba.point2dd import (
    solve2d_d,
    pos_cd_v, pos_cd_vp, pos_d_v, pos_d_vp,
    sep_cd_v, sep_cd_vp, sep_d_v, sep_d_vp,
)
from meepmeep.backends.numba.point3d import (
    solve3d,
    pos_c_v as _pos_c_v3, pos_c_vp as _pos_c_vp3,
    pos_v as _pos_v3, pos_vp as _pos_vp3,
    zpos_c_v, zpos_c_vp, zpos_v, zpos_vp,
    sep_c_v as _sep_c_v3, sep_c_vp as _sep_c_vp3,
    sep_v as _sep_v3, sep_vp as _sep_vp3,
    vel_c_v, vel_c_vp, vel_v, vel_vp,
    zvel_c_v, zvel_c_vp, zvel_v, zvel_vp,
)
from meepmeep.backends.numba.point3dd import (
    solve3d_d,
    pos_cd_v as _pos_cd_v3, pos_cd_vp as _pos_cd_vp3,
    pos_d_v as _pos_d_v3, pos_d_vp as _pos_d_vp3,
    zpos_cd_v, zpos_cd_vp, zpos_d_v, zpos_d_vp,
    sep_cd_v as _sep_cd_v3, sep_cd_vp as _sep_cd_vp3,
    sep_d_v as _sep_d_v3, sep_d_vp as _sep_d_vp3,
    vel_cd_v, vel_cd_vp, vel_d_v, vel_d_vp,
    zvel_cd_v, zvel_cd_vp, zvel_d_v, zvel_d_vp,
    rv_cd_v, rv_cd_vp, rv_d_v, rv_d_vp,
)

PARS = dict(p=3.4, a=8.0, i=1.54, e=0.1, w=0.4)
P = PARS["p"]
TC = 1.5
TE = 0.01
K_RV = 50.0
T_CEN = np.linspace(-0.06, 0.08, 2001)          # expansion-point-centred times
T_ABS = TC + TE + np.linspace(-0.06, 0.08, 2001)  # absolute times near the expansion point
TOL = {"rtol": 1e-12, "atol": 1e-14}


@pytest.fixture(scope="module")
def c2():
    return solve2d(TE, **PARS)


@pytest.fixture(scope="module")
def cdc2():
    return solve2d_d(TE, **PARS)


@pytest.fixture(scope="module")
def c3():
    return solve3d(TE, **PARS)


@pytest.fixture(scope="module")
def cdc3():
    return solve3d_d(TE, **PARS)


def _compare(serial, par, args):
    res_s = serial(*args)
    res_p = par(*args)
    if not isinstance(res_s, tuple):
        res_s, res_p = (res_s,), (res_p,)
    for u, v in zip(res_s, res_p):
        assert_allclose(v, u, **TOL)


class TestPoint2dValue:
    def test_pos_c(self, c2):
        _compare(pos_c_v, pos_c_vp, (T_CEN, c2))

    def test_pos(self, c2):
        _compare(pos_v, pos_vp, (T_ABS, TC, P, c2, TE))

    def test_sep_c(self, c2):
        _compare(sep_c_v, sep_c_vp, (T_CEN, c2))

    def test_sep(self, c2):
        _compare(sep_v, sep_vp, (T_ABS, TC, P, c2, TE))


class TestPoint2dGradient:
    def test_pos_cd(self, cdc2):
        _compare(pos_cd_v, pos_cd_vp, (T_CEN, *cdc2))

    def test_pos_d(self, cdc2):
        c, dc = cdc2
        _compare(pos_d_v, pos_d_vp, (T_ABS, TC, P, c, dc, TE))

    def test_sep_cd(self, cdc2):
        _compare(sep_cd_v, sep_cd_vp, (T_CEN, *cdc2))

    def test_sep_d(self, cdc2):
        c, dc = cdc2
        _compare(sep_d_v, sep_d_vp, (T_ABS, TC, P, c, dc, TE))


class TestPoint3dValue:
    def test_pos_c(self, c3):
        _compare(_pos_c_v3, _pos_c_vp3, (T_CEN, c3))

    def test_pos(self, c3):
        _compare(_pos_v3, _pos_vp3, (T_ABS, TC, P, c3, TE))

    def test_zpos_c(self, c3):
        _compare(zpos_c_v, zpos_c_vp, (T_CEN, c3))

    def test_zpos(self, c3):
        _compare(zpos_v, zpos_vp, (T_ABS, TC, P, c3, TE))

    def test_sep_c(self, c3):
        _compare(_sep_c_v3, _sep_c_vp3, (T_CEN, c3))

    def test_sep(self, c3):
        _compare(_sep_v3, _sep_vp3, (T_ABS, TC, P, c3, TE))

    def test_vel_c(self, c3):
        _compare(vel_c_v, vel_c_vp, (T_CEN, c3))

    def test_vel(self, c3):
        _compare(vel_v, vel_vp, (T_ABS, TC, P, c3, TE))

    def test_zvel_c(self, c3):
        _compare(zvel_c_v, zvel_c_vp, (T_CEN, c3))

    def test_zvel(self, c3):
        _compare(zvel_v, zvel_vp, (T_ABS, TC, P, c3, TE))


class TestPoint3dGradient:
    def test_pos_cd(self, cdc3):
        _compare(_pos_cd_v3, _pos_cd_vp3, (T_CEN, *cdc3))

    def test_pos_d(self, cdc3):
        c, dc = cdc3
        _compare(_pos_d_v3, _pos_d_vp3, (T_ABS, TC, P, c, dc, TE))

    def test_zpos_cd(self, cdc3):
        _compare(zpos_cd_v, zpos_cd_vp, (T_CEN, *cdc3))

    def test_zpos_d(self, cdc3):
        c, dc = cdc3
        _compare(zpos_d_v, zpos_d_vp, (T_ABS, TC, P, c, dc, TE))

    def test_sep_cd(self, cdc3):
        _compare(_sep_cd_v3, _sep_cd_vp3, (T_CEN, *cdc3))

    def test_sep_d(self, cdc3):
        c, dc = cdc3
        _compare(_sep_d_v3, _sep_d_vp3, (T_ABS, TC, P, c, dc, TE))

    def test_vel_cd(self, cdc3):
        _compare(vel_cd_v, vel_cd_vp, (T_CEN, *cdc3))

    def test_vel_d(self, cdc3):
        c, dc = cdc3
        _compare(vel_d_v, vel_d_vp, (T_ABS, TC, P, c, dc, TE))

    def test_zvel_cd(self, cdc3):
        _compare(zvel_cd_v, zvel_cd_vp, (T_CEN, *cdc3))

    def test_zvel_d(self, cdc3):
        c, dc = cdc3
        _compare(zvel_d_v, zvel_d_vp, (T_ABS, TC, P, c, dc, TE))


class TestExpansion2dParallelOptIn:
    """Expansion2D(parallel=True) must change performance, never results."""

    @pytest.fixture(params=[False, True], ids=["values", "derivatives"])
    def pair(self, request):
        times = TC + TE + np.linspace(-0.05, 0.05, 500)
        serial = Expansion2D(tc=TC, te=TE, derivatives=request.param, **PARS)
        par = Expansion2D(tc=TC, te=TE, derivatives=request.param, parallel=True, **PARS)
        # Force the parallel path regardless of array size.
        par._PARALLEL_NMIN_VALUE = 0
        par._PARALLEL_NMIN_GRAD = 0
        for k2 in (serial, par):
            k2.set_data(times)
        return serial, par

    def test_default_is_serial(self):
        assert Expansion2D(tc=TC, **PARS)._parallel is False

    def test_position(self, pair):
        s, p = pair
        for u, v in zip(s.position(), p.position()):
            assert_allclose(v, u, **TOL)

    def test_projected_separation(self, pair):
        s, p = pair
        res_s, res_p = s.projected_separation(), p.projected_separation()
        if not isinstance(res_s, tuple):
            res_s, res_p = (res_s,), (res_p,)
        for u, v in zip(res_s, res_p):
            assert_allclose(v, u, **TOL)

    def test_below_threshold_stays_serial(self):
        """With default thresholds, a small grid must use the serial path
        (kernel selection is observable through _select)."""
        from meepmeep.backends.numba.point2d import sep as sep_dispatcher
        k2 = Expansion2D(tc=TC, te=TE, parallel=True, **PARS)
        k2.set_data(TC + np.linspace(-0.05, 0.05, 100))
        assert k2._select(sep_dispatcher, sep_vp, k2._PARALLEL_NMIN_VALUE) is sep_dispatcher


class TestRvExplicitTwins:
    """rv gradients use per-thread z-velocity scratch in the parallel twin."""

    def test_rv_cd(self, cdc3):
        c, dc = cdc3
        a, i, e = PARS["a"], PARS["i"], PARS["e"]
        _compare(rv_cd_v, rv_cd_vp, (T_CEN, K_RV, P, a, i, e, c, dc))

    def test_rv_d(self, cdc3):
        c, dc = cdc3
        a, i, e = PARS["a"], PARS["i"], PARS["e"]
        _compare(rv_d_v, rv_d_vp, (T_ABS, K_RV, TC, P, a, i, e, c, dc, TE))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
