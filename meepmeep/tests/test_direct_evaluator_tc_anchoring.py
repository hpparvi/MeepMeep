"""Time anchoring of the direct single-expansion-point evaluators.

The direct evaluators (``pos`` / ``sep`` / ``zpos`` / ``zvel`` / ``rv`` and
their ``_d`` variants) take the transit-centre time ``tc`` as their time
anchor, plus an optional expansion-point offset ``te`` that has the same meaning and
value as the ``te`` argument of ``solve2d`` / ``solve3d`` (0.0 = expansion point at
the transit centre). These tests pin that contract with a transit centre
far from zero - the configuration in which the historical reading of the
second argument as "the expansion-point time" silently produced wrong orbits.
"""
import numpy as np
import pytest
from numba import njit
from numpy.testing import assert_allclose

from meepmeep import numba2d as mm2
from meepmeep import numba3d as mm3
from meepmeep.expansion2d import Expansion2D
from meepmeep.backends.numba.newton.newton import xy_newton_v, xyz_newton_v, rv_newton_v

PARS = dict(p=3.4, a=8.0, i=1.54, e=0.1, w=0.4)
P = PARS["p"]
TC = 100.3                 # absolute transit centre, not commensurate with P
TE = 0.02                  # off-centre expansion-point offset [days]
K_RV = 50.0
TIMES = TC + np.linspace(-0.06, 0.08, 11)
ORACLE_TOL = dict(rtol=0, atol=5e-5)


@pytest.fixture(scope="module")
def newton_xy():
    return xy_newton_v(TIMES, TC, **PARS)


@pytest.fixture(scope="module")
def newton_xyz():
    return xyz_newton_v(TIMES, TC, **PARS)


class TestAbsoluteTimes2d:
    def test_pos_ep_at_transit(self, newton_xy):
        c = mm2.solve2d(0.0, **PARS)
        x, y = mm2.pos(TIMES, TC, P, c)
        assert_allclose(x, newton_xy[0], **ORACLE_TOL)
        assert_allclose(y, newton_xy[1], **ORACLE_TOL)

    def test_pos_offcentre_ep(self, newton_xy):
        c = mm2.solve2d(TE, **PARS)
        x, y = mm2.pos(TIMES, TC, P, c, te=TE)
        assert_allclose(x, newton_xy[0], **ORACLE_TOL)
        assert_allclose(y, newton_xy[1], **ORACLE_TOL)

    def test_sep_offcentre_ep(self, newton_xy):
        c = mm2.solve2d(TE, **PARS)
        d = mm2.sep(TIMES, TC, P, c, te=TE)
        assert_allclose(d, np.hypot(*newton_xy), **ORACLE_TOL)

    def test_derivative_variants_match_values(self):
        c, dc = mm2.solve2d_d(TE, **PARS)
        x, y = mm2.pos(TIMES, TC, P, c, te=TE)
        xd, yd, dx, dy = mm2.pos_d(TIMES, TC, P, c, dc, te=TE)
        assert_allclose(xd, x, rtol=1e-12)
        assert_allclose(yd, y, rtol=1e-12)
        assert dx.shape == (TIMES.size, 7)
        d, dd = mm2.sep_d(TIMES, TC, P, c, dc, te=TE)
        assert_allclose(d, np.hypot(x, y), rtol=1e-12)
        assert dd.shape == (TIMES.size, 7)


class TestAbsoluteTimes3d:
    def test_pos_zpos_sep_offcentre_ep(self, newton_xyz):
        xn, yn, zn = newton_xyz
        c = mm3.solve3d(TE, **PARS)
        x, y, z = mm3.pos(TIMES, TC, P, c, te=TE)
        assert_allclose(x, xn, **ORACLE_TOL)
        assert_allclose(y, yn, **ORACLE_TOL)
        assert_allclose(z, zn, **ORACLE_TOL)
        assert_allclose(mm3.zpos(TIMES, TC, P, c, te=TE), zn, **ORACLE_TOL)
        assert_allclose(mm3.sep(TIMES, TC, P, c, te=TE), np.hypot(xn, yn), **ORACLE_TOL)

    def test_rv_offcentre_ep(self):
        c = mm3.solve3d(TE, **PARS)
        rv = mm3.rv(TIMES, K_RV, TC, P, PARS["a"], PARS["i"], PARS["e"], c, te=TE)
        rv_ref = rv_newton_v(TIMES, K_RV, TC, P, PARS["e"], PARS["w"])
        assert_allclose(rv, rv_ref, rtol=0, atol=0.05)

    def test_derivative_variants_match_values(self):
        c, dc = mm3.solve3d_d(TE, **PARS)
        x, y, z = mm3.pos(TIMES, TC, P, c, te=TE)
        xd, yd, zd, dx, dy, dz = mm3.pos_d(TIMES, TC, P, c, dc, te=TE)
        assert_allclose((xd, yd, zd), (x, y, z), rtol=1e-12)
        assert dx.shape == (TIMES.size, 7)
        zp, dzp = mm3.zpos_d(TIMES, TC, P, c, dc, te=TE)
        assert_allclose(zp, z, rtol=1e-12)
        vz, dvz = mm3.zvel_d(TIMES, TC, P, c, dc, te=TE)
        assert_allclose(vz, mm3.zvel(TIMES, TC, P, c, te=TE), rtol=1e-12)
        rv, drv = mm3.rv_d(TIMES, K_RV, TC, P, PARS["a"], PARS["i"], PARS["e"], c, dc, te=TE)
        assert_allclose(rv, mm3.rv(TIMES, K_RV, TC, P, PARS["a"], PARS["i"], PARS["e"], c, te=TE),
                        rtol=1e-12)
        assert drv.shape == (TIMES.size, 7)


class TestDispatchPaths:
    """The optional te must work for scalar times and inside @njit callers."""

    def test_scalar_array_parity_with_tk(self):
        c = mm2.solve2d(TE, **PARS)
        d = mm2.sep(TIMES, TC, P, c, te=TE)
        for j in (0, 5, 10):
            assert_allclose(mm2.sep(float(TIMES[j]), TC, P, c, te=TE), d[j], rtol=1e-12)

    def test_njit_caller_default_and_explicit_tk(self):
        sep3 = mm3.sep

        @njit
        def with_default(t, tc, p, c):
            return sep3(t, tc, p, c)

        @njit
        def with_te(t, tc, p, c, te):
            return sep3(t, tc, p, c, te)

        c0 = mm3.solve3d(0.0, **PARS)
        ck = mm3.solve3d(TE, **PARS)
        assert_allclose(with_default(TIMES, TC, P, c0), mm3.sep(TIMES, TC, P, c0), rtol=1e-12)
        assert_allclose(with_te(TIMES, TC, P, ck, TE), mm3.sep(TIMES, TC, P, ck, te=TE), rtol=1e-12)


class TestExpansion2dAbsoluteAnchoring:
    def test_offcentre_ep_far_tc(self, newton_xy):
        k2 = Expansion2D(tc=TC, te=TE, **PARS)
        k2.set_data(TIMES)
        assert_allclose(k2.projected_separation, np.hypot(*newton_xy), **ORACLE_TOL)
        x, y = k2.position
        assert_allclose(x, newton_xy[0], **ORACLE_TOL)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
