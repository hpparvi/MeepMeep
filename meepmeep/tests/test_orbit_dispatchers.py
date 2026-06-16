"""Smoke tests for the orbit3d/orbit3dd ``*_o`` / ``*_od`` overload dispatchers.

The dispatchers route at compile time (inside ``@njit``) or at call time
(from pure Python) based on whether the time argument is a scalar
``float`` or a 1-D float64 ``ndarray``. These tests confirm:

1. Pure-Python calls route correctly and return the expected shape/type.
2. ``@njit`` callers can compile both paths without TypingError.
3. The dispatched results equal direct calls to the underlying private
   ``_*_os`` / ``_*_ov`` (and ``_*_osd`` / ``_*_ovd``) kernels.
"""
import numpy as np
import pytest
from numba import njit
from numpy.testing import assert_allclose

from meepmeep.backends.numba.expansion_points import create_expansion_points
from meepmeep.backends.numba.utils import TWO_PI, mean_anomaly_at_transit
from meepmeep.backends.numba.orbit3d import (
    solve3d_orbit,
    pos_o, sep_o, rv_o,
    _pos_os, _pos_ov, _sep_os, _sep_ov, _rv_os, _rv_ov,
)
from meepmeep.backends.numba.orbit3dd import (
    solve3d_orbit_d,
    pos_od, sep_od,
    _pos_osd, _pos_ovd, _sep_osd, _sep_ovd,
)


NPT = 15


@pytest.fixture
def orbit_args(test_orbital_params):
    pars = test_orbital_params["eccentric"]
    p, e, w = pars["p"], pars["e"], pars["w"]
    ep_times, _, dt, pkt = create_expansion_points(NPT, max(e, 0.2), "ea")
    c = solve3d_orbit(ep_times, **pars, npt=NPT)
    c_d_pair = solve3d_orbit_d(ep_times, **pars, npt=NPT)
    tpa = -mean_anomaly_at_transit(e, w) / TWO_PI * p
    return {
        "pars": pars,
        "tpa": tpa, "dt": dt, "pkt": pkt, "pts": ep_times,
        "c": c, "c_d": c_d_pair[0], "dc_d": c_d_pair[1],
        "t_scalar": float(p * 0.3),
        "t_array": np.linspace(0.0, p, 25),
    }


class TestPythonDispatch:
    """Dispatchers route correctly when called from pure Python."""

    def test_pos_o_scalar_returns_tuple_of_floats(self, orbit_args):
        a = orbit_args
        result = pos_o(a["t_scalar"], a["tpa"], a["pars"]["p"], a["dt"],
                       a["pkt"], a["pts"], a["c"])
        assert len(result) == 3
        for component in result:
            assert isinstance(component, float)

    def test_pos_o_array_returns_tuple_of_arrays(self, orbit_args):
        a = orbit_args
        result = pos_o(a["t_array"], a["tpa"], a["pars"]["p"], a["dt"],
                       a["pkt"], a["pts"], a["c"])
        assert len(result) == 3
        for arr in result:
            assert isinstance(arr, np.ndarray)
            assert arr.shape == a["t_array"].shape

    def test_sep_o_scalar_matches_private(self, orbit_args):
        a = orbit_args
        via_dispatch = sep_o(a["t_scalar"], a["tpa"], a["pars"]["p"], a["dt"],
                             a["pkt"], a["pts"], a["c"])
        via_private = _sep_os(a["t_scalar"], a["tpa"], a["pars"]["p"], a["dt"],
                              a["pkt"], a["pts"], a["c"])
        assert via_dispatch == via_private

    def test_sep_o_array_matches_private(self, orbit_args):
        a = orbit_args
        via_dispatch = sep_o(a["t_array"], a["tpa"], a["pars"]["p"], a["dt"],
                             a["pkt"], a["pts"], a["c"])
        via_private = _sep_ov(a["t_array"], a["tpa"], a["pars"]["p"], a["dt"],
                              a["pkt"], a["pts"], a["c"])
        assert_allclose(via_dispatch, via_private, rtol=0, atol=0)

    def test_pos_od_scalar_returns_gradient_shape_6(self, orbit_args):
        a = orbit_args
        result = pos_od(a["t_scalar"], a["tpa"], a["pars"]["p"], a["dt"],
                        a["pkt"], a["pts"], a["c_d"], a["dc_d"])
        assert len(result) == 6  # x, y, z, dx, dy, dz
        # last three are derivative arrays of shape (6,)
        assert result[3].shape == (7,)
        assert result[4].shape == (7,)
        assert result[5].shape == (7,)

    def test_pos_od_array_returns_per_time_gradients(self, orbit_args):
        a = orbit_args
        result = pos_od(a["t_array"], a["tpa"], a["pars"]["p"], a["dt"],
                        a["pkt"], a["pts"], a["c_d"], a["dc_d"])
        n = a["t_array"].size
        # x, y, z arrays shape (N,); dx, dy, dz shape (N, 6)
        assert result[0].shape == (n,)
        assert result[3].shape == (n, 7)


class TestNjitDispatch:
    """Dispatchers compile and route correctly inside @njit callers."""

    def test_pos_o_njit_scalar(self, orbit_args):
        a = orbit_args

        @njit
        def caller(t, tpa, p, dt, pkt, pts, c):
            return pos_o(t, tpa, p, dt, pkt, pts, c)

        result = caller(a["t_scalar"], a["tpa"], a["pars"]["p"], a["dt"],
                        a["pkt"], a["pts"], a["c"])
        ref = _pos_os(a["t_scalar"], a["tpa"], a["pars"]["p"], a["dt"],
                      a["pkt"], a["pts"], a["c"])
        assert result == ref

    def test_pos_o_njit_array(self, orbit_args):
        a = orbit_args

        @njit
        def caller(t, tpa, p, dt, pkt, pts, c):
            return pos_o(t, tpa, p, dt, pkt, pts, c)

        result = caller(a["t_array"], a["tpa"], a["pars"]["p"], a["dt"],
                        a["pkt"], a["pts"], a["c"])
        ref = _pos_ov(a["t_array"], a["tpa"], a["pars"]["p"], a["dt"],
                      a["pkt"], a["pts"], a["c"])
        for r, e in zip(result, ref):
            assert_allclose(r, e, rtol=0, atol=0)

    def test_rv_o_njit_dispatch_both_shapes(self, orbit_args):
        a = orbit_args
        k = 50.0
        p = a["pars"]["p"]
        a_, i_, e_ = a["pars"]["a"], a["pars"]["i"], a["pars"]["e"]

        @njit
        def caller(t):
            return rv_o(t, k, a["tpa"], p, a_, i_, e_, a["dt"],
                        a["pkt"], a["pts"], a["c"])

        # Numba closes over a["tpa"] etc. — instead, pass everything.
        @njit
        def caller_full(t, k_, tpa, p_, a__, i__, e__, dt, pkt, pts, c):
            return rv_o(t, k_, tpa, p_, a__, i__, e__, dt, pkt, pts, c)

        scalar_result = caller_full(a["t_scalar"], k, a["tpa"], p, a_, i_, e_,
                                    a["dt"], a["pkt"], a["pts"], a["c"])
        array_result = caller_full(a["t_array"], k, a["tpa"], p, a_, i_, e_,
                                   a["dt"], a["pkt"], a["pts"], a["c"])

        scalar_ref = _rv_os(a["t_scalar"], k, a["tpa"], p, a_, i_, e_,
                            a["dt"], a["pkt"], a["pts"], a["c"])
        array_ref = _rv_ov(a["t_array"], k, a["tpa"], p, a_, i_, e_,
                           a["dt"], a["pkt"], a["pts"], a["c"])

        assert scalar_result == scalar_ref
        assert_allclose(array_result, array_ref, rtol=0, atol=0)

    def test_sep_od_njit_dispatch(self, orbit_args):
        a = orbit_args
        p = a["pars"]["p"]

        @njit
        def caller(t, tpa, p_, dt, pkt, pts, c, dc):
            return sep_od(t, tpa, p_, dt, pkt, pts, c, dc)

        s_scalar, ds_scalar = caller(a["t_scalar"], a["tpa"], p, a["dt"],
                                     a["pkt"], a["pts"], a["c_d"], a["dc_d"])
        s_array, ds_array = caller(a["t_array"], a["tpa"], p, a["dt"],
                                   a["pkt"], a["pts"], a["c_d"], a["dc_d"])
        s_scalar_ref, ds_scalar_ref = _sep_osd(a["t_scalar"], a["tpa"], p, a["dt"],
                                               a["pkt"], a["pts"], a["c_d"], a["dc_d"])
        s_array_ref, ds_array_ref = _sep_ovd(a["t_array"], a["tpa"], p, a["dt"],
                                             a["pkt"], a["pts"], a["c_d"], a["dc_d"])
        assert s_scalar == s_scalar_ref
        assert_allclose(ds_scalar, ds_scalar_ref, rtol=0, atol=0)
        assert_allclose(s_array, s_array_ref, rtol=0, atol=0)
        assert_allclose(ds_array, ds_array_ref, rtol=0, atol=0)
