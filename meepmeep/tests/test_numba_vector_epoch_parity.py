"""Vector/scalar parity of every gradient kernel over multi-epoch time grids.

The period chain term (``d/dp += epoch * d/dtc``) vanishes at epoch 0, and
the serial vector loops are only auto-vectorised by LLVM from about eight
samples upwards. numba 0.61 miscompiled ``zpos_d_v``/``zvel_d_v`` in exactly
that regime (the chain term was dropped, while the scalar path and the
parallel twins stayed correct), and the near-transit, few-sample grids of
the other parity suites could not see it. This suite runs every public
gradient vector kernel, serial and parallel, single- and
multi-expansion-point, on a 24-sample grid spanning many epochs and
compares it with a loop over the scalar path.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from meepmeep import numba2d, numba3d
from meepmeep.backends.numba.utils import eccentricity_vector, eccentricity_vector_d, mean_anomaly_at_transit

P, A, I, E, W, LAN = 5.0, 15.0, 1.55, 0.3, 0.5, 0.7
TC, TE = 0.4, 0.05
N = 24
RNG = np.random.default_rng(11)
T_POINT = TC + TE + P * RNG.integers(-9, 10, N) + RNG.uniform(-0.03, 0.03, N)
T_ORBIT = TC + P * RNG.uniform(-6.0, 7.0, N)

C2, DC2 = numba2d.solve2d_d(TE, P, A, I, E, W, LAN)
C3, DC3 = numba3d.solve3d_d(TE, P, A, I, E, W, LAN)
EP_TIMES, _, DT, EP_TABLE = numba3d.create_expansion_points(15, E, 'ea')
COEFFS, DCOEFFS = numba3d.solve3d_orbit_d(EP_TIMES, P, A, I, E, W, LAN)
TPA = TC - mean_anomaly_at_transit(E, W) / (2 * np.pi) * P
EV = tuple(eccentricity_vector(I, E, W, LAN))
DEV = eccentricity_vector_d(I, E, W, LAN)[1]
V = np.array([0.3, -1.2, 0.7])

# name: (dispatcher, serial vector kernel, parallel kernel, call(fn, t))
POINT = {
    "2d.pos_d": (numba2d.pos_d, numba2d.pos_d_v, numba2d.pos_d_vp, lambda f, t: f(t, TC, P, C2, DC2, TE)),
    "2d.sep_d": (numba2d.sep_d, numba2d.sep_d_v, numba2d.sep_d_vp, lambda f, t: f(t, TC, P, C2, DC2, TE)),
}
for _q in ("pos", "zpos", "sep", "vel", "zvel", "cos_alpha"):
    POINT[f"3d.{_q}_d"] = (getattr(numba3d, f"{_q}_d"), getattr(numba3d, f"{_q}_d_v"), getattr(numba3d, f"{_q}_d_vp"),
                           lambda f, t: f(t, TC, P, C3, DC3, TE))
POINT["3d.rv_d"] = (numba3d.rv_d, numba3d.rv_d_v, numba3d.rv_d_vp,
                    lambda f, t: f(t, 12.0, TC, P, A, I, E, C3, DC3, TE))
POINT["3d.lambert_d"] = (numba3d.lambert_phase_curve_d, numba3d.lambert_phase_curve_d_v,
                         numba3d.lambert_phase_curve_d_vp, lambda f, t: f(t, 0.3, 0.1, TC, P, C3, DC3, TE))
POINT["3d.ev_signal_d"] = (numba3d.ev_signal_d, numba3d.ev_signal_d_v, numba3d.ev_signal_d_vp,
                           lambda f, t: f(t, 1.3, 1e-3, I, TC, P, C3, DC3, TE))
POINT["3d.emission_d"] = (numba3d.emission_phase_curve_d, numba3d.emission_phase_curve_d_v,
                          numba3d.emission_phase_curve_d_vp, lambda f, t: f(t, 0.1, 0.02, 0.4, TC, P, C3, DC3, TE))

GRID = (DT, EP_TABLE, EP_TIMES, COEFFS, DCOEFFS)
ORBIT = {}
for _q in ("pos", "zpos", "sep", "vel", "zvel", "cos_alpha", "star_planet_distance"):
    ORBIT[_q] = (getattr(numba3d, f"{_q}_od"), getattr(numba3d, f"{_q}_ovd"), getattr(numba3d, f"{_q}_ovdp"),
                 lambda f, t: f(t, TPA, P, *GRID))
ORBIT["rv"] = (numba3d.rv_od, numba3d.rv_ovd, numba3d.rv_ovdp, lambda f, t: f(t, 12.0, TPA, P, A, I, E, *GRID))
ORBIT["lambert"] = (numba3d.lambert_phase_curve_od, numba3d.lambert_phase_curve_ovd,
                    numba3d.lambert_phase_curve_ovdp, lambda f, t: f(t, 0.3, 0.1, TPA, P, *GRID))
ORBIT["ev_signal"] = (numba3d.ev_signal_od, numba3d.ev_signal_ovd, numba3d.ev_signal_ovdp,
                      lambda f, t: f(1.3, 1e-3, I, t, TPA, P, *GRID))
ORBIT["emission"] = (numba3d.emission_phase_curve_od, numba3d.emission_phase_curve_ovd,
                     numba3d.emission_phase_curve_ovdp, lambda f, t: f(t, 0.1, 0.02, 0.4, TPA, P, *GRID))
ORBIT["light_travel_time"] = (numba3d.light_travel_time_od, numba3d.light_travel_time_ovd,
                              numba3d.light_travel_time_ovdp, lambda f, t: f(t, TPA, P, E, W, 0.9, *GRID, True))
ORBIT["true_anomaly"] = (numba3d.true_anomaly_od, numba3d.true_anomaly_ovd, numba3d.true_anomaly_ovdp,
                         lambda f, t: f(t, TPA, P, *EV, W, DEV, *GRID))
ORBIT["cos_v_p_angle"] = (numba3d.cos_v_p_angle_od, numba3d.cos_v_p_angle_ovd, numba3d.cos_v_p_angle_ovdp,
                          lambda f, t: f(V, t, TPA, P, *GRID))


def _scalar_loop(dispatcher, call, times):
    outs = [call(dispatcher, float(t)) for t in times]
    return tuple(np.array([o[m] for o in outs]) for m in range(len(outs[0])))


def _compare(actual, expected):
    for act, exp in zip(actual, expected):
        assert act.shape == exp.shape
        scale = max(np.abs(exp).max(), 1e-300)
        # atol covers ulp-level fastmath contraction differences between the
        # scalar and vector inlining contexts (see CLAUDE.md, write-into kernels).
        assert_allclose(act, exp, rtol=1e-11, atol=1e-13 * scale)


@pytest.mark.parametrize("kernel", ["dispatcher", "serial", "parallel"])
@pytest.mark.parametrize("name", list(POINT))
def test_single_expansion_point(name, kernel):
    dispatcher, serial, parallel, call = POINT[name]
    fn = {"dispatcher": dispatcher, "serial": serial, "parallel": parallel}[kernel]
    _compare(call(fn, T_POINT), _scalar_loop(dispatcher, call, T_POINT))


@pytest.mark.parametrize("kernel", ["dispatcher", "serial", "parallel"])
@pytest.mark.parametrize("name", list(ORBIT))
def test_multi_expansion_point(name, kernel):
    dispatcher, serial, parallel, call = ORBIT[name]
    fn = {"dispatcher": dispatcher, "serial": serial, "parallel": parallel}[kernel]
    _compare(call(fn, T_ORBIT), _scalar_loop(dispatcher, call, T_ORBIT))


def test_grids_span_many_epochs():
    epochs = np.floor((T_POINT - TC - TE + 0.5 * P) / P)
    assert len(np.unique(epochs)) >= 8 and N >= 16
