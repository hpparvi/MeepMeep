"""Serial/parallel parity for the prange kernel twins.

Every multi-expansion-point vector kernel (``_X_ov`` / ``_X_ovd``) has a parallel twin
(``_X_ovp`` / ``_X_ovdp``) compiled with ``parallel=True`` and a ``prange``
sample loop. These tests pin that the twins return the same results as the
serial kernels, and that the ``Orbit(parallel=True)`` opt-in routes through
them without changing any output.

Tolerances: the twins share the same write-into kernels as the serial
loops, but fastmath contraction can differ by an ulp between inlining
contexts, so the comparisons use a tiny atol instead of exact equality.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from meepmeep.backends.numba.expansion_points import create_expansion_points
from meepmeep.backends.numba.orbit3d import (
    solve3d_orbit,
    pos_ov, pos_ovp, zpos_ov, zpos_ovp, sep_ov, sep_ovp,
    vel_ov, vel_ovp, zvel_ov, zvel_ovp, rv_ov, rv_ovp,
    cos_alpha_ov, cos_alpha_ovp, cos_v_p_angle_ov, cos_v_p_angle_ovp,
    star_planet_distance_ov, star_planet_distance_ovp,
    true_anomaly_ov, true_anomaly_ovp,
    lambert_phase_curve_ov, lambert_phase_curve_ovp,
    ev_signal_ov, ev_signal_ovp,
    light_travel_time_ov, light_travel_time_ovp,
)
from meepmeep.backends.numba.orbit3dd import (
    solve3d_orbit_d,
    pos_ovd, pos_ovdp, zpos_ovd, zpos_ovdp, sep_ovd, sep_ovdp,
    vel_ovd, vel_ovdp, zvel_ovd, zvel_ovdp, rv_ovd, rv_ovdp,
    cos_alpha_ovd, cos_alpha_ovdp, cos_v_p_angle_ovd, cos_v_p_angle_ovdp,
    star_planet_distance_ovd, star_planet_distance_ovdp,
    true_anomaly_ovd, true_anomaly_ovdp,
    lambert_phase_curve_ovd, lambert_phase_curve_ovdp,
    ev_signal_ovd, ev_signal_ovdp,
    light_travel_time_ovd, light_travel_time_ovdp,
)
from meepmeep.backends.numba.utils import TWO_PI, mean_anomaly_at_transit, eccentricity_vector
from meepmeep.orbit import Orbit

NPT = 15
N = 3001
PARS = {"p": 5.0, "a": 15.0, "i": 1.55, "e": 0.3, "w": 0.5}
TOL = {"rtol": 1e-12, "atol": 1e-13}


@pytest.fixture(scope="module")
def setup():
    p, e, w = PARS["p"], PARS["e"], PARS["w"]
    ep_times, _, dt, pkt = create_expansion_points(NPT, max(e, 0.2), "ea")
    coeffs = solve3d_orbit(ep_times, **PARS, npt=NPT)
    _, dcoeffs = solve3d_orbit_d(ep_times, **PARS, npt=NPT)
    tpa = -mean_anomaly_at_transit(e, w) / TWO_PI * p
    times = np.linspace(0.0, p, N)
    ev = eccentricity_vector(PARS["i"], e, w)
    return times, tpa, p, dt, pkt, ep_times, coeffs, dcoeffs, ev


def _compare(serial, par, args):
    res_s = serial(*args)
    res_p = par(*args)
    if not isinstance(res_s, tuple):
        res_s, res_p = (res_s,), (res_p,)
    for u, v in zip(res_s, res_p):
        assert_allclose(v, u, **TOL)


class TestValueKernelParity:
    def test_pos(self, setup):
        t, tpa, p, dt, pkt, pts, c, _, _ = setup
        _compare(pos_ov, pos_ovp, (t, tpa, p, dt, pkt, pts, c))

    def test_zpos(self, setup):
        t, tpa, p, dt, pkt, pts, c, _, _ = setup
        _compare(zpos_ov, zpos_ovp, (t, tpa, p, dt, pkt, pts, c))

    def test_sep(self, setup):
        t, tpa, p, dt, pkt, pts, c, _, _ = setup
        _compare(sep_ov, sep_ovp, (t, tpa, p, dt, pkt, pts, c))

    def test_vel(self, setup):
        t, tpa, p, dt, pkt, pts, c, _, _ = setup
        _compare(vel_ov, vel_ovp, (t, tpa, p, dt, pkt, pts, c))

    def test_zvel(self, setup):
        t, tpa, p, dt, pkt, pts, c, _, _ = setup
        _compare(zvel_ov, zvel_ovp, (t, tpa, p, dt, pkt, pts, c))

    def test_rv(self, setup):
        t, tpa, p, dt, pkt, pts, c, _, _ = setup
        _compare(rv_ov, rv_ovp, (t, 0.05, tpa, p, PARS["a"], PARS["i"], PARS["e"], dt, pkt, pts, c))

    def test_cos_alpha(self, setup):
        t, tpa, p, dt, pkt, pts, c, _, _ = setup
        _compare(cos_alpha_ov, cos_alpha_ovp, (t, tpa, p, dt, pkt, pts, c))

    def test_cos_v_p_angle(self, setup):
        t, tpa, p, dt, pkt, pts, c, _, _ = setup
        v = np.array([0.3, 0.5, 0.81])
        _compare(cos_v_p_angle_ov, cos_v_p_angle_ovp, (v, t, tpa, p, dt, pkt, pts, c))

    def test_star_planet_distance(self, setup):
        t, tpa, p, dt, pkt, pts, c, _, _ = setup
        _compare(star_planet_distance_ov, star_planet_distance_ovp, (t, tpa, p, dt, pkt, pts, c))

    def test_true_anomaly(self, setup):
        t, tpa, p, dt, pkt, pts, c, _, ev = setup
        _compare(true_anomaly_ov, true_anomaly_ovp,
                 (t, tpa, p, ev[0], ev[1], ev[2], PARS["w"], dt, pkt, pts, c))

    def test_lambert_phase_curve(self, setup):
        t, tpa, p, dt, pkt, pts, c, _, _ = setup
        _compare(lambert_phase_curve_ov, lambert_phase_curve_ovp,
                 (t, 0.3, 0.1, tpa, p, dt, pkt, pts, c))

    def test_ev_signal(self, setup):
        t, tpa, p, dt, pkt, pts, c, _, _ = setup
        _compare(ev_signal_ov, ev_signal_ovp, (8e-6, 1e-3, PARS["i"], t, tpa, p, dt, pkt, pts, c))

    def test_light_travel_time(self, setup):
        t, tpa, p, dt, pkt, pts, c, _, _ = setup
        _compare(light_travel_time_ov, light_travel_time_ovp,
                 (t, tpa, p, PARS["e"], PARS["w"], 1.0, dt, pkt, pts, c))


class TestGradientKernelParity:
    def test_pos(self, setup):
        t, tpa, p, dt, pkt, pts, c, dc, _ = setup
        _compare(pos_ovd, pos_ovdp, (t, tpa, p, dt, pkt, pts, c, dc))

    def test_zpos(self, setup):
        t, tpa, p, dt, pkt, pts, c, dc, _ = setup
        _compare(zpos_ovd, zpos_ovdp, (t, tpa, p, dt, pkt, pts, c, dc))

    def test_sep(self, setup):
        t, tpa, p, dt, pkt, pts, c, dc, _ = setup
        _compare(sep_ovd, sep_ovdp, (t, tpa, p, dt, pkt, pts, c, dc))

    def test_vel(self, setup):
        t, tpa, p, dt, pkt, pts, c, dc, _ = setup
        _compare(vel_ovd, vel_ovdp, (t, tpa, p, dt, pkt, pts, c, dc))

    def test_zvel(self, setup):
        t, tpa, p, dt, pkt, pts, c, dc, _ = setup
        _compare(zvel_ovd, zvel_ovdp, (t, tpa, p, dt, pkt, pts, c, dc))

    def test_rv(self, setup):
        t, tpa, p, dt, pkt, pts, c, dc, _ = setup
        _compare(rv_ovd, rv_ovdp,
                 (t, 0.05, tpa, p, PARS["a"], PARS["i"], PARS["e"], dt, pkt, pts, c, dc))

    def test_cos_alpha(self, setup):
        t, tpa, p, dt, pkt, pts, c, dc, _ = setup
        _compare(cos_alpha_ovd, cos_alpha_ovdp, (t, tpa, p, dt, pkt, pts, c, dc))

    def test_cos_v_p_angle(self, setup):
        t, tpa, p, dt, pkt, pts, c, dc, _ = setup
        v = np.array([0.3, 0.5, 0.81])
        _compare(cos_v_p_angle_ovd, cos_v_p_angle_ovdp, (v, t, tpa, p, dt, pkt, pts, c, dc))

    def test_star_planet_distance(self, setup):
        t, tpa, p, dt, pkt, pts, c, dc, _ = setup
        _compare(star_planet_distance_ovd, star_planet_distance_ovdp,
                 (t, tpa, p, dt, pkt, pts, c, dc))

    def test_true_anomaly(self, setup):
        t, tpa, p, dt, pkt, pts, c, dc, ev = setup
        _compare(true_anomaly_ovd, true_anomaly_ovdp,
                 (t, tpa, p, ev[0], ev[1], ev[2], PARS["w"], dt, pkt, pts, c, dc))

    def test_lambert_phase_curve(self, setup):
        t, tpa, p, dt, pkt, pts, c, dc, _ = setup
        _compare(lambert_phase_curve_ovd, lambert_phase_curve_ovdp,
                 (t, 0.3, 0.1, tpa, p, dt, pkt, pts, c, dc))

    def test_ev_signal(self, setup):
        t, tpa, p, dt, pkt, pts, c, dc, _ = setup
        _compare(ev_signal_ovd, ev_signal_ovdp,
                 (8e-6, 1e-3, PARS["i"], t, tpa, p, dt, pkt, pts, c, dc))

    def test_light_travel_time(self, setup):
        t, tpa, p, dt, pkt, pts, c, dc, _ = setup
        _compare(light_travel_time_ovd, light_travel_time_ovdp,
                 (t, tpa, p, PARS["e"], PARS["w"], 1.0, dt, pkt, pts, c, dc))


class TestOrbitParallelOptIn:
    """Orbit(parallel=True) must change performance, never results."""

    @pytest.fixture(params=[False, True], ids=["values", "derivatives"])
    def orbit_pair(self, request):
        times = np.linspace(0.0, PARS["p"], 2000)
        serial = Orbit(npt=NPT, derivatives=request.param)
        par = Orbit(npt=NPT, derivatives=request.param, parallel=True)
        # Force the parallel path regardless of array size.
        par._PARALLEL_NMIN_VALUE = 0
        par._PARALLEL_NMIN_GRAD = 0
        for o in (serial, par):
            o.set_pars(tc=0.0, **PARS)
            o.set_data(times)
        return serial, par

    def _pair_check(self, res_s, res_p):
        if not isinstance(res_s, tuple):
            res_s, res_p = (res_s,), (res_p,)
        for u, v in zip(res_s, res_p):
            assert_allclose(v, u, **TOL)

    def test_default_is_serial(self):
        assert Orbit(npt=NPT)._parallel is False

    def test_xyz(self, orbit_pair):
        s, p = orbit_pair
        self._pair_check(s.xyz(), p.xyz())

    def test_vxyz(self, orbit_pair):
        s, p = orbit_pair
        self._pair_check(s.vxyz(), p.vxyz())

    def test_cos_phase(self, orbit_pair):
        s, p = orbit_pair
        self._pair_check(s.cos_phase(), p.cos_phase())

    def test_true_anomaly(self, orbit_pair):
        s, p = orbit_pair
        self._pair_check(s.true_anomaly(), p.true_anomaly())

    def test_star_planet_distance(self, orbit_pair):
        s, p = orbit_pair
        self._pair_check(s.star_planet_distance(), p.star_planet_distance())

    def test_radial_velocity(self, orbit_pair):
        s, p = orbit_pair
        self._pair_check(s.radial_velocity(0.05), p.radial_velocity(0.05))

    def test_lambert_phase_curve(self, orbit_pair):
        s, p = orbit_pair
        self._pair_check(s.lambert_phase_curve(0.1, 0.3), p.lambert_phase_curve(0.1, 0.3))

    def test_ellipsoidal_variation(self, orbit_pair):
        s, p = orbit_pair
        self._pair_check(s.ellipsoidal_variation(8e-6, 1e-3),
                         p.ellipsoidal_variation(8e-6, 1e-3))

    def test_light_travel_time(self, orbit_pair):
        s, p = orbit_pair
        self._pair_check(s.light_travel_time(1.0), p.light_travel_time(1.0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
