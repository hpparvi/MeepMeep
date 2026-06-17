"""Scalar/vector parity for the orbit3d / orbit3dd multi-expansion-point evaluators.

The orbit-spanning evaluators have both scalar (``_*_os`` / ``_*_osd``) and
vector (``_*_ov`` / ``_*_ovd``) variants. The vector variants either loop
over the scalar variants directly or duplicate the per-element body. Either
way the two should produce identical results on a single time point. These
tests pin that invariant so the planned overload dispatcher (Commit C) can
rely on routing scalar and vector cases to numerically equivalent code paths.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

from meepmeep.backends.numba.expansion_points import create_expansion_points
from meepmeep.backends.numba.utils import (
    TWO_PI,
    mean_anomaly_at_transit,
    eccentricity_vector,
)
from meepmeep.backends.numba.orbit3d import (
    solve3d_orbit,
    _pos_os, pos_ov,
    _zpos_os, zpos_ov,
    _sep_os, sep_ov,
    _vel_os, vel_ov,
    _zvel_os, zvel_ov,
    _true_anomaly_os, true_anomaly_ov,
    _cos_v_p_angle_os, cos_v_p_angle_ov,
    _cos_alpha_os, cos_alpha_ov,
    _star_planet_distance_os, star_planet_distance_ov,
    _lambert_phase_curve_os, lambert_phase_curve_ov,
    _ev_signal_os, ev_signal_ov,
    _emission_phase_curve_os, emission_phase_curve_ov,
    _rv_os, rv_ov,
    _light_travel_time_os, light_travel_time_ov,
)
from meepmeep.backends.numba.orbit3dd import (
    solve3d_orbit_d,
    _pos_osd, pos_ovd,
    _zpos_osd, zpos_ovd,
    _sep_osd, sep_ovd,
    _vel_osd, vel_ovd,
    _zvel_osd, zvel_ovd,
    _cos_alpha_osd, cos_alpha_ovd,
    _star_planet_distance_osd, star_planet_distance_ovd,
    _cos_v_p_angle_osd, cos_v_p_angle_ovd,
    _true_anomaly_osd, true_anomaly_ovd,
    _lambert_phase_curve_osd, lambert_phase_curve_ovd,
    _ev_signal_osd, ev_signal_ovd,
    _emission_phase_curve_osd, emission_phase_curve_ovd,
    _rv_osd, rv_ovd,
    _light_travel_time_osd, light_travel_time_ovd,
)


NPT = 15


def _setup(orbit_pars):
    p = orbit_pars["p"]
    e = orbit_pars["e"]
    w = orbit_pars["w"]
    ep_times, _, dt, ep_table = create_expansion_points(NPT, max(e, 0.2), "ea")
    coeffs = solve3d_orbit(ep_times, **orbit_pars, npt=NPT)
    tpa = -mean_anomaly_at_transit(e, w) / TWO_PI * p
    # Three representative times: near transit, mid-orbit, near eclipse.
    t_scalars = np.array([0.0, p * 0.25, p * 0.7])
    return t_scalars, tpa, dt, ep_table, ep_times, coeffs


def _setup_d(orbit_pars):
    p = orbit_pars["p"]
    e = orbit_pars["e"]
    w = orbit_pars["w"]
    ep_times, _, dt, ep_table = create_expansion_points(NPT, max(e, 0.2), "ea")
    coeffs, dcoeffs = solve3d_orbit_d(ep_times, **orbit_pars, npt=NPT)
    tpa = -mean_anomaly_at_transit(e, w) / TWO_PI * p
    t_scalars = np.array([0.0, p * 0.25, p * 0.7])
    return t_scalars, tpa, dt, ep_table, ep_times, coeffs, dcoeffs


@pytest.fixture(params=["circular", "eccentric", "high_e"])
def orbit_case(request, test_orbital_params):
    return test_orbital_params[request.param]


# ---------------------------------------------------------------------------
# orbit3d.py: value-only families
# ---------------------------------------------------------------------------

class TestForwardParity:

    def test_pos(self, orbit_case):
        ts, tpa, dt, pkt, pts, c = _setup(orbit_case)
        p = orbit_case["p"]
        xs, ys, zs = pos_ov(ts, tpa, p, dt, pkt, pts, c)
        for j, t in enumerate(ts):
            x, y, z = _pos_os(float(t), tpa, p, dt, pkt, pts, c)
            assert_allclose([x, y, z], [xs[j], ys[j], zs[j]], rtol=1e-14, atol=1e-14)

    def test_zpos(self, orbit_case):
        ts, tpa, dt, pkt, pts, c = _setup(orbit_case)
        p = orbit_case["p"]
        zs = zpos_ov(ts, tpa, p, dt, pkt, pts, c)
        for j, t in enumerate(ts):
            z = _zpos_os(float(t), tpa, p, dt, pkt, pts, c)
            assert_allclose(z, zs[j], rtol=1e-14, atol=1e-14)

    def test_sep(self, orbit_case):
        ts, tpa, dt, pkt, pts, c = _setup(orbit_case)
        p = orbit_case["p"]
        seps = sep_ov(ts, tpa, p, dt, pkt, pts, c)
        for j, t in enumerate(ts):
            s = _sep_os(float(t), tpa, p, dt, pkt, pts, c)
            assert_allclose(s, seps[j], rtol=1e-14, atol=1e-14)

    def test_vel(self, orbit_case):
        ts, tpa, dt, pkt, pts, c = _setup(orbit_case)
        p = orbit_case["p"]
        vxs, vys, vzs = vel_ov(ts, tpa, p, dt, pkt, pts, c)
        for j, t in enumerate(ts):
            vx, vy, vz = _vel_os(float(t), tpa, p, dt, pkt, pts, c)
            assert_allclose([vx, vy, vz], [vxs[j], vys[j], vzs[j]], rtol=1e-14, atol=1e-14)

    def test_zvel(self, orbit_case):
        ts, tpa, dt, pkt, pts, c = _setup(orbit_case)
        p = orbit_case["p"]
        vzs = zvel_ov(ts, tpa, p, dt, pkt, pts, c)
        for j, t in enumerate(ts):
            vz = _zvel_os(float(t), tpa, p, dt, pkt, pts, c)
            assert_allclose(vz, vzs[j], rtol=1e-14, atol=1e-14)

    def test_cos_alpha(self, orbit_case):
        ts, tpa, dt, pkt, pts, c = _setup(orbit_case)
        p = orbit_case["p"]
        cas = cos_alpha_ov(ts, tpa, p, dt, pkt, pts, c)
        for j, t in enumerate(ts):
            ca = _cos_alpha_os(float(t), tpa, p, dt, pkt, pts, c)
            assert_allclose(ca, cas[j], rtol=1e-14, atol=1e-14)

    def test_star_planet_distance(self, orbit_case):
        ts, tpa, dt, pkt, pts, c = _setup(orbit_case)
        p = orbit_case["p"]
        rs = star_planet_distance_ov(ts, tpa, p, dt, pkt, pts, c)
        for j, t in enumerate(ts):
            r = _star_planet_distance_os(float(t), tpa, p, dt, pkt, pts, c)
            assert_allclose(r, rs[j], rtol=1e-14, atol=1e-14)

    def test_cos_v_p_angle(self, orbit_case):
        ts, tpa, dt, pkt, pts, c = _setup(orbit_case)
        p = orbit_case["p"]
        v = np.array([0.3, -0.5, 0.8])
        cs = cos_v_p_angle_ov(v, ts, tpa, p, dt, pkt, pts, c)
        for j, t in enumerate(ts):
            cval = _cos_v_p_angle_os(v, float(t), tpa, p, dt, pkt, pts, c)
            assert_allclose(cval, cs[j], rtol=1e-14, atol=1e-14)

    def test_true_anomaly(self, orbit_case):
        ts, tpa, dt, pkt, pts, c = _setup(orbit_case)
        p = orbit_case["p"]
        ex, ey, ez = eccentricity_vector(orbit_case["i"], orbit_case["e"], orbit_case["w"])
        fs = true_anomaly_ov(ts, tpa, p, ex, ey, ez, orbit_case["w"], dt, pkt, pts, c)
        for j, t in enumerate(ts):
            f = _true_anomaly_os(float(t), tpa, p, ex, ey, ez, orbit_case["w"],
                                 dt, pkt, pts, c)
            assert_allclose(f, fs[j], rtol=1e-12, atol=1e-12)

    def test_lambert_phase_curve(self, orbit_case):
        ts, tpa, dt, pkt, pts, c = _setup(orbit_case)
        p = orbit_case["p"]
        ag, k = 0.3, 0.1
        fl = lambert_phase_curve_ov(ts, ag, k, tpa, p, dt, pkt, pts, c)
        for j, t in enumerate(ts):
            f = _lambert_phase_curve_os(float(t), ag, k, tpa, p, dt, pkt, pts, c)
            assert_allclose(f, fl[j], rtol=1e-14, atol=1e-14)

    def test_emission_phase_curve(self, orbit_case):
        ts, tpa, dt, pkt, pts, c = _setup(orbit_case)
        p = orbit_case["p"]
        fl = emission_phase_curve_ov(ts, 0.1, 0.25, 0.4, tpa, p, dt, pkt, pts, c)
        for j, t in enumerate(ts):
            f = _emission_phase_curve_os(float(t), 0.1, 0.25, 0.4, tpa, p, dt, pkt, pts, c)
            assert_allclose(f, fl[j], rtol=1e-14, atol=1e-14)

    def test_ev_signal(self, orbit_case):
        ts, tpa, dt, pkt, pts, c = _setup(orbit_case)
        p = orbit_case["p"]
        alpha, mr, inc = 1.0, 1e-3, orbit_case["i"]
        ev_v = ev_signal_ov(alpha, mr, inc, ts, tpa, p, dt, pkt, pts, c)
        for j, t in enumerate(ts):
            ev = _ev_signal_os(alpha, mr, inc, float(t), tpa, p, dt, pkt, pts, c)
            assert_allclose(ev, ev_v[j], rtol=1e-14, atol=1e-14)

    def test_rv(self, orbit_case):
        ts, tpa, dt, pkt, pts, c = _setup(orbit_case)
        p = orbit_case["p"]
        k_amp = 50.0
        rvs = rv_ov(ts, k_amp, tpa, p, orbit_case["a"], orbit_case["i"],
                     orbit_case["e"], dt, pkt, pts, c)
        for j, t in enumerate(ts):
            rv = _rv_os(float(t), k_amp, tpa, p, orbit_case["a"], orbit_case["i"],
                        orbit_case["e"], dt, pkt, pts, c)
            assert_allclose(rv, rvs[j], rtol=1e-14, atol=1e-14)

    def test_light_travel_time(self, orbit_case):
        ts, tpa, dt, pkt, pts, c = _setup(orbit_case)
        p = orbit_case["p"]
        e, w, rstar = orbit_case["e"], orbit_case["w"], 1.0
        ltts = light_travel_time_ov(ts, tpa, p, e, w, rstar, dt, pkt, pts, c)
        for j, t in enumerate(ts):
            ltt = _light_travel_time_os(float(t), tpa, p, e, w, rstar, dt, pkt, pts, c)
            assert_allclose(ltt, ltts[j], rtol=1e-14, atol=1e-14)


# ---------------------------------------------------------------------------
# orbit3dd.py: gradient-returning families
# ---------------------------------------------------------------------------

class TestGradientParity:

    def test_pos(self, orbit_case):
        ts, tpa, dt, pkt, pts, c, dc = _setup_d(orbit_case)
        p = orbit_case["p"]
        xs, ys, zs, dxs, dys, dzs = pos_ovd(ts, tpa, p, dt, pkt, pts, c, dc)
        for j, t in enumerate(ts):
            x, y, z, dx, dy, dz = _pos_osd(float(t), tpa, p, dt, pkt, pts, c, dc)
            assert_allclose([x, y, z], [xs[j], ys[j], zs[j]], rtol=1e-14, atol=1e-14)
            assert_allclose(dx, dxs[j], rtol=1e-14, atol=1e-14)
            assert_allclose(dy, dys[j], rtol=1e-14, atol=1e-14)
            assert_allclose(dz, dzs[j], rtol=1e-14, atol=1e-14)

    def test_zpos(self, orbit_case):
        ts, tpa, dt, pkt, pts, c, dc = _setup_d(orbit_case)
        p = orbit_case["p"]
        zs, dzs = zpos_ovd(ts, tpa, p, dt, pkt, pts, c, dc)
        for j, t in enumerate(ts):
            z, dz = _zpos_osd(float(t), tpa, p, dt, pkt, pts, c, dc)
            assert_allclose(z, zs[j], rtol=1e-14, atol=1e-14)
            assert_allclose(dz, dzs[j], rtol=1e-14, atol=1e-14)

    def test_sep(self, orbit_case):
        ts, tpa, dt, pkt, pts, c, dc = _setup_d(orbit_case)
        p = orbit_case["p"]
        ds, dds = sep_ovd(ts, tpa, p, dt, pkt, pts, c, dc)
        for j, t in enumerate(ts):
            d, dd = _sep_osd(float(t), tpa, p, dt, pkt, pts, c, dc)
            assert_allclose(d, ds[j], rtol=1e-14, atol=1e-14)
            assert_allclose(dd, dds[j], rtol=1e-14, atol=1e-14)

    def test_vel(self, orbit_case):
        ts, tpa, dt, pkt, pts, c, dc = _setup_d(orbit_case)
        p = orbit_case["p"]
        vxs, vys, vzs, dvxs, dvys, dvzs = vel_ovd(ts, tpa, p, dt, pkt, pts, c, dc)
        for j, t in enumerate(ts):
            vx, vy, vz, dvx, dvy, dvz = _vel_osd(float(t), tpa, p, dt, pkt, pts, c, dc)
            assert_allclose([vx, vy, vz], [vxs[j], vys[j], vzs[j]], rtol=1e-14, atol=1e-14)
            assert_allclose(dvx, dvxs[j], rtol=1e-14, atol=1e-14)
            assert_allclose(dvy, dvys[j], rtol=1e-14, atol=1e-14)
            assert_allclose(dvz, dvzs[j], rtol=1e-14, atol=1e-14)

    def test_zvel(self, orbit_case):
        ts, tpa, dt, pkt, pts, c, dc = _setup_d(orbit_case)
        p = orbit_case["p"]
        vzs, dvzs = zvel_ovd(ts, tpa, p, dt, pkt, pts, c, dc)
        for j, t in enumerate(ts):
            vz, dvz = _zvel_osd(float(t), tpa, p, dt, pkt, pts, c, dc)
            assert_allclose(vz, vzs[j], rtol=1e-14, atol=1e-14)
            assert_allclose(dvz, dvzs[j], rtol=1e-14, atol=1e-14)

    def test_cos_alpha(self, orbit_case):
        ts, tpa, dt, pkt, pts, c, dc = _setup_d(orbit_case)
        p = orbit_case["p"]
        cas, dcas = cos_alpha_ovd(ts, tpa, p, dt, pkt, pts, c, dc)
        for j, t in enumerate(ts):
            ca, dca = _cos_alpha_osd(float(t), tpa, p, dt, pkt, pts, c, dc)
            assert_allclose(ca, cas[j], rtol=1e-14, atol=1e-14)
            assert_allclose(dca, dcas[j], rtol=1e-14, atol=1e-14)

    def test_star_planet_distance(self, orbit_case):
        ts, tpa, dt, pkt, pts, c, dc = _setup_d(orbit_case)
        p = orbit_case["p"]
        rs, drs = star_planet_distance_ovd(ts, tpa, p, dt, pkt, pts, c, dc)
        for j, t in enumerate(ts):
            r, dr = _star_planet_distance_osd(float(t), tpa, p, dt, pkt, pts, c, dc)
            assert_allclose(r, rs[j], rtol=1e-14, atol=1e-14)
            assert_allclose(dr, drs[j], rtol=1e-14, atol=1e-14)

    def test_cos_v_p_angle(self, orbit_case):
        ts, tpa, dt, pkt, pts, c, dc = _setup_d(orbit_case)
        p = orbit_case["p"]
        v = np.array([0.3, -0.5, 0.8])
        cs, dcs = cos_v_p_angle_ovd(v, ts, tpa, p, dt, pkt, pts, c, dc)
        for j, t in enumerate(ts):
            cval, dc_val = _cos_v_p_angle_osd(v, float(t), tpa, p, dt, pkt, pts, c, dc)
            assert_allclose(cval, cs[j], rtol=1e-14, atol=1e-14)
            assert_allclose(dc_val, dcs[j], rtol=1e-14, atol=1e-14)

    def test_true_anomaly(self, orbit_case):
        ts, tpa, dt, pkt, pts, c, dc = _setup_d(orbit_case)
        p = orbit_case["p"]
        ex, ey, ez = eccentricity_vector(orbit_case["i"], orbit_case["e"], orbit_case["w"])
        fs, dfs = true_anomaly_ovd(ts, tpa, p, ex, ey, ez, orbit_case["w"],
                                    dt, pkt, pts, c, dc)
        for j, t in enumerate(ts):
            f, df = _true_anomaly_osd(float(t), tpa, p, ex, ey, ez, orbit_case["w"],
                                      dt, pkt, pts, c, dc)
            assert_allclose(f, fs[j], rtol=1e-12, atol=1e-12)
            assert_allclose(df, dfs[j], rtol=1e-12, atol=1e-12)

    def test_lambert_phase_curve(self, orbit_case):
        ts, tpa, dt, pkt, pts, c, dc = _setup_d(orbit_case)
        p = orbit_case["p"]
        ag, k = 0.3, 0.1
        fl, dfl = lambert_phase_curve_ovd(ts, ag, k, tpa, p, dt, pkt, pts, c, dc)
        for j, t in enumerate(ts):
            f, df = _lambert_phase_curve_osd(float(t), ag, k, tpa, p, dt, pkt, pts, c, dc)
            assert_allclose(f, fl[j], rtol=1e-14, atol=1e-14)
            assert_allclose(df, dfl[j], rtol=1e-14, atol=1e-14)

    def test_emission_phase_curve(self, orbit_case):
        ts, tpa, dt, pkt, pts, c, dc = _setup_d(orbit_case)
        p = orbit_case["p"]
        fl, dfl = emission_phase_curve_ovd(ts, 0.1, 0.25, 0.4, tpa, p, dt, pkt, pts, c, dc)
        for j, t in enumerate(ts):
            f, df = _emission_phase_curve_osd(float(t), 0.1, 0.25, 0.4, tpa, p, dt, pkt, pts, c, dc)
            assert_allclose(f, fl[j], rtol=1e-14, atol=1e-14)
            assert_allclose(df, dfl[j], rtol=1e-14, atol=1e-14)

    def test_ev_signal(self, orbit_case):
        ts, tpa, dt, pkt, pts, c, dc = _setup_d(orbit_case)
        p = orbit_case["p"]
        alpha, mr, inc = 1.0, 1e-3, orbit_case["i"]
        ev_v, dev_v = ev_signal_ovd(alpha, mr, inc, ts, tpa, p, dt, pkt, pts, c, dc)
        for j, t in enumerate(ts):
            ev, dev = _ev_signal_osd(alpha, mr, inc, float(t), tpa, p, dt, pkt, pts, c, dc)
            assert_allclose(ev, ev_v[j], rtol=1e-14, atol=1e-14)
            assert_allclose(dev, dev_v[j], rtol=1e-14, atol=1e-14)

    def test_rv(self, orbit_case):
        ts, tpa, dt, pkt, pts, c, dc = _setup_d(orbit_case)
        p = orbit_case["p"]
        k_amp = 50.0
        rvs, drvs = rv_ovd(ts, k_amp, tpa, p, orbit_case["a"], orbit_case["i"],
                            orbit_case["e"], dt, pkt, pts, c, dc)
        for j, t in enumerate(ts):
            rv, drv = _rv_osd(float(t), k_amp, tpa, p, orbit_case["a"], orbit_case["i"],
                              orbit_case["e"], dt, pkt, pts, c, dc)
            assert_allclose(rv, rvs[j], rtol=1e-14, atol=1e-14)
            assert_allclose(drv, drvs[j], rtol=1e-14, atol=1e-14)

    def test_light_travel_time(self, orbit_case):
        ts, tpa, dt, pkt, pts, c, dc = _setup_d(orbit_case)
        p = orbit_case["p"]
        e, w, rstar = orbit_case["e"], orbit_case["w"], 1.0
        ltts, dltts = light_travel_time_ovd(ts, tpa, p, e, w, rstar, dt, pkt, pts, c, dc)
        for j, t in enumerate(ts):
            ltt, dltt = _light_travel_time_osd(float(t), tpa, p, e, w, rstar,
                                               dt, pkt, pts, c, dc)
            assert_allclose(ltt, ltts[j], rtol=1e-14, atol=1e-14)
            assert_allclose(dltt, dltts[j], rtol=1e-14, atol=1e-14)
