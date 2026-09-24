"""Parity tests for the OpenCL multi-expansion-point gradient evaluators.

Compares every quantity in orbit3dd.cl against the numba reference over
times spanning several periods (the period chain term d[1] += epoch*d[0]
is unobservable at epoch 0), including the extended gradient layouts
(8 slots for rv_od, 9 for lambert and ev_signal, 10 for emission), the
k = 0 guard in rv_od's k slot, and light_travel_time_od in both timing
bases. The ``__kernel`` wrappers are test-only.
"""

import numpy as np
import pytest

pyopencl = pytest.importorskip("pyopencl")
try:
    _HAS_OPENCL_DEVICE = bool(pyopencl.get_platforms())
except Exception:
    _HAS_OPENCL_DEVICE = False
if not _HAS_OPENCL_DEVICE:
    pytest.skip("No OpenCL platform available", allow_module_level=True)

import pyopencl as cl

from meepmeep.backends.numba.expansion_points import create_expansion_points
from meepmeep.backends.numba.orbit3dd import (solve3d_orbit_d, pos_od, zpos_od,
                                              sep_od, vel_od, zvel_od, rv_od,
                                              cos_alpha_od, cos_v_p_angle_od,
                                              true_anomaly_od,
                                              lambert_phase_curve_od,
                                              ev_signal_od,
                                              emission_phase_curve_od,
                                              star_planet_distance_od,
                                              light_travel_time_od)
from meepmeep.backends.numba.utils import (TWO_PI, mean_anomaly_at_transit,
                                           eccentricity_vector, tc_to_tp_gradient)
from meepmeep.tests.opencl_utils import (build, get_queue, has_fp64, upload,
                                         upload_ep_table, output_buffer, read_back)

RTOL = 1e-12
ATOL = 1e-13

NPT = 15
V_FIXED = (0.3, -0.5, 0.8)
RSTAR = 2.0
K_RV = 25.0

# name -> (device call, device scalar names, gradient length)
SCALAR_GRAD_KERNELS = {
    'zpos_od': ('zpos_od(t[i], tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs, g)',
                ('tpa', 'p', 'dt'), 7),
    'sep_od': ('sep_od(t[i], tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs, g)',
               ('tpa', 'p', 'dt'), 7),
    'zvel_od': ('zvel_od(t[i], tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs, g)',
                ('tpa', 'p', 'dt'), 7),
    'cos_alpha_od': (
        'cos_alpha_od(t[i], tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs, g)',
        ('tpa', 'p', 'dt'), 7),
    'star_planet_distance_od': (
        'star_planet_distance_od(t[i], tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs, g)',
        ('tpa', 'p', 'dt'), 7),
    'rv_od': (
        'rv_od(t[i], k, tpa, p, aa, inc, e, dt, ep_table, ep_times, coeffs, dcoeffs, g)',
        ('k', 'tpa', 'p', 'aa', 'inc', 'e', 'dt'), 8),
    'cos_v_p_angle_od': (
        'cos_v_p_angle_od(vx, vy, vz, t[i], tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs, g)',
        ('vx', 'vy', 'vz', 'tpa', 'p', 'dt'), 7),
    'true_anomaly_od': (
        'true_anomaly_od(t[i], tpa, p, ex, ey, ez, w, dt, ep_table, ep_times, coeffs, dcoeffs, 1, g)',
        ('tpa', 'p', 'ex', 'ey', 'ez', 'w', 'dt'), 7),
    'true_anomaly_od_tp': (
        'true_anomaly_od(t[i], tpa, p, ex, ey, ez, w, dt, ep_table, ep_times, coeffs, dcoeffs, 0, g)',
        ('tpa', 'p', 'ex', 'ey', 'ez', 'w', 'dt'), 7),
    'lambert_phase_curve_od': (
        'lambert_phase_curve_od(t[i], ag, k, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs, g)',
        ('ag', 'k', 'tpa', 'p', 'dt'), 9),
    'ev_signal_od': (
        'ev_signal_od(al, mq, inc, t[i], tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs, g)',
        ('al', 'mq', 'inc', 'tpa', 'p', 'dt'), 9),
    'emission_phase_curve_od': (
        'emission_phase_curve_od(t[i], k, fr, off, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs, g)',
        ('k', 'fr', 'off', 'tpa', 'p', 'dt'), 10),
}


def _scalar_grad_kernel_source(name, call, scalar_names, ng):
    decls = ''.join(f'const REAL {s}, ' for s in scalar_names)
    return (f"__kernel void k_{name}(__global const REAL *t, {decls}"
            f"__global const int *ep_table, __global const REAL *ep_times, "
            f"__global const REAL *coeffs, __global const REAL *dcoeffs, "
            f"__global REAL *val, __global REAL *grad) {{\n"
            f"    int i = get_global_id(0);\n"
            f"    REAL g[{ng}];\n"
            f"    val[i] = {call};\n"
            f"    for (int m = 0; m < {ng}; m++)\n"
            f"        grad[{ng} * i + m] = g[m];\n"
            f"}}\n")


EXTRA_KERNELS = """
__kernel void k_pos_od(__global const REAL *t, const REAL tpa, const REAL p,
                       const REAL dt, __global const int *ep_table,
                       __global const REAL *ep_times, __global const REAL *coeffs,
                       __global const REAL *dcoeffs,
                       __global REAL *val, __global REAL *grad) {
    int i = get_global_id(0);
    REAL x, y, z, dx[7], dy[7], dz[7];
    pos_od(t[i], tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs,
           &x, &y, &z, dx, dy, dz);
    val[3 * i] = x; val[3 * i + 1] = y; val[3 * i + 2] = z;
    for (int m = 0; m < 7; m++) {
        grad[21 * i + m] = dx[m];
        grad[21 * i + 7 + m] = dy[m];
        grad[21 * i + 14 + m] = dz[m];
    }
}

__kernel void k_vel_od(__global const REAL *t, const REAL tpa, const REAL p,
                       const REAL dt, __global const int *ep_table,
                       __global const REAL *ep_times, __global const REAL *coeffs,
                       __global const REAL *dcoeffs,
                       __global REAL *val, __global REAL *grad) {
    int i = get_global_id(0);
    REAL x, y, z, dx[7], dy[7], dz[7];
    vel_od(t[i], tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs,
           &x, &y, &z, dx, dy, dz);
    val[3 * i] = x; val[3 * i + 1] = y; val[3 * i + 2] = z;
    for (int m = 0; m < 7; m++) {
        grad[21 * i + m] = dx[m];
        grad[21 * i + 7 + m] = dy[m];
        grad[21 * i + 14 + m] = dz[m];
    }
}

__kernel void k_light_travel_time_od(__global const REAL *t, const REAL tpa,
                                     const REAL p, const REAL e, const REAL w,
                                     const REAL rstar, const REAL dt,
                                     const int timing_is_tc,
                                     __global const int *ep_table,
                                     __global const REAL *ep_times,
                                     __global const REAL *coeffs,
                                     __global const REAL *dcoeffs,
                                     __global REAL *val, __global REAL *grad) {
    int i = get_global_id(0);
    REAL g[7];
    val[i] = light_travel_time_od(t[i], tpa, p, e, w, rstar, dt,
                                  ep_table, ep_times, coeffs, dcoeffs,
                                  timing_is_tc, g);
    for (int m = 0; m < 7; m++)
        grad[7 * i + m] = g[m];
}
"""

TEST_KERNELS = EXTRA_KERNELS + ''.join(
    _scalar_grad_kernel_source(name, call, scalars, ng)
    for name, (call, scalars, ng) in SCALAR_GRAD_KERNELS.items())


@pytest.fixture(scope='module')
def program():
    if not has_fp64():
        pytest.skip("Device lacks cl_khr_fp64")
    return build(TEST_KERNELS, 'orbit3dd.cl')


@pytest.fixture(params=['circular', 'eccentric'])
def orbit(request, test_orbital_params):
    pars = test_orbital_params[request.param]
    return pars['p'], pars['a'], pars['i'], pars['e'], pars['w']


def setup_orbit(orbit):
    p, a, i, e, w = orbit
    ep_times, _, dt, ep_table = create_expansion_points(NPT, max(e, 0.2), 'ea')
    coeffs, dcoeffs = solve3d_orbit_d(ep_times, p, a, i, e, w, npt=NPT)
    tpa = -mean_anomaly_at_transit(e, w) / TWO_PI * p
    times = tpa + np.linspace(-2.5 * p, 3.5 * p, 400)
    return times, tpa, dt, ep_table, ep_times, coeffs, dcoeffs


def extra_pars(orbit, tpa, dt):
    p, a, i, e, w = orbit
    ev = eccentricity_vector(i, e, w)
    return {'tpa': tpa, 'p': p, 'aa': a, 'inc': i, 'e': e, 'w': w, 'dt': dt,
            'k': 0.1, 'ag': 0.3, 'al': 1.2, 'mq': 1e-3, 'fr': 1e-3, 'off': 0.3,
            'vx': V_FIXED[0], 'vy': V_FIXED[1], 'vz': V_FIXED[2],
            'ex': ev[0], 'ey': ev[1], 'ez': ev[2]}


def scalar_values(name, x):
    _, names, _ = SCALAR_GRAD_KERNELS[name]
    values = dict(x)
    if name == 'rv_od':
        values['k'] = K_RV
    return [values[n] for n in names]


def run_grad(program, name, times, scalars, ep_table, ep_times, coeffs, dcoeffs,
             ng, n_val=1, int_scalars=()):
    queue = get_queue()
    n = times.size
    b_t = upload(queue, times)
    b_tab = upload_ep_table(queue, ep_table)
    b_ept = upload(queue, ep_times)
    b_c = upload(queue, coeffs)
    b_dc = upload(queue, dcoeffs)
    b_v, h_v = output_buffer(queue, n_val * n)
    b_g, h_g = output_buffer(queue, ng * n)
    kernel = cl.Kernel(program, f'k_{name}')
    kernel(queue, (n,), None, b_t, *[np.float64(s) for s in scalars],
           *[np.int32(s) for s in int_scalars], b_tab, b_ept, b_c, b_dc, b_v, b_g)
    vals = read_back(queue, b_v, h_v)
    grads = read_back(queue, b_g, h_g).reshape(n, ng)
    return vals, grads


NUMBA_REF = {
    'zpos_od': lambda t, s, x: zpos_od(t, x['tpa'], x['p'], *s),
    'sep_od': lambda t, s, x: sep_od(t, x['tpa'], x['p'], *s),
    'zvel_od': lambda t, s, x: zvel_od(t, x['tpa'], x['p'], *s),
    'cos_alpha_od': lambda t, s, x: cos_alpha_od(t, x['tpa'], x['p'], *s),
    'star_planet_distance_od': lambda t, s, x: star_planet_distance_od(
        t, x['tpa'], x['p'], *s),
    'rv_od': lambda t, s, x: rv_od(
        t, K_RV, x['tpa'], x['p'], x['aa'], x['inc'], x['e'], *s),
    'cos_v_p_angle_od': lambda t, s, x: cos_v_p_angle_od(
        np.array(V_FIXED), t, x['tpa'], x['p'], *s),
    'true_anomaly_od': lambda t, s, x: true_anomaly_od(
        t, x['tpa'], x['p'], x['ex'], x['ey'], x['ez'], x['w'], *s, True),
    'true_anomaly_od_tp': lambda t, s, x: true_anomaly_od(
        t, x['tpa'], x['p'], x['ex'], x['ey'], x['ez'], x['w'], *s, False),
    'lambert_phase_curve_od': lambda t, s, x: lambert_phase_curve_od(
        t, x['ag'], x['k'], x['tpa'], x['p'], *s),
    'ev_signal_od': lambda t, s, x: ev_signal_od(
        x['al'], x['mq'], x['inc'], t, x['tpa'], x['p'], *s),
    'emission_phase_curve_od': lambda t, s, x: emission_phase_curve_od(
        t, x['k'], x['fr'], x['off'], x['tpa'], x['p'], *s),
}


class TestScalarGradients:
    @pytest.mark.parametrize('name', sorted(SCALAR_GRAD_KERNELS))
    def test_parity(self, program, orbit, name):
        times, tpa, dt, ep_table, ep_times, coeffs, dcoeffs = setup_orbit(orbit)
        x = extra_pars(orbit, tpa, dt)
        ng = SCALAR_GRAD_KERNELS[name][2]
        v_cl, g_cl = run_grad(program, name, times, scalar_values(name, x),
                              ep_table, ep_times, coeffs, dcoeffs, ng)
        dispatch = (dt, ep_table, ep_times, coeffs, dcoeffs)
        v_nb, g_nb = NUMBA_REF[name](times, dispatch, x)
        np.testing.assert_allclose(v_cl, v_nb, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(g_cl, g_nb, rtol=RTOL, atol=ATOL)


class TestVectorGradients:
    @pytest.mark.parametrize('name,ref', [('pos_od', pos_od), ('vel_od', vel_od)])
    def test_parity(self, program, orbit, name, ref):
        times, tpa, dt, ep_table, ep_times, coeffs, dcoeffs = setup_orbit(orbit)
        p = orbit[0]
        v_cl, g_cl = run_grad(program, name, times, [tpa, p, dt],
                              ep_table, ep_times, coeffs, dcoeffs, 21, n_val=3)
        v_cl = v_cl.reshape(-1, 3)
        g_cl = g_cl.reshape(-1, 3, 7)
        x, y, z, dx, dy, dz = ref(times, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)
        np.testing.assert_allclose(v_cl, np.stack([x, y, z], axis=1), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(g_cl, np.stack([dx, dy, dz], axis=1), rtol=RTOL, atol=ATOL)


class TestRVKSlot:
    def test_zero_k_zeroes_k_slot(self, program, orbit):
        """rv_od must write all 8 slots even when k = 0 (no division)."""
        times, tpa, dt, ep_table, ep_times, coeffs, dcoeffs = setup_orbit(orbit)
        p, a, i, e, w = orbit
        scalars = [0.0, tpa, p, a, i, e, dt]
        v_cl, g_cl = run_grad(program, 'rv_od', times, scalars,
                              ep_table, ep_times, coeffs, dcoeffs, 8)
        v_nb, g_nb = rv_od(times, 0.0, tpa, p, a, i, e, dt,
                           ep_table, ep_times, coeffs, dcoeffs)
        np.testing.assert_allclose(v_cl, v_nb, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(g_cl, g_nb, rtol=RTOL, atol=ATOL)
        assert np.all(g_cl[:, 7] == 0.0)


class TestLightTravelTime:
    def test_transit_centre_basis(self, program, orbit):
        times, tpa, dt, ep_table, ep_times, coeffs, dcoeffs = setup_orbit(orbit)
        p, a, i, e, w = orbit
        scalars = [tpa, p, e, w, RSTAR, dt]
        v_cl, g_cl = run_grad(program, 'light_travel_time_od', times, scalars,
                              ep_table, ep_times, coeffs, dcoeffs, 7,
                              int_scalars=[1])
        v_nb, g_nb = light_travel_time_od(times, tpa, p, e, w, RSTAR, dt,
                                          ep_table, ep_times, coeffs, dcoeffs,
                                          timing_is_tc=True)
        np.testing.assert_allclose(v_cl, v_nb, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(g_cl, g_nb, rtol=RTOL, atol=ATOL)

    def test_periastron_basis(self, program, orbit):
        """timing_is_tc = 0 with a tc_to_tp_gradient-converted dcoeffs."""
        times, tpa, dt, ep_table, ep_times, coeffs, dcoeffs = setup_orbit(orbit)
        p, a, i, e, w = orbit
        dcoeffs_tp = np.stack([tc_to_tp_gradient(dcoeffs[j], p, e, w)
                               for j in range(dcoeffs.shape[0])])
        scalars = [tpa, p, e, w, RSTAR, dt]
        v_cl, g_cl = run_grad(program, 'light_travel_time_od', times, scalars,
                              ep_table, ep_times, coeffs, dcoeffs_tp, 7,
                              int_scalars=[0])
        v_nb, g_nb = light_travel_time_od(times, tpa, p, e, w, RSTAR, dt,
                                          ep_table, ep_times, coeffs, dcoeffs_tp,
                                          timing_is_tc=False)
        np.testing.assert_allclose(v_cl, v_nb, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(g_cl, g_nb, rtol=RTOL, atol=ATOL)


class TestTrueAnomalyBranches:
    @pytest.mark.parametrize('name, timing_is_tc', [('true_anomaly_od', True),
                                                    ('true_anomaly_od_tp', False)])
    def test_circular_fast_path(self, program, test_orbital_params, name, timing_is_tc):
        """The fast path's gradient depends on the declared basis (w = 0.7 so the
        transit-centre e and w slots are non-zero)."""
        pars = dict(test_orbital_params['circular'], w=0.7)
        orbit = (pars['p'], pars['a'], pars['i'], pars['e'], pars['w'])
        times, tpa, dt, ep_table, ep_times, coeffs, dcoeffs = setup_orbit(orbit)
        x = extra_pars(orbit, tpa, dt)
        x.update(ex=-1.0, ey=0.0, ez=0.0)
        v_cl, g_cl = run_grad(program, name, times,
                              scalar_values(name, x),
                              ep_table, ep_times, coeffs, dcoeffs, 7)
        v_nb, g_nb = true_anomaly_od(times, tpa, pars['p'], -1.0, 0.0, 0.0,
                                     pars['w'], dt, ep_table, ep_times,
                                     coeffs, dcoeffs, timing_is_tc)
        assert np.any(g_nb[:, 5] != 0.0) == timing_is_tc
        np.testing.assert_allclose(v_cl, v_nb, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(g_cl, g_nb, rtol=RTOL, atol=ATOL)
