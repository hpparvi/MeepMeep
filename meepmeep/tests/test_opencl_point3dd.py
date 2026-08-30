"""Parity tests for the OpenCL 3D single-expansion-point gradient evaluators.

Compares every quantity in point3dd.cl against the numba reference in
`meepmeep.numba3d`, centred and direct, including the extended gradient
layouts (9 slots for lambert and ev_signal, 10 for emission) and the
hoisted rv_scale/rv_cd_w path. Multi-epoch times make the period-folding
chain term observable in the direct tests.
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

from meepmeep.numba3d import (solve3d_d, pos_cd, pos_d, zpos_cd, zpos_d,
                              sep_cd, sep_d, vel_cd, vel_d, zvel_cd, zvel_d,
                              rv_cd, rv_d, cos_alpha_cd, cos_alpha_d,
                              lambert_phase_curve_cd, lambert_phase_curve_d,
                              ev_signal_cd, ev_signal_d,
                              emission_phase_curve_cd, emission_phase_curve_d)
from meepmeep.tests.opencl_utils import (build, get_queue, has_fp64, upload,
                                         output_buffer, read_back)

RTOL = 1e-12
ATOL = 1e-13

TC = 1.5
K_RV = 25.0

# name -> (device call, device scalar names, gradient length)
SCALAR_GRAD_KERNELS = {
    'zpos_cd': ('zpos_cd3(t[i], c, dc, g)', (), 7),
    'zpos_d': ('zpos_d3(t[i], tc, p, c, dc, te, g)', ('tc', 'p', 'te'), 7),
    'sep_cd': ('sep_cd3(t[i], c, dc, g)', (), 7),
    'sep_d': ('sep_d3(t[i], tc, p, c, dc, te, g)', ('tc', 'p', 'te'), 7),
    'zvel_cd': ('zvel_cd3(t[i], c, dc, g)', (), 7),
    'zvel_d': ('zvel_d3(t[i], tc, p, c, dc, te, g)', ('tc', 'p', 'te'), 7),
    'cos_alpha_cd': ('cos_alpha_cd3(t[i], c, dc, g)', (), 7),
    'cos_alpha_d': ('cos_alpha_d3(t[i], tc, p, c, dc, te, g)', ('tc', 'p', 'te'), 7),
    'rv_cd': ('rv_cd3(t[i], k, p, aa, inc, e, c, dc, g)',
              ('k', 'p', 'aa', 'inc', 'e'), 7),
    'rv_d': ('rv_d3(t[i], k, tc, p, aa, inc, e, c, dc, te, g)',
             ('k', 'tc', 'p', 'aa', 'inc', 'e', 'te'), 7),
    'lambert_phase_curve_cd': ('lambert_phase_curve_cd3(t[i], ag, k, c, dc, g)',
                               ('ag', 'k'), 9),
    'lambert_phase_curve_d': ('lambert_phase_curve_d3(t[i], ag, k, tc, p, c, dc, te, g)',
                              ('ag', 'k', 'tc', 'p', 'te'), 9),
    'ev_signal_cd': ('ev_signal_cd3(t[i], al, mq, inc, c, dc, g)',
                     ('al', 'mq', 'inc'), 9),
    'ev_signal_d': ('ev_signal_d3(t[i], al, mq, inc, tc, p, c, dc, te, g)',
                    ('al', 'mq', 'inc', 'tc', 'p', 'te'), 9),
    'emission_phase_curve_cd': ('emission_phase_curve_cd3(t[i], k, fr, off, c, dc, g)',
                                ('k', 'fr', 'off'), 10),
    'emission_phase_curve_d': ('emission_phase_curve_d3(t[i], k, fr, off, tc, p, c, dc, te, g)',
                               ('k', 'fr', 'off', 'tc', 'p', 'te'), 10),
}


def _scalar_grad_kernel_source(name, call, scalar_names, ng):
    decls = ''.join(f'const REAL {s}, ' for s in scalar_names)
    return (f"__kernel void k_{name}(__global const REAL *t, {decls}"
            f"__global const REAL *c, __global const REAL *dc, "
            f"__global REAL *val, __global REAL *grad) {{\n"
            f"    int i = get_global_id(0);\n"
            f"    REAL g[{ng}];\n"
            f"    val[i] = {call};\n"
            f"    for (int m = 0; m < {ng}; m++)\n"
            f"        grad[{ng} * i + m] = g[m];\n"
            f"}}\n")


VECTOR_GRAD_KERNELS = """
__kernel void k_pos_cd(__global const REAL *t, __global const REAL *c,
                       __global const REAL *dc,
                       __global REAL *val, __global REAL *grad) {
    int i = get_global_id(0);
    REAL x, y, z, dx[7], dy[7], dz[7];
    pos_cd3(t[i], c, dc, &x, &y, &z, dx, dy, dz);
    val[3 * i] = x; val[3 * i + 1] = y; val[3 * i + 2] = z;
    for (int m = 0; m < 7; m++) {
        grad[21 * i + m] = dx[m];
        grad[21 * i + 7 + m] = dy[m];
        grad[21 * i + 14 + m] = dz[m];
    }
}

__kernel void k_pos_d(__global const REAL *t, const REAL tc, const REAL p,
                      const REAL te, __global const REAL *c,
                      __global const REAL *dc,
                      __global REAL *val, __global REAL *grad) {
    int i = get_global_id(0);
    REAL x, y, z, dx[7], dy[7], dz[7];
    pos_d3(t[i], tc, p, c, dc, te, &x, &y, &z, dx, dy, dz);
    val[3 * i] = x; val[3 * i + 1] = y; val[3 * i + 2] = z;
    for (int m = 0; m < 7; m++) {
        grad[21 * i + m] = dx[m];
        grad[21 * i + 7 + m] = dy[m];
        grad[21 * i + 14 + m] = dz[m];
    }
}

__kernel void k_vel_cd(__global const REAL *t, __global const REAL *c,
                       __global const REAL *dc,
                       __global REAL *val, __global REAL *grad) {
    int i = get_global_id(0);
    REAL x, y, z, dx[7], dy[7], dz[7];
    vel_cd3(t[i], c, dc, &x, &y, &z, dx, dy, dz);
    val[3 * i] = x; val[3 * i + 1] = y; val[3 * i + 2] = z;
    for (int m = 0; m < 7; m++) {
        grad[21 * i + m] = dx[m];
        grad[21 * i + 7 + m] = dy[m];
        grad[21 * i + 14 + m] = dz[m];
    }
}

__kernel void k_vel_d(__global const REAL *t, const REAL tc, const REAL p,
                      const REAL te, __global const REAL *c,
                      __global const REAL *dc,
                      __global REAL *val, __global REAL *grad) {
    int i = get_global_id(0);
    REAL x, y, z, dx[7], dy[7], dz[7];
    vel_d3(t[i], tc, p, c, dc, te, &x, &y, &z, dx, dy, dz);
    val[3 * i] = x; val[3 * i + 1] = y; val[3 * i + 2] = z;
    for (int m = 0; m < 7; m++) {
        grad[21 * i + m] = dx[m];
        grad[21 * i + 7 + m] = dy[m];
        grad[21 * i + 14 + m] = dz[m];
    }
}

__kernel void k_rv_hoisted(__global const REAL *t, const REAL k, const REAL p,
                           const REAL aa, const REAL inc, const REAL e,
                           __global const REAL *c, __global const REAL *dc,
                           __global REAL *val, __global REAL *grad) {
    int i = get_global_id(0);
    REAL dsp, dsa, dsi, dse, g[7];
    REAL s = rv_scale(k, p, aa, inc, e, &dsp, &dsa, &dsi, &dse);
    val[i] = rv_cd_w(t[i], s, dsp, dsa, dsi, dse, c, dc, g);
    for (int m = 0; m < 7; m++)
        grad[7 * i + m] = g[m];
}
"""

TEST_KERNELS = VECTOR_GRAD_KERNELS + ''.join(
    _scalar_grad_kernel_source(name, call, scalars, ng)
    for name, (call, scalars, ng) in SCALAR_GRAD_KERNELS.items())


@pytest.fixture(scope='module')
def program():
    if not has_fp64():
        pytest.skip("Device lacks cl_khr_fp64")
    return build(TEST_KERNELS, 'point3dd.cl')


@pytest.fixture(params=['circular', 'eccentric'])
def orbit(request, test_orbital_params):
    pars = test_orbital_params[request.param]
    return pars['p'], pars['a'], pars['i'], pars['e'], pars['w']


def extra_pars(orbit):
    p, a, i, e, w = orbit
    return {'p': p, 'aa': a, 'inc': i, 'e': e, 'tc': TC,
            'k': 0.1, 'ag': 0.3, 'al': 1.2, 'mq': 1e-3, 'fr': 1e-3, 'off': 0.3}


def centered_times():
    return np.linspace(-0.1, 0.1, 101)


def multi_epoch_times(tc, p, te):
    offsets = np.linspace(-0.1, 0.1, 11)
    return np.concatenate([tc + te + k * p + offsets for k in range(-3, 4)])


def scalar_values(name, x, te):
    _, names, _ = SCALAR_GRAD_KERNELS[name]
    values = {'k': K_RV if name.startswith('rv') else x['k'], 'te': te, **{
        n: x[n] for n in ('tc', 'p', 'aa', 'inc', 'e', 'ag', 'al', 'mq', 'fr', 'off')}}
    return [values[n] for n in names]


def run_scalar_grad(program, name, times, scalars, c, dc, ng):
    queue = get_queue()
    n = times.size
    b_t, b_c, b_dc = upload(queue, times), upload(queue, c), upload(queue, dc)
    b_v, h_v = output_buffer(queue, n)
    b_g, h_g = output_buffer(queue, ng * n)
    kernel = cl.Kernel(program, f'k_{name}')
    kernel(queue, (n,), None, b_t, *[np.float64(s) for s in scalars], b_c, b_dc, b_v, b_g)
    return read_back(queue, b_v, h_v), read_back(queue, b_g, h_g).reshape(n, ng)


def run_vector_grad(program, name, times, scalars, c, dc):
    queue = get_queue()
    n = times.size
    b_t, b_c, b_dc = upload(queue, times), upload(queue, c), upload(queue, dc)
    b_v, h_v = output_buffer(queue, 3 * n)
    b_g, h_g = output_buffer(queue, 21 * n)
    kernel = cl.Kernel(program, f'k_{name}')
    kernel(queue, (n,), None, b_t, *[np.float64(s) for s in scalars], b_c, b_dc, b_v, b_g)
    vals = read_back(queue, b_v, h_v).reshape(n, 3)
    grads = read_back(queue, b_g, h_g).reshape(n, 3, 7)
    return vals, grads


NUMBA_REF = {
    'zpos_cd': lambda t, c, dc, x: zpos_cd(t, c, dc),
    'zpos_d': lambda t, c, dc, x, te: zpos_d(t, x['tc'], x['p'], c, dc, te),
    'sep_cd': lambda t, c, dc, x: sep_cd(t, c, dc),
    'sep_d': lambda t, c, dc, x, te: sep_d(t, x['tc'], x['p'], c, dc, te),
    'zvel_cd': lambda t, c, dc, x: zvel_cd(t, c, dc),
    'zvel_d': lambda t, c, dc, x, te: zvel_d(t, x['tc'], x['p'], c, dc, te),
    'cos_alpha_cd': lambda t, c, dc, x: cos_alpha_cd(t, c, dc),
    'cos_alpha_d': lambda t, c, dc, x, te: cos_alpha_d(t, x['tc'], x['p'], c, dc, te),
    'rv_cd': lambda t, c, dc, x: rv_cd(t, K_RV, x['p'], x['aa'], x['inc'], x['e'], c, dc),
    'rv_d': lambda t, c, dc, x, te: rv_d(
        t, K_RV, x['tc'], x['p'], x['aa'], x['inc'], x['e'], c, dc, te),
    'lambert_phase_curve_cd': lambda t, c, dc, x: lambert_phase_curve_cd(
        t, x['ag'], x['k'], c, dc),
    'lambert_phase_curve_d': lambda t, c, dc, x, te: lambert_phase_curve_d(
        t, x['ag'], x['k'], x['tc'], x['p'], c, dc, te),
    'ev_signal_cd': lambda t, c, dc, x: ev_signal_cd(t, x['al'], x['mq'], x['inc'], c, dc),
    'ev_signal_d': lambda t, c, dc, x, te: ev_signal_d(
        t, x['al'], x['mq'], x['inc'], x['tc'], x['p'], c, dc, te),
    'emission_phase_curve_cd': lambda t, c, dc, x: emission_phase_curve_cd(
        t, x['k'], x['fr'], x['off'], c, dc),
    'emission_phase_curve_d': lambda t, c, dc, x, te: emission_phase_curve_d(
        t, x['k'], x['fr'], x['off'], x['tc'], x['p'], c, dc, te),
}

CENTERED_NAMES = [n for n in SCALAR_GRAD_KERNELS if n.endswith('_cd')]
DIRECT_NAMES = [n for n in SCALAR_GRAD_KERNELS if n.endswith('_d') and n not in CENTERED_NAMES]


class TestCenteredGradients:
    @pytest.mark.parametrize('name', CENTERED_NAMES)
    def test_parity(self, program, orbit, name):
        p, a, i, e, w = orbit
        x = extra_pars(orbit)
        c, dc = solve3d_d(0.0, p, a, i, e, w)
        times = centered_times()
        ng = SCALAR_GRAD_KERNELS[name][2]
        v_cl, g_cl = run_scalar_grad(program, name, times,
                                     scalar_values(name, x, 0.0), c, dc, ng)
        v_nb, g_nb = NUMBA_REF[name](times, c, dc, x)
        np.testing.assert_allclose(v_cl, v_nb, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(g_cl, g_nb, rtol=RTOL, atol=ATOL)


class TestDirectGradients:
    """Multi-epoch times make the d[1] += epoch*d[0] chain term observable."""

    @pytest.mark.parametrize('name', DIRECT_NAMES)
    @pytest.mark.parametrize('te', [0.0, 0.02])
    def test_parity(self, program, orbit, name, te):
        p, a, i, e, w = orbit
        x = extra_pars(orbit)
        c, dc = solve3d_d(te, p, a, i, e, w)
        times = multi_epoch_times(TC, p, te)
        ng = SCALAR_GRAD_KERNELS[name][2]
        v_cl, g_cl = run_scalar_grad(program, name, times,
                                     scalar_values(name, x, te), c, dc, ng)
        v_nb, g_nb = NUMBA_REF[name](times, c, dc, x, te)
        np.testing.assert_allclose(v_cl, v_nb, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(g_cl, g_nb, rtol=RTOL, atol=ATOL)


class TestVectorGradients:
    def test_pos_cd(self, program, orbit):
        p, a, i, e, w = orbit
        c, dc = solve3d_d(0.0, p, a, i, e, w)
        times = centered_times()
        v_cl, g_cl = run_vector_grad(program, 'pos_cd', times, [], c, dc)
        px, py, pz, dpx, dpy, dpz = pos_cd(times, c, dc)
        np.testing.assert_allclose(v_cl, np.stack([px, py, pz], axis=1), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(g_cl, np.stack([dpx, dpy, dpz], axis=1), rtol=RTOL, atol=ATOL)

    def test_vel_cd(self, program, orbit):
        p, a, i, e, w = orbit
        c, dc = solve3d_d(0.0, p, a, i, e, w)
        times = centered_times()
        v_cl, g_cl = run_vector_grad(program, 'vel_cd', times, [], c, dc)
        vx, vy, vz, dvx, dvy, dvz = vel_cd(times, c, dc)
        np.testing.assert_allclose(v_cl, np.stack([vx, vy, vz], axis=1), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(g_cl, np.stack([dvx, dvy, dvz], axis=1), rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize('te', [0.0, 0.02])
    def test_pos_d(self, program, orbit, te):
        p, a, i, e, w = orbit
        c, dc = solve3d_d(te, p, a, i, e, w)
        times = multi_epoch_times(TC, p, te)
        v_cl, g_cl = run_vector_grad(program, 'pos_d', times, [TC, p, te], c, dc)
        px, py, pz, dpx, dpy, dpz = pos_d(times, TC, p, c, dc, te)
        np.testing.assert_allclose(v_cl, np.stack([px, py, pz], axis=1), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(g_cl, np.stack([dpx, dpy, dpz], axis=1), rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize('te', [0.0, 0.02])
    def test_vel_d(self, program, orbit, te):
        p, a, i, e, w = orbit
        c, dc = solve3d_d(te, p, a, i, e, w)
        times = multi_epoch_times(TC, p, te)
        v_cl, g_cl = run_vector_grad(program, 'vel_d', times, [TC, p, te], c, dc)
        vx, vy, vz, dvx, dvy, dvz = vel_d(times, TC, p, c, dc, te)
        np.testing.assert_allclose(v_cl, np.stack([vx, vy, vz], axis=1), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(g_cl, np.stack([dvx, dvy, dvz], axis=1), rtol=RTOL, atol=ATOL)


class TestRVHoistedPath:
    """rv_scale + rv_cd_w must equal the self-contained rv_cd."""

    def test_matches_rv_cd(self, program, orbit):
        p, a, i, e, w = orbit
        x = extra_pars(orbit)
        c, dc = solve3d_d(0.0, p, a, i, e, w)
        times = centered_times()
        queue = get_queue()
        n = times.size
        b_t, b_c, b_dc = upload(queue, times), upload(queue, c), upload(queue, dc)
        b_v, h_v = output_buffer(queue, n)
        b_g, h_g = output_buffer(queue, 7 * n)
        kernel = cl.Kernel(program, 'k_rv_hoisted')
        kernel(queue, (n,), None, b_t, np.float64(K_RV), np.float64(p),
               np.float64(a), np.float64(i), np.float64(e), b_c, b_dc, b_v, b_g)
        v_cl = read_back(queue, b_v, h_v)
        g_cl = read_back(queue, b_g, h_g).reshape(n, 7)
        v_nb, g_nb = rv_cd(times, K_RV, p, a, i, e, c, dc)
        np.testing.assert_allclose(v_cl, v_nb, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(g_cl, g_nb, rtol=RTOL, atol=ATOL)
