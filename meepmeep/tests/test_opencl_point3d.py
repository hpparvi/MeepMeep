"""Parity tests for the OpenCL 3D single-expansion-point value evaluators.

Compares every quantity in point3d.cl against the numba reference in
`meepmeep.numba3d`, centred and direct, over near-transit and multi-epoch
times. The ``__kernel`` wrappers are generated from a small table and are
test-only; the shipped backend contains device functions only.
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

from meepmeep.numba3d import (solve3d, pos_c, pos, zpos_c, zpos, sep_c, sep,
                              vel_c, vel, zvel_c, zvel, rv_c, rv,
                              cos_alpha_c, cos_alpha,
                              lambert_phase_curve_c, lambert_phase_curve,
                              ev_signal_c, ev_signal,
                              emission_phase_curve_c, emission_phase_curve)
from meepmeep.tests.opencl_utils import (build, get_queue, has_fp64, upload,
                                         output_buffer, read_back)

RTOL = 1e-12
ATOL = 1e-13

TC = 1.5

# Scalar-returning quantities: kernel name -> (device call, device scalar names).
# The scalar values are passed to the kernel in the listed order; names avoid
# colliding with the work-item index `i`.
SCALAR_KERNELS = {
    'zpos_c': ('zpos_c3(t[i], c)', ()),
    'zpos': ('zpos3(t[i], tc, p, c, te)', ('tc', 'p', 'te')),
    'sep_c': ('sep_c3(t[i], c)', ()),
    'sep': ('sep3(t[i], tc, p, c, te)', ('tc', 'p', 'te')),
    'zvel_c': ('zvel_c3(t[i], c)', ()),
    'zvel': ('zvel3(t[i], tc, p, c, te)', ('tc', 'p', 'te')),
    'cos_alpha_c': ('cos_alpha_c3(t[i], c)', ()),
    'cos_alpha': ('cos_alpha3(t[i], tc, p, c, te)', ('tc', 'p', 'te')),
    'rv_c': ('rv_c3(t[i], k, p, aa, inc, e, c)', ('k', 'p', 'aa', 'inc', 'e')),
    'rv': ('rv3(t[i], k, tc, p, aa, inc, e, c, te)', ('k', 'tc', 'p', 'aa', 'inc', 'e', 'te')),
    'lambert_phase_curve_c': ('lambert_phase_curve_c3(t[i], ag, k, c)', ('ag', 'k')),
    'lambert_phase_curve': ('lambert_phase_curve3(t[i], ag, k, tc, p, c, te)',
                            ('ag', 'k', 'tc', 'p', 'te')),
    'ev_signal_c': ('ev_signal_c3(t[i], al, mq, inc, c)', ('al', 'mq', 'inc')),
    'ev_signal': ('ev_signal3(t[i], al, mq, inc, tc, p, c, te)',
                  ('al', 'mq', 'inc', 'tc', 'p', 'te')),
    'emission_phase_curve_c': ('emission_phase_curve_c3(t[i], k, fr, off, c)',
                               ('k', 'fr', 'off')),
    'emission_phase_curve': ('emission_phase_curve3(t[i], k, fr, off, tc, p, c, te)',
                             ('k', 'fr', 'off', 'tc', 'p', 'te')),
}


def _scalar_kernel_source(name, call, scalar_names):
    decls = ''.join(f'const REAL {s}, ' for s in scalar_names)
    return (f"__kernel void k_{name}(__global const REAL *t, {decls}"
            f"__global const REAL *c, __global REAL *out) {{\n"
            f"    int i = get_global_id(0);\n"
            f"    out[i] = {call};\n"
            f"}}\n")


VECTOR_KERNELS = """
__kernel void k_pos_c(__global const REAL *t, __global const REAL *c,
                      __global REAL *ox, __global REAL *oy, __global REAL *oz) {
    int i = get_global_id(0);
    REAL x, y, z;
    pos_c3(t[i], c, &x, &y, &z);
    ox[i] = x; oy[i] = y; oz[i] = z;
}

__kernel void k_pos(__global const REAL *t, const REAL tc, const REAL p,
                    const REAL te, __global const REAL *c,
                    __global REAL *ox, __global REAL *oy, __global REAL *oz) {
    int i = get_global_id(0);
    REAL x, y, z;
    pos3(t[i], tc, p, c, te, &x, &y, &z);
    ox[i] = x; oy[i] = y; oz[i] = z;
}

__kernel void k_vel_c(__global const REAL *t, __global const REAL *c,
                      __global REAL *ox, __global REAL *oy, __global REAL *oz) {
    int i = get_global_id(0);
    REAL x, y, z;
    vel_c3(t[i], c, &x, &y, &z);
    ox[i] = x; oy[i] = y; oz[i] = z;
}

__kernel void k_vel(__global const REAL *t, const REAL tc, const REAL p,
                    const REAL te, __global const REAL *c,
                    __global REAL *ox, __global REAL *oy, __global REAL *oz) {
    int i = get_global_id(0);
    REAL x, y, z;
    vel3(t[i], tc, p, c, te, &x, &y, &z);
    ox[i] = x; oy[i] = y; oz[i] = z;
}
"""

TEST_KERNELS = VECTOR_KERNELS + ''.join(
    _scalar_kernel_source(name, call, scalars)
    for name, (call, scalars) in SCALAR_KERNELS.items())


@pytest.fixture(scope='module')
def program():
    if not has_fp64():
        pytest.skip("Device lacks cl_khr_fp64")
    return build(TEST_KERNELS, 'point3d.cl')


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


def run_scalar(program, name, times, scalars, c):
    queue = get_queue()
    b_t, b_c = upload(queue, times), upload(queue, c)
    b_o, h_o = output_buffer(queue, times.size)
    kernel = cl.Kernel(program, f'k_{name}')
    kernel(queue, (times.size,), None, b_t, *[np.float64(s) for s in scalars], b_c, b_o)
    return read_back(queue, b_o, h_o)


def run_vector(program, name, times, scalars, c):
    queue = get_queue()
    b_t, b_c = upload(queue, times), upload(queue, c)
    outs = [output_buffer(queue, times.size) for _ in range(3)]
    kernel = cl.Kernel(program, f'k_{name}')
    kernel(queue, (times.size,), None, b_t, *[np.float64(s) for s in scalars], b_c,
           *[b for b, _ in outs])
    return [read_back(queue, b, h) for b, h in outs]


# numba reference calls, keyed like SCALAR_KERNELS. `x` is the extra_pars dict.
NUMBA_REF = {
    'zpos_c': lambda t, c, x: zpos_c(t, c),
    'zpos': lambda t, c, x, te: zpos(t, x['tc'], x['p'], c, te),
    'sep_c': lambda t, c, x: sep_c(t, c),
    'sep': lambda t, c, x, te: sep(t, x['tc'], x['p'], c, te),
    'zvel_c': lambda t, c, x: zvel_c(t, c),
    'zvel': lambda t, c, x, te: zvel(t, x['tc'], x['p'], c, te),
    'cos_alpha_c': lambda t, c, x: cos_alpha_c(t, c),
    'cos_alpha': lambda t, c, x, te: cos_alpha(t, x['tc'], x['p'], c, te),
    'rv_c': lambda t, c, x: rv_c(t, 25.0, x['p'], x['aa'], x['inc'], x['e'], c),
    'rv': lambda t, c, x, te: rv(t, 25.0, x['tc'], x['p'], x['aa'], x['inc'], x['e'], c, te),
    'lambert_phase_curve_c': lambda t, c, x: lambert_phase_curve_c(t, x['ag'], x['k'], c),
    'lambert_phase_curve': lambda t, c, x, te: lambert_phase_curve(
        t, x['ag'], x['k'], x['tc'], x['p'], c, te),
    'ev_signal_c': lambda t, c, x: ev_signal_c(t, x['al'], x['mq'], x['inc'], c),
    'ev_signal': lambda t, c, x, te: ev_signal(
        t, x['al'], x['mq'], x['inc'], x['tc'], x['p'], c, te),
    'emission_phase_curve_c': lambda t, c, x: emission_phase_curve_c(
        t, x['k'], x['fr'], x['off'], c),
    'emission_phase_curve': lambda t, c, x, te: emission_phase_curve(
        t, x['k'], x['fr'], x['off'], x['tc'], x['p'], c, te),
}


def scalar_values(name, x, te):
    """Values for the device kernel's scalar arguments, in declaration order."""
    _, names = SCALAR_KERNELS[name]
    values = {'k': 25.0 if name in ('rv_c', 'rv') else x['k'], 'te': te, **{
        n: x[n] for n in ('tc', 'p', 'aa', 'inc', 'e', 'ag', 'al', 'mq', 'fr', 'off')}}
    return [values[n] for n in names]


CENTERED_NAMES = [n for n in SCALAR_KERNELS if n.endswith('_c') or n == 'rv_c']
DIRECT_NAMES = [n for n in SCALAR_KERNELS if n not in CENTERED_NAMES]


class TestCenteredScalars:
    @pytest.mark.parametrize('name', CENTERED_NAMES)
    def test_parity(self, program, orbit, name):
        p, a, i, e, w = orbit
        x = extra_pars(orbit)
        c = solve3d(0.0, p, a, i, e, w)
        times = centered_times()
        out_cl = run_scalar(program, name, times, scalar_values(name, x, 0.0), c)
        out_nb = NUMBA_REF[name](times, c, x)
        np.testing.assert_allclose(out_cl, out_nb, rtol=RTOL, atol=ATOL)


class TestDirectScalars:
    @pytest.mark.parametrize('name', DIRECT_NAMES)
    @pytest.mark.parametrize('te', [0.0, 0.02])
    def test_parity(self, program, orbit, name, te):
        p, a, i, e, w = orbit
        x = extra_pars(orbit)
        c = solve3d(te, p, a, i, e, w)
        times = multi_epoch_times(TC, p, te)
        out_cl = run_scalar(program, name, times, scalar_values(name, x, te), c)
        out_nb = NUMBA_REF[name](times, c, x, te)
        np.testing.assert_allclose(out_cl, out_nb, rtol=RTOL, atol=ATOL)


class TestVectorQuantities:
    def test_pos_c(self, program, orbit):
        p, a, i, e, w = orbit
        c = solve3d(0.0, p, a, i, e, w)
        times = centered_times()
        cl_xyz = run_vector(program, 'pos_c', times, [], c)
        nb_xyz = pos_c(times, c)
        for got, want in zip(cl_xyz, nb_xyz):
            np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)

    def test_vel_c(self, program, orbit):
        p, a, i, e, w = orbit
        c = solve3d(0.0, p, a, i, e, w)
        times = centered_times()
        cl_xyz = run_vector(program, 'vel_c', times, [], c)
        nb_xyz = vel_c(times, c)
        for got, want in zip(cl_xyz, nb_xyz):
            np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize('te', [0.0, 0.02])
    def test_pos(self, program, orbit, te):
        p, a, i, e, w = orbit
        c = solve3d(te, p, a, i, e, w)
        times = multi_epoch_times(TC, p, te)
        cl_xyz = run_vector(program, 'pos', times, [TC, p, te], c)
        nb_xyz = pos(times, TC, p, c, te)
        for got, want in zip(cl_xyz, nb_xyz):
            np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize('te', [0.0, 0.02])
    def test_vel(self, program, orbit, te):
        p, a, i, e, w = orbit
        c = solve3d(te, p, a, i, e, w)
        times = multi_epoch_times(TC, p, te)
        cl_xyz = run_vector(program, 'vel', times, [TC, p, te], c)
        nb_xyz = vel(times, TC, p, c, te)
        for got, want in zip(cl_xyz, nb_xyz):
            np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)
