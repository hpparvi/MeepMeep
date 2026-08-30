"""Parity tests for the OpenCL multi-expansion-point value evaluators.

Builds the expansion-point machinery on the host (`create_expansion_points`
+ `solve3d_orbit`), uploads the flattened coefficient stack and the int32
`ep_table`, and compares every quantity in orbit3d.cl against the numba
reference over times spanning several periods on both sides of the
periastron anchor (negative epochs exercise floor()'s rounding). The
``__kernel`` wrappers are test-only.
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
from meepmeep.backends.numba.orbit3d import (solve3d_orbit, pos_o, zpos_o, sep_o,
                                             vel_o, zvel_o, rv_o, cos_alpha_o,
                                             cos_v_p_angle_o, true_anomaly_o,
                                             lambert_phase_curve_o, ev_signal_o,
                                             emission_phase_curve_o,
                                             star_planet_distance_o,
                                             light_travel_time_o)
from meepmeep.backends.numba.orbit3d._common import ep_ix
from meepmeep.backends.numba.utils import (TWO_PI, mean_anomaly_at_transit,
                                           eccentricity_vector)
from meepmeep.tests.opencl_utils import (build, get_queue, has_fp64, upload,
                                         upload_ep_table, output_buffer, read_back)

RTOL = 1e-12
ATOL = 1e-13

NPT = 15
V_FIXED = (0.3, -0.5, 0.8)
RSTAR = 2.0

# name -> (device call, device scalar names)
SCALAR_KERNELS = {
    'zpos_o': ('zpos_o(t[i], tpa, p, dt, ep_table, ep_times, coeffs)',
               ('tpa', 'p', 'dt')),
    'sep_o': ('sep_o(t[i], tpa, p, dt, ep_table, ep_times, coeffs)',
              ('tpa', 'p', 'dt')),
    'zvel_o': ('zvel_o(t[i], tpa, p, dt, ep_table, ep_times, coeffs)',
               ('tpa', 'p', 'dt')),
    'cos_alpha_o': ('cos_alpha_o(t[i], tpa, p, dt, ep_table, ep_times, coeffs)',
                    ('tpa', 'p', 'dt')),
    'star_planet_distance_o': (
        'star_planet_distance_o(t[i], tpa, p, dt, ep_table, ep_times, coeffs)',
        ('tpa', 'p', 'dt')),
    'rv_o': ('rv_o(t[i], k, tpa, p, aa, inc, e, dt, ep_table, ep_times, coeffs)',
             ('k', 'tpa', 'p', 'aa', 'inc', 'e', 'dt')),
    'cos_v_p_angle_o': (
        'cos_v_p_angle_o(vx, vy, vz, t[i], tpa, p, dt, ep_table, ep_times, coeffs)',
        ('vx', 'vy', 'vz', 'tpa', 'p', 'dt')),
    'true_anomaly_o': (
        'true_anomaly_o(t[i], tpa, p, ex, ey, ez, w, dt, ep_table, ep_times, coeffs)',
        ('tpa', 'p', 'ex', 'ey', 'ez', 'w', 'dt')),
    'lambert_phase_curve_o': (
        'lambert_phase_curve_o(t[i], ag, k, tpa, p, dt, ep_table, ep_times, coeffs)',
        ('ag', 'k', 'tpa', 'p', 'dt')),
    'ev_signal_o': (
        'ev_signal_o(al, mq, inc, t[i], tpa, p, dt, ep_table, ep_times, coeffs)',
        ('al', 'mq', 'inc', 'tpa', 'p', 'dt')),
    'emission_phase_curve_o': (
        'emission_phase_curve_o(t[i], k, fr, off, tpa, p, dt, ep_table, ep_times, coeffs)',
        ('k', 'fr', 'off', 'tpa', 'p', 'dt')),
    'light_travel_time_o': (
        'light_travel_time_o(t[i], tpa, p, e, w, rstar, dt, ep_table, ep_times, coeffs)',
        ('tpa', 'p', 'e', 'w', 'rstar', 'dt')),
}


def _scalar_kernel_source(name, call, scalar_names):
    decls = ''.join(f'const REAL {s}, ' for s in scalar_names)
    return (f"__kernel void k_{name}(__global const REAL *t, {decls}"
            f"__global const int *ep_table, __global const REAL *ep_times, "
            f"__global const REAL *coeffs, __global REAL *out) {{\n"
            f"    int i = get_global_id(0);\n"
            f"    out[i] = {call};\n"
            f"}}\n")


EXTRA_KERNELS = """
__kernel void k_pos_o(__global const REAL *t, const REAL tpa, const REAL p,
                      const REAL dt, __global const int *ep_table,
                      __global const REAL *ep_times, __global const REAL *coeffs,
                      __global REAL *ox, __global REAL *oy, __global REAL *oz) {
    int i = get_global_id(0);
    REAL x, y, z;
    pos_o(t[i], tpa, p, dt, ep_table, ep_times, coeffs, &x, &y, &z);
    ox[i] = x; oy[i] = y; oz[i] = z;
}

__kernel void k_vel_o(__global const REAL *t, const REAL tpa, const REAL p,
                      const REAL dt, __global const int *ep_table,
                      __global const REAL *ep_times, __global const REAL *coeffs,
                      __global REAL *ox, __global REAL *oy, __global REAL *oz) {
    int i = get_global_id(0);
    REAL x, y, z;
    vel_o(t[i], tpa, p, dt, ep_table, ep_times, coeffs, &x, &y, &z);
    ox[i] = x; oy[i] = y; oz[i] = z;
}

__kernel void k_ep_ix(__global const REAL *t, const REAL tpa, const REAL p,
                      const REAL dt, __global const int *ep_table,
                      __global const REAL *ep_times, __global const REAL *coeffs,
                      __global REAL *out) {
    int i = get_global_id(0);
    out[i] = (REAL)ep_ix(t[i], tpa, p, dt, ep_table);
}
"""

TEST_KERNELS = EXTRA_KERNELS + ''.join(
    _scalar_kernel_source(name, call, scalars)
    for name, (call, scalars) in SCALAR_KERNELS.items())


@pytest.fixture(scope='module')
def program():
    if not has_fp64():
        pytest.skip("Device lacks cl_khr_fp64")
    return build(TEST_KERNELS, 'orbit3d.cl')


@pytest.fixture(params=['circular', 'eccentric'])
def orbit(request, test_orbital_params):
    pars = test_orbital_params[request.param]
    return pars['p'], pars['a'], pars['i'], pars['e'], pars['w']


def setup_orbit(orbit):
    """Host-side expansion-point machinery, mirroring test_orbit3d_evaluators."""
    p, a, i, e, w = orbit
    ep_times, _, dt, ep_table = create_expansion_points(NPT, max(e, 0.2), 'ea')
    coeffs = solve3d_orbit(ep_times, p, a, i, e, w, npt=NPT)
    tpa = -mean_anomaly_at_transit(e, w) / TWO_PI * p
    times = tpa + np.linspace(-2.5 * p, 3.5 * p, 400)
    return times, tpa, dt, ep_table, ep_times, coeffs


def extra_pars(orbit, tpa, dt):
    p, a, i, e, w = orbit
    ev = eccentricity_vector(i, e, w)
    return {'tpa': tpa, 'p': p, 'aa': a, 'inc': i, 'e': e, 'w': w, 'dt': dt,
            'k': 0.1, 'ag': 0.3, 'al': 1.2, 'mq': 1e-3, 'fr': 1e-3, 'off': 0.3,
            'rstar': RSTAR, 'vx': V_FIXED[0], 'vy': V_FIXED[1], 'vz': V_FIXED[2],
            'ex': ev[0], 'ey': ev[1], 'ez': ev[2]}


def scalar_values(name, x):
    _, names = SCALAR_KERNELS[name]
    values = dict(x)
    if name == 'rv_o':
        values['k'] = 25.0
    return [values[n] for n in names]


def run_kernel(program, name, times, scalars, ep_table, ep_times, coeffs, n_out=1):
    queue = get_queue()
    n = times.size
    b_t = upload(queue, times)
    b_tab = upload_ep_table(queue, ep_table)
    b_ept = upload(queue, ep_times)
    b_c = upload(queue, coeffs)
    outs = [output_buffer(queue, n) for _ in range(n_out)]
    kernel = cl.Kernel(program, f'k_{name}')
    kernel(queue, (n,), None, b_t, *[np.float64(s) for s in scalars],
           b_tab, b_ept, b_c, *[b for b, _ in outs])
    return [read_back(queue, b, h) for b, h in outs]


NUMBA_REF = {
    'zpos_o': lambda t, s, x: zpos_o(t, x['tpa'], x['p'], *s),
    'sep_o': lambda t, s, x: sep_o(t, x['tpa'], x['p'], *s),
    'zvel_o': lambda t, s, x: zvel_o(t, x['tpa'], x['p'], *s),
    'cos_alpha_o': lambda t, s, x: cos_alpha_o(t, x['tpa'], x['p'], *s),
    'star_planet_distance_o': lambda t, s, x: star_planet_distance_o(
        t, x['tpa'], x['p'], *s),
    'rv_o': lambda t, s, x: rv_o(t, 25.0, x['tpa'], x['p'], x['aa'], x['inc'], x['e'], *s),
    'cos_v_p_angle_o': lambda t, s, x: cos_v_p_angle_o(
        np.array(V_FIXED), t, x['tpa'], x['p'], *s),
    'true_anomaly_o': lambda t, s, x: true_anomaly_o(
        t, x['tpa'], x['p'], x['ex'], x['ey'], x['ez'], x['w'], *s),
    'lambert_phase_curve_o': lambda t, s, x: lambert_phase_curve_o(
        t, x['ag'], x['k'], x['tpa'], x['p'], *s),
    'ev_signal_o': lambda t, s, x: ev_signal_o(
        x['al'], x['mq'], x['inc'], t, x['tpa'], x['p'], *s),
    'emission_phase_curve_o': lambda t, s, x: emission_phase_curve_o(
        t, x['k'], x['fr'], x['off'], x['tpa'], x['p'], *s),
    'light_travel_time_o': lambda t, s, x: light_travel_time_o(
        t, x['tpa'], x['p'], x['e'], x['w'], RSTAR, *s),
}


class TestScalarQuantities:
    @pytest.mark.parametrize('name', sorted(NUMBA_REF))
    def test_parity(self, program, orbit, name):
        times, tpa, dt, ep_table, ep_times, coeffs = setup_orbit(orbit)
        x = extra_pars(orbit, tpa, dt)
        out_cl, = run_kernel(program, name, times, scalar_values(name, x),
                             ep_table, ep_times, coeffs)
        dispatch = (dt, ep_table, ep_times, coeffs)
        out_nb = NUMBA_REF[name](times, dispatch, x)
        np.testing.assert_allclose(out_cl, out_nb, rtol=RTOL, atol=ATOL)


class TestVectorQuantities:
    @pytest.mark.parametrize('name,ref', [('pos_o', pos_o), ('vel_o', vel_o)])
    def test_parity(self, program, orbit, name, ref):
        times, tpa, dt, ep_table, ep_times, coeffs = setup_orbit(orbit)
        p = orbit[0]
        cl_xyz = run_kernel(program, name, times, [tpa, p, dt],
                            ep_table, ep_times, coeffs, n_out=3)
        nb_xyz = ref(times, tpa, p, dt, ep_table, ep_times, coeffs)
        for got, want in zip(cl_xyz, nb_xyz):
            np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)


class TestEpIx:
    """Pins the fold, NaN guard, clamp, and table lookup in isolation."""

    def test_dense_grid_parity(self, program, orbit):
        times, tpa, dt, ep_table, ep_times, coeffs = setup_orbit(orbit)
        p = orbit[0]
        times = tpa + np.linspace(-2.0 * p, 3.0 * p, 5000)
        ix_cl, = run_kernel(program, 'ep_ix', times, [tpa, p, dt],
                            ep_table, ep_times, coeffs)
        ix_nb = np.array([ep_ix(t, tpa, p, dt, ep_table) for t in times])
        np.testing.assert_array_equal(ix_cl.astype(int), ix_nb)

    def test_nan_returns_first_index(self, program, orbit):
        times, tpa, dt, ep_table, ep_times, coeffs = setup_orbit(orbit)
        p = orbit[0]
        times = np.array([np.nan, tpa + 0.5 * p])
        ix_cl, = run_kernel(program, 'ep_ix', times, [tpa, p, dt],
                            ep_table, ep_times, coeffs)
        assert int(ix_cl[0]) == ep_table[0]
        assert int(ix_cl[1]) == ep_ix(times[1], tpa, p, dt, ep_table)


class TestTrueAnomalyBranches:
    def test_circular_fast_path(self, program, test_orbital_params):
        """ex = -1 with |e_vec| = 1 triggers the mean-anomaly fast path."""
        pars = test_orbital_params['circular']
        orbit = (pars['p'], pars['a'], pars['i'], pars['e'], pars['w'])
        times, tpa, dt, ep_table, ep_times, coeffs = setup_orbit(orbit)
        x = extra_pars(orbit, tpa, dt)
        x.update(ex=-1.0, ey=0.0, ez=0.0)
        out_cl, = run_kernel(program, 'true_anomaly_o', times,
                             scalar_values('true_anomaly_o', x),
                             ep_table, ep_times, coeffs)
        out_nb = true_anomaly_o(times, tpa, pars['p'], -1.0, 0.0, 0.0, pars['w'],
                                dt, ep_table, ep_times, coeffs)
        np.testing.assert_allclose(out_cl, out_nb, rtol=RTOL, atol=ATOL)
