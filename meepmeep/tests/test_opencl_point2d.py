"""Parity tests for the OpenCL 2D single-expansion-point evaluators.

Compares the device functions in point2d.cl (and, later, point2dd.cl)
against the numba reference in `meepmeep.numba2d` over near-transit and
multi-epoch times. The ``__kernel`` wrappers below are test-only; the
shipped backend contains device functions only.
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

from meepmeep.numba2d import (solve2d, pos_c, pos, sep_c, sep,
                              solve2d_d, pos_cd, pos_d, sep_cd, sep_d)
from meepmeep.tests.opencl_utils import (build, get_queue, has_fp64, upload,
                                         output_buffer, read_back, real_dtype)

RTOL = 1e-12
ATOL = 1e-13

VALUE_KERNELS = """
__kernel void k_pos_c2(__global const REAL *t, __global const REAL *c,
                       __global REAL *px, __global REAL *py) {
    int i = get_global_id(0);
    REAL x, y;
    pos_c2(t[i], c, &x, &y);
    px[i] = x;
    py[i] = y;
}

__kernel void k_pos2(__global const REAL *t, const REAL tc, const REAL p,
                     __global const REAL *c, const REAL te,
                     __global REAL *px, __global REAL *py) {
    int i = get_global_id(0);
    REAL x, y;
    pos2(t[i], tc, p, c, te, &x, &y);
    px[i] = x;
    py[i] = y;
}

__kernel void k_sep_c2(__global const REAL *t, __global const REAL *c,
                       __global REAL *d) {
    int i = get_global_id(0);
    d[i] = sep_c2(t[i], c);
}

__kernel void k_sep2(__global const REAL *t, const REAL tc, const REAL p,
                     __global const REAL *c, const REAL te,
                     __global REAL *d) {
    int i = get_global_id(0);
    d[i] = sep2(t[i], tc, p, c, te);
}
"""

GRAD_KERNELS = """
__kernel void k_pos_cd2(__global const REAL *t, __global const REAL *c,
                        __global const REAL *dc,
                        __global REAL *px, __global REAL *py,
                        __global REAL *dpx, __global REAL *dpy) {
    int i = get_global_id(0);
    REAL x, y, dx[7], dy[7];
    pos_cd2(t[i], c, dc, &x, &y, dx, dy);
    px[i] = x;
    py[i] = y;
    for (int m = 0; m < 7; m++) {
        dpx[7 * i + m] = dx[m];
        dpy[7 * i + m] = dy[m];
    }
}

__kernel void k_pos_d2(__global const REAL *t, const REAL tc, const REAL p,
                       __global const REAL *c, __global const REAL *dc,
                       const REAL te,
                       __global REAL *px, __global REAL *py,
                       __global REAL *dpx, __global REAL *dpy) {
    int i = get_global_id(0);
    REAL x, y, dx[7], dy[7];
    pos_d2(t[i], tc, p, c, dc, te, &x, &y, dx, dy);
    px[i] = x;
    py[i] = y;
    for (int m = 0; m < 7; m++) {
        dpx[7 * i + m] = dx[m];
        dpy[7 * i + m] = dy[m];
    }
}

__kernel void k_sep_cd2(__global const REAL *t, __global const REAL *c,
                        __global const REAL *dc,
                        __global REAL *d, __global REAL *dd) {
    int i = get_global_id(0);
    REAL g[7];
    d[i] = sep_cd2(t[i], c, dc, g);
    for (int m = 0; m < 7; m++)
        dd[7 * i + m] = g[m];
}

__kernel void k_sep_d2(__global const REAL *t, const REAL tc, const REAL p,
                       __global const REAL *c, __global const REAL *dc,
                       const REAL te,
                       __global REAL *d, __global REAL *dd) {
    int i = get_global_id(0);
    REAL g[7];
    d[i] = sep_d2(t[i], tc, p, c, dc, te, g);
    for (int m = 0; m < 7; m++)
        dd[7 * i + m] = g[m];
}
"""


@pytest.fixture(scope='module')
def program():
    if not has_fp64():
        pytest.skip("Device lacks cl_khr_fp64")
    return build(VALUE_KERNELS + GRAD_KERNELS, 'point2dd.cl')


@pytest.fixture(params=['circular', 'eccentric'])
def orbit(request, test_orbital_params):
    pars = test_orbital_params[request.param]
    return pars['p'], pars['a'], pars['i'], pars['e'], pars['w']


def centered_times():
    """Times relative to the expansion point, within its region of validity."""
    return np.linspace(-0.1, 0.1, 101)


def multi_epoch_times(tc, p, te):
    """Absolute times clustered around the expansion point over epochs -3..3."""
    offsets = np.linspace(-0.1, 0.1, 11)
    return np.concatenate([tc + te + k * p + offsets for k in range(-3, 4)])


def run_centered(program, kernel_name, times, c, n_out=1):
    queue = get_queue()
    b_t = upload(queue, times)
    b_c = upload(queue, c)
    outs = [output_buffer(queue, times.size) for _ in range(n_out)]
    kernel = cl.Kernel(program, kernel_name)
    kernel(queue, (times.size,), None, b_t, b_c, *[b for b, _ in outs])
    return [read_back(queue, b, h) for b, h in outs]


def run_direct(program, kernel_name, times, tc, p, c, te, n_out=1):
    queue = get_queue()
    b_t = upload(queue, times)
    b_c = upload(queue, c)
    outs = [output_buffer(queue, times.size) for _ in range(n_out)]
    real = real_dtype('double')
    kernel = cl.Kernel(program, kernel_name)
    kernel(queue, (times.size,), None, b_t, real(tc), real(p), b_c, real(te),
           *[b for b, _ in outs])
    return [read_back(queue, b, h) for b, h in outs]


def run_grad_centered(program, kernel_name, times, c, dc, n_val=1):
    queue = get_queue()
    n = times.size
    b_t, b_c, b_dc = upload(queue, times), upload(queue, c), upload(queue, dc)
    val_outs = [output_buffer(queue, n) for _ in range(n_val)]
    grad_outs = [output_buffer(queue, 7 * n) for _ in range(n_val)]
    kernel = cl.Kernel(program, kernel_name)
    kernel(queue, (n,), None, b_t, b_c, b_dc,
           *[b for b, _ in val_outs], *[b for b, _ in grad_outs])
    vals = [read_back(queue, b, h) for b, h in val_outs]
    grads = [read_back(queue, b, h).reshape(n, 7) for b, h in grad_outs]
    return vals, grads


def run_grad_direct(program, kernel_name, times, tc, p, c, dc, te, n_val=1):
    queue = get_queue()
    n = times.size
    b_t, b_c, b_dc = upload(queue, times), upload(queue, c), upload(queue, dc)
    val_outs = [output_buffer(queue, n) for _ in range(n_val)]
    grad_outs = [output_buffer(queue, 7 * n) for _ in range(n_val)]
    real = real_dtype('double')
    kernel = cl.Kernel(program, kernel_name)
    kernel(queue, (n,), None, b_t, real(tc), real(p), b_c, b_dc, real(te),
           *[b for b, _ in val_outs], *[b for b, _ in grad_outs])
    vals = [read_back(queue, b, h) for b, h in val_outs]
    grads = [read_back(queue, b, h).reshape(n, 7) for b, h in grad_outs]
    return vals, grads


class TestCenteredValues:
    def test_pos_c2(self, program, orbit):
        p, a, i, e, w = orbit
        c = solve2d(0.0, p, a, i, e, w)
        times = centered_times()
        px_cl, py_cl = run_centered(program, 'k_pos_c2', times, c, n_out=2)
        px_nb, py_nb = pos_c(times, c)
        np.testing.assert_allclose(px_cl, px_nb, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(py_cl, py_nb, rtol=RTOL, atol=ATOL)

    def test_sep_c2(self, program, orbit):
        p, a, i, e, w = orbit
        c = solve2d(0.0, p, a, i, e, w)
        times = centered_times()
        d_cl, = run_centered(program, 'k_sep_c2', times, c)
        d_nb = sep_c(times, c)
        np.testing.assert_allclose(d_cl, d_nb, rtol=RTOL, atol=ATOL)


class TestDirectValues:
    @pytest.mark.parametrize('te', [0.0, 0.02])
    def test_pos2(self, program, orbit, te):
        p, a, i, e, w = orbit
        tc = 1.5
        c = solve2d(te, p, a, i, e, w)
        times = multi_epoch_times(tc, p, te)
        px_cl, py_cl = run_direct(program, 'k_pos2', times, tc, p, c, te, n_out=2)
        px_nb, py_nb = pos(times, tc, p, c, te)
        np.testing.assert_allclose(px_cl, px_nb, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(py_cl, py_nb, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize('te', [0.0, 0.02])
    def test_sep2(self, program, orbit, te):
        p, a, i, e, w = orbit
        tc = 1.5
        c = solve2d(te, p, a, i, e, w)
        times = multi_epoch_times(tc, p, te)
        d_cl, = run_direct(program, 'k_sep2', times, tc, p, c, te)
        d_nb = sep(times, tc, p, c, te)
        np.testing.assert_allclose(d_cl, d_nb, rtol=RTOL, atol=ATOL)

    def test_negative_epochs_fold_correctly(self, program, orbit):
        """Times before tc exercise floor()'s round-toward-minus-infinity."""
        p, a, i, e, w = orbit
        tc = 1.5
        c = solve2d(0.0, p, a, i, e, w)
        times = tc - 2.0 * p + np.linspace(-0.05, 0.05, 21)
        d_cl, = run_direct(program, 'k_sep2', times, tc, p, c, 0.0)
        d_nb = sep(times, tc, p, c, 0.0)
        np.testing.assert_allclose(d_cl, d_nb, rtol=RTOL, atol=ATOL)


class TestCenteredGradients:
    def test_pos_cd2(self, program, orbit):
        p, a, i, e, w = orbit
        c, dc = solve2d_d(0.0, p, a, i, e, w)
        times = centered_times()
        (px_cl, py_cl), (dpx_cl, dpy_cl) = run_grad_centered(
            program, 'k_pos_cd2', times, c, dc, n_val=2)
        px_nb, py_nb, dpx_nb, dpy_nb = pos_cd(times, c, dc)
        np.testing.assert_allclose(px_cl, px_nb, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(py_cl, py_nb, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(dpx_cl, dpx_nb, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(dpy_cl, dpy_nb, rtol=RTOL, atol=ATOL)

    def test_sep_cd2(self, program, orbit):
        p, a, i, e, w = orbit
        c, dc = solve2d_d(0.0, p, a, i, e, w)
        times = centered_times()
        (d_cl,), (dd_cl,) = run_grad_centered(program, 'k_sep_cd2', times, c, dc)
        d_nb, dd_nb = sep_cd(times, c, dc)
        np.testing.assert_allclose(d_cl, d_nb, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(dd_cl, dd_nb, rtol=RTOL, atol=ATOL)


class TestDirectGradients:
    """Multi-epoch times make the d[1] += epoch*d[0] chain term observable."""

    @pytest.mark.parametrize('te', [0.0, 0.02])
    def test_pos_d2(self, program, orbit, te):
        p, a, i, e, w = orbit
        tc = 1.5
        c, dc = solve2d_d(te, p, a, i, e, w)
        times = multi_epoch_times(tc, p, te)
        (px_cl, py_cl), (dpx_cl, dpy_cl) = run_grad_direct(
            program, 'k_pos_d2', times, tc, p, c, dc, te, n_val=2)
        px_nb, py_nb, dpx_nb, dpy_nb = pos_d(times, tc, p, c, dc, te)
        np.testing.assert_allclose(px_cl, px_nb, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(py_cl, py_nb, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(dpx_cl, dpx_nb, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(dpy_cl, dpy_nb, rtol=RTOL, atol=ATOL)

    @pytest.mark.parametrize('te', [0.0, 0.02])
    def test_sep_d2(self, program, orbit, te):
        p, a, i, e, w = orbit
        tc = 1.5
        c, dc = solve2d_d(te, p, a, i, e, w)
        times = multi_epoch_times(tc, p, te)
        (d_cl,), (dd_cl,) = run_grad_direct(program, 'k_sep_d2', times, tc, p, c, dc, te)
        d_nb, dd_nb = sep_d(times, tc, p, c, dc, te)
        np.testing.assert_allclose(d_cl, d_nb, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(dd_cl, dd_nb, rtol=RTOL, atol=ATOL)

    def test_chain_term_nonzero_off_epoch(self, program, orbit):
        """Guard against a port that drops the period chain term entirely."""
        p, a, i, e, w = orbit
        tc = 1.5
        c, dc = solve2d_d(0.0, p, a, i, e, w)
        times = tc + 3.0 * p + np.linspace(-0.05, 0.05, 11)
        (_,), (dd_cl,) = run_grad_direct(program, 'k_sep_d2', times, tc, p, c, dc, 0.0)
        _, dd_c = sep_cd(times - tc - 3.0 * p, c, dc)
        assert not np.allclose(dd_cl[:, 1], dd_c[:, 1])
        np.testing.assert_allclose(dd_cl[:, 1], dd_c[:, 1] + 3.0 * dd_c[:, 0],
                                   rtol=RTOL, atol=ATOL)


class TestSinglePrecision:
    """fp32 smoke test: documents, rather than hides, single-precision limits.

    Times use a small origin (never BJD-scale): a float32 ulp at BJD ~2.4e6
    is ~0.25 d, so absolute-time consumers require host-shifted times in
    fp32 builds.
    """

    def test_sep2_fp32(self, orbit):
        p, a, i, e, w = orbit
        tc = 0.25
        c = solve2d(0.0, p, a, i, e, w)
        times = multi_epoch_times(tc, p, 0.0)

        program = build(VALUE_KERNELS, 'point2d.cl', precision='single')
        queue = get_queue()
        b_t = upload(queue, times, precision='single')
        b_c = upload(queue, c, precision='single')
        b_d, h_d = output_buffer(queue, times.size, precision='single')
        real = real_dtype('single')
        kernel = cl.Kernel(program, 'k_sep2')
        kernel(queue, (times.size,), None, b_t, real(tc), real(p), b_c, real(0.0), b_d)
        d_cl = read_back(queue, b_d, h_d)

        d_nb = sep(times, tc, p, c, 0.0)
        np.testing.assert_allclose(d_cl, d_nb, rtol=1e-4, atol=1e-4)
