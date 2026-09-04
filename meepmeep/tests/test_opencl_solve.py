"""Parity tests for the OpenCL Taylor coefficient solvers.

Compares solve2d.cl and solve3d.cl against the numba reference in
`meepmeep.numba2d` / `meepmeep.numba3d` over a batch of parameter sets
spanning a realistic range, in both precisions and both timing bases.

Unlike the evaluator tests these need no test-only ``__kernel`` wrappers: the
shipped ``solve_kernels.cl`` provides the batched entry points, so the tests
exercise exactly what ships.

The batch deliberately spans a range of eccentricities and expansion-point
times rather than fixing them:

- a shared eccentricity would hide the warp divergence in the Kepler
  solver's Newton loop, and never reach the ``e > 0.8`` branch of its
  initial guess;
- ``te == 0`` would make the period term of the mean-anomaly derivative
  (``-2 pi te / p**2``) identically zero, so no fp64 tolerance could see it
  go missing. That is the same blind spot the multi-epoch evaluator tests
  avoid for the ``d[1] += epoch*d[0]`` chain term.
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

from meepmeep.numba2d import solve2d, solve2d_d
from meepmeep.numba3d import solve3d, solve3d_d
from meepmeep.backends.opencl import read_kernel_source
from meepmeep.tests.opencl_utils import (build, get_context, get_queue, has_fp64,
                                         upload, real_dtype)

NSETS = 400
LOCAL_SIZE = 64

# fp64 tracks numba to a few ulp over the whole eccentricity range. fp32 is
# good to ~1e-6 up to e ~ 0.7 and then degrades sharply -- ~1e-5 by e = 0.85 and
# ~2.5e-4 by e = 0.95 -- so it gets its own, smaller envelope plus an explicit
# high-e test that pins the degradation instead of hiding it.
#
# The degradation is conditioning, not a porting error: it lives in the `e`
# gradient row of the snap column, in the transit-centre basis only. That row
# is the one routed through mean_anomaly_at_transit_with_derivatives, whose
# de_off_de term carries -e / sqrt(1 - e**2), and the snap column carries
# inv_r7. The periastron basis skips the transit offset entirely and stays at
# ~7e-7 fp32 even at e = 0.88.
TOL = {'double': 5e-12, 'single': 5e-6}
E_FP32 = 0.7          # eccentricity ceiling for the tight fp32 envelope
TOL_FP32_HIGH_E = 1e-3


def make_pars(n=NSETS, seed=20260904, e_max=0.94, e_min=0.0):
    """A batch of (te, p, a, i, e, w, lan) rows spanning a realistic range."""
    rng = np.random.default_rng(seed)
    pars = np.empty((n, 7))
    pars[:, 1] = rng.uniform(0.5, 30.0, n)                    # p
    pars[:, 0] = rng.uniform(-0.5, 0.5, n) * pars[:, 1]       # te, over a period
    pars[:, 2] = rng.uniform(2.0, 40.0, n)                    # a
    pars[:, 3] = rng.uniform(1.2, 0.5 * np.pi, n)             # i
    # Include exactly-circular orbits and the e > 0.8 Kepler-guess branch.
    ncirc = min(20, n // 2) if e_min == 0.0 else 0
    pars[:, 4] = np.concatenate([np.zeros(ncirc),
                                 np.linspace(e_min, e_max, n - ncirc)])
    pars[:, 5] = rng.uniform(0.0, 2.0 * np.pi, n)             # w
    pars[:, 6] = rng.uniform(0.0, 2.0 * np.pi, n)             # lan
    return np.ascontiguousarray(pars)


def numba_reference(pars, dim, gradients, from_periastron):
    """Coefficients (and derivatives) from the numba solvers."""
    solve, solve_d = (solve3d, solve3d_d) if dim == 3 else (solve2d, solve2d_d)
    n = pars.shape[0]
    cf = np.empty((n, dim, 5))
    dcf = np.empty((n, 7, dim, 5)) if gradients else None
    for j, q in enumerate(pars):
        if gradients:
            cf[j], dcf[j] = solve_d(q[0], q[1], q[2], q[3], q[4], q[5], q[6],
                                    from_periastron)
        else:
            cf[j] = solve(q[0], q[1], q[2], q[3], q[4], q[5], q[6])
    return cf, dcf


def run_solver(pars, dim, gradients, from_periastron, precision):
    """Run the shipped batch kernel and return the results as float64."""
    ctx, queue = get_context(), get_queue()
    dt = real_dtype(precision)
    n = pars.shape[0]
    prg = build('', 'solve_kernels.cl', precision=precision)

    pars_buf = upload(queue, pars, precision)
    cf_host = np.empty(n * dim * 5, dtype=dt)
    cf_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, cf_host.nbytes)
    gsize = (int(np.ceil(n / LOCAL_SIZE) * LOCAL_SIZE),)

    if gradients:
        dcf_host = np.empty(n * 7 * dim * 5, dtype=dt)
        dcf_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, dcf_host.nbytes)
        kern = cl.Kernel(prg, f'solve{dim}d_d_batch')
        kern.set_args(pars_buf, np.int32(n), np.int32(from_periastron),
                      cf_buf, dcf_buf)
    else:
        dcf_host = dcf_buf = None
        kern = cl.Kernel(prg, f'solve{dim}d_batch')
        kern.set_args(pars_buf, np.int32(n), cf_buf)

    cl.enqueue_nd_range_kernel(queue, kern, gsize, (LOCAL_SIZE,))
    cl.enqueue_copy(queue, cf_host, cf_buf)
    if gradients:
        cl.enqueue_copy(queue, dcf_host, dcf_buf)
    queue.finish()

    cf = cf_host.astype(np.float64).reshape(n, dim, 5)
    dcf = dcf_host.astype(np.float64).reshape(n, 7, dim, 5) if gradients else None
    return cf, dcf


def rel_dev(got, ref):
    """Max absolute deviation normalised by the scale of the reference."""
    scale = np.abs(ref).max()
    return np.abs(got - ref).max() / (scale if scale > 0 else 1.0)


def batch_for(precision):
    """The parameter batch a given precision is expected to handle tightly."""
    return make_pars() if precision == 'double' else make_pars(e_max=E_FP32)


def precisions():
    out = ['single']
    if has_fp64():
        out.insert(0, 'double')
    return out


@pytest.mark.parametrize('precision', precisions())
@pytest.mark.parametrize('dim', [2, 3])
def test_values_match_numba(dim, precision):
    pars = batch_for(precision)
    ref, _ = numba_reference(pars, dim, False, False)
    got, _ = run_solver(pars, dim, False, False, precision)
    assert rel_dev(got, ref) < TOL[precision]


@pytest.mark.parametrize('precision', precisions())
@pytest.mark.parametrize('from_periastron', [False, True])
@pytest.mark.parametrize('dim', [2, 3])
def test_gradients_match_numba(dim, from_periastron, precision):
    pars = batch_for(precision)
    ref_cf, ref_dcf = numba_reference(pars, dim, True, from_periastron)
    cf, dcf = run_solver(pars, dim, True, from_periastron, precision)
    assert rel_dev(cf, ref_cf) < TOL[precision]
    assert rel_dev(dcf, ref_dcf) < TOL[precision]


@pytest.mark.parametrize('dim', [2, 3])
def test_extreme_eccentricity_needs_the_kepler_initial_guess(dim):
    """`ea_from_ma` must keep its e > 0.8 initial guess of E0 = pi.

    Below e ~ 0.97 the guess is a convergence nicety: Newton starting from
    E0 = M reaches the same root, so no test in the ordinary range can tell
    whether the branch is there (a mutation removing it survives the rest of
    this module). From e ~ 0.99 it becomes load-bearing -- starting at M
    diverges outright, leaving a Kepler residual of order 1e18 -- so this is
    the band that actually pins it.

    fp64 only: fp32 cannot resolve this regime (see the envelope test below).
    """
    if not has_fp64():
        pytest.skip('needs fp64')
    pars = make_pars(n=120, e_min=0.985, e_max=0.995)
    ref, _ = numba_reference(pars, dim, False, False)
    got, _ = run_solver(pars, dim, False, False, 'double')
    assert np.isfinite(got).all()
    assert rel_dev(got, ref) < TOL['double']

    ref_cf, ref_dcf = numba_reference(pars, dim, True, False)
    cf, dcf = run_solver(pars, dim, True, False, 'double')
    assert np.isfinite(dcf).all()
    assert rel_dev(cf, ref_cf) < TOL['double']


@pytest.mark.parametrize('dim', [2, 3])
def test_high_eccentricity_single_precision_envelope(dim):
    """fp32 degrades near e -> 1, but only so far, and fp64 does not.

    Pins the measured envelope so a real regression is still visible through
    the conditioning. Skipped without fp64, which is the reference here.
    """
    if not has_fp64():
        pytest.skip('needs fp64 for the reference comparison')
    pars = make_pars(n=120, e_min=0.85, e_max=0.95)
    ref_cf, ref_dcf = numba_reference(pars, dim, True, False)

    _, dcf32 = run_solver(pars, dim, True, False, 'single')
    dev32 = rel_dev(dcf32, ref_dcf)
    assert dev32 < TOL_FP32_HIGH_E, f'fp32 worse than the known envelope: {dev32:.2e}'

    _, dcf64 = run_solver(pars, dim, True, False, 'double')
    assert rel_dev(dcf64, ref_dcf) < TOL['double']

    # The periastron basis avoids the ill-conditioned transit-offset term.
    _, ref_p = numba_reference(pars, dim, True, True)
    _, got_p = run_solver(pars, dim, True, True, 'single')
    assert rel_dev(got_p, ref_p) < dev32


@pytest.mark.parametrize('from_periastron', [False, True])
@pytest.mark.parametrize('dim', [2, 3])
def test_each_gradient_row_matches_numba(dim, from_periastron):
    """Per-parameter breakdown: one wrong chain-rule row can hide in the whole."""
    pars = make_pars()
    _, ref = numba_reference(pars, dim, True, from_periastron)
    _, got = run_solver(pars, dim, True, from_periastron, 'double')
    for k, name in enumerate(('tc', 'p', 'a', 'i', 'e', 'w', 'lan')):
        assert rel_dev(got[:, k], ref[:, k]) < TOL['double'], f'row {k} ({name})'


@pytest.mark.parametrize('dim', [2, 3])
def test_gradient_solver_value_matches_value_solver(dim):
    """The cf from solve*_d equals the cf from solve*, as in numba."""
    pars = make_pars()
    cf_val, _ = run_solver(pars, dim, False, False, 'double')
    cf_grad, _ = run_solver(pars, dim, True, False, 'double')
    assert rel_dev(cf_grad, cf_val) < TOL['double']


@pytest.mark.parametrize('dim', [2, 3])
def test_lan_row_leaves_z_alone(dim):
    """`lan` rotates the sky plane only, so the 3D lan row has no z component.

    The numba solver gets that zero from `zeros((7, 3, 5))`; the device
    assembles the tensor in uninitialised private memory and must write it.
    """
    pars = make_pars()
    _, dcf = run_solver(pars, dim, True, False, 'double')
    if dim == 3:
        assert np.all(dcf[:, 6, 2, :] == 0.0)
    assert np.isfinite(dcf).all()


def test_nsets_guard_leaves_surplus_untouched():
    """Work items past `nsets` must not write outside their block."""
    pars = make_pars(n=5)
    ctx, queue = get_context(), get_queue()
    prg = build('', 'solve_kernels.cl', precision='double')
    pars_buf = upload(queue, pars, 'double')
    # Room for 64 sets, but only 5 declared: the rest must survive untouched.
    host = np.full(64 * 10, -12345.0)
    buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                    hostbuf=host)
    kern = cl.Kernel(prg, 'solve2d_batch')
    kern.set_args(pars_buf, np.int32(5), buf)
    cl.enqueue_nd_range_kernel(queue, kern, (64,), (64,))
    cl.enqueue_copy(queue, host, buf)
    queue.finish()
    assert np.all(host[50:] == -12345.0)
    assert np.all(host[:50] != -12345.0)


class TestSourceContract:
    def test_only_solve_kernels_defines_kernels(self):
        """Every shipped file except solve_kernels.cl is device functions only."""
        from meepmeep.backends.opencl import SOURCE_FILES
        from importlib.resources import files
        pkg = files('meepmeep.backends.opencl')
        for name in SOURCE_FILES:
            body = '\n'.join(line for line in pkg.joinpath(name).read_text().splitlines()
                             if not line.lstrip().startswith(('*', '/*', '//')))
            if name == 'solve_kernels.cl':
                assert '__kernel' in body, name
            else:
                assert '__kernel' not in body, f'{name} defines a __kernel'

    def test_solvers_available_without_kernels(self):
        src = read_kernel_source('solve2d.cl', 'solve3d.cl')
        assert 'solve2d(' in src and 'solve3d(' in src
        assert '__kernel' not in src

    def test_kernels_pull_their_solvers(self):
        src = read_kernel_source('solve_kernels.cl')
        for needed in ('ea_from_ma', 'solve2d', 'solve3d', 'solve2d_d', 'solve3d_d'):
            assert needed in src, needed
