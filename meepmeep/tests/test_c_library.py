"""Parity tests for the C library build (``c/``) against the numba backend.

The C library is a unity build of the sources shared with the OpenCL backend
plus the C-only pieces: expansion-point placement, the orbit-wide solvers,
and the gradient basis transforms. The shared evaluators are covered
exhaustively by the OpenCL parity suites; this module compiles the library
with the host C compiler, loads it through ctypes, and checks

- that the committed public header is in sync with the shared sources,
- a representative sample of the shared functions, and
- every C-only function,

all against the numba reference. Needs a C compiler on PATH (``$CC``,
``cc``, ``gcc`` or ``clang``); the compile-dependent tests skip without one.
The header drift test always runs.
"""

import ctypes
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.ctypeslib import ndpointer

from meepmeep.numba2d import solve2d, solve2d_d, sep_d as sep_d2
from meepmeep.numba3d import (solve3d, solve3d_d, solve3d_orbit, solve3d_orbit_d,
                              tc_to_tp_gradient, tp_to_tc_gradient,
                              tp_to_tc_gradient_orbit, create_expansion_points,
                              sep, pos, zpos, sep_d, sep_o, sep_od, pos_od, ep_ix)
from meepmeep.backends.numba.utils import TWO_PI, mean_anomaly_at_transit

ROOT = Path(__file__).resolve().parents[2]
C_DIR = ROOT / 'c'
SHARED_DIR = ROOT / 'meepmeep' / 'backends' / 'opencl'

RTOL = 1e-12
ATOL = 1e-13
NPT = 15
TRES = 200

MM_OK, MM_ERR_N_EP, MM_ERR_QUANTITY, MM_ERR_TRES, MM_ERR_ECCENTRICITY = 0, 1, 2, 3, 4
QUANTITY = {'mm': 0, 'ea': 1, 'ta': 2}


# ---------------------------------------------------------------------------
# Header drift guard (no compiler needed)
# ---------------------------------------------------------------------------

def test_header_matches_shared_sources():
    """The prototype block of meepmeep.h must be regenerated after any
    signature change in the .cl files (python c/tools/generate_header.py)."""
    spec = importlib.util.spec_from_file_location('generate_header', C_DIR / 'tools' / 'generate_header.py')
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    current = gen.HEADER.read_text()
    assert gen.splice(current, gen.generate()) == current, \
        "c/include/meepmeep.h is stale; run python c/tools/generate_header.py"


# ---------------------------------------------------------------------------
# Build and bind
# ---------------------------------------------------------------------------

def _find_compiler():
    for candidate in (os.environ.get('CC'), 'cc', 'gcc', 'clang'):
        if candidate and shutil.which(candidate):
            return candidate
    return None


D = ctypes.c_double
I = ctypes.c_int
DP = ctypes.POINTER(ctypes.c_double)
d_in = ndpointer(dtype=np.float64, flags='C')
d_out = ndpointer(dtype=np.float64, flags='C,W')
i_in = ndpointer(dtype=np.int32, flags='C')
i_out = ndpointer(dtype=np.int32, flags='C,W')


def _bind(lib):
    """Declare the argument and return types of the functions under test."""
    sig = {
        'mm_status_string': ([I], ctypes.c_char_p),
        'create_expansion_points': ([I, D, I, I, d_out, d_out, DP, i_out], I),
        'solve2d': ([D] * 7 + [d_out], None),
        'solve2d_d': ([D] * 7 + [I, d_out, d_out], None),
        'solve3d': ([D] * 7 + [d_out], None),
        'solve3d_d': ([D] * 7 + [I, d_out, d_out], None),
        'solve3d_orbit': ([d_in, I] + [D] * 6 + [d_out], None),
        'solve3d_orbit_d': ([d_in, I] + [D] * 6 + [d_out, d_out], None),
        'tc_to_tp_gradient': ([d_out, I, D, D, D], None),
        'tp_to_tc_gradient': ([d_out, I, D, D, D], None),
        'tp_to_tc_gradient_orbit': ([d_out, I, D, D, D], None),
        'mean_anomaly_at_transit': ([D, D], D),
        'sep2': ([D, D, D, d_in, D], D),
        'sep_d2': ([D, D, D, d_in, d_in, D, d_out], D),
        'pos3': ([D, D, D, d_in, D, DP, DP, DP], None),
        'zpos3': ([D, D, D, d_in, D], D),
        'sep3': ([D, D, D, d_in, D], D),
        'sep_d3': ([D, D, D, d_in, d_in, D, d_out], D),
        'ep_ix': ([D, D, D, D, i_in], I),
        'sep_o': ([D, D, D, D, i_in, d_in, d_in], D),
        'sep_od': ([D, D, D, D, i_in, d_in, d_in, d_in, d_out], D),
        'pos_od': ([D, D, D, D, i_in, d_in, d_in, d_in, DP, DP, DP, d_out, d_out, d_out], None),
    }
    for name, (argtypes, restype) in sig.items():
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype
    return lib


@pytest.fixture(scope='module')
def lib(tmp_path_factory):
    cc = _find_compiler()
    if cc is None or sys.platform.startswith('win'):
        pytest.skip("No C compiler available")
    suffix = '.dylib' if sys.platform == 'darwin' else '.so'
    out = tmp_path_factory.mktemp('libmeepmeep') / f'libmeepmeep{suffix}'
    sources = sorted(str(f) for f in (C_DIR / 'src').glob('*.c'))
    cmd = [cc, '-std=c99', '-O2', '-Wall', '-fPIC', '-shared',
           f'-I{C_DIR / "include"}', f'-I{SHARED_DIR}', *sources, '-lm', '-o', str(out)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"C library build failed:\n{' '.join(cmd)}\n{result.stderr}")
    return _bind(ctypes.CDLL(str(out)))


@pytest.fixture(params=['circular', 'eccentric', 'high_e'])
def orbit(request, test_orbital_params):
    pars = test_orbital_params[request.param]
    return pars['p'], pars['a'], pars['i'], pars['e'], pars['w']


def flat(a):
    return np.ascontiguousarray(a, dtype=np.float64).ravel()


# ---------------------------------------------------------------------------
# Shared sources compiled as C: a representative sample
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('te', [0.0, 0.03, -0.12])
def test_solvers_match_numba(lib, orbit, te):
    p, a, i, e, w = orbit
    lan = 0.2

    cf = np.empty(10)
    lib.solve2d(te, p, a, i, e, w, lan, cf)
    np.testing.assert_allclose(cf, flat(solve2d(te, p, a, i, e, w, lan)), rtol=RTOL, atol=ATOL)

    cf = np.empty(15)
    lib.solve3d(te, p, a, i, e, w, lan, cf)
    np.testing.assert_allclose(cf, flat(solve3d(te, p, a, i, e, w, lan)), rtol=RTOL, atol=ATOL)

    for from_periastron in (0, 1):
        cf, dcf = np.empty(10), np.empty(70)
        lib.solve2d_d(te, p, a, i, e, w, lan, from_periastron, cf, dcf)
        rc, rdc = solve2d_d(te, p, a, i, e, w, lan, bool(from_periastron))
        np.testing.assert_allclose(cf, flat(rc), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(dcf, flat(rdc), rtol=RTOL, atol=ATOL)

        cf, dcf = np.empty(15), np.empty(105)
        lib.solve3d_d(te, p, a, i, e, w, lan, from_periastron, cf, dcf)
        rc, rdc = solve3d_d(te, p, a, i, e, w, lan, bool(from_periastron))
        np.testing.assert_allclose(cf, flat(rc), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(dcf, flat(rdc), rtol=RTOL, atol=ATOL)


def test_single_expansion_point_evaluators_match_numba(lib, orbit):
    p, a, i, e, w = orbit
    tc, te = 1.5, 0.01
    c3 = solve3d(te, p, a, i, e, w)
    c3d, dc3 = solve3d_d(te, p, a, i, e, w)
    c2d, dc2 = solve2d_d(te, p, a, i, e, w)
    fc3, fc3d, fdc3, fc2d, fdc2 = map(flat, (c3, c3d, dc3, c2d, dc2))
    times = tc + np.linspace(-2.2 * p, 2.7 * p, 200)

    dd = np.empty(7)
    px, py, pz = D(), D(), D()
    for t in times:
        np.testing.assert_allclose(lib.sep3(t, tc, p, fc3, te), sep(t, tc, p, c3, te), rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(lib.zpos3(t, tc, p, fc3, te), zpos(t, tc, p, c3, te), rtol=RTOL, atol=ATOL)
        lib.pos3(t, tc, p, fc3, te, px, py, pz)
        np.testing.assert_allclose([px.value, py.value, pz.value], pos(t, tc, p, c3, te), rtol=RTOL, atol=ATOL)

        # The lan slot of a separation gradient is analytically zero and
        # comes out as +-1e-13 roundoff whose sign depends on fastmath
        # contraction, so the gradient tolerance is scaled to the signal.
        v = lib.sep_d3(t, tc, p, fc3d, fdc3, te, dd)
        rv, rdd = sep_d(t, tc, p, c3d, dc3, te)
        np.testing.assert_allclose(v, rv, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(dd, rdd, rtol=RTOL, atol=ATOL * max(1.0, np.abs(rdd).max()))

        v = lib.sep_d2(t, tc, p, fc2d, fdc2, te, dd)
        rv, rdd = sep_d2(t, tc, p, c2d, dc2, te)
        np.testing.assert_allclose(v, rv, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(dd, rdd, rtol=RTOL, atol=ATOL * max(1.0, np.abs(rdd).max()))


# ---------------------------------------------------------------------------
# C-only functions: expansion-point placement
# ---------------------------------------------------------------------------

def c_expansion_points(lib, n_ep, e, quantity, tres=TRES):
    ep_times = np.empty(n_ep)
    change_times = np.empty(n_ep - 1)
    dt = D()
    ep_table = np.empty(tres, dtype=np.int32)
    status = lib.create_expansion_points(n_ep, e, QUANTITY[quantity], tres, ep_times, change_times, dt, ep_table)
    return status, ep_times, change_times, dt.value, ep_table


@pytest.mark.parametrize('quantity', ['mm', 'ea', 'ta'])
@pytest.mark.parametrize('e', [0.0, 0.3, 0.7, 0.95])
@pytest.mark.parametrize('n_ep', [3, 15, 21])
def test_create_expansion_points_matches_numba(lib, quantity, e, n_ep):
    status, ep_times, change_times, dt, ep_table = c_expansion_points(lib, n_ep, e, quantity)
    assert status == MM_OK
    r_ep, r_ct, r_dt, r_table = create_expansion_points(n_ep, e, quantity, TRES)
    # Both sides run Brent to xtol = 2e-12; the roots agree to that level.
    np.testing.assert_allclose(ep_times, r_ep, rtol=0, atol=1e-11)
    np.testing.assert_allclose(change_times, r_ct, rtol=0, atol=1e-11)
    assert dt == r_dt
    np.testing.assert_array_equal(ep_table, r_table)


def test_create_expansion_points_periodic_image_contract(lib):
    for quantity in ('mm', 'ea', 'ta'):
        _, ep_times, change_times, _, _ = c_expansion_points(lib, NPT, 0.4, quantity)
        assert ep_times[0] == 0.0 and ep_times[-1] == 1.0 and ep_times[NPT // 2] == 0.5
        assert np.all(np.diff(ep_times) > 0)
        assert np.all((change_times > ep_times[:-1]) & (change_times < ep_times[1:]))


@pytest.mark.edge_case
@pytest.mark.parametrize('n_ep, e, quantity, tres, expected', [
    (14, 0.3, 'ea', TRES, MM_ERR_N_EP),
    (1, 0.3, 'ea', TRES, MM_ERR_N_EP),
    (15, 0.3, 'xx', TRES, MM_ERR_QUANTITY),
    (15, 0.3, 'ea', 0, MM_ERR_TRES),
    (15, 1.0, 'ea', TRES, MM_ERR_ECCENTRICITY),
    (15, -0.1, 'ta', TRES, MM_ERR_ECCENTRICITY),
    (15, float('nan'), 'mm', TRES, MM_ERR_ECCENTRICITY),
])
def test_create_expansion_points_rejects_bad_input(lib, n_ep, e, quantity, tres, expected):
    ep_times = np.empty(max(n_ep, 1))
    change_times = np.empty(max(n_ep - 1, 1))
    ep_table = np.empty(max(tres, 1), dtype=np.int32)
    dt = D()
    status = lib.create_expansion_points(n_ep, e, QUANTITY.get(quantity, 99), tres, ep_times, change_times, dt, ep_table)
    assert status == expected
    assert lib.mm_status_string(status).decode() != 'unknown status'


# ---------------------------------------------------------------------------
# C-only functions: orbit-wide solvers and gradient basis transforms
# ---------------------------------------------------------------------------

def test_solve3d_orbit_matches_numba(lib, orbit):
    p, a, i, e, w = orbit
    lan = 0.1
    ep_times, _, _, _ = create_expansion_points(NPT, max(e, 0.2), 'ea', TRES)

    coeffs = np.empty(NPT * 15)
    lib.solve3d_orbit(ep_times, NPT, p, a, i, e, w, lan, coeffs)
    np.testing.assert_allclose(coeffs, flat(solve3d_orbit(ep_times, p, a, i, e, w, lan, NPT)), rtol=RTOL, atol=ATOL)

    coeffs, dcoeffs = np.empty(NPT * 15), np.empty(NPT * 105)
    lib.solve3d_orbit_d(ep_times, NPT, p, a, i, e, w, lan, coeffs, dcoeffs)
    rc, rdc = solve3d_orbit_d(ep_times, p, a, i, e, w, lan, NPT)
    np.testing.assert_allclose(coeffs, flat(rc), rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(dcoeffs, flat(rdc), rtol=RTOL, atol=ATOL)

    # The in-place orbit transform must reproduce numba's tc-basis stack.
    lib.tp_to_tc_gradient_orbit(dcoeffs, NPT, p, e, w)
    tp_to_tc_gradient_orbit(rdc, p, e, w)
    np.testing.assert_allclose(dcoeffs, flat(rdc), rtol=RTOL, atol=ATOL)


def test_single_block_basis_transforms_match_numba(lib, orbit):
    p, a, i, e, w = orbit
    e = max(e, 0.1)
    for dim, solver in ((2, solve2d_d), (3, solve3d_d)):
        block = 5 * dim
        _, dc = solver(0.02, p, a, i, e, w)
        for c_fn, r_fn in ((lib.tc_to_tp_gradient, tc_to_tp_gradient),
                           (lib.tp_to_tc_gradient, tp_to_tc_gradient)):
            buf = flat(dc).copy()
            c_fn(buf, block, p, e, w)
            np.testing.assert_allclose(buf, flat(r_fn(dc, p, e, w)), rtol=RTOL, atol=ATOL)
        # Round trip is the identity.
        buf = flat(dc).copy()
        lib.tc_to_tp_gradient(buf, block, p, e, w)
        lib.tp_to_tc_gradient(buf, block, p, e, w)
        np.testing.assert_allclose(buf, flat(dc), rtol=RTOL, atol=1e-12)


def test_mean_anomaly_at_transit_matches_numba(lib, orbit):
    _, _, _, e, w = orbit
    np.testing.assert_allclose(lib.mean_anomaly_at_transit(e, w), mean_anomaly_at_transit(e, w), rtol=RTOL, atol=ATOL)


# ---------------------------------------------------------------------------
# End to end: C placement -> C solve -> C evaluation, against the numba chain
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('quantity', ['mm', 'ea', 'ta'])
def test_orbit_pipeline_matches_numba(lib, orbit, quantity):
    p, a, i, e, w = orbit
    lan = 0.0
    e_ep = max(e, 0.2)

    status, ep_times, _, dt, ep_table = c_expansion_points(lib, NPT, e_ep, quantity)
    assert status == MM_OK
    coeffs, dcoeffs = np.empty(NPT * 15), np.empty(NPT * 105)
    lib.solve3d_orbit_d(ep_times, NPT, p, a, i, e, w, lan, coeffs, dcoeffs)
    lib.tp_to_tc_gradient_orbit(dcoeffs, NPT, p, e, w)

    r_ep, _, r_dt, r_table = create_expansion_points(NPT, e_ep, quantity, TRES)
    rc, rdc = solve3d_orbit_d(r_ep, p, a, i, e, w, lan, NPT)
    tp_to_tc_gradient_orbit(rdc, p, e, w)

    tpa = -mean_anomaly_at_transit(e, w) / TWO_PI * p
    times = tpa + np.linspace(-2.5 * p, 3.5 * p, 400)

    # The expansion points differ by the root-finder tolerance (~1e-12 in
    # phase), which perturbs the coefficients at that level, so this
    # comparison is looser than the same-grid tests above.
    r_z = sep_o(times, tpa, p, r_dt, r_table, r_ep, rc)
    r_zd, r_dz = sep_od(times, tpa, p, r_dt, r_table, r_ep, rc, rdc)
    r_px, r_py, r_pz, r_dpx, r_dpy, r_dpz = pos_od(times, tpa, p, r_dt, r_table, r_ep, rc, rdc)

    dz, dpx, dpy, dpz = (np.empty(7) for _ in range(4))
    px, py, pz = D(), D(), D()
    for k, t in enumerate(times):
        assert lib.ep_ix(t, tpa, p, dt, ep_table) == ep_ix(t, tpa, p, r_dt, r_table)
        np.testing.assert_allclose(lib.sep_o(t, tpa, p, dt, ep_table, ep_times, coeffs), r_z[k], rtol=1e-9)
        z = lib.sep_od(t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs, dz)
        np.testing.assert_allclose(z, r_zd[k], rtol=1e-9)
        np.testing.assert_allclose(dz, r_dz[k], rtol=1e-8, atol=1e-9)
        lib.pos_od(t, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs, px, py, pz, dpx, dpy, dpz)
        np.testing.assert_allclose([px.value, py.value, pz.value], [r_px[k], r_py[k], r_pz[k]], rtol=1e-9)
        np.testing.assert_allclose(np.stack([dpx, dpy, dpz]), np.stack([r_dpx[k], r_dpy[k], r_dpz[k]]), rtol=1e-8, atol=1e-9)
