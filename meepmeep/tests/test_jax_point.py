"""JAX backend: single-expansion-point evaluators against numba.

Values are compared with the numba ``point2d``/``point3d`` dispatchers, and
``jax.jacfwd`` through ``solve + evaluator`` with the numba ``_d`` kernels
fed by ``solve2d_d``/``solve3d_d``. The times span several epochs so that
the period chain term (``d/dp += epoch * d/dtc``) is observable.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

jax = pytest.importorskip("jax")

from meepmeep.tests.jax_utils import ORBITS, jacobian, assert_grad_close  # noqa: E402

from meepmeep import jax2d, jax3d, numba2d, numba3d  # noqa: E402

ORBIT_IDS = list(ORBITS)
TC = 2.3
TE = 0.13


def _times(p, n=64, seed=0):
    rng = np.random.default_rng(seed)
    epochs = rng.integers(-3, 5, n)
    return TC + TE + epochs * p + rng.uniform(-0.04, 0.04, n) * p


# name: (jax fn, numba value fn, numba _d fn, extra physical inputs, 3d?)
# Evaluators are called as f(times, *extras, tc, p, c, te); rv as f(times, k, tc, p, a, i, e, c, te).
EVALUATORS_3D = {
    "pos": (jax3d.pos, numba3d.pos, numba3d.pos_d, ()),
    "zpos": (jax3d.zpos, numba3d.zpos, numba3d.zpos_d, ()),
    "sep": (jax3d.sep, numba3d.sep, numba3d.sep_d, ()),
    "vel": (jax3d.vel, numba3d.vel, numba3d.vel_d, ()),
    "zvel": (jax3d.zvel, numba3d.zvel, numba3d.zvel_d, ()),
    "cos_alpha": (jax3d.cos_alpha, numba3d.cos_alpha, numba3d.cos_alpha_d, ()),
    "lambert": (jax3d.lambert_phase_curve, numba3d.lambert_phase_curve, numba3d.lambert_phase_curve_d,
                (0.3, 0.1)),
    "emission": (jax3d.emission_phase_curve, numba3d.emission_phase_curve, numba3d.emission_phase_curve_d,
                 (0.1, 0.02, 0.4)),
}


def _tup(x):
    return x if isinstance(x, tuple) else (x,)


def _split(numba_out, n_values):
    """Split a numba ``_d`` return into value tuple and gradient tuple."""
    return numba_out[:n_values], numba_out[n_values:]


@pytest.mark.parametrize("name", ORBIT_IDS)
@pytest.mark.parametrize("quantity", list(EVALUATORS_3D))
class TestPoint3D:
    def test_values(self, name, quantity):
        jfn, nfn, _, extras = EVALUATORS_3D[quantity]
        p, a, i, e, w, lan = ORBITS[name]
        c = numba3d.solve3d(TE, p, a, i, e, w, lan)
        times = _times(p)
        expected = nfn(times, *extras, TC, p, c, TE)
        actual = jfn(times, *extras, TC, p, c, TE)
        for act, exp in zip(_tup(actual), _tup(expected)):
            assert_allclose(np.asarray(act), exp, rtol=1e-12, atol=1e-13 * np.abs(exp).max())

    def test_scalar_time(self, name, quantity):
        jfn, nfn, _, extras = EVALUATORS_3D[quantity]
        p, a, i, e, w, lan = ORBITS[name]
        c = numba3d.solve3d(TE, p, a, i, e, w, lan)
        t = float(_times(p)[3])
        actual = jfn(t, *extras, TC, p, c, TE)
        expected = nfn(t, *extras, TC, p, c, TE)
        for act, exp in zip(_tup(actual), _tup(expected)):
            assert np.ndim(act) == 0
            assert_allclose(float(act), exp, rtol=1e-12, atol=1e-14)

    def test_gradient(self, name, quantity):
        jfn, _, nfn_d, extras = EVALUATORS_3D[quantity]
        p, a, i, e, w, lan = ORBITS[name]
        times = _times(p)
        c, dc = numba3d.solve3d_d(TE, p, a, i, e, w, lan)
        nb = nfn_d(times, *extras, TC, p, c, dc, TE)

        def f(tc, p, a, i, e, w, lan, *extras):
            return jfn(times, *extras, tc, p, jax3d.solve3d(TE, p, a, i, e, w, lan), TE)

        n = 7 + len(extras)
        jac = jacobian(f, n, TC, p, a, i, e, w, lan, *extras)
        jac = _tup(jac)
        _, grads = _split(nb, len(jac))
        for j, g in zip(jac, grads):
            assert_grad_close(j, g)


@pytest.mark.parametrize("name", ORBIT_IDS)
class TestPoint3DSpecial:
    def test_rv(self, name):
        p, a, i, e, w, lan = ORBITS[name]
        k = 12.0
        times = _times(p)
        c, dc = numba3d.solve3d_d(TE, p, a, i, e, w, lan)
        assert_allclose(np.asarray(jax3d.rv(times, k, TC, p, a, i, e, c, TE)),
                        numba3d.rv(times, k, TC, p, a, i, e, c, TE), rtol=1e-12, atol=1e-12 * k)

        def f(tc, p, a, i, e, w, lan):
            return jax3d.rv(times, k, tc, p, a, i, e, jax3d.solve3d(TE, p, a, i, e, w, lan), TE)

        # The single-expansion-point rv_d gradient is 7 wide (no k slot, unlike rv_od).
        _, drv = numba3d.rv_d(times, k, TC, p, a, i, e, c, dc, TE)
        assert_grad_close(jacobian(f, 7, TC, p, a, i, e, w, lan), drv)

    def test_ev_signal(self, name):
        """The inclination enters twice (coefficients and the sin^2 prefactor); numba sums both into slot 3."""
        p, a, i, e, w, lan = ORBITS[name]
        alpha, q = 1.3, 1e-3
        times = _times(p)
        c, dc = numba3d.solve3d_d(TE, p, a, i, e, w, lan)
        assert_allclose(np.asarray(jax3d.ev_signal(times, alpha, q, i, TC, p, c, TE)),
                        numba3d.ev_signal(times, alpha, q, i, TC, p, c, TE), rtol=1e-12)

        def f(tc, p, a, i, e, w, lan, alpha, q):
            return jax3d.ev_signal(times, alpha, q, i, tc, p, jax3d.solve3d(TE, p, a, i, e, w, lan), TE)

        _, dev = numba3d.ev_signal_d(times, alpha, q, i, TC, p, c, dc, TE)
        assert_grad_close(jacobian(f, 9, TC, p, a, i, e, w, lan, alpha, q), dev)

    def test_centered_matches_direct(self, name):
        p, a, i, e, w, lan = ORBITS[name]
        c = jax3d.solve3d(TE, p, a, i, e, w, lan)
        tau = np.linspace(-0.05, 0.05, 11) * p
        for fc, fd in [(jax3d.pos_c, jax3d.pos), (jax3d.sep_c, jax3d.sep), (jax3d.vel_c, jax3d.vel),
                       (jax3d.zpos_c, jax3d.zpos), (jax3d.zvel_c, jax3d.zvel),
                       (jax3d.cos_alpha_c, jax3d.cos_alpha)]:
            assert_allclose(np.asarray(fc(tau, c)), np.asarray(fd(TC + TE + tau, TC, p, c, TE)),
                            rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("name", ORBIT_IDS)
class TestPoint2D:
    def test_values(self, name):
        p, a, i, e, w, lan = ORBITS[name]
        c = numba2d.solve2d(TE, p, a, i, e, w, lan)
        times = _times(p)
        for act, exp in zip(jax2d.pos(times, TC, p, c, TE), numba2d.pos(times, TC, p, c, TE)):
            assert_allclose(np.asarray(act), exp, rtol=1e-12, atol=1e-13 * np.abs(exp).max())
        assert_allclose(np.asarray(jax2d.sep(times, TC, p, c, TE)), numba2d.sep(times, TC, p, c, TE), rtol=1e-12)

    def test_gradients(self, name):
        p, a, i, e, w, lan = ORBITS[name]
        times = _times(p)
        c, dc = numba2d.solve2d_d(TE, p, a, i, e, w, lan)

        def fpos(tc, p, a, i, e, w, lan):
            return jax2d.pos(times, tc, p, jax2d.solve2d(TE, p, a, i, e, w, lan), TE)

        def fsep(tc, p, a, i, e, w, lan):
            return jax2d.sep(times, tc, p, jax2d.solve2d(TE, p, a, i, e, w, lan), TE)

        pars = (TC, p, a, i, e, w, lan)
        _, _, dpx, dpy = numba2d.pos_d(times, TC, p, c, dc, TE)
        jx, jy = jacobian(fpos, 7, *pars)
        assert_grad_close(jx, dpx)
        assert_grad_close(jy, dpy)
        _, dd = numba2d.sep_d(times, TC, p, c, dc, TE)
        assert_grad_close(jacobian(fsep, 7, *pars), dd)


def test_lambert_gradient_finite_at_full_phase():
    """At full phase (alpha = 0) differentiating arccos would give NaN; the custom JVP does not."""
    p, a, i, e, w = 3.0, 10.0, 0.5 * np.pi, 0.0, 0.0
    c = jax3d.solve3d(0.5 * p, p, a, i, e, w)  # expand at the secondary eclipse

    def f(a, i, ag, k):
        return jax3d.lambert_phase_curve(TC + 0.5 * p, ag, k, TC, p, jax3d.solve3d(0.5 * p, p, a, i, e, w), 0.5 * p)

    assert float(jax3d.cos_alpha(TC + 0.5 * p, TC, p, c, 0.5 * p)) == pytest.approx(1.0, abs=1e-12)
    g = jax.grad(f, argnums=(0, 1, 2, 3))(a, i, 0.3, 0.1)
    assert np.all(np.isfinite(g))
