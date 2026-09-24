"""JAX backend: contact points, durations and minimum separation.

Values are compared with the numba ``point2d``/``point3d`` utilities (the
search loops are replicated step by step, so they agree to round-off).
Derivatives, which numba does not offer, come from implicit-function rules
and are checked against finite differences of tightly-converged roots.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import brentq, minimize_scalar

jax = pytest.importorskip("jax")

from meepmeep.tests.jax_utils import ORBITS  # noqa: E402

import jax.numpy as jnp  # noqa: E402

from meepmeep import jax2d, jax3d, numba2d, numba3d  # noqa: E402

TRANSITING = ["circular", "eccentric", "edge_on"]
K = 0.1


def _coeffs(name, ndim=3):
    p, a, i, e, w, lan = ORBITS[name]
    solver = jax3d.solve3d if ndim == 3 else jax2d.solve2d
    return np.asarray(solver(0.0, p, a, i, e, w, lan))


@pytest.mark.parametrize("name", TRANSITING)
@pytest.mark.parametrize("point", [1, 2, 3, 4, 12])
def test_contact_point_matches_numba(name, point):
    c3 = _coeffs(name)
    c2 = _coeffs(name, 2)
    assert_allclose(float(jax3d.find_contact_point(K, point, c3)), numba3d.find_contact_point(K, point, c3),
                    rtol=0, atol=1e-12)
    assert_allclose(float(jax2d.find_contact_point(K, point, c2)), numba2d.find_contact_point(K, point, c2),
                    rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", TRANSITING)
@pytest.mark.parametrize("fn", ["t14", "t23", "t12", "t34", "t1", "t4"])
def test_durations_match_numba(name, fn):
    c = _coeffs(name)
    assert_allclose(float(getattr(jax3d, fn)(K, c)), getattr(numba3d, fn)(K, c), rtol=0, atol=1e-12)


@pytest.mark.parametrize("name", TRANSITING)
def test_bounding_box_and_z_min_match_numba(name):
    c = _coeffs(name)
    assert_allclose(np.array(jax3d.bounding_box(K, c), dtype=float), numba3d.bounding_box(K, c), atol=1e-12)
    tj, zj = jax3d.find_z_min(0.001, c)
    tn, zn = numba3d.find_z_min(0.001, c)
    assert_allclose([float(tj), float(zj)], [tn, zn], rtol=0, atol=1e-12)


def _exact_contact(k, point, pars):
    """Tightly-converged contact time for finite differencing."""
    c = np.asarray(jax3d.solve3d(0.0, *pars))
    zt = {1: 1 + k, 4: 1 + k, 2: 1 - k, 3: 1 - k}[point]
    t = float(jax3d.find_contact_point(k, point, c))
    return brentq(lambda s: float(jax3d.sep_c(s, c)) - zt, t - 1e-3, t + 1e-3, xtol=1e-15, rtol=1e-15)


@pytest.mark.parametrize("name", ["eccentric", "edge_on"])
@pytest.mark.parametrize("point", [1, 2, 3, 4])
def test_contact_point_derivatives(name, point):
    pars = ORBITS[name]

    def f(k, a, i, e):
        p, _, _, _, w, lan = pars
        return jax3d.find_contact_point(k, point, jax3d.solve3d(0.0, p, a, i, e, w, lan))

    g = jax.grad(f, argnums=(0, 1, 2, 3))(K, pars[1], pars[2], pars[3])
    h = 1e-6
    for slot, (dk, da, di, de) in enumerate(np.eye(4) * h):
        pp = (pars[0], pars[1] + da, pars[2] + di, pars[3] + de, pars[4], pars[5])
        pm = (pars[0], pars[1] - da, pars[2] - di, pars[3] - de, pars[4], pars[5])
        fd = (_exact_contact(K + dk, point, pp) - _exact_contact(K - dk, point, pm)) / (2 * h)
        assert_allclose(float(g[slot]), fd, rtol=1e-3, atol=1e-7, err_msg=f"slot {slot}")


def test_t14_gradient_under_jit():
    p, a, i, e, w, lan = ORBITS["eccentric"]
    g = jax.jit(jax.grad(lambda a: jax3d.t14(K, jax3d.solve3d(0.0, p, a, i, e, w, lan))))(a)
    h = 1e-5

    def exact_t14(a_):
        pars = (p, a_, i, e, w, lan)
        return _exact_contact(K, 4, pars) - _exact_contact(K, 1, pars)

    # Difference tightly-converged roots: the bisection's 1e-6 day bracket is
    # far coarser than the change in t14 over the step.
    assert_allclose(float(g), (exact_t14(a + h) - exact_t14(a - h)) / (2 * h), rtol=1e-3)


@pytest.mark.parametrize("pars", [ORBITS["eccentric"], (2.5, 8.0, 1.52, 0.1, 0.3, 0.0)],
                         ids=["eccentric", "inclined"])
def test_z_min_derivatives(pars):
    """Needs a non-zero impact parameter: z_min has a kink in i at b = 0."""
    p, a, i, e, w, lan = pars

    def exact(i_):
        c = np.asarray(jax3d.solve3d(0.0, p, a, i_, e, w, lan))
        r = minimize_scalar(lambda s: float(jax3d.sep_c(s, c)) ** 2, bracket=(-0.01, 0.0, 0.01),
                            tol=1e-14)
        return r.x, np.sqrt(r.fun)

    def f(i_):
        return jax3d.find_z_min(0.0, jax3d.solve3d(0.0, p, a, i_, e, w, lan))

    dt, dz = jax.jacfwd(f)(i)
    h = 1e-5
    (tp_, zp), (tm, zm) = exact(i + h), exact(i - h)
    assert_allclose(float(dz), (zp - zm) / (2 * h), rtol=1e-5)
    assert_allclose(float(dt), (tp_ - tm) / (2 * h), rtol=1e-2, atol=1e-7)


def test_z_min_time_derivative_finite_for_central_transit():
    """The squared separation keeps the minimum-time derivative finite at b = 0."""
    p, a, e, w = 3.0, 10.0, 0.0, 0.0
    g = jax.grad(lambda i: jax3d.find_z_min(0.0, jax3d.solve3d(0.0, p, a, i, e, w))[0])(0.5 * jnp.pi)
    assert np.isfinite(float(g))
