"""JAX backend: Kepler solver and Taylor coefficient solvers against numba."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

jax = pytest.importorskip("jax")

from meepmeep.tests.jax_utils import ORBITS, assert_grad_close  # noqa: E402

import jax.numpy as jnp  # noqa: E402

from meepmeep.backends.jax.newton import ea_from_ma  # noqa: E402
from meepmeep.backends.jax.solve import solve2d, solve3d, solve3d_orbit  # noqa: E402
from meepmeep.backends.jax._common import require_x64  # noqa: E402
from meepmeep.backends.numba.newton.newton import ea_from_ma as nb_ea_from_ma  # noqa: E402
from meepmeep.numba2d import solve2d as nb_solve2d, solve2d_d as nb_solve2d_d  # noqa: E402
from meepmeep.numba3d import (solve3d as nb_solve3d, solve3d_d as nb_solve3d_d,  # noqa: E402
                              solve3d_orbit as nb_solve3d_orbit, create_expansion_points as nb_cep)

ORBIT_IDS = list(ORBITS)


class TestKepler:
    @pytest.mark.parametrize("e", [0.0, 0.1, 0.5, 0.8, 0.85, 0.95, 0.99])
    def test_matches_numba(self, e):
        ma = np.linspace(0.0, 2 * np.pi, 301)
        expected = np.array([nb_ea_from_ma(m, e) for m in ma])
        assert_allclose(np.asarray(ea_from_ma(ma, e)), expected, rtol=0, atol=1e-13)

    def test_solves_kepler(self):
        ma = np.linspace(-1.0, 7.0, 101)[:, None]
        e = np.array([0.0, 0.3, 0.9, 0.97])[None, :]
        ea = np.asarray(ea_from_ma(ma, e))
        assert_allclose(ea - e * np.sin(ea), np.broadcast_to(ma, ea.shape), atol=1e-12)

    def test_implicit_derivatives(self):
        ma, e = 1.3, 0.6
        ea = float(ea_from_ma(ma, e))
        dma, de = jax.grad(ea_from_ma, argnums=(0, 1))(ma, e)
        assert_allclose(dma, 1.0 / (1.0 - e * np.cos(ea)), rtol=1e-14)
        assert_allclose(de, np.sin(ea) / (1.0 - e * np.cos(ea)), rtol=1e-14)

    def test_reverse_mode_through_array(self):
        ma = jnp.linspace(0.1, 6.0, 50)
        g = jax.grad(lambda e: jnp.sum(ea_from_ma(ma, e)))(0.4)
        h = 1e-6
        fd = (np.sum(ea_from_ma(ma, 0.4 + h)) - np.sum(ea_from_ma(ma, 0.4 - h))) / (2 * h)
        assert_allclose(g, fd, rtol=1e-7)


@pytest.mark.parametrize("name", ORBIT_IDS)
@pytest.mark.parametrize("te", [0.0, -0.37, 1.9])
class TestSolve:
    def test_solve3d_values(self, name, te):
        pars = (te,) + ORBITS[name]
        expected = nb_solve3d(*pars)
        assert_allclose(np.asarray(solve3d(*pars)), expected, rtol=1e-13, atol=1e-13 * np.abs(expected).max())

    def test_solve2d_values(self, name, te):
        pars = (te,) + ORBITS[name]
        expected = nb_solve2d(*pars)
        assert_allclose(np.asarray(solve2d(*pars)), expected, rtol=1e-13, atol=1e-13 * np.abs(expected).max())

    @pytest.mark.parametrize("solver, nb_solver_d", [(solve3d, nb_solve3d_d), (solve2d, nb_solve2d_d)],
                             ids=["3d", "2d"])
    def test_jacobian_matches_numba_d(self, name, te, solver, nb_solver_d):
        """Slots 1..6 (p, a, i, e, w, lan) of the numba derivative tensor are exact
        coefficient derivatives at fixed te; slot 0 is the polynomial timing row,
        which has no coefficient-level autodiff counterpart."""
        pars = (te,) + ORBITS[name]
        _, dc = nb_solver_d(*pars)
        jac = jax.jacfwd(solver, argnums=tuple(range(1, 7)))(*pars)
        assert_grad_close(np.stack([np.asarray(j) for j in jac], -1), np.moveaxis(dc[1:], 0, -1), rtol=1e-10)


def test_solve_batched_te():
    pars = ORBITS["high_e"]
    te = np.linspace(-2.0, 2.0, 9)
    batched = np.asarray(solve3d(te, *pars))
    assert batched.shape == (9, 3, 5)
    for k, t in enumerate(te):
        assert_allclose(batched[k], np.asarray(solve3d(t, *pars)), rtol=0, atol=0)


@pytest.mark.parametrize("name", ORBIT_IDS)
@pytest.mark.parametrize("placement", ["ea", "ta", "mm"])
def test_solve3d_orbit_matches_numba(name, placement):
    p, a, i, e, w, lan = ORBITS[name]
    ep_times, _, _, _ = nb_cep(15, max(e, 0.2), placement)
    expected = nb_solve3d_orbit(ep_times, p, a, i, e, w, lan)
    actual = np.asarray(solve3d_orbit(ep_times, p, a, i, e, w, lan))
    assert actual.shape == (15, 3, 5)
    assert_allclose(actual, expected, rtol=1e-12, atol=1e-13 * np.abs(expected).max())


def test_require_x64_raises_in_single_precision():
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="double precision"):
            require_x64()
    finally:
        jax.config.update("jax_enable_x64", True)
    require_x64()
