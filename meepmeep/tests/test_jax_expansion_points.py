"""JAX backend: expansion-point placement against the numba (scipy brentq) version."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

jax = pytest.importorskip("jax")

from meepmeep.tests import jax_utils  # noqa: E402, F401  (enables x64)

import jax.numpy as jnp  # noqa: E402

from meepmeep.backends.jax.expansion_points import create_expansion_points  # noqa: E402
from meepmeep.backends.numba.expansion_points import create_expansion_points as nb_create  # noqa: E402


@pytest.mark.parametrize("quantity", ["mm", "ea", "ta"])
@pytest.mark.parametrize("n_ep", [7, 15, 25])
@pytest.mark.parametrize("e", [0.0, 0.2, 0.5, 0.8, 0.95])
def test_matches_numba(quantity, n_ep, e):
    expected = nb_create(n_ep, e, quantity)
    ep_times, change_times, dt, ep_table = create_expansion_points(n_ep, e, quantity)
    # brentq's default xtol is 2e-12
    assert_allclose(np.asarray(ep_times), expected[0], rtol=0, atol=1e-11)
    assert_allclose(np.asarray(change_times), expected[1], rtol=0, atol=1e-11)
    assert dt == expected[2]
    # A bin centre that coincides with a change time is decided by round-off:
    # brentq's root and the closed form can fall on opposite sides of it. Only
    # such ties may differ.
    table = np.asarray(ep_table)
    assert table.size == expected[3].size
    centres = (np.arange(len(table)) + 0.5) * dt
    ties = np.min(np.abs(centres[:, None] - expected[1][None, :]), axis=1) < 1e-10
    assert_array_equal(table[~ties], expected[3][~ties])
    assert np.all(np.abs(table - expected[3]) <= 1)
    assert table.dtype == np.int32


@pytest.mark.parametrize("quantity", ["mm", "ea", "ta"])
@pytest.mark.parametrize("n_ep", [15, 35])
@pytest.mark.parametrize("e", [0.0, 0.5, 0.9, 0.97])
def test_default_size_matches_numba(quantity, n_ep, e):
    from meepmeep.backends.jax.expansion_points import expansion_table_size
    from meepmeep.backends.numba.expansion_points import expansion_table_size as nb_size
    assert expansion_table_size(n_ep, e, quantity) == nb_size(n_ep, e, quantity)


@pytest.mark.parametrize("quantity", ["ea", "ta"])
def test_traced_table_matches_eager_up_to_e_0_9(quantity):
    """A traced e cannot size the table, so it is sized for e = 0.9, which is the
    size for every e up to 0.9: jitted and eager tables agree bin for bin."""
    grid = jax.jit(lambda e: create_expansion_points(15, e, quantity))
    for e in (0.2, 0.6, 0.9):
        traced, eager = grid(e), create_expansion_points(15, e, quantity)
        assert traced[2] == eager[2]
        assert_array_equal(np.asarray(traced[3]), np.asarray(eager[3]))


def test_jax_orbit_accuracy_at_high_eccentricity():
    """With the default table, the whole-orbit error keeps shrinking with npt at
    e = 0.9 (it stalled at ~0.1 R_star with a 200-bin table)."""
    from meepmeep.jax3d import JaxOrbit
    from meepmeep.backends.numba.newton.newton import xyz_newton_v
    p, a, i, e, w = 3.0, 8.5, np.radians(88.0), 0.9, np.radians(60.0)
    t = np.linspace(-2.0 * p, 3.0 * p, 100_001)
    x, y, z = (np.asarray(v) for v in JaxOrbit.from_tc(0.0, p, a, i, e, w, npt=35).xyz(jnp.asarray(t)))
    rx, ry, rz = xyz_newton_v(t, 0.0, p, a, i, e, w)
    assert np.max(np.sqrt((x - rx) ** 2 + (y - ry) ** 2 + (z - rz) ** 2)) < 1e-3


def test_traceable_in_eccentricity():
    @jax.jit
    def grid(e):
        return create_expansion_points(15, e, 'ea')

    for e in (0.1, 0.6):
        for act, exp in zip(grid(e), nb_create(15, e, 'ea')):
            assert_allclose(np.asarray(act), exp, atol=1e-11)


def test_placement_is_differentiable():
    g = jax.jacfwd(lambda e: create_expansion_points(15, e, 'ta')[0])(0.4)
    assert np.all(np.isfinite(np.asarray(g)))
    # The midpoint and the periodic endpoints do not move with e.
    assert_allclose(np.asarray(g)[[0, 7, 14]], 0.0, atol=0)


def test_periodic_image_contract():
    ep_times = np.asarray(create_expansion_points(15, 0.3, 'ea')[0])
    assert ep_times[0] == 0.0 and ep_times[-1] == 1.0 and ep_times[7] == 0.5


@pytest.mark.parametrize("kwargs, match", [(dict(n_ep=14, e=0.1), "odd"),
                                           (dict(n_ep=15, e=0.1, quantity="xx"), "Quantity")])
def test_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        create_expansion_points(**kwargs)


def test_int32_table_indexes_like_numba_table():
    _, _, dt, table = create_expansion_points(15, 0.3, 'ea')
    assert jnp.issubdtype(table.dtype, jnp.integer)
    assert int(table[0]) == 0 and int(table[-1]) == 14
