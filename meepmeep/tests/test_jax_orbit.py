"""JAX backend: the high-level ``JaxOrbit`` pytree against ``meepmeep.Orbit``.

The numba orbit's own grid is passed to ``JaxOrbit`` so both evaluate the
same polynomials.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

jax = pytest.importorskip("jax")

from meepmeep.tests.jax_utils import ORBITS, jacobian, assert_grad_close  # noqa: E402

import jax.numpy as jnp  # noqa: E402

from meepmeep import Orbit  # noqa: E402
from meepmeep.jax3d import JaxOrbit, create_expansion_points  # noqa: E402

ORBIT_IDS = list(ORBITS)
TC = 1.7
NPT = 15

# method: (numba call, jax call, extra parameters appended to the gradient)
METHODS = {
    "xyz": (lambda o: o.xyz(), lambda o, t: o.xyz(t), ()),
    "vxyz": (lambda o: o.vxyz(), lambda o, t: o.vxyz(t), ()),
    "cos_phase": (lambda o: o.cos_phase(), lambda o, t: o.cos_phase(t), ()),
    "phase": (lambda o: o.phase(), lambda o, t: o.phase(t), ()),
    "theta": (lambda o: o.theta(), lambda o, t: o.theta(t), ()),
    "distance": (lambda o: o.star_planet_distance(), lambda o, t: o.star_planet_distance(t), ()),
    "ltt": (lambda o: o.light_travel_time(0.8), lambda o, t: o.light_travel_time(t, 0.8), ()),
    "rv": (lambda o, k=11.0: o.radial_velocity(k), lambda o, t, k=11.0: o.radial_velocity(t, k), (11.0,)),
    "lambert": (lambda o, ag=0.3, k=0.1: o.lambert_phase_curve(k, ag),
                lambda o, t, ag=0.3, k=0.1: o.lambert_phase_curve(t, k, ag), (0.3, 0.1)),
    "emission": (lambda o, k=0.1, fr=0.02, off=0.4: o.emission_phase_curve(k, fr, off),
                 lambda o, t, k=0.1, fr=0.02, off=0.4: o.emission_phase_curve(t, k, fr, off), (0.1, 0.02, 0.4)),
    "ev": (lambda o, al=1.3, q=1e-3: o.ellipsoidal_variation(al, q),
           lambda o, t, al=1.3, q=1e-3: o.ellipsoidal_variation(t, al, q), (1.3, 1e-3)),
}


def _tup(x):
    return x if isinstance(x, tuple) else (x,)


def _numba_orbit(name, timing, times, derivatives=False):
    p, a, i, e, w, lan = ORBITS[name]
    orbit = Orbit(npt=NPT, derivatives=derivatives)
    orbit.set_pars(**{timing: TC}, p=p, a=a, i=i, e=e, w=w, lan=lan)
    orbit.set_data(times)
    return orbit


def _grid(orbit):
    return orbit._ep_times, orbit._change_times, orbit._dt, orbit._ep_table


def _times(name, n=80):
    rng = np.random.default_rng(5)
    return TC + rng.uniform(-2.5, 3.5, n) * ORBITS[name][0]


@pytest.mark.parametrize("name", ORBIT_IDS)
@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("timing", ["tc", "tp"])
def test_values(name, method, timing):
    times = _times(name)
    nb = _numba_orbit(name, timing, times)
    ctor = JaxOrbit.from_tc if timing == "tc" else JaxOrbit.from_tp
    jo = ctor(TC, *ORBITS[name], grid=_grid(nb))
    nfn, jfn, _ = METHODS[method]
    for act, exp in zip(_tup(jfn(jo, times)), _tup(nfn(nb))):
        assert_allclose(np.asarray(act), exp, rtol=1e-11, atol=1e-12 * max(np.abs(exp).max(), 1e-300))


@pytest.mark.parametrize("name", ORBIT_IDS)
@pytest.mark.parametrize("method", list(METHODS))
@pytest.mark.parametrize("timing", ["tc", "tp"])
def test_gradients(name, method, timing):
    times = _times(name)
    nb = _numba_orbit(name, timing, times, derivatives=True)
    grid = _grid(nb)
    nfn, jfn, extras = METHODS[method]
    ctor = JaxOrbit.from_tc if timing == "tc" else JaxOrbit.from_tp

    def f(t0, p, a, i, e, w, lan, *extra):
        return jfn(ctor(t0, p, a, i, e, w, lan, grid=grid), times, *extra)

    nb_out = _tup(nfn(nb))
    jac = _tup(jacobian(f, 7 + len(extras), TC, *ORBITS[name], *extras))
    for j, g in zip(jac, nb_out[len(jac):]):
        assert_grad_close(j, g, rtol=1e-8)


@pytest.mark.parametrize("name", ["eccentric", "high_e"])
@pytest.mark.parametrize("timing", ["tc", "tp"])
def test_true_anomaly(name, timing):
    """Values match numba and the exact solution (lan != 0 in both orbits); the
    gradient matches numba in the (tc|tp, p, a) slots, where numba's constant
    eccentricity vector makes no difference."""
    times = _times(name)
    nb = _numba_orbit(name, timing, times, derivatives=True)
    ctor = JaxOrbit.from_tc if timing == "tc" else JaxOrbit.from_tp
    grid = _grid(nb)
    jo = ctor(TC, *ORBITS[name], grid=grid)
    f_nb, df_nb = nb.true_anomaly()
    assert_allclose(np.asarray(jo.true_anomaly(times)), f_nb, rtol=1e-11, atol=1e-11)
    exact = np.mod(_numba_orbit(name, timing, times).true_anomaly(exact=True), 2 * np.pi)
    # Truncation error at e = 0.7 near periastron is a few 1e-3 rad; an eccentricity
    # vector that ignores the node would miss by the node angle (0.4 rad here).
    assert np.abs(np.angle(np.exp(1j * (f_nb - exact)))).max() < 1e-2

    def f(t0, p, a):
        return ctor(t0, p, a, *ORBITS[name][2:], grid=grid).true_anomaly(times)

    assert_grad_close(jacobian(f, 3, TC, *ORBITS[name][:2]), df_nb[:, :3], rtol=1e-8)


def test_mean_anomaly():
    times = _times("eccentric")
    nb = _numba_orbit("eccentric", "tc", times)
    jo = JaxOrbit.from_tc(TC, *ORBITS["eccentric"])
    assert_allclose(np.asarray(jo.mean_anomaly(times)), nb.mean_anomaly(), rtol=1e-13, atol=1e-13)


def test_default_grid_follows_numba_floor():
    """Without a grid the placement is built for max(e, 0.2), like meepmeep.Orbit after a rebuild."""
    p, a, i, e, w, lan = ORBITS["high_e"]
    jo = JaxOrbit.from_tc(TC, p, a, i, e, w, lan)
    expected = create_expansion_points(NPT, e, 'ea')[0]
    assert_allclose(np.asarray(jo.ep_times), np.asarray(expected), atol=0)
    jo0 = JaxOrbit.from_tc(TC, *ORBITS["circular"])
    assert_allclose(np.asarray(jo0.ep_times), np.asarray(create_expansion_points(NPT, 0.2, 'ea')[0]), atol=0)


def test_grid_is_not_differentiated():
    """The default grid is placed from a stop-gradient eccentricity: d(ep_times)/de = 0."""
    p, a, i, e, w, lan = ORBITS["high_e"]
    g = jax.jacfwd(lambda e: JaxOrbit.from_tc(TC, p, a, i, e, w, lan).ep_times)(e)
    assert_allclose(np.asarray(g), 0.0, atol=0)


def test_pytree_jit_vmap_grad():
    p, a, i, e, w, lan = ORBITS["eccentric"]
    times = jnp.linspace(TC - 0.2, TC + 0.2, 64)
    z_obs = JaxOrbit.from_tc(TC, p, a, i, e, w, lan).projected_separation(times)

    @jax.jit
    def evaluate(orbit):
        return orbit.projected_separation(times)

    orbit = JaxOrbit.from_tc(TC, p, a, i, e, w, lan)
    assert_allclose(np.asarray(evaluate(orbit)), np.asarray(z_obs), rtol=1e-14)

    batch = jax.vmap(lambda tc, a: JaxOrbit.from_tc(tc, p, a, i, e, w, lan))(jnp.array([TC, TC + 0.01]),
                                                                            jnp.array([a, a + 0.5]))
    assert batch.coeffs.shape == (2, NPT, 3, 5)
    assert jax.vmap(evaluate)(batch).shape == (2, 64)

    def loglike(theta):
        o = JaxOrbit.from_tc(*theta, lan)
        return -0.5 * jnp.sum((o.projected_separation(times) - z_obs) ** 2)

    theta = jnp.array([TC + 1e-3, p, a, i, e, w])
    g = jax.jit(jax.grad(loglike))(theta)
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(jnp.abs(g[0])) > 0.0


def test_true_anomaly_full_gradient_matches_finite_differences():
    p, a, i, e, w, lan = ORBITS["high_e"]
    times = jnp.array([TC + 0.3, TC + 1.9, TC - 2.2])
    grid = create_expansion_points(NPT, e, 'ea')

    def f(i, e, w, lan):
        return JaxOrbit.from_tp(0.0, p, a, i, e, w, lan, grid=grid).true_anomaly(times)

    jac = np.asarray(jacobian(f, 4, i, e, w, lan))
    h = 1e-6
    args = np.array([i, e, w, lan])
    for k in range(4):
        dp, dm = args.copy(), args.copy()
        dp[k] += h
        dm[k] -= h
        fd = (np.asarray(f(*dp)) - np.asarray(f(*dm))) / (2 * h)
        assert_allclose(jac[:, k], fd, rtol=1e-5, atol=1e-7, err_msg=f"slot {k}")
