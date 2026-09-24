"""JAX backend: multi-expansion-point evaluators against numba.

Values are compared with the numba ``orbit3d`` ``_o`` dispatchers on the same
grid. Gradients are ``jax.jacfwd`` through ``solve3d_orbit`` + evaluator,
compared with the numba ``_od`` kernels in both bases: the periastron basis
(``dcoeffs`` as returned by ``solve3d_orbit_d``) against a model taking
``tpa``, and the transit-centre basis (``dcoeffs`` transformed with
``tp_to_tc_gradient_orbit``) against a model deriving ``tpa`` from ``tc``.
Times span several epochs so the period chain term is exercised.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

jax = pytest.importorskip("jax")

from meepmeep.tests.jax_utils import ORBITS, jacobian, assert_grad_close  # noqa: E402

import jax.numpy as jnp  # noqa: E402

from meepmeep import jax3d, numba3d  # noqa: E402
from meepmeep.backends.numba.utils import eccentricity_vector as nb_eccentricity_vector  # noqa: E402

ORBIT_IDS = list(ORBITS)
TC = 1.7
RSTAR = 0.9


def _setup(name, n=96, seed=3):
    p, a, i, e, w, lan = ORBITS[name]
    ep_times, _, dt, ep_table = numba3d.create_expansion_points(15, max(e, 0.2), 'ea')
    tpa = TC - float(jax3d.mean_anomaly_at_transit(e, w)) / (2 * np.pi) * p
    rng = np.random.default_rng(seed)
    times = TC + rng.uniform(-3.5, 4.5, n) * p
    return (p, a, i, e, w, lan), tpa, dt, ep_table, ep_times, times


# name: (jax fn, numba value fn, numba gradient fn, extras before tpa, extras after tpa (numba layout))
# Standard layout: f(t, *pre, tpa, p, dt, ep_table, ep_times, coeffs).
STANDARD = {
    "pos": (jax3d.pos_o, numba3d.pos_o, numba3d.pos_od, ()),
    "zpos": (jax3d.zpos_o, numba3d.zpos_o, numba3d.zpos_od, ()),
    "sep": (jax3d.sep_o, numba3d.sep_o, numba3d.sep_od, ()),
    "vel": (jax3d.vel_o, numba3d.vel_o, numba3d.vel_od, ()),
    "zvel": (jax3d.zvel_o, numba3d.zvel_o, numba3d.zvel_od, ()),
    "cos_alpha": (jax3d.cos_alpha_o, numba3d.cos_alpha_o, numba3d.cos_alpha_od, ()),
    "distance": (jax3d.star_planet_distance_o, numba3d.star_planet_distance_o,
                 numba3d.star_planet_distance_od, ()),
    "lambert": (jax3d.lambert_phase_curve_o, numba3d.lambert_phase_curve_o, numba3d.lambert_phase_curve_od,
                (0.3, 0.1)),
    "emission": (jax3d.emission_phase_curve_o, numba3d.emission_phase_curve_o,
                 numba3d.emission_phase_curve_od, (0.1, 0.02, 0.4)),
}


def _tup(x):
    return x if isinstance(x, tuple) else (x,)


def _model(jfn, times, pre, dt, ep_table, ep_times, basis):
    """JAX model of (timing, p, a, i, e, w, lan, *pre) in the requested basis."""
    def f(timing, p, a, i, e, w, lan, *pre_args):
        if basis == "tc":
            tpa = timing - jax3d.mean_anomaly_at_transit(e, w) / (2 * jnp.pi) * p
        else:
            tpa = timing
        coeffs = jax3d.solve3d_orbit(ep_times, p, a, i, e, w, lan)
        return jfn(times, *pre_args, tpa, p, dt, ep_table, ep_times, coeffs)
    return f


def _numba_dcoeffs(ep_times, pars, basis):
    """Numba coefficient-derivative tensors in the requested basis."""
    p, a, i, e, w, lan = pars
    coeffs, dcoeffs = numba3d.solve3d_orbit_d(ep_times, p, a, i, e, w, lan)
    if basis == "tc":
        numba3d.tp_to_tc_gradient_orbit(dcoeffs, p, e, w)
    return coeffs, dcoeffs


@pytest.mark.parametrize("name", ORBIT_IDS)
@pytest.mark.parametrize("quantity", list(STANDARD))
class TestStandard:
    def test_values(self, name, quantity):
        jfn, nfn, _, pre = STANDARD[quantity]
        pars, tpa, dt, ep_table, ep_times, times = _setup(name)
        coeffs = numba3d.solve3d_orbit(ep_times, *pars)
        expected = _tup(nfn(times, *pre, tpa, *pars[:1], dt, ep_table, ep_times, coeffs))
        actual = _tup(jfn(times, *pre, tpa, pars[0], dt, ep_table, ep_times, coeffs))
        for act, exp in zip(actual, expected):
            assert_allclose(np.asarray(act), exp, rtol=1e-12, atol=1e-13 * np.abs(exp).max())

    @pytest.mark.parametrize("basis", ["tc", "tp"])
    def test_gradient(self, name, quantity, basis):
        jfn, _, nfn_d, pre = STANDARD[quantity]
        pars, tpa, dt, ep_table, ep_times, times = _setup(name)
        coeffs, dcoeffs = _numba_dcoeffs(ep_times, pars, basis)
        nb = nfn_d(times, *pre, tpa, pars[0], dt, ep_table, ep_times, coeffs, dcoeffs)
        timing = TC if basis == "tc" else tpa
        jac = _tup(jacobian(_model(jfn, times, pre, dt, ep_table, ep_times, basis), 7 + len(pre),
                            timing, *pars, *pre))
        grads = nb[len(jac):]
        for j, g in zip(jac, grads):
            assert_grad_close(j, g)


@pytest.mark.parametrize("name", ORBIT_IDS)
@pytest.mark.parametrize("basis", ["tc", "tp"])
class TestSpecial:
    def test_rv(self, name, basis):
        pars, tpa, dt, ep_table, ep_times, times = _setup(name)
        p, a, i, e, w, lan = pars
        k = 12.0
        coeffs, dcoeffs = _numba_dcoeffs(ep_times, pars, basis)
        rv, drv = numba3d.rv_od(times, k, tpa, p, a, i, e, dt, ep_table, ep_times, coeffs, dcoeffs)
        assert_allclose(np.asarray(jax3d.rv_o(times, k, tpa, p, a, i, e, dt, ep_table, ep_times, coeffs)), rv,
                        rtol=1e-12, atol=1e-12 * k)

        def f(timing, p, a, i, e, w, lan, k):
            tpa_ = timing - jax3d.mean_anomaly_at_transit(e, w) / (2 * jnp.pi) * p if basis == "tc" else timing
            co = jax3d.solve3d_orbit(ep_times, p, a, i, e, w, lan)
            return jax3d.rv_o(times, k, tpa_, p, a, i, e, dt, ep_table, ep_times, co)

        timing = TC if basis == "tc" else tpa
        assert_grad_close(jacobian(f, 8, timing, *pars, k), drv)

    def test_ev_signal(self, name, basis):
        pars, tpa, dt, ep_table, ep_times, times = _setup(name)
        p, a, i, e, w, lan = pars
        alpha, q = 1.3, 1e-3
        coeffs, dcoeffs = _numba_dcoeffs(ep_times, pars, basis)
        ev, dev = numba3d.ev_signal_od(alpha, q, i, times, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)
        assert_allclose(np.asarray(jax3d.ev_signal_o(alpha, q, i, times, tpa, p, dt, ep_table, ep_times, coeffs)),
                        ev, rtol=1e-12, atol=1e-13 * np.abs(ev).max())

        def f(timing, p, a, i, e, w, lan, alpha, q):
            tpa_ = timing - jax3d.mean_anomaly_at_transit(e, w) / (2 * jnp.pi) * p if basis == "tc" else timing
            co = jax3d.solve3d_orbit(ep_times, p, a, i, e, w, lan)
            return jax3d.ev_signal_o(alpha, q, i, times, tpa_, p, dt, ep_table, ep_times, co)

        timing = TC if basis == "tc" else tpa
        assert_grad_close(jacobian(f, 9, timing, *pars, alpha, q), dev)

    def test_light_travel_time(self, name, basis):
        pars, tpa, dt, ep_table, ep_times, times = _setup(name)
        p, a, i, e, w, lan = pars
        coeffs, dcoeffs = _numba_dcoeffs(ep_times, pars, basis)
        ltt, dltt = numba3d.light_travel_time_od(times, tpa, p, e, w, RSTAR, dt, ep_table, ep_times, coeffs,
                                                 dcoeffs, basis == "tc")
        assert_allclose(np.asarray(jax3d.light_travel_time_o(times, tpa, p, e, w, RSTAR, dt, ep_table, ep_times,
                                                             coeffs)),
                        ltt, rtol=1e-11, atol=1e-13 * np.abs(ltt).max())

        def f(timing, p, a, i, e, w, lan):
            tpa_ = timing - jax3d.mean_anomaly_at_transit(e, w) / (2 * jnp.pi) * p if basis == "tc" else timing
            co = jax3d.solve3d_orbit(ep_times, p, a, i, e, w, lan)
            return jax3d.light_travel_time_o(times, tpa_, p, e, w, RSTAR, dt, ep_table, ep_times, co)

        timing = TC if basis == "tc" else tpa
        assert_grad_close(jacobian(f, 7, timing, *pars), dltt)

    def test_cos_v_p_angle(self, name, basis):
        pars, tpa, dt, ep_table, ep_times, times = _setup(name)
        v = np.array([0.3, -1.2, 0.7])
        coeffs, dcoeffs = _numba_dcoeffs(ep_times, pars, basis)
        val, dval = numba3d.cos_v_p_angle_od(v, times, tpa, pars[0], dt, ep_table, ep_times, coeffs, dcoeffs)
        assert_allclose(np.asarray(jax3d.cos_v_p_angle_o(v, times, tpa, pars[0], dt, ep_table, ep_times, coeffs)),
                        val, rtol=1e-12, atol=1e-14)
        f = _model(lambda t, *rest: jax3d.cos_v_p_angle_o(v, t, *rest), times, (), dt, ep_table, ep_times, basis)
        timing = TC if basis == "tc" else tpa
        assert_grad_close(jacobian(f, 7, timing, *pars), dval)

    def test_true_anomaly(self, name, basis):
        """numba treats the eccentricity vector as constant; closing over it as numpy constants does the same."""
        pars, tpa, dt, ep_table, ep_times, times = _setup(name)
        p, a, i, e, w, lan = pars
        ex, ey, ez = nb_eccentricity_vector(i, e, w, lan)
        coeffs, dcoeffs = _numba_dcoeffs(ep_times, pars, basis)
        f_nb, df_nb = numba3d.true_anomaly_od(times, tpa, p, ex, ey, ez, w, dt, ep_table, ep_times, coeffs, dcoeffs,
                                              basis == "tc")
        f_j = jax3d.true_anomaly_o(times, tpa, p, ex, ey, ez, w, dt, ep_table, ep_times, coeffs)
        assert_allclose(np.asarray(f_j), f_nb, rtol=1e-12, atol=1e-12)
        f = _model(lambda t, tpa_, p_, *rest: jax3d.true_anomaly_o(t, tpa_, p_, ex, ey, ez, w, *rest),
                   times, (), dt, ep_table, ep_times, basis)
        timing = TC if basis == "tc" else tpa
        assert_grad_close(jacobian(f, 7, timing, *pars), df_nb)


def test_true_anomaly_matches_newton():
    """With a non-zero node the eccentricity vector must be rotated with the positions."""
    p, a, i, e, w, lan = ORBITS["eccentric"]
    assert lan != 0.0
    ep_times, _, dt, ep_table = jax3d.create_expansion_points(25, e, 'ea')
    tpa = TC - float(jax3d.mean_anomaly_at_transit(e, w)) / (2 * np.pi) * p
    coeffs = jax3d.solve3d_orbit(ep_times, p, a, i, e, w, lan)
    times = TC + np.linspace(-1.3, 2.1, 400) * p
    f = np.asarray(jax3d.true_anomaly_o(times, tpa, p, *jax3d.eccentricity_vector(i, e, w, lan), w,
                                        dt, ep_table, ep_times, coeffs))
    from meepmeep.backends.jax.newton import ta_newton
    f_exact = np.mod(np.asarray(ta_newton(times, TC, p, e, w)), 2 * np.pi)
    d = np.angle(np.exp(1j * (f - f_exact)))
    assert np.abs(d).max() < 1e-4  # truncation error; an unrotated vector misses by ~0.1 rad


def test_circular_true_anomaly_gradient_tc_basis():
    p, a, i, e, w, lan = ORBITS["circular"]
    ep_times, _, dt, ep_table = jax3d.create_expansion_points(15, 0.2, 'ea')
    times = TC + np.array([0.3, 2.6, -1.4]) * p
    ev = jax3d.eccentricity_vector(i, e, w)

    def f(tc, p):
        tpa = tc - jax3d.mean_anomaly_at_transit(e, w) / (2 * jnp.pi) * p
        coeffs = jax3d.solve3d_orbit(ep_times, p, a, i, e, w, lan)
        return jax3d.true_anomaly_o(times, tpa, p, *ev, w, dt, ep_table, ep_times, coeffs)

    h = 1e-6
    fd = np.stack([(np.asarray(f(TC + h, p)) - np.asarray(f(TC - h, p))) / (2 * h),
                   (np.asarray(f(TC, p + h)) - np.asarray(f(TC, p - h))) / (2 * h)], -1)
    assert_allclose(np.asarray(jacobian(f, 2, TC, p)), fd, rtol=1e-7)


def test_true_anomaly_gradient_finite_at_periastron_and_circular():
    p, a, i, e, w, lan = ORBITS["eccentric"]
    ep_times, _, dt, ep_table = jax3d.create_expansion_points(15, e, 'ea')

    def f(tpa, e, i, w):
        coeffs = jax3d.solve3d_orbit(ep_times, p, a, i, e, w, lan)
        ex, ey, ez = jax3d.eccentricity_vector(i, e, w, lan)
        return jax3d.true_anomaly_o(jnp.array([tpa, tpa + p, tpa + 0.3]), tpa, p, ex, ey, ez, w,
                                    dt, ep_table, ep_times, coeffs)

    for e_ in (e, 0.0):
        jac = jax.jacfwd(f, argnums=(0, 1, 2, 3))(0.4, e_, i, w)
        assert all(np.all(np.isfinite(np.asarray(j))) for j in jac)


def test_full_true_anomaly_gradient_is_physical():
    """With the eccentricity vector differentiated too, the true anomaly at fixed
    time and tp depends only on (p, e): the (a, i, w, lan) slots vanish."""
    p, a, i, e, w, lan = ORBITS["high_e"]
    ep_times, _, dt, ep_table = jax3d.create_expansion_points(15, e, 'ea')
    times = jnp.array([0.2, 1.1, 3.9])

    def f(tp, p, a, i, e, w, lan):
        coeffs = jax3d.solve3d_orbit(ep_times, p, a, i, e, w, lan)
        return jax3d.true_anomaly_o(times, tp, p, *jax3d.eccentricity_vector(i, e, w, lan), w, dt, ep_table, ep_times,
                                    coeffs)

    jac = np.asarray(jacobian(f, 7, 0.0, p, a, i, e, w, lan))
    assert_allclose(jac[:, [2, 3, 5, 6]], 0.0, atol=1e-6)
    assert np.all(np.abs(jac[:, [0, 1, 4]]) > 1e-3)


def test_jit_and_vmap_over_parameters():
    p, a, i, e, w, lan = ORBITS["eccentric"]
    ep_times, _, dt, ep_table = jax3d.create_expansion_points(15, e, 'ea')
    times = jnp.linspace(0.0, 2 * p, 50)

    @jax.jit
    def model(pars):
        tc, p, a, i, e, w = pars
        tpa = tc - jax3d.mean_anomaly_at_transit(e, w) / (2 * jnp.pi) * p
        coeffs = jax3d.solve3d_orbit(ep_times, p, a, i, e, w)
        return jax3d.sep_o(times, tpa, p, dt, ep_table, ep_times, coeffs)

    batch = jnp.array([[TC, p, a, i, e, w], [TC + 0.1, p, a + 1.0, i, e, w], [TC, p * 1.1, a, i, 0.5, 2.0]])
    out = jax.vmap(model)(batch)
    for k in range(3):
        assert_allclose(np.asarray(out[k]), np.asarray(model(batch[k])), rtol=1e-14)
    g = jax.jit(jax.grad(lambda q: jnp.sum(model(q) ** 2)))(batch[0])
    assert np.all(np.isfinite(np.asarray(g)))


def test_ep_ix_matches_numba():
    pars, tpa, dt, ep_table, ep_times, times = _setup("high_e", n=500)
    expected = np.array([numba3d.ep_ix(t, tpa, pars[0], dt, ep_table) for t in times])
    assert_allclose(np.asarray(jax3d.ep_ix(times, tpa, pars[0], dt, ep_table)), expected, rtol=0, atol=0)


def test_nan_time_does_not_read_out_of_bounds():
    pars, tpa, dt, ep_table, ep_times, _ = _setup("eccentric")
    coeffs = jax3d.solve3d_orbit(ep_times, *pars)
    out = np.asarray(jax3d.sep_o(jnp.array([np.nan, TC]), tpa, pars[0], dt, ep_table, ep_times, coeffs))
    assert np.isnan(out[0]) and np.isfinite(out[1])


def test_period_gradient_in_periodic_image_segment():
    """Times just before periastron are served by the periodic-image expansion point;
    the period derivative there must match finite differences of the values."""
    p, a, i, e, w, lan = ORBITS["eccentric"]
    ep_times, _, dt, ep_table = numba3d.create_expansion_points(15, e, 'ea')
    tpa = -0.2
    times = np.array([tpa + 0.99 * p, tpa + 2.985 * p, tpa + 0.5 * p])
    assert list(np.asarray(jax3d.ep_ix(times, tpa, p, dt, ep_table))) == [14, 14, 7]

    def f(p):
        return jax3d.sep_o(times, tpa, p, dt, ep_table, ep_times, jax3d.solve3d_orbit(ep_times, p, a, i, e, w, lan))

    h = 1e-6
    fd = (np.asarray(f(p + h)) - np.asarray(f(p - h))) / (2 * h)
    assert_allclose(np.asarray(jax.jacfwd(f)(p)), fd, rtol=1e-6)
