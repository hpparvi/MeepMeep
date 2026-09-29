"""JAX backend in single precision.

The backend computes in the floating dtype of its inputs (see
``meepmeep.backends.jax._common.working_dtype``). These tests pin that rule
(float32 in, float32 out, through the expansion-point grid and ``JaxOrbit``
too), the float32 accuracy against the float64 numba backend, and that the
iterative solvers converge in float32 instead of running to their caps.

Accuracy is measured on identical inputs: the numba reference is evaluated at
the float32-rounded times and parameters cast back to float64, so the
comparison isolates float32 arithmetic from input rounding. Times use a small
origin, as the float32 contract requires (see docs/source/jax_backend.rst).
"""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose

jax = pytest.importorskip("jax")

from meepmeep.tests.jax_utils import ORBITS, jacobian  # noqa: E402  (enables x64)

import jax.numpy as jnp  # noqa: E402

from meepmeep import jax2d, jax3d, numba2d, numba3d  # noqa: E402
from meepmeep.backends.numba.utils import eccentricity_vector as nb_eccentricity_vector  # noqa: E402
from meepmeep.backends.jax._common import EA_TOLERANCE, working_dtype  # noqa: E402
from meepmeep.backends.jax.newton import _ea_newton_loop  # noqa: E402

F32 = jnp.dtype(jnp.float32)
F64 = jnp.dtype(jnp.float64)
ORBIT_IDS = list(ORBITS)
TC = 1.7
K = 0.1


def f32(x):
    """``x`` as a float32 JAX array."""
    return jnp.asarray(x, jnp.float32)


class TestWorkingDtype:
    def test_float32_with_python_scalars_stays_float32(self):
        assert working_dtype(f32(1.0), 2.0, 3) == F32

    def test_python_scalars_give_the_default_float(self):
        assert working_dtype(1.0, 2.0) == F64  # jax_utils enables x64 for this suite

    def test_explicit_float64_array_promotes(self):
        assert working_dtype(f32(np.ones(3)), np.ones(3)) == F64

    def test_integers_fall_back_to_the_default_float(self):
        assert working_dtype(jnp.arange(3)) == F64
        assert working_dtype(jnp.arange(3, dtype=jnp.int32), f32(1.0)) == F32

    def test_lists_count_element_by_element(self):
        assert working_dtype([0.0, 0.0, 1.0], f32(1.0)) == F32

    def test_default_float_follows_x64(self):
        with jax.enable_x64(False):
            assert working_dtype(1.0) == F32

    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.complex64], ids=["f16", "bf16", "c64"])
    def test_unsupported_dtypes_raise(self, dtype):
        with pytest.raises(TypeError, match="float32 or float64"):
            working_dtype(jnp.ones(3, dtype))

    def test_solver_rejects_half_precision(self):
        with pytest.raises(TypeError, match="float32 or float64"):
            jax3d.solve3d(jnp.float16(0.0), 3.0, 10.0, 1.5, 0.0, 0.0)


class TestKeplerFloat32:
    @pytest.mark.parametrize("e", [0.0, 0.3, 0.7, 0.9, 0.95])
    def test_converges_before_the_cap(self, e):
        ma = jnp.linspace(0.0, 2 * np.pi, 301, dtype=jnp.float32)
        ea, steps = _ea_newton_loop(ma, f32(e))
        assert ea.dtype == F32
        assert int(steps.max()) < 20  # the cap is 50
        ea64 = np.asarray(ea, np.float64)
        residual = ea64 - float(f32(e)) * np.sin(ea64) - np.asarray(ma, np.float64)
        assert np.abs(residual).max() < 2e-6

    def test_float64_keeps_the_numba_tolerance(self):
        assert EA_TOLERANCE[F64] == 1e-13
        ea, steps = _ea_newton_loop(jnp.linspace(0.0, 2 * np.pi, 301), 0.5)
        assert ea.dtype == F64 and int(steps.max()) < 50

    def test_implicit_derivative_in_float32(self):
        ma, e = f32(1.3), f32(0.6)
        dma, de = jax.grad(jax3d.ea_from_ma, argnums=(0, 1))(ma, e)
        ea = float(jax3d.ea_from_ma(ma, e))
        e64 = float(e)
        assert dma.dtype == de.dtype == F32
        assert_allclose(float(dma), 1.0 / (1.0 - e64 * np.cos(ea)), rtol=1e-5)
        assert_allclose(float(de), np.sin(ea) / (1.0 - e64 * np.cos(ea)), rtol=1e-5)


# ---------------------------------------------------------------------------
# Dtype propagation: float32 in, float32 out, over the whole public surface
# ---------------------------------------------------------------------------

# Public names whose outputs are not floating point (JaxOrbit has its own tests).
NOT_FLOATING = {"JaxOrbit", "expansion_table_size", "ep_ix"}
DURATIONS = ("t1", "t4", "t12", "t14", "t23", "t34")


@pytest.fixture(scope="module")
def ctx():
    """float32 inputs for every public evaluator: an eccentric orbit, times, and both expansions."""
    p, a, i, e, w, lan = (f32(v) for v in ORBITS["eccentric"])
    tc = f32(TC)
    ep_times, _, dt, ep_table = jax3d.create_expansion_points(15, e)
    coeffs = jax3d.solve3d_orbit(ep_times, p, a, i, e, w, lan)
    return dict(p=p, a=a, i=i, e=e, w=w, lan=lan, tc=tc, k=f32(K),
                t=f32(TC + np.linspace(-1.2, 2.3, 17) * float(p)),
                tr=f32(np.linspace(-0.1, 0.1, 17)),
                c2=jax2d.solve2d(0.0, p, a, i, e, w, lan),
                c3=jax3d.solve3d(0.0, p, a, i, e, w, lan),
                tpa=tc - jax3d.mean_anomaly_at_transit(e, w) / (2 * jnp.pi) * p,
                ep_times=ep_times, grid=(dt, ep_table, ep_times, coeffs))


def _util_calls(agg, c):
    calls = {
        "find_contact_point": lambda x: agg.find_contact_point(x["k"], 1, x[c]),
        "bounding_box": lambda x: agg.bounding_box(x["k"], x[c]),
        "find_z_min": lambda x: agg.find_z_min(0.0, x[c]),
    }
    calls.update({n: (lambda x, n=n: getattr(agg, n)(x["k"], x[c])) for n in DURATIONS})
    return calls


def _point_calls(agg, c, names):
    calls = {}
    for q in names:
        calls[f"{q}_c"] = lambda x, q=q: getattr(agg, f"{q}_c")(x["tr"], x[c])
        calls[q] = lambda x, q=q: getattr(agg, q)(x["t"], x["tc"], x["p"], x[c])
    return calls


def _orbit_calls(names):
    return {n: (lambda x, n=n: getattr(jax3d, n)(x["t"], x["tpa"], x["p"], *x["grid"])) for n in names}


CALLS_2D = {
    "solve2d": lambda x: jax2d.solve2d(0.0, x["p"], x["a"], x["i"], x["e"], x["w"], x["lan"]),
    **_point_calls(jax2d, "c2", ("pos", "sep")),
    **_util_calls(jax2d, "c2"),
}

CALLS_3D = {
    "solve3d": lambda x: jax3d.solve3d(0.0, x["p"], x["a"], x["i"], x["e"], x["w"], x["lan"]),
    "solve3d_orbit": lambda x: jax3d.solve3d_orbit(x["ep_times"], x["p"], x["a"], x["i"], x["e"], x["w"],
                                                   x["lan"]),
    "create_expansion_points": lambda x: jax3d.create_expansion_points(15, x["e"])[:2],
    "ea_from_ma": lambda x: jax3d.ea_from_ma(x["tr"] + 1.0, x["e"]),
    "eccentricity_vector": lambda x: jax3d.eccentricity_vector(x["i"], x["e"], x["w"], x["lan"]),
    "eclipse_time_offset": lambda x: jax3d.eclipse_time_offset(x["p"], x["i"], x["e"], x["w"]),
    "mean_anomaly_at_transit": lambda x: jax3d.mean_anomaly_at_transit(x["e"], x["w"]),
    "rv_c": lambda x: jax3d.rv_c(x["tr"], 11.0, x["p"], x["a"], x["i"], x["e"], x["c3"]),
    "rv": lambda x: jax3d.rv(x["t"], 11.0, x["tc"], x["p"], x["a"], x["i"], x["e"], x["c3"]),
    "rv_o": lambda x: jax3d.rv_o(x["t"], 11.0, x["tpa"], x["p"], x["a"], x["i"], x["e"], *x["grid"]),
    "lambert_phase_curve_c": lambda x: jax3d.lambert_phase_curve_c(x["tr"], 0.3, K, x["c3"]),
    "lambert_phase_curve": lambda x: jax3d.lambert_phase_curve(x["t"], 0.3, K, x["tc"], x["p"], x["c3"]),
    "lambert_phase_curve_o": lambda x: jax3d.lambert_phase_curve_o(x["t"], 0.3, K, x["tpa"], x["p"], *x["grid"]),
    "ev_signal_c": lambda x: jax3d.ev_signal_c(x["tr"], 1.3, 1e-3, x["i"], x["c3"]),
    "ev_signal": lambda x: jax3d.ev_signal(x["t"], 1.3, 1e-3, x["i"], x["tc"], x["p"], x["c3"]),
    "ev_signal_o": lambda x: jax3d.ev_signal_o(1.3, 1e-3, x["i"], x["t"], x["tpa"], x["p"], *x["grid"]),
    "emission_phase_curve_c": lambda x: jax3d.emission_phase_curve_c(x["tr"], K, 0.02, 0.4, x["c3"]),
    "emission_phase_curve": lambda x: jax3d.emission_phase_curve(x["t"], K, 0.02, 0.4, x["tc"], x["p"],
                                                                 x["c3"]),
    "emission_phase_curve_o": lambda x: jax3d.emission_phase_curve_o(x["t"], K, 0.02, 0.4, x["tpa"], x["p"],
                                                                     *x["grid"]),
    "light_travel_time_o": lambda x: jax3d.light_travel_time_o(x["t"], x["tpa"], x["p"], x["e"], x["w"], 0.9,
                                                               *x["grid"]),
    # A plain list: its Python floats must not promote the computation.
    "cos_v_p_angle_o": lambda x: jax3d.cos_v_p_angle_o([0.3, -1.2, 0.7], x["t"], x["tpa"], x["p"], *x["grid"]),
    "true_anomaly_o": lambda x: jax3d.true_anomaly_o(
        x["t"], x["tpa"], x["p"], *jax3d.eccentricity_vector(x["i"], x["e"], x["w"], x["lan"]), x["w"],
        *x["grid"]),
    **_point_calls(jax3d, "c3", ("pos", "zpos", "sep", "vel", "zvel", "cos_alpha")),
    **_orbit_calls(("pos_o", "zpos_o", "sep_o", "vel_o", "zvel_o", "cos_alpha_o", "star_planet_distance_o")),
    **_util_calls(jax3d, "c3"),
}

CALL_CASES = ([pytest.param(f, id=f"jax2d.{n}") for n, f in CALLS_2D.items()]
              + [pytest.param(f, id=f"jax3d.{n}") for n, f in CALLS_3D.items()])


def _floating_leaves(out):
    leaves = [jnp.asarray(leaf) for leaf in jax.tree_util.tree_leaves(out)]
    return [leaf for leaf in leaves if jnp.issubdtype(leaf.dtype, jnp.floating)]


@pytest.mark.parametrize("agg, calls", [(jax2d, CALLS_2D), (jax3d, CALLS_3D)], ids=["jax2d", "jax3d"])
def test_every_public_function_has_a_float32_call(agg, calls):
    """A new public function must get a float32 call below, or this fails."""
    missing = set(agg.__all__) - set(calls) - NOT_FLOATING
    assert not missing, f"public functions without a float32 dtype test: {sorted(missing)}"


@pytest.mark.parametrize("call", CALL_CASES)
def test_float32_in_float32_out(ctx, call):
    leaves = _floating_leaves(call(ctx))
    assert leaves
    assert all(leaf.dtype == F32 for leaf in leaves), [str(leaf.dtype) for leaf in leaves]


def test_ep_ix_runs_in_float32(ctx):
    dt, ep_table, _, _ = ctx["grid"]
    ix = jax3d.ep_ix(ctx["t"], ctx["tpa"], ctx["p"], dt, ep_table)
    assert jnp.issubdtype(ix.dtype, jnp.integer)


@pytest.mark.parametrize("quantity", ["mm", "ea", "ta"])
def test_grid_follows_the_eccentricity_dtype(quantity):
    ep_times, change_times, _, ep_table = jax3d.create_expansion_points(15, f32(0.3), quantity)
    assert ep_times.dtype == change_times.dtype == F32
    assert ep_table.dtype == jnp.int32
    ep64 = jax3d.create_expansion_points(15, 0.3, quantity)[0]
    assert ep64.dtype == F64
    assert_allclose(np.asarray(ep_times), np.asarray(ep64), atol=1e-6)


def test_python_float_parameters_follow_float32_arrays():
    c = jax3d.solve3d(f32(0.0), 3.0, 10.0, 1.5, 0.1, 0.3)
    assert c.dtype == F32
    assert jax3d.sep(f32(0.05), 0.0, 3.0, c).dtype == F32


def test_float64_numpy_times_promote():
    """Standard JAX promotion, documented rather than fought: a NumPy float64 array is strongly typed."""
    c = jax3d.solve3d(f32(0.0), 3.0, 10.0, 1.5, 0.1, 0.3)
    assert jax3d.sep(np.linspace(-0.1, 0.1, 5), 0.0, 3.0, c).dtype == F64


def test_integer_inputs_use_the_default_float():
    assert jax3d.solve3d(0, 3, 10, 1, 0, 0).dtype == F64


def test_jit_retraces_per_dtype():
    f = jax.jit(lambda t, p: jax3d.sep(t, 0.0, p, jax3d.solve3d(0.0, p, 10.0, 1.5, 0.1, 0.3)))
    assert f(f32(0.05), f32(3.0)).dtype == F32
    assert f(jnp.float64(0.05), jnp.float64(3.0)).dtype == F64
    assert f(f32(0.05), f32(3.0)).dtype == F32


# ---------------------------------------------------------------------------
# JaxOrbit
# ---------------------------------------------------------------------------

JAXORBIT_METHODS = {
    "mean_anomaly": lambda o, t: o.mean_anomaly(t),
    "true_anomaly": lambda o, t: o.true_anomaly(t),
    "xyz": lambda o, t: o.xyz(t),
    "vxyz": lambda o, t: o.vxyz(t),
    "projected_separation": lambda o, t: o.projected_separation(t),
    "cos_phase": lambda o, t: o.cos_phase(t),
    "phase": lambda o, t: o.phase(t),
    "theta": lambda o, t: o.theta(t),
    "star_planet_distance": lambda o, t: o.star_planet_distance(t),
    "light_travel_time": lambda o, t: o.light_travel_time(t, 0.9),
    "radial_velocity": lambda o, t: o.radial_velocity(t, 11.0),
    "lambert_phase_curve": lambda o, t: o.lambert_phase_curve(t, K, 0.3),
    "emission_phase_curve": lambda o, t: o.emission_phase_curve(t, K, 0.02, 0.4),
    "ellipsoidal_variation": lambda o, t: o.ellipsoidal_variation(t, 1.3, 1e-3),
}
JAXORBIT_FIELDS = ("tc", "tp", "p", "a", "i", "e", "w", "lan", "ep_times", "dt", "coeffs")


def test_every_jax_orbit_method_is_covered():
    public = {n for n in dir(jax3d.JaxOrbit) if not n.startswith("_") and callable(getattr(jax3d.JaxOrbit, n))}
    assert public - {"from_tc", "from_tp"} == set(JAXORBIT_METHODS)


@pytest.mark.parametrize("builder", ["from_tc", "from_tp"])
@pytest.mark.parametrize("grid_source", ["jax", "numba"])
def test_jax_orbit_float32(builder, grid_source):
    """A numba-built float64 grid must not promote a float32 orbit."""
    grid = None if grid_source == "jax" else numba3d.create_expansion_points(15, 0.3, "ea")
    orbit = getattr(jax3d.JaxOrbit, builder)(f32(TC), *(f32(v) for v in ORBITS["eccentric"]), grid=grid)
    for field in JAXORBIT_FIELDS:
        assert getattr(orbit, field).dtype == F32, field
    t = f32(TC + np.linspace(-0.2, 3.1, 33) * ORBITS["eccentric"][0])
    for name, method in JAXORBIT_METHODS.items():
        assert all(leaf.dtype == F32 for leaf in _floating_leaves(method(orbit, t))), name



def test_absolute_timing_follows_the_parameters():
    """The documented contract for an unshifted BJD: a Python float adopts the float32 parameters
    (and is rounded by up to 0.25 d); an explicit float64 promotes the whole orbit and stays exact."""
    tc = 2459000.123456
    pars32 = [f32(v) for v in ORBITS["eccentric"]]
    assert jax3d.JaxOrbit.from_tc(tc, *pars32).tc.dtype == F32
    o64 = jax3d.JaxOrbit.from_tc(np.float64(tc), *pars32)
    assert o64.tc.dtype == o64.coeffs.dtype == F64
    t = tc + np.linspace(-0.1, 0.1, 5)
    expected = jax3d.JaxOrbit.from_tc(tc, *ORBITS["eccentric"]).projected_separation(t)
    assert _scaled_error(o64.projected_separation(t), expected) < 1e-6

def test_vmap_over_float32_parameter_batch():
    p, a, i, _, w, lan = (f32(v) for v in ORBITS["eccentric"])
    t = f32(np.linspace(-0.2, 0.2, 21))
    z = jax.vmap(lambda e: jax3d.JaxOrbit.from_tc(f32(0.0), p, a, i, e, w, lan).projected_separation(t))(
        f32(np.linspace(0.05, 0.5, 4)))
    assert z.shape == (4, 21) and z.dtype == F32


def test_phase_gradient_finite_where_cos_phase_rounds_to_one():
    """An edge-on circular orbit has cos(phase) = -1 at transit and +1 at eclipse, exactly so in
    float32. The clip must keep the arccos gradient finite there; 1 - 1e-15 rounds to 1 in float32."""
    t = f32([0.0, 1.5])
    pars = (f32(3.0), f32(10.0), f32(0.5 * np.pi), f32(0.0), f32(0.0))
    orbit = jax3d.JaxOrbit.from_tc(f32(0.0), *pars)
    assert np.any(np.abs(np.asarray(orbit.cos_phase(t))) == 1.0)  # the case this test is about
    for method in ("phase", "theta"):
        g = jax.grad(lambda tc: jnp.sum(getattr(jax3d.JaxOrbit.from_tc(tc, *pars), method)(t)))(f32(0.0))
        assert g.dtype == F32 and np.isfinite(float(g)), method


# ---------------------------------------------------------------------------
# Accuracy against the float64 numba backend on identical (float32-rounded) inputs
# ---------------------------------------------------------------------------

# Scaled errors (max |jax - numba| / signal scale), pinned at about three times the
# worst case measured on 2026-09-25 (values 2.2e-6 up to e = 0.7 and 1.2e-5 at
# e = 0.9, gradients 1.9e-5 and 1.1e-4, true anomaly 1.7e-5 rad). e = 0.9 has its own
# limit: float32 conditioning degrades near periastron at high eccentricity.
TOL_VALUE = 1e-5
TOL_VALUE_HIGH_E = 5e-5
TOL_GRAD = 1e-4
TOL_GRAD_HIGH_E = 5e-4
TOL_TA = 1e-4        # [radians]
TOL_CONTACT = 2e-6   # [days]; both bisections stop at a 1e-6 day bracket
TOL_ZMIN_T = 1e-4    # [days]; the separation is flat at its minimum
TOL_ZMIN_Z = 1e-5    # [R_star]
HIGH_E = 0.75
TRANSITING = ["circular", "eccentric", "edge_on"]


def _tol(name, low, high):
    return high if ORBITS[name][3] > HIGH_E else low


def _rounded(x):
    """``x`` rounded to float32 and cast back to float64: the input both backends see."""
    return np.asarray(np.asarray(x, np.float32), np.float64)


def _tup(x):
    return x if isinstance(x, tuple) else (x,)


def _scaled_error(actual, desired, floor=0.0):
    """Max absolute error over the signal scale, the scale floored at ``floor``."""
    actual, desired = np.asarray(actual, np.float64), np.asarray(desired, np.float64)
    return float(np.abs(actual - desired).max() / max(np.abs(desired).max(), floor))


def _grad_error(actual, desired):
    """Largest per-slot error, scaled like jax_utils.assert_grad_close scales its atol."""
    a2 = np.asarray(actual, np.float64).reshape(-1, desired.shape[-1])
    d2 = np.asarray(desired, np.float64).reshape(-1, desired.shape[-1])
    scale = np.maximum(np.abs(d2).max(axis=0), 1e-3 * np.abs(d2).max())
    return float((np.abs(a2 - d2).max(axis=0) / scale).max())


def _setup32(name, n=96, seed=3):
    """float32-rounded inputs (as float64) for the multi-expansion-point evaluators."""
    p, a, i, e, w, lan = (float(v) for v in _rounded(ORBITS[name]))
    ep_times, _, dt, ep_table = numba3d.create_expansion_points(15, max(e, 0.2), "ea")
    tpa = float(_rounded(TC - numba3d.mean_anomaly_at_transit(e, w) / (2 * np.pi) * p))
    rng = np.random.default_rng(seed)
    times = _rounded(TC + rng.uniform(-3.5, 4.5, n) * p)
    return (p, a, i, e, w, lan), tpa, float(_rounded(dt)), ep_table, _rounded(ep_times), times


def _grid32(dt, ep_table, ep_times):
    return f32(dt), jnp.asarray(ep_table, jnp.int32), f32(ep_times)


# name: (jax fn, numba value fn, numba gradient fn, physical inputs between t and tpa)
ORBIT_EVALUATORS = {
    "pos": (jax3d.pos_o, numba3d.pos_o, numba3d.pos_od, ()),
    "zpos": (jax3d.zpos_o, numba3d.zpos_o, numba3d.zpos_od, ()),
    "sep": (jax3d.sep_o, numba3d.sep_o, numba3d.sep_od, ()),
    "vel": (jax3d.vel_o, numba3d.vel_o, numba3d.vel_od, ()),
    "zvel": (jax3d.zvel_o, numba3d.zvel_o, numba3d.zvel_od, ()),
    "cos_alpha": (jax3d.cos_alpha_o, numba3d.cos_alpha_o, numba3d.cos_alpha_od, ()),
    "distance": (jax3d.star_planet_distance_o, numba3d.star_planet_distance_o,
                 numba3d.star_planet_distance_od, ()),
    "lambert": (jax3d.lambert_phase_curve_o, numba3d.lambert_phase_curve_o,
                numba3d.lambert_phase_curve_od, (0.3, 0.1)),
    "emission": (jax3d.emission_phase_curve_o, numba3d.emission_phase_curve_o,
                 numba3d.emission_phase_curve_od, (0.1, 0.02, 0.4)),
}
GRADIENT_QUANTITIES = ("pos", "sep", "zvel", "cos_alpha", "lambert")


def value_error(name, quantity):
    jfn, nfn, _, pre = ORBIT_EVALUATORS[quantity]
    pars, tpa, dt, ep_table, ep_times, times = _setup32(name)
    expected = _tup(nfn(times, *pre, tpa, pars[0], dt, ep_table, ep_times,
                        numba3d.solve3d_orbit(ep_times, *pars)))
    dt32, table32, ep32 = _grid32(dt, ep_table, ep_times)
    coeffs32 = jax3d.solve3d_orbit(ep32, *(f32(v) for v in pars))
    actual = _tup(jfn(f32(times), *pre, f32(tpa), f32(pars[0]), dt32, table32, ep32, coeffs32))
    assert all(act.dtype == F32 for act in actual)
    return max(_scaled_error(act, exp) for act, exp in zip(actual, expected))


def gradient_error(name, quantity, basis):
    jfn, _, nfn_d, pre = ORBIT_EVALUATORS[quantity]
    pars, tpa, dt, ep_table, ep_times, times = _setup32(name)
    p, a, i, e, w, lan = pars
    coeffs, dcoeffs = numba3d.solve3d_orbit_d(ep_times, *pars)
    if basis == "tc":
        numba3d.tp_to_tc_gradient_orbit(dcoeffs, p, e, w)
    nb = nfn_d(times, *pre, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)
    dt32, table32, ep32 = _grid32(dt, ep_table, ep_times)
    t32 = f32(times)

    def model(timing, p, a, i, e, w, lan, *pre_args):
        tpa_ = timing - jax3d.mean_anomaly_at_transit(e, w) / (2 * jnp.pi) * p if basis == "tc" else timing
        co = jax3d.solve3d_orbit(ep32, p, a, i, e, w, lan)
        return jfn(t32, *pre_args, tpa_, p, dt32, table32, ep32, co)

    timing = TC if basis == "tc" else tpa
    jac = _tup(jacobian(model, 7 + len(pre), *(f32(v) for v in (timing, *pars, *pre))))
    assert all(j.dtype == F32 for j in jac)
    return max(_grad_error(j, g) for j, g in zip(jac, nb[len(jac):]))


def true_anomaly_error(name):
    pars, tpa, dt, ep_table, ep_times, times = _setup32(name)
    p, a, i, e, w, lan = pars
    expected = numba3d.true_anomaly_o(times, tpa, p, *nb_eccentricity_vector(i, e, w, lan), w, dt, ep_table,
                                      ep_times, numba3d.solve3d_orbit(ep_times, *pars))
    dt32, table32, ep32 = _grid32(dt, ep_table, ep_times)
    p32, a32, i32, e32, w32, lan32 = (f32(v) for v in pars)
    actual = jax3d.true_anomaly_o(f32(times), f32(tpa), p32, *jax3d.eccentricity_vector(i32, e32, w32, lan32),
                                  w32, dt32, table32, ep32, jax3d.solve3d_orbit(ep32, p32, a32, i32, e32, w32,
                                                                                lan32))
    assert actual.dtype == F32
    diff = np.asarray(actual, np.float64) - expected
    return float(np.abs(np.angle(np.exp(1j * diff))).max())


@pytest.mark.accuracy
@pytest.mark.parametrize("name", ORBIT_IDS)
@pytest.mark.parametrize("quantity", list(ORBIT_EVALUATORS))
def test_orbit_values_match_numba(name, quantity):
    assert value_error(name, quantity) < _tol(name, TOL_VALUE, TOL_VALUE_HIGH_E)


@pytest.mark.accuracy
@pytest.mark.parametrize("basis", ["tc", "tp"])
@pytest.mark.parametrize("name", ORBIT_IDS)
@pytest.mark.parametrize("quantity", GRADIENT_QUANTITIES)
def test_orbit_gradients_match_numba(quantity, name, basis):
    assert gradient_error(name, quantity, basis) < _tol(name, TOL_GRAD, TOL_GRAD_HIGH_E)


@pytest.mark.accuracy
@pytest.mark.parametrize("name", ORBIT_IDS)
def test_true_anomaly_matches_numba(name):
    assert true_anomaly_error(name) < TOL_TA


@pytest.mark.accuracy
@pytest.mark.parametrize("name", ORBIT_IDS)
@pytest.mark.parametrize("te", [0.0, -0.37, 1.9])
@pytest.mark.parametrize("dim", [2, 3])
def test_solve_matches_numba_per_order(name, te, dim):
    """Each Taylor order (column) against its own scale, so the small snap column is not hidden.

    The scale is floored at the order's natural magnitude a n**k / k!: a column can
    vanish analytically (the 2D order-0 column of an edge-on orbit at transit).
    """
    pars = tuple(float(v) for v in _rounded((te,) + ORBITS[name]))
    n = 2 * np.pi / pars[1]
    expected = (numba3d.solve3d if dim == 3 else numba2d.solve2d)(*pars)
    actual = (jax3d.solve3d if dim == 3 else jax2d.solve2d)(*(f32(v) for v in pars))
    assert actual.dtype == F32
    tol = _tol(name, TOL_VALUE, TOL_VALUE_HIGH_E)
    for order in range(5):
        floor = pars[2] * n ** order / math.factorial(order)
        assert _scaled_error(actual[:, order], expected[:, order], floor) < tol, f"order {order}"


def _coeffs_pair(name):
    pars = tuple(float(v) for v in _rounded((0.0,) + ORBITS[name]))
    return numba3d.solve3d(*pars), jax3d.solve3d(*(f32(v) for v in pars))


@pytest.mark.accuracy
@pytest.mark.parametrize("name", TRANSITING)
@pytest.mark.parametrize("point", [1, 2, 3, 4, 12])
def test_contact_points_match_numba(name, point):
    c64, c32 = _coeffs_pair(name)
    t = jax3d.find_contact_point(f32(K), point, c32)
    assert t.dtype == F32
    assert abs(float(t) - numba3d.find_contact_point(K, point, c64)) < TOL_CONTACT


@pytest.mark.accuracy
@pytest.mark.parametrize("name", TRANSITING)
def test_find_z_min_matches_numba(name):
    c64, c32 = _coeffs_pair(name)
    tj, zj = jax3d.find_z_min(0.001, c32)
    tn, zn = numba3d.find_z_min(0.001, c64)
    assert tj.dtype == zj.dtype == F32
    assert abs(float(tj) - tn) < TOL_ZMIN_T
    assert abs(float(zj) - zn) < TOL_ZMIN_Z


def test_util_gradients_in_float32():
    """The custom_jvp rules must produce float32 tangents, finite and close to float64."""
    def grads(dtype):
        p, a, i, e, w, lan = (jnp.asarray(v, dtype) for v in ORBITS["eccentric"])
        k, zero = jnp.asarray(K, dtype), jnp.asarray(0.0, dtype)
        g_t14 = jax.grad(lambda a_: jax3d.t14(k, jax3d.solve3d(zero, p, a_, i, e, w, lan)))(a)
        g_z = jax.grad(lambda i_: jax3d.find_z_min(zero, jax3d.solve3d(zero, p, a, i_, e, w, lan))[1])(i)
        return g_t14, g_z

    for g32, g64 in zip(grads(jnp.float32), grads(jnp.float64)):
        assert g32.dtype == F32 and np.isfinite(float(g32))
        assert_allclose(float(g32), float(g64), rtol=1e-3)



def test_contact_point_gradient_with_a_wider_radius_ratio():
    """A float64 k with float32 coefficients promotes to float64 instead of breaking the custom_jvp rule."""
    p, i, e, w, lan = (f32(v) for v in (3.0, 1.5, 0.3, 0.4, 0.0))
    a = f32(10.0)
    t14 = jax3d.t14(np.float64(K), jax3d.solve3d(f32(0.0), p, a, i, e, w, lan))
    assert t14.dtype == F64
    g = jax.grad(lambda a_: jax3d.t14(np.float64(K), jax3d.solve3d(f32(0.0), p, a_, i, e, w, lan)))(a)
    assert np.isfinite(float(g))
    g32 = jax.grad(lambda a_: jax3d.t14(f32(K), jax3d.solve3d(f32(0.0), p, a_, i, e, w, lan)))(a)
    assert_allclose(float(g), float(g32), rtol=1e-3)

class TestX64Disabled:
    """The interoperability case: with x64 off, Python and NumPy inputs all become float32."""

    def test_numba_grid_and_python_inputs(self):
        # float32-representable inputs (as float64), so the comparison measures float32
        # arithmetic, not input rounding; the numba grid keeps its int64 table.
        pars, tpa, dt, ep_table, ep_times, times = _setup32("eccentric")
        p, a, i, e, w, lan = pars
        grid = (ep_times, None, dt, ep_table)
        with jax.enable_x64(False):
            c = jax3d.solve3d(0.0, p, a, i, e, w, lan)
            coeffs = jax3d.solve3d_orbit(ep_times, p, a, i, e, w, lan)
            z = jax3d.sep_o(times, tpa, p, dt, ep_table, ep_times, coeffs)
            t14 = jax3d.t14(K, c)
            z_orbit = jax3d.JaxOrbit.from_tc(TC, p, a, i, e, w, lan, grid=grid).projected_separation(times)
            z_default = jax3d.JaxOrbit.from_tc(TC, p, a, i, e, w, lan).projected_separation(times)
        for out in (c, coeffs, z, t14, z_orbit, z_default):
            assert out.dtype == F32
        z_nb = numba3d.sep_o(times, tpa, p, dt, ep_table, ep_times, numba3d.solve3d_orbit(ep_times, p, a, i, e, w,
                                                                                            lan))
        for zz in (z, z_orbit, z_default):
            assert _scaled_error(zz, z_nb) < TOL_VALUE
        assert abs(float(t14) - numba3d.t14(K, numba3d.solve3d(0.0, p, a, i, e, w, lan))) < 2 * TOL_CONTACT
