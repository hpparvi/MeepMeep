"""JAX-vs-numba evaluation benchmark behind the table in docs/source/jax_backend.rst.

Unlike the solve2d spike in this directory, this one is kept: it is what the
"Performance" section of the JAX backend page reports.

The projected separation over a whole eccentric orbit (p = 3 d, a = 8.5,
e = 0.3, npt = 15), evaluated from the orbital parameters, i.e. the coefficient
solve plus the evaluation, as a model does inside a likelihood. The
expansion-point grid is built once outside the timed call, as the docs
recommend:

- numba value / grad       : solve3d_orbit + sep_ov, and solve3d_orbit_d +
                             tp_to_tc_gradient_orbit + sep_ovd -> (N, 7)
                             (serial kernels, what the dispatchers route to)
- numba parallel value/grad: the same with the prange twins sep_ovp / sep_ovdp
- JAX value                : jit(model)
- JAX jacfwd               : jit(jacfwd(model)) -> (N, 7)
- JAX grad                 : jit(grad(scalar likelihood))

Before timing, the JAX model is checked against numba (values to 1e-10,
gradients to 1e-8 relative). Times are the best over repeated batches of calls,
after compilation; JAX calls end with block_until_ready().

Usage (from the repository root):

    python benchmarks/bench_jax_vs_numba.py cpu
    python benchmarks/bench_jax_vs_numba.py cuda     # numba columns stay on the CPU
"""
import os
import sys

platform = sys.argv[1] if len(sys.argv) > 1 else "cpu"
os.environ["JAX_PLATFORMS"] = platform

import platform  # noqa: E402
import subprocess  # noqa: E402
import timeit  # noqa: E402

import numpy as np  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from meepmeep import numba3d as nb  # noqa: E402
from meepmeep import jax3d as jx  # noqa: E402

TC, P, A, I, E, W, LAN = 0.0, 3.0, 8.5, np.radians(88.0), 0.3, np.radians(60.0), 0.3
NPT = 15
THETA = np.array([TC, P, A, I, E, W, LAN])


def best(f, budget=0.5):
    """Best time per call [s]: repeat batches until ~budget seconds, take the minimum."""
    f()
    n = 1
    while True:
        t = timeit.timeit(f, number=n)
        if t > 0.02 or n >= 1 << 20:
            break
        n *= 4
    reps = max(3, int(budget / max(t, 1e-9)))
    return min(timeit.repeat(f, number=n, repeat=min(reps, 25))) / n


def run(N, parallel=False):
    times = np.linspace(0.0, P, N, endpoint=False)
    ep_times, _, dt, ep_table = nb.create_expansion_points(NPT, max(E, 0.2), "ea")

    # numba (CPU): value and (N, 7) gradient from the parameters
    def nb_value():
        c = nb.solve3d_orbit(ep_times, P, A, I, E, W, LAN, npt=NPT)
        tpa = TC - nb.mean_anomaly_at_transit(E, W) / (2 * np.pi) * P
        return nb.sep_o(times, tpa, P, dt, ep_table, ep_times, c)

    def nb_grad():
        c, dc = nb.solve3d_orbit_d(ep_times, P, A, I, E, W, LAN, npt=NPT)
        nb.tp_to_tc_gradient_orbit(dc, P, E, W)
        tpa = TC - nb.mean_anomaly_at_transit(E, W) / (2 * np.pi) * P
        return nb.sep_od(times, tpa, P, dt, ep_table, ep_times, c, dc)

    def nb_value_par():
        c = nb.solve3d_orbit(ep_times, P, A, I, E, W, LAN, npt=NPT)
        tpa = TC - nb.mean_anomaly_at_transit(E, W) / (2 * np.pi) * P
        return nb.sep_ovp(times, tpa, P, dt, ep_table, ep_times, c)

    def nb_grad_par():
        c, dc = nb.solve3d_orbit_d(ep_times, P, A, I, E, W, LAN, npt=NPT)
        nb.tp_to_tc_gradient_orbit(dc, P, E, W)
        tpa = TC - nb.mean_anomaly_at_transit(E, W) / (2 * np.pi) * P
        return nb.sep_ovdp(times, tpa, P, dt, ep_table, ep_times, c, dc)

    # JAX: the same model, grid held out of the differentiated arguments
    t_j = jnp.asarray(times)
    ep_j, table_j = jnp.asarray(ep_times), jnp.asarray(ep_table.astype(np.int32))
    z_obs = jnp.asarray(nb_value()) + 1e-3

    def model(theta):
        tc, p, a, i, e, w, lan = theta
        tpa = tc - jx.mean_anomaly_at_transit(e, w) / (2 * jnp.pi) * p
        c = jx.solve3d_orbit(ep_j, p, a, i, e, w, lan)
        return jx.sep_o(t_j, tpa, p, dt, table_j, ep_j, c)

    def loglike(theta):
        return -0.5 * jnp.sum((model(theta) - z_obs) ** 2)

    theta = jnp.asarray(THETA)
    f_val = jax.jit(model)
    f_jac = jax.jit(jax.jacfwd(model))
    f_grad = jax.jit(jax.grad(loglike))

    # Parity: the JAX model reproduces numba (values and the (N, 7) gradient).
    z_nb, dz_nb = nb_grad()
    assert np.allclose(np.asarray(f_val(theta)), z_nb, rtol=0, atol=1e-10)
    assert np.allclose(np.asarray(f_jac(theta)), dz_nb, rtol=1e-8, atol=1e-8 * np.abs(dz_nb).max())

    if parallel:
        return {"numba par value": best(nb_value_par), "numba par grad": best(nb_grad_par)}
    return {
        "numba value": best(nb_value),
        "numba grad": best(nb_grad),
        "JAX value": best(lambda: f_val(theta).block_until_ready()),
        "JAX jacfwd": best(lambda: f_jac(theta).block_until_ready()),
        "JAX grad": best(lambda: f_grad(theta).block_until_ready()),
    }


def cpu_model():
    """CPU model name for the report (Linux /proc/cpuinfo, macOS sysctl)."""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.processor() or "unknown CPU"


def fmt(t):
    return f"{t * 1e6:8.1f} us" if t < 1e-3 else f"{t * 1e3:8.3f} ms"


if __name__ == "__main__":
    dev = jax.devices()[0]
    print(f"JAX {jax.__version__} on {dev.platform}: {dev.device_kind}; numba on {cpu_model()} "
          f"({os.cpu_count()} logical CPUs); e = {E}, npt = {NPT}")
    cols = ["numba value", "numba par value", "JAX value", "numba grad", "numba par grad", "JAX jacfwd",
            "JAX grad"]
    print(f"{'N':>9}  " + "  ".join(f"{c:>15}" for c in cols))
    sizes = (1_000, 10_000, 100_000, 1_000_000)
    # The parallel kernels go last: numba's worker threads keep spinning for a
    # while after a prange region and would slow down whatever is timed next.
    results = {N: run(N) for N in sizes}
    for N in sizes:
        results[N].update(run(N, parallel=True))
    for N in sizes:
        print(f"{N:>9,}  " + "  ".join(f"{fmt(results[N][c]):>15}" for c in cols))
