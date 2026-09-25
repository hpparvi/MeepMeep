# Spike: is a GPU-side `solve2d` worth it?

**Throwaway prototype, not part of the shipped package.** It exists to test the
decision recorded in `meepmeep/backends/opencl/source.py` ("solve on the host,
upload the arrays, evaluate on the device"), not to replace it.

## Contents

| file | what it is |
|---|---|
| `solve2d_proto.cl` | prototype device port of `solve2d`, `solve2d_d`, and `ea_from_ma`, plus batched `__kernel` entry points (one work item per parameter set) |
| `_solve2d_harness.py` | shared plumbing: program build, parameter generation, numba batch loops |
| `check_solve2d_opencl.py` | correctness against the numba twins |
| `bench_solve2d_opencl.py` | the GPU-vs-CPU benchmark |
| `bench_solve_share.py` | evaluation throughput, to put the solve cost in context |
| `results.txt` | raw output of a benchmark run |

## Setup

RTX 5070 (48 CUs, consumer Blackwell: fp64 at 1/64 rate), Ryzen 7 5800X
(8 cores / 16 threads), numba 0.63.1. Parameters drawn over a realistic range
with `e ~ U(0, 0.5)`, so the Kepler Newton loop diverges within a warp.

Both paths are timed to the same finish line: coefficients resident in device
memory, ready for an evaluation kernel.

Timings are the **minimum over interleaved rounds**. A median over a contiguous
block is not safe on a loaded machine: it produced self-contradictory numbers
(a strictly smaller workload timing slower than a larger one) before the
methodology was changed.

## Correctness

The port agrees with the numba solvers to **1.4e-14** relative (fp64) and
**2.2e-5** (fp32) over 2000 parameter sets spanning `e` up to 0.95.

Mutation-tested: sign flips and dropped chain-rule terms in the `dm10`, `dmu`,
`dinv_r7`, `dma`, `dea`, `u_ddot`, `lan` and `tc` rows all turn the check red.
Two mutations survived and are documented in `solve2d_proto.cl`:

- **`te = 0` blindness.** The period term of the mean-anomaly derivative is
  `-2*pi*te/p**2`, identically zero at the transit centre, so a `te == 0` batch
  cannot see it. `make_pars` now spans `te` over a full period. Same shape as
  the `d[1] += epoch*d[0]` epoch-0 trap documented in `CLAUDE.md`.
- **NumPy-vs-C mod convention is unobservable here.** `E - e sin(E) = M` is
  strictly monotonic, so `root(M + 2*pi) = root(M) + 2*pi` exactly, and the
  solver's only outputs are `sin(E)` and `cos(E)`. The positive-mod helper is
  kept anyway: it is only safe while nothing downstream consumes `E` itself.

## Result: end to end, coefficients resident on device [us]

`solve2d` (values only):

| | N=1 | N=50 | N=100 | N=1000 |
|---|---|---|---|---|
| CPU numba serial -> upload | 12.8 | 26.6 | 40.4 | 322.2 |
| CPU numba prange -> upload | 21.1 | 37.5 | 40.3 | 84.3 |
| GPU solve (fp64) | 40.0 | 45.9 | 46.3 | 52.8 |
| GPU solve (fp32) | 27.9 | 28.4 | 28.4 | 33.0 |

`solve2d_d` (values + gradients):

| | N=1 | N=50 | N=100 | N=1000 |
|---|---|---|---|---|
| CPU numba serial -> upload | 25.8 | 94.2 | 173.5 | 1402.1 |
| CPU numba prange -> upload | 38.7 | 65.7 | 89.3 | 339.7 |
| GPU solve (fp64) | 67.3 | 72.3 | 70.8 | 78.0 |
| GPU solve (fp32) | 28.9 | 30.1 | 32.6 | 37.7 |

Crossovers: values, the GPU overtakes one core near **N ~ 60**; gradients, near
**N ~ 15**. At N=1000 with gradients the GPU is **37x** faster than a single
core and **9x** faster than all eight.

## Why: the GPU path is launch-bound at these sizes

Device command duration barely moves with N:

| gradients, kernel only [us] | N=1 | N=50 | N=100 | N=1000 |
|---|---|---|---|---|
| fp64 | 41.3 | 45.2 | 45.6 | 46.3 |
| fp32 | 3.1 | 5.0 | 6.1 | 6.9 |

Solving one expansion costs about what solving a thousand costs. Confirmed
independently: an A/B run while the card ramped from 1005 to 2910 MHz (2.9x)
produced **identical** wall times to 0.1 us.

Fixed costs, fp32: ~13 us kernel launch + sync, plus ~13 us for the parameter
upload (a second enqueue). Marginal cost per parameter set, from the scaling
study out to N=100,000:

| | ns per parameter set (gradients) |
|---|---|
| GPU fp32, end to end | 4.6 |
| GPU fp32, kernel only | 1.4 |
| GPU fp64, kernel only | 7.5 |
| CPU, prange (8 cores) | 166 |
| CPU, one core | 1330 |

## fp32 vs fp64

On this consumer card fp64 costs **6.7x** on the solve kernel and **9x** on the
shipped `sep_cd2` evaluator (0.55 vs 0.06 ns/point), well short of the 64x
FLOP-rate ratio, because the kernel is bound by transcendentals, memory traffic
and occupancy rather than raw fp64 ALU. On a datacenter card (1/2 rate) the gap
would largely close.

fp32 accuracy against numba: ~1e-6 relative for `e < 0.5`, degrading to ~2e-5
(6e-5 in the `w` gradient row) as `e` approaches 0.95.

Note that numba's literal `1e-13` Kepler tolerance is unreachable in float32
(eps ~ 1.2e-7), so a verbatim port pins every fp32 work item at the
50-iteration cap. `solve2d_proto.cl` makes the tolerance precision-aware; the
`strict Kepler tol` benchmark rows show the cost of not doing so (kernel time
roughly doubles). Warp divergence itself is mild.

## Register pressure: a non-issue

`PRIVATE_MEM_SIZE` is **40 bytes** for the fp64 gradient kernel: essentially
no spilling, despite ~2.5 kB of declared private data. NVVM scalarises the ~39
private derivative vectors into registers and dead-codes the structural zeros.
The macro-based column fill matters here: an array of pointers for
`dq_xi`/`dq_eta` would make them address-taken, defeat SROA, and push the whole
working set to local memory. Occupancy is register-limited (max work-group 256,
not 1024) but not spill-bound, so a hand-optimised sparse rewrite would buy
little.

## Context: how much of the pipeline is the solve?

Throughput of the shipped `sep_cd2` gradient evaluator on the same device,
N=1000 parameter sets x M cadences:

| | M=100 | M=1000 | M=10000 |
|---|---|---|---|
| eval fp32 [us] | 9.6 | 57.3 | 580.0 |
| eval fp64 [us] | 60.1 | 552.9 | 5465.6 |

Against a **1402 us** CPU-solve-and-upload (or 340 us with prange) at N=1000,
the solve dominates every one of those pipelines. Even at M=10000 the
single-core solve is 2.4x the entire fp32 evaluation. Moving the solve to the
device (37.7 us) removes it as a bottleneck.

For N=1 (one parameter set per likelihood call, the ordinary `Orbit` and
`Expansion2D` use), the solve is 1.6 us and irrelevant either way.

## The CPU baseline is not the CPU's ceiling

`solve2d_d` allocates ~40 `zeros(6)` temporaries per call. Numba's NRT
heap-allocates every one; it does not stack-promote `np.zeros`:

| | ns/call |
|---|---|
| `solve2d_d` total | 1193 |
| its 40 `zeros(6)` allocations alone | 705 |
| the same data as one `zeros((40, 6))` | 73 |
| as scalars | ~0 |

**59% of the numba gradient solver is allocation, not arithmetic**, and ~630
ns/call is recoverable by a purely local change to one function, a ~2.1x CPU
speedup with no GPU involved.

## Outcome

Recommendation 1 was implemented (`point2dd/solve.py`, `point3dd/solve.py`):
the ~40 separate `zeros(6)` scratch vectors became one `zeros((n, 6))` block
with named row views, and five vectors that were written but never read
(`dci`, `dsi`, `dcw`, `dsw`, `dr2`; the rotation-matrix rows are
hand-differentiated from `si`/`ci`/`sw`/`cw` directly) were deleted.
`solve2d_d` 1166 -> 689 ns, `solve3d_d` 1274 -> 730 ns (**1.7x**), bit-identical
with fastmath disabled and within 1.8e-15 with it on.

That work also turned up a **larger** win, since done. 88 per cent of
`Orbit.set_pars(tc=..., derivatives=True)` was the Python-level loop calling
`tp_to_tc_gradient` once per expansion point (94.3 of 107.4 us at npt=15). Root
cause: `tp_to_tc_gradient` was missing the `@njit(fastmath=True)` its inverse
`tc_to_tp_gradient` carries, so it ran as plain NumPy: roughly fifteen small
array operations per expansion point. Restoring the decorator took `set_pars`
to 37.2 us; replacing the Python loop with a new in-place whole-orbit transform
`tp_to_tc_gradient_orbit` took it to 13.5 us (**7.9x**; 8.4x at npt=25). The
tc-basis reparametrisation is now free relative to the tp path.

Recommendation 2 was then implemented in the shipped backend: `solve2d.cl`,
`solve3d.cl` (device functions) and the opt-in `solve_kernels.cl` (the only
shipped file with `__kernel` entry points). Recommendation 3 (fusing the
solve into the evaluation kernel) was deliberately deferred: it would need
an address-space variant of every evaluator, because OpenCL C 1.2 has no
generic address space and this machine's device does not advertise
`__opencl_c_generic_address_space` even though NVIDIA's compiler accepts one.
Solve and evaluate are separate launches for now.

Shipped `solve3d_d`, end to end with coefficients resident on device [us]:

| N | CPU serial | CPU prange | GPU fp64 | GPU fp32 |
|---|---|---|---|---|
| 1 | 15.2 | 28.2 | 57.6 | 17.0 |
| 100 | 119.2 | 73.6 | 68.3 | 21.8 |
| 1000 | 1016.7 | 317.2 | 78.0 | 26.8 |
| 10000 | 9836.9 | 2547.0 | 177.9 | 83.0 |

## Recommendation

1. ~~**Do the allocation fix first.**~~ *(done; see Outcome above)* ~2.1x on `solve2d_d` (and very likely
   `solve3d_d` / `solve3d_orbit_d`, which have the same structure), local to
   one function, no new backend surface, benefits every existing user.
2. **A GPU solve is justified only for the population-sampler case**: many
   parameter sets per likelihood call (emcee/DE walkers), evaluated on-device.
   There it is a real 9-37x on a component that currently dominates the
   pipeline. It is worthless for the single-parameter-set case.
3. **If it ships, fuse rather than add a kernel.** Two thirds of the GPU path's
   cost at these sizes is the launch and the separate parameter upload, not the
   arithmetic. A `solve` kernel launched separately from the evaluation kernel
   throws most of the win away; solving per work-group inside the evaluation
   kernel, or at minimum batching the enqueues, is where the value is.
4. **The `solve*` exclusion in `source.py` should stay the default** and the
   docstring rationale is still right for MeepMeep's normal use. What this
   spike shows is that it is the wrong default for one specific consumer
   (device-resident population sampling), not that it is wrong generally.

## Caveats

Single device, single host. NVIDIA's OpenCL launch latency (~13 us) is high
relative to CUDA and dominates these measurements. Only the 2D solver was
ported; `solve3d_orbit` does npt (~15) expansions per parameter set, which
multiplies the arithmetic without changing the fixed launch cost, so the
GPU's advantage there would be larger than measured here.

## Also here: the JAX-vs-numba evaluation benchmark

`bench_jax_vs_numba.py` is not part of the spike above and is kept: it produces
the table in the "Performance" section of `docs/source/jax_backend.rst` (the
projected separation over a whole eccentric orbit, values and `(N, 7)`
gradients, numba serial and parallel kernels against jitted JAX `jit`,
`jacfwd` and `grad`). Run it from the repository root with
`python benchmarks/bench_jax_vs_numba.py cpu`, or `cuda` for the JAX columns on
a GPU.
