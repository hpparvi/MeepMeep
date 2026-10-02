.. _jax_backend:

JAX backend
===========

.. warning::

   The JAX backend is experimental. Its API and behaviour may still change
   between releases.

The JAX backend is a port of the numba evaluators to JAX. It covers the
coefficient solvers, the single- and multi-expansion-point evaluators, the
transit-geometry utilities, expansion-point placement and the exact
Newton-Raphson references. The references live in
``meepmeep.backends.jax.newton`` (``ea_newton``, ``ta_newton``,
``xyz_newton``, ``z_newton``, ``rv_newton``, ...), outside the two
aggregators, as their numba twins do; only
:func:`~meepmeep.jax3d.ea_from_ma` is exported. Everything traces, so a model
built on it can be ``jit``-compiled, ``vmap``-ed over parameter sets, run on
a GPU, and differentiated with ``jax.grad`` or ``jax.jacfwd``.

The difference from the numba backend is where the gradients come from.
Numba ships hand-derived ``_d``/``_od`` kernels. The JAX backend ships only
the value code and lets autodiff do the rest. Because the evaluators compute
a polynomial whose coefficients the solver returns, autodiff gives the exact
derivative of what is actually evaluated, which is also what the numba
kernels compute analytically. The test suite pins the two against each other
at round-off.

Install JAX with the optional dependency group:

.. code-block:: bash

   pip install "meepmeep[jax]"

With conda, install ``jax`` from conda-forge alongside MeepMeep
(``conda install -c conda-forge meepmeep jax``).

.. contents::
   :local:
   :depth: 1


Precision
---------

The backend computes in the floating dtype of its inputs, following JAX's
promotion rules. Float32 arrays in give float32 out, and Python floats adopt
the dtype of the arrays they are mixed with. An explicit float64 array (a
NumPy array, say) promotes the whole computation to float64. Half precision
raises a ``TypeError``.

Float64 is the reference: it agrees with the numba backend to round-off.
Enable it before any JAX computation:

.. code-block:: python

   import jax
   jax.config.update("jax_enable_x64", True)

Float32 is for consumer GPUs, where it runs many times faster than float64,
and for float32 JAX pipelines. It works with x64 off (everything is float32)
and with x64 on (pass float32 arrays). Two things to know.

**Shift the times first.** Absolute times in transit work are BJDs around
2.46e6, where a float32 ulp is a quarter of a day. The information is lost at
the cast, so subtract a reference epoch in float64 on the host, then cast:

.. code-block:: python

   import numpy as np
   import jax.numpy as jnp
   from meepmeep.jax3d import JaxOrbit

   t0 = 2460000.0                                   # near the middle of the data
   times = jnp.asarray(bjd - t0, jnp.float32)       # bjd: a float64 NumPy array
   orbit = JaxOrbit.from_tc(*(jnp.float32(v) for v in (tc - t0, p, a, i, e, w)))

Resolution still falls with distance from ``t0``. 500 days away a float32
ulp is 3e-5 d (2.6 s), about 6e-4 R_star of motion for a 3-day orbit at
a = 10, or tens of ppm near ingress. Keep ``t0`` central and spans to a few
hundred days for precise photometry. The timing parameter follows the same
rule: an absolute Python-float ``tc`` or ``tp`` passed with float32 parameters
is stored in float32 (rounded by up to 0.25 d), whatever the dtype of the
times. Shift it by ``t0``, or pass it as ``np.float64`` to make the whole orbit
float64. Likewise, a float64 NumPy time array mixed into float32 inputs
promotes the computation to float64, and so does a numba-built grid passed
straight to the ``*_o`` functions: cast ``ep_times`` and ``dt`` to float32
first (``JaxOrbit`` casts its grid for you).

**Expect five to six significant digits.** On identical inputs, float32
values agree with numba to 1e-5 of the signal scale up to e = 0.7 (5e-5 at
e = 0.9), gradients to 1e-4 (5e-4), and the true anomaly to 1e-4 rad. The
Kepler solver stops at a 1e-6 step in float32 (numba's 1e-13 in float64).
The contact-point and minimum-separation searches keep their 1e-6 and 1e-7
day brackets, which float32 resolves near the expansion point.


High level: ``JaxOrbit``
------------------------

:class:`~meepmeep.jax3d.JaxOrbit` is the functional counterpart of
:class:`~meepmeep.orbit.Orbit`. It is an immutable pytree built by
``from_tc`` or ``from_tp``, and every evaluator takes the times explicitly.

.. code-block:: python

   import jax
   import jax.numpy as jnp
   from meepmeep.jax3d import JaxOrbit

   jax.config.update("jax_enable_x64", True)
   times = jnp.linspace(-0.1, 0.1, 500)

   def loglike(theta, z_obs):
       tc, p, a, i, e, w = theta
       orbit = JaxOrbit.from_tc(tc, p, a, i, e, w)
       return -0.5 * jnp.sum((orbit.projected_separation(times) - z_obs) ** 2)

   grad = jax.jit(jax.grad(loglike))

The gradient basis follows the constructor. ``from_tc`` differentiates with
respect to the transit centre, and ``from_tp`` with respect to the periastron
time. There are no basis-transform helpers.

Without a ``grid`` argument the expansion points are placed for
``max(e, 0.2)``, as :class:`~meepmeep.orbit.Orbit` does. The placement is held
fixed under differentiation (a ``stop_gradient`` on ``e``), so the expansion
points sit at fixed phases, as in numba. Pass ``grid=`` (the tuple from
:func:`~meepmeep.jax3d.create_expansion_points`, or from the numba version) to
pin the grid yourself. One thing ``JaxOrbit`` does not do is the numba class's
hysteresis: the grid follows ``e`` on every call.

The methods mirror :class:`~meepmeep.orbit.Orbit`, with the times as the
first argument:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - ``JaxOrbit``
     - ``Orbit``
   * - ``xyz(times)``, ``vxyz(times)``, ``star_planet_distance(times)``,
       ``cos_phase(times)``, ``phase(times)``, ``theta(times)``,
       ``mean_anomaly(times)``, ``true_anomaly(times)``
     - the same names without ``times`` (``xyz`` and
       ``star_planet_distance`` take an optional ``times``)
   * - ``projected_separation(times)``
     - none; use :func:`~meepmeep.numba3d.sep_o`
   * - ``radial_velocity(times, k)``
     - ``radial_velocity(k)``
   * - ``lambert_phase_curve(times, k, ag)``
     - ``lambert_phase_curve(k, ag, times=None)``
   * - ``emission_phase_curve(times, k, fratio, offset)``
     - ``emission_phase_curve(k, fratio, offset, times=None)``
   * - ``ellipsoidal_variation(times, alpha, mass_ratio)``
     - ``ellipsoidal_variation(alpha, mass_ratio, times=None)``
   * - ``light_travel_time(times, rstar)``
     - ``light_travel_time(rstar)``

``JaxOrbit`` has no ``plot`` and no Newton-Raphson ``exact`` switch, and its
methods return values only; take gradients with ``jax.grad`` or
``jax.jacfwd`` of whatever function uses them.

Because a ``JaxOrbit`` is built inside the traced function, ``jax.vmap``
evaluates many parameter sets in one call:

.. code-block:: python

   def model(theta):
       tc, p, a, i, e, w = theta
       return JaxOrbit.from_tc(tc, p, a, i, e, w).projected_separation(times)

   thetas = jnp.array([[0.0, 3.0, 8.5, 1.54, 0.1, 0.5],
                       [0.001, 3.0, 9.0, 1.53, 0.3, 1.0]])   # (n_sets, 6)
   z = jax.jit(jax.vmap(model))(thetas)                      # (n_sets, N)
   dz = jax.jit(jax.vmap(jax.jacfwd(model)))(thetas)         # (n_sets, N, 6)


Low level: ``jax2d`` and ``jax3d``
----------------------------------

``meepmeep.jax2d`` and ``meepmeep.jax3d`` mirror
``meepmeep.numba2d`` and ``meepmeep.numba3d``. Names and argument order
are the same, so porting a numba model is mostly an import change. Every
function is element-wise: a scalar time gives a scalar and an array gives an
array. The ``_v``/``_vp`` vector kernels, the ``@overload`` dispatch and the
``_d``/``_cd``/``_od`` gradient variants have no counterparts. Get gradients by
differentiating a function that calls the solver and then an evaluator:

.. code-block:: python

   from meepmeep.jax3d import (create_expansion_points, solve3d_orbit, sep_o,
                               mean_anomaly_at_transit)

   ep_times, _, dt, ep_table = create_expansion_points(15, 0.3)

   def model(tc, p, a, i, e, w):
       tpa = tc - mean_anomaly_at_transit(e, w) / (2 * jnp.pi) * p
       coeffs = solve3d_orbit(ep_times, p, a, i, e, w)
       return sep_o(times, tpa, p, dt, ep_table, ep_times, coeffs)

   theta = (0.0, 3.0, 10.0, 1.52, 0.3, 0.5)                          # tc, p, a, i, e, w
   z = model(*theta)
   dz = jnp.stack(jax.jacfwd(model, argnums=range(6))(*theta), -1)   # (N, 6)

Keep the grid out of the differentiated arguments, as above. For a scalar
likelihood, ``jax.grad`` (reverse mode) costs a small multiple of one forward
pass whatever the number of parameters.

Other differences from numba:

- :func:`~meepmeep.jax3d.create_expansion_points` needs no root finder. The
  ``'ea'`` and ``'ta'`` placements map to time in closed form, so the grid can
  be built inside ``jit`` from a traced eccentricity. It agrees with the
  numba grid to scipy's ``brentq`` tolerance.
- :func:`~meepmeep.jax3d.solve3d_orbit` has no ``npt`` argument; it is the
  length of ``ep_times``. The solvers also accept an array of expansion times
  and return a stack of coefficient matrices.
- The contact points, durations and :func:`~meepmeep.jax3d.find_z_min`
  replicate the numba search loops, and they are differentiable. Their
  derivatives come from the implicit function theorem at the found point, not
  from differentiating the iteration.
- Four functions carry a ``custom_jvp`` rule instead of being
  differentiated through their code: :func:`~meepmeep.jax3d.ea_from_ma`
  (Kepler's equation, by implicit differentiation at the converged
  :math:`E`), ``lambert_kernel`` (finite at full phase),
  :func:`~meepmeep.jax3d.find_contact_point` and
  :func:`~meepmeep.jax3d.find_z_min` (implicit function theorem at the found
  point). Everything else is plain autodiff.
- The orbital-mechanics utilities beyond the exported ones
  (``i_from_baew``, ``as_from_rhop``, ``d_from_pkaiews``,
  ``impact_parameter``, ``transit_distance_factor``, ...) live in
  ``meepmeep.backends.jax.utils``, with the same signatures as their numba
  twins in ``meepmeep.backends.numba.utils``.
- :func:`~meepmeep.jax3d.true_anomaly_o` follows the gradient wherever the
  eccentricity vector came from. Build it from traced ``(i, e, w, lan)`` for
  the full derivative. numba's ``true_anomaly_od`` takes the vector's Jacobian
  explicitly (``dev`` from :func:`~meepmeep.numba3d.eccentricity_vector_d`), so
  the two agree when both differentiate it.


Performance
-----------

On a CPU, a jitted JAX model pays a fixed per-call overhead when the time
grid is small, and matches or beats numba's serial kernels once it is large.
Numba's parallel kernels (``sep_ovp``, ``sep_ovdp``, or ``Orbit(parallel=True)``)
stay ahead wherever they pay off. The table times the projected separation
over a whole eccentric orbit (``e = 0.3``, ``npt = 15``), evaluated from the
orbital parameters (coefficient solve plus evaluation), with serial / parallel
numba kernels. It was measured with ``benchmarks/bench_jax_vs_numba.py`` on an
AMD Ryzen 7 5800X (8 cores, 16 threads) with JAX 0.11.1 (best of repeated
calls, after compilation). The absolute numbers vary by machine, and so do the
ratios to a lesser degree. On a MacBook Pro, JAX ran about as fast as here
while the serial numba kernels were faster (by about 2x for values and 1.5x for
gradients at 100 000 points), and ``jax.grad`` cost about twice the ``jacfwd``
time.

=========  ==============  =========  ==================  ==============  ==========
N          numba value     JAX value  numba (N, 7) grad   JAX ``jacfwd``  JAX
           (serial / par)             (serial / par)                      ``grad``
=========  ==============  =========  ==================  ==============  ==========
1 000      9.5 / 29 us     29 us      35 / 68 us          121 us          101 us
10 000     49 / 47 us      63 us      234 / 113 us        294 us          332 us
100 000    0.43 / 0.13 ms  0.19 ms    2.2 / 0.43 ms       1.9 ms          3.1 ms
1 000 000  4.4 / 0.73 ms   1.1 ms     52 / 15 ms          83 ms           113 ms
=========  ==============  =========  ==================  ==============  ==========

``jax.grad`` of a scalar likelihood costs about 1.5 times the ``jacfwd`` time
at the larger sizes on this machine. So there is no reason to switch a working
numba model. The
JAX backend earns its keep in models that are JAX already (numpyro, blackjax,
jaxoplanet), in ``vmap`` over many parameter sets, and on accelerators; the
same script runs the JAX columns on a CUDA device with
``python benchmarks/bench_jax_vs_numba.py cuda``, and in float32 with
``--precision single``.


API
---

The full public surface of the two aggregators.

.. autoclass:: meepmeep.jax3d.JaxOrbit
   :members:
   :member-order: bysource

.. currentmodule:: meepmeep.jax3d

.. autosummary::
   :toctree: api/generated

   bounding_box
   cos_alpha
   cos_alpha_c
   cos_alpha_o
   cos_v_p_angle_o
   create_expansion_points
   ea_from_ma
   eccentricity_vector
   eclipse_time_offset
   emission_phase_curve
   emission_phase_curve_c
   emission_phase_curve_o
   ep_ix
   ev_signal
   ev_signal_c
   ev_signal_o
   expansion_table_size
   find_contact_point
   find_z_min
   lambert_phase_curve
   lambert_phase_curve_c
   lambert_phase_curve_o
   light_travel_time_o
   mean_anomaly_at_transit
   pos
   pos_c
   pos_o
   rv
   rv_c
   rv_o
   sep
   sep_c
   sep_o
   solve3d
   solve3d_orbit
   star_planet_distance_o
   t1
   t12
   t14
   t23
   t34
   t4
   true_anomaly_o
   vel
   vel_c
   vel_o
   zpos
   zpos_c
   zpos_o
   zvel
   zvel_c
   zvel_o

.. currentmodule:: meepmeep.jax2d

.. autosummary::
   :toctree: api/generated

   bounding_box
   find_contact_point
   find_z_min
   pos
   pos_c
   sep
   sep_c
   solve2d
   t1
   t12
   t14
   t23
   t34
   t4
