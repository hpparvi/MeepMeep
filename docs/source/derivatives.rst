.. _derivatives:

Analytic parameter derivatives
==============================

Gradient-based optimisers (Levenberg-Marquardt, L-BFGS) and HMC
samplers need the gradient of the model w.r.t. the parameters at every
iteration. Finite-differencing the orbit works but is expensive and
loses accuracy; automatic differentiation through JIT-compiled numba
code is not supported in general. The numba backend takes a third route:
hand-derived analytic gradients shipped as sibling routines next to
each evaluator. Each ``_d`` call costs only a few times what the
value-only call costs, the gradient is exact up to floating-point
error, and the result drops straight into a fitter or sampler. (The
optional JAX backend, :doc:`jax_backend`, ports the value evaluators to JAX
and gets the same gradients from autodiff instead; the test suites pin the
two together at round-off.)

In concrete terms, every quantity that the Taylor backend evaluates is
also exposed in a ``_d``-suffixed variant that returns the value
alongside its analytic partial derivatives w.r.t. the seven orbital
parameters (the transit-geometry searches, contact points, durations and
``find_z_min``, are the exception; their derivatives come only from the
JAX backend)

.. math::

   \boldsymbol{\theta} = (t_c,\, p,\, a,\, i,\, e,\, w,\, \Omega),

where :math:`t_c` is the transit-centre time (time of inferior
conjunction). The leading axis of every ``dc`` tensor follows this
ordering. The :math:`t_c` row is special: the evaluators depend on
:math:`t_c` only through the polynomial argument
:math:`t_\mathrm{obs} - t_c`, so it is the derivative of the truncated
polynomial itself, :math:`\partial c_n / \partial t_c = -(n+1)\, c_{n+1}`,
rather than a derivative propagated through Kepler's equation; see
:ref:`transit-centre-row` for why that distinction matters. This page
documents how those derivatives are computed, the explicit formulas at
each stage, and the practical regime in which they are accurate — useful
when you are verifying the math, extending the backend with a new
observable, or debugging a gradient mismatch.

.. contents::
   :local:
   :depth: 2


Using the gradients
-------------------

The high-level classes return the gradients when constructed with
``derivatives=True``. With an RV model, the gradient of
:math:`\chi^2` follows from one call:

.. code-block:: python

   import numpy as np
   from meepmeep import Orbit

   tc, p, a, i, e, w = 0.0, 3.0, 8.5, np.radians(88.0), 0.1, np.radians(60.0)
   o = Orbit(derivatives=True)
   o.set_pars(tc=tc, p=p, a=a, i=i, e=e, w=w)
   o.set_data(times)                        # times, rv_obs, rv_err: your data

   rv, drv = o.radial_velocity(k=50.0)      # drv: (N, 8), (tc, p, a, i, e, w, lan, k)
   dchi2 = -2.0 * ((rv_obs - rv) / rv_err**2) @ drv

Binding the orbit with ``set_pars(tp=...)`` instead returns every gradient in
the periastron basis ``(tp, p, a, i, e, w, lan)``. The single-expansion
classes, :class:`~meepmeep.expansion2d.Expansion2D` and
:class:`~meepmeep.expansion3d.Expansion3D`, are anchored at ``tc`` and
always return the transit-centre basis.

At the low level, a single expansion returns the transit-centre basis, and
:func:`~meepmeep.numba3d.tc_to_tp_gradient` converts the derivative
coefficients to the periastron basis. The evaluators still take ``tc`` as
their time anchor, because slot 0 means the same in both bases:

.. code-block:: python

   from meepmeep.numba3d import solve3d_d, sep_d, tc_to_tp_gradient

   c, dc = solve3d_d(0.0, p, a, i, e, w)       # expansion at the transit centre
   d, dd = sep_d(times, tc, p, c, dc)          # dd: (N, 7), transit-centre basis
   dc_tp = tc_to_tp_gradient(dc, p, e, w)      # a new array; dc is unchanged
   d, dd_tp = sep_d(times, tc, p, c, dc_tp)    # periastron basis

The whole-orbit solver works the other way round. It returns the
periastron basis, and :func:`~meepmeep.numba3d.tp_to_tc_gradient_orbit`
converts it in place:

.. code-block:: python

   from meepmeep.numba3d import (create_expansion_points, solve3d_orbit_d, sep_od,
                                 tp_to_tc_gradient_orbit, mean_anomaly_at_transit)

   ep_times, _, dt, ep_table = create_expansion_points(15, max(e, 0.2), "ea")
   coeffs, dcoeffs = solve3d_orbit_d(ep_times, p, a, i, e, w, npt=15)
   tpa = tc - mean_anomaly_at_transit(e, w) / (2 * np.pi) * p
   z, dz_tp = sep_od(times, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)
   tp_to_tc_gradient_orbit(dcoeffs, p, e, w)   # in place
   z, dz_tc = sep_od(times, tpa, p, dt, ep_table, ep_times, coeffs, dcoeffs)

Every gradient here matches central finite differences of the matching
value function to about :math:`10^{-8}` relative, on time grids that span
several epochs. The vector and parallel kernels (``sep_d_v`` / ``sep_d_vp``,
``sep_ovd`` / ``sep_ovdp``, ...) return the same arrays for large time
grids; see :doc:`naming_conventions`.


The two-layer chain
-------------------

The gradient computation splits into two layers. The boundary between
them is exactly where the analytic difficulty sits: everything that
depends on Kepler's equation lives in Layer A and is computed once per
expansion point; the per-evaluator math in Layer B is just polynomial
manipulation and one-line chain rules.

* **Layer A — derivative coefficients.** The solvers
  :func:`~meepmeep.backends.numba.point2dd.solve.solve2d_d` and
  :func:`~meepmeep.backends.numba.point3dd.solve.solve3d_d` produce
  the Taylor coefficient matrix ``c`` of shape ``(D, 5)`` *and* a
  parameter-derivative tensor ``dc`` of shape ``(7, D, 5)``. The
  element ``dc[k, d, n]`` is
  :math:`\partial c[d, n] / \partial \theta_k`. All non-trivial
  calculus lives here: Kepler's equation, the orbital-plane state, the
  rotation into the sky frame.

* **Layer B — evaluator propagation.** Every ``_d`` evaluator
  (positions, distances, velocities, RVs, phase-curve outputs) takes
  ``c`` and ``dc`` and reduces them to the final quantity together with
  its gradient: the seven orbital slots (the seventh being the longitude of
  the ascending node), followed by the physical inputs of the quantities
  that have them — ``(ag, k)`` for the Lambert phase curve and
  ``(alpha, mass_ratio)`` for the ellipsoidal variation (width 9),
  ``(k, fratio, offset)`` for the emission phase curve (width 10), and
  ``k`` for the multi-expansion-point ``rv_od`` (width 8; the
  single-expansion-point ``rv_d`` stays at 7). The reductions are either
  trivial polynomial evaluations or simple chain-rule applications.

The two layers are documented separately below.


Layer A: derivative coefficients
--------------------------------

The walk-through below mirrors
:func:`~meepmeep.backends.numba.point2dd.solve.solve2d_d` step by
step; the 3D solver
:func:`~meepmeep.backends.numba.point3dd.solve.solve3d_d` follows the
same structure with one extra row in the final rotation matrix.

The parameter indexing throughout is

==  ==============  ========================================================
k   Parameter       Comment
==  ==============  ========================================================
0   :math:`t_c`     Transit-centre time [days], inferior conjunction
1   :math:`p`       Orbital period
2   :math:`a`       Scaled semi-major axis
3   :math:`i`       Inclination
4   :math:`e`       Eccentricity
5   :math:`w`       Argument of periastron
6   :math:`\Omega`  Longitude of the ascending node
==  ==============  ========================================================

The first six parameters drive Kepler's equation and the orbital-plane
state; their analytic partials are built with the length-6 working
arrays described below. The seventh, :math:`\Omega`, is a constant
rotation of the sky-plane :math:`(x, y)` about the line of sight and
does *not* enter the Kepler solve. It is applied as a post-processing
rotation of the assembled coefficients: the six Kepler-parameter rows
are rotated by :math:`R(\Omega)`, and the new :math:`\Omega` row is
:math:`R'(\Omega)` applied to the unrotated position coefficients (the
line-of-sight :math:`z` row of the :math:`\Omega` derivative is zero).


Step 1 — auxiliary partials
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Several auxiliary quantities depend on only one parameter each, so
their gradient vectors are sparse:

.. math::

   n &= \frac{2\pi}{p},
     \qquad \frac{\partial n}{\partial p} = -\frac{2\pi}{p^2}, \\
   \mu &= n^2 a^3,
     \qquad \frac{\partial \mu}{\partial p} = 2 n\, \frac{\partial n}{\partial p}\, a^3,
     \qquad \frac{\partial \mu}{\partial a} = 3 n^2 a^2, \\
   \sqrt{1-e^2}\; &\Longrightarrow\; \frac{\partial}{\partial e}\sqrt{1-e^2}
     = -\frac{e}{\sqrt{1-e^2}}, \\
   (\cos i,\, \sin i) &\Longrightarrow
     \frac{\partial \cos i}{\partial i} = -\sin i,\quad
     \frac{\partial \sin i}{\partial i} = \cos i, \\
   (\cos w,\, \sin w) &\Longrightarrow
     \frac{\partial \cos w}{\partial w} = -\sin w,\quad
     \frac{\partial \sin w}{\partial w} = \cos w.

These feed every later step. The code stores each as a length-6 array
with one non-zero entry; the surrounding loops therefore mostly carry
zeros until eccentricity and orientation enter the chain.


Step 2 — mean anomaly at transit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mean anomaly at the moment of inferior conjunction,
:math:`M_\text{tr}(e, w)`, is non-trivial in :math:`e` and :math:`w`.
The helper
:func:`~meepmeep.backends.numba.utils.mean_anomaly_at_transit_with_derivatives`
returns the value and the two partials in closed form (derived by
implicit differentiation of an arctangent expression for the
eccentric anomaly at transit). Schematically,

.. math::

   M_\text{tr} = E_\text{tr} - e \sin E_\text{tr},
   \qquad
   E_\text{tr} = \operatorname{atan2}\!\bigl(\sqrt{1-e^2}\,\cos w,\; e + \sin w\bigr),

with :math:`\partial M_\text{tr} / \partial e` and
:math:`\partial M_\text{tr} / \partial w` formed by differentiating
this composite. The solver stores the result in ``doffset[4]`` and
``doffset[5]``.


Step 3 — mean anomaly and its gradient
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mean anomaly at the expansion-point time :math:`t_k` (the solver's ``te``
argument, a fixed time measured relative to the transit centre; the
evaluators later take the polynomial argument
:math:`\tau = t_\mathrm{obs} - t_c - t_k`, see :ref:`transit-centre-row`)
is

.. math::

   M(t_k; p, e, w) \;=\; \frac{2\pi\, t_k}{p} \;+\; M_\text{tr}(e, w) \pmod{2\pi},

so, differentiating w.r.t. the parameter vector,

.. math::

   \frac{\partial M}{\partial p} = -\frac{2\pi\, t_k}{p^2}, \qquad
   \frac{\partial M}{\partial e} = \frac{\partial M_\text{tr}}{\partial e}, \qquad
   \frac{\partial M}{\partial w} = \frac{\partial M_\text{tr}}{\partial w},

with the ``a`` and ``i`` entries zero. Slot 0, the transit-centre time,
is *not* propagated from here (its entry is zero throughout Layer A):
the transit centre enters the evaluators only through the polynomial
argument, so its row is assembled from the coefficients at the end
(:ref:`transit-centre-row`).


Step 4 — eccentric anomaly via implicit differentiation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Kepler's equation

.. math::

   E \;-\; e \sin E \;=\; M

is solved numerically by
:func:`~meepmeep.backends.numba.newton.newton.ea_from_ma` (Newton-
Raphson). Once :math:`E` is known, differentiating both sides with
respect to a generic parameter :math:`\theta_k` gives

.. math::

   \bigl(1 - e \cos E\bigr)\, \frac{\partial E}{\partial \theta_k}
     \;=\; \frac{\partial M}{\partial \theta_k}
       \;+\; \sin E\; \frac{\partial e}{\partial \theta_k},

so

.. math::

   \boxed{\;
     \frac{\partial E}{\partial \theta_k}
       \;=\; \frac{1}{1 - e \cos E}\,
             \left(\frac{\partial M}{\partial \theta_k}
                   \;+\; \sin E \cdot \delta_{k=4}\right).
   \;}

Here :math:`\delta_{k=4}` is the Kronecker delta selecting the
eccentricity slot. The :math:`(1 - e \cos E)^{-1}` factor is the same
Jacobian that appears in the Newton-Raphson iteration of the forward
solve; the solver re-evaluates it once at the converged :math:`E`.

The partials of :math:`\sin E` and :math:`\cos E` then follow trivially
by the chain rule.


Step 5 — orbital-plane position and velocity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the orbital plane, with :math:`\xi` along the line to periastron
and :math:`\eta` perpendicular to it,

.. math::

   r &= a\bigl(1 - e \cos E\bigr), \\
   \xi &= a\bigl(\cos E - e\bigr), \\
   \eta &= a\,\sqrt{1-e^2}\; \sin E, \\
   \dot E &= \frac{n a}{r}, \\
   v_\xi &= -a \sin E \cdot \dot E, \\
   v_\eta &= a \sqrt{1-e^2}\,\cos E \cdot \dot E.

Each of these is a product/quotient of factors whose partials are
either already in scope (steps 1, 3, 4) or zero. The solver expands
each derivative as a sum of product-rule terms and stores the result
in ``dr[k]``, ``dxi[k]``, ``deta[k]``, ``dv_xi[k]``, ``dv_eta[k]``.


Step 6 — higher-order Taylor terms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 4th-order Taylor expansion needs position, velocity, acceleration,
jerk, and snap at the expansion point. Rather than differentiating
:math:`r(t)` and the angles directly, the solver expresses each
higher derivative in terms of :math:`(\xi, \eta)`,
:math:`(v_\xi, v_\eta)`, and the Newtonian gravitational acceleration.

Define the auxiliary radial scalar

.. math::

   u \;=\; -\frac{\mu}{r^3},

so the acceleration in the orbital plane is :math:`(a_\xi, a_\eta) =
u\,(\xi, \eta)`. Two further radial scalars are needed:

.. math::

   \dot u &= \frac{3 \mu\, (\mathbf r \cdot \mathbf v)}{r^5}
          \;=\; \frac{3 \mu\, (\xi v_\xi + \eta v_\eta)}{r^5}, \\
   \ddot u &= 3 \mu \left( \frac{v^2}{r^5} - \frac{5 (\mathbf r \cdot \mathbf v)^2}{r^7} \right)
            \;-\; 3 u^2,

with :math:`v^2 = v_\xi^2 + v_\eta^2`. From these, the jerk and snap
in the orbital plane are

.. math::

   (j_\xi, j_\eta) &= \dot u\, (\xi, \eta) \;+\; u\, (v_\xi, v_\eta), \\
   (s_\xi, s_\eta) &= (\ddot u + u^2)\,(\xi, \eta)
                      \;+\; 2 \dot u\, (v_\xi, v_\eta).

Each of :math:`u, \dot u, \ddot u` and the resulting jerk and snap
vectors is differentiated by the product rule. The recurring building
blocks are :math:`\partial r^{-n} / \partial \theta_k =
-n\, r^{-n-1}\, \partial r / \partial \theta_k`, which the solver
computes once per inverse power and reuses.


Step 7 — rotation into the sky frame
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 2D sky-plane projection is a constant rotation depending on
:math:`(i, w)`:

.. math::

   \begin{pmatrix} m_{00} & m_{01} \\ m_{10} & m_{11} \end{pmatrix}
   \;=\;
   \begin{pmatrix} -\cos w & \sin w \\ -\sin w\, \cos i & -\cos w\, \cos i \end{pmatrix}.

The 3D solver adds a third row, :math:`(\sin w\, \sin i,\; \cos w\, \sin i)`,
yielding the line-of-sight component. The non-zero entries of the
:math:`\partial m_{rc} / \partial \theta_k` arrays are immediate from
step 1.

For each Taylor order :math:`n \in \{0, 1, 2, 3, 4\}` and each spatial
dimension :math:`d`, the stored coefficient is

.. math::

   c[d, n] \;=\; \frac{1}{n!}\, \bigl(R\, q^{(n)}\bigr)[d],

where :math:`q^{(n)}` is the :math:`n`-th orbital-plane vector
(:math:`(\xi, \eta)` for :math:`n=0`, :math:`(v_\xi, v_\eta)` for
:math:`n=1`, acceleration for :math:`n=2`, and so on). The
factorial pre-scaling means the coefficient is the actual Taylor
coefficient, not the raw derivative; consequently the evaluator does
no factorial divisions.

The derivative coefficient follows from the product rule on the
rotation:

.. math::

   \boxed{\;
     \frac{\partial c[d, n]}{\partial \theta_k}
       \;=\; \frac{1}{n!}\,
             \Bigl(
               \frac{\partial R}{\partial \theta_k}\, q^{(n)}
               \;+\; R\, \frac{\partial q^{(n)}}{\partial \theta_k}
             \Bigr)[d].
   \;}

:math:`R` carries the dependence on :math:`(i, w)`, and :math:`q^{(n)}` the
dependence on :math:`(p, a, e)`. When the expansion time is measured from
the transit centre, :math:`q^{(n)}` also depends on :math:`w`, through the
mean anomaly at transit :math:`M_\mathrm{tr}(e, w)` that places the
expansion point on the orbit; when it is measured from periastron
(``from_periastron=True``) it does not. Neither factor carries :math:`t_c`:
that slot is zero at this stage and is built from the coefficients
themselves in the final step (see :ref:`transit-centre-row`), and the node
:math:`\Omega` enters through a last rotation. Up to those two steps, the
output ``dcf`` is the tensor whose entries are exactly the right-hand side
of this boxed identity.


Step 8 — longitude of the ascending node
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The node is applied last, as described under the parameter table above:
the six Kepler-parameter rows and the coefficients are rotated by
:math:`R(\Omega)` in the sky plane, and the :math:`\Omega` row is
:math:`R'(\Omega)` applied to the unrotated coefficients, with a zero
line-of-sight row.


.. _transit-centre-row:

Step 9 — the transit-centre row
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every evaluator computes a polynomial in the elapsed time,
:math:`P(\tau) = \sum_{n=0}^{4} c_n \tau^n` with
:math:`\tau = t_\mathrm{obs} - t_c - t_k`, so the transit centre enters
only through the argument and

.. math::

   \frac{\partial P}{\partial t_c} = -P'(\tau)
   \quad\Longrightarrow\quad
   \frac{\partial c_n}{\partial t_c} = -(n+1)\, c_{n+1}
   \ \ (n = 0, \dots, 3), \qquad
   \frac{\partial c_4}{\partial t_c} = 0 .

The solvers build the row from this identity rather than by propagating
:math:`\partial M / \partial t_c = -2\pi/p` through Kepler's equation
as they do for the other parameters. The two are not the same thing.
Propagating through Kepler's equation differentiates the *exact* orbit
and its fourth-order entry is :math:`-5\, c_5`, the fifth-order
coefficient the expansion does not carry; evaluated in the polynomial
it adds :math:`x^{(5)}\, \tau^4 / 4!` to the :math:`t_c` derivative,
which is exact for the true orbit but is not the derivative of the value
the evaluator returns. The two differ by about :math:`10^{-5}` relative
at :math:`\tau \approx 0.07` d for a typical hot-Jupiter orbit
(:math:`a = 6`, :math:`p = 3` d) and grow as :math:`\tau^4`, reaching
about :math:`10^{-2}` at :math:`\tau \approx 0.3` d, where a single
expansion point is stretched anyway.

MeepMeep ships the derivative of the polynomial because the gradient is
consumed by optimisers and samplers that need the gradient of the model
*actually evaluated*: Hamiltonian Monte Carlo, for instance, relies on
the gradient and the value belonging to the same function, and finite
differences or automatic differentiation of the evaluator give this
form. The accuracy of either form against the true orbit is set by the
expansion's own truncation and is the same. The period column inherits
the choice through the period-folding term ``dd[1] += epoch * dd[0]``
of the absolute-time evaluators, so multi-epoch period gradients are
consistent as well. The identity is exact, so the row can be checked to
round-off: finite differences of ``sep_d`` or ``pos_d`` with respect to
their ``tc`` argument agree with slot 0 to :math:`10^{-9}` at any time,
not just near the expansion point. If you need the derivative of the
true orbit with respect to time instead (a physical velocity), use the
velocity evaluators.

Layer B: evaluator propagation
------------------------------

Once ``c`` and ``dc`` are in hand, every quantity the backend can
evaluate is a closed-form function of them and the centered time
``t``. The propagation rules below are all that the ``_d`` evaluators
contain.


Position
~~~~~~~~

The Horner polynomial in ``c`` already gives the position. The
gradient is the corresponding Horner polynomial in ``dc[k, d, :]``
for each parameter :math:`k`:

.. math::

   p_d(t) \;=\; \sum_{n=0}^{4} c[d, n]\, t^n,
   \qquad
   \frac{\partial p_d}{\partial \theta_k}(t)
     \;=\; \sum_{n=0}^{4} dc[k, d, n]\, t^n.

Implemented in
:func:`~meepmeep.backends.numba.point2dd.position.pos_cd`,
:func:`~meepmeep.backends.numba.point3dd.position.pos_cd`,
and their direct counterparts
:func:`~meepmeep.backends.numba.point2dd.position.pos_d` and
:func:`~meepmeep.backends.numba.point3dd.position.pos_d` (which epoch-fold
``t`` first; the ``-epoch * p`` term in the folded time makes the period
derivative gain ``epoch`` times the ``tc`` column, which the kernels add).


Projected separation
~~~~~~~~~~~~~~~~~~~~

For :math:`d = \sqrt{p_x^2 + p_y^2}`, differentiating
:math:`d^2` gives

.. math::

   \boxed{\;
     \frac{\partial d}{\partial \theta_k}
       \;=\; \frac{p_x\, \partial p_x/\partial \theta_k
                   \;+\; p_y\, \partial p_y/\partial \theta_k}{d}.
   \;}

The same reduction is applied in 2D and 3D
(:func:`~meepmeep.backends.numba.point2dd.separation.sep_cd`,
:func:`~meepmeep.backends.numba.point3dd.separation.sep_cd`); both
treat :math:`d` as the **projected** separation. The expression is
regular for :math:`d > 0` and ill-defined at exactly zero projected
separation; transit modelling stays well clear of this geometric
singularity.


Z-coordinate
~~~~~~~~~~~~

The line-of-sight coordinate :math:`z` is just the third row of the
position polynomial, so its gradient is the polynomial in
``dc[k, 2, :]``. See
:func:`~meepmeep.backends.numba.point3dd.zposition.zpos_cd`.


Line-of-sight velocity
~~~~~~~~~~~~~~~~~~~~~~

The velocity polynomial is the term-by-term derivative of the position
polynomial, with the factorial pre-scaling exactly cancelling the
:math:`n` in front of :math:`t^{n-1}`:

.. math::

   v_z(t) \;=\; \frac{\mathrm d}{\mathrm d t}
                \sum_{n=0}^{4} c[2, n]\, t^n
            \;=\; c[2, 1]
                  + 2 c[2, 2]\, t
                  + 3 c[2, 3]\, t^2
                  + 4 c[2, 4]\, t^3.

The gradient is the same polynomial pattern on ``dc``. See
:func:`~meepmeep.backends.numba.point3dd.zvelocity.zvel_cd`.


Radial velocity
~~~~~~~~~~~~~~~

The radial velocity carries an additional parameter dependence through
the conversion between the internal velocity (in :math:`R_\star /
\text{day}`) and the observed RV in :math:`\text{m s}^{-1}`:

.. math::

   \mathrm{RV} \;=\; K \cdot \frac{v_z}{n_z},
   \qquad
   n_z \;=\; \frac{2\pi}{p}\, \frac{a \sin i}{\sqrt{1-e^2}}.

Pulling the scalar :math:`s = K / n_z` out front,

.. math::

   \frac{\partial \mathrm{RV}}{\partial \theta_k}
     \;=\; s\, \frac{\partial v_z}{\partial \theta_k}
           \;+\; v_z\, \frac{\partial s}{\partial \theta_k},

with closed-form non-zero entries

.. math::

   \frac{\partial s}{\partial p} = +\frac{s}{p},
   \qquad
   \frac{\partial s}{\partial a} = -\frac{s}{a},
   \qquad
   \frac{\partial s}{\partial i} = -s\, \cot i,
   \qquad
   \frac{\partial s}{\partial e} = -\frac{s\, e}{1 - e^2}.

The :math:`(t_c, w)` partials of :math:`s` vanish. Implemented in
:func:`~meepmeep.backends.numba.point3dd.radial_velocity.rv_cd` and
:func:`~meepmeep.backends.numba.point3dd.radial_velocity.rv_d`.


Phase angle and 3D separation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The whole-orbit dispatchers in
:mod:`~meepmeep.backends.numba.orbit3dd` compose further chain
rules on top of the position gradient. The two recurring patterns are
the 3D separation

.. math::

   r \;=\; \sqrt{x^2 + y^2 + z^2}
   \;\Longrightarrow\;
   \frac{\partial r}{\partial \theta_k}
     \;=\; \frac{x\, \partial x/\partial \theta_k
                 \;+\; y\, \partial y/\partial \theta_k
                 \;+\; z\, \partial z/\partial \theta_k}{r}

and the cosine of the phase angle :math:`\cos \alpha = -z / r`:

.. math::

   \frac{\partial}{\partial \theta_k} \!\left(-\frac{z}{r}\right)
     \;=\; -\frac{1}{r}\, \frac{\partial z}{\partial \theta_k}
           \;+\; \frac{z}{r^3}\,
                 \Bigl(x\, \tfrac{\partial x}{\partial \theta_k}
                       + y\, \tfrac{\partial y}{\partial \theta_k}
                       + z\, \tfrac{\partial z}{\partial \theta_k}\Bigr).

These appear in
:func:`~meepmeep.backends.numba.orbit3dd.star_planet_distance_od`
and
:func:`~meepmeep.backends.numba.orbit3dd.cos_alpha_od`. The
single-expansion-point Lambertian phase-curve evaluator
(:func:`~meepmeep.backends.numba.point3dd.lambert.lambert_phase_curve_cd`,
which the whole-orbit
:func:`~meepmeep.backends.numba.orbit3dd.lambert_phase_curve_od`
delegates to) forms the flux :math:`(k/r)^2 A_g\, f(\cos\alpha)`, combining the
Lambert kernel
:math:`f(\cos\alpha) = (\sin\alpha + (\pi - \alpha)\cos\alpha)/\pi`
(with its closed-form derivative :math:`\mathrm d f / \mathrm d \cos\alpha`)
and the inverse-square illumination :math:`1/r^2` set by the instantaneous
star-planet distance. Its gradient therefore chains through both
:math:`\cos\alpha` and :math:`r`, using the two derivatives above.


Velocity
~~~~~~~~

The velocity evaluators (``vel_cd`` / ``vel_d`` / ``vel_od``) differentiate
the position polynomial in time,
:math:`v(\tau) = \sum_{n=1}^{4} n\, c_n\, \tau^{n-1}`, and apply the same
differentiation to each derivative-coefficient row, so
:math:`\partial v / \partial \theta_k = \sum_n n\, \partial c_n/\partial\theta_k\,
\tau^{n-1}`. The direct and whole-orbit forms add the period-folding
chain term as for positions.

Ellipsoidal variation
~~~~~~~~~~~~~~~~~~~~~

The signal is

.. math::

   \mathrm{EV} \;=\; -\alpha\, q\, \sin^2 i\;
                    \frac{2\,(z/r)^2 - 1}{r^3},

with the gravity-darkening coefficient :math:`\alpha` and the mass ratio
:math:`q`. Its gradient chains through :math:`z` and :math:`r` as above,
and the explicit :math:`\sin^2 i` adds
:math:`2\,\mathrm{EV}\cot i` to the inclination slot. The two extra slots
are :math:`\mathrm{EV}/\alpha` and :math:`\mathrm{EV}/q`.

Thermal emission
~~~~~~~~~~~~~~~~

The emission model is
:math:`F = k^2 f_\mathrm{ratio}\, \tfrac12\bigl(1 + \cos\delta\; c_z
+ \sin\delta\; s\bigr)`, where :math:`c_z = -z/r` is the phase-angle cosine
and :math:`s = -(w_x y - w_y x)/(|w|\, r)` a signed in-plane component
built from the orbit normal :math:`w = x \times v`. The gradient chains
through the position *and* velocity gradients (the latter via
:math:`w`). The extra slots are :math:`2F/k`,
:math:`F/f_\mathrm{ratio}` and
:math:`k^2 f_\mathrm{ratio}\,\tfrac12(-\sin\delta\; c_z + \cos\delta\; s)`
for the hotspot offset :math:`\delta`.

Light travel time
~~~~~~~~~~~~~~~~~

The delay relative to the transit is
:math:`\Delta t = -(z(t) - z(t_\mathrm{tr}))\, R_\star / c`, with
:math:`t_\mathrm{tr}` the transit time. The second term moves with the
parameters in two ways, through the orbit and through the transit time
itself:

.. math::

   \frac{\mathrm d\, z(t_\mathrm{tr})}{\mathrm d \theta_k}
     \;=\; \left.\frac{\partial z}{\partial \theta_k}\right|_{t_\mathrm{tr}}
     \;+\; v_z(t_\mathrm{tr})\, \frac{\mathrm d t_\mathrm{tr}}{\mathrm d \theta_k}.

In the transit-centre basis :math:`t_\mathrm{tr} = t_c`, so only the timing
slot of :math:`\mathrm d t_\mathrm{tr}/\mathrm d\theta` is non-zero. In the
periastron basis :math:`t_\mathrm{tr} = t_p + M_\mathrm{tr}(e, w)\, p/2\pi`
also moves with :math:`p`, :math:`e` and :math:`w`. That is why
``light_travel_time_od`` takes ``timing_is_tc``. The :math:`R_\star`
derivative is not returned.

True anomaly
~~~~~~~~~~~~

The true anomaly is the angle between the position and the eccentricity
vector :math:`\mathbf{e}`:
:math:`f = \arccos(\hat{\mathbf x}\cdot\hat{\mathbf e})`, with the branch
chosen from the mean anomaly. Both vectors depend on the parameters, so
with :math:`u = \mathbf x\cdot\mathbf e / (|\mathbf x|\,|\mathbf e|)`

.. math::

   \frac{\partial u}{\partial \theta_k}
     = \frac{\partial_k\mathbf x\cdot\mathbf e + \mathbf x\cdot\partial_k\mathbf e}{|\mathbf x|\,|\mathbf e|}
       - u\,\frac{\mathbf x\cdot\partial_k\mathbf x}{|\mathbf x|^2}
       - u\,\frac{\mathbf e\cdot\partial_k\mathbf e}{|\mathbf e|^2},
   \qquad
   \frac{\partial f}{\partial \theta_k} = \mp\frac{\partial_k u}{\sqrt{1 - u^2}}.

The eccentricity vector turns with :math:`w` and :math:`\Omega`, and its
Jacobian :math:`\partial\mathbf e/\partial\theta` comes from
:func:`~meepmeep.numba3d.eccentricity_vector_d` (the ``dev`` argument of
:func:`~meepmeep.numba3d.true_anomaly_od`). Without that term the ``w``
and :math:`\Omega` slots would carry only the position's half of the
rotation. At the apsides (:math:`u = \pm 1`) the expression is 0/0 and the
gradient is set to zero. For a circular orbit (the eccentricity-vector
sentinel) the true anomaly is the mean anomaly,
:math:`f = 2\pi(t - t_p)/p`, differentiated directly.

Angle to a fixed vector
~~~~~~~~~~~~~~~~~~~~~~~

``cos_v_p_angle_od`` returns the cosine of the angle between the position
and a fixed vector :math:`\mathbf v`,
:math:`\mathbf x\cdot\mathbf v/(|\mathbf x|\,|\mathbf v|)`, whose gradient is
:math:`(\partial_k\mathbf x\cdot\mathbf v)/(|\mathbf x|\,|\mathbf v|)
- (\mathbf x\cdot\mathbf v)(\mathbf x\cdot\partial_k\mathbf x)/(|\mathbf x|^3|\mathbf v|)`.

The node slot
~~~~~~~~~~~~~

:math:`\Omega` rotates the sky plane about the line of sight, which leaves
:math:`z`, :math:`r`, the projected separation :math:`d` and the
line-of-sight velocity unchanged. Their :math:`\Omega` slots, and those of
everything built only from them (the phase-angle cosine, the Lambert and
emission phase curves, the ellipsoidal variation, the radial velocity, the
light travel time), are zero up to round-off (about :math:`10^{-16}`). Only
the sky-plane coordinates and velocities carry a non-zero :math:`\Omega`
slot.


Multi-expansion-point propagation
---------------------------------

The orbit-spanning solver
:func:`~meepmeep.backends.numba.orbit3dd.solve3d_orbit_d`
applies
:func:`~meepmeep.backends.numba.point3dd.solve.solve3d_d` once per
expansion point and stacks the results into arrays of shape ``(N, 3, 5)`` for
``coeffs`` and ``(N, 7, 3, 5)`` for ``dcoeffs``. The expansion points sit at
fixed phases from periastron, so the solver runs in the periastron basis
(``from_periastron=True``), and each expansion time ``p * phase`` moves with
the period: every slot's period row gains ``phase * dcf[0]``. The last slot
is the periodic image of the first, one period later. ``coeffs[-1]`` is a
copy of ``coeffs[0]``, and ``dcoeffs[-1]`` is a copy of ``dcoeffs[0]`` whose
period row additionally gains ``(ep_times[-1] - ep_times[0]) * dcoeffs[0, 0]``
for its own phase. :class:`~meepmeep.orbit.Orbit` converts the result to the
transit-centre basis with
:func:`~meepmeep.numba3d.tp_to_tc_gradient_orbit` when it is bound by ``tc``.

Each multi-expansion-point ``_d`` dispatcher then performs the same single-expansion-point
chain rule documented above after a ``ep_table``-driven expansion point lookup:

#. Epoch-fold the absolute time against the periastron anchor ``tpa``:
   ``epoch = floor((t - tpa) / p)``, so the folded time is
   ``t - tpa - epoch * p``.
#. Look up the expansion-point index ``ix`` from the time-to-expansion-point table.
#. Subtract the expansion point phase to get the centered time.
#. Call the matching centered single-expansion-point ``_d`` evaluator with
   ``coeffs[ix]`` and ``dcoeffs[ix]``.
#. Add the period-folding chain term. The folded time depends on the
   period through ``-epoch * p``, so each gradient's period slot gains
   ``epoch`` times its timing slot: ``d[1] += epoch * d[0]``. The term
   vanishes at epoch 0 and grows with the orbit count.

The chain rule for higher-level outputs (phase angle, Lambert curve, RV,
light travel time) is then applied identically to the single-expansion-point
case, using the dispatched value and gradient, which already include the
chain term.


Numerical regime and pitfalls
-----------------------------

* **Validity window.** Each expansion point's Taylor expansion is accurate within
  a region around the expansion point whose size depends on the orbit; near
  periastron of an eccentric orbit the window is narrowest. The
  *gradient* is accurate inside the same region — its truncation error
  has the same order as the value's.

* **Projected-separation singularity.** The chain rule for
  :math:`\partial d / \partial \theta` diverges as :math:`d \to 0`.
  This is a geometric, not numerical, singularity (the direction of
  ascent is undefined when the projected separation vanishes). It is
  outside the transit-modelling regime.

* **Eccentricity edge cases.** The implicit-differentiation form of
  :math:`\partial E / \partial \theta` remains finite for all
  :math:`e \in [0, 1)`: the denominator :math:`1 - e \cos E` is
  bounded below by :math:`1 - e > 0`. The small-eccentricity branch
  used in
  :func:`~meepmeep.backends.numba.utils.eccentricity_vector` is a
  numerical convenience for the orientation calculation and does not
  enter the derivative chain documented here.

* **Floating-point precision.** The ``_d`` routines run under
  :func:`numba.njit` with ``fastmath=True``, except the true-anomaly
  kernels, which keep strict math because their ``arccos`` argument sits near
  :math:`\pm 1`. In practice this yields
  gradients agreeing with finite-difference checks to roughly
  :math:`10^{-9}` relative error for typical transit parameters — the
  same envelope the value-only evaluators inhabit.

* **Slot-0 convention.** Slot 0 is the partial with respect to the
  transit-centre time :math:`t_c`, taken of the truncated polynomial the
  evaluators compute (:ref:`transit-centre-row`). It is built once at the
  source (``dcf[0, :, n] = -(n + 1) cf[:, n + 1]`` in ``solve2d_d`` /
  ``solve3d_d``) and propagates linearly through every evaluator, so all
  ``_d`` / ``_od`` outputs report :math:`\partial / \partial t_c`
  consistently, and finite differences of any evaluator with respect to
  its ``tc`` argument reproduce it to round-off.

Transit-centre vs periastron parametrisation
--------------------------------------------

The two timing parametrisations, :math:`(t_c, p, a, i, e, w, \Omega)` with the
shape derivatives taken holding the transit centre fixed and
:math:`(t_p, p, a, i, e, w, \Omega)` holding the periastron time fixed, are
related by the exact, parameter-dependent offset

.. math::

   t_c = t_p + M_\mathrm{tr}(e, w)\, \frac{p}{2\pi},

so the periastron-basis gradient is the transit-centre gradient with multiples
of the timing row added to the :math:`p`, :math:`e`, and :math:`w` rows:

.. math::

   \frac{\partial f}{\partial p}\Big|_{t_p} &=
       \frac{\partial f}{\partial p}\Big|_{t_c}
       + \frac{\partial f}{\partial t_c}\, \frac{M_\mathrm{tr}}{2\pi}, \\
   \frac{\partial f}{\partial e}\Big|_{t_p} &=
       \frac{\partial f}{\partial e}\Big|_{t_c}
       + \frac{\partial f}{\partial t_c}\, \frac{\partial M_\mathrm{tr}}{\partial e}\, \frac{p}{2\pi}, \\
   \frac{\partial f}{\partial w}\Big|_{t_p} &=
       \frac{\partial f}{\partial w}\Big|_{t_c}
       + \frac{\partial f}{\partial t_c}\, \frac{\partial M_\mathrm{tr}}{\partial w}\, \frac{p}{2\pi},

and the reverse transform subtracts the same multiples. The :math:`a`,
:math:`i`, and :math:`\Omega` rows are unchanged, and slot 0 is numerically
identical in both bases (it equals :math:`\partial/\partial t_c =
\partial/\partial t_p` when the other parameters are held fixed). The
transforms are :func:`~meepmeep.numba3d.tc_to_tp_gradient` and
:func:`~meepmeep.numba3d.tp_to_tc_gradient`, with the mean-anomaly-at-transit
terms from
:func:`~meepmeep.backends.numba.utils.mean_anomaly_at_transit_with_derivatives`.

**Which basis is native depends on where the expansion points are anchored**,
and this is not a formality. A transform is exact for the truncated polynomial
only if the timing parameter enters the evaluated model purely through the
polynomial argument, so that its row is the polynomial's own derivative
(:ref:`transit-centre-row`). If instead the timing parameter moves the
expansion points relative to the orbit, the shape derivatives differentiated
through Kepler's equation at fixed expansion time carry an implicit exact-orbit
time shift, and the transform leaks the fifth-order term.

* The single-expansion-point solvers
  :func:`~meepmeep.backends.numba.point2dd.solve.solve2d_d` /
  :func:`~meepmeep.backends.numba.point3dd.solve.solve3d_d` anchor the
  expansion time ``te`` at the transit centre by default and return the
  **transit-centre** basis, exact for a model whose ``te`` is held fixed from
  :math:`t_c`. ``tc_to_tp_gradient`` converts such a block exactly. With
  ``from_periastron=True`` they anchor ``te`` at periastron and return the
  periastron basis instead.

* The orbit-spanning solver
  :func:`~meepmeep.backends.numba.orbit3dd.solve3d_orbit_d` places its
  expansion points at fixed phases from periastron and therefore calls the
  point solver with ``from_periastron=True``: its rows are natively the
  **periastron** basis, exact at fixed phase, with the period row holding
  the phase (not the expansion time) fixed and the solver adding
  :math:`\phi_k\, \partial c/\partial t_p` for the shift of the expansion
  time :math:`p\,\phi_k` with the period. Differentiating at fixed
  expansion time from the transit centre there would leave the ``e``, ``w``
  and ``p`` columns off by the fifth-order term (about :math:`10^{-4}` in a
  0.04 d window around transit for a hot-Jupiter orbit), whichever basis is
  reported.

An :class:`~meepmeep.orbit.Orbit` therefore returns the periastron basis
untouched when bound with ``set_pars(tp=...)`` and applies
``tp_to_tc_gradient_orbit`` to the per-expansion-point coefficient derivatives
when bound with ``set_pars(tc=...)``, so the transit-centre basis propagates
consistently to every derivative-returning quantity (radial velocity, position,
separation, phase curves, ...). Either way the gradient is the exact gradient
of the value the class returns, and central finite differences of any quantity
with respect to any parameter, in either basis, reproduce it to round-off, with
two caveats. The expansion-point grid must stay fixed: in ``e`` the class
rebuilds it when the eccentricity drifts by more than 0.05 (see
:doc:`orbit_overview`), which makes the value jump by the truncation error.
And the true anomaly's gradient is set to zero exactly at the apsides, where
its analytic form is 0/0.
