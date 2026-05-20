.. _derivatives:

Analytic parameter derivatives
==============================

Every quantity that the Taylor backend evaluates is also exposed in a
``_d``-suffixed variant that returns the value alongside its
analytic partial derivatives with respect to the six orbital
parameters

.. math::

   \boldsymbol{\theta} = (\phi_0,\, p,\, a,\, i,\, e,\, w),

where :math:`\phi_0` is the expansion phase passed to the solver (in
the single-knot transit-modelling workflow, :math:`\phi_0 \equiv t_0`,
the time of transit center). The leading axis of every ``dc`` tensor
follows this ordering. This page documents how the derivatives are
computed, the explicit formulas at each stage, and the practical
regime in which they are accurate.

.. contents::
   :local:
   :depth: 2


Why analytic, and the two-layer chain
-------------------------------------

The numba backend is JIT-compiled with :func:`numba.njit` and
``fastmath=True``; automatic differentiation through such code is not
supported in general, so MeepMeep derives every gradient by hand and
ships it as a sibling routine to each evaluator. The arrangement is
fast (each ``_d`` call is at most a few times slower than the value-only
call), exact up to floating-point error, and compatible with
gradient-based fitters (Levenberg-Marquardt, HMC, L-BFGS) operating in
inner loops.

The chain splits into two layers:

* **Layer A — derivative coefficients.** The solvers
  :func:`~meepmeep.backends.numba.taylor.solve2dd.solve2d_d` and
  :func:`~meepmeep.backends.numba.taylor.solve3dd.solve3d_d` produce
  the Taylor coefficient matrix ``c`` of shape ``(D, 5)`` *and* a
  parameter-derivative tensor ``dc`` of shape ``(6, D, 5)``. The
  element ``dc[k, d, n]`` is
  :math:`\partial c[d, n] / \partial \theta_k`. All non-trivial
  calculus lives here: Kepler's equation, the orbital-plane state, the
  rotation into the sky frame.

* **Layer B — evaluator propagation.** Every ``_d`` evaluator
  (positions, distances, velocities, RVs, phase-curve outputs) takes
  ``c`` and ``dc`` and reduces them to the final quantity together with
  its 6-vector gradient. The reductions are either trivial polynomial
  evaluations or simple chain-rule applications.

The two layers are documented separately below.


Layer A: derivative coefficients
--------------------------------

The walk-through below mirrors
:func:`~meepmeep.backends.numba.taylor.solve2dd.solve2d_d` step by
step; the 3D solver
:func:`~meepmeep.backends.numba.taylor.solve3dd.solve3d_d` follows the
same structure with one extra row in the final rotation matrix.

The parameter indexing throughout is

==  ==========  ==============================================
k    Parameter   Comment
==  ==========  ==============================================
0    :math:`\phi_0`   Taylor expansion phase / time of transit
1    :math:`p`        Orbital period
2    :math:`a`        Scaled semi-major axis
3    :math:`i`        Inclination
4    :math:`e`        Eccentricity
5    :math:`w`        Argument of periastron
==  ==========  ==============================================


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

The mean anomaly at the expansion phase is

.. math::

   M(\phi_0; p, e, w) \;=\; \frac{2\pi\, \phi_0}{p} \;+\; M_\text{tr}(e, w) \pmod{2\pi},

so

.. math::

   \frac{\partial M}{\partial \phi_0} = \frac{2\pi}{p}, \qquad
   \frac{\partial M}{\partial p} = -\frac{2\pi\, \phi_0}{p^2}, \qquad
   \frac{\partial M}{\partial e} = \frac{\partial M_\text{tr}}{\partial e}, \qquad
   \frac{\partial M}{\partial w} = \frac{\partial M_\text{tr}}{\partial w},

with the other two entries (``a``, ``i``) zero.


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
solve, so it is already at hand.

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

The 5th-order Taylor expansion needs position, velocity, acceleration,
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

Only :math:`R` depends on :math:`(i, w)`; only :math:`q^{(n)}` carries
the dependence on :math:`(\phi_0, p, a, e)`. The output ``dcf`` is the
tensor whose entries are exactly the right-hand side of this boxed
identity.


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
:func:`~meepmeep.backends.numba.taylor.position2dd.pos_cd`,
:func:`~meepmeep.backends.numba.taylor.position3dd.pos_cd`,
and their direct counterparts ``pos_d``, ``pos_d`` (which epoch-fold
``t`` first; the discrete epoch shift contributes no derivative).


Projected distance
~~~~~~~~~~~~~~~~~~

For :math:`d = \sqrt{p_x^2 + p_y^2}`, differentiating
:math:`d^2` gives

.. math::

   \boxed{\;
     \frac{\partial d}{\partial \theta_k}
       \;=\; \frac{p_x\, \partial p_x/\partial \theta_k
                   \;+\; p_y\, \partial p_y/\partial \theta_k}{d}.
   \;}

The same reduction is applied in 2D and 3D
(:func:`~meepmeep.backends.numba.taylor.position2dd.sep_cd`,
:func:`~meepmeep.backends.numba.taylor.position3dd.sep_cd`); both
treat :math:`d` as the **projected** distance. The expression is
regular for :math:`d > 0` and ill-defined at exactly zero projected
separation; transit modelling stays well clear of this geometric
singularity.


Z-coordinate
~~~~~~~~~~~~

The line-of-sight coordinate :math:`z` is just the third row of the
position polynomial, so its gradient is the polynomial in
``dc[k, 2, :]``. See
:func:`~meepmeep.backends.numba.taylor.position3dd.pz_cd`.


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
:func:`~meepmeep.backends.numba.taylor.velocity3dd.zvel_cd`.


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

The :math:`(\phi_0, w)` partials of :math:`s` vanish. Implemented in
:func:`~meepmeep.backends.numba.taylor.velocity3dd.rv_cd` and
:func:`~meepmeep.backends.numba.taylor.velocity3dd.rv_d`.


Phase angle and 3D separation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The whole-orbit dispatchers in
:mod:`~meepmeep.backends.numba.taylor.orbit3dd` compose further chain
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
:func:`~meepmeep.backends.numba.taylor.orbit3dd.star_planet_distance_o5v_d`
and
:func:`~meepmeep.backends.numba.taylor.orbit3dd.cos_alpha_o5v_d`. The
Lambertian phase-curve dispatchers
(:func:`~meepmeep.backends.numba.taylor.orbit3dd.lambert_phase_curve_o5v_d`,
:func:`~meepmeep.backends.numba.taylor.orbit3dd.lambert_and_emission_o5v_d`)
compose the Lambert kernel
:math:`f(\cos\alpha) = (\sin\alpha + (\pi - \alpha)\cos\alpha)/\pi`
on top of :math:`\cos\alpha`, with its own closed-form derivative
:math:`\mathrm d f / \mathrm d \cos\alpha`.


Multi-knot propagation
----------------------

The orbit-spanning solver
:func:`~meepmeep.backends.numba.taylor.orbit3dd.solve3d_orbit_d`
applies
:func:`~meepmeep.backends.numba.taylor.solve3dd.solve3d_d` once per
knot and stacks the results into arrays of shape ``(N, 3, 5)`` for
``coeffs`` and ``(N, 6, 3, 5)`` for ``dcoeffs``. The periodic-image
contract on the last knot is honoured for both arrays — they share
their final slot with the first.

Each multi-knot ``_d`` dispatcher then performs the same single-knot
chain rule documented above after a ``pktable``-driven knot lookup:

#. Epoch-fold the absolute time into a single period.
#. Look up the knot index ``ix`` from the time-to-knot table.
#. Subtract the knot phase to get the centered time.
#. Call the matching centered single-knot ``_d`` evaluator with
   ``coeffs[ix]`` and ``dcoeffs[ix]``.

No additional algebra is needed at the dispatcher level for positions,
velocities, or distances; the chain rule for higher-level outputs
(phase angle, Lambert curve, RV, light travel time) is applied
identically to the single-knot case but using the dispatched value and
gradient.


Numerical regime and pitfalls
-----------------------------

* **Validity window.** Each knot's Taylor expansion is accurate within
  a region around the knot whose size depends on the orbit; near
  periastron of an eccentric orbit the window is narrowest. The
  *gradient* is accurate inside the same region — its truncation error
  has the same order as the value's.

* **Projected-distance singularity.** The chain rule for
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

* **Floating-point precision.** All ``_d`` routines run under
  :func:`numba.njit` with ``fastmath=True``. In practice this yields
  gradients agreeing with finite-difference checks to roughly
  :math:`10^{-9}` relative error for typical transit parameters — the
  same envelope the value-only evaluators inhabit.

* **Slot-0 convention.** In the single-knot transit workflow the
  expansion phase :math:`\phi_0` coincides with the time of transit
  center :math:`t_0` and slot 0 is read directly as
  :math:`\partial / \partial t_0`. In the multi-knot workflow each
  knot's expansion phase is fixed at construction time; slot 0 then
  describes the sensitivity to that shared phase offset, which the
  high-level fitting layer maps onto :math:`t_0` as appropriate.
