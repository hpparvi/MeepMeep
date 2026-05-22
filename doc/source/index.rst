MeepMeep documentation
======================

MeepMeep is a fast Keplerian orbit evaluator for exoplanet
light-curve and radial-velocity modelling. It returns the quantities
you actually fit against: the sky-projected separation of the star
and planet centres (in units of the stellar radius), the
line-of-sight velocity and the stellar radial-velocity signal,
reflected-light and thermal phase curves, ellipsoidal variation, and
the light-travel-time correction.

The speed comes from a Taylor-series shortcut. Kepler's equation is
solved exactly only at a small set of knot points along the orbit.
Every evaluation in the fitting loop is then a short polynomial in
time — no transcendental solve per sample. The inner loop of a
typical light-curve or RV fit runs roughly an order of magnitude
faster than it would with a per-point Newton-Raphson solver.

There are two equally first-class ways to use the package. The
:class:`~meepmeep.orbit.Orbit` class is the convenience entry point:
instantiate it once with the time array of your observations, then
inside the fitting loop update the orbital parameters and read out
whichever observable you need — positions, velocities, the
sky-projected separation, the phase angle, the radial velocity,
reflected-light and thermal phase curves, ellipsoidal variation, or
the light-travel-time correction. The low-level single-knot Taylor
functions in :mod:`meepmeep.backends.numba.taylor` cover the same
ground at a lower level: each is a small ``@njit`` function that
takes a coefficient matrix and a time and returns the requested
quantity, so it composes directly inside your own ``@njit``
likelihood or transit model with no Python-boxing overhead. Use the
class when you want the batteries-included workflow; use the
low-level functions when you are building a custom evaluator and want
the orbit math to inline into your hot loop.

Derivative mode adds analytic gradients of every quantity w.r.t. the
six orbital parameters (and any method-specific extras), making the
package a natural fit for gradient-based optimisers and HMC samplers.

Both the high-level class and the low-level functions are backed by a
numba implementation and a JAX implementation. The numba backend is
currently complete; the JAX backend is partially implemented and
slated for future development.

The pages below introduce the :class:`~meepmeep.orbit.Orbit` class
first, then map the low-level Taylor-series backend it uses under the
hood.

.. toctree::
   :maxdepth: 2
   :caption: High-level Orbit class

   orbit_overview
   api/high_level

.. toctree::
   :maxdepth: 2
   :caption: Taylor-series backend

   taylor_overview
   naming_conventions
   derivatives
   api/low_level
