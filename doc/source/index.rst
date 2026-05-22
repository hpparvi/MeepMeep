MeepMeep documentation
======================

MeepMeep is an extremely fast Keplerian orbit evaluator for exoplanet
light-curve and radial-velocity modelling. It calculates sky-projected
planet-star separations, phase curves, RV signals, and other quantities
useful in exoplanet research — an order of magnitude faster than
standard Newton-Raphson approaches.

MeepMeep's speed comes from a Taylor-series shortcut presented in
`Parviainen and Korth (2020) <https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.3356P/abstract>`_.
Kepler's equation is solved exactly only at a small
set of knot points along the orbit, and every orbit evaluation
in between is just a short polynomial in time.

There are two ways to use the package. The
:class:`~meepmeep.orbit.Orbit` class is the convenience entry point:
instantiate it once with the time array of your observations, then
update the orbital parameters inside the fitting loop and read out
whichever observable you need — positions, velocities, the
sky-projected separation, the phase angle, the radial velocity,
reflected-light and thermal phase curves, ellipsoidal variation, or
the light-travel-time correction. The low-level functions cover the
same ground more directly: they are all numba-jitted, so they drop
straight into a custom transit or RV model with minimal overhead. Use
the class when you want the batteries-included workflow; use the
low-level functions when you are building a custom transit or RV model
and want the orbit math to inline into your hot loop.

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
