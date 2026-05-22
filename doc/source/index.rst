MeepMeep documentation
======================

MeepMeep is a Python package for fast Keplerian orbit calculations. It
is built for the case that dominates exoplanet light-curve and
radial-velocity work: the same orbit is evaluated thousands of times
under different orbital parameters during a modelling or fitting run,
and that inner loop has to be fast.

The :class:`~meepmeep.orbit.Orbit` class is the everyday entry point.
You instantiate it once with the time array of your observations.
Inside your modelling or fitting loop you then only update the
orbital parameters and read out whichever quantities you need —
positions, velocities, the sky-projected planet-star separation, the
phase angle, the radial velocity, reflected-light and thermal phase
curves, ellipsoidal variation, or the light-travel-time correction.
All the per-orbit setup is done once; the per-iteration cost is one
small Taylor solve plus the cost of the requested observable.

Derivative mode adds analytic gradients of every quantity w.r.t. the
six orbital parameters (and any method-specific extras), making the
package a natural fit for gradient-based optimisers and HMC samplers.

MeepMeep will provide both a high-level interface (the
:class:`~meepmeep.orbit.Orbit` class and friends) and a low-level
interface (for users who need to compose their own evaluators or
differentiate orbit-derived quantities through their own chain rule),
backed by both a numba and a JAX implementation. The numba backend is
currently complete; a JAX backend is partially implemented and slated
for future development.

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
