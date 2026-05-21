MeepMeep documentation
======================

MeepMeep is a Python package for fast Keplerian orbit calculations targeted at
exoplanet transit modelling. The high-level :class:`~meepmeep.orbit.Orbit`
class is the everyday entry point: bind a few orbital elements, get any of
the standard observables (position, velocity, projected separation, phase
angle, radial velocity, phase curves, ellipsoidal variation, light travel
time) at arbitrary times, optionally with analytic gradients for fitting.
Below the surface, the numba-compiled Taylor-series routines under
:mod:`meepmeep.backends.numba.taylor` expose the core evaluator primitives
for users who need to compose their own pipelines or differentiate
orbit-derived quantities directly.

The pages below introduce the :class:`~meepmeep.orbit.Orbit` class first,
then describe the conceptual model behind the low-level Taylor backend and
catalogue its functions.

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
