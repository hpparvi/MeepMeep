MeepMeep documentation
======================

MeepMeep is a Python package for fast Keplerian orbit calculations targeted at
exoplanet transit modelling. The high-level :class:`~meepmeep.orbit.Orbit`
interface is meant for everyday use, while the low-level numba-compiled
Taylor-series routines under :mod:`meepmeep.backends.numba.taylor` expose the
core evaluator primitives for users who need to compose their own pipelines
or differentiate orbit-derived quantities.

The pages below describe the conceptual model behind the low-level layer and
catalogue its functions.

.. toctree::
   :maxdepth: 2
   :caption: Taylor-series backend

   taylor_overview
   naming_conventions
   derivatives
   api/low_level
