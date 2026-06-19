MeepMeep documentation
======================

MeepMeep is an extremely fast Keplerian orbit evaluator for exoplanet
light-curve and radial-velocity modelling. It computes sky-projected
planet-star separations, phase curves, RV signals, and other quantities
used in exoplanet research up to two orders of magnitude faster than
per-point Newton-Raphson.

For every supported quantity it can also return the partial derivatives
with respect to the orbital parameters (and any other inputs), which feed
directly into gradient-based optimisers and MCMC samplers. The gradients are
computed analytically, without JAX or other automatic-differentiation tools;
the code is pure Numba-jitted Python (a JAX backend is underway).

MeepMeep's speed comes from the Taylor-series approach presented in
`Parviainen and Korth (2020) <https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.3356P/abstract>`_.
Kepler's equation is solved exactly at a single point in time (for transit or
eclipse modelling) or a small set of points along the orbit (for phase curves, RVs, and
similar). The planet's position is expanded into a Taylor series around these expansion
points, so every subsequent quantity evaluation is based on the planet's orbital position
expressed as a short polynomial in time.

Installation
------------

.. code-block:: bash

   pip install meepmeep

For a development checkout, clone the repository and install in editable
mode:

.. code-block:: bash

   git clone https://github.com/hpparvi/meepmeep
   cd meepmeep
   pip install -e .

MeepMeep runs on Python 3 and depends only on NumPy, Numba, SciPy, and
Matplotlib, all pulled in automatically.


Quickstart
----------

.. code-block:: python

   import numpy as np
   from meepmeep import Orbit

   o = Orbit(npt=15)
   o.set_pars(tc=0.0, p=3.0, a=8.5, i=np.radians(89.0), e=0.1, w=np.radians(90.0))
   o.set_data(np.linspace(-0.05, 0.05, 1001))

   x, y, z = o.xyz()                    # planet position per time
   d = o.star_planet_distance()         # 3D star-planet separation

See :ref:`orbit_overview` for the full tour and
:ref:`taylor_overview` for the low-level backend.


Two ways to use MeepMeep
------------------------

MeepMeep offers two ways in: a low-level API and a convenience high-level
API. The high-level classes —
:class:`~meepmeep.expansion2d.Expansion2D`,
:class:`~meepmeep.expansion3d.Expansion3D`, and
:class:`~meepmeep.orbit.Orbit` — wrap the orbit math behind a stateful
object: instantiate one with your observation times, then update the orbital
parameters inside a fitting loop and read out whichever observable you need.
The low-level functions cover the same ground more directly — they are all
numba-jitted and drop straight into a custom transit or RV model with minimal
overhead. Use a class for the batteries-included workflow; use the low-level
functions when you want the orbit math to inline into your own hot loop.

Both paths expose the same derivative mode: analytic gradients of every
quantity with respect to the seven orbital parameters (plus any
method-specific extras).

There are three high-level classes. :class:`~meepmeep.expansion2d.Expansion2D`
is the lightweight one for transit or eclipse geometry: a single Taylor expansion in
the sky plane, giving planet position, projected separation, and
contact-point / duration utilities. :class:`~meepmeep.expansion3d.Expansion3D`
keeps the full 3D motion of that single expansion, adding the line-of-sight
coordinate, velocity, radial velocity, phase angle, and phase-curve
observables within the event window. :class:`~meepmeep.orbit.Orbit` spans
the whole orbit in 3D and adds those same dynamical and photometric
quantities at *any* orbital phase, plus light travel time. The pages below
introduce ``Expansion2D`` first, then ``Expansion3D``, then ``Orbit``, then
map the low-level Taylor-series backend they all use under the hood.

.. toctree::
   :maxdepth: 2
   :caption: High-level Expansion2D class

   expansion2d_overview
   api/high_level_expansion2d

.. toctree::
   :maxdepth: 2
   :caption: High-level Expansion3D class

   expansion3d_overview
   api/high_level_expansion3d

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

.. toctree::
   :maxdepth: 1
   :caption: Project

   citing
