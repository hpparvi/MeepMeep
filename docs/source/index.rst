MeepMeep documentation
======================

MeepMeep is an extremely fast Keplerian orbit evaluator for exoplanet
light-curve and radial-velocity modelling. It calculates sky-projected
planet-star separations, phase curves, RV signals, and other quantities
useful in exoplanet research, up to three orders of magnitude faster than
standard Newton-Raphson approaches.

In addition, MeepMeep can return the partial derivatives w.r.t. the orbital
parameters (and other input parameters) for all the supported quantities.
These are useful when developing code used with gradient-aware optimisers or
MCMC samplers. This is all done without JAX or other autograd-approaches
(although a JAX implementation is underway). The code is pure Numba-jitted
Python.

MeepMeep's speed comes from a Taylor-series approach presented in
`Parviainen and Korth (2020) <https://ui.adsabs.harvard.edu/abs/2020MNRAS.499.3356P/abstract>`_.
Kepler's equation is solved exactly only at a single point in time (for transit or eclipse modelling),
or a small set of points along the orbit (for modelling phase curves, RVs, etc.). The planet's position
is expanded into a Taylor series in these expansion points, after which every
quantity evaluation is based on the planet's orbital position expressed as a short polynomial in time.

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

The :class:`~meepmeep.expansion2d.Expansion2D` and :class:`~meepmeep.orbit.Orbit` classes are convenience entry points:
instantiate them once with your observation times, then update the orbital
parameters inside a fitting loop and read out whichever observable you
need. The low-level functions cover the same ground more directly — they
are all numba-jitted and drop straight into a custom transit or RV model
with minimal overhead. Use the class for the batteries-included workflow;
use the low-level functions when you want the orbit math to inline into
your own hot loop.

Derivative mode adds analytic gradients of every quantity w.r.t. the
seven orbital parameters (and any method-specific extras), making the
package a natural fit for gradient-based optimisers and HMC samplers.

Both the high-level class and the low-level functions are backed by a
numba implementation and a JAX implementation. The numba backend is
currently complete; the JAX backend is partially implemented and
slated for future development.

There are two high-level classes. :class:`~meepmeep.expansion2d.Expansion2D`
is the lightweight one for transit or eclipse geometry: a single Taylor expansion in
the sky plane, giving planet position, projected separation, and
contact-point / duration utilities. :class:`~meepmeep.orbit.Orbit` spans
the whole orbit in 3D and adds velocity, radial velocity, phase curves,
and light travel time. The pages below introduce ``Expansion2D`` first,
then ``Orbit``, then map the low-level Taylor-series backend they both
use under the hood.

.. toctree::
   :maxdepth: 2
   :caption: High-level Expansion2D class

   expansion2d_overview
   api/high_level_expansion2d

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
