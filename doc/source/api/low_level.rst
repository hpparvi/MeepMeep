.. _api.lowlevel:

Low level functions
===================

Two-dimensional functions
-------------------------

The 2D functions in `meepmeep.xy` are used to calculate a Keplerian orbit projected to a (x, y) viewing planet. These
are mainly useful for modeling transiting planets.

Taylor series expansion around a single point
*********************************************

.. currentmodule:: meepmeep

The Taylor series expansion around a single point in phase.

.. autosummary::
    :toctree: api

    xy.position.solve_xy_p5s

Position
********

.. currentmodule:: meepmeep.xy.position

.. autosummary::
    :toctree: api.2d

    solve_xy_p5s
    xy_t15sc
    xy_t15s
    xy_t15v
    xyd_t15s
    pd_t15
    pd_t15sc
    pd_t25s
    pd_t25v

Three-dimensional functions
---------------------------

Taylor series expansion
***********************
.. currentmodule:: meepmeep

The Taylor series expansion around a single point in phase.

.. autosummary::
    :toctree: api

    xyz.position.solve_xyz_p5s

Position
********

.. currentmodule:: meepmeep.xyz.position

The 3d functions in `meepmeep.xyz` can be used to calculate (x, y, z) positions, phase angles, light travel time
delays, and more.

.. autosummary::
    :toctree: api.3d

    solve_xyz_p5s
    xyz_t15s

