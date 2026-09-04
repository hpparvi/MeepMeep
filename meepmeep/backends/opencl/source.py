#  MeepMeep: fast orbit calculations for exoplanet modelling
#  Copyright (C) 2022-2026 Hannu Parviainen
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""OpenCL device-function source for the MeepMeep evaluators.

This backend ships the numba backend's evaluation functionality, and the
Taylor coefficient solvers, as OpenCL C *device functions*: packages using
MeepMeep prepend the source returned by :func:`read_kernel_source` to their
own kernel code and call the functions from their kernels. Context, queue, and
program management are deliberately left to the caller.

Every file except ``solve_kernels.cl`` contains device functions only.
``solve_kernels.cl`` is opt-in and holds the batched ``__kernel`` entry points
for the solvers (one work item per orbital parameter set); request it by name
to get launchable solvers, or omit it and drive the ``solve2d``/``solve3d``
device functions from your own kernel.

Solving on the device pays off for *batches* of parameter sets -- population
samplers, where the coefficients then never leave the device -- and not for the
single parameter set per likelihood call that ``Orbit`` and ``Expansion2D``
issue: a solve kernel is launch-bound at these sizes, so solving one expansion
costs about what solving a thousand does. Expansion-point placement
(``create_expansion_points``, which needs scipy), the Newton reference solvers,
and the contact-point bisection in ``util.py`` remain host-side.

The device functions mirror the numba backend one-to-one (same names, same
argument order, values first and gradient output buffers last), with these
C-imposed deviations, documented per file:

- The single-expansion-point functions carry a trailing dimension digit
  (``pos_c2``/``pos_c3``, ``sep_cd2``/``sep_cd3``, ...): OpenCL C has a
  single flat namespace, so the digit replaces the numba backend's
  ``point2d``/``point3d`` package split. The multi-expansion-point ``_o``/
  ``_od`` evaluators and the dimension-agnostic helpers (``lambert_kernel``,
  ``rv_scale``, ``ep_ix``, ...) are unsuffixed.
- Only the scalar evaluators are ported; the numba vector/parallel kernels
  (``*_v``/``*_vp``/``*_ov*``) have no OpenCL counterpart because the kernel
  NDRange supplies the loop over samples.
- Optional arguments (``te``, ``lan``, ``timing_is_tc``) become mandatory.
- The solvers return their matrices through ``__global`` output pointers
  instead of returning them, and spell the inclination ``inc`` because ``i``
  is the loop index. They keep the numba names unsuffixed (``solve2d``,
  ``solve3d``): unlike the evaluators, those names already carry the
  dimension.
- The Kepler solver's convergence tolerance is precision-aware
  (``MM_EA_TOL``): numba's literal ``1e-13`` is unreachable in float32, so a
  verbatim port would run every fp32 work item to the 50-iteration cap.

Conventions the calling kernel must follow:

- Gradients use the seven-parameter order ``(tc, p, a, i, e, w, lan)``;
  functions with extra physical inputs append their derivatives after the
  orbital block in argument order.
- Coefficient arrays are the C-contiguous flattenings of the host solver
  outputs (``ascontiguousarray(c).ravel()``): ``c`` (D, 5), ``dc`` (7, D, 5),
  ``coeffs`` (npt, 3, 5), ``dcoeffs`` (npt, 7, 3, 5).
- ``ep_table`` must be uploaded as int32 (``ep_table.astype(np.int32)``);
  the numba backend builds it as int64.
- In single precision, absolute times (and ``tc``/``tpa``) must be shifted by
  a float64 reference epoch host-side before casting: a float32 ulp at
  BJD ~2.4e6 is ~0.25 days.
"""

from importlib.resources import files

__all__ = ['SOURCE_FILES', 'read_kernel_source', 'read_full_source', 'build_options']

SOURCE_FILES: tuple[str, ...] = (
    'common.cl',
    'solve2d.cl',
    'solve3d.cl',
    'point2d.cl',
    'point2dd.cl',
    'point3d.cl',
    'point3dd.cl',
    'orbit3d.cl',
    'orbit3dd.cl',
    'solve_kernels.cl',
)
"""The shipped ``.cl`` files in a valid concatenation (dependency) order."""

_DEPENDS: dict[str, tuple[str, ...]] = {
    'common.cl': (),
    'solve2d.cl': ('common.cl',),
    'solve3d.cl': ('common.cl',),
    'solve_kernels.cl': ('common.cl', 'solve2d.cl', 'solve3d.cl'),
    'point2d.cl': ('common.cl',),
    'point2dd.cl': ('common.cl', 'point2d.cl'),
    'point3d.cl': ('common.cl',),
    'point3dd.cl': ('common.cl', 'point3d.cl'),
    'orbit3d.cl': ('common.cl', 'point3d.cl'),
    'orbit3dd.cl': ('common.cl', 'point3d.cl', 'point3dd.cl', 'orbit3d.cl'),
}


def read_kernel_source(*names: str) -> str:
    """Concatenate the named ``.cl`` files with their dependencies.

    The requested files are expanded transitively with the files they
    require (every file needs ``common.cl``; the gradient and orbit files
    build on the single-expansion-point files), deduplicated, and
    concatenated in dependency order, so the result always compiles when
    prepended to kernel code. ``read_kernel_source('point2d.cl')`` returns
    ``common.cl`` + ``point2d.cl``.

    Parameters
    ----------
    names : str
        Names from :data:`SOURCE_FILES`.

    Returns
    -------
    str
        The concatenated OpenCL C source.
    """
    unknown = [name for name in names if name not in _DEPENDS]
    if unknown:
        raise ValueError(f"Unknown source file(s) {unknown}; expected names from {SOURCE_FILES}.")

    selected: set[str] = set()

    def _add(name: str) -> None:
        if name not in selected:
            for dep in _DEPENDS[name]:
                _add(dep)
            selected.add(name)

    for name in names:
        _add(name)

    package = files('meepmeep.backends.opencl')
    return '\n'.join(package.joinpath(name).read_text() for name in SOURCE_FILES if name in selected)


def read_full_source() -> str:
    """Concatenate every shipped ``.cl`` file in dependency order."""
    return read_kernel_source(*SOURCE_FILES)


def build_options(precision: str = 'double') -> str:
    """Preprocessor options selecting the device floating-point type.

    Deliberately does not enable ``-cl-fast-relaxed-math``: the OpenCL
    functions are tested for ~1e-12 agreement with the numba backend, and
    relaxed math breaks that. It would also be less strict than the numba
    kernels that are compiled without ``fastmath`` on purpose
    (``true_anomaly``, ``rv``).

    Parameters
    ----------
    precision : str
        Either ``'double'`` (defines ``REAL=double`` and ``USE_FP64``, which
        enables the ``cl_khr_fp64`` extension) or ``'single'``.

    Returns
    -------
    str
        The option string to pass to ``pyopencl.Program.build``.
    """
    if precision == 'double':
        return '-DREAL=double -DUSE_FP64'
    elif precision == 'single':
        return '-DREAL=float'
    raise ValueError(f"Unknown precision '{precision}', expected 'double' or 'single'.")
