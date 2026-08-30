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

"""OpenCL backend: MeepMeep evaluators as OpenCL C device functions.

See :mod:`meepmeep.backends.opencl.source` for the calling conventions.
This module is import-safe on machines without pyopencl or an OpenCL
driver: it only reads packaged source files.
"""

from .source import SOURCE_FILES, read_kernel_source, read_full_source, build_options

__all__ = ['SOURCE_FILES', 'read_kernel_source', 'read_full_source', 'build_options']
