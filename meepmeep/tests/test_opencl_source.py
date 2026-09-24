"""Tests for the OpenCL backend source reader and build options.

The reader tests need no OpenCL runtime; the build tests are guarded so a
machine without pyopencl or an OpenCL platform skips instead of erroring.
"""

import pytest

from meepmeep.backends.opencl import SOURCE_FILES, read_kernel_source, read_full_source, build_options


class TestReader:
    def test_single_file_pulls_dependencies(self):
        src = read_kernel_source('point2d.cl')
        assert 'MM_INLINE REAL taylor5' in src  # from common.cl
        assert 'MM_INLINE REAL sep_c2' in src

    def test_common_alone(self):
        src = read_kernel_source('common.cl')
        assert 'taylor5' in src
        assert 'sep_c2' not in src

    def test_deduplication(self):
        once = read_kernel_source('point2d.cl')
        twice = read_kernel_source('point2d.cl', 'common.cl', 'point2d.cl')
        assert once == twice
        assert twice.count('MM_INLINE REAL taylor5(') == 1

    def test_concatenation_order(self):
        src = read_full_source()
        # common.cl must precede every user of taylor5.
        assert src.index('MM_INLINE REAL taylor5(') < src.index('MM_INLINE void pos_c2(')

    def test_full_source_covers_all_files(self):
        src = read_full_source()
        assert src == read_kernel_source(*SOURCE_FILES)

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match='nonexistent.cl'):
            read_kernel_source('nonexistent.cl')


class TestBuildOptions:
    def test_double(self):
        opts = build_options('double')
        assert '-DREAL=double' in opts
        assert '-DUSE_FP64' in opts
        assert 'fast-relaxed-math' not in opts

    def test_single(self):
        opts = build_options('single')
        assert '-DREAL=float' in opts
        assert 'USE_FP64' not in opts

    def test_default_is_double(self):
        assert build_options() == build_options('double')

    def test_unknown_precision_raises(self):
        with pytest.raises(ValueError):
            build_options('half')


pyopencl = pytest.importorskip("pyopencl")
try:
    _HAS_OPENCL_DEVICE = bool(pyopencl.get_platforms())
except Exception:
    _HAS_OPENCL_DEVICE = False


@pytest.mark.skipif(not _HAS_OPENCL_DEVICE, reason="No OpenCL platform available")
class TestBuild:
    """The full concatenated source must compile without kernels appended."""

    def _build(self, precision):
        from meepmeep.tests.opencl_utils import build, has_fp64
        if precision == 'double' and not has_fp64():
            pytest.skip("Device lacks cl_khr_fp64")
        # An empty dummy kernel: some drivers refuse to build a program
        # with no __kernel entry point.
        build('__kernel void _dummy(__global REAL *x) { x[0] = (REAL)0.0; }',
              *SOURCE_FILES, precision=precision)

    def test_builds_fp64(self):
        self._build('double')

    def test_builds_fp32(self):
        self._build('single')
