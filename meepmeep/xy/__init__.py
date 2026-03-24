"""Compatibility shim for xy subpackage."""
from ..backends.numba.ts2d.position import pd_t15
from ..backends.numba.taylor.solve2d import solve_xy_p5
from ..backends.numba.ts2d.derivatives import pd_with_derivatives_v, pd_with_derivatives_s
from ..backends.numba.ts2d import position, derivatives

# Legacy alias for backward compatibility
solve_xy_p5s = solve_xy_p5

__all__ = ['solve_xy_p5s', 'pd_t15', 'pd_with_derivatives_v',
           'pd_with_derivatives_s', 'position', 'derivatives']