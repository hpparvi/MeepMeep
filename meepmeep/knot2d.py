from numpy import asarray, zeros, arctan2

from .utils import as_from_rhop, i_from_baew
from .xy.position import pd_t15, solve_xy_p5s, xy_t15
from .xy.derivatives import pd_with_derivatives_v, xy_derivative_coeffs
from .xy.par_direct import diffs as diffs_natural
from .xy.par_fitting import partial_derivatives as diffs_fitting

class Knot2D:

    def __init__(self, phase: float, t0: float, p: float, a: float, i: float, e: float, w: float,
                 derivatives: bool = False):
        self.derivatives = derivatives
        self.phase = phase
        self.t0 = t0
        self.p = p
        self.a = a
        self.i = i
        self.e = e
        self.w = w

        self._coeffs = solve_xy_p5s(phase, p, a, i, e, w)
        if derivatives:
            self._c_derivative_coeffs()

    def _c_derivative_coeffs(self):
        d = diffs_natural(self.p, self.a, self.i, self.e, self.w, 1e-4)
        self._coeffs_d = xy_derivative_coeffs(d, 1e-4, self._coeffs)

    def position(self, t):
        return xy_t15(t, self.t0, self.p, self._coeffs)

    def projected_distance(self, t):
        if self.derivatives:
            return pd_with_derivatives_v(t, self.t0, self.p, self._coeffs, self._coeffs_d)
        else:
            return pd_t15(t, self.t0, self.p, self._coeffs)

    def _pd_numerical_derivatives(self, t, e=1e-4):
        t = asarray(t)
        res = zeros((6, t.size))
        r0 = pd_t15(t, self.t0, self.p, solve_xy_p5s(self.phase, self.p, self.a, self.i, self.e, self.w))
        res[0] = pd_t15(t, self.t0, self.p, solve_xy_p5s(self.phase + e, self.p, self.a, self.i, self.e, self.w))
        res[1] = pd_t15(t, self.t0, self.p + e, solve_xy_p5s(self.phase, self.p + e, self.a, self.i, self.e, self.w))
        res[2] = pd_t15(t, self.t0, self.p, solve_xy_p5s(self.phase, self.p, self.a + e, self.i, self.e, self.w))
        res[3] = pd_t15(t, self.t0, self.p, solve_xy_p5s(self.phase, self.p, self.a, self.i + e, self.e, self.w))
        res[4] = pd_t15(t, self.t0, self.p, solve_xy_p5s(self.phase, self.p, self.a, self.i, self.e + e, self.w))
        res[5] = pd_t15(t, self.t0, self.p, solve_xy_p5s(self.phase, self.p, self.a, self.i, self.e, self.w + e))
        res = (res - r0) / e
        return res


class Knot2DFit(Knot2D):

    def __init__(self, phase: float, t0: float, p: float, rho: float, b: float, secw: float, sesw: float,
                 derivatives: bool = False):
        self.rho = rho
        self.b = b
        self.secw = secw
        self.sesw = sesw
        a = as_from_rhop(rho, p)
        e = secw ** 2 + sesw ** 2
        w = arctan2(sesw, secw)
        i = i_from_baew(b, a, e, w)
        super().__init__(phase, t0, p, a, i, e, w, derivatives)

    def _c_derivative_coeffs(self):
        d = diffs_fitting(self.p, self.rho, self.b, self.secw, self.sesw, 1e-4)
        self._coeffs_d = xy_derivative_coeffs(d, 1e-4, self._coeffs)