from typing import Optional

from matplotlib.patches import Circle
from matplotlib.pyplot import subplots, setp
from numpy import arccos, ndarray

from .xyz5 import solve_xyz_o5s, xyz_o5v, cos_alpha_o5v, light_travel_time_o5v


class Orbit:
    def __init__(self, npt: int = 11):
        self.npt: int = npt
        self.times: Optional[ndarray] = None

        self._dt: Optional[float] = None
        self._points: Optional[float] = None
        self._coeffs: Optional[ndarray] = None
        self._t0: Optional[float] = None
        self._p: Optional[float] = None

    def set_data(self, times):
        self.times = times

    def set_pars(self, t0, p, a, i, e, w):
        self._t0 = t0
        self._p = p
        self._dt, self._points, self._coeffs = solve_xyz_o5s(p, a, i, e, w, self.npt)

    @property
    def xyz(self):
        return xyz_o5v(self.times, self._t0, self._p, self._dt, self._points, self._coeffs)

    @property
    def cos_alpha(self):
        return cos_alpha_o5v(self.times, self._t0, self._p, self._dt, self._points, self._coeffs)

    @property
    def alpha(self):
        return arccos(cos_alpha_o5v(self.times, self._t0, self._p, self._dt, self._points, self._coeffs))

    def light_travel_time(self, rstar: float):
        return light_travel_time_o5v(self.times, self._t0, self._p, rstar, self._dt, self._points, self._coeffs)

    def plot(self, figsize=None):
        x, y, z = self.xyz
        xl, yl, zl = 1.1 * abs(x).max(), 1.1 * abs(y).max(), 1.1 * abs(z).max()
        al = max([xl, yl, zl])
        #TODO: Add truths using Newton's method

        fig, axs = subplots(1, 3, figsize=figsize)
        axs[0].plot(x, y)
        axs[0].plot(self._coeffs[0, 0], self._coeffs[0, 1], 'ok')
        axs[1].plot(x, z)
        axs[1].plot(self._coeffs[0, 0], self._coeffs[0, 2], 'ok')
        axs[2].plot(z, y)
        axs[2].plot(self._coeffs[0, 2], self._coeffs[0, 1], 'ok')
        [ax.add_patch(Circle((0, 0), 1, fc='y', ec='k')) for ax in axs]
        [ax.set_aspect(1) for ax in axs]
        setp(axs, xlim=(-al, al), ylim=(-al, al))
        setp(axs[1:], yticks=[])
        fig.tight_layout()