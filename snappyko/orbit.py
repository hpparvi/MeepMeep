from typing import Optional

from matplotlib.patches import Circle, Wedge
from matplotlib.pyplot import subplots, setp
from numpy import arccos, ndarray, mod, argmin, degrees, linspace, array, sqrt, sin, cos

from .newton import xyz_newton_v, ta_newton_v
from .utils import mean_anomaly_offset, TWO_PI, eccentricity_vector
from .xyz5 import solve_xyz_o5s, xyz_o5v, cos_alpha_o5v, light_travel_time_o5v, vxyz_o5v, true_anomaly_o5v, rv_o5v, \
    star_planet_distance_o5v, ev_signal_o5v


class Orbit:
    def __init__(self, npt: int = 15):
        self.npt: int = npt
        self.times: Optional[ndarray] = None

        self._dt: Optional[float] = None
        self._points: Optional[float] = None
        self._coeffs: Optional[ndarray] = None
        self._t0: Optional[float] = None
        self._p: Optional[float] = None
        self._a: Optional[float] = None
        self._i: Optional[float] = None
        self._e: Optional[float] = None
        self._w: Optional[float] = None

    def set_data(self, times):
        self.times = times

    def set_pars(self, t0, p, a, i, e, w):
        self._t0 = t0
        self._p = p
        self._a = a
        self._i = i
        self._e = e
        self._w = w
        self._dt, self._points, self._coeffs = solve_xyz_o5s(p, a, i, e, w, self.npt)

    def mean_anomaly(self):
        offset = mean_anomaly_offset(self._e, self._w)
        return mod(TWO_PI * (self.times - (self._t0 - offset * self._p / TWO_PI)) / self._p, TWO_PI)

    def true_anomaly(self):
        ev = eccentricity_vector(self._i, self._e, self._w)
        return true_anomaly_o5v(self.times, self._t0, self._p, ev[0], ev[1], ev[2], self._w, self._dt, self._points, self._coeffs)

    def xyz(self, times: Optional[ndarray] = None):
        times = times or self.times
        return xyz_o5v(times, self._t0, self._p, self._dt, self._points, self._coeffs)

    def _xyz_error(self):
        x, y, z = self.xyz()
        xt, yt, zt = xyz_newton_v(self.times, self._t0, self._p, self._a, self._i, self._e, self._w)
        return x-xt, y-yt, z-zt

    def vxyz(self):
        return vxyz_o5v(self.times, self._t0, self._p, self._dt, self._points, self._coeffs)

    def cos_phase(self):
        return cos_alpha_o5v(self.times, self._t0, self._p, self._dt, self._points, self._coeffs)

    def _cos_phase_error(self):
        ta = ta_newton_v(self.times, self._t0, self._p, self._e, self._w)
        cos_alpha_t = ta
        return cos_alpha_o5v(self.times, self._t0, self._p, self._dt, self._points, self._coeffs) - cos_alpha_t

    def phase(self):
        return arccos(cos_alpha_o5v(self.times, self._t0, self._p, self._dt, self._points, self._coeffs))

    def theta(self):
        return arccos(-cos_alpha_o5v(self.times, self._t0, self._p, self._dt, self._points, self._coeffs))

    def star_planet_distance(self, times: Optional[ndarray] = None):
        return star_planet_distance_o5v(times or self.times, self._t0, self._p, self._dt, self._points, self._coeffs)

    def light_travel_time(self, rstar: float):
        return light_travel_time_o5v(self.times, self._t0, self._p, rstar, self._dt, self._points, self._coeffs)

    def radial_velocity(self, k: float):
        return rv_o5v(self.times, k, self._t0, self._p, self._a, self._i, self._e, self._dt, self._points, self._coeffs)

    def ellipsoidal_variation(self, alpha: float, mass_ratio: float, times: Optional[ndarray] = None):
        """Ellipsoidal variation signal.

        NOTES: See Eqs. 6-10 in Lillo-Box al. (2014).
        """
        return ev_signal_o5v(alpha, mass_ratio, self._i, times or self.times, self._t0, self._p, self._dt, self._points, self._coeffs)

    def plot(self, figsize: Optional[tuple] = None, show_exact: bool = False, sr: float = 1.0, pr: float = 0.5, pc='k', npt: int = 1000):
        tcur = self.times
        self.set_data(linspace(0, self._p, npt))

        x, y, z = self.xyz()
        xl, yl, zl = 1.1*abs(x).max(), 1.1*abs(y).max(), 1.1*abs(z).max()
        al = max([xl, yl, zl])

        fig, axs = subplots(1, 3, figsize=figsize)
        axs[0].plot(x, y, zorder=0)
        axs[0].add_patch(Circle((self._coeffs[0, 0], self._coeffs[0, 1]), pr, fc=pc, ec='k', zorder=10))
        axs[1].plot(x, z, zorder=1)
        axs[1].add_patch(Circle((self._coeffs[0, 0], self._coeffs[0, 2]), pr, fc=pc, ec='k', zorder=11))
        axs[1].add_patch(Wedge((0, 0), 1.3*sr, 180 - degrees(self._w), 180, fc=pc, ec='k', zorder=-10))

        axs[2].plot(z, y, zorder=2)
        axs[2].add_patch(Circle((self._coeffs[0, 2], self._coeffs[0, 1]), pr, fc=pc, ec='k', zorder=12))

        di = self.times.size//6
        for i in range(6):
            axs[1].arrow(x[i*di], z[i*di], x[i*di + 1] - x[i*di], z[i*di + 1] - z[i*di], shape='full', lw=6,
                         length_includes_head=True, head_width=.1, color='k')

        m = x < 0.0
        axs[1].plot((0, x[m][argmin(abs(z[m]))]), (0, 0), 'k', zorder=-10, ls='--')
        omega_ix = argmin(x**2 + y**2 + z**2)
        axs[1].plot((0, x[omega_ix]), (0, z[omega_ix]), 'k', zorder=-10, ls='--')

        if show_exact:
            xt, yt, zt = xyz_newton_v(self.times, self._t0, self._p, self._a, self._i, self._e, self._w)
            axs[0].plot(xt, yt, 'k--')
            axs[1].plot(xt, zt, 'k--')
            axs[2].plot(zt, yt, 'k--')

        [ax.add_patch(Circle((0, 0), sr, fc='y', ec='k')) for ax in axs]
        [ax.set_aspect(1) for ax in axs]
        setp(axs, xlim=(-al, al), ylim=(-al, al))
        setp(axs[0], xlabel='X', ylabel='Y', title='Front')
        setp(axs[1], xlabel='X', ylabel='Z', ylim=(al, -al), title='Top')
        setp(axs[2], xlabel='Z', ylabel='Y', title='Side')
        fig.tight_layout()
        self.set_data(tcur)
