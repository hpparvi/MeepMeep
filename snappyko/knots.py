from numba import njit
from numpy import pi, linspace, zeros
from scipy.optimize import root_scalar

from .newton import ea_newton_s, ea_newton_v, ta_newton_s, ta_newton_v, xyz_newton_v

@njit
def eccentric_anomaly(t, e):
    return ea_newton_s(t, 0.0, 1.0, e, 0.5*pi)

@njit
def true_anomaly(t, e):
    f = ta_newton_s(t, 0.0, 1.0, e, 0.5*pi)
    if f < 0.0:
        f += 2*pi
    return f


def create_knots(n_knots: int, e: float, quantity: str = 'ea', tres: int = 200):
    if quantity not in ('mm', 'ea', 'ta'):
        raise ValueError("Quantity needs to be either 'mm' for mean motion, 'ea' for eccentric anomaly, or 'ta' for true anomaly.")
    if n_knots % 2 != 1:
        raise ValueError("Number of knots should be odd.")

    if quantity == 'mm':
        knot_times = linspace(0, 1, n_knots++1, endpoint=False)
        dt = 1 / n_knots
        change_times = linspace(0.5 * dt, 1 - 0.5 * dt, n_knots)
    else:
        if quantity == 'ea':
            def cfun(t, e, v):
                return eccentric_anomaly(t, e) - v
        elif quantity == 'ta':
            def cfun(t, e, v):
                return true_anomaly(t, e) - v
        else:
            raise NotImplementedError

        knot_sep = 2 * pi / n_knots

        knot_times = zeros(n_knots)
        knot_times[n_knots // 2] = 0.5
        t0 = 1e-5
        for i in range(1, n_knots // 2):
            knot_times[i] = root_scalar(cfun, args=(e, i * knot_sep), bracket=(t0, 1.0 - 1e-5)).root
            t0 = knot_times[i]
        knot_times[n_knots // 2 + 1:-1] = 1 - knot_times[n_knots // 2 - 1:0:-1]
        knot_times[-1] = 1.0

        change_times = zeros(n_knots-1)
        t0 = 1e-5
        for i in range(0, n_knots // 2):
            change_times[i] = root_scalar(cfun, args=(e, (i + 0.5) * knot_sep), bracket=(t0, 1.0 - 1e-5)).root
            t0 = change_times[i]
        change_times[n_knots // 2:] = 1 - change_times[n_knots // 2 - 1::-1]

    # Create the time-to-knot table
    dt = 1 / tres
    tktable = zeros(tres, int)
    ik = 0
    for i in range(tres):
        if i*dt > change_times[ik]:
            ik += 1
        if ik >= n_knots-1:
            tktable[i:] = ik
            break
        tktable[i] = ik

    return knot_times, change_times, dt, tktable
