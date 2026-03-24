#  MeepMeep: fast orbit calculations for exoplanet modelling
#  Copyright (C) 2022 Hannu Parviainen
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

from numba import njit
from numpy import floor, sqrt, ndarray, pi

from ..utils import TWO_PI, HALF_PI

@njit(fastmath=True)
def p2d(tc, t0: float, p: float, c: ndarray):
    """Calculate planet's (x, y) position using Taylor series expansion.

    Automatically works with both scalar and array time inputs through broadcasting.

    Parameters
    ----------
    tc : float or ndarray
        The current time(s).
    t0 : float
        The Taylor series expansion time.
    p : float
        The orbital period.
    c : ndarray
        A 2x5 coefficient matrix where each element is a coefficient for Taylor series expansion.

    Returns
    -------
    tuple[float, float] or tuple[ndarray, ndarray]
        The (x, y) position(s). Returns scalars for scalar input, arrays for array input.
    """
    epoch = floor((tc - t0 + 0.5 * p) / p)
    t = tc - (t0 + epoch * p)
    px = c[0,0] + t*(c[0,1] + t*(c[0,2] + t*(c[0, 3] + t*c[0,4])))
    py = c[1,0] + t*(c[1,1] + t*(c[1,2] + t*(c[1, 3] + t*c[1,4])))
    return px, py


@njit(fastmath=True)
def p2dc(t: float, c: ndarray) -> tuple[float, float]:
    """Calculate planet's (x,y) position using Taylor series expansion for t centered on the expansion time.

    Parameters
    ----------
    t : float
        Time centered on the expansion time.
    c : ndarray
        A 2x5 coefficient matrix where each element is a coefficient for Taylor series expansion.

    Returns
    -------
    (float, float)
        The (x, y) position.
    """
    px = c[0,0] + t*(c[0,1] + t*(c[0,2] + t*(c[0, 3] + t*c[0,4])))
    py = c[1,0] + t*(c[1,1] + t*(c[1,2] + t*(c[1, 3] + t*c[1,4])))
    return px, py


@njit(fastmath=True)
def d2d(tc, t0, p, c):
    """Calculate the projected planet-star center (d)istance near transit."""
    px, py = p2d(tc, t0, p, c)
    return sqrt(px ** 2 + py ** 2)


@njit(fastmath=True)
def d2dc(tc, c):
    """Calculate the projected planet-star center (d)istance near transit."""
    px, py = p2dc(tc, c)
    return sqrt(px ** 2 + py ** 2)


@njit(fastmath=True)
def pd2d(t: float, c: ndarray) -> tuple[float, float, float]:
    """Calculate planet's (x,y) position and the projected distance for t centered on the expansion time.

    Parameters
    ----------
    t : float
        Time centered on the expansion time.
    c : ndarray
        A 2x5 coefficient matrix where each element is a coefficient for Taylor series expansion.

    Returns
    -------
    (float, float, float)
        The (x, y) position and the projected star-planet distance.
    """
    px = c[0,0] + t*(c[0,1] + t*(c[0,2] + t*(c[0, 3] + t*c[0,4])))
    py = c[1,0] + t*(c[1,1] + t*(c[1,2] + t*(c[1, 3] + t*c[1,4])))
    return px, py, sqrt(px**2 + py**2)

