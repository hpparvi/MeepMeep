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

from numba import njit
from numpy import zeros, sqrt, cos, sin, pi
from numpy.typing import NDArray

from ..newton.newton import ea_from_ma
from ..utils import mean_anomaly_at_transit_with_derivatives, TWO_PI


@njit(fastmath=True)
def solve3d_d(phase, p, a, i, e, w, lan: float = 0.0) -> tuple[NDArray, NDArray]:
    """Calculate Taylor expansion coefficients and their parameter derivatives for 3D position.

    Parameters
    ----------
    phase : float
        Phase angle (time) for the Taylor series expansion [days].
    p : float
        Orbital period [days].
    a : float
        Semi-major axis of the orbit [R_star].
    i : float
        Inclination of the orbit [rad].
    e : float
        Eccentricity of the orbit.
    w : float
        Argument of periastron [rad].
    lan : float, optional
        Longitude of the ascending node [rad]. A constant counterclockwise rotation
        of the sky-plane (x, y) coordinates about the line of sight; the line-of-sight
        (z) coordinate is unaffected. Defaults to 0.0.

    Returns
    -------
    cf : ndarray (3, 5)
        Position Taylor coefficients (identical to solve3d output).
    dcf : ndarray (7, 3, 5)
        Parameter derivative coefficients. dcf[k] = d(cf)/d(theta_k)
        for theta = (phase, p, a, i, e, w, lan). Row 6 is the derivative
        with respect to the longitude of the ascending node.
    """
    # Parameter indices: 0=phase, 1=p, 2=a, 3=i, 4=e, 5=w, 6=lan

    # ================================================================
    # Step 1: Constants and their derivatives
    # ================================================================
    n = TWO_PI / p
    dn = zeros(6)
    dn[1] = -TWO_PI / p**2

    mu = n**2 * a**3
    dmu = zeros(6)
    dmu[1] = 2.0 * n * dn[1] * a**3
    dmu[2] = 3.0 * n**2 * a**2

    sqe2 = sqrt(1.0 - e**2)
    dsqe2 = zeros(6)
    dsqe2[4] = -e / sqe2

    ci = cos(i)
    si = sin(i)
    dci = zeros(6)
    dci[3] = -si
    dsi = zeros(6)
    dsi[3] = ci

    cw = cos(w)
    sw = sin(w)
    dcw = zeros(6)
    dcw[5] = -sw
    dsw = zeros(6)
    dsw[5] = cw

    # ================================================================
    # Step 2: Mean anomaly offset and its derivatives
    # ================================================================
    offset, d_offset_de, d_offset_dw = mean_anomaly_at_transit_with_derivatives(e, w)
    doffset = zeros(6)
    doffset[4] = d_offset_de
    doffset[5] = d_offset_dw

    # ================================================================
    # Step 3: Mean anomaly and Kepler's equation
    # ================================================================
    ma = (TWO_PI * phase / p + offset) % TWO_PI

    dma = zeros(6)
    dma[0] = TWO_PI / p
    dma[1] = -TWO_PI * phase / p**2
    dma[4] = doffset[4]
    dma[5] = doffset[5]

    # Solve Kepler's equation: E - e*sin(E) = M
    ea = ea_from_ma(ma, e)
    sea = sin(ea)
    cea = cos(ea)

    # Implicit differentiation: dE/dq = (dma/dq + sin(E)*de/dq) / (1 - e*cos(E))
    inv_denom = 1.0 / (1.0 - e * cea)
    dea = zeros(6)
    for k in range(6):
        de_k = 1.0 if k == 4 else 0.0
        dea[k] = (dma[k] + sea * de_k) * inv_denom

    # Derivatives of sin(E), cos(E)
    dsea = zeros(6)
    dcea = zeros(6)
    for k in range(6):
        dsea[k] = cea * dea[k]
        dcea[k] = -sea * dea[k]

    # ================================================================
    # Step 4: Orbital plane position & velocity
    # ================================================================
    r_val = a * (1.0 - e * cea)
    dr = zeros(6)
    for k in range(6):
        da_k = 1.0 if k == 2 else 0.0
        de_k = 1.0 if k == 4 else 0.0
        dr[k] = da_k * (1.0 - e * cea) + a * (-de_k * cea - e * dcea[k])

    xi = a * (cea - e)
    dxi = zeros(6)
    for k in range(6):
        da_k = 1.0 if k == 2 else 0.0
        de_k = 1.0 if k == 4 else 0.0
        dxi[k] = da_k * (cea - e) + a * (dcea[k] - de_k)

    eta = a * sqe2 * sea
    deta = zeros(6)
    for k in range(6):
        da_k = 1.0 if k == 2 else 0.0
        deta[k] = da_k * sqe2 * sea + a * dsqe2[k] * sea + a * sqe2 * dsea[k]

    # E_dot = n * a / r
    ea_dot = n * a / r_val
    dea_dot = zeros(6)
    for k in range(6):
        da_k = 1.0 if k == 2 else 0.0
        dea_dot[k] = (dn[k] * a + n * da_k) / r_val - n * a * dr[k] / r_val**2

    # v_xi = -a * sin(E) * E_dot
    v_xi = -a * sea * ea_dot
    dv_xi = zeros(6)
    for k in range(6):
        da_k = 1.0 if k == 2 else 0.0
        dv_xi[k] = -(da_k * sea * ea_dot + a * dsea[k] * ea_dot + a * sea * dea_dot[k])

    # v_eta = a * sqe2 * cos(E) * E_dot
    v_eta = a * sqe2 * cea * ea_dot
    dv_eta = zeros(6)
    for k in range(6):
        da_k = 1.0 if k == 2 else 0.0
        dv_eta[k] = (da_k * sqe2 * cea * ea_dot
                     + a * dsqe2[k] * cea * ea_dot
                     + a * sqe2 * dcea[k] * ea_dot
                     + a * sqe2 * cea * dea_dot[k])

    # ================================================================
    # Step 5: Higher-order derivatives
    # ================================================================
    r2 = r_val**2
    v2 = v_xi**2 + v_eta**2
    rv = xi * v_xi + eta * v_eta

    dr2 = zeros(6)
    dv2 = zeros(6)
    drv = zeros(6)
    for k in range(6):
        dr2[k] = 2.0 * r_val * dr[k]
        dv2[k] = 2.0 * v_xi * dv_xi[k] + 2.0 * v_eta * dv_eta[k]
        drv[k] = dxi[k] * v_xi + xi * dv_xi[k] + deta[k] * v_eta + eta * dv_eta[k]

    inv_r3 = 1.0 / (r2 * r_val)
    inv_r5 = inv_r3 / r2
    inv_r7 = inv_r5 / r2

    dinv_r3 = zeros(6)
    dinv_r5 = zeros(6)
    dinv_r7 = zeros(6)
    for k in range(6):
        dinv_r3[k] = -3.0 * inv_r3 * dr[k] / r_val
        dinv_r5[k] = -5.0 * inv_r5 * dr[k] / r_val
        dinv_r7[k] = -7.0 * inv_r7 * dr[k] / r_val

    # u = -mu * inv_r3
    u = -mu * inv_r3
    du = zeros(6)
    for k in range(6):
        du[k] = -dmu[k] * inv_r3 - mu * dinv_r3[k]

    # u_dot = 3 * mu * rv * inv_r5
    u_dot = 3.0 * mu * rv * inv_r5
    du_dot = zeros(6)
    for k in range(6):
        du_dot[k] = 3.0 * (dmu[k] * rv * inv_r5
                           + mu * drv[k] * inv_r5
                           + mu * rv * dinv_r5[k])

    # u_ddot = 3*mu*(v2*inv_r5 - 5*rv^2*inv_r7) - 3*u^2
    rv2 = rv**2
    drv2 = zeros(6)
    for k in range(6):
        drv2[k] = 2.0 * rv * drv[k]

    u_ddot = 3.0 * mu * (v2 * inv_r5 - 5.0 * rv2 * inv_r7) - 3.0 * u**2
    du_ddot = zeros(6)
    for k in range(6):
        du_ddot[k] = (3.0 * (dmu[k] * (v2 * inv_r5 - 5.0 * rv2 * inv_r7)
                             + mu * (dv2[k] * inv_r5 + v2 * dinv_r5[k]
                                     - 5.0 * (drv2[k] * inv_r7 + rv2 * dinv_r7[k])))
                      - 6.0 * u * du[k])

    # Acceleration
    a_xi = u * xi
    a_eta = u * eta
    da_xi = zeros(6)
    da_eta = zeros(6)
    for k in range(6):
        da_xi[k] = du[k] * xi + u * dxi[k]
        da_eta[k] = du[k] * eta + u * deta[k]

    # Jerk
    j_xi = u_dot * xi + u * v_xi
    j_eta = u_dot * eta + u * v_eta
    dj_xi = zeros(6)
    dj_eta = zeros(6)
    for k in range(6):
        dj_xi[k] = du_dot[k] * xi + u_dot * dxi[k] + du[k] * v_xi + u * dv_xi[k]
        dj_eta[k] = du_dot[k] * eta + u_dot * deta[k] + du[k] * v_eta + u * dv_eta[k]

    # Snap
    s_coeff = u_ddot + u**2
    ds_coeff = zeros(6)
    for k in range(6):
        ds_coeff[k] = du_ddot[k] + 2.0 * u * du[k]

    s_xi = s_coeff * xi + 2.0 * u_dot * v_xi
    s_eta = s_coeff * eta + 2.0 * u_dot * v_eta
    ds_xi = zeros(6)
    ds_eta = zeros(6)
    for k in range(6):
        ds_xi[k] = ds_coeff[k] * xi + s_coeff * dxi[k] + 2.0 * (du_dot[k] * v_xi + u_dot * dv_xi[k])
        ds_eta[k] = ds_coeff[k] * eta + s_coeff * deta[k] + 2.0 * (du_dot[k] * v_eta + u_dot * dv_eta[k])

    # ================================================================
    # Step 6: Rotation matrix and its derivatives
    # ================================================================
    # X = -xi * cw + eta * sw          (toward observer)
    # Y = (-xi * sw - eta * cw) * ci   (sky plane)
    # Z = (xi * sw + eta * cw) * si    (above sky plane)

    m00 = -cw
    m01 = sw
    m10 = -sw * ci
    m11 = -cw * ci
    m20 = sw * si
    m21 = cw * si

    dm00 = zeros(6)
    dm01 = zeros(6)
    dm10 = zeros(6)
    dm11 = zeros(6)
    dm20 = zeros(6)
    dm21 = zeros(6)

    dm00[5] = sw
    dm01[5] = cw

    dm10[3] = sw * si
    dm10[5] = -cw * ci
    dm11[3] = cw * si
    dm11[5] = sw * ci

    dm20[3] = sw * ci
    dm20[5] = cw * si
    dm21[3] = cw * ci
    dm21[5] = -sw * si

    # ================================================================
    # Step 7: Assemble output
    # ================================================================
    cf = zeros((3, 5))
    dcf = zeros((7, 3, 5))

    # Orbital plane quantities grouped by Taylor order
    q_xi = (xi, v_xi, a_xi, j_xi, s_xi)
    q_eta = (eta, v_eta, a_eta, j_eta, s_eta)
    dq_xi = (dxi, dv_xi, da_xi, dj_xi, ds_xi)
    dq_eta = (deta, dv_eta, da_eta, dj_eta, ds_eta)
    scale = (1.0, 1.0, 0.5, 1.0/6.0, 1.0/24.0)

    # Row rotation elements: (m_row0, m_row1, dm_row0, dm_row1) for each row
    m_rows = ((m00, m01, dm00, dm01),
              (m10, m11, dm10, dm11),
              (m20, m21, dm20, dm21))

    for col in range(5):
        qx = q_xi[col]
        qe = q_eta[col]
        s = scale[col]

        for row in range(3):
            mr0, mr1, dmr0, dmr1 = m_rows[row]
            cf[row, col] = (mr0 * qx + mr1 * qe) * s
            for k in range(6):
                dqx = dq_xi[col][k]
                dqe = dq_eta[col][k]
                dcf[k, row, col] = (dmr0[k] * qx + mr0 * dqx + dmr1[k] * qe + mr1 * dqe) * s

    # ================================================================
    # Step 8: Longitude of the ascending node
    # ================================================================
    # `lan` is a constant rotation R(lan) of the sky-plane (x, y) about the line of
    # sight, independent of the other six parameters and leaving z unchanged. The
    # product rule collapses: cf and the existing six derivative rows are rotated by
    # R(lan) in (x, y), and the new lan-derivative row (index 6) is R'(lan) . cf_base
    # in (x, y) with a zero z-component.
    cO = cos(lan)
    sO = sin(lan)
    for col in range(5):
        x0 = cf[0, col]
        y0 = cf[1, col]

        # New lan-derivative row: R'(lan) . cf_base (uses the pre-rotation coords);
        # the z-row stays zero because z is independent of lan.
        dcf[6, 0, col] = -sO * x0 - cO * y0
        dcf[6, 1, col] = cO * x0 - sO * y0

        # Rotate the position (z unchanged)
        cf[0, col] = cO * x0 - sO * y0
        cf[1, col] = sO * x0 + cO * y0

        # Rotate the existing six derivative rows (z unchanged)
        for k in range(6):
            dx0 = dcf[k, 0, col]
            dy0 = dcf[k, 1, col]
            dcf[k, 0, col] = cO * dx0 - sO * dy0
            dcf[k, 1, col] = sO * dx0 + cO * dy0

    return cf, dcf
