"""
Accelerated Proximal Gradient (APG)
"""

from math import sqrt

import numpy
from numpy.linalg import norm, svd

from tools import shrinking


def _check_converged(
    s_kp1_l, s_kp1_s, l_kp1, s_kp1, tol, converged, max_iter, k
) -> bool:
    stopping_criterion = norm([s_kp1_l, s_kp1_s]) / (2 * max(1, norm([l_kp1, s_kp1])))
    if (stopping_criterion <= tol) or (converged == 0 and k >= max_iter):
        return True
    return False


def rpca_apg(data_mat, lmbda: float, max_iter, tol):
    """
    Solve RPCA-PCP via APG

    Args:
        D: (m x n) data matrix
        lambda: weight of sparse error
        tol: tolerance for stopping criterion (DEFAULT=1e-6)
        max_iter: maximum number of iterations (DEFAULT=1000)

    Returns:
        s_hat (s_^) - estimate of S
    """
    _, sigma_i, __ = svd(data_mat, full_matrices=False)

    l_k = l_m1 = numpy.zeros(data_mat.shape)
    s_k = s_m1 = numpy.zeros(data_mat.shape)
    t_k = t_m1 = 1

    mu_k = sigma_i[1]
    mu_bar = 0.05 * sigma_i[3]
    eta = 0.99

    # Solving RPCA-PCP via APG
    converged = False
    k = 0
    while not converged:
        y_k_l = l_k + ((t_m1 - 1) / t_k) * (l_k - l_m1)
        y_k_s = s_k + ((t_m1 - 1) / t_k) * (s_k - s_m1)

        g_k_l = y_k_l - (1 / 2) * (y_k_l + y_k_s - data_mat)
        U, sigma, v = svd(g_k_l, full_matrices=False)
        sigma = numpy.diag(sigma)
        l_kp1 = numpy.dot(U, shrinking(sigma, mu_k / 2))
        l_kp1 = numpy.dot(l_kp1, v)

        g_k_s = y_k_s - (1 / 2) * (y_k_l + y_k_s - data_mat)
        g_k_s = numpy.squeeze(numpy.asarray(g_k_s))
        s_kp1 = shrinking(g_k_s, lmbda * mu_k / 2)

        t_kp1 = 0.5 * (1 + sqrt(1 + 4 * t_k * t_k))
        temp = l_kp1 + s_kp1 - y_k_l - y_k_s
        s_kp1_l = 2 * (y_k_l - l_kp1) + temp
        s_kp1_s = 2 * (y_k_s - s_kp1) + temp
        mu_k = max(mu_k * eta, mu_bar)

        k = k + 1
        t_m1 = t_k
        t_k = t_kp1
        l_m1 = l_k
        s_m1 = s_k
        l_k = l_kp1
        s_k = s_kp1

        converged = _check_converged(
            s_kp1_l, s_kp1_s, l_kp1, s_kp1, tol, converged, max_iter, k
        )

    return s_k
