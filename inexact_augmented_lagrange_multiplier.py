"""
Inexact augmented Lagrange multiplier (IALM)
"""

import numpy
from numpy.linalg import norm, svd

from tools import shrinking


def _jay_func(y_mat, lambd: float):
    """
    Implements J(D) = max(norm_{2}(D), lambda^(-1)*norm_{inf}(D))

    Args:
        TBA

    Returns:
        TBA
    """
    return max(
        norm(y_mat, 2),
        numpy.dot(numpy.reciprocal(lambd), norm(y_mat, numpy.inf)),
    )


def rpca_ialm(data_mat, lmbda: float, max_iter, tol):
    """
    Solving RPCA-PCP via IALM

    Args:
        D: (m x n) data matrix
        lambda: weight of sparse error
        tol: tolerance for stopping criterion (DEFAULT=1e-2)
        max_iter: maximum number of iterations (DEFAULT=1000)

    Returns:
        s_k (s_^) - estimate of S
    """

    d_norm = norm(data_mat)
    l_k = numpy.zeros(data_mat.shape)
    s_k = numpy.zeros(data_mat.shape)
    y_k = data_mat / _jay_func(data_mat, lmbda)
    mu_k = 1.25 / norm(data_mat, 2)
    mu_bar = mu_k * 1e7
    rho = 1.6

    converged = k = 0
    while converged == 0:
        U, sigma, v = svd(
            data_mat - s_k + numpy.reciprocal(mu_k) * y_k, full_matrices=False
        )  # economy SVD

        sigma = numpy.diag(sigma)
        l_kp1 = numpy.dot(U, shrinking(sigma, numpy.reciprocal(mu_k)))
        l_kp1 = numpy.dot(l_kp1, v)

        shr = data_mat - l_kp1 + numpy.dot(numpy.reciprocal(mu_k), y_k)
        s_kp1 = shrinking(shr, lmbda * numpy.reciprocal(mu_k))
        mu_k = min(mu_k * rho, mu_bar)

        k = k + 1
        l_k = l_kp1
        s_k = s_kp1

        stop_criterion = norm(data_mat - l_k - s_k, "fro") / d_norm
        if (converged == 0 and k >= max_iter) or stop_criterion < tol:
            converged = 1

    return s_k
