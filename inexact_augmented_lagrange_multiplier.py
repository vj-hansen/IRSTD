"""
Inexact augmented Lagrange multiplier (IALM)
"""

import numpy
from numpy import linalg

from tools import shrinking


def _jay_func(y_mat, lambd):
    """
    implements
        J(D) = max(norm_{2}(D), lambda^(-1)*norm_{inf}(D))
    """
    return max(
        linalg.norm(y_mat, 2),
        numpy.dot(numpy.reciprocal(lambd), linalg.norm(y_mat, numpy.inf)),
    )


def rpca_ialm(data_mat, lmbda, max_iter, tol):
    """
    Required input:
        D       - (m x n) data matrix
        lambda  - weight of sparse error

    Adjustable parameters:
        tol         - tolerance for stopping criterion (DEFAULT=1e-2)
        max_iter    - maximum number of iterations (DEFAULT=1000)

    Return:
        s_hat - estimate of S
    """

    d_norm = linalg.norm(data_mat)
    l_k = numpy.zeros(data_mat.shape)
    s_k = numpy.zeros(data_mat.shape)
    y_k = data_mat / _jay_func(data_mat, lmbda)
    mu_k = 1.25 / linalg.norm(data_mat, 2)
    mu_bar = mu_k * 1e7
    RHO = 1.6

    # Solving RPCA-PCP via IALM
    converged = k = 0
    while converged == 0:
        U, sigm, v = linalg.svd(
            data_mat - s_k + numpy.reciprocal(mu_k) * y_k, full_matrices=False
        )  # economy SVD
        sigm = numpy.diag(sigm)
        l_kp1 = numpy.dot(U, shrinking(sigm, numpy.reciprocal(mu_k)))
        l_kp1 = numpy.dot(l_kp1, v)
        shr = data_mat - l_kp1 + numpy.dot(numpy.reciprocal(mu_k), y_k)
        s_kp1 = shrinking(shr, lmbda * numpy.reciprocal(mu_k))
        mu_k = min(mu_k * RHO, mu_bar)
        k = k + 1
        l_k = l_kp1
        s_k = s_kp1

        stop_criterion = linalg.norm(data_mat - l_k - s_k, "fro") / d_norm
        if (converged == 0 and k >= max_iter) or stop_criterion < tol:
            converged = 1
    s_hat = s_k
    return s_hat
