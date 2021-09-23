"""
Inexact augmented Lagrange multiplier (IALM)
"""

import numpy as np
from numpy import linalg, ndarray

from md_utils import shrinking


def jay_func(y_mat, lambd: int):

    """
    implements
        J(D) = max(norm_{2}(D), lambda^(-1)*norm_{inf}(D))
    
    Args:
        y_mat (TYPE): Description
        lambd (int): lambda
    
    Returns:
        TYPE: Description
    """
    return max(linalg.norm(y_mat, 2), np.dot(np.reciprocal(lambd),
                                             linalg.norm(y_mat, np.inf)))


def rpca_ialm(data_mat, lmbda: int, max_iter: int, tol: float) -> ndarray:

    """
    Args:
        data_mat (TYPE): (m x n) data matrix
        lmbda (int): weight of sparse error
        max_iter (int): maximum number of iterations (DEFAULT=1000)
        tol (float): tolerance for stopping criterion (DEFAULT=1e-2)
    
    Returns:
        ndarray: Description
    """


    d_norm = linalg.norm(data_mat)
    l_k = np.zeros(data_mat.shape)
    s_k = np.zeros(data_mat.shape)
    y_k = data_mat/jay_func(data_mat, lmbda)
    mu_k = 1.25/linalg.norm(data_mat, 2)
    mu_bar = mu_k*1e7
    rho = 1.6

    converged = k = 0
    while converged == 0:
        U, sigm, v = linalg.svd(data_mat-s_k+np.reciprocal(mu_k)*y_k,
                                full_matrices=False)  # economy SVD
        sigm = np.diag(sigm)
        l_kp1 = np.dot(U, shrinking(sigm, np.reciprocal(mu_k)))
        l_kp1 = np.dot(l_kp1, v)
        shr = data_mat - l_kp1 + np.dot(np.reciprocal(mu_k), y_k)
        s_kp1 = shrinking(shr, lmbda*np.reciprocal(mu_k))
        mu_k = min(mu_k*rho, mu_bar)
        k = k+1
        l_k = l_kp1
        s_k = s_kp1

        stop_criterion = linalg.norm(data_mat - l_k - s_k, 'fro')/d_norm
        if (converged == 0 and k >= max_iter) or stop_criterion < tol:
            converged = 1
    s_hat = s_k
    return s_hat
