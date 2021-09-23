"""
Accelerated Proximal Gradient (APG)
"""

from math import sqrt

import numpy as np
from numpy import linalg

from md_utils import shrinking


def rpca_apg(
        data_mat, 
        lmbda: int,
        max_iter: int, 
        tol: float) -> np.ndarray:

    """
    Args:
        data_mat (TYPE): (m x n) data matrix
        lmbda (int): weight of sparse error
        max_iter (int): maximum number of iterations (DEFAULT=1000)
        tol (float): tolerance for stopping criterion (DEFAULT=1e-6)
    
    Returns:
        np.ndarray: Description
    """
    U_i, sigm_i, v_i = linalg.svd(
                                data_mat,
                                full_matrices=False)

    l_k = l_m1 = np.zeros(data_mat.shape)
    s_k = s_m1 = np.zeros(data_mat.shape)
    t_k = t_m1 = 1

    mu_k = sigm_i[1]
    mu_bar = 0.05*sigm_i[3]
    eta = 0.99

    converged = k = 0
    while converged == 0:
        y_k_l = l_k+((t_m1-1)/t_k)*(l_k-l_m1)
        y_k_s = s_k+((t_m1-1)/t_k)*(s_k-s_m1)
        g_k_l = y_k_l-(1/2)*(y_k_l+y_k_s-data_mat)
        U, sigm, v = linalg.svd(g_k_l,
                                full_matrices=False)
        sigm = np.diag(sigm)
        l_kp1 = np.dot(U, shrinking(sigm, mu_k/2))
        l_kp1 = np.dot(l_kp1, v)
        g_k_s = y_k_s - (1/2)*(y_k_l + y_k_s - data_mat)
        g_k_s = np.squeeze(np.asarray(g_k_s))
        s_kp1 = shrinking(g_k_s, lmbda*mu_k/2)
        t_kp1 = 0.5*(1 + sqrt(1 + 4*t_k*t_k))
        temp = l_kp1+s_kp1-y_k_l-y_k_s
        s_kp1_l = 2*(y_k_l-l_kp1)+temp
        s_kp1_s = 2*(y_k_s-s_kp1)+temp
        mu_k = max(mu_k*eta, mu_bar)
        k = k+1
        t_m1 = t_k
        t_k = t_kp1
        l_m1 = l_k
        s_m1 = s_k
        l_k = l_kp1
        s_k = s_kp1
        stopping_criterion = linalg.norm([s_kp1_l, s_kp1_s]) \
            / (2*max(1, linalg.norm([l_kp1, s_kp1])))
        if (stopping_criterion <= tol) \
                or (converged == 0 and k >= max_iter):
            converged = 1
    s_hat = s_k
    return s_hat
