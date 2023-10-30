"""
Principal Component Pursuit (PCP)

Based on:
    https://github.com/dfm/pcp, Daniel Foreman-Mackey, 2015, MIT license.
    https://github.com/dfm/pcp/blob/main/pcp.py
"""

import numpy

from accelerated_proximal_gradient import rpca_apg
from inexact_augmented_lagrange_multiplier import rpca_ialm
from tools import matrix_to_grayscale, sliding_window


def pcp_func(
    original_image,
    image_shape,
    method: str,
    max_iter: int = 500,
    tol: float = 1e-2,
    sw_step_size: int = 10,
    sw_patch_size: int = 50,
):
    """
    Principal Component Pursuit
    """
    m, n = image_shape
    window_size = sw_patch_size
    step_size = sw_step_size
    orig_img = sliding_window(
        img_input=original_image, window_size=window_size, step_size=step_size, m=m, n=n
    )

    orig_img = matrix_to_grayscale(mat=orig_img)

    lam = 1.0 / numpy.sqrt(numpy.max((m, n)))
    if method == "apg":
        s_o = rpca_apg(orig_img, lam, max_iter, tol)

    elif method == "ialm":
        s_o = rpca_ialm(orig_img, lam, max_iter, tol)

    target_patch = numpy.zeros((m, n, 100))
    s_ret = numpy.zeros((m, n))
    y = numpy.zeros((m, n))
    temp_patch = numpy.zeros((window_size, window_size))

    idx = 0
    # build target patch
    for i in range(0, m - window_size + 1, step_size):
        for j in range(0, n - window_size + 1, step_size):
            idx += 1
            temp_patch = temp_patch.ravel(order="F")
            temp_patch = s_o[:, [idx - 1]]
            y[i : i + window_size - 1, j : j + window_size - 1] = (
                y[i : i + window_size - 1, j : j + window_size - 1] + 1
            )
            temp_patch = numpy.reshape(
                temp_patch, (window_size, window_size), order="F"
            )

            for u in range(i, i + window_size - 1):
                for v in range(j, j + window_size - 1):
                    target_patch[u, v, int(y[u, v])] = temp_patch[u - i + 1, v - j + 1]

    for i in range(0, m):
        for j in range(0, n):
            if int(y[i, j]) > 0:
                s_ret[i, j] = numpy.median(target_patch[i, j, 0 : int(y[i, j])])

    return s_ret
