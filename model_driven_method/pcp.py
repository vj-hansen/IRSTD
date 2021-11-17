
"""
Principal Component Pursuit (PCP)

Based on:
    https://github.com/dfm/pcp, Daniel Foreman-Mackey, 2015, MIT license.
    https://github.com/dfm/pcp/blob/main/pcp.py
"""

import numpy as np

from apg import rpca_apg
from ialm import rpca_ialm
from md_utils import mat2gray, sliding_window


def pcp_func(
    o_image, im_shape, max_iter=500,
    tol=1e-2, method='ialm',
        sw_step_size=10, sw_ptch_sz=50):
    '''
        Principal Component Pursuit
    '''
    m, n = im_shape
    wndw_sz = sw_ptch_sz
    step_sz = sw_step_size
    orig_img = sliding_window(
        o_image, wndw_sz,
        step_sz, m, n)

    orig_img = mat2gray(orig_img)

    lam = 1.0 / np.sqrt(np.max((m, n)))
    if method == 'apg':
        s_o = rpca_apg(orig_img, lam, max_iter, tol)

    elif method == 'ialm':
        s_o = rpca_ialm(orig_img, lam, max_iter, tol)

    trgt_patch = np.zeros((m, n, 100))
    s_ret = np.zeros((m, n))
    y = np.zeros((m, n))
    temp1 = np.zeros((wndw_sz, wndw_sz))

    idx = 0
    # build target patch
    for i in range(0, m - wndw_sz+1, step_sz):
        for j in range(0, n - wndw_sz+1, step_sz):
            idx += 1
            temp1 = temp1.ravel(order='F')
            temp1 = s_o[:, [idx-1]]
            y[i:i + wndw_sz-1, j:j + wndw_sz -
                1] = y[i:i + wndw_sz-1, j:j + wndw_sz-1]+1
            temp1 = np.reshape(temp1, (wndw_sz, wndw_sz), order='F')
            for u in range(i, i + wndw_sz-1):
                for v in range(j, j + wndw_sz-1):
                    trgt_patch[u, v, int(y[u, v])] = temp1[u - i+1, v - j+1]
    # median from IPI paper
    for i in range(0, m):
        for j in range(0, n):
            if int(y[i, j]) > 0:
                s_ret[i, j] = np.median(trgt_patch[i, j, 0:int(y[i, j])])
                # s_ret[i, j] = np.percentile(x, 10) # 10th percentile: alternative to median
    return s_ret
