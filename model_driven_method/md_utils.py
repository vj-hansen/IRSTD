"""
Utilities used for model-driven method.
"""

import math
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pandas as pd
import PIL


def shrinking(mat_in, epsilon):
    """Soft-thresholding (shrinkage) operator: S_epsilon[x]"""
    sgn = np.sign(mat_in)  # sign returns -1 if x < 0, 0 if x==0, 1 if x > 0
    return np.multiply(sgn, np.maximum(np.abs(mat_in) - epsilon, 0))


def mat2gray(m):
    """
    Matrix to grayscale img
    """
    m = np.asmatrix(m)
    m_min = np.min(m)
    m_max = np.max(m)
    img = np.zeros(np.shape(m))
    divisor_mat = float(m_max - m_min) * (m - m_min)
    if np.max(divisor_mat) > 0:
        img = np.add(
            img,
            np.multiply(
                np.logical_and(np.greater_equal(m, m_min), np.less(m, m_max)),
                (1 / float(m_max - m_min) * (m - m_min)),
            ),
        )
    img = np.add(img, (np.greater_equal(m, m_max)))
    return img


def sliding_window(img_1, wndw_sz, step_sz, m, n):
    """
    Sliding window
    --------------
    Input:
        img_1: input image
        wndw_sz: size of sliding window
        step_sz: step size
        m, n: image shape
    --------------
    Return:
        image patch [2500 x ]
    """
    img = np.array(img_1)
    org_img = []  # empty array
    for i in range(0, m - wndw_sz + 1, step_sz):
        for j in range(0, n - wndw_sz + 1, step_sz):
            temp = img[i : i + wndw_sz, j : j + wndw_sz]
            # order='F' is Fortran-style
            org_img = np.append(org_img, [temp.flatten("F")])
    org_img = np.reshape(
        org_img, (wndw_sz * wndw_sz, org_img.size // (wndw_sz * wndw_sz)), order="F"
    )
    return org_img


def image_to_mat(o_img):
    """Image to matrix"""
    mat = []
    shape = None
    img = PIL.Image.open(o_img).convert("L")
    if shape is None:
        shape = img.size
        pix = list(img.getdata())
        # getdata(): returns the contents of this image as a sequence object
        # containing pixel values. The sequence object is flattened,
        # so that values for line 1 follow directly after the values of line 0
        mat.append(pix)
    return np.array(mat), shape[::-1]


def rgb2gray(rgb):
    """RGB image to grayscale image"""
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def read_xml(path, in_file):
    """Iterates through all .xml files in a given directory and combines
    them in a single Pandas dataframe.

    Parameters:
    ----------
    path : str
        The path containing the .xml files
    Returns:
        Numpy array
    """
    xml_list = []
    full_path = path + in_file + ".xml"
    tree = ET.parse(full_path)
    root = tree.getroot()
    for member in root.findall("object"):
        # the number of 'object' in the file dictates how many targets we have
        if len(member) == 7:  # some xml files contain extra info on "pixels"
            value = (
                root.find("filename").text,
                int(member[6][0].text),
                int(member[6][1].text),
                int(member[6][2].text),
                int(member[6][3].text),
            )
        elif len(member) == 5:  # 1 object
            value = (
                root.find("filename").text,
                int(member[4][0].text),
                int(member[4][1].text),
                int(member[4][2].text),
                int(member[4][3].text),
            )
        xml_list.append(value)
    column_name = ["filename", "xmin", "ymin", "xmax", "ymax"]
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    xml_np = xml_df.to_numpy()
    return xml_np


def pts_near(gt_bbx, pred_bbx, rad):  # change name
    """
    Determine if two points are within a radius.

    Parameters
    ----------
    gt_bbx : dict
            [centre_x, centre_y]
    pred_bbx : dict
            [centre_x, centre_y]
    Returns
    -------
    True or False
    """

    # create a box region where anything outside
    # the box is not within the radius (rad).
    if (
        abs(gt_bbx["centre_x"] - pred_bbx["centre_x"]) > rad
        or abs(gt_bbx["centre_y"] - pred_bbx["centre_y"]) > rad
    ):
        pt_cls = False

    rad_sqrd = math.pow(rad, 2)
    # return true if the points are close
    pt_cls = bool(
        rad_sqrd
        > (
            math.pow(gt_bbx["centre_x"] - pred_bbx["centre_x"], 2)
            + math.pow((gt_bbx["centre_y"] - pred_bbx["centre_y"]), 2)
        )
    )
    return pt_cls


def get_target_loc(img_file, thresh, delta):
    """
    Find location of pixels which have a different
    value than the black background (0 = black, 255 = white).

    Find (x, y)-position for pixels above a threshold.
    The target will have pixel-values above 0, the brightest targets
        have a value close to 255
    """
    x_p_a = []
    y_p_a = []
    r_x_p_a = []
    r_y_p_a = []
    radius = 5

    img = cv2.imread(img_file, 0)
    circ_img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    y_v, x_v = np.where(img > thresh)

    if len(x_v) == 0 or len(y_v) == 0:
        circ_img = circ_img_rgb
    for y_pos, x_pos in zip(y_v, x_v):
        x_p_a.append(x_pos)
        y_p_a.append(y_pos)

    for i, (x, y) in enumerate(zip(x_p_a, y_p_a)):
        if i == 0 or i == len(x_v) - 1:
            r_x_p_a.append(x_p_a[i])
            r_y_p_a.append(y_p_a[i])
            if i == 0:
                circ_img = cv2.circle(circ_img_rgb, (x, y), radius, (0, 255, 0), 2)
        else:
            diff_x = abs(x_p_a[i] - x_p_a[i - 1])  # e.g. x[1] - x[0]
            diff_y = abs(y_p_a[i] - y_p_a[i - 1])
            # the placement of [x(i), y(i)] and [x(i-1), y(i-1)] must be
            # different by at least delta pixels
            if (
                (diff_x > delta)
                and (diff_y > delta)
                and x_p_a[0] != x_p_a[len(x_v) - 1]
                and y_p_a[0] != y_p_a[len(x_v) - 1]
            ):
                r_x_p_a.append(x_p_a[i])
                r_y_p_a.append(y_p_a[i])
                circ_img = cv2.circle(circ_img_rgb, (x, y), radius, (0, 255, 0), 2)
            else:
                circ_img = circ_img_rgb
    if len(x_v) != 0 and len(y_v) != 0:
        if (
            abs(x_p_a[0] - x_p_a[len(x_v) - 1]) < delta
            and abs(y_p_a[0] - y_p_a[len(x_v) - 1]) < delta
        ):
            r_x_p_a.pop()
            r_y_p_a.pop()
    return circ_img, r_x_p_a, r_y_p_a
