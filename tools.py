"""
Utilities used for model-driven method.
"""

import math
import xml.etree.ElementTree

import cv2
import numpy
import pandas


def shrinking(mat_in, epsilon):
    """Soft-thresholding (shrinkage) operator: S_epsilon[x]"""
    sgn = numpy.sign(mat_in)  # sign returns -1 if x < 0, 0 if x==0, 1 if x > 0
    return numpy.multiply(sgn, numpy.maximum(numpy.abs(mat_in) - epsilon, 0))


def matrix_to_grayscale(mat):
    """
    Matrix to grayscale image
    """
    mat = numpy.asmatrix(mat)
    m_min = numpy.min(mat)
    m_max = numpy.max(mat)
    img = numpy.zeros(numpy.shape(mat))
    divisor_mat = float(m_max - m_min) * (mat - m_min)
    if numpy.max(divisor_mat) > 0:
        img = numpy.add(
            img,
            numpy.multiply(
                numpy.logical_and(
                    numpy.greater_equal(mat, m_min), numpy.less(mat, m_max)
                ),
                (1 / float(m_max - m_min) * (mat - m_min)),
            ),
        )
    img = numpy.add(img, (numpy.greater_equal(mat, m_max)))
    return img


def sliding_window(img_input, wndw_sz, step_sz, m, n):
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
    img = numpy.array(img_input)
    org_img = []
    for i in range(0, m - wndw_sz + 1, step_sz):
        for j in range(0, n - wndw_sz + 1, step_sz):
            temp = img[i : i + wndw_sz, j : j + wndw_sz]
            org_img = numpy.append(org_img, [temp.flatten("F")])
    org_img = numpy.reshape(
        org_img, (wndw_sz * wndw_sz, org_img.size // (wndw_sz * wndw_sz)), order="F"
    )
    return org_img


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
    tree = xml.etree.ElementTree.parse(full_path)
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
    xml_df = pandas.DataFrame(xml_list, columns=column_name)
    xml_np = xml_df.to_numpy()
    return xml_np


def pts_near(gt_bbx, pred_bbx, rad) -> bool:
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


def get_target_loc(img_file, thresh: int, delta: int):
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
    y_v, x_v = numpy.where(img > thresh)

    if 0 in (len(x_v), len(y_v)):
        circ_img = circ_img_rgb
    for y_pos, x_pos in zip(y_v, x_v):
        x_p_a.append(x_pos)
        y_p_a.append(y_pos)

    for i, (x, y) in enumerate(zip(x_p_a, y_p_a)):
        if i in (0, len(x_v) - 1):
            r_x_p_a.append(x_p_a[i])
            r_y_p_a.append(y_p_a[i])
            if i == 0:
                circ_img = cv2.circle(circ_img_rgb, (x, y), radius, (0, 255, 0), 2)
        else:
            diff_x = abs(x_p_a[i] - x_p_a[i - 1])
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
