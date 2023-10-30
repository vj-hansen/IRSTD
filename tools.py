"""
Utilities used for model-driven method.
"""

import math
import xml.etree.ElementTree
from pathlib import Path

import cv2
import numpy
import pandas


def shrinking(mat_in, epsilon) -> numpy.ndarray:
    """
    Soft-thresholding (shrinkage) operator: S_epsilon[x]

    Args:
        TBA

    Returns:
        TBA
    """
    sgn = numpy.sign(mat_in)  # sign returns -1 if x < 0, 0 if x==0, 1 if x > 0
    return numpy.multiply(sgn, numpy.maximum(numpy.abs(mat_in) - epsilon, 0))


def matrix_to_grayscale(mat) -> numpy.ndarray:
    """
    Matrix to grayscale image

    Args:
        TBA

    Returns:
        img: numpy.ndarray
    """
    mat = numpy.asmatrix(mat)
    mat_min = numpy.min(mat)
    mat_max = numpy.max(mat)
    img = numpy.zeros(numpy.shape(mat))
    divisor_mat = float(mat_max - mat_min) * (mat - mat_min)

    if numpy.max(divisor_mat) > 0:
        img = numpy.add(
            img,
            numpy.multiply(
                numpy.logical_and(
                    numpy.greater_equal(mat, mat_min), numpy.less(mat, mat_max)
                ),
                (1 / float(mat_max - mat_min) * (mat - mat_min)),
            ),
        )

    return numpy.add(img, (numpy.greater_equal(mat, mat_max)))


def sliding_window(img_input, window_size, step_size, m, n) -> numpy.ndarray:
    """
    Sliding window

    Args:
        img_input: input image
        wndw_sz: size of sliding window
        step_sz: step size
        m, n: image shape

    Returns:
        image patch [2500 x ]
    """
    img = numpy.array(img_input)
    original_image = []
    for i in range(0, m - window_size + 1, step_size):
        for j in range(0, n - window_size + 1, step_size):
            temp = img[i : i + window_size, j : j + window_size]
            original_image = numpy.append(original_image, [temp.flatten("F")])

    return numpy.reshape(
        original_image,
        (window_size * window_size, original_image.size // (window_size * window_size)),
        order="F",
    )


def read_xml(xml_path, in_file) -> numpy.ndarray:
    """Iterates through all .xml files in a given directory and combines
    them in a single Pandas dataframe.

    Args:
        xml_path : str
            The path containing the .xml files

    Returns:
        xml_np: Numpy array
    """
    xml_list = []
    file_path = f"{in_file}.xml"
    full_path = Path(xml_path, file_path)
    root = xml.etree.ElementTree.parse(full_path).getroot()

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
    return pandas.DataFrame(xml_list, columns=column_name).to_numpy()


def points_near(gt_bbx, pred_bbx, rad) -> bool:
    """
    Determine if two points are within a radius.

    Args:
        gt_bbx : dict
                [centre_x, centre_y]
        pred_bbx : dict
                [centre_x, centre_y]

    Returns:
        True if two points are within a radius, else False
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


def get_target_location(img_file: str, thresh: int, delta: int):
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

    # TODO get rid off cv
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
