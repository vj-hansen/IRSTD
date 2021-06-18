"""
Utils for data-driven method
"""

import xml.etree.ElementTree as ET
import cv2
import pandas as pd


def save_image(
        img_filename, image, acc,
        model, score_thresh,
        top_left_crner, btm_right_crner):
    cv2.rectangle(
            image,
            top_left_crner,
            btm_right_crner,
            color = (0, 0, 255),
            thickness = 1)
    cv2.imwrite(
            '/Users/victor/Google Drive/detection_pics/'
            + model + "_"
    		+ str(score_thresh) + acc + "_" + img_filename
            + '.jpg', image)


def read_xml(path, in_file):
    """ Iterates through all .xml files in a given directory and combines
    them in a single Pandas dataframe.

    Parameters:
    ----------
    path : str
        The path containing the .xml files
    Returns:
        Numpy array
    """

    xml_list    = []
    full_path   = path+in_file+'.xml'
    tree        = ET.parse(full_path)
    root        = tree.getroot()
    for member in root.findall('object'):
        # the number of 'object' in the file dictates how many targets we have
        if len(member) == 7: # some xml files contain extra info on "pixels"
            value = (root.find('filename').text,
                    int(member[6][0].text),
                    int(member[6][1].text),
                    int(member[6][2].text),
                    int(member[6][3].text))
        elif len(member) == 5: # 1 object
            value = (root.find('filename').text,
                    int(member[4][0].text),
                    int(member[4][1].text),
                    int(member[4][2].text),
                    int(member[4][3].text))
        xml_list.append(value)
    column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df      = pd.DataFrame(xml_list, columns=column_name)
    xml_np      = xml_df.to_numpy()
    return xml_np


def get_iou(gt_bbx, pred_bbx):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Based on:
       https://stackoverflow.com/questions/25349178/
       calculating-percentage-of-bounding-box-overlap-for-image-detector-evaluation

    Parameters
    ----------
    gt_bbx : dict
        ymin, xmin, ymax, xmax]
        Keys: {'xmin', 'xmax', 'ymin', 'ymax'}
        The (xmin, ymin) position is at the top left corner,
        the (xmax, ymax) position is at the bottom right corner
    pred_bbx : dict
        Keys: {'xmin', 'xmax', 'ymin', 'ymax'}
        The (xmin, ymin) position is at the top left corner,
        the (xmax, ymax) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """

    assert gt_bbx['xmin']   < gt_bbx['xmax']
    assert gt_bbx['ymin']   < gt_bbx['ymax']
    assert pred_bbx['xmin'] < pred_bbx['xmax']
    assert pred_bbx['ymin'] < pred_bbx['ymax']

    # determine the coordinates of the intersection rectangle
    x_left      = max(gt_bbx['xmin'], pred_bbx['xmin'])
    y_top       = max(gt_bbx['ymin'], pred_bbx['ymin'])
    x_right     = min(gt_bbx['xmax'], pred_bbx['xmax'])
    y_bottom    = min(gt_bbx['ymax'], pred_bbx['ymax'])

    if (x_right < x_left) or (y_bottom < y_top):
        iou = 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
    else:
        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both BBs
        gt_bbx_area   = (gt_bbx['xmax']-gt_bbx['xmin'])*(gt_bbx['ymax']-gt_bbx['ymin'])
        pred_bbx_area = (pred_bbx['xmax']-pred_bbx['xmin'])*(pred_bbx['ymax']-pred_bbx['ymin'])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(gt_bbx_area + pred_bbx_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
    return iou, intersection_area
