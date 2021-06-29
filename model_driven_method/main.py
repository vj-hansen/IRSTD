"""
Model-driven approach for IR Small Target Detection

Concept based on:
   C. Gao, D. Meng, Y. Yang, Y. Wang, X. Zhou and A. G. Hauptmann,
"Infrared Patch-Image Model for Small Target Detection in a Single Image,"
in IEEE Transactions on Image Processing, vol. 22, no. 12, pp. 4996-5009,
Dec. 2013, doi: 10.1109/TIP.2013.2281420.

"""

import os
import time
import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from pcp import pcp_func
from md_utils import read_xml, pts_near, get_target_loc

cwd = os.getcwd()
TEST_DIR    = cwd+"/model_driven_method/test_imgs/"
#TEST_DIR   = "../dataset/dataset_images/target_test/"
img_dir     = os.listdir(TEST_DIR)
SAVE_DIR    = 'model_driven_method/detection_pics/'

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

filelist = [file for file in img_dir if file.endswith('.png')]
TOTAL_TIME      = 0
TRUE_POS        = 0
FALSE_POS       = 0
FALSE_NEG       = 0
TOTAL_GT_OBJ    = 0

images          = []
img_filename    = []
total_detc      = []

MAX_IT_PARAM        = 500
TOL_PARAM           = 1e-2
METHOD_PARAM        = 'ialm' # or apg
THRESH_PARAM        = 150
SLIDEWIN_STEP_SIZE  = 20
SLIDEWIN_PATCH_SIZE = 80
DELTA               = 4

for it, file in enumerate(filelist):
    if file.split(".")[-1] == 'png':
        fullpath    = TEST_DIR + file
        tmp_img     = Image.open(fullpath).convert("L")
        tmp_img.save('img.jpg')
        if os.path.isfile(fullpath):
            read_xml_file       = read_xml(TEST_DIR, file.split(".")[0])
            GT_OBJECTS_IN_IMG   = len(read_xml_file)
        else:
            GT_OBJECTS_IN_IMG = 0

    img         = plt.imread('img.jpg')
    m, n        = img.shape
    im_shape    = (m, n)
    start       = time.time()

    T = pcp_func(
            img,
            im_shape,
            max_iter        = MAX_IT_PARAM,
            tol             = TOL_PARAM,
            method          = METHOD_PARAM,
            sw_step_size    = SLIDEWIN_STEP_SIZE,
            sw_ptch_sz      = SLIDEWIN_PATCH_SIZE)

    end         = time.time()
    round_time  = end-start
    TOTAL_TIME  = TOTAL_TIME + round_time
    print("Total time: %.2f s" % round_time)
    TOTAL_GT_OBJ = GT_OBJECTS_IN_IMG + TOTAL_GT_OBJ

    img_filename.append(file.split(".")[0])
    plt.imsave('t_img.jpg', T.reshape(im_shape), cmap = 'gray')
    print(str(GT_OBJECTS_IN_IMG) + ' object(s) in ' + file)

    circ_img_rgb, pcx_pos, pcy_pos = get_target_loc('t_img.jpg',
                                                    thresh  = THRESH_PARAM,
                                                    delta   = DELTA)
    total_detc.append(pcx_pos)

    gtcx_arr    = []
    gtcy_arr    = []
    status_img  = []
    if GT_OBJECTS_IN_IMG != 0: # GT objects in image
        for iter1 in range(GT_OBJECTS_IN_IMG):
            ymin_xml    = read_xml_file[iter1][2]
            xmin_xml    = read_xml_file[iter1][1]
            ymax_xml    = read_xml_file[iter1][4]
            xmax_xml    = read_xml_file[iter1][3]
            cx_xml      = int((xmax_xml + xmin_xml) // 2)
            cy_xml      = int((ymax_xml + ymin_xml) // 2)
            cv2.circle(circ_img_rgb, (cx_xml, cy_xml), 10, (0, 0, 255), 2)
            gtcx_arr.append(cx_xml)
            gtcy_arr.append(cy_xml)

    if len(pcx_pos) != 0:
        p_order     = np.argsort(pcx_pos)
        gt_order    = np.argsort(gtcx_arr)
        if GT_OBJECTS_IN_IMG == len(pcx_pos):
            TRUE_POS    += 1
            IM_STATUS   = 'TP_'
        elif GT_OBJECTS_IN_IMG - len(pcx_pos) > 0:
            FALSE_NEG   += 1
            IM_STATUS   = 'FN_'
        elif (len(pcx_pos) - GT_OBJECTS_IN_IMG > 0) or \
        	(GT_OBJECTS_IN_IMG == 0 and len(pcx_pos) != 0):
            FALSE_POS   += 1
            IM_STATUS   = 'FP_'
        for it1, it2 in zip(range(len(pcx_pos)),
                            range(GT_OBJECTS_IN_IMG)):
            pred_bbx = {
                    "centre_x" : pcx_pos[p_order[it1]],
                    "centre_y" : pcy_pos[p_order[it1]]
                    }

            gt_bbx = {
                    "centre_x" : gtcx_arr[gt_order[it2]],
                    "centre_y" : gtcy_arr[gt_order[it2]]
                    }

            # return true if objects are within proximity
            PTS_CLOSE = pts_near(gt_bbx, pred_bbx, rad=5)
            status_img.append(PTS_CLOSE)
            if PTS_CLOSE and GT_OBJECTS_IN_IMG == len(pcx_pos):
                TRUE_POS += 1
                if sum(status_img) == GT_OBJECTS_IN_IMG:
                    IM_STATUS = 'TP_' # only if num(TRUE_POS) for this file == num(gt_obj_in_img)
                else:
                    FALSE_NEG   += 1
                    IM_STATUS   = 'FN_'
            elif not(PTS_CLOSE) and len(pcx_pos) > GT_OBJECTS_IN_IMG:
                FALSE_POS   += 1
                IM_STATUS   = 'FP_' # only if num(False_POS) > num(gt_obj_in_img)

    elif GT_OBJECTS_IN_IMG == 0 and len(pcx_pos) == 0:
        IM_STATUS = 'TN_'

    elif GT_OBJECTS_IN_IMG - len(pcx_pos) > 0 and len(pcx_pos) == 0:
        FALSE_NEG   += 1
        IM_STATUS   = 'FN_'

    cv2.imwrite(SAVE_DIR+IM_STATUS+'_'+METHOD_PARAM+'_'+str(TOL_PARAM)+'_'+str(MAX_IT_PARAM)+
                '_'+str(THRESH_PARAM)+'_'+file.split(".")[0]+'_target.jpg', circ_img_rgb)

avg_time = TOTAL_TIME/(len(filelist))
print("Avg. time per img.: %.2f s" % avg_time)
print("TP: ", TRUE_POS)
print("FP: ", FALSE_POS)
print("FN: ", FALSE_NEG)
