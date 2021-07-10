'''
Based on:
    https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_object_detection.ipynb

Test trained data-driven model on images.
'''

import os
import warnings
import time
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import cv2

from dd_utils import read_xml, get_iou, save_image


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
warnings.filterwarnings('ignore')

SCORE_THRESH = 0.5
MODEL = "DD-v1"  # DD-v1 or DD-v2
PATH_TO_SAVED_MODEL = "../saved_models/"+MODEL+"/saved_model/"
TEST_DIR = "../dataset/dataset_images/target_test/"
LABEL_MAP_DIR = "../saved_models/label_map.pbtxt"
GT_OBJECTS_IN_IMG = 0

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
category_index = label_map_util.create_category_index_from_labelmap(
                                        LABEL_MAP_DIR,
                                        use_display_name=True)
img_dir = os.listdir(TEST_DIR)

if not os.path.exists('detection_pics/'):
    os.makedirs('detection_pics/')

filelist = [file for file in img_dir if file.endswith('.png')]

images = []
img_filename = []
SUM_NOB = 0
SUM_IOU = 0
TRUE_POS = 0
TRUE_NEG = 0
FALSE_POS = 0
FALSE_NEG = 0
TOTAL_TIME = 0
IOU = 0

for it, file in enumerate(filelist):
    if file.split(".")[-1] == 'png':
        fullpath = TEST_DIR+file
        image_np = Image.open(fullpath).convert("RGB")
        image_np = np.array(image_np)
        if os.path.isfile(TEST_DIR+file.split(".")[0]):
            read_xml_file = read_xml(TEST_DIR, file.split(".")[0])
            GT_OBJECTS_IN_IMG = len(read_xml_file)
        else:
            GT_OBJECTS_IN_IMG = 0
        if len(image_np.shape) < 2:
            nimg = (image_np)
            nimg[:, :, 0] = image_np
            nimg[:, :, 1] = image_np
            nimg[:, :, 2] = image_np
        else:
            nimg = image_np

    input_tensor = tf.convert_to_tensor(nimg)
    input_tensor = input_tensor[tf.newaxis, ...]

    start = time.time()
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    end = time.time()
    round_time = end-start
    TOTAL_TIME = TOTAL_TIME+round_time
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    img_w_detections = nimg.copy()
    images.append(img_w_detections)
    img_filename.append(file.split(".")[0])

    viz_utils.visualize_boxes_and_labels_on_image_array(
            img_w_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=10,
            agnostic_mode=True,
            line_thickness=1,
            min_score_thresh=SCORE_THRESH,
            skip_labels=True)

    boxes = np.squeeze(detections['detection_boxes'])
    scores = np.squeeze(detections['detection_scores'])
    number_of_boxes = np.sum(scores > SCORE_THRESH)
    SUM_NOB = number_of_boxes+SUM_NOB

    if (number_of_boxes-GT_OBJECTS_IN_IMG > 0) and (GT_OBJECTS_IN_IMG != 0):
        FALSE_POS += 1
        # FP: predicts that an object exist when it DNE
        for i in range(GT_OBJECTS_IN_IMG):
            top_left_crner = (read_xml_file[i][1], read_xml_file[i][2])
            btm_right_crner = (read_xml_file[i][3], read_xml_file[i][4])
            save_image(
                img_filename[it], images[it], "fp", MODEL,
                SCORE_THRESH, top_left_crner, btm_right_crner)

    elif (GT_OBJECTS_IN_IMG-number_of_boxes > 0) and (GT_OBJECTS_IN_IMG != 0):
        FALSE_NEG += 1
        # FN: predicts that an object DNE when it does
        for j in range(GT_OBJECTS_IN_IMG):
            top_left_crner = (read_xml_file[j][1], read_xml_file[j][2])
            btm_right_crner = (read_xml_file[j][3], read_xml_file[j][4])
            save_image(
                img_filename[it], images[it], "fn", MODEL,
                SCORE_THRESH, top_left_crner, btm_right_crner)

    elif (GT_OBJECTS_IN_IMG == 0) and (number_of_boxes > 0):
        FALSE_POS += 1
        cv2.imwrite(
            'detection_pics/' + MODEL + "_"
            + str(SCORE_THRESH) + 'fp' + "_"
            + img_filename[it] + '.jpg', images[it])

    elif (GT_OBJECTS_IN_IMG == 0) and (number_of_boxes == 0):
        TRUE_NEG += 1

    elif GT_OBJECTS_IN_IMG == number_of_boxes:
        for i in range(number_of_boxes):
            ymin = boxes[i][0]
            xmin = boxes[i][1]
            ymax = boxes[i][2]
            xmax = boxes[i][3]
            W = (nimg.shape[1])
            H = (nimg.shape[0])

            # csv_rows = [img_filename, bbx_ymin, bbx_xmin, bbx_ymax, bbx_xmax]
            bbx_ymin = int(ymin*H)
            bbx_xmin = int(xmin*W)
            bbx_ymax = int(ymax*H)
            bbx_xmax = int(xmax*W)

            pred_bbx = {
                "file": file,
                "ymin": int(bbx_ymin),
                "xmin": int(bbx_xmin),
                "ymax": int(bbx_ymax),
                "xmax": int(bbx_xmax)
            }

            gt_bbx = {
                "file": read_xml_file[i][0],
                "ymin": read_xml_file[i][2],
                "xmin": read_xml_file[i][1],
                "ymax": read_xml_file[i][4],
                "xmax": read_xml_file[i][3]
            }

            top_left_crner = (read_xml_file[i][1], read_xml_file[i][2])
            btm_right_crner = (read_xml_file[i][3], read_xml_file[i][4])

            IOU, intersection_area = get_iou(gt_bbx, pred_bbx)
            SUM_IOU = IOU+SUM_IOU
            if intersection_area > 0:
                TRUE_POS += 1
            elif intersection_area < 0:
                FALSE_POS += 1

            save_image(
                img_filename[it], images[it],
                "tp", MODEL, SCORE_THRESH,
                top_left_crner, btm_right_crner)

avg_time = TOTAL_TIME/(len(filelist))
print("Avg. time: %.2f s" % avg_time)
fps = (1/avg_time)
print("fps: %.2f" % fps)
print("TP:", TRUE_POS)
print("TN:", TRUE_NEG)
print("FP:", FALSE_POS)
print("FN:", FALSE_NEG)
print("Objects detected", SUM_NOB)
