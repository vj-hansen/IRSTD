"""
Code is based on:
    https://github.com/tensorflow/hub/blob/master/examples/colab/tf2_object_detection.ipynb
"""


import os
import warnings
import cv2
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
warnings.filterwarnings('ignore')

SCORE_THRESH = 0.5
MODEL = "DD-v1"  # DD-v1 or DD-v2
PATH_TO_SAVED_MODEL = "../saved_models/"+MODEL+"/saved_model/"
LABEL_MAP_DIR = "../saved_models/label_map.pbtxt"

cap = cv2.VideoCapture('helicopter.avi')
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
category_index = label_map_util.create_category_index_from_labelmap(
                                    LABEL_MAP_DIR,
                                    use_display_name=True)

while True:
    ret, nimg = cap.read()
    timer = cv2.getTickCount()
    IM_W = 320
    IM_H = 240
    input_tensor = tf.convert_to_tensor(nimg)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    img_w_detections = nimg.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
            img_w_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=5,
            agnostic_mode=True,
            line_thickness=1,
            min_score_thresh=SCORE_THRESH,
            skip_labels=True)

    fps = cv2.getTickFrequency() / (cv2.getTickCount()-timer)
    cv2.flip(img_w_detections, 0)
    cv2.putText(
        img_w_detections, "FPS: "+str(int(fps)),
        (100, 60), cv2.FONT_HERSHEY_SIMPLEX,
        1.0, (0, 0, 255), 2)

    cv2.imshow('CenterNet + ResNet50 FPN', cv2.resize(img_w_detections, (IM_W, IM_H)))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
