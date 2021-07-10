"""
    Zoom, shift, mirror and adjust brightness of images.
    Used for increasing the size of our data set
"""
import os
import random
import cv2
import numpy as np


IMG_IDX = 1
SRC_DIR = "../dataset/dataset_images/"
DST_DIR = "../dataset/dataset_images/agmntd_dataset/"
src_files = os.listdir(SRC_DIR)

if not os.path.exists(DST_DIR):
    os.makedirs(DST_DIR)


def fill(f_img, img_h, img_w):
    f_img = cv2.resize(
                f_img,
                (img_h, img_w),
                cv2.INTER_CUBIC)
    return f_img


def horizontal_shift(h_img, ratio=0.0):
    # zoom and shift image along x-axis
    if ratio > 1 or ratio < 0:
        print('Value should be less than 1 and greater than 0')
        return h_img

    ratio = random.uniform(-ratio, ratio)
    img_h, img_w = h_img.shape[:2]
    to_shift = img_w*ratio
    if ratio > 0:
        h_img = h_img[:, :int(img_w-to_shift), :]
    if ratio < 0:
        h_img = h_img[:, int(-1*to_shift):, :]
    h_img = fill(h_img, img_h, img_w)
    return h_img


def zoom(z_img, value):
    # zoom in on area of image
    if value > 1 or value < 0:
        return z_img
    value = random.uniform(value, 1)
    img_h, img_w = z_img.shape[:2]
    h_taken = int(value*img_h)
    w_taken = int(value*img_w)
    h_start = random.randint(0, img_h - h_taken)
    w_start = random.randint(0, img_w - w_taken)
    z_img = z_img[h_start:h_start + h_taken, w_start:w_start + w_taken, :]
    z_img = fill(z_img, img_h, img_w)
    return z_img


def brightness(b_img, low, high):
    # adjust image brightness
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(b_img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)

    hsv[:, :, 1] = hsv[:, :, 1]*value
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv[:, :, 2] = hsv[:, :, 2]*value
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255

    hsv = np.array(hsv, dtype=np.uint8)
    b_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return b_img


def mirroring(m_img, flip_dir):
    """
    mirror image around x-axis
    FLIP_HORZ = 1
    FLIP_VERT = 0
    """
    m_img = cv2.flip(m_img, flip_dir)
    return m_img


def rotation(r_img):
    # rotate image
    r_img = cv2.rotate(r_img, cv2.cv2.ROTATE_90_CLOCKWISE)
    return r_img


for file in src_files:
    if os.path.isfile(SRC_DIR+file):
        a, b = os.path.splitext(SRC_DIR + file)
        img = cv2.imread(str(a+b), 1)
        img = cv2.imread(str(a+b), 1)
        img = brightness(img, 0.4, 1)
        img = horizontal_shift(img, 0.4)
        img = zoom(img, 0.2)
        img = mirroring(img, 0)
        img = rotation(img)
        cv2.imwrite(DST_DIR + 'aug_misc_'+str(IMG_IDX) + ".png", img)
        IMG_IDX += 1
print("Done..")
