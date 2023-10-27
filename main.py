"""
Model-driven approach for IR Small Target Detection

Concept based on:
   C. Gao, D. Meng, Y. Yang, Y. Wang, X. Zhou and A. G. Hauptmann,
"Infrared Patch-Image Model for Small Target Detection in a Single Image,"
in IEEE Transactions on Image Processing, vol. 22, no. 12, pp. 4996-5009,
Dec. 2013, doi: 10.1109/TIP.2013.2281420.

"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from matplotlib import pyplot
from PIL import Image

from tools import get_target_loc, pts_near, read_xml
from principal_component_pursuit import pcp_func


class MainClass:
    def __init__(self, method: str, input_image_dir: str):
        self.total_time = 0
        self.true_pos = 0
        self.false_pos = 0
        self.false_neg = 0
        self.total_gt_obj = 0
        self.img_filename = []
        self.total_detc = []
        self.max_it_param = 500
        self.tol_param = 1e-2
        self.method_param = method
        self.thresh_param = 150
        self.slidewin_step_size = 20
        self.slidewin_patch_size = 80
        self.delta = 4
        self.test_dir = f"{Path.cwd()}/{input_image_dir}/"
        self.img_dir = os.listdir(self.test_dir)
        self.save_dir = "detection_pics/"

        self._set_dirs()

    def _set_dirs(self):
        if not Path(self.save_dir).exists():
            Path(self.save_dir).mkdir(parents=True)

    def _get_files(self):
        filelist = [file for file in self.img_dir if file.endswith(".png")]
        logger.debug(filelist)
        return filelist

    def _run_pcp(self, img, im_shape):
        T = pcp_func(
            img,
            im_shape,
            max_iter=self.max_it_param,
            tol=self.tol_param,
            method=self.method_param,
            sw_step_size=self.slidewin_step_size,
            sw_ptch_sz=self.slidewin_patch_size,
        )
        return T

    def _check_files(self, file):
        if file.split(".")[-1] == "png":
            fullpath = self.test_dir + file
            tmp_img = Image.open(fullpath).convert("L")
            tmp_img.save("img.jpg")
            if os.path.isfile(fullpath):
                read_xml_file = read_xml(self.test_dir, file.split(".")[0])
                gt_objects_in_img = len(read_xml_file)
            else:
                gt_objects_in_img = 0
        return gt_objects_in_img, read_xml_file

    def _get_time(self, start, end):
        round_time = end - start
        self.total_time = self.total_time + round_time
        logger.debug("Total time: %.2f s" % round_time)

    def _iter_image(self, iter_, read_xml_file, circ_img_rgb, gtcx_arr, gtcy_arr):
        ymin_xml = read_xml_file[iter_][2]
        xmin_xml = read_xml_file[iter_][1]
        ymax_xml = read_xml_file[iter_][4]
        xmax_xml = read_xml_file[iter_][3]
        cx_xml = int((xmax_xml + xmin_xml) // 2)
        cy_xml = int((ymax_xml + ymin_xml) // 2)
        cv2.circle(circ_img_rgb, (cx_xml, cy_xml), 10, (0, 0, 255), 2)
        gtcx_arr.append(cx_xml)
        gtcy_arr.append(cy_xml)
        return gtcx_arr, gtcy_arr

    def _assert_img(self, pcx_pos, gtcx_arr, gt_objects_in_img):
        p_order = np.argsort(pcx_pos)
        gt_order = np.argsort(gtcx_arr)
        if gt_objects_in_img == len(pcx_pos):
            self.true_pos += 1
            im_status = "TP_"
        elif gt_objects_in_img - len(pcx_pos) > 0:
            self.false_neg += 1
            im_status = "FN_"
        elif (len(pcx_pos) - gt_objects_in_img > 0) or (
            gt_objects_in_img == 0 and len(pcx_pos) != 0
        ):
            self.false_pos += 1
            im_status = "FP_"
        return im_status, p_order, gt_order

    def _pts_prox(
        self, gt_bbx, pred_bbx, pcx_pos, gt_objects_in_img, status_img
    ) -> str:
        pts_close = pts_near(
            gt_bbx, pred_bbx, rad=5
        )  # return true if objects are within proximity
        status_img.append(pts_close)
        im_status = ""
        if pts_close and gt_objects_in_img == len(pcx_pos):
            self.true_pos += 1
            if sum(status_img) == gt_objects_in_img:
                # only if num(TRUE_POS) for this file == num(gt_obj_in_img)
                im_status = "TP_"
            else:
                self.false_neg += 1
                im_status = "FN_"
        elif not (pts_close) and len(pcx_pos) > gt_objects_in_img:
            self.false_pos += 1
            # only if num(False_POS) > num(gt_obj_in_img)
            im_status = "FP_"
        return im_status

    def _get_boxes(
        self, it1, it2, pcx_pos, pcy_pos, p_order, gt_order, gtcx_arr, gtcy_arr
    ):
        pred_bbx = {
            "centre_x": pcx_pos[p_order[it1]],
            "centre_y": pcy_pos[p_order[it1]],
        }

        gt_bbx = {
            "centre_x": gtcx_arr[gt_order[it2]],
            "centre_y": gtcy_arr[gt_order[it2]],
        }
        return gt_bbx, pred_bbx

    def run(self) -> None:
        filelist = self._get_files()
        im_status = ""
        for _, file in enumerate(filelist):
            gt_objects_in_img, read_xml_file = self._check_files(file)
            img = pyplot.imread("img.jpg")
            m, n = img.shape
            im_shape = (m, n)
            start = time.time()
            T = self._run_pcp(img, im_shape)
            end = time.time()
            self._get_time(start, end)
            self.total_gt_obj = gt_objects_in_img + self.total_gt_obj

            self.img_filename.append(file.split(".")[0])
            pyplot.imsave("t_img.jpg", T.reshape(im_shape), cmap="gray")
            logger.info(f"{str(gt_objects_in_img)} object(s) in {file}")

            circ_img_rgb, pcx_pos, pcy_pos = get_target_loc(
                "t_img.jpg", thresh=self.thresh_param, delta=self.delta
            )
            self.total_detc.append(pcx_pos)

            gtcx_arr = []
            gtcy_arr = []
            status_img = []
            if gt_objects_in_img != 0:
                for iter_ in range(gt_objects_in_img):
                    gtcx_arr, gtcy_arr = self._iter_image(
                        iter_, read_xml_file, circ_img_rgb, gtcx_arr, gtcy_arr
                    )
            if len(pcx_pos) != 0:
                im_status, p_order, gt_order = self._assert_img(
                    pcx_pos, gtcx_arr, gt_objects_in_img
                )
                for it1, it2 in zip(range(len(pcx_pos)), range(gt_objects_in_img)):
                    gt_bbx, pred_bbx = self._get_boxes(
                        it1,
                        it2,
                        pcx_pos,
                        pcy_pos,
                        p_order,
                        gt_order,
                        gtcx_arr,
                        gtcy_arr,
                    )
                    im_status = self._pts_prox(
                        gt_bbx, pred_bbx, pcx_pos, gt_objects_in_img, status_img
                    )
            elif gt_objects_in_img == 0 and len(pcx_pos) == 0:
                im_status = "TN_"

            elif gt_objects_in_img - len(pcx_pos) > 0 and len(pcx_pos) == 0:
                self.false_neg += 1
                im_status = "FN_"

            image_write = os.path.join(
                self.save_dir,
                f"{im_status}{self.method_param}_{self.tol_param}_{self.max_it_param}_{self.thresh_param}_{file.split('.')[0]}_target.jpg",
            )
            logger.debug(image_write)
            cv2.imwrite(
                image_write,
                circ_img_rgb,
            )
            self._print_results(filelist)

    def _print_results(self, filelist) -> None:
        avg_time = self.total_time / (len(filelist))
        logger.info("Avg. time per img.: %.2f s" % avg_time)
        logger.info(f"TP: {self.true_pos}")
        logger.info(f"FP: {self.false_pos}")
        logger.info(f"FN: {self.false_neg}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method", type=str, help="Model driven method", choices=["ialm", "apg"]
    )
    parser.add_argument("--image-dir", type=str, help="Image dir name")
    args = parser.parse_args()

    assert args.method, "No method provided... Aborting!"
    assert args.image_dir, "No image dir provided... Aborting!"

    MainClass(method=args.method, input_image_dir=args.image_dir).run()


if __name__ == "__main__":
    main()
