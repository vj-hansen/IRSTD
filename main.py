#!/usr/bin/python3

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
import numpy
from matplotlib import pyplot
from PIL import Image

from principal_component_pursuit import pcp_func
from tools import get_target_location, points_near, read_xml


class Irstd:
    _MAX_ITERATIONS = 500
    _TOLERANCE = 1e-2
    _THRESHOLD = 150
    _SLIDEWIN_STEP_SIZE = 20
    _SLIDEWIN_PATCH_SIZE = 80
    _DELTA = 4
    _RESULT_DIR = "results/"

    def __init__(self, method: str, input_image_dir: str):
        self.total_time = 0
        self.true_pos = 0
        self.false_pos = 0
        self.false_neg = 0
        self.total_gt_obj = 0
        self.img_filename = []
        self.total_detections = []

        self.method_param = method

        self.test_dir = Path.cwd() / input_image_dir
        self.img_dir = os.listdir(self.test_dir)
        self._set_dirs()

    def _set_dirs(self) -> None:
        if not Path(self._RESULT_DIR).exists():
            Path(self._RESULT_DIR).mkdir(parents=True)

    def _get_files(self) -> list[str]:
        return [file for file in self.img_dir if file.endswith(".png")]

    def _run_pcp(self, img, im_shape) -> numpy.ndarray:
        """
        Args:
            img:
            im_shape:

        Returns:
            numpy.ndarray
        """
        return pcp_func(
            img,
            im_shape,
            max_iter=self._MAX_ITERATIONS,
            tol=self._TOLERANCE,
            method=self.method_param,
            sw_step_size=self._SLIDEWIN_STEP_SIZE,
            sw_patch_size=self._SLIDEWIN_PATCH_SIZE,
        )

    def _check_files(self, file: str) -> tuple[int, numpy.ndarray]:
        """
        Args:
            file: str

        Returns:
            gt_objects_in_img: int
            read_xml_file: numpy.ndarray
        """
        gt_objects_in_img = 0
        if file.endswith("png"):
            fullpath = Path(self.test_dir, file)
            tmp_img = Image.open(fullpath).convert("L")
            tmp_img.save("img.jpg")

            if fullpath.is_file():
                read_xml_file = read_xml(self.test_dir, file.split(".")[0])
                gt_objects_in_img = len(read_xml_file)
            else:
                gt_objects_in_img = 0

        else:
            gt_objects_in_img = 0
            read_xml_file = []

        return gt_objects_in_img, read_xml_file

    def _get_time(self, start, end):
        round_time = end - start
        self.total_time = self.total_time + round_time
        print(f"Total time: {round_time:.3f} s")

    def _iter_image(self, iter_, read_xml_file, circ_img_rgb, gtcx_arr, gtcy_arr):
        """
        Args:
            TBA

        Returns:
            TBA
        """
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
        """
        Args:
            pcx_pos:
            gtcx_arr:
            gt_objects_in_img:

        Returns:
            im_status: str
            p_order:
            gt_order:
        """
        im_status = ""
        p_order = numpy.argsort(pcx_pos)
        gt_order = numpy.argsort(gtcx_arr)

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

    def _points_within_proximity(
        self, gt_bbx, pred_bbx, pcx_pos, gt_objects_in_img, status_img
    ) -> str:
        """Check if predicted points are within proximity of ground-truth points"""
        points_close = points_near(
            gt_bbx, pred_bbx, rad=5
        )  # return true if objects are within proximity

        status_img.append(points_close)
        im_status = ""

        if points_close and gt_objects_in_img == len(pcx_pos):
            self.true_pos += 1
            if sum(status_img) == gt_objects_in_img:
                # only if num(TRUE_POS) for this file == num(gt_obj_in_img)
                im_status = "TP_"
            else:
                self.false_neg += 1
                im_status = "FN_"

        elif not (points_close) and len(pcx_pos) > gt_objects_in_img:
            self.false_pos += 1
            # only if num(False_POS) > num(gt_obj_in_img)
            im_status = "FP_"

        return im_status

    def _get_boxes(
        self, it1, it2, pcx_pos, pcy_pos, p_order, gt_order, gtcx_arr, gtcy_arr
    ) -> tuple[dict, dict]:
        """
        Args:
            TBA

        Returns:
            TBA
        """
        pred_bbx = {
            "centre_x": pcx_pos[p_order[it1]],
            "centre_y": pcy_pos[p_order[it1]],
        }

        gt_bbx = {
            "centre_x": gtcx_arr[gt_order[it2]],
            "centre_y": gtcy_arr[gt_order[it2]],
        }
        return gt_bbx, pred_bbx

    def run(self):
        """
        FIXME: Too long / doing too much stuff here
        Args:
            TBA

        Returns:
            TBA
        """
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
            print(f"{gt_objects_in_img} object(s) in {file}")

            circ_img_rgb, pcx_pos, pcy_pos = get_target_location(
                "t_img.jpg", thresh=self._THRESHOLD, delta=self._DELTA
            )
            self.total_detections.append(pcx_pos)

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
                    im_status = self._points_within_proximity(
                        gt_bbx, pred_bbx, pcx_pos, gt_objects_in_img, status_img
                    )

            elif gt_objects_in_img == 0 and len(pcx_pos) == 0:
                im_status = "TN_"

            elif gt_objects_in_img - len(pcx_pos) > 0 and len(pcx_pos) == 0:
                self.false_neg += 1
                im_status = "FN_"

            image_write = os.path.join(
                self._RESULT_DIR,
                f"{im_status}{self.method_param}_{self._TOLERANCE}_{self._MAX_ITERATIONS}"
                f"_{self._THRESHOLD}_{file.split('.')[0]}_target.jpg",
            )

            cv2.imwrite(
                image_write,
                circ_img_rgb,
            )
            self._print_results(filelist)

    def _print_results(self, filelist) -> None:
        avg_time = self.total_time / (len(filelist))
        print(f"Avg. time per img.: {avg_time:.3f} s")
        print(f"TP: {self.true_pos}")
        print(f"FP: {self.false_pos}")
        print(f"FN: {self.false_neg}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--method", type=str, help="Model driven method", choices=["ialm", "apg"]
    )
    parser.add_argument("--image-dir", type=str, help="Image dir name")
    args = parser.parse_args()

    assert args.method, "No method provided... Aborting!"
    assert args.image_dir, "No image dir provided... Aborting!"

    Irstd(method=args.method, input_image_dir=args.image_dir).run()


if __name__ == "__main__":
    main()
