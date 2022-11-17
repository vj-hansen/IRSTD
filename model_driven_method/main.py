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

import cv2
import numpy
from matplotlib import pyplot
from md_utils import get_target_loc, pts_near, read_xml
from pcp import pcp_func
from PIL import Image


class MainClass:
    def __init__(self):
        self.total_time = 0
        self.true_pos = 0
        self.false_pos = 0
        self.false_neg = 0
        self.total_gt_obj = 0
        self.img_filename = []
        self.total_detc = []
        self.max_it_param = 500
        self.tol_param = 1e-2
        self.method_param = "ialm"  # or apg
        self.thresh_param = 150
        self.slidewin_step_size = 20
        self.slidewin_patch_size = 80
        self.delta = 4
        cwd = os.getcwd()
        self.test_dir = f"{cwd}/model_driven_method/test_imgs/"
        self.img_dir = os.listdir(self.test_dir)
        self.save_dir = "model_driven_method/detection_pics/"

    def set_dirs(self) -> None:
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def get_files(self) -> list[str]:
        filelist = [file for file in self.img_dir if file.endswith(".png")]
        return filelist

    def run_pcp(self, img, im_shape):
        # returns NDArray[float64]
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

    def check_files(self, file):
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

    def get_time(self, start, end) -> None:
        round_time = end - start
        self.total_time = self.total_time + round_time
        print("Total time: %.2f s" % round_time)

    def iter_image(self, iter_, read_xml_file, circ_img_rgb, gtcx_arr, gtcy_arr):
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

    def assert_img(self, pcx_pos, gtcx_arr, gt_objects_in_img):
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

    def pts_prox(self, gt_bbx, pred_bbx, pcx_pos, gt_objects_in_img, status_img) -> str:
        # return true if objects are within proximity
        pts_close = pts_near(gt_bbx, pred_bbx, rad=5)
        status_img.append(pts_close)
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

    def get_boxes(
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
        filelist = self.get_files()
        for _, file in enumerate(filelist):
            gt_objects_in_img, read_xml_file = self.check_files(file)
            img = pyplot.imread("img.jpg")
            m, n = img.shape
            im_shape = (m, n)
            start = time.time()
            T = self.run_pcp(img, im_shape)
            end = time.time()
            self.get_time(start, end)
            self.total_gt_obj = gt_objects_in_img + self.total_gt_obj

            self.img_filename.append(file.split(".")[0])
            pyplot.imsave("t_img.jpg", T.reshape(im_shape), cmap="gray")
            print(f"{str(gt_objects_in_img)} object(s) in {file}")

            circ_img_rgb, pcx_pos, pcy_pos = get_target_loc(
                "t_img.jpg", thresh=self.thresh_param, delta=self.delta
            )
            self.total_detc.append(pcx_pos)

            gtcx_arr = []
            gtcy_arr = []
            status_img = []
            if gt_objects_in_img != 0:
                for iter_ in range(gt_objects_in_img):
                    gtcx_arr, gtcy_arr = self.iter_image(
                        iter_, read_xml_file, circ_img_rgb, gtcx_arr, gtcy_arr
                    )
            if len(pcx_pos) != 0:
                im_status, p_order, gt_order = self.assert_img(
                    pcx_pos, gtcx_arr, gt_objects_in_img
                )
                for it1, it2 in zip(range(len(pcx_pos)), range(gt_objects_in_img)):
                    gt_bbx, pred_bbx = self.get_boxes(
                        it1,
                        it2,
                        pcx_pos,
                        pcy_pos,
                        p_order,
                        gt_order,
                        gtcx_arr,
                        gtcy_arr,
                    )
                    im_status = self.pts_prox(
                        gt_bbx, pred_bbx, pcx_pos, gt_objects_in_img, status_img
                    )
            elif gt_objects_in_img == 0 and len(pcx_pos) == 0:
                im_status = "TN_"

            elif gt_objects_in_img - len(pcx_pos) > 0 and len(pcx_pos) == 0:
                self.false_neg += 1
                im_status = "FN_"

            cv2.imwrite(
                f"{self.save_dir}{im_status}_{self.method_param}_{self.tol_param}_{self.max_it_param}_{self.thresh_param}_{file.split('.')[0]}_target.jpg",
                circ_img_rgb,
            )
            self.print_results(filelist=filelist)

    def print_results(self, filelist: list[str]) -> None:
        avg_time = self.total_time / (len(filelist))
        print("Avg. time per img.: %.2f s" % avg_time)
        print("TP: ", self.true_pos)
        print("FP: ", self.false_pos)
        print("FN: ", self.false_neg)


if __name__ == "__main__":
    MainClass().run()
