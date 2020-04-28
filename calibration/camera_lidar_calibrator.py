"""Top-level class for Calibration"""
import numpy as np
import cv2 as cv

from calibration.utils.data_utils import *
from calibration.utils.pc_utils import *

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from scipy.ndimage.filters import gaussian_filter, convolve

from calibration.pc_edge_detector import PcEdgeDetector
from calibration.img_edge_detector import ImgEdgeDetector


class CameraLidarCalibrator:

    def __init__(self, cfg):
        self.pc_detecter = PcEdgeDetector(cfg)
        self.img_detector = ImgEdgeDetector(cfg)

        self.pixels = None
        self.K = cfg.K
        self.R, self.T = load_lid_cal(cfg.calib_dir)

        # TODO: Change the methods below to use the new variables in pc_detector and img_detector

    def pc_to_pixels(self):
        '''
        Generate pixel coordinate for all points
        '''
        one_mat = np.ones((self.points.shape[0], 1))
        point_cloud = np.concatenate((self.points, one_mat), axis=1)

        # TODO: Perform transform without homogeneous term,
        #       if too memory intensive

        # Project point into Camera Frame
        point_cloud_cam = np.matmul(np.hstack((self.R, self.T)), point_cloud.T)

        # Remove the Homogeneous Term
        point_cloud_cam = np.matmul(self.P_rect, point_cloud_cam)

        # Normalize the Points into Camera Frame
        self.pixels = point_cloud_cam[::] / point_cloud_cam[::][-1]
        self.pixels = np.delete(self.pixels, 2, axis=0)

    def draw_points(self, image=None, FULL=True):
        """
        Draw points within corresponding camera's FoV on image provided.
        If no image provided, points are drawn on an empty(black) background.
        """

        if image is not None:
            image = np.dstack((image, image, image))
            hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        else:
            hsv_image = np.zeros(self.img.shape).astype(np.uint8)

        color = self.pc_to_colors()
        if FULL:
            index = range(self.pixels.shape[1])
        else:
            index = np.random.choice(self.pixels.shape[1],
                                     size=int(self.pixels.shape[1] / 10),
                                     replace=False)
        for i in index:
            if self.points[i, 0] < 0:
                continue
            if ((self.pixels[0, i] < 0) | (self.pixels[1, i] < 0) |
                (self.pixels[0, i] > hsv_image.shape[1]) |
                    (self.pixels[1, i] > hsv_image.shape[0])):
                continue
            cv.circle(
                hsv_image,
                (np.int32(self.pixels[0, i]), np.int32(self.pixels[1, i])), 1,
                (int(color[i]), 255, 255), -1)

        return cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)

    def pc_to_colors(self, min_d=0, max_d=120):
        """
        print Color(HSV's H value) corresponding to distance(m)
        close distance = red , far distance = blue
        """
        dist = np.sqrt(
            np.add(np.power(self.points[:, 0],
                            2), np.power(self.points[:, 1], 2),
                   np.power(self.points[:, 2], 2)))
        np.clip(dist, 0, max_d, out=dist)
        # max distance is 120m but usually not usual
        return (((dist - min_d) / (max_d - min_d)) * 120).astype(np.uint8)
