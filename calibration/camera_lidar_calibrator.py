"""Top-level class for Calibration"""
import numpy as np
import cv2 as cv
import gc
import pyquaternion as pyquat
from scipy.stats import multivariate_normal
from scipy.linalg import expm
from scipy.stats import norm

from calibration.utils.data_utils import *
from calibration.utils.pc_utils import *
from calibration.utils.img_utils import *

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from scipy.ndimage.filters import gaussian_filter, convolve

from calibration.pc_edge_detector import PcEdgeDetector
from calibration.img_edge_detector import ImgEdgeDetector


class CameraLidarCalibrator:

    def __init__(self, cfg, visualize=False, tau_init=None):
        self.pc_detector = PcEdgeDetector(cfg, visualize=visualize)
        gc.collect()
        self.img_detector = ImgEdgeDetector(cfg, visualize=visualize)
        gc.collect()
        self.pixels = None
        self.pixels_mask = None
        self.K = np.asarray(cfg.K)
        self.R, self.T = load_lid_cal(cfg.calib_dir)

        if tau_init:
            self.tau = tau_init
        elif isinstance(self.R, np.ndarray) and isinstance(self.T, np.ndarray):
            self.tau = self.transform_to_tau(self.R, self.T)
        else:
            self.tau = np.zeros((1, 5))

        self.visualize = visualize
        # TODO: Change the methods below to use the new variables in pc_detector and img_detector

    @staticmethod
    def transform_to_tau(R, T):
        r_vec, _ = cv2.Rodrigues(R)
        return np.hstack((r_vec.T, T.T)).reshape(6, )

    @staticmethod
    def tau_to_transform(tau):
        R, _ = cv2.Rodrigues(tau[:3])
        T = tau[3:].reshape((3, 1))
        return R, T

    def pc_to_pixels(self):
        '''
        Generate pixel coordinate for all points
        '''
        # Compute R and T from current tau
        self.R, self.T = self.tau_to_transform(self.tau)

        one_mat = np.ones((self.pc_detector.pcs.shape[0], 1))
        point_cloud = np.concatenate((self.pc_detector.pcs, one_mat), axis=1)

        # TODO: Perform transform without homogeneous term,
        #       if too memory intensive

        # Project point into Camera Frame
        point_cloud_cam = np.matmul(np.hstack((self.R, self.T)), point_cloud.T)

        # Remove the Homogeneous Term
        point_cloud_cam = np.matmul(self.K, point_cloud_cam)

        # Normalize the Points into Camera Frame
        self.pixels = point_cloud_cam[::] / point_cloud_cam[::][-1]
        self.pixels = np.delete(self.pixels, 2, axis=0)
        self.pixels = self.pixels.T

        # Remove pixels that are outside image
        inside_mask_x = np.logical_and((self.pixels[:, 0] >= 0),
                                       (self.pixels[:, 0] <= self.img_detector.img_w))
        inside_mask_y = np.logical_and((self.pixels[:, 1] >= 0),
                                       (self.pixels[:, 1] <= self.img_detector.img_h))
        inside_mask = np.logical_and(inside_mask_x, inside_mask_y)

        # if self.visualize:
        #     blank = np.zeros(
        #         (self.img_detector.img_h, self.img_detector.img_w))
        #     blank[self.pixels[inside_mask, 1].astype(
        #         np.int), self.pixels[inside_mask, 0].astype(np.int)] = 255
        #     cv.imshow('Projected Lidar Edges', blank)
        #     cv.waitKey(0)

        self.pixels_mask = inside_mask

    def draw_points(self, image=None, FULL=True):
        """
        Draw points within corresponding camera's FoV on image provided.
        If no image provided, points are drawn on an empty(black) background.
        """

        if image is not None:
            image = np.uint8(np.dstack((image, image, image)))*255
            cv.imshow('Before projection', image)
            cv.waitKey(0)

            hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        else:
            hsv_image = np.zeros(self.img_detector.imgs.shape).astype(np.uint8)

        color = self.pc_to_colors()
        if FULL:
            index = range(self.pixels.shape[0])
        else:
            index = np.random.choice(self.pixels.shape[0],
                                     size=int(self.pixels.shape[0] / 10),
                                     replace=False)
        for i in index:
            if self.pc_detector.pcs[i, 0] < 0:
                continue
            if self.pixels_mask[i] is False:
                continue

            cv.circle(
                hsv_image,
                (np.int32(self.pixels[i, 0]), np.int32(self.pixels[i, 1])), 1,
                (int(color[i]), 255, 255), -1)

        return cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)

    def pc_to_colors(self, min_d=0, max_d=120):
        """
        print Color(HSV's H value) corresponding to distance(m)
        close distance = red , far distance = blue
        """
        dist = np.sqrt(
            np.add(np.power(self.pc_detector.pcs[:, 0], 2),
                   np.power(self.pc_detector.pcs[:, 1], 2),
                   np.power(self.pc_detector.pcs[:, 2], 2)))
        np.clip(dist, 0, max_d, out=dist)
        # max distance is 120m but usually not usual
        return (((dist - min_d) / (max_d - min_d)) * 120).astype(np.uint8)

    def compute_bf_cost(self, sigma_in):
        """Compute cost for the current tau (extrinsics)"""
        start_t = time.time()
        # Project lidar points
        self.pc_to_pixels()

        # GMM Cost
        cost = 0
        # iterate over lidar edge points
        for idx in range(self.pc_detector.pcs_edge_idxs.shape[0]):
            pt_idx = self.pc_detector.pcs_edge_idxs[idx]

            # check if pixel lands within image bounds
            if self.pixels_mask[pt_idx]:

                # lidar edge weight
                w_i = self.pc_detector.pcs_edge_scores[pt_idx]

                # gaussian parameters
                mu = self.pixels[pt_idx, :]
                sigma = sigma_in / \
                    np.linalg.norm(self.pc_detector.pcs[pt_idx, :])
                cov_mat = np.diag([sigma, sigma])

                # neighborhood params
                min_x = max(0, int(mu[0] - 3*sigma))
                max_x = min(self.img_detector.img_w, int(mu[0] + 3*sigma))
                min_y = max(0, int(mu[1] - 3*sigma))
                max_y = min(self.img_detector.img_h, int(mu[1] + 3*sigma))
                num_ed_pixels = np.sum(
                    self.img_detector.imgs_edges[min_y: max_y, min_x: max_x])

                # iterate over 3-sigma neighborhood
                for x in range(min_x, max_x):
                    for y in range(min_y, max_y):

                        # check if current img pixel is an edge pixel
                        if self.img_detector.imgs_edges[y, x]:
                            w_j = self.img_detector.imgs_edge_scores[y, x]
                            w_ij = 0.5*(w_i + w_j)/num_ed_pixels
                            cost += w_ij * \
                                multivariate_normal.pdf([x, y], mu, cov_mat)
        gc.collect()
        print(f"Brute Force cost computation time:{time.time() - start_t}")
        return cost

    @staticmethod
    def gaussian_pdf(u, v, sigma, mu=0):
        """Compute P(d) according to the 1d gaussian pdf"""
        d = np.sqrt(u**2 + v**2)
        return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(d**2)/(2*(sigma**2)))

    @staticmethod
    def gaussian_pdf_deriv(u, v, sigma, mu=0, wrt='u'):
        d = np.sqrt(u ** 2 + v ** 2)
        if wrt == 'u':
            factor = u
        else:
            factor = v
        return (1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(d**2)/(2*(sigma**2)))*(-factor/sigma)

    def compute_gradient(self, sigma_in):
        """Assuming lidar edge points have already been projected, compute gradient at current tau"""
        print('Computing Gradient')

        # GMM Cost
        gradient = np.zeros(6)
        omega = self.tau[:3]
        jac = jacobian(omega)

        # TODO: Simplify
        angle = np.linalg.norm(omega, 2)
        axis = omega/angle
        quat = pyquat.Quaternion(axis=axis, radians=angle)
        R = quat.rotation_matrix

        # TODO: Simplify
        f_x = self.K[0, 0]
        f_y = self.K[1, 1]
        c_x = self.K[0, 2]
        c_y = self.K[1, 2]

        # iterate over lidar edge points
        for idx in range(self.pc_detector.pcs_edge_idxs.shape[0]):
            pt_idx = self.pc_detector.pcs_edge_idxs[idx]

            # check if pixel lands within image bounds
            if self.pixels_mask[pt_idx]:
                start_time = time.time()
                # lidar edge weight
                w_i = self.pc_detector.pcs_edge_scores[pt_idx]

                # gaussian parameters
                mu = self.pixels[pt_idx, :]
                sigma = sigma_in / \
                    np.linalg.norm(self.pc_detector.pcs[pt_idx, :])
                cov_mat = np.diag([sigma, sigma])

                # neighborhood params
                min_x = max(0, int(mu[0] - 3 * sigma))
                max_x = min(self.img_detector.img_w, int(mu[0] + 3 * sigma))
                min_y = max(0, int(mu[1] - 3 * sigma))
                max_y = min(self.img_detector.img_h, int(mu[1] + 3 * sigma))
                num_ed_pixels = np.sum(
                    self.img_detector.imgs_edges[min_y: max_y, min_x: max_x])

                # iterate over 3-sigma neighborhood
                for x in range(min_x, max_x):
                    for y in range(min_y, max_y):

                        # check if current img pixel is an edge pixel
                        if self.img_detector.imgs_edges[y, x]:

                            w_j = self.img_detector.imgs_edge_scores[y, x]
                            w_ij = 0.5 * (w_i + w_j) / num_ed_pixels

                            M = - \
                                np.dot(
                                    skew(np.dot(self.R, self.pc_detector.pcs[pt_idx])), jac)

                            dxc_dtau = dyc_dtau = dzc_dtau = np.zeros((1, 6))
                            dxc_dtau[0, :3] = M[0, :]
                            dxc_dtau[0, 3] = 1

                            dyc_dtau[0, :3] = M[1, :]
                            dyc_dtau[0, 4] = 1

                            dyc_dtau[0, :3] = M[2, :]
                            dyc_dtau[0, 5] = 1

                            # TODO: Correctly derive gaussian derivative wrt u and v
                            u, v = x - mu[0], y - mu[1]
                            dG_du = self.gaussian_pdf_deriv(u, v, sigma, wrt='u')
                            dG_dv = self.gaussian_pdf_deriv(u, v, sigma, wrt='v')

                            x_c, y_c, z_c = self.pc_detector.pcs[pt_idx]
                            du_dxc = f_x/z_c
                            du_dyc = 0
                            du_dzc = -(f_x*x_c)/(z_c**2)
                            dv_dxc = 0
                            dv_dyc = f_y / z_c
                            dv_dzc = -(f_y*y_c)/(z_c**2)

                            du_dtau = (du_dxc * dxc_dtau) + \
                                      (du_dyc * dyc_dtau) + (du_dzc * dzc_dtau)
                            dv_dtau = (dv_dxc * dxc_dtau) + \
                                      (dv_dyc * dyc_dtau) + (dv_dzc * dzc_dtau)

                            gradient = gradient + \
                                       w_ij*((dG_du*du_dtau) + (dG_dv*dv_dtau))

                # print(f"One projected lidar point gradient component time={time.time()-start_time}")

        return gradient

    def compute_conv_cost(self, sigma_in):
        """Project lidar points onto image using current extrinsics, then compute cost"""
        start_t = time.time()
        self.pc_to_pixels()

        cost_map = np.zeros(self.img_detector.imgs_edge_scores.shape)
        for idx_pc in range(self.pc_detector.pcs_edge_idxs.shape[0]):

            idx = self.pc_detector.pcs_edge_idxs[idx_pc]

            # check if projected pixel lands within image bounds
            if not self.pixels_mask[idx]:
                continue

            # TODO: Use camera frame pointcloud for sigma scaling
            sigma = int(sigma_in /
                        np.linalg.norm(self.pc_detector.pcs[idx, :], 2))

            mu_x, mu_y = self.pixels[idx].astype(np.int)
            # Get gaussian kernel
            # Distance > 3 sigma is set to 0
            # and normalized so that the total Kernel = 1
            # BUG: In getGaussianKernel2D
            gauss2d = getGaussianKernel2D(sigma, False)
            top, bot, left, right = get_boundry(
                self.img_detector.imgs_edge_scores, (mu_y, mu_x), sigma)
            # Get image patch inside the kernel
            edge_scores_patch = \
                self.img_detector.imgs_edge_scores[mu_y - top:mu_y + bot,
                                                   mu_x - left:mu_x + right]

            # weight = (normalized img score + normalized pc score) / 2
            # weight = weight / |Omega_i|
            # Cost = Weight * Gaussian Kernal
            # BUG: Only the pixels that contain values > 0 in the edge_scores_patch
            # should be added self.pc_detector_pcs_edge_scores[idx] to. The 0 pixels
            # should remain 0
            nonzero_idxs = np.argwhere(edge_scores_patch)
            if len(nonzero_idxs) == 0:
                continue


            edge_scores_patch[nonzero_idxs[:, 0], nonzero_idxs[:, 1]] += self.pc_detector.pcs_edge_scores[idx]

            kernel_patch = gauss2d[3 * sigma - top:3 * sigma + bot,
                                   3 * sigma - left:3 * sigma + right]

            cost_patch = np.multiply(edge_scores_patch, kernel_patch)

            # Normalize by number of edge pixels in the neighborhood
            # print(f'Cost: {np.sum(cost_patch)/(2*np.sum(cost_patch>0))}')

            cost_map[mu_y, mu_x] = \
                np.sum(cost_patch) / (2 * np.sum(edge_scores_patch > 0))

        # plot_2d(cost_map)
        gc.collect()
        print(f"Convolution Cost Computation time:{time.time() - start_t}")

        return np.sum(cost_map)

    def optimize(self, sigma_in, max_iters=10):

        iter = 0
        cost_history = []
        learning_rate = 1e-15
        # TODO: Backtracking line learning rate

        while iter < max_iters:

            # Visualize current projection
            if self.visualize:
                self.pc_to_pixels()
                proj_img = self.draw_points()
                cv.imshow('PC Projection', proj_img)
                cv.waitKey(0)

            cost = self.compute_conv_cost(sigma_in)
            cost_history.append(cost)
            start_time = time.time()
            gradient = self.compute_gradient(sigma_in).reshape((6, ))
            self.tau -= learning_rate*gradient
            print(f'Gradient time = {time.time()-start_time}')
            print('hi')
        # TODO: Plot cost over the iterations
