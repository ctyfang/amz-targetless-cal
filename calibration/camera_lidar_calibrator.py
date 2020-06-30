"""Top-level class for Calibration"""

import numpy as np
import cv2 as cv
from pyquaternion import Quaternion
import gc
import itertools as iter
import scipy
from datetime import datetime
import matplotlib.pyplot as plt

from scipy.optimize import least_squares, minimize, basinhopping
from scipy.stats import multivariate_normal, entropy
from scipy.spatial.ckdtree import cKDTree
from KDEpy import FFTKDE

from calibration.img_edge_detector import ImgEdgeDetector
from calibration.pc_edge_detector import PcEdgeDetector
from calibration.utils.data_utils import *
from calibration.utils.img_utils import *
from calibration.utils.pc_utils import *


class CameraLidarCalibrator:

    def __init__(self, cfg, visualize=False, tau_init=None):
        self.visualize = visualize
        self.projected_points = []
        self.points_cam_frame = []
        self.projection_mask = []
        self.K = np.asarray(cfg.K)
        self.R, self.T = load_lid_cal(cfg.calib_dir)
        self.correspondences = []

        if tau_init:
            self.tau = tau_init
        elif isinstance(self.R, np.ndarray) and isinstance(self.T, np.ndarray):
            self.tau = self.transform_to_tau(self.R, self.T)
        else:
            self.tau = np.zeros((1, 6))

        # Load point clouds/images into the detectors
        self.img_detector = ImgEdgeDetector(cfg, visualize=False)
        self.pc_detector = PcEdgeDetector(cfg, visualize=visualize)
        self.num_frames = len(self.img_detector.imgs)

        # Calculate projected_points, points_cam_frame, projection_mask
        self.project_point_cloud()

        # User input of correspondences
        self.select_correspondences()

        # Detect edges
        self.img_detector.img_detect(method=cfg.im_ed_method,
                                     visualize=visualize)
        gc.collect()

        self.pc_detector.pc_detect(self.points_cam_frame,
                                   cfg.pc_ed_score_thr,
                                   cfg.pc_ed_num_nn,
                                   cfg.pc_ed_rad_nn,
                                   visualize=visualize)
        gc.collect()

        if visualize:
            # self.draw_all_points(score=self.pc_detector.pcs_edge_scores)
            self.draw_all_points()
            self.draw_edge_points()
            self.draw_edge_points(score=self.pc_detector.pcs_edge_scores[-1],
                                  image=self.img_detector.img_edge_scores[-1])

        # Optimization parameters
        self.opt_save_every = 10
        self.num_iterations = 0

    def select_correspondences(self):
        self.correspondences = []

        curr_img_name = "gray"
        points_selected = 0
        ref_pt = np.asarray([0, 0], dtype=np.uint)
        ref_pt_3D = np.asarray([0, 0, 0], dtype=np.float32)

        pc_pixel_tree = None
        curr_pc_pixels = None
        curr_pc = None

        def correspondence_cb(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                if curr_img_name == "gray":
                    ref_pt[0], ref_pt[1] = x, y
                else:
                    d, i = pc_pixel_tree.query(
                        np.asarray([x, y]).reshape((1, 2)), 1)
                    nearest_pixel = curr_pc_pixels[i, :].astype(np.uint)[0]
                    ref_pt[0], ref_pt[1] = nearest_pixel[0], nearest_pixel[1]
                    ref_pt_3D[:] = curr_pc[i, :]

        for frame_idx in range(self.num_frames):
            curr_pc = self.pc_detector.pcs[frame_idx]
            curr_pc_pixels = self.projected_points[frame_idx]
            pc_pixel_tree = cKDTree(curr_pc_pixels)

            img_gray = cv.cvtColor(self.img_detector.imgs[frame_idx],
                                   cv.COLOR_BGR2GRAY)
            img_synthetic = gen_synthetic_image(curr_pc,
                             self.pc_detector.reflectances[frame_idx],
                             self.R, self.T, self.K,
                             (self.img_detector.img_h,
                             self.img_detector.img_w))

            cv.namedWindow("Correspondences")
            cv.setMouseCallback("Correspondences", correspondence_cb)

            gray_pixels = []
            lidar_pixels = []
            lidar_points = []
            while True:
                if points_selected % 2 == 0:
                    curr_img = img_gray
                    curr_img_name = "gray"
                else:
                    curr_img = img_synthetic
                    curr_img_name = "synthetic"

                curr_img = np.repeat(np.expand_dims(curr_img, axis=2), 3, axis=2)
                curr_img = cv.circle(curr_img, tuple(ref_pt), radius=2,
                                     color=(255, 0, 0),
                                     thickness=-1)
                cv.imshow("Correspondences", curr_img)
                key = cv.waitKey(1)

                if key == ord('y'):
                    points_selected += 1
                    if curr_img_name == "synthetic":
                        lidar_pixels.append(ref_pt.copy())
                        lidar_points.append(ref_pt_3D.copy())
                    else:
                        gray_pixels.append(ref_pt.copy())

                elif key == ord('q'):
                    if points_selected % 2 == 0:
                        break
                    else:
                        print("Uneven number of points. Select one more.")

            gray_pixels = np.asarray(gray_pixels)
            lidar_pixels = np.asarray(lidar_pixels)
            lidar_points = np.asarray(lidar_points)

            if self.visualize:
                cv.imshow("Matches", draw_point_matches(img_gray, gray_pixels,
                                     img_synthetic, lidar_pixels))
                cv.waitKey(0)
                cv.destroyAllWindows()

            self.correspondences.append((gray_pixels,
                                         lidar_points))

    def update_extrinsics(self, tau_new):
        """Given new extrinsics, update rotation matrices and translation vec"""
        R, T = self.tau_to_transform(tau_new)
        self.R, self.T = R, T
        self.tau = tau_new

    @staticmethod
    def transform_to_tau(R, T):
        r_vec, _ = cv2.Rodrigues(R)
        return np.hstack((r_vec.T, T.T)).reshape(6,)

    @staticmethod
    def tau_to_transform(tau):
        tau = np.squeeze(tau)
        R, _ = cv2.Rodrigues(tau[:3])
        T = tau[3:].reshape((3, 1))
        return R, T

    def project_point_cloud(self):
        '''
        Transform all points of the point cloud into the camera frame and then
        projects all points to the image plane. Also return a binary mask to 
        obtain all points with a valid projection.
        '''
        # Compute R and T from current tau
        self.R, self.T = self.tau_to_transform(self.tau)

        # Remove previous projection
        self.points_cam_frame = []
        self.projected_points = []
        self.projection_mask = []

        for pc in self.pc_detector.pcs:
            one_mat = np.ones((pc.shape[0], 1))
            point_cloud = np.concatenate((pc, one_mat), axis=1)

            # Transform points into the camera frame
            self.points_cam_frame.append(
                np.matmul(np.hstack((self.R, self.T)), point_cloud.T).T)

            # Project points into image plane and normalize
            projected_points = np.dot(self.K, self.points_cam_frame[-1].T)
            projected_points = projected_points[::] / projected_points[::][-1]
            projected_points = np.delete(projected_points, 2, axis=0)
            self.projected_points.append(projected_points.T)

            # Remove points that were behind the camera
            # self.points_cam_frame = self.points_cam_frame.T
            in_front_of_camera_mask = self.points_cam_frame[-1][:, 2] > 0

            # Remove projected points that are outside of the image
            inside_mask_x = np.logical_and(
                (projected_points.T[:, 0] >= 0),
                (projected_points.T[:, 0] <= self.img_detector.img_w))
            inside_mask_y = np.logical_and(
                (projected_points.T[:, 1] >= 0),
                (projected_points.T[:, 1] <= self.img_detector.img_h))
            inside_mask = np.logical_and(inside_mask_x, inside_mask_y)

            # Final projection mask
            self.projection_mask.append(
                np.logical_and(inside_mask, in_front_of_camera_mask))

    def draw_all_points(self, score=None, img=None, frame=-1, show=False):
        """
        Draw all points within corresponding camera's FoV on image provided.
        """
        if img is None:
            image = self.img_detector.imgs[frame].copy()
        else:
            image = img

        colors = self.scalar_to_color(score=score, frame=frame)
        colors_valid = colors[self.projection_mask[frame]]

        projected_points_valid = self.projected_points[frame][
            self.projection_mask[frame]]

        for pixel, color in zip(projected_points_valid, colors_valid):
            cv2.circle(image,
                       (pixel[0].astype(np.int), pixel[1].astype(np.int)), 1,
                       color.tolist(), -1)
            # image[pixel[1].astype(np.int), pixel[0].astype(np.int), :] = color

        if show:
            cv.imshow('Projected Point Cloud on Image', image)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return image

    def draw_reflectance(self, frame=-1):
        """Given frame, draw reflectance image"""
        img_h, img_w = self.img_detector.imgs[frame].shape[:2]
        refl_img = np.zeros((img_h, img_w), dtype=np.float32)

        projected_points_valid = self.projected_points[frame][
            self.projection_mask[frame]]
        reflectance_values = self.pc_detector.reflectances[frame][
            self.projection_mask[frame]]

        for pixel, reflectance in zip(projected_points_valid,
                                      reflectance_values):
            refl_img[pixel[1].astype(np.int),
                     pixel[0].astype(np.int)] = reflectance

        cv.imshow('Projected Point Cloud Reflectance Image', refl_img)
        cv.imshow('Grayscale img',
                  cv.cvtColor(self.img_detector.imgs[frame], cv.COLOR_BGR2GRAY))
        cv.waitKey(0)
        cv.destroyAllWindows()

    def draw_depth_image(self, frame=-1, show=False):
        img_h, img_w = self.img_detector.imgs[frame].shape[:2]
        image = np.zeros((img_h, img_w), dtype=np.float32)

        grid_x, grid_y = np.mgrid[0:img_w, 0:img_h]

        depth = np.linalg.norm(self.pc_detector.pcs[frame], ord=2, axis=1)
        depth_valid = depth[self.projection_mask[frame]]

        # depth_valid = self.pc_detector.reflectances[frame][
        #     self.projection_mask[frame]] * 255
        projected_points_valid = self.projected_points[frame][
            self.projection_mask[frame]]

        depth_img = griddata(projected_points_valid,
                             depth_valid, (grid_x, grid_y),
                             method='linear').T

        depth_img = (depth_img * 255 / np.nanmax(depth_img)).astype(np.uint8)
        if show:
            cv.imshow('Depth image with linear interpolation', depth_img)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return depth_img

    def draw_edge_points_binary(self,
                                 frame=-1,
                                 show=False):
        """
        Draw only edge points within corresponding camera's FoV.
        """

        image = np.zeros((self.img_detector.img_h, self.img_detector.img_w),
                         dtype=np.bool)

        projected_points_valid = self.projected_points[frame][np.logical_and(
            self.projection_mask[frame],
            self.pc_detector.pcs_edge_masks[frame])]

        for pixel in projected_points_valid:
            image[pixel[1].astype(np.int), pixel[0].astype(np.int)] = True

        if show:
            cv.imshow('Projected Edge Points on Image', image)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return image

    def draw_edge_points(self,
                         score=None,
                         image=None,
                         append_string='',
                         frame=-1,
                         save=False,
                         show=False):
        """
        Draw only edge points within corresponding camera's FoV on image provided.
        """

        if image is None:
            image = self.img_detector.imgs[frame].copy()
        else:
            image = (image.copy() * 255).astype(np.uint8)
            image = np.dstack((image, image, image))

        colors = self.scalar_to_color(frame=frame)
        colors_valid = colors[np.logical_and(
            self.projection_mask[frame],
            self.pc_detector.pcs_edge_masks[frame])]

        projected_points_valid = self.projected_points[frame][np.logical_and(
            self.projection_mask[frame],
            self.pc_detector.pcs_edge_masks[frame])]

        for pixel, color in zip(projected_points_valid, colors_valid):
            image[pixel[1].astype(np.int), pixel[0].astype(np.int), :] = color

        if save:
            now = datetime.now()
            cv.imwrite(
                append_string + now.strftime("%y%m%d-%H%M%S-%f") + '.jpg',
                image)

        if show:
            cv.imshow('Projected Edge Points on Image', image)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return image

    def scalar_to_color(self, score=None, min_d=0, max_d=60, frame=-1):
        """
        print Color(HSV's H value) corresponding to score
        """
        if score is None:
            score = np.sqrt(
                np.power(self.points_cam_frame[frame][:, 0], 2) +
                np.power(self.points_cam_frame[frame][:, 1], 2) +
                np.power(self.points_cam_frame[frame][:, 2], 2))

        np.clip(score, 0, max_d, out=score)
        # max distance is 120m but usually not usual

        norm = plt.Normalize()
        colors = plt.cm.jet(norm(score))

        return (colors[:, :3] * 255).astype(np.uint8)

    def draw_points(self, image=None, FULL=True):
        """
        Draw points within corresponding camera's FoV on image provided.
        If no image provided, points are drawn on an empty(black) background.
        """

        if image is not None:
            image = np.uint8(np.dstack((image, image, image))) * 255
            cv.imshow('Before projection', image)
            cv.waitKey(0)

            hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        else:
            hsv_image = np.zeros(self.img_detector.imgs.shape).astype(np.uint8)

        color = self.pc_to_colors()
        if FULL:
            index = range(self.projected_points.shape[0])
        else:
            index = np.random.choice(self.projected_points.shape[0],
                                     size=int(self.projected_points.shape[0] /
                                              10),
                                     replace=False)
        for i in index:
            if pc[i, 0] < 0:
                continue
            if self.projection_mask[i] is False:
                continue

            cv.circle(hsv_image, (np.int32(self.projected_points[i, 0]),
                                  np.int32(self.projected_points[i, 1])), 1,
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

    @staticmethod
    def gaussian_pdf(u, v, sigma, mu=0):
        """Compute P(d) according to the 1d gaussian pdf"""
        d = np.sqrt(u**2 + v**2)
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(d**2) /
                                                           (2 * (sigma**2)))

    @staticmethod
    def gaussian_pdf_deriv(u, v, sigma, mu=0, wrt='u'):
        d = np.sqrt(u**2 + v**2)
        if wrt == 'u':
            factor = u
        else:
            factor = v
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
            -(d**2) / (2 * (sigma**2))) * (-factor / sigma)

    def compute_gradient(self, sigma_in):
        """Assuming lidar edge points have already been projected, compute gradient at current tau
           Also assumes tau has already been converted to rot mat and trans vec."""

        # GMM Cost
        gradient = np.zeros(6)
        omega = self.tau[:3]
        jac = jacobian(omega)

        # TODO: Simplify
        f_x = self.K[0, 0]
        f_y = self.K[1, 1]

        # iterate over lidar edge points
        for idx in range(self.pc_detector.pcs_edge_idxs.shape[0]):
            pt_idx = self.pc_detector.pcs_edge_idxs[idx]

            # check if projected point lands within image bounds
            if self.projection_mask[pt_idx]:
                # lidar edge weight
                w_i = self.pc_detector.pcs_edge_scores[pt_idx]

                # gaussian parameters
                mu = self.projected_points[pt_idx, :]
                # sigma = sigma_in / np.linalg.norm(self.pc_detector.pcs[pt_idx, :])
                sigma = sigma_in

                # neighborhood params
                min_x = max(0, int(mu[0] - 3 * sigma))
                max_x = min(self.img_detector.img_w, int(mu[0] + 3 * sigma))
                min_y = max(0, int(mu[1] - 3 * sigma))
                max_y = min(self.img_detector.img_h, int(mu[1] + 3 * sigma))
                num_ed_projected_points = np.sum(
                    self.img_detector.imgs_edges[min_y:max_y, min_x:max_x])

                # iterate over 3-sigma neighborhood
                for x in range(min_x, max_x):
                    for y in range(min_y, max_y):

                        # check if current img pixel is an edge pixel
                        if self.img_detector.imgs_edges[y, x]:
                            w_j = self.img_detector.img_edge_scores[y, x]
                            w_ij = 0.5 * (w_i + w_j) / num_ed_projected_points

                            M = -np.dot(
                                skew(
                                    np.dot(self.R,
                                           self.pc_detector.pcs[pt_idx])), jac)

                            dxc_dtau = dyc_dtau = dzc_dtau = np.zeros((1, 6))
                            dxc_dtau[0, :3] = M[0, :]
                            dxc_dtau[0, 3] = 1

                            dyc_dtau[0, :3] = M[1, :]
                            dyc_dtau[0, 4] = 1

                            dyc_dtau[0, :3] = M[2, :]
                            dyc_dtau[0, 5] = 1

                            u, v = abs(x - mu[0]), abs(y - mu[1])
                            dG_du = self.gaussian_pdf_deriv(u,
                                                            v,
                                                            sigma,
                                                            wrt='u')
                            dG_dv = self.gaussian_pdf_deriv(u,
                                                            v,
                                                            sigma,
                                                            wrt='v')

                            x_c, y_c, z_c = self.points_cam_frame[pt_idx, :]
                            du_dxc = f_x / z_c
                            du_dyc = 0
                            du_dzc = -(f_x * x_c) / (z_c**2)
                            dv_dxc = 0
                            dv_dyc = f_y / z_c
                            dv_dzc = -(f_y * y_c) / (z_c**2)

                            du_dtau = ((du_dxc * dxc_dtau) +
                                       (du_dyc * dyc_dtau) +
                                       (du_dzc * dzc_dtau))
                            dv_dtau = ((dv_dxc * dxc_dtau) +
                                       (dv_dyc * dyc_dtau) +
                                       (dv_dzc * dzc_dtau))

                            gradient = gradient + \
                                       w_ij * ((dG_du * du_dtau) + (
                                        dG_dv * dv_dtau))
        return -gradient

    def compute_mi_cost(self, frame=-1):
        """Compute mutual info cost for one frame"""
        self.project_point_cloud()
        grayscale_img = cv.cvtColor(self.img_detector.imgs[frame],
                                    cv.COLOR_BGR2GRAY)
        projected_points_valid = self.projected_points[frame] \
                                 [self.projection_mask[frame], :]

        grayscale_vector = np.expand_dims(
            grayscale_img[projected_points_valid[:, 1].astype(np.uint),
                          projected_points_valid[:, 0].astype(np.uint)], 1)
        reflectance_vector = np.expand_dims(
            (self.pc_detector.reflectances[frame][self.projection_mask[frame]] *
             255.0), 1).astype(np.int)

        if len(reflectance_vector) > 0 and len(grayscale_vector) > 0:

            joint_data = np.hstack([grayscale_vector, reflectance_vector])
            intensity_vector = np.linspace(0, 255, 510)
            grid_x, grid_y = np.meshgrid(intensity_vector,
                                         intensity_vector)
            grid_data = np.vstack([grid_y.ravel(), grid_x.ravel()])
            grid_data = grid_data.T

            gray_probs = FFTKDE(
                bw='silverman').fit(grayscale_vector).evaluate(intensity_vector)

            refl_probs = FFTKDE(
                bw='silverman').fit(reflectance_vector).evaluate(intensity_vector)
            joint_probs = FFTKDE().fit(joint_data).evaluate(grid_data)

            gray_probs /= np.sum(gray_probs)
            refl_probs /= np.sum(refl_probs)

            joint_probs /= np.sum(joint_probs)
            mi_cost = entropy(gray_probs) + \
                      entropy(refl_probs) - entropy(joint_probs)
            mi_cost = mi_cost

        else:
            mi_cost = 0
        return -mi_cost

    def compute_conv_cost(self, sigma_in, frame=-1, sigma_scaling=True):
        """Compute cost"""
        # start_t = time.time()
        cost_map = np.zeros(self.img_detector.img_edge_scores[frame].shape)
        for idx_pc in range(self.pc_detector.pcs_edge_idxs[frame].shape[0]):

            idx = self.pc_detector.pcs_edge_idxs[frame][idx_pc]

            # check if projected projected point lands within image bounds
            if not self.projection_mask[frame][idx]:
                continue

            # TODO: Use camera frame pointcloud for sigma scaling
            if sigma_scaling:
                sigma = (
                    sigma_in /
                    np.linalg.norm(self.points_cam_frame[frame][idx, :], 2))
            else:
                sigma = sigma_in

            mu_x, mu_y = self.projected_points[frame][idx].astype(np.int)
            # Get gaussian kernel
            # Distance > 3 sigma is set to 0
            # and normalized so that the total Kernel = 1
            gauss2d = getGaussianKernel2D(sigma, False)
            top, bot, left, right = get_boundry(
                self.img_detector.img_edge_scores[frame], (mu_y, mu_x),
                int(sigma))
            # Get image patch inside the kernel
            edge_scores_patch = \
                self.img_detector.img_edge_scores[frame][
                mu_y - top:mu_y + bot,
                mu_x - left:mu_x + right
                ].copy()

            # weight = (normalized img score + normalized pc score) / 2
            # weight = weight / |Omega_i|
            # Cost = Weight * Gaussian Kernel
            num_nonzeros = np.sum(edge_scores_patch != 0)
            if num_nonzeros == 0:
                continue

            edge_scores_patch[edge_scores_patch != 0] += \
                self.pc_detector.pcs_edge_scores[frame][idx]

            kernel_patch = gauss2d[3 * int(sigma) - top:3 * int(sigma) + bot,
                                   3 * int(sigma) - left:3 * int(sigma) + right]

            cost_patch = np.multiply(edge_scores_patch, kernel_patch)

            # Normalize by number of edge projected_points in the neighborhood
            cost_map[mu_y, mu_x] = \
                np.sum(cost_patch) / (2 * np.sum(edge_scores_patch > 0))

        # plot_2d(cost_map)
        gc.collect()
        return -np.sum(cost_map)

    def compute_corresp_cost(self, norm="L2"):
        """Return average distance between all correspondences"""
        pixel_distances = []
        num_corresp = 0
        dist_offset = np.sqrt(self.img_detector.img_w**2 +
                              self.img_detector.img_h**2)*3

        for matches in self.correspondences:
            gray_pixels = matches[0]
            lidar_points = matches[1]
            lidar_points_cam = np.matmul(self.R, lidar_points.T) + self.T
            lidar_pixels_homo = (np.matmul(self.K, lidar_points_cam).T)
            lidar_pixels_homo = lidar_pixels_homo / \
                                np.expand_dims(lidar_pixels_homo[:, 2], axis=1)
            lidar_pixels = lidar_pixels_homo[:, :2]

            pixel_diff = gray_pixels - lidar_pixels
            pixel_distances += np.linalg.norm(pixel_diff, axis=1, ord=1).tolist()
            num_corresp += lidar_pixels.shape[0]

        total_dist = 0
        for dist in pixel_distances:
            if dist <= 5:
                total_dist += dist
            else:
                total_dist += (dist**2)
        average_dist = total_dist/num_corresp
        return -dist_offset + 3*average_dist

    def compute_chamfer_dists(self):
        total_dist = 0
        total_edge_pts = 0
        for frame_idx in range(self.num_frames):
            cam_edges = self.img_detector.imgs_edges[frame_idx]
            cam_edges_inv = 255*np.logical_not(cam_edges).astype(np.uint8)
            cam_dist_map = cv.distanceTransform(cam_edges_inv, cv.DIST_L2,
                                                cv.DIST_MASK_PRECISE)
            lid_edges = self.draw_edge_points_binary(frame_idx)
            num_edge_pts = lid_edges.sum()
            dist = np.multiply(lid_edges, cam_dist_map).sum()

            total_dist += dist
            total_edge_pts += num_edge_pts

        return total_dist/total_edge_pts

    def compute_points_cost(self, frame=-1):
        """Penalize the deviation from the initial number of points for
        each frame."""
        num_points = self.projection_mask[frame].sum()
        points_diff = abs(self.numpoints_preopt[frame] - num_points)
        return points_diff

    def ls_optimize(self, hyperparams, maxiter=600, save_every=100):
        """Optimize cost over all image-scan pairs."""
        cost_history = []

        # Store initial number of points for points-based cost
        self.numpoints_preopt = [np.sum(self.projection_mask[i])
                                 for i in range(self.num_frames)]

        # Generate initial simplex for Nelder-Mead
        initial_deltas = [0.10, 0.10, 0.10, 1.0, 1.0, 1.0]
        opt_options = {'disp': True, 'maxiter': maxiter, 'adaptive': True,
                       'initial_simplex': get_mixed_delta_simplex(self.tau,
                                                                  initial_deltas,
                                                                  scales=hyperparams['scales'])}
        self.num_iterations = 0
        self.opt_save_every = save_every

        def loss_callback(xk, state=None):
            """Save loss graph and point-cloud projection for debugging"""
            self.num_iterations += 1
            if len(cost_history):
                plt.close('all')
                plt.figure()
                plt.plot(cost_history)
                plt.savefig('current_loss.png')

            if self.num_iterations % self.opt_save_every == 0:
                img = self.draw_all_points()
                cv.imwrite('current_projection.jpg', img)
            return False

        update_deltas = [0.10, 0.10, 0.10, 0.5, 0.5, 0.5]
        def bh_callback(x, f, accepted):
            print(f"Minimum found at x: {x}, f: {f}. Accepted: {accepted}")

            if accepted:
                print(f"Modifying simplex")
                opt_options['initial_simplex'] = \
                    get_mixed_delta_simplex(np.multiply(x,
                                            hyperparams['scales']),
                                            update_deltas,
                                            scales=hyperparams['scales'])
            return False

        # Compute cost
        start = time.time()
        tau_scaled = np.divide(self.tau, hyperparams['scales'])

        # Nelder-Mead
        # opt_results = minimize(loss,
        #                        tau_scaled,
        #                        method='Nelder-Mead',
        #                        args=(self, hyperparams, cost_history, False),
        #                        options=opt_options,
        #                        callback=loss_callback)
        # self.tau = np.multiply(opt_results.x, hyperparams['scales'])

        # Basin-Hopping
        step_vector = [0.10, 0.10, 0.10, 1.0, 1.0, 1.0]
        opt_results = basinhopping(loss, tau_scaled, 10, T=1.5,
                                   niter_success=3,
                                   disp=True, callback=bh_callback,
                                   take_step=RandomDisplacement(step_vector),
                                   minimizer_kwargs={'method': 'Nelder-Mead',
                                                     'args': (self, hyperparams,
                                                             cost_history,
                                                             False),
                                                     'callback': loss_callback,
                                                     'options': opt_options})
        self.tau = np.multiply(opt_results.lowest_optimization_result.x,
                               hyperparams['scales'])

        print(f"NL optimizer time={time.time() - start}")
        return self.tau, cost_history


    def batch_optimization(self, sigma_in=6):
        # cost_history = []
        self.num_iterations = 0
        def loss(tau_init, calibrator, sigma, cost_history):
            local_cost = []
            calibrator.tau = tau_init
            calibrator.project_point_cloud()
            # print(len(calibrator.projected_points))
            for i in range(len(calibrator.img_detector.imgs)):
                hm_ptc = calibrator.compute_heat_map(
                    sigma=sigma, frame=i, ptCloud=True)
                hm_img = calibrator.compute_heat_map(
                    sigma=sigma, frame=i, ptCloud=False)
                diff = hm_img - hm_ptc
                # cost = -np.linalg.norm(diff, ord=2)
                local_cost.append(np.linalg.norm(diff, ord=2))

            cost_history.append(np.sum(local_cost))

            if self.num_iterations % 100 == 0:
                img = self.draw_all_points(frame=0)
                cv.imwrite('generated/'+ str(self.num_iterations) + '.jpg', img)
            self.num_iterations += 1
            print((cost_history[-1]))

            return cost_history[-1]

        start = time.time()
        threshold = 0.05
        err = np.random.uniform(-threshold, threshold, (6,))
        self.tau += err
        while sigma_in > 1:
            cost_history = []
            opt_results = minimize(loss,
                                self.tau + err,
                                method='Nelder-Mead',
                                args=(
                                self, sigma_in, cost_history))#,
                                #options=opt_options,
                                #callback=loss_callback)
            sigma_in -= 3
            cost_history = np.array(cost_history)
            print(cost_history.shape)

            plt.plot(range(len(cost_history)), cost_history)           

        print(f"Batch optimizer time={time.time() - start}")

        # fig, ax = plt.subplots(len(self.img_detector.imgs))
        # # sys.exit()
        # for i in range(len(ax)):
        #     ax[i].plot(range(len(cost_history)), cost_history[:, i])
        plt.show()
        img = self.draw_all_points(frame=0)
        cv.imwrite('generated/'+ str(self.num_iterations) + '.jpg', img)
        self.tau = opt_results.x
        return opt_results.x

    def compute_heat_map(self,
                         sigma,
                         frame=-1,
                         ptCloud=False,
                         show=False):
        """Compute heat map"""
        # start_t = time.time()
        cost_map = np.zeros(self.img_detector.img_edge_scores[frame].shape)
        gauss2d = getGaussianKernel2D(sigma, False)
        if ptCloud:
            for idx_pc in range(self.pc_detector.pcs_edge_idxs[frame].shape[0]):

                idx = self.pc_detector.pcs_edge_idxs[frame][idx_pc]

                # check if projected projected point lands within image bounds
                if not self.projection_mask[frame][idx]:
                    continue

                mu_x, mu_y = self.projected_points[frame][idx].astype(np.int)
                # Get gaussian kernel
                # Distance > 3 sigma is set to 0
                # and normalized so that the total Kernel = 1

                top, bot, left, right = get_boundry(
                    self.img_detector.img_edge_scores[frame], (mu_y, mu_x),
                    int(sigma))

                kernel_patch = gauss2d[3 * int(sigma) - top:3 * int(sigma) +
                                       bot, 3 * int(sigma) -
                                       left:3 * int(sigma) + right].copy()

                # Normalize by number of edge projected_points in the neighborhood
                cost_map[mu_y - top:mu_y + bot,
                         mu_x - left:mu_x + right] += kernel_patch
            threshold = np.amax(gauss2d)
            cost_map[cost_map < threshold] = 0
            if show:
                plot_2d(cost_map,
                        figname='generated/heat_map_ptc_{}'.format(frame))
            gc.collect()
            return cost_map

        for mu_x in range(self.img_detector.imgs_edges[frame].shape[1]):
            for mu_y in range(self.img_detector.imgs_edges[frame].shape[0]):
                # mu_x, mu_y = self.projected_points[frame][idx].astype(np.int)
                if not self.img_detector.imgs_edges[frame][mu_y, mu_x]:
                    continue

                top, bot, left, right = get_boundry(
                    self.img_detector.img_edge_scores[frame], (mu_y, mu_x),
                    int(sigma))

                kernel_patch = gauss2d[3 * int(sigma) - top:3 * int(sigma) +
                                       bot, 3 * int(sigma) -
                                       left:3 * int(sigma) + right].copy()

                cost_map[mu_y - top:mu_y + bot,
                         mu_x - left:mu_x + right] += kernel_patch

        threshold = np.amax(gauss2d)
        cost_map[cost_map < threshold] = 0
        if show:
            plot_2d(cost_map, figname='generated/heat_map_img_{}'.format(frame))
        gc.collect()
        return cost_map


def loss(tau_scaled, calibrator, hyperparams, cost_history,
         return_components=False):
    """Compute loss for given tau with hyperparams dict. If return components
    is true, returns list of components instead of their sum. Hyperparams dict
    must have the keys ['alphas', 'scales'], where hyperparams['alphas']
    contains the coefficients for each component
    ['mi', 'gmm', 'points', 'corr', 'sigma'],

    hyperparams['scales'] describes how each parameter axis is
    normalized in the optimization."""

    # Rescale Extrinsics
    tau = np.multiply(tau_scaled, hyperparams['scales'])
    calibrator.update_extrinsics(tau)
    calibrator.project_point_cloud()

    # Compute loss components
    cost_components = np.zeros((4, 1))
    for frame_idx in range(calibrator.num_frames):
        if hyperparams['alphas']['mi']:
            cost_components[0] += calibrator.compute_mi_cost(frame_idx)

        if hyperparams['alphas']['gmm']:
            cost_components[1] += calibrator.compute_conv_cost(hyperparams['alphas']['sigma'],
                                                               frame_idx,
                                                               sigma_scaling=False)

        if hyperparams['alphas']['points']:
            cost_components[2] += calibrator.compute_points_cost(frame_idx)

    if hyperparams['alphas']['corr']:
        cost_components[3] += calibrator.compute_corresp_cost()

    # Scale loss components
    cost_components[0] *= hyperparams['alphas']['mi']
    cost_components[1] *= hyperparams['alphas']['gmm']
    cost_components[2] *= hyperparams['alphas']['points']
    cost_components[3] *= hyperparams['alphas']['corr']

    total_cost = sum(cost_components)
    cost_history.append(total_cost)

    print(cost_components)
    if return_components:
        return cost_components
    else:
        return sum(cost_components)


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape[:2]
    img1 = img1.copy()
    img2 = img2.copy()
    dists1 = []
    dists2 = []
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


from scipy._lib._util import check_random_state
class RandomDisplacement(object):
    """
    Add a random displacement of maximum size `stepsize` to each coordinate
    Calling this updates `x` in-place.
    Parameters
    ----------
    stepsize : float, optional
        Maximum stepsize in any dimension
    random_gen : {None, `np.random.RandomState`, `np.random.Generator`}
        The random number generator that generates the displacements
    """
    def __init__(self, stepsize, random_gen=None):
        self.stepsize = stepsize
        self.random_gen = check_random_state(random_gen)

    def __call__(self, x):
        for i in range(len(x)):
            x[i] *= (1+self.random_gen.uniform(-self.stepsize[i],
                                               self.stepsize[i]))
        return x
