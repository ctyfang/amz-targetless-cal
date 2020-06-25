"""Top-level class for Calibration"""
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from pyquaternion import Quaternion
import gc
import itertools as iter
import scipy
from datetime import datetime

from scipy.ndimage.filters import gaussian_filter, convolve
from scipy.optimize import least_squares, minimize, root, differential_evolution, basinhopping
from scipy.stats import multivariate_normal, gaussian_kde, entropy, norm
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
from scipy.linalg import expm
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import scipy.optimize as optimize
from KDEpy import FFTKDE
from scipy.interpolate import griddata

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

        if tau_init:
            self.tau = tau_init
        elif isinstance(self.R, np.ndarray) and isinstance(self.T, np.ndarray):
            self.tau = self.transform_to_tau(self.R, self.T)
        else:
            self.tau = np.zeros((1, 6))

        # Load point clouds/images into the detectors
        self.img_detector = ImgEdgeDetector(cfg, visualize=False)
        self.pc_detector = PcEdgeDetector(cfg, visualize=visualize)

        # Calculate projected_points, points_cam_frame, projection_mask
        self.project_point_cloud()

        # Detect edges
        self.img_detector.img_detect(method=cfg.pc_ed_method,
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

    @staticmethod
    def transform_to_tau(R, T):
        r_vec, _ = cv2.Rodrigues(R)
        return np.hstack((r_vec.T, T.T)).reshape(6,)

    @staticmethod
    def tau_to_transform(tau):
        R, _ = cv2.Rodrigues(tau[:3])
        T = tau[3:].reshape((3, 1))
        return R, T

    @staticmethod
    def tau_to_tauquat(tau):
        tau_quat = np.zeros((7,))
        # test = Quaternion(axis=tau[:3]/np.linalg.norm(tau[:3], 2),
        #                           angle=np.linalg.norm(tau[:3], 2))
        tau_quat[:4] = Quaternion(axis=tau[:3] / np.linalg.norm(tau[:3], 2),
                                  angle=np.linalg.norm(tau[:3], 2)).elements
        tau_quat[4:] = tau[3:]
        return tau_quat

    @staticmethod
    def tauquat_to_tau(tau_quat):
        quat = Quaternion(tau_quat[:4])
        rot_vec = quat.angle * quat.axis
        tau = np.zeros((6,))
        tau[:3] = rot_vec
        tau[3:] = tau_quat[4:]
        return tau

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

            # TODO: Perform transform without homogeneous term,
            #       if too memory intensive

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

    def compute_bf_cost(self, sigma_in):
        """Compute cost for the current tau (extrinsics)"""
        start_t = time.time()

        # GMM Cost
        cost = 0
        # iterate over lidar edge points
        for idx in range(self.pc_detector.pcs_edge_idxs.shape[0]):
            pt_idx = self.pc_detector.pcs_edge_idxs[idx]

            # check if projected point lands within image bounds
            if self.projection_mask[pt_idx]:

                # lidar edge weight
                w_i = self.pc_detector.pcs_edge_scores[pt_idx]

                # gaussian parameters
                mu = self.projected_points[pt_idx, :]
                sigma = sigma_in / \
                        np.linalg.norm(self.pc_detector.pcs[pt_idx, :])
                cov_mat = np.diag([sigma, sigma])

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
                            cost += w_ij * \
                                    multivariate_normal.pdf([x, y], mu, cov_mat)
        gc.collect()
        # print(f"Brute Force cost computation time:{time.time() - start_t}")
        return -cost

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
        grayscale_img = cv.cvtColor(self.img_detector.imgs[frame],
                                    cv.COLOR_BGR2GRAY)
        projected_points_valid = self.projected_points[frame] \
            [self.projection_mask[frame]]
        # print(f"Num valid points: {projected_points_valid.shape[0]}")
        grayscale_vector = np.expand_dims(
            grayscale_img[projected_points_valid[:, 1].astype(np.uint),
                          projected_points_valid[:, 0].astype(np.uint)], 1)
        reflectance_vector = np.expand_dims(
            (self.pc_detector.reflectances[frame][self.projection_mask[frame]] *
             255.0), 1).astype(np.int)

        if len(reflectance_vector) > 0 and len(grayscale_vector) > 0:

            joint_data = np.hstack([grayscale_vector, reflectance_vector])
            grid_x, grid_y = np.meshgrid(range(-1, 257), range(-1, 257))
            grid_data = np.vstack([grid_y.ravel(), grid_x.ravel()])

            # # Using KDEpy
            gray_probs = FFTKDE(bw='silverman').fit(grayscale_vector).evaluate(
                range(-1, 257))
            refl_probs = FFTKDE(
                bw='silverman').fit(reflectance_vector).evaluate(range(-1, 257))
            joint_probs = FFTKDE().fit(joint_data).evaluate(grid_data.T)

            gray_probs /= np.sum(gray_probs)
            refl_probs /= np.sum(refl_probs)
            joint_probs /= np.sum(joint_probs)
            mi_cost = entropy(gray_probs) + \
                      entropy(refl_probs) - entropy(joint_probs)
            mi_cost = max(0.0, mi_cost)
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

    def ls_optimize(self,
                    sigma_in,
                    method='lm',
                    alpha_gmm=1,
                    alpha_mi=8e2,
                    maxiter=600,
                    save_every=100):
        """Optimize cost over all image-scan pairs using mutual info and gmm.
            Scale the contributions from two loss sources using alphas."""
        cost_history = []
        """Optimization config"""
        self.tau_ord_mags = np.log10(np.abs(self.tau))
        self.tau_ord_mags[3:] += 1
        self.tau_ord_mags = np.power(10 * np.ones(self.tau.shape),
                                     self.tau_ord_mags)

        opt_options = {'disp': True, 'maxiter': maxiter, 'adaptive': True}
        self.num_iterations = 0
        self.tau_preoptimize = self.tau
        self.opt_save_every = save_every

        def loss_callback(xk, state=None):
            # print(xk*self.tau_ord_mags)
            self.num_iterations += 1
            """Monitor number of points being projected onto the image"""
            total_valid_points = 0
            for frame_idx in range(len(self.projection_mask)):
                total_valid_points += np.sum(self.projection_mask[frame_idx])

            if total_valid_points < 10000:
                raise BadProjection

            if self.num_iterations % self.opt_save_every == 0:
                img = self.draw_all_points()
                cv.imwrite(str(self.num_iterations) + '.jpg', img)

            return False

        def bh_callback(x, f, accepted):
            self.num_iterations = 0
            print(f"at minimum {f} accepted {np.multiply(x, self.tau_ord_mags)}"
                  f"?"
                  f" {accepted}")

        """Re-try optimization until final projection lands on image"""
        optim_successful = False
        start = time.time()
        while not optim_successful:
            print('Optimizing over all extrinsics...')
            try:
                self.tau = np.divide(self.tau, self.tau_ord_mags)
                # opt_results = minimize(loss_scaled,
                #                        self.tau,
                #                        method='Nelder-Mead',
                #                        args=(
                #                        self, sigma_in, alpha_mi, alpha_gmm,
                #                        cost_history, None, False),
                #                        options=opt_options,
                #                        callback=loss_callback)

                opt_results = basinhopping(loss_scaled,
                                           self.tau,
                                           niter=20,
                                           T=10,
                                           stepsize=0.05,
                                           callback=bh_callback,
                                           minimizer_kwargs={
                                               'method': 'Nelder-Mead',
                                               'args':
                                                   (self, sigma_in, alpha_mi,
                                                    alpha_gmm, cost_history,
                                                    self.tau_ord_mags, False),
                                               'options': opt_options,
                                               'callback': loss_callback
                                           })
                # self.tau = opt_results.x
                self.tau = opt_results.x * self.tau_ord_mags

                # print("Optimization successful: {}\n {}\n".format(
                #     opt_results.success, opt_results.message))
                # optim_successful = True

            except BadProjection:
                print("Bad projection.. trying again")
                self.tau = perturb_tau(self.tau_preoptimize, 0.005, 0.5)
                cost_history = []

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


def loss_scaled(tau, calibrator, sigma_in, alpha_mi, alpha_gmm, cost_history,
                tau_scales, return_components):
    calibrator.tau = tau

    if tau_scales is not None:
        calibrator.tau = np.multiply(calibrator.tau, tau_scales)

    calibrator.project_point_cloud()
    num_frames = len(calibrator.pc_detector.pcs)

    cost_mi = cost_gmm = 0
    for frame_idx in range(num_frames):
        cost_mi += alpha_mi * calibrator.compute_mi_cost(frame_idx)
        cost_gmm += alpha_gmm * calibrator.compute_conv_cost(
            sigma_in, frame_idx)

    total_cost = (cost_mi / num_frames) + (cost_gmm / num_frames)
    cost_history.append(total_cost)
    # print([cost_mi, cost_gmm])

    if return_components:
        return [cost_mi / num_frames, cost_gmm / num_frames]
    else:
        return cost_history[-1]


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


class BadProjection(Exception):
    """Bad Projection exception"""
