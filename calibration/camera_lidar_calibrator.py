"""Top-level class for Calibration"""
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
from pyquaternion import Quaternion
import gc

from scipy.ndimage.filters import gaussian_filter, convolve
from scipy.optimize import least_squares, minimize
from scipy.stats import multivariate_normal
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score
from scipy.linalg import expm
from scipy.stats import norm

from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

from calibration.img_edge_detector import ImgEdgeDetector
from calibration.pc_edge_detector import PcEdgeDetector
from calibration.utils.data_utils import *
from calibration.utils.img_utils import *
from calibration.utils.pc_utils import *
from calibration.loss_functions import *

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
        self.pc_detector = PcEdgeDetector(cfg, visualize=visualize)
        self.img_detector = ImgEdgeDetector(cfg, visualize=visualize)

        # Calculate projected_points, points_cam_frame, projection_mask
        self.project_point_cloud()

        # Detect edges
        self.pc_detector.pc_detect(self.points_cam_frame,
                                   cfg.pc_ed_score_thr,
                                   cfg.pc_ed_num_nn,
                                   cfg.pc_ed_rad_nn,
                                   visualize=visualize)
        gc.collect()
        self.img_detector.img_detect(visualize=visualize)
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
        tau_quat[:4] = Quaternion(axis=tau[:3]/np.linalg.norm(tau[:3], 2),
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

        projected_points_valid = self.projected_points[frame][self.projection_mask[frame]]

        for pixel, color in zip(projected_points_valid, colors_valid):
            image[pixel[1].astype(np.int), pixel[0].astype(np.int), :] = color

        if show:
            cv.imshow('Projected Point Cloud on Image', image)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return image

    def draw_reflectance(self, frame=-1):
        """Given frame, draw reflectance image"""
        img_h, img_w = self.img_detector.imgs[frame].shape[:2]
        refl_img = np.zeros((img_h, img_w), dtype=np.float32)

        projected_points_valid = self.projected_points[frame][self.projection_mask[frame]]
        reflectance_values = self.pc_detector.reflectances[frame][self.projection_mask[frame]]

        for pixel, reflectance in zip(projected_points_valid, reflectance_values):
            refl_img[pixel[1].astype(np.int), pixel[0].astype(np.int)] = reflectance

        cv.imshow('Projected Point Cloud Reflectance Image', refl_img)
        cv.imshow('Grayscale img', cv.cvtColor(self.img_detector.imgs[frame], cv.COLOR_BGR2GRAY))
        cv.waitKey(0)
        cv.destroyAllWindows()

    def draw_edge_points(self, score=None, image=None, frame=-1, save=False, show=False):
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
            cv.imwrite(time.strftime("%Y%m%d-%H%M%S") + '.jpg', image)

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
        #print(f"Brute Force cost computation time:{time.time() - start_t}")
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
                                w_ij*((dG_du*du_dtau) + (dG_dv*dv_dtau))

        return -gradient

    def compute_mi_cost(self, frame=-1):
        """Compute mutual info cost for one frame"""
        grayscale_img = cv.cvtColor(self.img_detector.imgs[frame], cv.COLOR_BGR2GRAY)
        projected_points_valid = self.projected_points[frame][self.projection_mask[frame]]
        grayscale_vector = grayscale_img[projected_points_valid[:, 1].astype(np.uint),
                                         projected_points_valid[:, 0].astype(np.uint)]
        reflectance_vector = (self.pc_detector.reflectances[frame][self.projection_mask[frame]]*255.0)

        if len(reflectance_vector) > 0 and len(grayscale_vector) > 0:
            mi_cost = mutual_info_score(grayscale_vector,
                                        reflectance_vector)
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
                sigma = (sigma_in / np.linalg.norm(self.points_cam_frame[frame][idx, :], 2))
            else:
                sigma = sigma_in

            mu_x, mu_y = self.projected_points[frame][idx].astype(np.int)
            # Get gaussian kernel
            # Distance > 3 sigma is set to 0
            # and normalized so that the total Kernel = 1
            # BUG: In getGaussianKernel2D
            gauss2d = getGaussianKernel2D(sigma, False)
            top, bot, left, right = get_boundry(
                self.img_detector.img_edge_scores[frame], (mu_y, mu_x), int(sigma))
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
            # print(f'Cost: {np.sum(cost_patch)/(2*np.sum(cost_patch>0))}')

            cost_map[mu_y, mu_x] = \
                np.sum(cost_patch) / (2 * np.sum(edge_scores_patch > 0))

        # plot_2d(cost_map)
        # gc.collect()
        # print(f"Convolution Cost Computation time:{time.time() - start_t}")

        return -np.sum(cost_map)

    def optimize(self,
                 sigma_in,
                 max_iters=100,
                 show_every_n=10,
                 verbose=True,
                 convergence_thresh=1e-6):
        iteration = 0
        cost_history = []
        learning_rate = 7.5e-7
        # TODO: Backtracking line learning rate

        last_tau = self.tau
        last_cost = None
        while iteration <= max_iters:
            iteration += 1

            # Project pointcloud edge points, generate R and T from tau
            self.project_point_cloud()

            # Visualize current projection
            if self.visualize and (iteration - 1 % show_every_n) == 0:
                self.draw_edge_points(score=self.pc_detector.pcs_edge_scores,
                                      image=self.img_detector.img_edge_scores,
                                      save=True)

            cost = self.compute_bf_cost(sigma_in)

            # Check for overshoot
            if last_cost is not None and last_cost < cost:
                if verbose:
                    print("Overshot minimum, adjusting learning params...")

                self.tau = last_tau
                sigma_in *= 0.9
                learning_rate /= 2
                continue

            cost_history.append(cost)
            start_time = time.time()
            gradient = self.compute_gradient(sigma_in).reshape((6,))
            self.tau -= learning_rate * gradient

            if verbose:
                print(f"----- ITERATION {iteration} -----")
                print(f"Cost = {cost}")
                print(f"Gradient Magnitude = {np.linalg.norm(gradient, 2)}")
                print(f'Gradient time = {time.time()-start_time}')
                print("---------------------------")

            # Check for convergence
            if len(cost_history) > 2 and abs(cost -
                                             last_cost) < convergence_thresh:
                if verbose:
                    print(
                        f"Converged in {iteration} iterations. LR={learning_rate}. Sig={sigma_in}. Cost={cost}"
                    )
                break

            # Store current cost and tau in-case overshoot
            last_cost = cost
            last_tau = self.tau

        self.draw_all_points()
        self.draw_edge_points(score=self.pc_detector.pcs_edge_scores,
                              image=self.img_detector.img_edge_scores)

        # TODO: Plot cost over the iterations

    def ls_optimize(self, sigma_in, method='lm', alpha_gmm=1, alpha_mi=30, maxiter=1000, translation_only=False):
        """Optimize cost over all image-scan pairs using mutual info and gmm.
            Scale the contributions from two loss sources using alphas."""
        cost_history = []
        tau_preoptimize = self.tau
        minimize_options = {'maxiter': maxiter, 'xtol': 1e-8, 'ftol': 1e-10, 'disp': True}

        def loss_callback(xk):
            total_valid_points = 0
            for frame_idx in range(len(self.projection_mask)):
                total_valid_points += np.sum(self.projection_mask[frame_idx])

            if total_valid_points < 1000:
                raise BadProjection
            return True

        optim_successful = False

        # Re-try optimization until final projection lands on image
        start = time.time()
        while not optim_successful:
            self.tau = tau_preoptimize

            if translation_only:
                print('Optimizing over translation...')
                trans_vec = self.tau[3:]
                try:
                    trans_vec_optimized = minimize(loss_translation,
                                                 trans_vec,
                                                 method='Nelder-Mead',
                                                 args=(self, sigma_in, cost_history),
                                                 options=minimize_options,
                                                 callback=loss_callback)

                    self.tau[3:] = trans_vec_optimized.x
                    optim_successful = True

                except BadProjection:
                    print("Bad projection.. perturbed tau, trying again")
                    tau_preoptimize = perturb_tau(tau_preoptimize, 0.005, 0.001)
                    cost_history = []

            else:
                print('Optimizing over all extrinsics...')
                tau_quat = self.tau_to_tauquat(self.tau)
                try:
                    tau_optimized = minimize(loss,
                                             tau_quat,
                                             method='Nelder-Mead',
                                             args=(self, sigma_in, cost_history),
                                             options=minimize_options,
                                             callback=loss_callback)

                    self.tau = self.tauquat_to_tau(tau_optimized.x)
                    optim_successful = True

                except BadProjection:
                    print("Bad projection.. trying again")
                    tau_preoptimize = perturb_tau(tau_preoptimize, 0.005, 0.5)
                    cost_history = []

        print(f"NL optimizer time={time.time()-start}")
        plt.plot(range(len(cost_history)), cost_history)
        plt.show()

        return self.tau, cost_history
