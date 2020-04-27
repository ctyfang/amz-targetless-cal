"""Top-level class for Calibration"""
import numpy as np
import cv2 as cv
from scipy.spatial import ckdtree

from data_utils import *
from pc_utils import *
from pc_utils import visualize_edges as pc_visualize_edges

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from scipy.ndimage.filters import gaussian_filter, convolve


class Calibrator:

    def __init__(self, calib_dir, input_dir, frame):
        self.pc = np.fromfile(str(input_dir) + '/velodyne_points/data/' +
                              str(frame).zfill(10) + '.bin',
                              dtype=np.float32).reshape(-1, 4)[:, :3]
        self.pc_edge_scores = None
        self.pc_edge_idxs = None
        self.pc_nn_sizes = None
        self.max_pc_edge_score = None

        self.img = cv.imread(
            str(input_dir) + '/image_00/data/' + str(frame).zfill(10) + '.png')
        self.img_edge_scores = None
        self.img_edges = None
        self.pixels = None

        self.P_rect = load_cam_cal(calib_dir)
        self.R, self.T = load_lid_cal(calib_dir)

    def pc_detect(self, thresh=0.6, num_nn=100, rad_nn=0.1, visualize=False):
        """
        Compute edge scores for pointcloud
        Get edge points via thresholding
        """

        # Init outputs
        num_points = self.pc.shape[0]
        center_scores = np.zeros(num_points)
        planar_scores = np.zeros(num_points)
        self.pc_edge_scores = np.zeros(num_points)
        self.pc_nn_sizes = np.zeros(num_points)

        start_t = time.time()
        kdtree = ckdtree.cKDTree(self.pc)
        for point_idx in range(num_points):
            curr_xyz = self.pc[point_idx, :]

            neighbor_d1, neighbor_i1 = kdtree.query(curr_xyz, num_nn)
            neighbor_i2 = kdtree.query_ball_point(curr_xyz, rad_nn)
            # remove duplicates
            neighbor_i2 = list(set(neighbor_i2) - set(neighbor_i1))
            neighbor_d2 = np.linalg.norm(self.pc[neighbor_i2, :] - curr_xyz,
                                         ord=2,
                                         axis=1)
            neighbor_i = np.append(neighbor_i1, neighbor_i2).astype(np.int)
            neighbor_d = np.append(neighbor_d1, neighbor_d2)

            self.pc_nn_sizes[point_idx] = neighbor_d.shape[0]
            neighborhood_xyz = self.pc[neighbor_i.tolist(), :]

            center_score = compute_centerscore(neighborhood_xyz, curr_xyz,
                                               np.max(neighbor_d))
            planarity_score = compute_planarscore(neighborhood_xyz, curr_xyz)

            center_scores[point_idx] = center_score
            planar_scores[point_idx] = planarity_score

        # Combine two edge scores
        # (Global normalization, local neighborhood size normalization)
        max_center_score = np.max(center_scores)
        max_planar_score = np.max(planar_scores)
        self.pc_edge_scores = 0.5 * \
            (center_scores / max_center_score
             + planar_scores / max_planar_score)

        print(f"Total pc scoring time:{time.time() - start_t}")

        # Remove all points with an edge score below the threshold
        score_mask = self.pc_edge_scores > thresh
        self.pc_edge_idxs = np.argwhere(score_mask)
        self.pc_edge_idxs = np.squeeze(self.pc_edge_idxs)

        # Exclude boundary points in final thresholding
        # and max score calculation
        pc_boundary_idxs = get_first_and_last_channel_idxs(self.pc)
        boundary_mask = [
            (edge_idx not in pc_boundary_idxs) for edge_idx in self.pc_edge_idxs
        ]
        self.pc_edge_idxs = self.pc_edge_idxs[boundary_mask]

        pc_nonbound_edge_scores = np.delete(self.pc_edge_scores,
                                            pc_boundary_idxs,
                                            axis=0)
        self.max_pc_edge_score = np.max(pc_nonbound_edge_scores)

        if visualize:
            pc_visualize_edges(self.pc, self.pc_edge_idxs, self.pc_edge_scores)

    def img_detect(self, visualize=False):
        '''
        Compute pixel-wise edge score with non-maximum suppression
        Scores are normalized so that maximum score is 1
        '''
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        blurred = gaussian_filter(gray, sigma=2, order=0, mode='reflect')

        gradient_x = convolve(blurred, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        gradient_y = convolve(blurred, [[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.img_edge_scores = np.sqrt(
            np.power(gradient_x, 2) + np.power(gradient_y, 2))
        self.img_edges = cv.Canny(self.img, 100, 200, L2gradient=True)
        self.img_edge_scores[np.where(self.img_edges == 0)] = 0
        self.img_edge_scores = \
            self.img_edge_scores/np.amax(self.img_edge_scores)

        if visualize:
            im_x, im_y = np.meshgrid(
                np.linspace(0, self.img_edge_scores.shape[1],
                            self.img_edge_scores.shape[1] + 1),
                np.linspace(0, self.img_edge_scores.shape[0],
                            self.img_edge_scores.shape[0] + 1))

            levels = MaxNLocator(nbins=15).tick_values(0, 1)
            cmap = plt.get_cmap('hot')
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

            fig, (ax0, ax1) = plt.subplots(nrows=2)
            plot = ax0.pcolormesh(im_x,
                                  im_y,
                                  self.img_edge_scores[::-1, :],
                                  cmap=cmap,
                                  norm=norm)
            fig.colorbar(plot, ax=ax0)
            ax0.set_title('edge scores')

            binary_edge = ax1.pcolormesh(im_x,
                                         im_y,
                                         self.img_edges[::-1, :],
                                         cmap=cmap,
                                         norm=norm)
            fig.colorbar(binary_edge, ax=ax1)
            ax1.set_title('Binary Edge')
            plt.axis('equal')
            plt.show()

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
