import numpy as np
from scipy.spatial import ckdtree
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm as cm
import open3d as o3d
from sklearn.cluster import KMeans

import time
import os
from glob import glob
from tqdm import tqdm


class PcEdgeDetector:
    """Helper class for Calibrator. Load pointcloud files from .bin's,
    run edge-detection, and generate masks to extract edge points for
    projection during optimization."""

    def __init__(self, cfg, visualize=True):
        if os.path.exists(cfg.dir):
            self.pcs, self.reflectances = self.load_pcs(cfg.dir,
                                                        cfg.frames)
        else:
            print("Pointcloud directory does not exist.")
            exit()

        self.pcs_edge_idxs = []
        self.pcs_edge_masks = []
        self.pcs_edge_scores = []

        self.PC_ED_SCORE_THR = cfg.pc_ed_score_thr
        self.PC_ED_RAD_NN = cfg.pc_ed_rad_nn
        self.PC_ED_NUM_NN = cfg.pc_ed_num_nn
        self.PC_NUM_CHANNELS = 64

    def __len__(self):
        if len(self.pcs) == len(self.reflectances):
            return len(self.pcs)
        else:
            print('Number of xyz coordinates does not match number of reflectance frame')
            return -1

    def pc_detect(self, pcs_cam_frame, thresh=60, num_nn=100, rad_nn=0.1,
                  visualize=False):
        """
        Compute edge scores for pointcloud
        Get edge points via thresholding
        """

        # Init outputs
        for idx, (pc, pc_cam_frame) in enumerate(zip(self.pcs, pcs_cam_frame)):
            print('Point Cloud #{}/{}'.format(idx+1, len(self.pcs)))
            num_points = pc.shape[0]
            center_scores = np.zeros(num_points)
            planar_scores = np.zeros(num_points)

            start_t = time.time()
            kdtree = ckdtree.cKDTree(pc)
            for point_idx in tqdm(range(num_points)):
                curr_xyz = pc[point_idx, :]

                neighbor_d1, neighbor_i1 = kdtree.query(curr_xyz, num_nn)
                neighbor_i2 = kdtree.query_ball_point(curr_xyz, rad_nn)
                # remove duplicates
                neighbor_i2 = list(set(neighbor_i2) - set(neighbor_i1))
                neighbor_d2 = np.linalg.norm(pc[neighbor_i2, :] - curr_xyz,
                                             ord=2,
                                             axis=1)
                neighbor_i = np.append(neighbor_i1, neighbor_i2).astype(np.int)
                neighbor_d = np.append(neighbor_d1, neighbor_d2)

                neighborhood_xyz = pc[neighbor_i.tolist(), :]

                center_score = self.compute_centerscore(neighborhood_xyz,
                                                        curr_xyz,
                                                        np.max(neighbor_d))
                planarity_score = self.compute_planarscore(
                    neighborhood_xyz, curr_xyz)

                center_scores[point_idx] = center_score
                planar_scores[point_idx] = planarity_score

            # Combine three edge scores
            # (Global normalization, local neighborhood size normalization)
            pc_edge_scores_1 = center_scores
            pc_edge_scores_1 = (pc_edge_scores_1 - pc_edge_scores_1.min())/(pc_edge_scores_1.max() - pc_edge_scores_1.min())

            pc_edge_scores_2 = planar_scores
            pc_edge_scores_2 = (pc_edge_scores_2 - pc_edge_scores_2.min())/(pc_edge_scores_2.max() - pc_edge_scores_2.min())
            pc_edge_scores = pc_edge_scores_1*pc_edge_scores_2

            # Calculate the depth discontinuity score
            depth_discontinuity_scores = self.compute_depth_discontinuity_score(
                pc, self.PC_NUM_CHANNELS)
            pc_edge_scores_3 = depth_discontinuity_scores
            pc_edge_scores_3 = (pc_edge_scores_3 - pc_edge_scores_3.min())/(pc_edge_scores_3.max() - pc_edge_scores_3.min())
            pc_edge_scores = pc_edge_scores*pc_edge_scores_3

            # NMS
            points_suppressed = 0
            for point_idx in range(num_points):
                curr_score = pc_edge_scores[point_idx]
                neighbor_i = kdtree.query_ball_point(curr_xyz, 0.10)
                neighbor_scores = pc_edge_scores[neighbor_i]

                if (neighbor_scores > curr_score).any():
                    pc_edge_scores[point_idx] = 0
                    points_suppressed += 1

            self.pcs_edge_scores.append(pc_edge_scores)

            # Remove all points with an edge score below the threshold
            thresh = np.percentile(pc_edge_scores, thresh)
            self.pcs_edge_masks.append(self.pcs_edge_scores[-1] > thresh)
            # Exclude boundary points in final thresholding
            pc_boundary_idxs = self.get_first_and_last_channels_idxs(pc)
            self.pcs_edge_masks[-1][pc_boundary_idxs] = False
            # Exclude points that in the camera frame have a euclidean distance
            # greater than a treshold
            outside_point_idxs = self.get_points_outside_radius(
                pc_cam_frame, radius=25)
            self.pcs_edge_masks[-1][outside_point_idxs] = False

            self.pcs_edge_idxs.append(
                np.squeeze(np.argwhere(self.pcs_edge_masks[-1])))

        if visualize:
            for idx in range(len(self.pcs)):
                self.pc_visualize_edges(self.pcs[idx], self.pcs_edge_idxs[idx],
                                        self.pcs_edge_scores[idx])

    @staticmethod
    def load_pcs(path, frames, subsample=1.0):
        """Load pointclouds, separate XYZ and Reflectance components"""
        pcs = []
        reflectances = []

        if frames == -1:
            frame_paths = sorted(
                glob(os.path.join(path, 'velodyne_points', 'data', '*.bin')))
        else:
            frame_paths = [os.path.join(path, 'velodyne_points', 'data', str(
                frame).zfill(10)) + ".bin" for frame in frames]

        if len(frame_paths) == 0:
            frame_paths = sorted(
                glob(os.path.join(path, 'fw_lidar_filtered', '*.bin'))
            )

        for path in frame_paths:
            _, ext = os.path.splitext(path)
            if ext == '.bin':
                curr_pc = (np.fromfile(path,
                                       dtype=np.float64).reshape(-1, 6))[:, :]
            elif ext == '.txt':
                curr_pc = (np.loadtxt(path,
                                       dtype=np.float64).reshape(-1, 6))[:, :]
            elif ext == '.npy':
                curr_pc = (np.fromfile(path,
                                       dtype=np.float64).reshape(-1, 6))[:, :]
            else:
                print("Invalid point-cloud format encountered.")
                exit()

            pc = curr_pc[:int(subsample * curr_pc.shape[0]), :3]
            refl = curr_pc[:int(subsample * curr_pc.shape[0]), 3]

            xyz_mask = ~np.isnan(pc)
            ref_mask = ~np.isnan(refl)
            mask = np.logical_and(xyz_mask[:,0], xyz_mask[:,1])
            mask = np.logical_and(mask, xyz_mask[:, 2])
            mask = np.logical_and(mask, ref_mask)
 
            pcs.append(pc[mask])
            reflectances.append(refl[mask])

        return pcs, reflectances

    @staticmethod
    def compute_centerscore(nn_xyz, center_xyz, max_nn_d):
        """Description: Compute modified center-based score. Distance between
        center (center_xyz), and the neighborhood (N x 3) around center_xyz.
        Result is scaled by max distance in the neighborhood, as done in
        Kang 2019."""

        centroid = np.mean(nn_xyz, axis=0)
        norm_dist = np.linalg.norm(center_xyz - centroid) / max_nn_d
        return norm_dist

    @staticmethod
    def compute_planarscore(nn_xyz, center_xyz):

        complete_xyz = np.concatenate(
            [nn_xyz, center_xyz.reshape((1, 3))], axis=0)
        centroid = np.mean(complete_xyz, axis=0)
        centered_xyz = complete_xyz - centroid

        # Build structure tensor
        n_points = centered_xyz.shape[0]
        s = np.zeros((3, 3))
        for i in range(n_points):
            test = np.dot(centered_xyz[i, :].reshape((3, 1)), centered_xyz[i, :].reshape((1, 3)))
            s += test
        s /= n_points

        # Compute planarity of neighborhood using SVD (Xia & Wang 2017)
        _, eig_vals, _ = np.linalg.svd(s)
        planarity = 1 - (eig_vals[1] - eig_vals[2]) / eig_vals[0]

        return planarity

    @staticmethod
    def compute_depth_discontinuity_score(point_cloud, num_channels):
        """Description: Compute vertical edges using depth discontinuity as the edge indicator. First
                     the points of the point cloud are sorted according to their polar angle to
                     group the points into their corresponding channels. Then the points of the
                     channels are sorted according to their azimuth angle. After calculating the depth
                     of each point, the change of depth between consecutive points in the sorted
                     channel is calculated. If this change is above a threshold for a certain point,
                     it is considered an edge.
        Return:      A binary mask indicating edges in the point cloud."""

        # Calculate the polar angle for every point
        polar_angle = 180 * np.arctan2(point_cloud[:, 2], np.sqrt(
            np.square(point_cloud[:, 0]) + np.square(point_cloud[:, 1]))) / np.pi

        # Calculate 64 cluster means
        max_polar_angle = np.max(polar_angle)
        min_polar_angle = np.min(polar_angle)
        # Calculate an initial guess
        init_guess = np.linspace(
            min_polar_angle, max_polar_angle, num=num_channels)
        kmeans = KMeans(n_clusters=num_channels, init=init_guess.reshape(-1, 1)
                        ).fit(polar_angle.reshape(-1, 1))

        # Fill list with the idxs of edges
        edge_idxs = []

        # Return depth discontinuity scores
        point_cloud_ddepth = np.zeros(point_cloud.shape[0])

        for channel in range(num_channels):
            # Binary mask for the points of the current channel
            current_channel_mask = (kmeans.labels_ == channel)
            # Indices of the points of the current channel
            current_channel_idxs = np.argwhere(current_channel_mask)
            # Array of the points of the current channel
            current_channel_points = point_cloud[current_channel_mask]
            # Array that assigns every point in current_channel_points an index
            current_channel_order = np.arange(current_channel_points.shape[0])

            # Calculate the azimuth angles of the points of the channel
            azimuth_angle = 180 * \
                np.arctan2(
                    current_channel_points[:, 1], current_channel_points[:, 0]) / np.pi

            # Find the array that sorts the points of the current channel according to its azimuth angle
            sort_by_azimuth_angle = np.argsort(azimuth_angle)

            azimuth_angle_sorted = azimuth_angle[sort_by_azimuth_angle]
            current_channel_points_sorted = current_channel_points[sort_by_azimuth_angle]
            current_channel_order = current_channel_order[sort_by_azimuth_angle]

            # Visualize current ring colored by azimuth value
            VISUALIZE_CURRENT_RING = False
            if VISUALIZE_CURRENT_RING:
                PcEdgeDetector.visualize_xyz_scores(
                    current_channel_points_sorted, azimuth_angle_sorted, cmap=cm.summer)

            # Calculate the depth of each point
            depth = np.linalg.norm(current_channel_points_sorted, axis=1)

            # Ben method
            # rml: depth of right point minus depth of left point
            delta_depth_rml = depth - np.roll(depth, -1)
            # lmr: depth of left point minus depth of right point
            delta_depth_lmr = depth - np.roll(depth, +1)

            # Visualize depth, and delta depth against azimuth_angle_sorted
            VISUALIZE_DELTA_DEPTH_AGAINST_AZIMUTH_ANGLE = False
            if VISUALIZE_DELTA_DEPTH_AGAINST_AZIMUTH_ANGLE:
                fig, ax = plt.subplots(figsize=(20, 10))
                ax.scatter(
                    azimuth_angle_sorted[:300], depth[:300], s=1, alpha=0.75)
                ax.scatter(azimuth_angle_sorted[:300],
                           delta_depth_lmr[:300], s=1, c='red')
                plt.tight_layout(pad=3.0)
                plt.show()

            edges_filter = np.logical_or(
                delta_depth_rml < -0.5, delta_depth_lmr < -0.5)


            current_edges = current_channel_points_sorted[edges_filter]

            # Get the idxs of the points in the current channel that are edges
            current_channel_order = current_channel_order[edges_filter]

            # Get the idxs of the points in the point cloud that are edges
            current_ring_edge_idxs = current_channel_idxs[current_channel_order]

            # Visualize edges in current ring
            VISUALIZE_EDGES_IN_RING = False
            if VISUALIZE_EDGES_IN_RING:
                PcEdgeDetector.visualize_xyz_scores(
                    current_channel_points_sorted, edges_filter, cmap=cm.summer)

            # Delete current edges if too many are detected
            if current_edges.shape[0] > 500:
                continue
            edge_idxs.extend(current_ring_edge_idxs.tolist())

        # Get the binary mask indicating the edge points in the point cloud
        point_cloud_mask = np.zeros(point_cloud.shape[0])
        point_cloud_mask[edge_idxs] = True
        return point_cloud_mask

    @staticmethod
    def get_first_and_last_channels_idxs(pc, ch_to_remove=3, hor_res=0.2):
        """
        Returns numpy array of the point indices that are in the first and last channels
        """
        x = pc[:, 0]
        y = pc[:, 1]
        z = pc[:, 2]

        polar_angle = np.arctan2(np.sqrt(np.square(x) + np.square(y)), z)

        # Find the indices of the 360 / horizontal_resolution smallest/largest polar angles
        size_channel = 3 * int(360 / hor_res)
        neg_mask_smallest_angle = np.argpartition(polar_angle,
                                                  size_channel)[:size_channel]
        neg_mask_largest_angle = np.argpartition(polar_angle,
                                                 -size_channel)[-size_channel:]
        boundary_idxs = np.concatenate(
            (neg_mask_largest_angle, neg_mask_smallest_angle), axis=0)

        return np.unique(boundary_idxs)

    @staticmethod
    def get_points_outside_radius(pc, radius=20):
        """
        Return point indices of points outside radius
        """
        distance = np.sqrt(
            np.power(pc[:, 0], 2) +
            np.power(pc[:, 1], 2) +
            np.power(pc[:, 2], 2))

        return np.argwhere(distance > radius)

    def pc_visualize_edges(self, xyz, edge_idxs, edge_scores):
        v_min = np.min(edge_scores)
        v_max = np.max(edge_scores)

        edge_points = xyz[edge_idxs, :]
        edge_scores = edge_scores[edge_idxs]
        self.visualize_xyz_scores(edge_points,
                                  edge_scores,
                                  vmin=v_min,
                                  vmax=v_max)

    @staticmethod
    def visualize_xyz_scores(xyz, scores, vmin=None, vmax=None, cmap=cm.tab20c):
        if vmin is None:
            vmin = np.min(scores)
        if vmax is None:
            vmax = np.max(scores)

        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        colors = np.asarray(mapper.to_rgba(scores))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        ctr = vis.get_view_control()
        print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
        ctr.change_field_of_view(step=4)
        print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
        vis.run()
        cam = ctr.convert_to_pinhole_camera_parameters()
        print(cam.extrinsic)
        print(cam.intrinsic.intrinsic_matrix)
