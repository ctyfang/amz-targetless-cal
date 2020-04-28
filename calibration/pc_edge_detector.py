import numpy as np
from scipy.spatial import ckdtree
import time
from time import sleep
import matplotlib
from matplotlib import cm as cm
import open3d as o3d


class PcEdgeDetector:

    def __init__(self, cfg, visualize=True):
        self.pcs = self.load_pc(cfg.pc_dir, cfg.frames)

        self.pcs_edge_idxs = None
        self.pcs_edge_scores = None
        self.pcs_max_edge_score = None
        self.pcs_nn_sizes = None

        self.PC_ED_SCORE_THR = cfg.pc_ed_score_thr
        self.PC_ED_RAD_NN = cfg.pc_ed_rad_nn
        self.PC_ED_NUM_NN = cfg.pc_ed_num_nn

        self.pc_detect(self.PC_ED_SCORE_THR, self.PC_ED_NUM_NN,
                       self.PC_ED_RAD_NN, visualize=visualize)

    def pc_detect(self, thresh=0.6, num_nn=100, rad_nn=0.1, visualize=False):
        """
        Compute edge scores for pointcloud
        Get edge points via thresholding
        """

        # TODO: Be able to process several point clouds
        # Init outputs
        num_points = self.pcs.shape[0]
        center_scores = np.zeros(num_points)
        planar_scores = np.zeros(num_points)
        self.pcs_edge_scores = np.zeros(num_points)
        self.pcs_nn_sizes = np.zeros(num_points)

        start_t = time.time()
        kdtree = ckdtree.cKDTree(self.pcs)
        for point_idx in range(num_points):
            curr_xyz = self.pcs[point_idx, :]

            neighbor_d1, neighbor_i1 = kdtree.query(curr_xyz, num_nn)
            neighbor_i2 = kdtree.query_ball_point(curr_xyz, rad_nn)
            # remove duplicates
            neighbor_i2 = list(set(neighbor_i2) - set(neighbor_i1))
            neighbor_d2 = np.linalg.norm(self.pcs[neighbor_i2, :] - curr_xyz,
                                         ord=2,
                                         axis=1)
            neighbor_i = np.append(neighbor_i1, neighbor_i2).astype(np.int)
            neighbor_d = np.append(neighbor_d1, neighbor_d2)

            self.pcs_nn_sizes[point_idx] = neighbor_d.shape[0]
            neighborhood_xyz = self.pcs[neighbor_i.tolist(), :]

            center_score = self.compute_centerscore(neighborhood_xyz, curr_xyz,
                                                    np.max(neighbor_d))
            planarity_score = self.compute_planarscore(
                neighborhood_xyz, curr_xyz)

            center_scores[point_idx] = center_score
            planar_scores[point_idx] = planarity_score

        # Combine two edge scores
        # (Global normalization, local neighborhood size normalization)
        max_center_score = np.max(center_scores)
        max_planar_score = np.max(planar_scores)
        self.pcs_edge_scores = 0.5 * \
            (center_scores / max_center_score
             + planar_scores / max_planar_score)

        print(f"Total pc scoring time:{time.time() - start_t}")

        # Remove all points with an edge score below the threshold
        score_mask = self.pcs_edge_scores > thresh
        self.pcs_edge_idxs = np.argwhere(score_mask)
        self.pcs_edge_idxs = np.squeeze(self.pcs_edge_idxs)

        # Exclude boundary points in final thresholding
        # and max score calculation
        pc_boundary_idxs = self.get_first_and_last_channel_idxs(self.pcs)
        boundary_mask = [
            (edge_idx not in pc_boundary_idxs) for edge_idx in self.pcs_edge_idxs
        ]
        self.pcs_edge_idxs = self.pcs_edge_idxs[boundary_mask]

        pc_nonbound_edge_scores = np.delete(self.pcs_edge_scores,
                                            pc_boundary_idxs,
                                            axis=0)
        self.pcs_max_edge_score = np.max(pc_nonbound_edge_scores)

        if visualize:
            self.pc_visualize_edges(
                self.pcs, self.pcs_edge_idxs, self.pcs_edge_scores)

    @staticmethod
    def load_pc(path, frames):
        if len(frames) <= 1:
            return np.fromfile(str(path) + '/velodyne_points/data/' +
                               str(frames[0]).zfill(10) + '.bin',
                               dtype=np.float32).reshape(-1, 4)[:, :3]
        else:
            pcs = []
            for frame in frames:
                pcs.append(np.fromfile(str(path) + '/velodyne_points/data/' +
                                       str(frame).zfill(10) + '.bin',
                                       dtype=np.float32).reshape(-1, 4)[:, :3])
            return pcs

    @staticmethod
    def compute_centerscore(nn_xyz, center_xyz, max_nn_d):
        # Description: Compute modified center-based score. Distance between center (center_xyz),
        #              and the neighborhood (N x 3) around center_xyz. Result is scaled by max
        #              distance in the neighborhood, as done in Kang 2019.

        centroid = np.mean(nn_xyz, axis=0)
        norm_dist = np.linalg.norm(center_xyz - centroid)/max_nn_d
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
            s += np.dot(centered_xyz[i, :].T, centered_xyz[i, :])
        s /= n_points

        # Compute planarity of neighborhood using SVD (Xia & Wang 2017)
        _, eig_vals, _ = np.linalg.svd(s)
        planarity = 1 - (eig_vals[1] - eig_vals[2])/eig_vals[0]

        return planarity

    @staticmethod
    def get_first_and_last_channel_idxs(pc, hor_res=0.2):
        x = pc[:, 0]
        y = pc[:, 1]
        z = pc[:, 2]

        polar_angle = np.arctan2(np.sqrt(np.square(x) + np.square(y)), z)

        # Find the indices of the 360 / horizontal_resolution smallest/largest polar angles
        size_channel = 2 * int(360 / hor_res)
        neg_mask_smallest_angle = np.argpartition(
            polar_angle, size_channel)[:size_channel]
        neg_mask_largest_angle = np.argpartition(
            polar_angle, -size_channel)[-size_channel:]
        boundary_idxs = np.concatenate(
            (neg_mask_largest_angle, neg_mask_smallest_angle), axis=0)

        return np.unique(boundary_idxs)

    def pc_visualize_edges(self, xyz, edge_idxs, edge_scores):
        v_min = np.min(edge_scores)
        v_max = np.max(edge_scores)

        edge_points = xyz[edge_idxs, :]
        edge_scores = edge_scores[edge_idxs]
        self.visualize_xyz_scores(
            edge_points, edge_scores, vmin=v_min, vmax=v_max)

    @staticmethod
    def visualize_xyz_scores(xyz, scores, vmin=None, vmax=None):
        if vmin is None:
            vmin = np.min(scores)
        if vmax is None:
            vmax = np.max(scores)

        norm = matplotlib.colors.Normalize(
            vmin=vmin, vmax=vmax, clip=True)
        mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)
        colors = np.asarray([mapper.to_rgba(x)[:3] for x in scores])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()
