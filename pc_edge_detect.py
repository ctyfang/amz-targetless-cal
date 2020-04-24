"""Utilities for detecting edges in pointclouds"""
import os
import time
import json

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import ckdtree
import numpy as np
import open3d as o3d

from data_utils import *


def detect_pc_edges(pc, thr, num_nn):
    """ Calculates edge scores of a point cloud as in Kang 2019

    Parameters:
    pc      -- input point cloud 
    thr     -- threshold for discarding points with a lower edge score 
    num_nn  -- number of nearest neighbors for calculating the edge score 

    Return:
    pc_edge_points      
    pc_edge_scores
    max_pc_edge_score

    """

    O3D_VIEW_PATH = './configs/o3d_camera_model.json'

    CAMERA_PARAMS = o3d.io.read_pinhole_camera_parameters(O3D_VIEW_PATH)

    # View pointcloud from approx. perspective of camera
    # custom_draw_xyz_with_params(pc_xyz, camera_params)

    # Visualize neighborhoods around each point
    # visualize_neighborhoods(pc_xyz)

    # Edge Score Computation
    kdtree = ckdtree.cKDTree(pc)
    num_points = pc.shape[0]
    center_scores = np.zeros(num_points)
    planar_scores = np.zeros(num_points)
    pc_edge_scores = np.zeros(num_points)
    nn_sizes = np.zeros(num_points)

    start_t = time.time()
    for point_idx in range(pc.shape[0]):

        curr_xyz = pc[point_idx, :]
        # TODO: Modify neighborhood extraction
        neighbor_d, neighbor_i = kdtree.query(curr_xyz, num_nn)
        nn_sizes[point_idx] = neighbor_d.shape[0]

        neighborhood_xyz = pc[neighbor_i, :]

        t = time.time()
        center_score = compute_centerscore(
            neighborhood_xyz, curr_xyz, np.max(neighbor_d))
        planarity_score = compute_planarscore(neighborhood_xyz, curr_xyz)
        # print(f'Center score:{center_score}\n Planarity score:{planarity_score}\n')
        # print(f'Elapsed for computation:{time.time()-t}')

        t = time.time()
        center_scores[point_idx] = center_score
        planar_scores[point_idx] = planarity_score
        # print(f'Elapsed for appending:{time.time()-t}')

    # Combine two edge scores (Global normalization, local neighborhood size normalization)
    max_center_score = np.max(center_scores)
    max_planar_score = np.max(planar_scores)
    pc_edge_scores = 0.5 * \
        (center_scores/max_center_score + planar_scores/max_planar_score)

    print(f"Total pc scoring time:{time.time()-start_t}")

    # Store min/max value of pc_edge_scores for visualization
    vmin = np.min(pc_edge_scores)
    vmax = np.max(pc_edge_scores)

    # Remove first and last channel of rotating lidar point cloud using polar angle
    pc, pc_edge_scores = rm_first_and_last_channel(pc, pc_edge_scores)

    # Remove all points with an edge score below the threshold
    filter = pc_edge_scores > thr
    pc_edge_scores = pc_edge_scores[filter]
    pc = pc[filter]

    pc_edge_points = pc
    max_pc_edge_score = np.max(pc_edge_scores)

    # Normalize scores and map to colors
    # visualize_xyz_scores(pc_edge_points, center_scores)
    # visualize_xyz_scores(pc_edge_points, planar_scores)
    visualize_xyz_scores(pc_edge_points, pc_edge_scores, vmin=vmin, vmax=vmax)

    return pc_edge_points, pc_edge_scores, max_pc_edge_score


if __name__ == "__main__":

    # Read point cloud from file
    DATA_PATH_LIST = [('/media/carter/Samsung_T5/3dv/2011_09_26/2011_09_26_drive_0106_sync/'
                       'velodyne_points/data'), ('/home/benjin/Development/Data/'
                                                 '2011_09_26_drive_0106_sync/velodyne_points/data')]

    DATA_PATH = getPath(DATA_PATH_LIST)
    FILE_IDX = 29
    pc = load_from_bin(os.path.join(
        DATA_PATH, str(FILE_IDX).zfill(10) + '.bin'))

    # Select a subset of the point cloud
    DATA_FRACTION = 1
    fraction_idxs = np.random.randint(
        0, pc.shape[0], size=round(DATA_FRACTION*pc.shape[0]))
    pc = pc[fraction_idxs, :]

    # Define constants
    THRESHOLD = 0.6
    NUM_NN = 100

    pc_edge_points, pc_edge_scores, max_pc_edge_score = detect_pc_edges(
        pc, THRESHOLD, NUM_NN)
