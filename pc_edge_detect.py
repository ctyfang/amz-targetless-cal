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

with open('./configs/data_paths.json') as paths_handle:
    data = json.load(paths_handle)
    PC_DIR = data['PC_DIR']

O3D_VIEW_PATH = './configs/o3d_camera_model.json'
FILE_IDX = 29
pc_xyz = load_from_bin(os.path.join(PC_DIR, str(FILE_IDX).zfill(10) + '.bin'))
CAMERA_PARAMS = o3d.io.read_pinhole_camera_parameters(O3D_VIEW_PATH)

# Select a subset of the points
DATA_FRACTION = 0.01
fraction_idxs = np.random.randint(0, pc_xyz.shape[0], size=round(DATA_FRACTION*pc_xyz.shape[0]))
pc_xyz = pc_xyz[fraction_idxs, :]

## View pointcloud from approx. perspective of camera
# custom_draw_xyz_with_params(pc_xyz, camera_params)

## Visualize neighborhoods around each point
# visualize_neighborhoods(pc_xyz)

# Edge Score Computation
kdtree = ckdtree.cKDTree(pc_xyz)
num_points = pc_xyz.shape[0]
center_score_arr = np.zeros(num_points)
planar_score_arr = np.zeros(num_points)
total_score_arr = np.zeros(num_points)
nn_size_arr = np.zeros(num_points)

start_t = time.time()
for point_idx in range(pc_xyz.shape[0]):

    curr_xyz = pc_xyz[point_idx, :]
    # TODO: Modify neighborhood extraction
    neighbor_d, neighbor_i = kdtree.query(curr_xyz, 100)
    nn_size_arr[point_idx] = neighbor_d.shape[0]

    neighborhood_xyz = pc_xyz[neighbor_i, :]

    t = time.time()
    center_score = compute_centerscore(neighborhood_xyz, curr_xyz, np.max(neighbor_d))
    planarity_score = compute_planarscore(neighborhood_xyz, curr_xyz)
    # print(f'Center score:{center_score}\n Planarity score:{planarity_score}\n')
    # print(f'Elapsed for computation:{time.time()-t}')

    t = time.time()
    center_score_arr[point_idx] = center_score
    planar_score_arr[point_idx] = planarity_score
    # print(f'Elapsed for appending:{time.time()-t}')

# Combine two edge scores (Global normalization, local neighborhood size normalization)
max_center_score = np.max(center_score_arr)
max_planar_score = np.max(planar_score_arr)
total_score_arr = np.divide(0.5*(center_score_arr/max_center_score + planar_score_arr/max_planar_score), nn_size_arr)
print(f"Total pc scoring time:{time.time()-start_t}")

# Normalize scores and map to colors
# visualize_xyz_scores(pc_xyz, center_score_arr)
# visualize_xyz_scores(pc_xyz, planar_score_arr)
visualize_xyz_scores(pc_xyz, total_score_arr)
