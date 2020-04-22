import numpy as np
import open3d as o3d
from data_utils import *
import os
from scipy.spatial import ckdtree
import time

data_path = '/media/carter/Samsung_T5/3dv/2011_09_26/2011_09_26_drive_0106_sync/velodyne_points/data'
o3d_view_path = './configs/o3d_camera_model.json'
file_idx = 29
pc_xyz = load_from_bin(os.path.join(data_path, str(file_idx).zfill(10) + '.bin'))
camera_params = o3d.io.read_pinhole_camera_parameters(o3d_view_path)

## View pointcloud from approx. perspective of camera
# custom_draw_xyz_with_params(pc_xyz, camera_params)

## Visualize neighborhoods around each point
# visualize_neighborhoods(pc_xyz)

# Edge Score Computation
kdtree = ckdtree.cKDTree(pc_xyz)
center_score_arr = np.zeros(pc_xyz.shape[0])
planar_score_arr = np.zeros(pc_xyz.shape[0])
for point_idx in range(pc_xyz.shape[0]):

    curr_xyz = pc_xyz[point_idx, :]
    neighbor_d, neighbor_i = kdtree.query(curr_xyz, 100)

    neighborhood_xyz = pc_xyz[neighbor_i, :]

    t = time.time()
    center_score = compute_centerscore(neighborhood_xyz, curr_xyz, np.max(neighbor_d))
    planarity_score = compute_planarscore(neighborhood_xyz, curr_xyz)
    # print(f'Center score:{center_score}\n Planarity score:{planarity_score}\n')
    # print(f'Elapsed:{t-time.time()}')

    center_score_arr[point_idx] = center_score
    planar_score_arr[point_idx] = planarity_score
