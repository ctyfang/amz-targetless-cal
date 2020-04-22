import numpy as np
import open3d as o3d
from data_utils import *
import os

data_path = '/media/carter/Samsung_T5/3dv/2011_09_26/2011_09_26_drive_0106_sync/velodyne_points/data'
o3d_view_path = './configs/o3d_camera_model.json'
file_idx = 29
pc_xyz = load_from_bin(os.path.join(data_path, str(file_idx).zfill(10) + '.bin'))
camera_params = o3d.io.read_pinhole_camera_parameters(o3d_view_path)

custom_draw_xyz_with_params(pc_xyz, camera_params)



