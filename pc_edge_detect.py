import numpy as np
import open3d as o3d
from data_utils import load_from_bin
import os

data_path = '/media/carter/Samsung_T5/3dv/2011_09_26/2011_09_26_drive_0106_sync/velodyne_points/data'
file_idx = 29
pc_xyz = load_from_bin(os.path.join(data_path, str(file_idx).zfill(10) + '.bin'))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pc_xyz)

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
vis.run()

