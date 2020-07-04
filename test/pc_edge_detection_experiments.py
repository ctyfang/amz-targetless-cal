from calibration.camera_lidar_calibrator import *
from calibration.utils.config import command_line_parser
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import open3d as o3d
import pickle
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from scipy import signal


def visualize_point_cloud(point_cloud, scores, cmap=cm.tab20c):
    vmin = np.min(scores)
    vmax = np.max(scores)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = np.asarray(mapper.to_rgba(scores))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()


##### FLAGS ####
VISUALIZE_POLAR_ANGLE_SCATTER_PLOT = True
VISUALIZE_SEGMENTED_POINT_CLOUD = False
################


input_dir_list = ['/media/carter/Samsung_T5/3dv/2011_09_28/collection',
                  '/media/benjin/Data/Data/3dv_kitti_2011_09_26_static_scenes/2011_09_26',
                  'data/2011_09_26_0017']
calib_dir_list = ['/media/carter/Samsung_T5/3dv/2011_09_28/calibration',
                  '/media/benjin/Data/Data/3dv_kitti_2011_09_26_calib/2011_09_26',
                  'data']

cfg = command_line_parser()

input_dir = getPath(input_dir_list)
calib_dir = getPath(calib_dir_list)

cfg.pc_dir = input_dir
cfg.img_dir = input_dir
cfg.calib_dir = calib_dir

# Create calibrator and detect edges from scratch
# calibrator = CameraLidarCalibrator(cfg, visualize=False)
# with open('./output/calibrator_collection-8-0928.pkl', 'wb') as output_pkl:
#     pickle.dump(calibrator, output_pkl, pickle.HIGHEST_PROTOCOL)

# # Load calibrator with detected edges from pickled object
with open('./output/calibrator_imgthresh200-300_pcthresh04_3_imgs_dataset_34.pkl', 'rb') as input_pkl:
    calibrator = pickle.load(input_pkl)
    calibrator.visualize = True

# Velodyne Point Cloud
pc1 = calibrator.pc_detector.pcs[0]

# Calculate the polar angle for every point
polar_angle = 180 * np.arctan2(pc1[:, 2], np.sqrt(
    np.square(pc1[:, 0]) + np.square(pc1[:, 1]))) / np.pi

# Calculate 64 cluster means
max_polar_angle = np.max(polar_angle)
min_polar_angle = np.min(polar_angle)
# Calculate an initial guess
init_guess = np.linspace(min_polar_angle, max_polar_angle, num=64)
kmeans = KMeans(n_clusters=64, init=init_guess.reshape(-1, 1)
                ).fit(polar_angle.reshape(-1, 1))

# Visualize the angles in a scatter plot
if VISUALIZE_POLAR_ANGLE_SCATTER_PLOT:
    rng = np.random.default_rng()
    y = rng.random(size=polar_angle.shape)

    fig, ax = plt.subplots(figsize=(20, 10))

    # Modify labels for better visualization
    modulo_vector = np.ones(shape=kmeans.labels_.shape) * 20
    labels = np.mod(kmeans.labels_.copy(), modulo_vector)

    ax.scatter(polar_angle, y, s=1, c=labels, cmap='tab20c',
               alpha=0.75)

    plt.tight_layout(pad=3.0)
    plt.show()

# Visualize point cloud
if VISUALIZE_SEGMENTED_POINT_CLOUD:
    modulo_vector = np.ones(shape=kmeans.labels_.shape) * 20
    labels = np.mod(kmeans.labels_.copy(), modulo_vector)
    visualize_point_cloud(pc1, labels)

# Fill list with the idxs of edges
edge_idxs = []

for ring in range(64):
    current_ring_filter = (kmeans.labels_ == ring)
    current_ring_idxs = np.argwhere(current_ring_filter)
    current_ring = pc1[current_ring_filter]
    # For keeping track of which points are edges
    current_ring_order = np.arange(current_ring.shape[0])

    # Calculate the azimuth angles of the ring
    azimuth_angle = 180 * \
        np.arctan2(current_ring[:, 1], current_ring[:, 0]) / np.pi

    sort_by_azimuth_angle = np.argsort(azimuth_angle)

    azimuth_angle_sorted = azimuth_angle[sort_by_azimuth_angle]

    current_ring_sorted = current_ring[sort_by_azimuth_angle]

    current_ring_order = current_ring_order[sort_by_azimuth_angle]

    # Visualize current ring colored by azimuth value
    VISUALIZE_CURRENT_RING = False
    if VISUALIZE_CURRENT_RING:
        visualize_point_cloud(
            current_ring_sorted, azimuth_angle_sorted, cmap=cm.summer)

    # Calculate the depth of each point
    depth = np.linalg.norm(current_ring_sorted, axis=1)

    # rml: depth of right point minus depth of left point
    delta_depth_rml = depth - np.roll(depth, -1)
    # lmr: depth of left point minus depth of right point
    delta_depth_lmr = depth - np.roll(depth, +1)

    # Visualize depth, and delta depth against azimuth_angle_sorted
    VISUALIZE_DELTA_DEPTH_AGAINST_AZIMUTH_ANGLE = False
    if VISUALIZE_DELTA_DEPTH_AGAINST_AZIMUTH_ANGLE:
        fig, ax = plt.subplots(figsize=(20, 10))

        ax.scatter(azimuth_angle_sorted[:1000], depth[:1000], s=1, alpha=0.75)

        ax.scatter(azimuth_angle_sorted[:1000],
                   delta_depth_lmr[:1000], s=1, c='red')

        plt.tight_layout(pad=3.0)
        plt.show()

    edges_filter = np.logical_or(
        delta_depth_rml < -0.5, delta_depth_lmr < -0.5)

    current_edges = current_ring_sorted[edges_filter]

    current_ring_order = current_ring_order[edges_filter]

    # Get the idxs that are edges
    current_ring_edge_idxs = current_ring_idxs[current_ring_order]

    # Visualize edges in current ring
    VISUALIZE_EDGES_IN_RING = True
    if VISUALIZE_EDGES_IN_RING:
        visualize_point_cloud(
            current_ring_sorted, edges_filter, cmap=cm.summer)

    # Delete current edges if too many are detected
    if current_edges.shape[0] > 500:
        continue

    edge_idxs.extend(current_ring_edge_idxs.tolist())


pc_mask = np.zeros(pc1.shape[0])
pc_mask[edge_idxs] = True
# Visualize edges in whole point cloud
visualize_point_cloud(pc1, pc_mask, cmap='summer')


print('Done')
