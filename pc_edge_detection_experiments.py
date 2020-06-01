from calibration.camera_lidar_calibrator import *
from calibration.utils.config import command_line_parser
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

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

# Calculate 64 means
# Calculate an initial guess
max_polar_angle = np.max(polar_angle)
min_polar_angle = np.min(polar_angle)
init_guess = np.linspace(min_polar_angle, max_polar_angle, num=64)
kmeans = KMeans(n_clusters=64, init=init_guess.reshape(-1, 1)
                ).fit(polar_angle.reshape(-1, 1))

# Visualize the angles in a scatter plot
VISUALIZE_POLAR_ANGLES = True
if VISUALIZE_POLAR_ANGLES:
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

# Fill list with np arrays that contain the points of one specific ring
rings_list = []

for ring in range(64):
    rings_list.append(polar_angle[kmeans.labels_ == ring])


print('Done')
