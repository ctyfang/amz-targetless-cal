"""Utility functions for handling images and pointclouds"""
# Deep Learning for Autonomous Driving
# Material for the 3rd and 4th problem of Project 1
# For further questions contact Dengxin Dai (dai@vision.ee.ethz.ch) or Ozan Unal (ouenal@ee.ethz.ch)

import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial import ckdtree
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm as cm

from copy import deepcopy
from random import randint
from time import sleep

import scipy
import json
import os
from os.path import dirname, abspath
import argparse
import math as m


def visualize_xyz_scores(xyz, scores):
    norm = matplotlib.colors.Normalize(vmin=np.min(scores), vmax=np.max(scores), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    colors = np.asarray([mapper.to_rgba(x)[:3] for x in scores])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()

def compute_centerscore(nn_xyz, center_xyz, max_nn_d):
    # Description: Compute modified center-based score. Distance between center (center_xyz),
    #              and the neighborhood (N x 3) around center_xyz. Result is scaled by max
    #              distance in the neighborhood, as done in Kang 2019.

    centroid = np.mean(nn_xyz, axis=0)
    norm_dist = np.linalg.norm(center_xyz - centroid)/max_nn_d
    return norm_dist


def compute_planarscore(nn_xyz, center_xyz):

    complete_xyz = np.concatenate([nn_xyz, center_xyz.reshape((1, 3))], axis=0)
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
    planarity = (eig_vals[1] - eig_vals[2])/eig_vals[0]

    return planarity

def visualize_neighborhoods(xyz):
    kdtree = ckdtree.cKDTree(xyz)

    colors = np.zeros((xyz.shape[0], 3))
    colors[:, 0] = 1

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    def highlight_neighborhood(vis):
        point_idx = randint(0, xyz.shape[0])
        new_colors = deepcopy(colors)
        new_colors[point_idx] = [0, 1, 0]

        neighbors_d, neighbors_i = kdtree.query(xyz[point_idx, :], 1000)
        new_colors[neighbors_i] = [0, 1, 0]
        pcd.colors = o3d.utility.Vector3dVector(new_colors)
        vis.update_geometry(pcd)
        sleep(0.5)

    o3d.visualization.draw_geometries_with_animation_callback([pcd], highlight_neighborhood)


def custom_draw_xyz_with_params(xyz, camera_params):
    # Construct pcd object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pc', visible=True, width=camera_params.intrinsic.width, height=camera_params.intrinsic.height)
    ctr = vis.get_view_control()
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(camera_params)
    vis.run()


def custom_draw_geometry_with_key_callbacks(pcd):

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def load_render_option(vis):
        vis.get_render_option().load_from_json(
            "../../TestData/renderoption.json")
        return False

    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False

    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.show()
        return False

    def save_camera_model(vis):
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters('./configs/o3d_camera_model.json', param)

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord("R")] = load_render_option
    key_to_callback[ord(",")] = capture_depth
    key_to_callback[ord(".")] = capture_image
    key_to_callback[ord("C")] = save_camera_model
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)

def load_from_bin(bin_path):
    # load point cloud from a binary file
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # ignore reflectivity info
    return obj[:, :3]


def depth_color(val, min_d=0, max_d=120):
    """
    print Color(HSV's H value) corresponding to distance(m)
    close distance = red , far distance = blue
    """
    np.clip(val, 0, max_d, out=val)  # max distance is 120m but usually not usual
    return (((val - min_d) / (max_d - min_d)) * 120).astype(np.uint8)


def line_color(val, min_d=1, max_d=64):
    """
    print Color(HSV's H value) corresponding to laser id
    """
    alter_num = 4
    return (((val - min_d)%alter_num) * 127/alter_num).astype(np.uint8)


def calib_velo2cam(filepath):
    """
    get Rotation(R : 3x3), Translation(T : 3x1) matrix info
    using R,T matrix, we can convert velodyne coordinates to camera coordinates
    """
    with open(filepath, "r") as f:
        file = f.readlines()

        for line in file:
            (key, val) = line.split(':', 1)
            if key == 'R':
                R = np.fromstring(val, sep=' ')
                R = R.reshape(3, 3)
            if key == 'T':
                T = np.fromstring(val, sep=' ')
                T = T.reshape(3, 1)
    return R, T

def calib_imu2velo(filepath):
    """
    get Rotation(R : 3x3), Translation(T : 3x1) matrix info
    using R,T matrix, we can convert imu coordinates to velodyne coordinates
    """
    with open(filepath, "r") as f:
        file = f.readlines()

        for line in file:
            (key, val) = line.split(':', 1)
            if key == 'R':
                R = np.fromstring(val, sep=' ')
                R = R.reshape(3, 3)
            if key == 'T':
                T = np.fromstring(val, sep=' ')
                T = T.reshape(3, 1)
    return R, T

def calib_cam2cam(filepath, mode):
    """
    If your image is 'rectified image' :
        get only Projection(P : 3x4) matrix is enough
    but if your image is 'distorted image'(not rectified image) :
        you need undistortion step using distortion coefficients(5 : D)

    in this code, we'll get P matrix since we're using rectified image.
    in this code, we set filepath = 'yourpath/2011_09_26_drive_0029_sync/calib_cam_to_cam.txt' and mode = '02'
    """
    with open(filepath, "r") as f:
        file = f.readlines()

        for line in file:
            (key, val) = line.split(':', 1)
            if key == ('P_rect_' + mode):
                P_ = np.fromstring(val, sep=' ')
                P_ = P_.reshape(3, 4)
                # erase 4th column ([0,0,0])
                P_ = P_[:3, :3]
    return P_



def print_projection_plt(points, color, image):
    """ project converted velodyne points into camera image """

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(points.shape[1]):
        cv2.circle(hsv_image, (np.int32(points[0][i]), np.int32(points[1][i])), 2, (int(color[i]), 255, 255), -1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)



def compute_timestamps(timestamps_f, ind):
    # return timestamps of the the ind^th sample (line) in seconds
    # in this code, timestamps_f can be 'image_02/timestamps.txt', 'oxts/timestamps.txt', 'velodyne_points/timestamps_start.txt', ...
    #  'velodyne_points/timestamps_end.txt',  or 'velodyne_points/timestamps.txt'. ind is the index (name) of the sample like '0000000003'
    with open(timestamps_f) as f:
        timestamps_ = f.readlines()
        #file_id = file[7:10]
        timestamps_ = timestamps_[int(ind)]
        timestamps_ = timestamps_[11:]
        timestamps_ = np.double(timestamps_[:2]) * 3600 + np.double(timestamps_[3:5]) * 60 + np.double(timestamps_[6:])
    return timestamps_



def load_oxts_velocity(oxts_f):
    # return the speed of the vehicle given the oxts file
    with open(oxts_f) as f:
        data = [list(map(np.double, line.strip().split(' '))) for line in f]
        speed_f = data[0][8]
        speed_l = data[0][9]
        speed_u = data[0][10]
    return np.array((speed_f, speed_l, speed_u))


def load_oxts_angular_rate(oxts_f):
    # return the angular rate of the vehicle given the oxts file
    with open(oxts_f) as f:
        data = [list(map(np.double, line.strip().split(' '))) for line in f]
        angular_rate_f = data[0][20]
        angular_rate_l = data[0][21]
        angular_rate_u = data[0][22]
    return angular_rate_f, angular_rate_l, angular_rate_u

