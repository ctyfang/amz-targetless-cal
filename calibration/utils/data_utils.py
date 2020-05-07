"""Utility functions for working with the KITTI dataset"""
import numpy as np
import cv2
import os
import sys


def load_cam_cal(calib_dir):
    """Get camera matrix for camera 0 given directory with KITTI calibration files"""
    with open(str(calib_dir) + '/calib_cam_to_cam.txt', "r") as file:
        lines = file.readlines()

        for line in lines:
            (key, val) = line.split(':', 1)
            if key == ('P_rect_' + "00"):
                P_rect = np.fromstring(val, sep=' ') \
                                  .reshape(3, 4)[:3, :3]

                return P_rect


def load_lid_cal(calib_dir):
    """Get [R, T] that brings points from LiDAR frame to camera 0 frame."""
    with open(str(calib_dir) + '/calib_velo_to_cam.txt', "r") as file:
        lines = file.readlines()

        for line in lines:
            (key, val) = line.split(':', 1)
            if key == 'R':
                R = np.fromstring(val, sep=' ').reshape(3, 3)
            if key == 'T':
                T = np.fromstring(val, sep=' ').reshape(3, 1)

    return [R, T]


def getPath(list_of_paths):
    for path in list_of_paths:
        if os.path.isdir(path):
            return path
    print("ERROR: Data paths don't exist!")
    sys.exit()


def load_from_bin(bin_path):
    # load point cloud from a binary file
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    # ignore reflectivity info
    return obj[:, :3]


def load_from_csv(csv_path, delimiter=',', skip_header=0):
    # load point cloud from a csv file
    obj = np.genfromtxt(csv_path, delimiter=delimiter, skip_header=skip_header)
    # ignore unnecessary indices in first column
    return obj[:, 1:4]


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
    return (((val - min_d) % alter_num) * 127/alter_num).astype(np.uint8)


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
        cv2.circle(hsv_image, (np.int32(points[0][i]), np.int32(
            points[1][i])), 2, (int(color[i]), 255, 255), -1)

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
        timestamps_ = np.double(timestamps_[
                                :2]) * 3600 + np.double(timestamps_[3:5]) * 60 + np.double(timestamps_[6:])
    return timestamps_


def skew(vector):
    """
    this function returns a numpy array with the skew symmetric cross product matrix for vector.
    the skew symmetric cross product matrix is defined such that
    np.cross(a, b) = np.dot(skew(a), b)

    :param vector: An array like vector to create the skew symmetric cross product matrix for
    :return: A numpy array of the skew symmetric cross product vector
    """

    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


def jacobian(omega):
    """Given rotation vector compute Jacobian (3x3). ith column represents derivative of camera frame
       position wrt rotation vector"""

    omega_mag = np.linalg.norm(omega, 2)
    a = (omega/omega_mag).reshape((3, 1))


    jac = (np.sin(omega_mag)/omega_mag)*np.eye(3) + (1-np.sin(omega_mag)/omega_mag)*np.dot(a, a.T) + \
          ((1-np.cos(omega_mag))/omega_mag)*skew(a)

    return jac
