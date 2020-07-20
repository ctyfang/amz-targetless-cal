"""Utility functions for working with the KITTI dataset"""
import numpy as np
from sklearn.metrics import mutual_info_score
from pyquaternion import Quaternion
import cv2
import os
import sys

from copy import deepcopy

def load_cam_cal(calib_dir):
    """
    Given the directory path with the calibration file 'calib_cam_to_cam.txt',
    open it and extract camera 0's intrinsic matrix from the projection
    matrix

    :param calib_dir: str, directory path with calibration txt file
    :return: (3, 3) numpy array of camera intrinsics
    """
    with open(str(calib_dir) + '/calib_cam_to_cam.txt', "r") as file:
        lines = file.readlines()

        for line in lines:
            (key, val) = line.split(':', 1)
            if key == ('P_rect_' + "00"):
                P_rect = np.fromstring(val, sep=' ') \
                                  .reshape(3, 4)[:3, :3]

                return P_rect


def load_lid_cal(calib_dir):
    """
    Given the directory path with the calibration file 'calib_velo_to_cam.txt',
    open it and extract the extrinsics from LiDAR to camera

    :param calib_dir: str, directory path with calibration txt file
    :return: [R, T], (3x3) rotation matrix R, (3x1) translation vector T
    """
    with open(str(calib_dir) + '/calib_velo_to_cam.txt', "r") as file:
        lines = file.readlines()

        for line in lines:
            (key, val) = line.split(':', 1)
            if key == 'R':
                R = np.fromstring(val, sep=' ').reshape(3, 3)
            if key == 'T':
                T = np.fromstring(val, sep=' ').reshape(3, 1)

    return [R, T]


def load_from_bin(bin_path, incl_refl=False):
    """
    Given path to a pointcloud BIN file, load the XYZ data. Assumes XYZ
    are the first three columns. Also returns reflectance if incl_refl
    boolean is specified.

    :param bin_path: str, path to pointcloud data file
    :param incl_refl: boolean, whether to extract reflectance from the pc or not
    :return: (N, 3) numpy array if only XYZ, (N, 4) if incl_refl is true
    """
    # load point cloud from a binary file
    obj = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

    if incl_refl:
        return obj
    else:
        return obj[:, :3]


def load_from_csv(path, delimiter=',', skip_header=0, incl_refl=False):
    """
    Given path to a pointcloud CSV or TXT file, load the XYZ data. Assumes XYZ
    are the first three columns. Also returns reflectance if incl_refl
    boolean is specified.

    :param bin_path: str, path to pointcloud data file
    :param incl_refl: boolean, whether to extract reflectance from the pc or not
    :return: (N, 3) numpy array if only XYZ, (N, 4) if incl_refl is true
    """
    _, ext = os.path.splitext(path)
    if ext == '.csv':
        obj = np.genfromtxt(path, delimiter=delimiter,
                            skip_header=skip_header)
        return obj[:, 1:4]

    elif ext == '.txt':
        obj = np.genfromtxt(path, skip_header=skip_header)
        return obj[:, :]

    else:
        print("Unsupported extension for pc data.")


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


def print_projection_plt(points, color, image):
    """ project converted velodyne points into camera image """

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    for i in range(points.shape[1]):
        cv2.circle(hsv_image, (np.int32(points[0][i]), np.int32(
            points[1][i])), 2, (int(color[i]), 255, 255), -1)

    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)


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


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert euler angles in radians to a quaternion.

    :param roll
    :param pitch
    :param yaw
    :return: (4,) array of quaternion values
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return np.asarray([qw, qx, qy, qz]).reshape((4,))

def perturb_tau(tau_in, trans_range=0.05, angle_range=2.5):
    """
    Given a (6, ) extrinsics array, tau_in, and ranges for the translation
    and rotation disturbances, perturb tau_in and return the perturbed array.
    tau_in[0:3] is a rotation vector. tau_in[3:] is a translation vector.
    Rotation and translation errors are sampled uniformly. Translation error
    is added to the translation vector. A noise rotation is generated using the
    angle errors as euler angles.

    :param tau_in: initial extrinsics vector
    :param trans_range: range in meters to perturb the x, y, and z axes
    :param angle_range: range in degrees to perturb the rotation parameters
    :return: tau_perturbed
    """

    # Unpack tau
    trans_vec = deepcopy(tau_in[3:])
    rot_vec = deepcopy(tau_in[:3])
    angle = np.linalg.norm(rot_vec, 2)
    axis = rot_vec/angle

    # Sample noise
    x_noise = np.random.uniform(-trans_range, trans_range)
    y_noise = np.random.uniform(-trans_range, trans_range)
    z_noise = np.random.uniform(-trans_range, trans_range)
    rot_x_noise = np.deg2rad(np.random.uniform(-angle_range, angle_range))
    rot_y_noise = np.deg2rad(np.random.uniform(-angle_range, angle_range))
    rot_z_noise = np.deg2rad(np.random.uniform(-angle_range, angle_range))

    # Add noise by sampling an added noisy rotation
    R_noise = Quaternion(euler_to_quaternion(rot_x_noise, rot_y_noise, rot_z_noise)).rotation_matrix
    R_old = Quaternion(axis=axis, angle=angle).rotation_matrix
    R_new = np.dot(R_noise, R_old)
    quat_new = Quaternion(matrix=R_new)
    angle_new = quat_new.angle
    axis_new = quat_new.axis
    rot_vec_new = angle_new*axis_new

    # Apply noise
    trans_vec_new = trans_vec + [x_noise, y_noise, z_noise]
    # angle += angle_noise
    # new_rot_vec = angle*axis
    return np.asarray([rot_vec_new, trans_vec_new]).reshape(tau_in.shape)


def get_initial_simplex(x0, nonzdelt=0.05, zdelt=0.00025):
    """
    Generate a simplex from an initial guess by modifying each dimension of the
    initial guess. For x0 (D, ), a (D+1, D) simplex is generated. In each
    dimension, if x0(i) == 0, the axis is scaled by (1+zdelt), else it
    is scaled by (1+nonzdelt).

    :param x0: Initial point to build simplex around.
    :param deltas: scaling parameter for non-zero axes
    :param zdelt: scaling parameter for zero axes
    :return: (D+1, D) simplex
    """
    D = np.max(x0.shape)
    simplex = np.zeros((D+1, D))
    simplex[0, :] = x0

    for i in range(0, D):
        x = x0.copy()
        if x[i] == 0:
            x[i] = (1 + zdelt)*x[i]
        else:
            x[i] = (1 + nonzdelt)*x[i]
        simplex[i+1, :] = x
    return simplex


def get_mixed_delta_simplex(x0, deltas, zdelt=[0.00025, 0.00025, 0.00025,
                                               0.00025, 0.00025, 0.00025],
                            scales=np.ones((1, 6))):
    """
    Generate a simplex from an initial guess by modifying each dimension of the
    initial guess. For x0 (D, ), a (D+1, D) simplex is generated. In each
    dimension, if x0(i) == 0, the axis is scaled by (1+zdelt(i)), else it
    is scaled by (1+deltas(i)).

    :param x0: Initial point to build simplex around.
    :param deltas: (D, ) array of scaling parameters for non-zero axes
    :param zdelt: (D, ) array of scaling parameters for zero axes
    :param scales: (D, ) Optional output scaling of the final simplex vertices
    :return: (D+1, D) simplex
    """
    D = np.max(x0.shape)
    simplex = np.zeros((D+1, D))
    simplex[0, :] = np.divide(x0, scales)

    for i in range(0, D):
        x = np.squeeze(x0).copy()
        if x[i] == 0:
            x[i] = (1 + zdelt[i])*x[i]
        else:
            x[i] = (1 + deltas[i])*x[i]

        x = np.divide(x, scales)
        simplex[i+1, :] = x
    return simplex
