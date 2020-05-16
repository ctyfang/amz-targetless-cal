import numpy as np

from calibration.camera_lidar_calibrator import CameraLidarCalibrator

def loss(tau_quat_init, calibrator, sigma_in, alpha_mi, alpha_gmm, cost_history):
    calibrator.tau = CameraLidarCalibrator.tauquat_to_tau(tau_quat_init)
    calibrator.project_point_cloud()

    cost_mi = alpha_mi * calibrator.compute_mi_cost()
    cost_gmm = alpha_gmm * calibrator.compute_conv_cost(sigma_in)

    cost_history.append(cost_gmm + cost_mi)
    # print(cost_history[-1])
    return cost_history[-1]


def loss_translation(trans_vec_init, calibrator, sigma_in, alpha_mi, alpha_gmm, cost_history):
    calibrator.tau[3:] = trans_vec_init
    calibrator.project_point_cloud()

    cost_mi = alpha_mi * calibrator.compute_mi_cost()
    cost_gmm = alpha_gmm * calibrator.compute_conv_cost(sigma_in)

    cost_history.append(cost_gmm + cost_mi)
    # print(cost_history[-1])
    return cost_history[-1]


class BadProjection(Exception):
    """Bad Projection exception"""
