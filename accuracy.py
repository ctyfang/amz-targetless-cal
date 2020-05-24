from calibration.camera_lidar_calibrator import CameraLidarCalibrator
from calibration.utils.config import command_line_parser
from calibration.utils.data_utils import getPath, perturb_tau
import pickle
import cv2
import sys
import time
import numpy as np
import csv


def fields():
    list_of_fields = ['Initial Cost', 'Final Cost',
                      'GT_axis_angle_x', 'GT_axis_angle_y', 'GT_axis_angle_z', 'GT_translation_x', 'GT_translation_y', 'GT_translation_z',
                      'Init_guess_axis_angle_x', 'Init_guess_axis_angle_y', 'Init_guess_axis_angle_z', 'Init_guess_translation_x', 'Init_guess_translation_y', 'Init_guess_translation_z',
                      'Optimized_axis_angle_x', 'Optimized_axis_angle_y', 'Optimized_axis_angle_z', 'Optimized_translation_x', 'Optimized_translation_y', 'Optimized_translation_z',
                      'Init_guess_translation_error_x', 'Init_guess_translation_error_y', 'Init_guess_translation_error_z', 'Init_guess_angle_error', 'Init_guess_angle_error_axis_x', 'Init_guess_angle_error_axis_y', 'Init_guess_angle_error_axis_z',
                      'Optimized_translation_error_x', 'Optimized_translation_error_y', 'Optimized_translation_error_z', 'Optimized_angle_error', 'Optimized_angle_error_axis_x', 'Optimized_angle_error_axis_y', 'Optimized_angle_error_axis_z']
    return list_of_fields


def evaluate_accuracy(gt_tau, init_tau, optimized_tau, cost_history):
    list = []
    list.append(cost_history[0])
    list.append(cost_history[-1])
    list.extend(gt_tau.tolist())
    list.extend(init_tau.tolist())
    list.extend(optimized_tau.tolist())

    # Calculate the deviation of initial guess from the ground truth
    init_tau_translation_error = gt_tau[3:] - init_tau[3:]

    gt_rot_mat, _ = cv2.Rodrigues(gt_tau[:3])
    init_tau_rot_mat, _ = cv2.Rodrigues(init_tau[:3])
    init_tau_rotation_error, _ = cv2.Rodrigues(gt_rot_mat @ init_tau_rot_mat.T)
    init_tau_angle_error = np.linalg.norm(init_tau_rotation_error)
    init_tau_angle_error_axis = init_tau_rotation_error / init_tau_angle_error

    list.extend(init_tau_translation_error.tolist())
    list.append(np.degrees(init_tau_angle_error))
    list.extend(np.squeeze(init_tau_angle_error_axis).tolist())

    # Calculate the deviation of the optimized value from the ground truth
    optimized_tau_translation_error = gt_tau[3:] - optimized_tau[3:]

    optimized_tau_rot_mat, _ = cv2.Rodrigues(optimized_tau[:3])
    optimized_tau_rotation_error, _ = cv2.Rodrigues(
        gt_rot_mat @ optimized_tau_rot_mat.T)
    optimized_tau_angle_error = np.linalg.norm(optimized_tau_rotation_error)
    optimized_tau_angle_error_axis = optimized_tau_rotation_error / \
        optimized_tau_angle_error

    list.extend(optimized_tau_translation_error.tolist())
    list.append(np.degrees(optimized_tau_angle_error))
    list.extend(np.squeeze(optimized_tau_angle_error_axis).tolist())

    return list


input_dir_list = ['/media/carter/Samsung_T5/3dv/2011_09_28/2011_09_28_drive_0035_sync',
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

cfg.frames = [20, 26, 48, 64, 77, 99, 224, 266, 374, 428]
# cfg.frames = [26, 374, 428]
# cfg.frames = [34, 35, 185]


DETECT_FROM_SCRATCH = True
SAVE_IMAGES = True
CALIBRATION_OBJECT_PATH = './output/calibrator_imgthresh200-300_pcthresh04_10_imgs_dataset.pkl'

if DETECT_FROM_SCRATCH:
    # Detect edges from scratch
    calibrator = CameraLidarCalibrator(cfg, visualize=True)
    with open(CALIBRATION_OBJECT_PATH, 'wb') as output_pkl:
        pickle.dump(calibrator, output_pkl, pickle.HIGHEST_PROTOCOL)
else:
    # Load calibration object
    with open(CALIBRATION_OBJECT_PATH, 'rb') as input_pkl:
        calibrator = pickle.load(input_pkl)
        calibrator.visualize = True

for frame in range(len(cfg.frames)):
    calibrator.draw_edge_points(
        score=calibrator.pc_detector.pcs_edge_scores, frame=frame, append_string='gt_', show=False, save=SAVE_IMAGES)

gt_calibration = calibrator.tau.copy()

with open('output/calibration_runs_{}.txt'.format(time.strftime("%Y%m%d-%H%M%S")), 'w') as csv_file:

    csv_writer = csv.writer(csv_file)

    csv_writer.writerow(fields())
    csv_file.flush()

    for i in range(10):

        initial_guess = perturb_tau(gt_calibration, 0.05, 5)

        calibrator.tau = initial_guess.copy()

        calibrator.project_point_cloud()

        for frame in range(len(cfg.frames)):
            calibrator.draw_edge_points(
                score=calibrator.pc_detector.pcs_edge_scores, frame=frame, append_string='init_guess_', save=SAVE_IMAGES)

        optimized_tau, cost_history = calibrator.ls_optimize(120)

        for frame in range(len(cfg.frames)):
            calibrator.draw_edge_points(
                score=calibrator.pc_detector.pcs_edge_scores, frame=frame, append_string='optimized_', save=SAVE_IMAGES)

        print("The optimized tau is: {}".format(optimized_tau))
        print("The former tau is: {}".format(initial_guess))
        print("Cost history initial value: {}".format(cost_history[:1]))
        print("Cost history final value: {}".format(cost_history[-1:]))
        print("---------------------------------------------")

        # Calculate relevant values to evaluate accuracy and format to a row of the csv file
        row = evaluate_accuracy(
            gt_calibration, initial_guess, optimized_tau, cost_history)
        csv_writer.writerow(row)
        csv_file.flush()


print("Done writing to {}".format(csv_file.name))
