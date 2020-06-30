from calibration.camera_lidar_calibrator import *
from calibration.utils.config import command_line_parser
from calibration.utils.data_utils import *
from calibration.utils.img_utils import *
from calibration.utils.exp_utils import *

import pickle
import json
from prettytable import PrettyTable

"""Script parameters"""
save_every = 1
select_new_correspondences = False
calibrator_path = '../generated/calibrators/0928-6frames.pkl'
LOG_DIR = '../generated/optimizer_logs/test'

"""Experiment parameters"""
exp_params = {'NUM_SAMPLES': 10, 'TRANS_ERR_SIGMA': 0.10, 'ANGLE_ERR_SIGMA': 5,
              'ALPHA_MI': [1], 'ALPHA_GMM': [5],
              'ALPHA_POINTS': [0], 'ALPHA_CORR': [0],
              'SIGMAS': [7.0],
              'MAX_ITERS': 100, 'SCALES': np.power(10*np.ones((1, 6)),
                                                   [0, 0, 0, -2, -2, -2]).tolist()}

"""Default Hyperparameter Template"""
hyperparams = {'alphas': {'mi': 1, 'gmm': 5, 'points': 0, 'corr': 0},
               'scales': exp_params['SCALES']}

"""Calibration directory specification"""
calib_dir_list = ['/media/carter/Samsung_T5/3dv/2011_09_28/calibration',
                  '/home/benjin/Development/Data/2011_09_26_calib/2011_09_26',
                  'data',
                  '/home/carter/pycharm_project_534/data/calibration',
                  './data/calibration']

calib_dir = getPath(calib_dir_list)

"""Load pre-made calibrator object"""
if os.path.exists(calibrator_path):
    with open(calibrator_path, 'rb') as input_pkl:
        calibrator = pickle.load(input_pkl)
        calibrator.visualize = True
else:
    print('Calibrator does not exist at specified path!')
    exit()

"""Load ground-truth extrinsics"""
R, T = load_lid_cal(calib_dir)
tau_gt = calibrator.tau = calibrator.transform_to_tau(R, T)
exp_params['tau_gt'] = tau_gt.tolist()

"""Select correspondences and save calibrator"""
if select_new_correspondences:
    calibrator.select_correspondences()
    with open(calibrator_path, 'wb') as overwrite_pkl:
        pickle.dump(calibrator, overwrite_pkl)

"""Create log directory and save GT projections"""
os.makedirs(LOG_DIR, exist_ok=True)
with open(os.path.join(LOG_DIR, 'params.json'), 'w') as json_file:
    json.dump(exp_params, json_file, indent=4)

os.makedirs(os.path.join(LOG_DIR, 'gt'), exist_ok=True)
calibrator.project_point_cloud()
for frame_idx in range(len(calibrator.img_detector.imgs)):
    img_all_points = calibrator.draw_all_points(frame=frame_idx)
    img_edges = calibrator.draw_edge_points(frame=frame_idx,
                                            image=calibrator.img_detector.imgs_edges[frame_idx])
    cv.imwrite(os.path.join(LOG_DIR, 'gt', f'all_points_frame_{frame_idx}.jpg'), img_all_points)
    cv.imwrite(os.path.join(LOG_DIR, 'gt', f'edge_points_frame_{frame_idx}.jpg'), img_edges)

"""Run the experiment"""
tau_data = []
tau_inits = []
for sample_idx in range(exp_params['NUM_SAMPLES']):
    print(f'----- SAMPLE {sample_idx} -----')
    calibrator.update_extrinsics(perturb_tau(tau_gt,
                                 trans_std=exp_params['TRANS_ERR_SIGMA'],
                                 angle_std=exp_params['ANGLE_ERR_SIGMA']))
    compare_taus(tau_gt, calibrator.tau)

    calibrator.project_point_cloud()
    tau_inits.append(calibrator.tau)

    """Make directory for this trial"""
    os.makedirs(os.path.join(LOG_DIR, f'trial_{sample_idx}'), exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, f'trial_{sample_idx}', 'initial'), exist_ok=True)

    """Save initial images"""
    for frame_idx in range(len(calibrator.img_detector.imgs)):
        img_all_points = calibrator.draw_all_points(frame=frame_idx)
        img_edges = calibrator.draw_edge_points(frame=frame_idx,
                                                image=calibrator.img_detector.imgs_edges[frame_idx])
        cv.imwrite(os.path.join(LOG_DIR, f'trial_{sample_idx}', 'initial',
                                f'all_points_{sample_idx}_frame_{frame_idx}.jpg'), img_all_points)
        cv.imwrite(os.path.join(LOG_DIR, f'trial_{sample_idx}', 'initial',
                                f'edge_points_{sample_idx}_frame_{frame_idx}.jpg'), img_edges)

    """Run optimizer"""
    for stage_idx in range(len(exp_params["SIGMAS"])):
        sigma_in, alpha_gmm, alpha_mi, alpha_points, alpha_corr = \
            exp_params['SIGMAS'][stage_idx], \
            exp_params['ALPHA_GMM'][stage_idx], \
            exp_params['ALPHA_MI'][stage_idx], \
            exp_params['ALPHA_POINTS'][stage_idx], \
            exp_params['ALPHA_CORR'][stage_idx]

        curr_hyperparams = hyperparams.copy()
        curr_hyperparams['alphas']['mi'] = alpha_mi
        curr_hyperparams['alphas']['gmm'] = alpha_gmm
        curr_hyperparams['alphas']['points'] = alpha_points
        curr_hyperparams['alphas']['corr'] = alpha_corr
        curr_hyperparams['alphas']['sigma'] = sigma_in

        tau_opt, cost_history = calibrator.ls_optimize(hyperparams,
                                                       maxiter=exp_params['MAX_ITERS'],
                                                       save_every=save_every)
        calibrator.tau = tau_opt
        compare_taus(tau_gt, tau_opt)
        calibrator.project_point_cloud()

    """Save results from this optimization stage"""
    tau_data.append(np.squeeze(tau_opt))

    """Save projections with final optimized tau"""
    os.makedirs(os.path.join(LOG_DIR, f'trial_{sample_idx}', f'stage_{stage_idx}'), exist_ok=True)
    plt.figure()
    plt.title('Loss history')
    plt.plot(range(len(cost_history)), cost_history)
    plt.savefig(os.path.join(LOG_DIR, f'trial_{sample_idx}', f'stage_{stage_idx}', f'loss_history.png'))
    plt.show()

    calibrator.project_point_cloud()
    for frame_idx in range(len(calibrator.img_detector.imgs)):
        os.makedirs(os.path.join(LOG_DIR, f'trial_{sample_idx}', f'stage_{stage_idx}'), exist_ok=True)
        img_all_points = calibrator.draw_all_points(frame=frame_idx)
        img_edges = calibrator.draw_edge_points(frame=frame_idx,
                                                image=calibrator.img_detector.imgs_edges[frame_idx])
        cv.imwrite(os.path.join(LOG_DIR, f'trial_{sample_idx}', f'stage_{stage_idx}',
                                f'all_points_frame_{frame_idx}.jpg'), img_all_points)
        cv.imwrite(os.path.join(LOG_DIR, f'trial_{sample_idx}', f'stage_{stage_idx}',
                                f'edge_points_frame_{frame_idx}.jpg'), img_edges)

"""Save initial taus, optimized taus, tau_scale"""
tau_inits = np.asarray(tau_inits)
np.save(os.path.join(LOG_DIR, 'tau_inits'), tau_inits)

tau_data = np.asarray(tau_data)
np.save(os.path.join(LOG_DIR, 'tau_data'), tau_data)

plt.figure()
plt.clf()
plt.subplot(1, 5, 1)
plt.title('X')
plt.scatter(range(tau_data.shape[0]), tau_data[:, 3], c='b')
plt.scatter(range(tau_inits.shape[0]), tau_inits[:, 3], c='r')
axes = plt.gca()
xlims = axes.get_xlim()
plt.hlines(tau_gt[3], xlims[0], xlims[1])

plt.subplot(1, 5, 2)
plt.title('Y')
plt.scatter(range(tau_data.shape[0]), tau_data[:, 4], c='b')
plt.scatter(range(tau_inits.shape[0]), tau_inits[:, 4], c='r')
axes = plt.gca()
xlims = axes.get_xlim()
plt.hlines(tau_gt[4], xlims[0], xlims[1])

plt.subplot(1, 5, 3)
plt.title('Z')
plt.scatter(range(tau_data.shape[0]), tau_data[:, 5], c='b')
plt.scatter(range(tau_inits.shape[0]), tau_inits[:, 5], c='r')
axes = plt.gca()
xlims = axes.get_xlim()
plt.hlines(tau_gt[5], xlims[0], xlims[1])

plt.subplot(1, 5, 4)
plt.title('Angle')
angle_vec = np.linalg.norm(tau_data[:, :3], axis=1, ord=2)
angle_gt = np.linalg.norm(tau_gt[:3], ord=2)
angle_vec_init = np.linalg.norm(tau_inits[:, :3], axis=1, ord=2)
plt.scatter(range(tau_data.shape[0]), angle_vec, c='b')
plt.scatter(range(tau_inits.shape[0]), angle_vec_init, c='r')
axes = plt.gca()
xlims = axes.get_xlim()
plt.hlines(angle_gt, xlims[0], xlims[1])

plt.subplot(1, 5, 5)
plt.title('Axis')
axis_vec = tau_data[:, :3]/np.expand_dims(np.linalg.norm(tau_data[:, :3], axis=1, ord=2), 1)
axis_gt = tau_gt[:3]/np.linalg.norm(tau_gt[:3], ord=2)
axis_vec_init = tau_inits[:, :3] / np.expand_dims(np.linalg.norm(tau_inits[:, :3], axis=1, ord=2), 1)
axis_diffs = np.linalg.norm(axis_vec - axis_gt, axis=1, ord=2)
plt.scatter(range(axis_diffs.shape[0]), axis_diffs, c='b')
axes = plt.gca()
xlims = axes.get_xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.show()
plt.savefig(os.path.join(LOG_DIR, 'extrinsics_deviation.png'))
plt.show()

# Mean, Median, Std Deviation
table = PrettyTable()
table.field_names = ["X_med", "X_std", "Y_med", "Y_std", "Z_med", "Z_std",
                     "angle_med", "angle_std", "axis_med", "axis_std"]

x_error = tau_data[:, 3] - tau_gt[3]
y_error = tau_data[:, 4] - tau_gt[4]
z_error = tau_data[:, 5] - tau_gt[5]
angle_error = np.linalg.norm(tau_data[:, :3], ord=2, axis=1) - np.linalg.norm(tau_gt[:3], ord=2)
axis_error = axis_diffs
errors = [x_error, y_error, z_error, angle_error, axis_error]

row = []
for param_idx, param_name in enumerate(['X', 'Y', 'Z', 'angle', 'axis']):
    row += [np.median(errors[param_idx]), np.std(errors[param_idx])]
table.add_row(row)
print(table)

table_data = table.get_string()
with open(os.path.join(LOG_DIR, 'error_metrics.txt'), 'w') as table_file:
    table_file.write(table_data)

