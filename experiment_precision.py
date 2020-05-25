from calibration.camera_lidar_calibrator import *
from calibration.utils.config import command_line_parser
from calibration.utils.data_utils import *
from calibration.utils.img_utils import *

import pickle
import json
import sys
import gc
from prettytable import PrettyTable

input_dir_list = [
    '/media/carter/Samsung_T5/3dv/2011_09_28/collection',
    '/home/benjin/Development/Data/2011_09_26_drive_0106_sync',
    'data/2011_09_26_0017'
]
calib_dir_list = [
    '/media/carter/Samsung_T5/3dv/2011_09_28/calibration',
    '/home/benjin/Development/Data/2011_09_26_calib/2011_09_26',
    'data'
]
trial = 'set_3_t_10_r_15'


filename_init = 'tau_init' + trial + '.npy'
filename_data = 'tau_data' + trial + '.npy'

if __name__ == '__main__':
    cfg = command_line_parser()

    input_dir = getPath(input_dir_list)
    calib_dir = getPath(calib_dir_list)
    cfg.pc_dir = input_dir
    cfg.img_dir = input_dir
    cfg.calib_dir = calib_dir

    # # Load calibrator with detected edges from pickled object
    with open('generated/calibratorset3.pkl', 'rb') as input_pkl:
        calibrator = pickle.load(input_pkl)
        calibrator.visualize = True

    # Ground-truth Tau
    R, T = load_lid_cal(cfg.calib_dir)
    tau_gt = calibrator.tau = calibrator.transform_to_tau(R, T)

    # Experiment Parameters
    exp_params = {
        'NUM_SAMPLES': 1000,
        'TRANS_ERR_SIGMA': 0.2,
        'ANGLE_ERR_SIGMA': 15,
        'ALPHA_MI': [30.0],
        'ALPHA_GMM': [1.0],
        'SIGMAS': [15.0, 5.0, 3.0, 1.0],
        'MAX_ITERS': 500,
        'tau_gt': tau_gt.tolist()
    }

    LOG_DIR = os.path.join('generated','logs', trial)
    with open(os.path.join(LOG_DIR, 'params.json'), 'w') as json_file:
        json.dump(exp_params, json_file, indent=4)

    # # Errors
    new = True
    # filename_init = 'tau_init_set3_large_disturb.npy'
    # filename_data = 'tau_data_set3_large_disturb.npy'
    if new:
        tau_data = []
        tau_inits = []
    else:
        with open(os.path.join(LOG_DIR, filename_init), 'rb') as f:
            tau_inits = np.load(f).tolist()
        with open(os.path.join(LOG_DIR, filename_data), 'rb') as f:
            tau_data = np.load(f).tolist()

    for sample_idx in range(exp_params['NUM_SAMPLES']):
        print(f'----- SAMPLE {sample_idx} -----')
        # Reset tau to gt, sample noise, perturb
        calibrator.tau = perturb_tau(tau_gt,
                                     trans_std=exp_params['TRANS_ERR_SIGMA'],
                                     angle_std=exp_params['ANGLE_ERR_SIGMA'])
        calibrator.project_point_cloud()

        tau_inits.append(calibrator.tau)
        np.save(os.path.join(LOG_DIR, filename_init), np.asarray(tau_inits))

        # Run optimizer
        for stage_idx, [sigma_in, alpha_mi, alpha_gmm] in enumerate(
                zip(exp_params['SIGMAS'], exp_params['ALPHA_MI'],
                    exp_params['ALPHA_GMM'])):

            print(f'Optimization {stage_idx + 1}/{len(exp_params["SIGMAS"])}')
            if stage_idx != 2:
                tau_opt, cost_history = calibrator.ls_optimize(
                    sigma_in,
                    alpha_gmm=alpha_gmm,
                    alpha_mi=alpha_mi,
                    maxiter=exp_params['MAX_ITERS'],
                    translation_only=False)
            else:
                tau_opt, cost_history = calibrator.ls_optimize(
                    sigma_in,
                    alpha_gmm=alpha_gmm,
                    alpha_mi=alpha_mi,
                    maxiter=exp_params['MAX_ITERS'],
                    translation_only=True)

            calibrator.tau = tau_opt
            calibrator.project_point_cloud()
            print(extrinsics_error(tau_gt, tau_opt))

        tau_data.append(tau_opt)
        np.save(os.path.join(LOG_DIR, filename_data), np.asarray(tau_data))
        gc.collect()
        # sys.eixt()
