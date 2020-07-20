from calibration.camera_lidar_calibrator import *
from calibration.utils.config import command_line_parser
from calibration.utils.data_utils import *
from calibration.utils.img_utils import *
import pickle

def main():
    cfg = command_line_parser()

    # Create calibrator and detect edges from scratch
    calibrator = CameraLidarCalibrator(cfg, visualize=False)
    with open('generated/calibrators/new_calibrator.pkl', 'wb') as output_pkl:
        pickle.dump(calibrator, output_pkl, pickle.HIGHEST_PROTOCOL)

    # Run different optimizer depending on MANUAL or AUTOMATIC
    hyperparams = {'alphas': {'mi': 0, 'gmm': 0, 'points': 0, 'corr': 0,
                            'sigma': 7},
                'scales': np.power(10*np.ones((1, 6)),
                                    [0, 0, 0, -2, -2, -2]).tolist()}

    if cfg.calibration_method == 'manual':
        print('Select at least 6 correspondences')
        calibrator.select_correspondences()
        tau_opt = calibrator.batch_optimization()

    elif cfg.calibration_method == 'automatic':
        hyperparams['alphas']['gmm'] = 0.2
        hyperparams['alphas']['mi'] = 1.0
        tau_opt, cost_history = calibrator.ls_optimize(hyperparams)

    calibrator.tau = tau_opt
    with open('generated/calibrators/new_calibrator.pkl', 'wb') as output_pkl:
        pickle.dump(calibrator, output_pkl, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()