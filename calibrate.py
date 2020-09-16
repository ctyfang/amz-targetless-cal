from calibration.camera_lidar_calibrator import *
from calibration.utils.config import command_line_parser
from calibration.utils.data_utils import *
from calibration.utils.img_utils import *
import pickle

def main():
    cfg = command_line_parser()

    # Create calibrator and detect edges from scratch
    # R = np.array([0.39617389417871668, 0.91206160557644977, -0.10578219701149925]).reshape((1,3))
    # print(R.shape)
    with open('extrinsics.npy', 'rb') as f:
        extrinsics = np.load(f)

    R = extrinsics[:3, :3].reshape((3,3))
    T = extrinsics[:3, 3].reshape((3,1))
    # T = np.array([-1.657708471611965, 0.739773004848766, 1.953301910912623]).reshape(3,1)
    # T = np.zeros((3,1))
    tau_init = CameraLidarCalibrator.transform_to_tau(R, T)
    print(tau_init)

    calibrator = CameraLidarCalibrator(cfg, visualize=False, tau_init=tau_init)

    with open('generated/calibrators/new_calibrator.pkl', 'wb') as output_pkl:
        pickle.dump(calibrator, output_pkl, pickle.HIGHEST_PROTOCOL)

    # exit()
    if cfg.calibration_method == 'manual':
        print('Select at least 6 correspondences')
        calibrator.select_correspondences()
        tau_opt = calibrator.batch_optimization()

    calibrator.tau = tau_opt
    with open('generated/calibrators/new_calibrator.pkl', 'wb') as output_pkl:
        pickle.dump(calibrator, output_pkl, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()