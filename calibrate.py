from calibration.camera_lidar_calibrator import *
from calibration.utils.config import command_line_parser
from calibration.utils.data_utils import *
from calibration.utils.img_utils import *
import pickle
import cv2


def main():
    pwd = os.path.abspath(".")
    cfg = command_line_parser()

    # Create calibrator and detect edges from scratch
    init_guess_file = 'extrinsics_fw_forward.npz'
    with np.load(os.path.join(pwd, 'init', init_guess_file)) as data:
        R = data['R'].squeeze()
        t = data['t'].squeeze()

    tau_init = np.hstack((R, t)).squeeze()
    print(tau_init)

    calibrator = CameraLidarCalibrator(cfg, visualize=False, tau_init=tau_init)

    if cfg.calibration_method == 'manual':
        print('Select at least 6 correspondences')
        calibrator.select_correspondences()
        tau_opt = calibrator.batch_optimization()
        # _, R, t = cv2.solvePnP(
        #     objectPoints=calibrator.correspondences[0][1],
        #     imagePoints=calibrator.correspondences[0][0].astype(np.float),
        #     cameraMatrix=calibrator.K,
        #     distCoeffs=calibrator.distortion, rvec=R,
        #     tvec=t, useExtrinsicGuess=True)
        # tau_opt = np.hstack((R.squeeze(), t.squeeze())).squeeze()
        # print(tau_opt)

    calibrator.tau = tau_opt
    with open(os.path.join(pwd, init_guess_file), 'wb') as f:
        np.savez(f, R=tau_opt[:3], t=tau_opt[3:])

    for idx in range(len(calibrator.img_detector.imgs)):
        calibrator.draw_all_points(frame=idx, show=True)


if __name__ == "__main__":
    main()
