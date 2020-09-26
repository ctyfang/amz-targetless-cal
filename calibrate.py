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
    with np.load(init_guess_file) as data:
        R = data['R']
        t = data['t']
    print(R.shape)
    print(t.shape)

    tau_init = np.hstack((R, t)).squeeze()
    print(tau_init)

    calibrator = CameraLidarCalibrator(cfg, visualize=False, tau_init=tau_init)

    # with open('generated/calibrators/new_calibrator.pkl', 'wb') as output_pkl:
    #     pickle.dump(calibrator, output_pkl, pickle.HIGHEST_PROTOCOL)

    # exit()
    if cfg.calibration_method == 'manual':
        print('Select at least 6 correspondences')
        calibrator.select_correspondences()
        tau_opt = calibrator.batch_optimization()

    calibrator.tau = tau_opt
    with open(os.path.join(pwd, init_guess_file), 'wb') as f:
        np.savez(f, R=tau_opt[:3], t=tau_opt[3:])
    with open('generated/calibrators/new_calibrator.pkl', 'wb') as output_pkl:
        pickle.dump(calibrator, output_pkl, pickle.HIGHEST_PROTOCOL)
    # with open('generated/calibrators/new_calibrator.pkl', 'rb') as input_pkl:
    #     calibrator = pickle.load(input_pkl)
    #     calibrator.visualize = True

    for idx in range(len(calibrator.img_detector.imgs)):
        calibrator.draw_all_points(frame=idx, show=True)
    # cv2.imshow('projection', projection)

if __name__ == "__main__":
    main()