from calibration.camera_lidar_calibrator import *
from calibration.utils.config import command_line_parser
from calibration.utils.data_utils import *
from calibration.utils.img_utils import *
import pickle

input_dir_list = ['/media/carter/Samsung_T5/3dv/2011_09_28/2011_09_28_drive_0035_sync',
                  '/home/benjin/Development/Data/2011_09_26_drive_0106_sync',
                  'data/2011_09_26_0017']
calib_dir_list = ['/media/carter/Samsung_T5/3dv/2011_09_28/calibration',
                  '/home/benjin/Development/Data/2011_09_26_calib/2011_09_26',
                  'data']

cfg = command_line_parser()

input_dir = getPath(input_dir_list)
calib_dir = getPath(calib_dir_list)

cfg.pc_dir = input_dir
cfg.img_dir = input_dir
cfg.calib_dir = calib_dir

# Detect edges from scratch
# calibrator = CameraLidarCalibrator(cfg, visualize=True)
# with open('./output/calibrator_imgthresh200-300_pcthresh04.pkl', 'wb') as output_pkl:
#     pickle.dump(calibrator, output_pkl, pickle.HIGHEST_PROTOCOL)

# Load calibrator from pickle
with open('./output/calibrator_imgthresh200-300_pcthresh04.pkl', 'rb') as input_pkl:
    calibrator = pickle.load(input_pkl)
    calibrator.visualize = True

# Perturb tau with noise
# calibrator.tau = perturb_tau(calibrator.tau, trans_std=0.025, angle_std=2.5)

# print(calibrator.compute_conv_cost(120))
# print(calibrator.compute_bf_cost(120))
calibrator.optimize(15, max_iters=1000)
calibrator.optimize(5, max_iters=1000)
# calibrator.optimize(3, max_iters=1000)
