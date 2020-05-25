from calibration.camera_lidar_calibrator import *
from calibration.utils.config import command_line_parser
from calibration.utils.data_utils import *
from calibration.utils.img_utils import *
import pickle

input_dir_list = ['/media/carter/Samsung_T5/3dv/2011_09_28/collection',
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

# Create calibrator and detect edges from scratch
calibrator = CameraLidarCalibrator(cfg, visualize=False)
with open('./output/calibrator_collection-8-0928.pkl', 'wb') as output_pkl:
    pickle.dump(calibrator, output_pkl, pickle.HIGHEST_PROTOCOL)

# # Load calibrator with detected edges from pickled object
# with open('./output/calibrator_collection-3.pkl', 'rb') as input_pkl:
#     calibrator = pickle.load(input_pkl)
#     calibrator.visualize = True

# # Perturb tau with noise
# calibrator.tau = perturb_tau(calibrator.tau, trans_std=0.05, angle_std=5)

# Draw, Optimize, Re-draw
calibrator.project_point_cloud()
for frame_idx in range(len(calibrator.projection_mask)):
    calibrator.draw_reflectance(frame=frame_idx)
    calibrator.draw_all_points(frame=frame_idx)
    calibrator.draw_edge_points(frame=frame_idx,
                                image=calibrator.img_detector.imgs_edges[frame_idx])

