from calibration.camera_lidar_calibrator import *
from calibration.utils.config import command_line_parser
from calibration.utils.data_utils import *
from calibration.utils.img_utils import *
import pickle

input_dir_list = ['/media/carter/Samsung_T5/3dv/2011_09_28/collection',
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

cfg.frames = [34, 1, 6, 19, 257, 185]
cfg.im_ed_score_thr1 = 100
cfg.im_ed_score_thr2 = 200

cfg.pc_ed_score_thr = 0.275

# Create calibrator and detect edges from scratch
calibrator = CameraLidarCalibrator(cfg, visualize=True)
with open('generated/calibrators/0928-6frames-corresps.pkl', 'wb') as output_pkl:
    pickle.dump(calibrator, output_pkl, pickle.HIGHEST_PROTOCOL)

# Load calibrator with detected edges from pickled object
# with open('./output/calibrator_3dv_kitti_2011_09_26_static_scenes_thr1_100_thr2_200_thr_0p25.pkl', 'rb') as input_pkl:
#     calibrator = pickle.load(input_pkl)
#     calibrator.visualize = True

# Draw, Optimize, Re-draw
calibrator.project_point_cloud()
for frame_idx in range(len(calibrator.projection_mask)):
    calibrator.draw_edge_points(
        frame=frame_idx, image=calibrator.img_detector.imgs_edges[frame_idx], show=True)
    calibrator.draw_edge_points(
        frame=frame_idx, show=True)
