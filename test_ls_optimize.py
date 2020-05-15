from calibration.camera_lidar_calibrator import *
from calibration.utils.config import command_line_parser
from calibration.utils.data_utils import *
from calibration.utils.img_utils import *
import pickle
import cv2
import sys


input_dir_list = ['/media/carter/Samsung_T5/3dv/2011_09_26/2011_09_26_drive_0106_sync',
                  '/home/benjin/Development/Data/2011_09_26_drive_0106_sync',
                  'data/2011_09_26_0017']
calib_dir_list = ['/media/carter/Samsung_T5/3dv/2011_09_26/calibration',
                  '/home/benjin/Development/Data/2011_09_26_calib/2011_09_26',
                  'data']

cfg = command_line_parser()

input_dir = getPath(input_dir_list)
calib_dir = getPath(calib_dir_list)

cfg.pc_dir = input_dir
cfg.img_dir = input_dir
cfg.calib_dir = calib_dir

new = False
if new:
    # Detect edges from scratch
    calibrator = CameraLidarCalibrator(cfg, visualize=False)
    with open('generated/calibrator.pkl', 'wb') as output_pkl:
        pickle.dump(calibrator, output_pkl, pickle.HIGHEST_PROTOCOL)
    print('Initialization Done')
else:
    # Load calibrator from pickle
    with open('generated/calibrator.pkl', 'rb') as input_pkl:
        calibrator = pickle.load(input_pkl)
        calibrator.visualize = True

print('Parameter Sets')
frame = -1
initial = calibrator.draw_edge_points(
    score=calibrator.pc_detector.pcs_edge_scores[frame])#, image=calibrator.img_detector.imgs_edge_scores)
print('Done Drawing')


print(calibrator.compute_conv_cost(120, frame=frame))
sys.exit()
# print(calibrator.compute_bf_cost(120))

print(calibrator.ls_optimize(120))
calibrator.project_point_cloud()
post = calibrator.draw_edge_points(
    score=calibrator.pc_detector.pcs_edge_scores)#, image=calibrator.img_detector.imgs_edge_scores)
vertical = np.concatenate((initial, post), axis=0)
cv2.imshow('Comparison', vertical)
cv2.waitKey(0)
cv2.destroyAllWindows()
print('Done')
# calibrator.optimize(120, max_iters=1)
