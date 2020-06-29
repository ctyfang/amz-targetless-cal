from calibration.camera_lidar_calibrator import *
from calibration.utils.config import command_line_parser
from calibration.utils.data_utils import *
from calibration.utils.img_utils import *
import pickle

"""Script Parameters"""
visualize_edges = True

input_dir_list = ['/media/carter/Samsung_T5/3dv/2011_09_28/collection',
                  '/media/benjin/Data/Data/3dv_kitti_2011_09_26_static_scenes/2011_09_26',
                  'data/2011_09_26_0017']
calib_dir_list = ['/media/carter/Samsung_T5/3dv/2011_09_28/calibration',
                  '/media/benjin/Data/Data/3dv_kitti_2011_09_26_calib/2011_09_26',
                  'data']

cfg = command_line_parser()
input_dir = getPath(input_dir_list)
calib_dir = getPath(calib_dir_list)
calibrator_path = '../generated/calibrators/0928-6frames-corresps.pkl'

"""Load calibrator with detected edges"""
with open(calibrator_path, 'rb') as input_pkl:
    calibrator = pickle.load(input_pkl)
    calibrator.visualize = True

"""Load ground-truth extrinsics"""
R, T = load_lid_cal(calib_dir)
tau_gt = calibrator.tau = calibrator.transform_to_tau(R, T)

"""Get Camera and LiDAR edge images"""
frame_idx = 0
cam_edges = calibrator.img_detector.imgs_edges[frame_idx]
lid_edges = calibrator.draw_edge_points_binary(frame_idx)

if visualize_edges:
    cat_edges = np.vstack([255*cam_edges.astype(np.uint8),
                           255*lid_edges.astype(np.uint8)])
    cv.imshow("Concatenated", cat_edges)
    cv.waitKey(0)
    cv.destroyAllWindows()

"""Distance Transform"""
cam_dist = cv.distanceTransform(255*np.logical_not(cam_edges).astype(np.uint8),
                                cv.DIST_L2, cv.DIST_MASK_PRECISE)
cv.imshow("Dist", cam_dist)
cv.waitKey(0)
cv.destroyAllWindows()

"""Chamfer Distance"""
chamfer_dist = np.multiply(lid_edges, cam_dist).sum()
print('hi')