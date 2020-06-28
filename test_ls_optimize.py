from calibration.camera_lidar_calibrator import *
from calibration.utils.config import command_line_parser
from calibration.utils.data_utils import *
from calibration.utils.img_utils import *
import pickle
import cv2
import sys


# input_dir_list = ['data/2011_09_26_0017']
input_dir_list = ['data/collection']
calib_dir_list = ['data']

cfg = command_line_parser()

input_dir = getPath(input_dir_list)
calib_dir = getPath(calib_dir_list)

cfg.pc_dir = input_dir
cfg.img_dir = input_dir
cfg.calib_dir = calib_dir
# cfg.frames = [99, 139, 149, 150, 151, 152, 153]
cfg.frames = [34, 35, 185]

new = False
if new:
    # Detect edges from scratch
    calibrator = CameraLidarCalibrator(cfg, visualize=False)
    with open('generated/calibratorset3.pkl', 'wb') as output_pkl:
        pickle.dump(calibrator, output_pkl, pickle.HIGHEST_PROTOCOL)
    print('Initialization Done')
else:
    # Load calibrator from pickle
    with open('generated/calibratorset3.pkl', 'rb') as input_pkl:
        calibrator = pickle.load(input_pkl)
        calibrator.visualize = True

print('Parameter Sets')

print(len(calibrator.img_detector.imgs))

# frames = range(len(calibrator.img_detector.imgs))
# for frame in frames:
#     initial = calibrator.draw_edge_points(
#         score=calibrator.pc_detector.pcs_edge_scores[frame], frame=frame)#, image=calibrator.img_detector.imgs_edge_scores)
# print('Done Drawing')

# print(calibrator.compute_conv_cost(120, frame=frame))
# print(calibrator.batch_optimization(120))
# sys.exit()
# print(calibrator.compute_bf_cost(120))
sigma = 6.0
hm_ptc = calibrator.dist_transform(frame=0, show=True)
sys.exit()
error = np.linspace(0, 0.01, 10)
cost_history = []
# print(calibrator.batch_optimization(sigma_in=sigma))
# hm_ptc = calibrator.compute_heat_map(sigma=sigma, frame=0,
#                                      ptCloud=True, show=True)
# hm_img = calibrator.compute_heat_map(sigma=sigma, frame=0,
#                                      ptCloud=False, show=True)
# for i in range(len(calibrator.img_detector.imgs)):
# for i in range (10):
    # calibrator.tau[2] += 0.001
    # calibrator.project_point_cloud()
    # sigma=7
    # hm_ptc = calibrator.compute_heat_map(sigma=sigma, frame=0, ptCloud=True)
    # hm_img = calibrator.compute_heat_map(sigma=sigma, frame=0, ptCloud=False)
    # diff = hm_img - hm_ptc
    # cost_history.append(np.linalg.norm(diff, ord=2))
    # print(cost_history[-1])
# plt.plot(range(10), cost_history)
# plt.show()
sys.exit()
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
