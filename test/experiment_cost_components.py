from calibration.camera_lidar_calibrator import *
from calibration.utils.config import command_line_parser
from calibration.utils.data_utils import *
from calibration.utils.img_utils import *

from tqdm import tqdm
import pickle
import json
import pyquaternion as pyquat
from prettytable import PrettyTable
from scipy.spatial.transform import Rotation as scipyrot

# Configuration parameters
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

BASE_DIR = '../output/logs2/exp_m=3_n=9_trials=5'
# Load calibrator with detected edges from pickled object
with open('../output/calibrator_collection-8-0928.pkl', 'rb') as input_pkl:
    calibrator = pickle.load(input_pkl)
    calibrator.visualize = True

print(len(calibrator.pc_detector.pcs))
print(len(calibrator.img_detector.imgs))
# Experiment Parameters
with open(os.path.join(BASE_DIR, 'params.json')) as params:
    opt_params = json.load(params)
exp_params = {'trans_range': 0.20, 'angle_range': 20}

trial_idx = 3
tau_data = np.load(os.path.join(BASE_DIR, 'tau_data.npy'))
tau_gt = tau_data[trial_idx, :]
stage_idx = 0

## TRANSLATION ##

xlabels = ['X Displacement (m)', 'Y Displacement (m)', 'Z Displacement']
ylabels = ['MI Cost', 'GMM Cost', 'Points Cost', 'Cost Sum']
base_idx = 3
trans_res = 0.00125
trans_range = (-exp_params['trans_range'],
               +exp_params['trans_range'])
offset_vals = np.linspace(trans_range[0], trans_range[1], round((trans_range[1] - trans_range[0])/trans_res) + 1)

costs = []
plt.figure(figsize=(18, 12))
for offset_idx in range(3):
    costs.append(np.zeros((4, offset_vals.shape[0])))
    for idx, val in tqdm(enumerate(offset_vals)):
        tau_temp = deepcopy(tau_gt)
        tau_temp[base_idx+offset_idx] += val

        cost_mi, cost_gmm, cost_points = loss_components(tau_temp, calibrator,
                                         opt_params['SIGMAS'][stage_idx], opt_params['ALPHA_MI'][stage_idx],
                                         opt_params['ALPHA_GMM'][stage_idx],  opt_params['ALPHA_POINTS'][stage_idx], [])

        costs[offset_idx][0, idx] = cost_mi
        costs[offset_idx][1, idx] = cost_gmm
        costs[offset_idx][2, idx] = cost_points
        costs[offset_idx][3, idx] = cost_mi + cost_gmm + cost_points

    plt.subplot(4, 3, 1+offset_idx)
    plt.plot(offset_vals, costs[offset_idx][0, :])
    if offset_idx == 0:
        plt.ylabel(ylabels[0])

    plt.subplot(4, 3, 4+offset_idx)
    plt.plot(offset_vals, costs[offset_idx][1, :])
    if offset_idx == 0:
        plt.ylabel(ylabels[1])

    plt.subplot(4, 3, 7+offset_idx)
    plt.plot(offset_vals, costs[offset_idx][2, :])
    if offset_idx == 0:
        plt.ylabel(ylabels[2])

    plt.subplot(4, 3, 10+offset_idx)
    plt.plot(offset_vals, costs[offset_idx][3, :])
    if offset_idx == 0:
        plt.ylabel(ylabels[3])
    plt.xlabel(xlabels[offset_idx])



plt.show()

## ROTATION ##
plt.figure(figsize=(18, 12))
xlabels = ['Roll (degrees)', 'Pitch (degrees)', 'Yaw (degrees)']
angle_res = 0.125
angle_range = (-exp_params['angle_range'],
               exp_params['angle_range'])
offset_vals = np.linspace(angle_range[0], angle_range[1], round((angle_range[1] - angle_range[0])/angle_res) + 1)
costs = []
for offset_idx in range(3):

    costs.append(np.zeros((4, offset_vals.shape[0])))
    for idx, val in tqdm(enumerate(offset_vals)):
        euler_offset = [0, 0, 0]
        euler_offset[offset_idx] = val

        r = scipyrot.from_rotvec(tau_gt[:3])
        euler_angles = r.as_euler('xyz', degrees=True)
        euler_new = euler_angles + euler_offset

        r_new = scipyrot.from_euler('xyz', euler_new, degrees=True)
        rvec_new = r_new.as_rotvec()

        tau_temp = deepcopy(tau_gt)
        tau_temp[:3] = rvec_new

        cost_mi, cost_gmm, cost_points = loss_components(tau_temp, calibrator, opt_params['SIGMAS'][stage_idx], opt_params['ALPHA_MI'][stage_idx],
                                         opt_params['ALPHA_GMM'][stage_idx], opt_params['ALPHA_POINTS'][stage_idx], [])

        costs[offset_idx][0, idx] = cost_mi
        costs[offset_idx][1, idx] = cost_gmm
        costs[offset_idx][2, idx] = cost_points
        costs[offset_idx][3, idx] = cost_mi + cost_gmm + cost_points

    plt.subplot(4, 3, 1+offset_idx)
    plt.plot(offset_vals, costs[offset_idx][0, :])
    plt.subplot(4, 3, 4+offset_idx)
    plt.plot(offset_vals, costs[offset_idx][1, :])
    plt.subplot(4, 3, 7+offset_idx)
    plt.plot(offset_vals, costs[offset_idx][2, :])
    plt.subplot(4, 3, 10+offset_idx)
    plt.plot(offset_vals, costs[offset_idx][3, :])
    plt.xlabel(xlabels[offset_idx])

plt.show()


