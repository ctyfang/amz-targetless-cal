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

BASE_DIR = '../generated/optimizer_logs/test'
# Load calibrator with detected edges from pickled object
with open('../generated/calibrators/correspondence_test.pkl', 'rb') as input_pkl:
    calibrator = pickle.load(input_pkl)
    calibrator.visualize = True

# Experiment Parameters
with open(os.path.join(BASE_DIR, 'params.json')) as params:
    opt_params = json.load(params)
exp_params = {'trans_range': 0.10, 'angle_range': 10}

tau_scales_log = np.asarray([1, 1, 1, -2, -2, -2])
tau_scales = np.power(10 * np.ones(tau_scales_log.shape),
                      tau_scales_log)
trial_idx = 0
tau_data = np.load(os.path.join(BASE_DIR, 'tau_data.npy'))
tau_gt = tau_data[trial_idx, :]
stage_idx = 0

## TRANSLATION ##

plt.subplot(1, 2, 1)
base_idx = 3
trans_res = 0.0250
trans_range = (-exp_params['trans_range'],
               +exp_params['trans_range'])
offset_vals = np.linspace(trans_range[0], trans_range[1], round((trans_range[1] - trans_range[0])/trans_res) + 1)

for offset_idx in range(3):
    cost = np.zeros(offset_vals.shape)
    for [idx, val] in tqdm(enumerate(offset_vals)):
        tau_temp = deepcopy(tau_gt)
        tau_temp[base_idx+offset_idx] += val

        curr_cost = loss_scaled(tau_temp, calibrator,
                                 opt_params['SIGMAS'][stage_idx],
                                 opt_params['ALPHA_MI'][stage_idx],
                                 opt_params['ALPHA_GMM'][stage_idx],
                                 opt_params['ALPHA_CORR'][stage_idx], [],
                                 tau_scales, False)

        cost[idx] = curr_cost
    plt.plot(offset_vals, cost)
plt.legend(['x', 'y', 'z'])
plt.xlabel('Translation Displacement (m)')
plt.show()

## ROTATION ##

plt.subplot(1, 2, 2)
angle_res = 0.125
angle_range = (-exp_params['angle_range'],
               exp_params['angle_range'])
offset_vals = np.linspace(angle_range[0], angle_range[1], round((angle_range[1] - angle_range[0])/angle_res) + 1)

for dim in range(3):
    cost = np.zeros(offset_vals.shape)

    for [idx, val] in tqdm(enumerate(offset_vals)):
        euler_offset = [0, 0, 0]
        euler_offset[dim] = val

        r = scipyrot.from_rotvec(tau_gt[:3])
        euler_angles = r.as_euler('xyz', degrees=True)
        euler_new = euler_angles + euler_offset

        r_new = scipyrot.from_euler('xyz', euler_new, degrees=True)
        rvec_new = r_new.as_rotvec()

        tau_temp = deepcopy(tau_gt)
        tau_temp[:3] = rvec_new

        curr_cost = loss_scaled(tau_temp, calibrator,
                                opt_params['SIGMAS'][stage_idx],
                                opt_params['ALPHA_MI'][stage_idx],
                                opt_params['ALPHA_GMM'][stage_idx],
                                opt_params['ALPHA_CORR'][stage_idx], [],
                                tau_scales, False)

        cost[idx] = curr_cost
    plt.plot(offset_vals, cost)
plt.legend(['Roll', 'Pitch', 'Yaw'])
plt.xlabel('Rotation Displacement (Degrees)')
plt.show()

