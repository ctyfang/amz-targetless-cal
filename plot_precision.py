from calibration.camera_lidar_calibrator import *
from calibration.utils.config import command_line_parser
from calibration.utils.data_utils import *
from calibration.utils.img_utils import *
from experiment_precision import filename_init, filename_data

import pickle
import json
import sys
from prettytable import PrettyTable
from scipy.spatial.transform import Rotation as R

input_dir_list = [
    '/media/carter/Samsung_T5/3dv/2011_09_28/collection',
    '/home/benjin/Development/Data/2011_09_26_drive_0106_sync',
    'data/2011_09_26_0017'
]
calib_dir_list = [
    '/media/carter/Samsung_T5/3dv/2011_09_28/calibration',
    '/home/benjin/Development/Data/2011_09_26_calib/2011_09_26', 'data'
]
cfg = command_line_parser()

input_dir = getPath(input_dir_list)
calib_dir = getPath(calib_dir_list)
cfg.pc_dir = input_dir
cfg.img_dir = input_dir
cfg.calib_dir = calib_dir


LOG_DIR = 'generated/logs'
with open(os.path.join(LOG_DIR, filename_init), 'rb') as f:
    tau_inits = np.load(f)
with open(os.path.join(LOG_DIR, filename_data), 'rb') as f:
    tau_data = np.load(f)

fig = plt.figure(figsize=[12, 5])
ax = [fig.add_subplot(1, 6, 1)]
for i in range(2, 4):
    ax.append(fig.add_subplot(1, 6, i, sharey=ax[0]))
ax.append(fig.add_subplot(1, 6, 4))
for i in range(5, 7):
    ax.append(fig.add_subplot(1, 6, i, sharey=ax[3]))

for param_idx, param_name in enumerate(['Roll', 'Pitch', 'Yaw', 'X', 'Y', 'Z']):
    if param_idx == 0:
        ax[param_idx].set_ylabel('Deviation from mean(degree)')
    # fig, ax = plt.subplots()
    ax[param_idx].set_title(param_name)
    if param_idx < 3:
        ax[param_idx].boxplot(
            np.rad2deg(tau_data[:, param_idx] - np.mean(tau_data[:, param_idx]))
        )
    else:
        ax[param_idx].boxplot(
            tau_data[:, param_idx] - np.mean(tau_data[:, param_idx])
        )
ax[-1].set_ylabel('Deviation from mean(m)')
ax[-1].yaxis.set_label_position("right")
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, 'extrinsics_deviation_box.png'))
# plt.show()
plt.close()

plt.figure(figsize=[12, 5])
for param_idx, param_name in enumerate(['Roll', 'Pitch', 'Yaw', 'X', 'Y', 'Z']):
    plt.subplot(1, 6, 1+param_idx)
    plt.title(param_name)
    plt.scatter(range(tau_data.shape[0]), tau_data[:, param_idx], c='b')
    plt.scatter(range(tau_inits.shape[0]), tau_inits[:, param_idx], c='r')
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, 'extrinsics_deviation.png'))
# plt.show()
plt.close()

# Mean, Median, Std Deviation
table = PrettyTable()
table.field_names = [
    "N_sample", "X_std", "Y_std", "Z_std", 'Roll_std', 'Pitch_std', 'Yaw_std'
]

row = [tau_data.shape[0]]
for param_idx, param_name in enumerate(['Roll', 'Pitch', 'Yaw', 'X', 'Y', 'Z']):
    row += [np.std(tau_data[:, param_idx])]
table.add_row(row)
print(table)

table_data = table.get_string()
with open(os.path.join(LOG_DIR, 'error_metrics.txt'), 'w') as table_file:
    table_file.write(table_data)