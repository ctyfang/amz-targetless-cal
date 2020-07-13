from calibration.camera_lidar_calibrator import *
from calibration.utils.config import command_line_parser
from calibration.utils.data_utils import *
from calibration.utils.img_utils import *
from experiment_precision import filename_init, filename_data, LOG_DIR

import pickle
import json
import sys
from prettytable import PrettyTable
from scipy.spatial.transform import Rotation as R

# LOG_DIR = 'generated/logs'
# filename_init = 'tau_inits_set3.npy'
# filename_data = 'tau_data_set3.npy'
# generated/logs/tau_init_set3_large_disturb.npy
def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((ys < upper_bound) & (ys > lower_bound))

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


with open(os.path.join(LOG_DIR, filename_init), 'rb') as f:
    tau_inits = np.load(f)
with open(os.path.join(LOG_DIR, filename_data), 'rb') as f:
    tau_data = np.load(f)

print(tau_inits.shape)
print(tau_data.shape)
sys.exit()
fields = ['R1', 'R2', 'R3', 'X', 'Y', 'Z']
R, T = load_lid_cal(calib_dir)
tau_gt = CameraLidarCalibrator.transform_to_tau(R, T)
# for idx in range(tau_data.shape[0]):
#     tau_data[idx,:3] = \
#         R.from_rotvec(tau_data[idx,:3]).as_euler('xyz', degrees=True)
#     tau_inits[idx,:3] = \
#         R.from_rotvec(tau_inits[idx,:3]).as_euler('xyz', degrees=True)

fig = plt.figure(figsize=[12, 5])
ax = [fig.add_subplot(1, 6, 1)]
for i in range(2, 4):
    ax.append(fig.add_subplot(1, 6, i, sharey=ax[0]))

ax.append(fig.add_subplot(1, 6, 4))
ax[-1].yaxis.tick_right()
for i in range(5, 7):
    ax.append(fig.add_subplot(1, 6, i, sharey=ax[3]))
    ax[-1].yaxis.tick_right()

for param_idx, param_name in enumerate(fields):
    if param_idx == 0:
        ax[param_idx].set_ylabel('Rotational deviation from mean(deg)')
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

ax[-1].set_ylabel('Translational deviation from mean(m)')
ax[-1].yaxis.set_label_position("right")
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, 'extrinsics_deviation_box.png'))
# plt.show()
plt.close()

plt.figure(figsize=[12, 5])
for param_idx, param_name in enumerate(fields):
    plt.subplot(1, 6, 1+param_idx)
    plt.title(param_name)
    plt.scatter(range(tau_data.shape[0]), tau_data[:, param_idx], c='b')
    plt.scatter(range(tau_inits.shape[0]), tau_inits[:, param_idx], c='r')
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, 'extrinsics_deviation.png'))
# plt.show()
plt.close()

plt.figure(figsize=[12, 5])
for param_idx, param_name in enumerate(fields):
    plt.subplot(2, 3, 1+param_idx)
    plt.title(param_name)
    counts, bins = np.histogram(
        tau_data[:, param_idx]-np.mean(tau_data[:, param_idx])
    )
    plt.hist(bins[:-1], bins, weights=counts)
plt.tight_layout()
plt.savefig(os.path.join(LOG_DIR, 'extrinsics_distribution_hist.png'))
plt.close()

# Std Deviation
table = PrettyTable()
table.field_names = [
    "N_sample", 'R1_std', 'P2_std', 'R3_std', "X_std", "Y_std", "Z_std"
]

row = [tau_data.shape[0]]
for param_idx in range(6):
    inliner = outliers_iqr(tau_data[:,param_idx])
    if param_idx < 3:
        row += [np.std(np.rad2deg(tau_data[inliner, param_idx]))]
    else:
        row += [np.std(tau_data[inliner, param_idx])]
table.add_row(row)
print(table)

table_data = table.get_string()
with open(os.path.join(LOG_DIR, 'error_metrics.txt'), 'w') as table_file:
    table_file.write(table_data)

# sys.exit()
plt.figure()
plt.clf()
plt.subplot(1, 5, 1)
plt.title('X')
plt.scatter(range(tau_inits.shape[0]), tau_inits[:, 3], c='r')
plt.scatter(range(tau_data.shape[0]), tau_data[:, 3], c='b')
axes = plt.gca()
xlims = axes.get_xlim()
plt.hlines(tau_gt[3], xlims[0], xlims[1])

plt.subplot(1, 5, 2)
plt.title('Y')
plt.scatter(range(tau_inits.shape[0]), tau_inits[:, 4], c='r')
plt.scatter(range(tau_data.shape[0]), tau_data[:, 4], c='b')
axes = plt.gca()
xlims = axes.get_xlim()
plt.hlines(tau_gt[4], xlims[0], xlims[1])

plt.subplot(1, 5, 3)
plt.title('Z')
plt.scatter(range(tau_inits.shape[0]), tau_inits[:, 5], c='r')
plt.scatter(range(tau_data.shape[0]), tau_data[:, 5], c='b')
axes = plt.gca()
xlims = axes.get_xlim()
plt.hlines(tau_gt[5], xlims[0], xlims[1])

plt.subplot(1, 5, 4)
plt.title('Angle')
angle_vec = np.linalg.norm(tau_data[:, :3], axis=1, ord=2)
angle_gt = np.linalg.norm(tau_gt[:3], ord=2)
angle_vec_init = np.linalg.norm(tau_inits[:, :3], axis=1, ord=2)
plt.scatter(range(tau_inits.shape[0]), angle_vec_init, c='r')
plt.scatter(range(tau_data.shape[0]), angle_vec, c='b')
axes = plt.gca()
xlims = axes.get_xlim()
plt.hlines(angle_gt, xlims[0], xlims[1])

plt.subplot(1, 5, 5)
plt.title('Axis')
axis_vec = tau_data[:, :3]/np.expand_dims(np.linalg.norm(tau_data[:, :3], axis=1, ord=2), 1)
axis_gt = tau_gt[:3]/np.linalg.norm(tau_gt[:3], ord=2)
axis_vec_init = tau_inits[:, :3] / np.expand_dims(np.linalg.norm(tau_inits[:, :3], axis=1, ord=2), 1)
axis_diffs = np.linalg.norm(axis_vec - axis_gt, axis=1, ord=2)
plt.scatter(range(axis_diffs.shape[0]), axis_diffs, c='b')
axes = plt.gca()
xlims = axes.get_xlim()
plt.hlines(0, xlims[0], xlims[1])
# plt.show()
plt.savefig(os.path.join(LOG_DIR, 'extrinsics_deviation.png'))
plt.show()

# Mean, Median, Std Deviation
table = PrettyTable()
table.field_names = ["X_med", "X_std", "Y_med", "Y_std", "Z_med", "Z_std",
                     "angle_med", "angle_std", "axis_med", "axis_std"]

x_error = tau_data[:, 3] - tau_gt[3]
y_error = tau_data[:, 4] - tau_gt[4]
z_error = tau_data[:, 5] - tau_gt[5]
angle_error = np.linalg.norm(tau_data[:, :3], ord=2, axis=1) - np.linalg.norm(tau_gt[:3], ord=2)
axis_error = axis_diffs
errors = [x_error, y_error, z_error, angle_error, axis_error]

row = []
for param_idx, param_name in enumerate(['X', 'Y', 'Z', 'angle', 'axis']):
    row += [np.median(errors[param_idx]), np.std(errors[param_idx])]
table.add_row(row)
print(table)

table_data = table.get_string()
with open(os.path.join(LOG_DIR, 'error_metrics.txt'), 'w') as table_file:
    table_file.write(table_data)

