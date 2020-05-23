import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
import json
import os

# Experimental data
TAU_INIT_PATH = '../output/logs/tau_inits.npy'
TAU_FINAL_PATH = '../output/logs/tau_data.npy'
EXP_PARAM_PATH = '../output/logs/params.json'
LOG_DIR = '../output/logs'

# Plot deviation of solutions wrt ground-truth
tau_inits = np.load(TAU_INIT_PATH)
tau_data = np.load(TAU_FINAL_PATH)

with open(EXP_PARAM_PATH, 'r') as param_json:
    exp_params = json.load(param_json)
tau_gt = np.asarray(exp_params['tau_gt'])

plt.figure()
plt.clf()
plt.subplot(1, 5, 1)
plt.title('X')
plt.scatter(range(tau_data.shape[0]), tau_data[:, 3], c='b')
plt.scatter(range(tau_inits.shape[0]), tau_inits[:, 3], c='r')
axes = plt.gca()
xlims = axes.get_xlim()
plt.hlines(tau_gt[3], xlims[0], xlims[1])

plt.subplot(1, 5, 2)
plt.title('Y')
plt.scatter(range(tau_data.shape[0]), tau_data[:, 4], c='b')
plt.scatter(range(tau_inits.shape[0]), tau_inits[:, 4], c='r')
axes = plt.gca()
xlims = axes.get_xlim()
plt.hlines(tau_gt[4], xlims[0], xlims[1])

plt.subplot(1, 5, 3)
plt.title('Z')
plt.scatter(range(tau_data.shape[0]), tau_data[:, 5], c='b')
plt.scatter(range(tau_inits.shape[0]), tau_inits[:, 5], c='r')
axes = plt.gca()
xlims = axes.get_xlim()
plt.hlines(tau_gt[5], xlims[0], xlims[1])

plt.subplot(1, 5, 4)
plt.title('Angle')
angle_vec = np.linalg.norm(tau_data[:, :3], axis=1, ord=2)
angle_gt = np.linalg.norm(tau_gt[:3], ord=2)
angle_vec_init = np.linalg.norm(tau_inits[:, :3], axis=1, ord=2)
plt.scatter(range(tau_data.shape[0]), angle_vec, c='b')
plt.scatter(range(tau_inits.shape[0]), angle_vec_init, c='r')
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
plt.show()

# Mean, Median, Std Deviation
table = PrettyTable()
table.field_names = ["X_med", "X_std", "Y_med", "Y_std", "Z_med", "Z_std",
                     "angle_med", "angle_std", "axis_med", "axis_std"]

x_error = abs(tau_data[:, 3] - tau_gt[3])
y_error = abs(tau_data[:, 4] - tau_gt[4])
z_error = abs(tau_data[:, 5] - tau_gt[5])
angle_error = abs(np.linalg.norm(tau_data[:, :3], ord=2, axis=1) - np.linalg.norm(tau_gt[:3], ord=2))
axis_error = axis_diffs
errors = [x_error, y_error, z_error, angle_error, axis_error]

row = []
for param_idx, param_name in enumerate(['X', 'Y', 'Z', 'angle', 'axis']):
    row += [np.median(errors[param_idx]), np.std(errors[param_idx])]
table.add_row(row)
print('ERROR METRICS (wrt KITTI)')
print(table)

std_table = PrettyTable()
std_table.field_names = ["X_std", 'Y_std', "Z_std", "Angle_std"]
def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

x_data = reject_outliers(tau_inits[:, 3], 2)
y_data = reject_outliers(tau_inits[:, 4], 2)
z_data = reject_outliers(tau_inits[:, 5], 2)
angle_data = reject_outliers(np.linalg.norm(tau_inits[:, :3], ord=2, axis=1), 2)
std_table.add_row([np.std(x_data), np.std(y_data), np.std(z_data),
                   np.std(angle_data)])
print('DISTRIBUTION OF Initializations')
print(std_table)

plt.figure()
plt.subplot(1, 4, 1)
plt.scatter(range(x_data.shape[0]), x_data)

plt.subplot(1, 4, 2)
plt.scatter(range(y_data.shape[0]), y_data)

plt.subplot(1, 4, 3)
plt.scatter(range(z_data.shape[0]), z_data)

plt.subplot(1, 4, 4)
plt.scatter(range(angle_data.shape[0]), angle_data)
plt.show()

std_table = PrettyTable()
std_table.field_names = ["X_std", 'Y_std', "Z_std", "Angle_std"]

x_data = reject_outliers(tau_data[:, 3], 2)
y_data = reject_outliers(tau_data[:, 4], 2)
z_data = reject_outliers(tau_data[:, 5], 2)
angle_data = reject_outliers(np.linalg.norm(tau_data[:, :3], ord=2, axis=1), 2)
std_table.add_row([np.std(x_data), np.std(y_data), np.std(z_data),
                   np.std(angle_data)])
print('DISTRIBUTION OF SOLUTIONS')
print(std_table)

plt.figure()
plt.subplot(1, 4, 1)
plt.scatter(range(x_data.shape[0]), x_data)

plt.subplot(1, 4, 2)
plt.scatter(range(y_data.shape[0]), y_data)

plt.subplot(1, 4, 3)
plt.scatter(range(z_data.shape[0]), z_data)

plt.subplot(1, 4, 4)
plt.scatter(range(angle_data.shape[0]), angle_data)
plt.show()
table_data = table.get_string()
with open(os.path.join(LOG_DIR, 'error_metrics.txt'), 'w') as table_file:
    table_file.write(table_data)