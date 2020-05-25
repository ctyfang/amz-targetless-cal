import numpy as np
import cv2 as cv
from tqdm import tqdm
import matplotlib.pyplot as plt

from calibration.camera_lidar_calibrator import CameraLidarCalibrator
from calibration.pc_edge_detector import PcEdgeDetector
from calibration.img_edge_detector import ImgEdgeDetector
from calibration.utils.data_utils import load_cam_cal
from calibration.utils.img_utils import draw_pc_points

import os
from glob import glob

# Set Parameters
BASE_DIR = '/home/carter/git/amz-targetless-cal2/output/logs2/exp_m=3_n=9_trials=5'
trial_idx = 1
# frames = [1, 3]
frames = [7, 8]

# Create plot
img_h, img_w = [370, 1224]
fig_h = 4
plt.figure(figsize=((img_w/img_h)*fig_h*len(frames), fig_h*len(frames)))

# Write frames
for idx, frame in enumerate(frames):
    before_img = cv.imread(os.path.join(BASE_DIR, 'trial_'+str(trial_idx), 'initial',
                                        'all_points_'+str(trial_idx)+'_frame_'+str(frame)+'.jpg'))
    before_img = cv.cvtColor(before_img, cv.COLOR_BGR2RGB)

    after_img = cv.imread(os.path.join(BASE_DIR, 'trial_'+str(trial_idx), 'stage_0',
                                        'all_points_frame_'+str(frame)+'.jpg'))
    after_img = cv.cvtColor(after_img, cv.COLOR_BGR2RGB)

    plt.subplot(2, len(frames), 1+idx)
    plt.imshow(before_img)
    axes = plt.gca()
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)

    plt.subplot(2, len(frames), len(frames)+1+idx)
    plt.imshow(after_img)
    axes = plt.gca()
    axes.get_xaxis().set_visible(False)
    axes.get_yaxis().set_visible(False)

plt.tight_layout()
plt.show()

