import numpy as np
import cv2 as cv
from tqdm import tqdm

from calibration.camera_lidar_calibrator import CameraLidarCalibrator
from calibration.pc_edge_detector import PcEdgeDetector
from calibration.img_edge_detector import ImgEdgeDetector
from calibration.utils.data_utils import load_cam_cal
from calibration.utils.img_utils import draw_pc_points

import os
from glob import glob

# Set Parameters
image_dir = './opt_seq'
ext = 'jpg'
extra_end_frames = 10

# Get frames in sorted order
frame_paths = sorted(glob(os.path.join(image_dir, '*.'+ext)))
test_img = cv.imread(frame_paths[0])
img_h, img_w = test_img.shape[:2]

# Create video
os.makedirs(image_dir, exist_ok=True)
out = cv.VideoWriter(os.path.join(image_dir, 'video.avi'), cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 4, (img_w, img_h))

out.write(cv.imread(frame_paths[-1]))
# Write frames
for idx, path in tqdm(enumerate(frame_paths)):

    out.write(cv.imread(path))

    if idx == len(frame_paths)-1:
        for extra_idx in range(extra_end_frames):
            out.write(cv.imread(path))

out.release()

