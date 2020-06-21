import cv2 as cv

import os

from calibration.utils.data_utils import *
from calibration.utils.pc_utils import gen_reflectance_image

"""Script parameters"""
visualize_imgs = True
visualize_kps = True
visualize_matches = True

"""Load image, pointcloud, extrinsics"""
data_dir = '/media/carter/Samsung_T5/3dv/2011_09_28/collection'
calib_dir = '/media/carter/Samsung_T5/3dv/2011_09_28/calibration'
frame_idx = 34

img = cv.imread(os.path.join(data_dir, 'image_00', 'data',
                             str(frame_idx).zfill(10) + '.png'))

pc = load_from_bin(os.path.join(data_dir, 'velodyne_points', 'data',
                                str(frame_idx).zfill(10) + '.bin'),
                   incl_refl=True)

K = load_cam_cal(calib_dir)
R, T = load_lid_cal(calib_dir)

# TODO
"""Perturb the extrinsics """

"""Generate reflectance and grayscale images"""
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_gray = cv.blur(img_gray, (7, 7))
img_refl, refl_mask = gen_reflectance_image(pc, R, T, K, img_gray.shape,
                                            fill=True, fill_rad=5)
img_refl = cv.blur(img_refl, (3, 3))

if visualize_imgs:
    img_cat = np.concatenate([img_gray, img_refl], axis=0)
    cv.imshow('Concatenated', img_cat)
    cv.waitKey(0)
    cv.destroyAllWindows()

"""Detect, visualize features"""
detector = cv.ORB_create(patchSize=91, edgeThreshold=91)
descriptor = cv.xfeatures2d_VGG.create(desc=101, isigma=1.0, scale_factor=0.75)
kp_g = detector.detect(img_gray, None)
kp_r = detector.detect(img_refl, refl_mask)
desc_g = descriptor.compute(img_gray, kp_g, None)
desc_r = descriptor.compute(img_refl, kp_r, refl_mask)

if visualize_kps:
    img_g_kp = np.repeat(np.expand_dims(img_gray, 2), 3, axis=2)
    cv.drawKeypoints(img_gray, kp_g, img_g_kp, color=(255, 0, 0), flags=0)
    img_r_kp = np.repeat(np.expand_dims(img_refl, 2), 3, axis=2)
    cv.drawKeypoints(img_refl, kp_r, img_r_kp, color=(255, 0, 0), flags=0)

    img_cat = np.concatenate([img_g_kp, img_r_kp], axis=0)
    cv.imshow('Gray (top) and Refl (bot) KPs', img_cat)
    cv.waitKey(0)
    cv.destroyAllWindows()

"""Match, visualize matches"""
bf = cv.BFMatcher()
matches = bf.knnMatch(desc_g, desc_r, k=1)

good = matches
# good = []
# for m, n in matches:
#     if m.distance < n.distance:
#         good.append([m])

if visualize_matches:
    img_gray_stack = np.repeat(np.expand_dims(img_gray, 2), 3, axis=2)
    img_refl_stack = np.repeat(np.expand_dims(img_refl, 2), 3, axis=2)
    img_match = cv.drawMatchesKnn(img_gray_stack, kp_g,
                                  img_refl_stack, kp_r,
                                  good[:10],
                                  None,
                                  flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    h, w, _ = img_match.shape
    cv.imshow('Matches', cv.resize(img_match, (w//2, h//2)))
    cv.waitKey(0)
    cv.destroyAllWindows()
