import cv2 as cv
from scipy.spatial.ckdtree import cKDTree

import os
import pickle

from calibration.utils.data_utils import *
from calibration.utils.pc_utils import *
from calibration.utils.img_utils import *

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
img_depth, _ = gen_depth_image(pc, R, T, K, img.shape[:2])
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_gray = cv.blur(img_gray, (7, 7))
img_refl, refl_mask = gen_reflectance_image(pc, R, T, K, img_gray.shape,
                                            fill=True, fill_rad=1)
img_refl = cv.blur(img_refl, (3, 3))

cv.imshow("refl", img_refl)
cv.waitKey(0)
cv.destroyAllWindows()

img_blend = cv.addWeighted(img_refl, 0.7, img_depth, 0.3, gamma=0)
if visualize_imgs:
    img_cat = np.concatenate([img_gray, img_blend], axis=0)
    cv.imshow('Concatenated', img_cat)
    cv.waitKey(0)
    cv.destroyAllWindows()

"""Manually Selection of Matching Keypoints"""
pc_pixels = get_pc_pixels(pc, R, T, K, img_gray.shape)
pc_pixel_tree = cKDTree(pc_pixels)

points_selected = 0
curr_img_name = "gray"
valid_img_names = ["gray", "blend"]
points_dict = {"gray": [], "blend": []}
refPt = np.asarray([0, 0])
def correspondence_cb(event, x, y, flags, param):
    global refPt
    global curr_img_name

    if event == cv.EVENT_LBUTTONDOWN:
        if curr_img_name == "gray":
            refPt[0], refPt[1] = x, y
        else:
            d, i = pc_pixel_tree.query(np.asarray([x, y]).reshape((1, 2)), 1)
            nearest_pixel = pc_pixels[i, :].astype(np.uint)[0]
            refPt[0], refPt[1] = nearest_pixel[0], nearest_pixel[1]

        print(refPt)

cv.namedWindow("Correspondences")
cv.setMouseCallback("Correspondences", correspondence_cb)

while True:
    if points_selected % 2 == 0:
        curr_img = img_gray
        curr_img_name = "gray"
    else:
        curr_img = img_blend
        curr_img_name = "blend"

    curr_img = np.repeat(np.expand_dims(curr_img, axis=2), 3, axis=2)
    curr_img = cv.circle(curr_img, tuple(refPt), radius=2, color=(255, 0, 0),
                         thickness=-1)
    cv.imshow("Correspondences", curr_img)
    key = cv.waitKey(1)

    if key == ord('y'):
        points_selected += 1
        points_dict[curr_img_name].append(np.asarray([refPt[0], refPt[1]]))

    if key == ord('q'):
        if points_selected % 2 == 0:
            break
        else:
            print("Uneven number of points. Select one more and try again.")

"""Draw Manually Selected Correspondences"""
test = np.asarray(points_dict["gray"])
print(test.shape)
match_img = draw_point_matches(img_gray, np.asarray(points_dict["gray"]),
                               img_blend, np.asarray(points_dict["blend"]))
cv.imshow("Matches", match_img)
cv.waitKey(0)
cv.destroyAllWindows()

"""Detect, visualize features"""
# detector = cv.ORB_create(patchSize=91, edgeThreshold=91)
# descriptor = cv.xfeatures2d_VGG.create(desc=100, isigma=1.0, scale_factor=0.75)
# kp_g = detector.detect(img_gray, None)
# kp_r = detector.detect(img_blend, refl_mask)
# desc_g = descriptor.compute(img_gray, kp_g, None)
# desc_r = descriptor.compute(img_refl, kp_r, refl_mask)
#
# if visualize_kps:
#     img_g_kp = np.repeat(np.expand_dims(img_gray, 2), 3, axis=2)
#     cv.drawKeypoints(img_gray, kp_g, img_g_kp, color=(255, 0, 0), flags=0)
#     img_r_kp = np.repeat(np.expand_dims(img_refl, 2), 3, axis=2)
#     cv.drawKeypoints(img_refl, kp_r, img_r_kp, color=(255, 0, 0), flags=0)
#
#     img_cat = np.concatenate([img_g_kp, img_r_kp], axis=0)
#     cv.imshow('Gray (top) and Refl (bot) KPs', img_cat)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
#
# """Match, visualize matches"""
# bf = cv.BFMatcher()
# matches = bf.knnMatch(desc_g, desc_r, k=2)
#
# good = []
# for m, n in matches:
#     if m.distance < 0.9*n.distance:
#         good.append([m])
#
# if visualize_matches:
#     img_gray_stack = np.repeat(np.expand_dims(img_gray, 2), 3, axis=2)
#     img_refl_stack = np.repeat(np.expand_dims(img_refl, 2), 3, axis=2)
#     img_match = cv.drawMatchesKnn(img_gray_stack, kp_g,
#                                   img_refl_stack, kp_r,
#                                   good,
#                                   None,
#                                   flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#     h, w, _ = img_match.shape
#     cv.imshow('Matches', cv.resize(img_match, (w//2, h//2)))
#     cv.waitKey(0)
#     cv.destroyAllWindows()
