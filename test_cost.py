import cv2
from calibration.camera_lidar_calibrator import *
from calibration.utils.config import command_line_parser
from calibration.utils.data_utils import *
from calibration.utils.img_utils import plot_2d
from calibration import img_edge_detector
from calibration.camera_lidar_calibrator import *
from cost import *

def pc_to_pixels(points, R, T, P_rect):
    '''
    Generate pixel coordinate for all points
    '''
    one_mat = np.ones((points.shape[0], 1))
    point_cloud = np.concatenate((points, one_mat), axis=1)

    # TODO: Perform transform without homogeneous term,
    #       if too memory intensive

    # Project point into Camera Frame
    point_cloud_cam = np.matmul(np.hstack((R, T)), point_cloud.T)

    # Remove the Homogeneous Term
    point_cloud_cam = np.matmul(P_rect, point_cloud_cam)

    # Normalize the Points into Camera Frame
    pixels = point_cloud_cam[::] / point_cloud_cam[::][-1]
    return np.delete(pixels, 2, axis=0)

def draw_points(image, points, FULL=True):
    """
    Draw points within corresponding camera's FoV on image provided.
    If no image provided, points are drawn on an empty(black) background.
    """

    if len(image.shape) == 2:
        image = np.dstack((image, image, image))
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    else:
        hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    rect = load_cam_cal('data')
    R, T = calib_velo2cam('data/calib_velo_to_cam.txt')
    pixels = pc_to_pixels(points, R, T, rect)
    color = pc_to_colors(points)
    if FULL:
        index = range(pixels.shape[1])
    else:
        index = np.random.choice(pixels.shape[1],
                                    size=int(pixels.shape[1] / 10),
                                    replace=False)
    for i in index:
        if points[i, 0] < 0:
            continue
        if ((pixels[0, i] < 0) | (pixels[1, i] < 0) |
            (pixels[0, i] > hsv_image.shape[1]) |
                (pixels[1, i] > hsv_image.shape[0])):
            continue
        cv.circle(
            hsv_image,
            (np.int32(pixels[0, i]), np.int32(pixels[1, i])), 1,
            (int(color[i]), 255, 255), -1)

    return cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)

def pc_to_colors(points, min_d=0, max_d=120):
    """
    print Color(HSV's H value) corresponding to distance(m)
    close distance = red , far distance = blue
    """
    dist = np.sqrt(
        np.add(np.power(points[:, 0],
                        2), np.power(points[:, 1], 2),
                np.power(points[:, 2], 2)))
    np.clip(dist, 0, max_d, out=dist)
    # max distance is 120m but usually not usual
    return (((dist - min_d) / (max_d - min_d)) * 120).astype(np.uint8)

# load image
input_dir_list = ['/media/carter/Samsung_T5/3dv/2011_09_26/2011_09_26_drive_0106_sync',
                  '/home/benjin/Development/Data/2011_09_26_drive_0106_sync',
                  'data/2011_09_26_0017']
calib_dir_list = ['/media/carter/Samsung_T5/3dv/2011_09_26/calibration',
                  '/home/benjin/Development/Data/2011_09_26_calib/2011_09_26',
                  'data']

cfg = command_line_parser()

input_dir = getPath(input_dir_list)
calib_dir = getPath(calib_dir_list)

cfg.pc_dir = input_dir
cfg.img_dir = input_dir
cfg.calib_dir = calib_dir

SHOW = True
img = img_edge_detector.ImgEdgeDetector(cfg, visualize=SHOW)

# load pointcloud
with open('test/pc_edge_scores.npy', 'rb') as f:
    pcs_edge_scores = np.load(f)
with open('test/pc_edge_idx.npy', 'rb') as f:
    pcs_edge_idxs = np.load(f)
# with open('pc_max_score.npy', 'rb') as f:
#     pcs_max_edge_score = np.load(f)

ptCloud = load_from_bin('data/2011_09_26_0017/velodyne_points/data/0000000029.bin')

print(ptCloud.shape)
print(pcs_edge_scores.shape)
print(pcs_edge_idxs.shape)

if SHOW:
    img_with_point = draw_points(cv2.Canny(img.imgs, 100, 200, L2gradient=True), ptCloud[pcs_edge_idxs])
    cv2.imshow('img_with_point', img_with_point)
    cv2.imwrite('generated/edge_demo.png', img_with_point)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# cost
rot_mtx, tra_vec = calib_velo2cam('data/calib_velo_to_cam.txt')
rot_vec, _ = cv2.Rodrigues(rot_mtx)
camera_matrix = load_cam_cal('data')
cost = cost(rot_vec, tra_vec, camera_matrix, 360, ptCloud[pcs_edge_idxs], pcs_edge_scores[pcs_edge_idxs], img.imgs_edges, img.imgs_edge_scores)
print(cost)
print('Done')