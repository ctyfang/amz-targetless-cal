import os
import open3d as o3d
import numpy as np
import yaml
import cv2
from glob import glob
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm as cm
from mpl_toolkits.mplot3d import Axes3D
import argparse


def create_pcd(xyz, scores=None, vmin=None, vmax=None, cmap=cm.tab20c):
    if vmin is None:
        vmin = np.min(scores)
    if vmax is None:
        vmax = np.max(scores)

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    colors = np.asarray(mapper.to_rgba(scores))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd


def create_pinhole_camera(file):
    with open(file) as f:
        try:
            data = yaml.load(f, Loader=yaml.CLoader)
        except AttributeError:
            data = yaml.load(f, Loader=yaml.Loader)

    # camera = o3d.camera.PinholeCameraParameters()
    # camera.intrinsic.set_intrinsics(data['image_width'], data['image_height'],
    #                                 data['camera_matrix']['data'][0],
    #                                 data['camera_matrix']['data'][4],
    #                                 data['camera_matrix']['data'][3],
    #                                 data['camera_matrix']['data'][5])
    intrinsic_matrix = np.array(data['camera_matrix']['data']).reshape(
        (data['camera_matrix']['rows'], data['camera_matrix']['cols']))
    distortion = np.array(data['distortion_coefficients']['data']).reshape(
        (data['distortion_coefficients']['rows'],
         data['distortion_coefficients']['cols']))
    # exit()
    return intrinsic_matrix, distortion


def get_extrinsics(pcd, camera):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=camera.intrinsic.width,
                      height=camera.intrinsic.height)

    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    cam = ctr.convert_to_pinhole_camera_parameters()
    cam.intrinsic.intrinsic_matrix = camera.intrinsic.intrinsic_matrix
    ctr.convert_from_pinhole_camera_parameters(cam)
    ctr.change_field_of_view(step=4)
    print(f'FoV: {vis.get_view_control().get_field_of_view()}')
    vis.run()
    return ctr.convert_to_pinhole_camera_parameters().extrinsic


def commandline_parser():
    parser = argparse.ArgumentParser(
        add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir_pc',
                        type=str,
                        default='',
                        required=True,
                        help='Path to directory containing point clouds')
    parser.add_argument('--dir_img',
                        type=str,
                        default='',
                        required=True,
                        help='Path to directory containing point clouds')
    cfg = parser.parse_args()

    return cfg


def select_img_points(img):
    ref_pt = np.zeros((2,))
    img_points = []

    def correspondence_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ref_pt[0], ref_pt[1] = x, y

    cv2.namedWindow("Correspondences")
    cv2.setMouseCallback("Correspondences", correspondence_cb)

    while True:
        curr_img = cv2.circle(img.copy(),
                              tuple(ref_pt.astype(np.uint)),
                              radius=2,
                              color=(255, 0, 0),
                              thickness=-1)
        cv2.imshow("Correspondences", curr_img)
        key = cv2.waitKey(1)
        if key == ord('y'):
            img_points.append(ref_pt.copy())
        elif key == ord('q'):
            if len(img_points) < 3:
                print("At least 3 points needed.")
            else:
                cv2.destroyAllWindows()
                break
    return np.array(img_points)


def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

def to_pixel(pcd, R, T, rect=np.eye(3)):
    '''
    Generate pixel coordinate for all points
    '''
    one_mat = np.ones((pcd.shape[0], 1))
    point_cloud = np.concatenate((pcd, one_mat), axis=1)

    transformation = np.hstack((R, T))

    # Project point into Camera Frame
    point_cloud_cam = np.matmul(transformation, point_cloud.T)

    # Remove the Homogenious Term
    point_cloud_cam = np.matmul(rect, point_cloud_cam)

    # Normalize the Points into Camera Frame
    pixels = point_cloud_cam[::] / point_cloud_cam[::][-1]
    pixels = np.delete(pixels, 2, axis=0)
    return pixels

def depth_color(points, min_d=0, max_d=120):
    """
    print Color(HSV's H value) corresponding to distance(m)
    close distance = red , far distance = blue
    """
    dist = np.sqrt(
        np.add(np.power(points[:, 0], 2),
                np.power(points[:, 1], 2),
                np.power(points[:, 2], 2)))
    np.clip(dist, 0, max_d, out=dist)
    # max distance is 120m but usually not usual
    return (((dist - min_d) / (max_d - min_d)) * 120).astype(np.uint8)

def draw_points(points, image, R, t):
    '''
    Draw points within corresponding camera's FoV on image provided.
    If no image provided, points are drawn on an empty(black) background.
    '''
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    color = depth_color(points)
    pixels = to_pixel(points, R, t)
    for i in range(pixels.shape[1]):
        if points[i, 0] < 0:
            continue
        if ((pixels[0, i] < 0) | (pixels[1, i] < 0) |
            (pixels[0, i] > hsv_image.shape[1]) |
            (pixels[1, i] > hsv_image.shape[0])):
            continue
        cv2.circle(
            hsv_image,
            (np.int32(pixels[0, i]), np.int32(pixels[1, i])),
            1, (int(color[i]), 255, 255), -1)
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


if __name__ == '__main__':
    pwd = os.path.abspath(".")
    cfg = commandline_parser()
    pcfiles = sorted(glob(os.path.join(cfg.dir_pc, '*.bin')))
    imgfile = sorted(glob(os.path.join(cfg.dir_img, '*.png')))

    img = cv2.imread(imgfile[0])
    img_points = select_img_points(img)

    pc = (np.fromfile(pcfiles[0], dtype=np.float64).reshape(-1, 6))[:, :4]
    pcd = create_pcd(pc[:, :3], pc[:, 3])
    pc_index = pick_points(pcd)

    K_mtx, distort = create_pinhole_camera(
        '/Users/jpzhong/Documents/git/sensor_fusion_2020/preprocessing/preprocessing/resources/forward.yaml'
    )

    _, R, t = cv2.solvePnP(pc[pc_index, :3], img_points, K_mtx, distort)
    R, _ = cv2.Rodrigues(R)
    print(R)
    print(t)
    cv2.imshow('projection', draw_points(pc[:,:3], img, R, t))
    cv2.waitKey(0)
    # camera = create_pinhole_camera(
    #     '/Users/jpzhong/Documents/git/sensor_fusion_2020/preprocessing/preprocessing/resources/forward.yaml'
    # )
    # pcd = create_pcd(pc[:, :3], pc[:, 3])
    # extrinsics = get_extrinsics(pcd, camera)
    # os.chdir(pwd)
    # with open('extrinsics.npy', 'wb') as f:
    #     np.save(f, extrinsics)
    # with open('extrinsics.npy', 'rb') as f:
    #     extrinsic = np.load(f)
    # print(extrinsic)
    # np.save('R.npy', extrinsics[:3, :3])
    # np.save('T.npy', extrinsics[:3, 3])

    # with open(f'tau_init_{}_to{}.yaml') as data:
