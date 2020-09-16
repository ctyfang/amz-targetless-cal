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

    camera = o3d.camera.PinholeCameraParameters()
    camera.intrinsic.set_intrinsics(data['image_width'], data['image_height'],
                                    data['camera_matrix']['data'][0],
                                    data['camera_matrix']['data'][4],
                                    data['camera_matrix']['data'][3],
                                    data['camera_matrix']['data'][5])
    # exit()
    return camera


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
        add_help=True,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--dir_pc', type=str, default='', required=True,
        help='Path to directory containing point clouds')
    parser.add_argument(
        '--dir_img', type=str, default='', required=True,
        help='Path to directory containing point clouds')
    cfg = parser.parse_args()

    return cfg


if __name__ == '__main__':
    pwd = os.path.abspath(".")
    cfg = commandline_parser()
    pcfiles = sorted(
        glob(
            os.path.join(cfg.dir_pc, '*.bin')))
    imgfile = sorted(
        glob(
            os.path.join(cfg.dir_img, '*.png')))

    pc = (np.fromfile(pcfiles[0], dtype=np.float64).reshape(-1, 6))[:, :4]
    img = cv2.imread(imgfile[0])
    cv2.imshow('image', img)
    cv2.waitKey(0)

    camera = create_pinhole_camera(
        '/Users/jpzhong/Documents/git/sensor_fusion_2020/preprocessing/preprocessing/resources/forward.yaml'
    )
    pcd = create_pcd(pc[:, :3], pc[:, 3])
    extrinsics = get_extrinsics(pcd, camera)
    os.chdir(pwd)
    with open('extrinsics.npy', 'wb') as f:
        np.save(f, extrinsics)
    with open('extrinsics.npy', 'rb') as f:
        extrinsic = np.load(f)
    print(extrinsic)
    # np.save('R.npy', extrinsics[:3, :3])
    # np.save('T.npy', extrinsics[:3, 3])
    cv2.destroyAllWindows()
    # with open(f'tau_init_{}_to{}.yaml') as data:
