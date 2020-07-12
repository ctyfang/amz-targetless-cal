"""Utility functions for handling images and pointclouds"""
import numpy as np
import open3d as o3d
from scipy.spatial import ckdtree
from scipy.interpolate import griddata
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import cv2 as cv

from copy import deepcopy
from random import randint
import time
from time import sleep


def pc_visualize_edges(xyz, edge_idxs, edge_scores):
    v_min = np.min(edge_scores)
    v_max = np.max(edge_scores)

    edge_points = xyz[edge_idxs, :]
    edge_scores = edge_scores[edge_idxs]
    visualize_xyz_scores(edge_points, edge_scores, vmin=v_min, vmax=v_max)


def get_first_and_last_channel_idxs(pc, hor_res=0.2):
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]
    polar_angle = np.arctan2(np.sqrt(np.square(x) + np.square(y)), z)

    # Find the indices of the 360 / horizontal_resolution smallest/largest polar angles
    size_channel = 2 * int(360 / hor_res)
    neg_mask_smallest_angle = np.argpartition(
        polar_angle, size_channel)[:size_channel]
    neg_mask_largest_angle = np.argpartition(
        polar_angle, -size_channel)[-size_channel:]
    boundary_idxs = np.concatenate(
        (neg_mask_largest_angle, neg_mask_smallest_angle), axis=0)

    return np.unique(boundary_idxs)


def visualize_xyz_scores(xyz, scores, vmin=None, vmax=None):
    if vmin is None:
        vmin = np.min(scores)
    if vmax is None:
        vmax = np.max(scores)

    norm = matplotlib.colors.Normalize(
        vmin=vmin, vmax=vmax, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    colors = np.asarray([mapper.to_rgba(x)[:3] for x in scores])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()


def compute_centerscore(nn_xyz, center_xyz, max_nn_d):
    # Description: Compute modified center-based score. Distance between center (center_xyz),
    #              and the neighborhood (N x 3) around center_xyz. Result is scaled by max
    #              distance in the neighborhood, as done in Kang 2019.

    centroid = np.mean(nn_xyz, axis=0)
    norm_dist = np.linalg.norm(center_xyz - centroid)/max_nn_d
    return norm_dist


def compute_planarscore(nn_xyz, center_xyz):

    complete_xyz = np.concatenate([nn_xyz, center_xyz.reshape((1, 3))], axis=0)
    centroid = np.mean(complete_xyz, axis=0)
    centered_xyz = complete_xyz - centroid

    # Build structure tensor
    n_points = centered_xyz.shape[0]
    s = np.zeros((3, 3))
    for i in range(n_points):
        s += np.dot(centered_xyz[i, :].T, centered_xyz[i, :])
    s /= n_points

    # Compute planarity of neighborhood using SVD (Xia & Wang 2017)
    _, eig_vals, _ = np.linalg.svd(s)
    planarity = 1 - (eig_vals[1] - eig_vals[2])/eig_vals[0]

    return planarity


def visualize_neighborhoods(xyz):
    kdtree = ckdtree.cKDTree(xyz)

    colors = np.zeros((xyz.shape[0], 3))
    colors[:, 0] = 1

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    def highlight_neighborhood(vis):
        point_idx = randint(0, xyz.shape[0])
        new_colors = deepcopy(colors)
        new_colors[point_idx] = [0, 1, 0]

        neighbors_d, neighbors_i = kdtree.query(xyz[point_idx, :], 1000)
        new_colors[neighbors_i] = [0, 1, 0]
        pcd.colors = o3d.utility.Vector3dVector(new_colors)
        vis.update_geometry(pcd)
        sleep(0.5)

    o3d.visualization.draw_geometries_with_animation_callback(
        [pcd], highlight_neighborhood)


def custom_draw_xyz_with_params(xyz, camera_params):
    # Construct pcd object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pc', visible=True,
                      width=camera_params.intrinsic.width, height=camera_params.intrinsic.height)
    ctr = vis.get_view_control()
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(camera_params)
    vis.run()


def custom_draw_geometry_with_key_callbacks(pcd):

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def load_render_option(vis):
        vis.get_render_option().load_from_json(
            "../../TestData/renderoption.json")
        return False

    def capture_depth(vis):
        depth = vis.capture_depth_float_buffer()
        plt.imshow(np.asarray(depth))
        plt.show()
        return False

    def capture_image(vis):
        image = vis.capture_screen_float_buffer()
        plt.imshow(np.asarray(image))
        plt.show()
        return False

    def save_camera_model(vis):
        ctr = vis.get_view_control()
        param = ctr.convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters(
            './configs/o3d_camera_model.json', param)

    key_to_callback = {}
    key_to_callback[ord("K")] = change_background_to_black
    key_to_callback[ord("R")] = load_render_option
    key_to_callback[ord(",")] = capture_depth
    key_to_callback[ord(".")] = capture_image
    key_to_callback[ord("C")] = save_camera_model
    o3d.visualization.draw_geometries_with_key_callbacks(
        [pcd], key_to_callback)


def get_pc_pixels(pc, R, T, K, img_dims):
    R = R.reshape((3, 3))
    T = T.reshape((3, 1))
    K = K.reshape((3, 3))

    """Remove points behind the vehicle"""
    pc = pc[pc[:, 0] > 0, :]

    """Separate position from reflectance"""
    refl = pc[:, 3]
    pc = pc[:, :3]

    """Transform from LiDAR to camera frame, project onto image"""
    pc_cam = np.matmul(R, np.transpose(pc)) + T
    pc_pixels = np.matmul(K, pc_cam)
    pc_pixels /= pc_pixels[2, :]
    pc_pixels = np.transpose(pc_pixels[:2, :])  # Reshape to Nx2

    """Remove pixels outside image bounds"""
    x_mask = np.logical_and(pc_pixels[:, 0] >= 0,
                            pc_pixels[:, 0] <= img_dims[1])
    y_mask = np.logical_and(pc_pixels[:, 1] >= 0,
                            pc_pixels[:, 1] <= img_dims[0])
    valid_idxs = np.squeeze(np.argwhere(np.logical_and(x_mask, y_mask)))
    pc_pixels = pc_pixels[valid_idxs, :]
    return pc_pixels


def gen_reflectance_image(pc, R, T, K, img_dims, fill=False, fill_rad=3):
    """Given a pointcloud, rotation matrix, translation vector, and camera
     intrinsics, generate the reflectance image with specified image dimensions.
     image_dims specified as (h, w). PC is a numpy array of shape (N, 4)
     where each row is [X, Y, Z, Reflectance] with reflectance in range [0, 1].

     Return the reflectance image, and a mask image. Mask image indicates the
     areas with pixels.
     """

    R = R.reshape((3, 3))
    T = T.reshape((3, 1))
    K = K.reshape((3, 3))

    """Remove points behind the vehicle"""
    pc = pc[pc[:, 0] > 0, :]

    """Separate position from reflectance"""
    refl = pc[:, 3]
    pc = pc[:, :3]

    """Transform from LiDAR to camera frame, project onto image"""
    pc_cam = np.matmul(R, np.transpose(pc)) + T
    pc_pixels = np.matmul(K, pc_cam)
    pc_pixels /= pc_pixels[2, :]
    pc_pixels = np.transpose(pc_pixels[:2, :]) # Reshape to Nx2

    """Remove pixels outside image bounds"""
    x_mask = np.logical_and(pc_pixels[:, 0] >= 0,
                            pc_pixels[:, 0] <= img_dims[1])
    y_mask = np.logical_and(pc_pixels[:, 1] >= 0,
                            pc_pixels[:, 1] <= img_dims[0])
    valid_idxs = np.squeeze(np.argwhere(np.logical_and(x_mask, y_mask)))
    pc_pixels = pc_pixels[valid_idxs, :]
    refl = refl[valid_idxs]

    """Generate image and color with reflectance"""
    refl *= 255
    refl_img = np.zeros(img_dims, dtype=np.uint8)
    mask_img = np.zeros(img_dims, dtype=np.uint8)

    for i in range(pc_pixels.shape[0]):
        x, y = pc_pixels[i, :]
        x, y = int(x), int(y)
        refl_img[y-fill_rad:y+fill_rad, x-fill_rad:x+fill_rad] = refl[i]

    x_min, x_max = int(np.min(pc_pixels[:, 0])), int(np.max(pc_pixels[:, 0]))
    y_min, y_max = int(np.min(pc_pixels[:, 1])), int(np.max(pc_pixels[:, 1]))
    mask_img[y_min:y_max, x_min:x_max] = 1

    return refl_img, mask_img


def gen_depth_image(pc, R, T, K, img_dims, fill_rad=3):
    """Given a pointcloud, rotation matrix, translation vector, and camera
     intrinsics, generate the reflectance image with specified image dimensions.
     image_dims specified as (h, w). PC is a numpy array of shape (N, 4)
     where each row is [X, Y, Z, Reflectance] with reflectance in range [0, 1].

     Return the reflectance image, and a mask image. Mask image indicates the
     areas with pixels.
     """

    R = R.reshape((3, 3))
    T = T.reshape((3, 1))
    K = K.reshape((3, 3))

    """Remove points behind the vehicle"""
    pc = pc[pc[:, 0] > 0, :]
    pc = pc[pc[:, 0] < 30, :]

    """Separate position from reflectance"""
    pc = pc[:, :3]

    """Transform from LiDAR to camera frame, project onto image"""
    pc_cam = np.matmul(R, np.transpose(pc)) + T
    pc_pixels = np.matmul(K, pc_cam)
    pc_pixels /= pc_pixels[2, :]
    pc_pixels = np.transpose(pc_pixels[:2, :]) # Reshape to Nx2
    pc_cam = pc_cam.T

    """Remove pixels outside image bounds"""
    x_mask = np.logical_and(pc_pixels[:, 0] >= 0,
                            pc_pixels[:, 0] <= img_dims[1])
    y_mask = np.logical_and(pc_pixels[:, 1] >= 0,
                            pc_pixels[:, 1] <= img_dims[0])
    valid_idxs = np.squeeze(np.argwhere(np.logical_and(x_mask, y_mask)))
    pc_pixels = pc_pixels[valid_idxs, :]

    img_h, img_w = img_dims
    grid_x, grid_y = np.mgrid[0:img_w, 0:img_h]

    depth = np.linalg.norm(pc_cam, ord=2, axis=1)
    depth_valid = depth[valid_idxs]
    depth_img = griddata(pc_pixels,
                         depth_valid, (grid_x, grid_y),
                         method='linear').T

    depth_img = (depth_img * 255 / np.nanmax(depth_img)).astype(np.uint8)

    cv.imshow('Depth image with linear interpolation', depth_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return depth_img, None


def gen_synthetic_image(pc, refl, R, T, K, img_dims, fill_rad=3,
                        alpha_depth=0.5, alpha_refl=0.5):
    """Given a pointcloud, rotation matrix, translation vector, and camera
     intrinsics, generate the reflectance image with specified image dimensions.
     image_dims specified as (h, w). PC is a numpy array of shape (N, 4)
     where each row is [X, Y, Z, Reflectance] with reflectance in range [0, 1].

     Return the reflectance image, and a mask image. Mask image indicates the
     areas with pixels.
     """

    R = R.reshape((3, 3))
    T = T.reshape((3, 1))
    K = K.reshape((3, 3))

    """Remove points behind the vehicle"""
    # pc_mask = np.logical_and(pc[:, 0] > 0, pc[:, 0] < 30)
    # pc = pc[pc_mask, :]
    # refl = refl[pc_mask]

    """Separate position from reflectance"""
    pc = pc[:, :3]

    """Transform from LiDAR to camera frame, project onto image"""
    pc_cam = np.matmul(R, np.transpose(pc)) + T
    pc_pixels = np.matmul(K, pc_cam)
    pc_pixels /= pc_pixels[2, :]
    pc_pixels = np.transpose(pc_pixels[:2, :]) # Reshape to Nx2
    pc_cam = pc_cam.T

    """Remove pixels outside image bounds"""
    x_mask = np.logical_and(pc_pixels[:, 0] >= 0,
                            pc_pixels[:, 0] <= img_dims[1])
    y_mask = np.logical_and(pc_pixels[:, 1] >= 0,
                            pc_pixels[:, 1] <= img_dims[0])
    valid_idxs = np.squeeze(np.argwhere(np.logical_and(x_mask, y_mask)))
    pc_pixels = pc_pixels[valid_idxs, :]
    refl = refl[valid_idxs]

    img_h, img_w = img_dims
    grid_x, grid_y = np.mgrid[0:img_w, 0:img_h]

    """Depth Image"""
    depth = np.linalg.norm(pc_cam, ord=2, axis=1)
    depth_valid = depth[valid_idxs]
    depth_img = griddata(pc_pixels,
                         depth_valid, (grid_x, grid_y),
                         method='linear').T
    depth_img = (1 - depth_img * 255 / np.nanmax(depth_img)).astype(np.uint8)

    """Reflectance Image"""
    refl *= 255
    refl_img = np.zeros(img_dims, dtype=np.uint8)
    for i in range(pc_pixels.shape[0]):
        x, y = pc_pixels[i, :]
        x, y = int(x), int(y)
        refl_img[y - fill_rad:y + fill_rad, x - fill_rad:x + fill_rad] = refl[i]
    refl_img = cv.blur(refl_img, (3, 3))

    """Blend Image"""
    blend_img = cv.addWeighted(depth_img, alpha_depth, refl_img, alpha_refl, 0)
    return blend_img