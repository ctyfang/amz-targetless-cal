"""Cost Gradient Computation according to Kang 2019"""
import numpy as np
import cv2 as cv
from scipy.ndimage import correlate
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from scipy.ndimage.filters import gaussian_filter,convolve
import gc
from calibration.utils.img_utils import *

def skew(vector):
    """
    this function returns a numpy array with the skew symmetric cross product matrix for vector.
    the skew symmetric cross product matrix is defined such that
    np.cross(a, b) = np.dot(skew(a), b)

    :param vector: An array like vector to create the skew symmetric cross product matrix for
    :return: A numpy array of the skew symmetric cross product vector
    """

    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])


def pairwise_gradient(tau, kernel, camera_matrix, edge_point):
    """ Single component of gradient dG(d_ij, sigma_i)/dtau

    Parameters:
    tau      -- (6, 1) vector. tau[:3] is the rotation vector, tau[3:] is the translation vector
    kernel   -- Current kernel used for pairwise cost G(d_ij, sigma_i),
    thr     -- threshold for discarding points with a lower edge score
    num_nn  -- number of nearest neighbors for calculating the edge score
    rad_nn  -- Edge scores use num_nn nearest neighbors, as well as all neighbors within a certain radius.
               rad_nn is the radius

    Return:a
    pc_edge_points
    pc_edge_scores
    max_pc_edge_score

    """
    return


def cost(rot_vec, trans_vec, camera_matrix, sigma_in, pc_edge_points,
         pc_edge_scores, edge_image, im_edge_scores):
    # Transform pc_edge_points to camera frame
    rot_mat, _ = cv.Rodrigues(rot_vec)
    # need to make sure trans_vec (3x1)
    pc_edge_points_c = (np.dot(rot_mat, pc_edge_points.T) + trans_vec).T
    cost_map = np.zeros(edge_image.shape)
    gaus_map = np.zeros(edge_image.shape)
    cost = 0
    for idx in range(pc_edge_points_c.shape[0]):
        # Project edge point onto image
        curr_point = pc_edge_points_c[idx, :]
        proj_edge_point = np.dot(camera_matrix, curr_point)
        proj_edge_point /= proj_edge_point[2]
        mu_x, mu_y = proj_edge_point[:2].astype(np.int)
        if outside_image(edge_image, (mu_y, mu_x)):
            continue

        sigma = int(sigma_in / np.linalg.norm(curr_point, 2))
        # Get gaussian kernel
        # Distance > 3 sigma is set to 0
        # and normalized so that the total Kernel = 1
        gauss2d = getGaussianKernel2D(sigma, False)
        top, bot, left, right = get_boundry(edge_image, (mu_y, mu_x), sigma)
        # Get image patch inside the kernel
        edge_scores_patch = im_edge_scores[mu_y - top: mu_y + bot,
                                           mu_x - left: mu_x + right]

        # weight = (normalized img score + normalized pc score) / 2 / |Omega_i|
        # Cost = Weight * Gaussian Kernal
        edge_scores_patch += pc_edge_scores[idx]
        edge_scores_patch /= 2
        kernal_patch = gauss2d[3*sigma-top: 3*sigma+bot,
                               3*sigma-left: 3*sigma+right]

        cost_patch = np.multiply(edge_scores_patch, kernal_patch)
        if cost_patch.all == 0:
            continue
    
        print(f'Cost: {np.sum(cost_patch)/(2*np.sum(cost_patch>0))}')
        cost_map[mu_y, mu_x] = np.sum(cost_patch) / (2 * np.sum(cost_patch > 0))
        gaus_map[mu_y - top: mu_y + bot,
                 mu_x - left: mu_x + right] += gauss2d[3*sigma-top: 3*sigma+bot,
                                                3*sigma-left: 3*sigma+right] 
    plot_2d(cost_map)
    plot_2d(gaus_map)
    gc.collect()
    return np.sum(cost_map)
