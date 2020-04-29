"""Cost Gradient Computation according to Kang 2019"""
import numpy as np
import cv2 as cv
from scipy.ndimage import correlate
from scipy.linalg import expm


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


def cost(tau, sigma_in, camera_matrix, pc_edge_points, pc_edge_scores,
         pc_nn_sizes, edge_image, im_edge_scores):

    # Transform pc_edge_points to camera frame
    rot_vec = tau[:2]
    trans_vec = tau[2:]
    rot_mat = expm(skew(rot_vec))
    pc_edge_points_c = np.dot(rot_mat, pc_edge_points.T) + trans_vec.T

    cost = 0
    for idx in range(pc_edge_points_c.shape[0]):

        # Project edge point onto image
        curr_point = pc_edge_points_c[idx, :]
        proj_edge_point = np.dot(camera_matrix, curr_point)
        proj_edge_point /= proj_edge_point[2]

        # Get gaussian kernel
        curr_sigma = sigma_in / np.linalg.norm(curr_point, 2)
        mu_x, mu_y = proj_edge_point[:2]
        gauss_kernel = cv.getGaussianKernel(
            (6 * int(curr_sigma), 6 * int(curr_sigma)), curr_sigma, cv.CV_64F)

        # Get image patch inside the kernel
        edge_image_patch = edge_image[mu_y - 3 * int(curr_sigma):mu_y +
                                      3 * int(curr_sigma),
                                      mu_x - 3 * int(curr_sigma):mu_x +
                                      3 * int(curr_sigma)]

        edge_scores_patch = im_edge_scores[mu_y - 3 * int(curr_sigma):mu_y +
                                           3 * int(curr_sigma),
                                           mu_x - 3 * int(curr_sigma):mu_x +
                                           3 * int(curr_sigma)]

        # TODO: Handle edge cases (outside bounds, even kernel size)
        # TODO: Ensure edge_image_patch size matches kernel size
        # Weight image patch by edge scores
        weighted_edge_image_patch = np.multiply(edge_image_patch,
                                                edge_scores_patch)
        weighted_edge_image_patch /= np.max(im_edge_scores)

        weighted_edge_image_patch += (pc_edge_scores[idx] /
                                      np.max(pc_edge_scores))

        weighted_edge_image_patch /= (2 * pc_nn_sizes[idx])
        weighted_edge_image_patch *= (-1)

        curr_point_cost = correlate(weighted_edge_image_patch, gauss_kernel)
        cost += curr_point_cost

    return cost
