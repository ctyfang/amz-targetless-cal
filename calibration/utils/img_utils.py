#!/usr/bin/env python
"""
Module for Generating Point Cloud Projections
used for Fusion Miscalibration Detection Network
"""

import cv2
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

def get_boundry(image, center, sigma):
    """
    For a given point, find the boundry of the 3-sigma batch.

    param: image: base image
    param: center: center pixel location (x, y)
    param: sigma: standard deviation of the Gaussian kernel
    return: distance to top, bottom, left, and right end of the batch
    """
    top = min(3 * sigma, center[0])
    bot = min(3*sigma+1, image.shape[0] - center[0] - 1)
    left = min(3 * sigma, center[1])
    right = min(3*sigma+1, image.shape[1] - center[1] - 1)
    return top, bot, left, right


def plot_2d(values, figname=None):
    """ plot a 2d numpy array """
    y = range(values.shape[0]+1)
    x = range(values.shape[1]+1)
    levels = MaxNLocator(nbins=15).tick_values(np.amin(values),
                                               np.amax(values))
    cmap = plt.get_cmap('hot')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    scale = np.power(10, np.log10(max(values.shape)).astype(int)-1)

    fig = plt.figure(figsize=values.T.shape[:2]/scale)
    ax0 = fig.add_subplot(111)
    plot = ax0.pcolormesh(x,
                          y,
                          values[::-1, :],
                          cmap=cmap,
                          norm=norm)
    fig.colorbar(plot, ax=ax0)
    ax0.set_title('pcolormesh with levels')
    ax0.set_xlim(0, values.shape[1])
    ax0.set_ylim(0, values.shape[0])
    plt.axis('equal')
    if figname is not None:
        plt.savefig(figname)
    plt.pause(3)
    plt.close()


def getGaussianKernel2D(sigma):
    """Given sigma, get 2D kernel of dimensions (6*int(sigma), 6*int(sigma))"""
    gauss1d = cv2.getGaussianKernel(6*int(sigma)+1, sigma).astype(np.float32)
    gaussian = np.dot(gauss1d, gauss1d.T)
    return gaussian


def outside_image(image, pixel):
    """Check whether a given pixel location (x, y) is outside the given image"""
    return (pixel[0] < 0 or pixel[0] >= image.shape[0] or
            pixel[1] < 0 or pixel[1] >= image.shape[1])


def scalar_to_color(pc, score=None, min_d=0, max_d=60):
    """
    print Color(HSV's H value) corresponding to score
    """
    if score is None:
        score = np.sqrt(
            np.power(pc[:, 0], 2) +
            np.power(pc[:, 1], 2) +
            np.power(pc[:, 2], 2))

    np.clip(score, 0, max_d, out=score)
    # max distance is 120m but usually not usual

    norm = plt.Normalize()
    colors = plt.cm.jet(norm(score))

    return (colors[:, :3] * 255).astype(np.uint8)


def draw_point_matches(img_1, points_1, img_2, points_2):
    """
    Draw corresponding points between 2 images.
    Corresponding points are drawn with same color.
    Lines connecting correspondences are also drawn.

    param: img_1    : 1st image
    param: points_1 : points on the 1st image
    param: img_2    : 2nd image
    param: points_2 : points on the 2nd image
    return: concatenated image with points and lines drawn
    """
    h, w = img_1.shape[0], img_1.shape[1]
    img_cat = np.concatenate([img_1, img_2], axis=1)
    img_cat = np.repeat(np.expand_dims(img_cat, axis=2), 3, axis=2)

    for point_idx in range(points_1.shape[0]):
        pt_1 = points_1[point_idx, :]
        pt_2 = points_2[point_idx, :]
        pt_2[0] += w

        color = (np.random.uniform()*255.0,
                 np.random.uniform()*255.0,
                 np.random.uniform()*255.0)
        print(color)
        img_cat = cv2.line(
            img_cat, tuple(pt_1.tolist()), tuple(pt_2.tolist()), color
        )

    return img_cat