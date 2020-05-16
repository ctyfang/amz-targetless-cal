#!/usr/bin/env python
"""
Module for Generating Point Cloud Projections
used for Fusion Miscalibration Detection Network
"""
from pathlib import Path
import argparse
import glob
import sys
import os
import gc

import cv2
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from scipy.ndimage.filters import gaussian_filter, convolve

# Path
# ptCloud_dir = "data/2011_09_26/sync/velodyne_points/data/"
INPUT_DIR = 'data/2011_09_26_0017'
# calib_velo2cam_path = "data/calib_velo_to_cam.txt"
# calib_cam2cam_path = "data/calib_cam_to_cam.txt"


class PtCloud():
    '''
    Point Cloud Class
    '''

    def __init__(self, calib_dir, input_dir, frame):
        # data
        self.points = np.fromfile(str(input_dir) + '/velodyne_points/data/' +
                                  frame + '.bin',
                                  dtype=np.float32).reshape(-1, 4)[:, :3]
        self.image = cv2.imread(
            str(input_dir) + '/image_00/data/' + frame + '.png')

        self.p_rect = np.zeros((3, 3))
        self.R = np.zeros((3, 3))
        self.T = np.zeros((3, 1))
        self.load_cam(calib_dir)
        self.load_lid(calib_dir)
        self.pixels = np.ones((self.points.shape[0], 2))
        self.__to_pixel()
        self.cost = np.zeros((self.image.shape[0], self.image.shape[1]))

    def __to_pixel(self):
        '''
        Generate pixel coordinate for all points
        '''
        one_mat = np.ones((self.points.shape[0], 1))
        point_cloud = np.concatenate((self.points, one_mat), axis=1)

        # Project point into Camera Frame
        point_cloud_cam = np.matmul(np.hstack((self.R, self.T)), point_cloud.T)

        # Remove the Homogenious Term
        point_cloud_cam = np.matmul(self.p_rect, point_cloud_cam)

        # Normalize the Points into Camera Frame
        self.pixels = point_cloud_cam[::] / point_cloud_cam[::][-1]
        self.pixels = np.delete(self.pixels, 2, axis=0)

    def __add_gaussian_cost(self, pixel, sigma, show=False):
        '''
        Add Gaussian-like cost to cost map
        pixel: location of the Gaussian center
        sigma: standard deviation of the Gaussian
        show: Display the current cost map after adding
        '''
        x, y = np.meshgrid(np.linspace(-3 * sigma, 3 * sigma, 6 * sigma + 1),
                           np.linspace(-3 * sigma, 3 * sigma, 6 * sigma + 1))
        dist = np.sqrt(x * x + y * y)
        gaussian = np.exp(-(dist**2 / (2.0 * sigma**2)))

        # for pixel in pixels:
        for i in range(len(x)):
            for j in range(len(y)):
                if ((int(pixel[0] + y[i, j]) < 0) |
                    (int(pixel[1] + x[i, j]) < 0) |
                    (int(pixel[0] + y[i, j]) >= self.cost.shape[1]) |
                        (int(pixel[1] + x[i, j]) >= self.cost.shape[0])):
                    continue
                self.cost[int(pixel[1]+x[i, j]), int(pixel[0]+y[i, j])] += \
                    int(255 * gaussian[i, j])

        if show:
            self.visualize_cost()

    def __depth_color(self, min_d=0, max_d=120):
        """
        print Color(HSV's H value) corresponding to distance(m)
        close distance = red , far distance = blue
        """
        dist = np.sqrt(
            np.add(np.power(self.points[:, 0],
                            2), np.power(self.points[:, 1], 2),
                   np.power(self.points[:, 2], 2)))
        np.clip(dist, 0, max_d, out=dist)
        # max distance is 120m but usually not usual
        return (((dist - min_d) / (max_d - min_d)) * 120).astype(np.uint8)

    def img_detect(self, visualize=False):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blurred = gaussian_filter(gray, sigma=2, order=0, mode='reflect')

        ###### Gradients x and y (Sobel filters) ######
        im_x = convolve(blurred, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        im_y = convolve(blurred, [[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        ###### gradient and direction ########
        gradient = np.sqrt(np.power(im_x, 2.0) + np.power(im_y, 2.0))
        edge = cv2.Canny(self.image, 100, 200, L2gradient=True)
        gradient[np.where(edge == 0)] = 0
        if visualize:
            im_x, im_y = np.meshgrid(
                np.linspace(0, gradient.shape[1], gradient.shape[1] + 1),
                np.linspace(0, gradient.shape[0], gradient.shape[0] + 1))

            levels = MaxNLocator(nbins=15).tick_values(0, np.amax(gradient))
            cmap = plt.get_cmap('hot')
            norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

            fig, (ax0, ax1) = plt.subplots(nrows=2)
            plot = ax0.pcolormesh(im_x,
                                  im_y,
                                  gradient[::-1, :],
                                  cmap=cmap,
                                  norm=norm)
            fig.colorbar(plot, ax=ax0)
            ax0.set_title('pcolormesh with levels')
            edge_img = ax1.pcolormesh(im_x,
                                      im_y,
                                      edge[::-1, :],
                                      cmap=cmap,
                                      norm=norm)
            fig.colorbar(edge_img, ax=ax1)
            ax1.set_title('Binary Edge')
            plt.axis('equal')
            plt.show()
            plt.pause(5)

    def load_cam(self, calib_dir):
        '''
        Load camera parameters from file
        KITTI format expected
        '''
        with open(str(calib_dir) + '/calib_cam_to_cam.txt', "r") as file:
            lines = file.readlines()

            for line in lines:
                (key, val) = line.split(':', 1)
                if key == ('P_rect_' + "00"):
                    self.p_rect = np.fromstring(val, sep=' ')\
                        .reshape(3, 4)[:3, :3]

    def load_lid(self, calib_dir):
        '''
        Load LiDAR parameters from file
        KITTI format expected
        '''
        with open(str(calib_dir) + '/calib_velo_to_cam.txt', "r") as file:
            lines = file.readlines()

            for line in lines:
                (key, val) = line.split(':', 1)
                if key == 'R':
                    self.R = np.fromstring(val, sep=' ').reshape(3, 3)
                if key == 'T':
                    self.T = np.fromstring(val, sep=' ').reshape(3, 1)

    def visualize_cost(self):
        '''
        Visualize the current cost map
        '''
        np.clip(self.cost, 0, 255, out=self.cost)
        im_x, im_y = np.meshgrid(
            np.linspace(0, self.cost.shape[1], self.cost.shape[1] + 1),
            np.linspace(0, self.cost.shape[0], self.cost.shape[0] + 1))

        levels = MaxNLocator(nbins=15).tick_values(0, 255)
        cmap = plt.get_cmap('binary_r')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        fig, ax0 = plt.subplots(nrows=1)
        plot = ax0.pcolormesh(im_x,
                              im_y,
                              self.cost[::-1, :],
                              cmap=cmap,
                              norm=norm)
        fig.colorbar(plot, ax=ax0)
        ax0.set_title('pcolormesh with levels')
        plt.axis('equal')
        plt.show()

    def draw_points(self, image=None, scale=None):
        '''
        Draw points within corresponding camera's FoV on image provided.
        If no image provided, points are drawn on an empty(black) background.
        '''
        if image is not None:
            image = np.dstack((image, image, image))
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            hsv_image = np.zeros(self.image.shape).astype(np.uint8)

        color = self.__depth_color()
        if scale is None:
            index = range(self.pixels.shape[1])
        else:
            index = np.random.choice(self.pixels.shape[1],
                                     size=int(self.pixels.shape[1] / scale),
                                     replace=False)

        for i in index:
            if self.points[i, 0] < 0:
                continue
            if ((self.pixels[0, i] < 0) | (self.pixels[1, i] < 0) |
                (self.pixels[0, i] > hsv_image.shape[1]) |
                    (self.pixels[1, i] > hsv_image.shape[0])):
                continue
            cv2.circle(
                hsv_image,
                (np.int32(self.pixels[0, i]), np.int32(self.pixels[1, i])), 1,
                (int(color[i]), 255, 255), -1)
            self.__add_gaussian_cost(
                (np.int32(self.pixels[0, i]), np.int32(self.pixels[1, i])),
                sigma=10,
                show=False)

        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)


def get_boundry(image, center, sigma):
    top = min(3 * sigma, center[0])
    bot = min(3*sigma+1, image.shape[0] - center[0] - 1)
    left = min(3 * sigma, center[1])
    right = min(3*sigma+1, image.shape[1] - center[1] - 1)
    return top, bot, left, right


def plot_2d(values):
    y = range(values.shape[0]+1)
    x = range(values.shape[1]+1)
    levels = MaxNLocator(nbins=15).tick_values(np.amin(values),
                                               np.amax(values))
    cmap = plt.get_cmap('hot')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    fig, ax0 = plt.subplots(nrows=1)
    plot = ax0.pcolormesh(x,
                          y,
                          values[::-1, :],
                          cmap=cmap,
                          norm=norm)
    fig.colorbar(plot, ax=ax0)
    ax0.set_title('pcolormesh with levels')
    plt.axis('equal')
    plt.show()


def getGaussianKernel2D(sigma, visualize=False):
    """Given sigma, get 2D kernel of dimensions (6*int(sigma), 6*int(sigma))"""
    # sigma_int = int(sigma)
    # x, y = np.meshgrid(np.linspace(-3 * sigma_int, 3 * sigma_int, 6 * sigma_int + 1, dtype=np.float8),
    #                    np.linspace(-3 * sigma_int, 3 * sigma_int, 6 * sigma_int + 1, dtype=np.float8))
    # dist = np.sqrt(x * x + y * y)
    # # BUG: Square root over 2 * np.pi missing
    # gaussian = np.exp(-(dist**2 / (2.0 * sigma**2))) / (sigma*np.sqrt(2 * np.pi))
    # gaussian[dist > 3*sigma] = 0

    gauss1d = cv2.getGaussianKernel(6*int(sigma)+1, sigma).astype(np.float32)
    # gaussian = np.multiply(gauss1d.T, gauss1d)
    gaussian = np.dot(gauss1d, gauss1d.T)

    # BUG: I am not sure but I don't get why it's necessesary to make the
    # values in the grid add to one? Carter doesn't go it either.
    # gaussian = gaussian / np.sum(gaussian)

    # if visualize:
    #     levels = MaxNLocator(nbins=15).tick_values(0, np.amax(gaussian))
    #     cmap = plt.get_cmap('hot')
    #     norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    #
    #     fig, ax0 = plt.subplots(nrows=1)
    #     plot = ax0.pcolormesh(x,
    #                           y,
    #                           gaussian[::-1, :],
    #                           cmap=cmap,
    #                           norm=norm)
    #     fig.colorbar(plot, ax=ax0)
    #     ax0.set_title('pcolormesh with levels')
    #     plt.axis('equal')
    #     plt.show()
    return gaussian


def outside_image(image, pixel):
    return (pixel[0] < 0 or pixel[0] >= image.shape[0] or
            pixel[1] < 0 or pixel[1] >= image.shape[1])


def get_argument():
    '''
    Retrieve Required Input from User
    '''
    parser = argparse.ArgumentParser(
        description='Perform weighted edge detection based on image gradients')
    parser.add_argument('input_dir',
                        help='Directory containing all Sensor Data.\n\
                              Default KITTI Structure assumed.')
    parser.add_argument('output_dir',
                        help='Output directory for generated images')
    parser.add_argument('calib_dir',
                        help='Directory containing all calibration files')
    args = vars(parser.parse_args())

    # GET IMAGE NAMES
    input_dir = Path(args['input_dir'])
    output_dir = Path(args['output_dir'])
    calib_dir = Path(args['calib_dir'])

    if not input_dir.exists():
        print('Trying default input directory')
        input_dir = Path(INPUT_DIR)
        if not input_dir.exists():
            print('Invalid input image directory')
            print('please enter correct directory')
            sys.exit(1)
        print('Using default input directory')

    if not output_dir.exists():
        print('Output image directory did not exist')
        print('Directory created according to given name')
        output_dir.mkdir(parents=True, exist_ok=True)

    if not calib_dir.exists():
        print('Invalid File Path to Calibration File')
        sys.exit(1)
    return input_dir, output_dir, calib_dir


def main():
    '''
    MAIN() function.
    If called directly, an edge image is generated
    with randomly sampled partial point cloud projected
    '''
    input_dir, output_dir, calib_dir = get_argument()
    images = sorted(glob.glob(str(input_dir) + '/image_02/data/*.png'))

    for img in images:
        image = cv2.imread(img)
        edge = cv2.Canny(image, 100, 200, L2gradient=True)
        frame = os.path.splitext(os.path.basename(img))[0]
        # load data from files - transform lidar to imu frame
        ptcloud = PtCloud(calib_dir, input_dir, frame)
        ptcloud.img_detect(visualize=True)
        # img_with_proj = ptcloud.draw_points(edge)
        # cv2.imwrite(str(output_dir) + '/edge_full_proj.png', img_with_proj)

        img_with_proj = ptcloud.draw_points(edge, scale=1000)
        cv2.imwrite(str(output_dir) + '/edge_sub_proj.png', img_with_proj)
        # ptcloud.visualize_cost()

        cv2.imshow('Edge with projection', img_with_proj)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return


if __name__ == '__main__':
    main()
