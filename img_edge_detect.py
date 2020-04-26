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

import cv2
import numpy as np

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
        self.points = np.fromfile(
            str(input_dir)+'/velodyne_points/data/'+frame+'.bin',
            dtype=np.float32).reshape(-1, 4)[:, :3]
        self.image = cv2.imread(
            str(input_dir)+'/image_00/data/'+frame+'.png')

        self.P_rect = np.zeros((3, 3))
        self.R = np.zeros((3, 3))
        self.T = np.zeros((3, 1))
        self.load_cam(calib_dir)
        self.load_lid(calib_dir)
        self.pixels = np.ones((self.points.shape[0], 2))
        self.to_pixel()

    def load_cam(self, calib_dir):
        with open(str(calib_dir) + '/calib_cam_to_cam.txt', "r") as file:
            lines = file.readlines()

            for line in lines:
                (key, val) = line.split(':', 1)
                if key == ('P_rect_' + "00"):
                    self.P_rect = np.fromstring(val, sep=' ')\
                        .reshape(3, 4)[:3, :3]

    def load_lid(self, calib_dir):
        with open(str(calib_dir) + '/calib_velo_to_cam.txt', "r") as file:
            lines = file.readlines()

            for line in lines:
                (key, val) = line.split(':', 1)
                if key == 'R':
                    self.R = np.fromstring(val, sep=' ').reshape(3, 3)
                if key == 'T':
                    self.T = np.fromstring(val, sep=' ').reshape(3, 1)

    def to_pixel(self):
        '''
        Generate pixel coordinate for all points
        '''
        one_mat = np.ones((self.points.shape[0], 1))
        point_cloud = np.concatenate((self.points, one_mat), axis=1)

        # Project point into Camera Frame
        point_cloud_cam = np.matmul(np.hstack((self.R,
                                               self.T)),
                                    point_cloud.T)

        # Remove the Homogenious Term
        point_cloud_cam = np.matmul(self.P_rect, point_cloud_cam)

        # Normalize the Points into Camera Frame
        self.pixels = point_cloud_cam[::]/point_cloud_cam[::][-1]
        self.pixels = np.delete(self.pixels, 2, axis=0)

    def draw_points(self, image=None, FULL=True):
        '''
        Draw points within corresponding camera's FoV on image provided.
        If no image provided, points are drawn on an empty(black) background.
        '''
        if image is not None:
            image = np.dstack((image, image, image))
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            hsv_image = np.zeros(self.image.shape).astype(np.uint8)

        color = self.depth_color()
        if FULL:
            index = range(self.pixels.shape[1])
        else:
            index = np.random.choice(self.pixels.shape[1],
                                     size=int(self.pixels.shape[1]/10),
                                     replace=False)
        for i in index:
            if self.points[i, 0] < 0:
                continue
            if ((self.pixels[0, i] < 0) |
                    (self.pixels[1, i] < 0) |
                    (self.pixels[0, i] > hsv_image.shape[1]) |
                    (self.pixels[1, i] > hsv_image.shape[0])):
                continue
            cv2.circle(hsv_image,
                       (np.int32(self.pixels[0, i]),
                        np.int32(self.pixels[1, i])),
                       1, (int(color[i]), 255, 255), -1)

        return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

    def depth_color(self, min_d=0, max_d=120):
        """
        print Color(HSV's H value) corresponding to distance(m)
        close distance = red , far distance = blue
        """
        dist = np.sqrt(np.add(
            np.power(self.points[:, 0], 2),
            np.power(self.points[:, 1], 2),
            np.power(self.points[:, 2], 2)))
        np.clip(dist, 0, max_d, out=dist)
        # max distance is 120m but usually not usual
        return (((dist - min_d) / (max_d - min_d)) * 120).astype(np.uint8)


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
    images = sorted(glob.glob(str(input_dir)+'/image_02/data/*.png'))

    for img in images:
        image = cv2.imread(img)
        edge = cv2.Canny(image, 100, 200, L2gradient=True)
        frame = os.path.splitext(os.path.basename(img))[0]
        # load data from files - transform lidar to imu frame
        ptcloud = PtCloud(calib_dir, input_dir, frame)
        ptcloud.to_pixel()
        img_with_proj = ptcloud.draw_points(edge, FULL=True)
        cv2.imwrite(str(output_dir) + '/edge_full_proj.png', img_with_proj)
        img_with_proj = ptcloud.draw_points(edge, FULL=False)
        cv2.imwrite(str(output_dir) + '/edge_sub_proj.png', img_with_proj)

        cv2.imshow('Edge with projection', img_with_proj)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return


if __name__ == '__main__':
    main()
