import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

import os
from glob import glob

from scipy.ndimage.filters import gaussian_filter, convolve
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


class ImgEdgeDetector:
    """Helper class for CameraLidarCalibrator. Load images from .png's,
    run edge-detection using Structured-Edge Detector or Canny to
    extract edge points for projection during optimization."""
    def __init__(self, cfg, visualize=False):
        if os.path.exists(cfg.dir):
            self.imgs = self.load_imgs(cfg.dir, cfg.frames)
        else:
            print("Image directory does not exist.")
            exit()

        self.img_edge_scores = []
        self.imgs_edges = []

        self.ed_thresh_low = cfg.im_ced_score_lower_thr
        self.ed_thresh_high = cfg.im_ced_score_upper_thr

        self.img_h, self.img_w = self.imgs[0].shape[:2]
        for frame_idx in range(len(self.imgs)):
            curr_h, curr_w = self.imgs[frame_idx].shape[:2]
            self.img_h, self.img_w = min(
                self.img_h, curr_h), min(self.img_w, curr_w)

    def __len__(self):
        return len(self.imgs)

    def img_detect(self, method='canny', visualize=False):
        """
        Detect edges on all the loaded images using Structured Edge Detector
        or the Canny Edge Detector. For SED, the edge scores are the model
        outputs. For Canny, the gradient magnitude is taken as the edge score.
        Edge scores are normalized to the [0, 1] range. NMS is performed in
        both cases.

        :param method: one of ['canny', 'sed']
        :param visualize: boolean, whether to show detected edge image or not.
        """

        model = os.path.join(os.getcwd(), 'calibration/configs/sed_model.yml')
        sed_model = cv.ximgproc.createStructuredEdgeDetection(model)
        for img in self.imgs:

            if method == 'canny':
                gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                blurred = gaussian_filter(gray, sigma=2, order=0, mode='reflect')

                gradient_x = convolve(
                    blurred, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                gradient_y = convolve(
                    blurred, [[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

                img_edge_scores = np.sqrt(
                    np.power(gradient_x, 2) + np.power(gradient_y, 2))
                img_edges = cv.Canny(img,
                                     self.ed_thresh_low,
                                     self.ed_thresh_high,
                                     L2gradient=True).astype(bool)

            else: # Use Structured Edge Detector
                img = img / 255
                img = img.astype(np.float32)
                img_edge_scores = sed_model.detectEdges(img)
                orientations = sed_model.computeOrientation(img_edge_scores)
                img_edges = sed_model.edgesNms(img_edge_scores, orientations, r=2, s=0, m=1)
                _, img_edges = cv.threshold(img_edges, 0.25, 1.0, cv.THRESH_BINARY)
                img_edges = img_edges.astype(np.bool)

            img_edge_scores[~img_edges] = 0
            img_edge_scores = img_edge_scores / np.amax(img_edge_scores)
            self.img_edge_scores.append(img_edge_scores)
            self.imgs_edges.append(img_edges)

        if visualize:
            self.visualize_img_edges(range(len(self.imgs)))

    def visualize_img_edges(self, frames=[0]):
        """
        Given the frame index [0, num_frames_loaded], draw the edge image
        colored by edge strength using matplotlib.pyplot.

        :param frames: iterable, frames to draw the edges for
        """
        im_x, im_y = np.meshgrid(
            np.linspace(0, self.img_edge_scores[0].shape[1],
                        self.img_edge_scores[0].shape[1] + 1),
            np.linspace(0, self.img_edge_scores[0].shape[0],
                        self.img_edge_scores[0].shape[0] + 1))

        levels = MaxNLocator(nbins=15).tick_values(0, 1)
        cmap = plt.get_cmap('hot')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        for frame in frames:
            fig, (ax0, ax1) = plt.subplots(nrows=2)
            plot = ax0.pcolormesh(im_x,
                                  im_y,
                                  self.img_edge_scores[frame][::-1, :],
                                  cmap=cmap,
                                  norm=norm)
            fig.colorbar(plot, ax=ax0)
            ax0.set_title('edge scores')
            plt.axis('equal')

            binary_edge = ax1.pcolormesh(im_x,
                                         im_y,
                                         self.imgs_edges[frame][::-1, :],
                                         cmap=cmap,
                                         norm=norm)
            fig.colorbar(binary_edge, ax=ax1)
            ax1.set_title('Binary Edge')
            plt.axis('equal')
            plt.show()
            plt.pause(0.001)

    @staticmethod
    def load_imgs(path, frames):
        """
        Load specified frames given KITTI dataset base-path. Base-path should
        contain the image_00 directory. Frames is an iterable that contains
        the indices of frames to be loaded. If frames=-1, loads all frames
        in the directory

        :param path: string, path to kitti directory containing image_00 dir
        :param frames: iterable, or -1
        :return:
        """

        imgs = []

        if frames == -1:
            frame_paths = sorted(
                glob(os.path.join(path, 'image_00', 'data', '*.png')))
        else:
            frame_paths = [os.path.join(path, 'image_00', 'data', str(
                frame).zfill(10)) + '.png' for frame in frames]

        if len(frame_paths) == 0:
            frame_paths = sorted(
                glob(os.path.join(path, 'forward_camera_filtered', '*.png'))
            )[0:6]

        for path in frame_paths:
            imgs.append(cv.imread(path))

        return imgs
