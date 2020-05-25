import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

import os
from glob import glob

from scipy.ndimage.filters import gaussian_filter, convolve
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


class ImgEdgeDetector:

    def __init__(self, cfg, visualize=False):
        self.imgs = self.load_imgs(cfg.img_dir, cfg.frames)
        self.img_edge_scores = []
        self.imgs_edges = []

        self.ed_thresh_low = cfg.im_ed_score_thr1
        self.ed_thresh_high = cfg.im_ed_score_thr2

        # TODO: handle multiple images
        self.img_h, self.img_w = self.imgs[0].shape[:2]
        for frame_idx in range(len(self.imgs)):
            curr_h, curr_w = self.imgs[frame_idx].shape[:2]
            self.img_h, self.img_w = min(self.img_h, curr_h), min(self.img_w, curr_w)

    def img_detect(self, visualize=False):
        '''
        Compute pixel-wise edge score with non-maximum suppression
        Scores are normalized so that maximum score is 1
        '''
        for img in self.imgs:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            blurred = gaussian_filter(gray, sigma=2, order=0, mode='reflect')

            gradient_x = convolve(blurred, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            gradient_y = convolve(blurred, [[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

            img_edge_scores = np.sqrt(
                np.power(gradient_x, 2) + np.power(gradient_y, 2))
            img_edges = cv.Canny(img,
                                 self.ed_thresh_low,
                                 self.ed_thresh_high,
                                 L2gradient=True).astype(bool)
            img_edge_scores[~img_edges] = 0
            img_edge_scores = img_edge_scores / np.amax(img_edge_scores)
            self.img_edge_scores.append(img_edge_scores)
            self.imgs_edges.append(img_edges)

        if visualize:
            self.visualize_img_edges(range(len(self.imgs)))

    def visualize_img_edges(self, frames=[0]):
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
        """Load specified frames given KITTI dataset base-path. If frames=-1, loads all frames"""
        imgs = []

        if frames == -1:
            frame_paths = sorted(glob(os.path.join(path, 'image_00', 'data', '*.png')))
        else:
            frame_paths = [os.path.join(path, 'image_00', 'data', str(frame).zfill(10)) for frame in frames]

        for path in frame_paths:
            imgs.append(cv.imread(path))

        return imgs
