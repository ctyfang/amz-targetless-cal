import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from scipy.ndimage.filters import gaussian_filter, convolve


class ImgEdgeDetector:

    def __init__(self, cfg):
        self.imgs = self.load_imgs(cfg.img_dir, cfg.frames)
        self.imgs_edge_scores = None
        self.imgs_edges = None

        self.img_detect(visualize=True)

    def img_detect(self, visualize=False):
        '''
        Compute pixel-wise edge score with non-maximum suppression
        Scores are normalized so that maximum score is 1
        '''
        gray = cv.cvtColor(self.imgs, cv.COLOR_BGR2GRAY)
        blurred = gaussian_filter(gray, sigma=2, order=0, mode='reflect')

        gradient_x = convolve(blurred, [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        gradient_y = convolve(blurred, [[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        self.imgs_edge_scores = np.sqrt(
            np.power(gradient_x, 2) + np.power(gradient_y, 2))
        self.imgs_edges = cv.Canny(self.imgs, 100, 200, L2gradient=True)
        self.imgs_edge_scores[np.where(self.imgs_edges == 0)] = 0
        self.imgs_edge_scores = \
            self.imgs_edge_scores/np.amax(self.imgs_edge_scores)

        if visualize:
            self.visualize_img_edges()

    def visualize_img_edges(self):
        im_x, im_y = np.meshgrid(
            np.linspace(0, self.imgs_edge_scores.shape[1],
                        self.imgs_edge_scores.shape[1] + 1),
            np.linspace(0, self.imgs_edge_scores.shape[0],
                        self.imgs_edge_scores.shape[0] + 1))

        levels = MaxNLocator(nbins=15).tick_values(0, 1)
        cmap = plt.get_cmap('hot')
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        fig, (ax0, ax1) = plt.subplots(nrows=2)
        plot = ax0.pcolormesh(im_x,
                              im_y,
                              self.imgs_edge_scores[::-1, :],
                              cmap=cmap,
                              norm=norm)
        fig.colorbar(plot, ax=ax0)
        ax0.set_title('edge scores')

        binary_edge = ax1.pcolormesh(im_x,
                                     im_y,
                                     self.imgs_edges[::-1, :],
                                     cmap=cmap,
                                     norm=norm)
        fig.colorbar(binary_edge, ax=ax1)
        ax1.set_title('Binary Edge')
        plt.axis('equal')
        plt.show()

    @staticmethod
    def load_imgs(path, frames):
        # TODO: Should always return an array of images, but rest of the code
        # needs to be changed for that
        if len(frames) <= 1:
            return cv.imread(str(path) + '/image_00/data/' +
                             str(frames[0]).zfill(10) + '.png')
        else:
            imgs = []
            for frame in frames:
                imgs.append(cv.imread(str(path) + '/image_00/data/' +
                                      str(frame).zfill(10) + '.png'))
            return imgs
