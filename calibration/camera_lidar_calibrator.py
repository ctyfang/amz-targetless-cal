from datetime import datetime
import gc
import yaml
import warnings

from KDEpy import FFTKDE

from scipy._lib._util import check_random_state
from scipy.optimize import minimize, basinhopping, least_squares
from scipy.stats import entropy
from scipy.spatial.ckdtree import cKDTree

from calibration.img_edge_detector import ImgEdgeDetector
from calibration.pc_edge_detector import PcEdgeDetector
from calibration.utils.data_utils import *
from calibration.utils.img_utils import *
from calibration.utils.pc_utils import *


class CameraLidarCalibrator:
    """Top-level class for lidar-camera calibration"""

    def __init__(self, cfg, visualize=False, tau_init=None):
        """Extract paths and thresholds from config dict. Load images and pcs,
        run edge detection on images and pcs. Select correspondences.

        :param cfg: Dictionary with calibration parameters. See utils/config.py.
        :param visualize: Boolean specifying whether or not to visualize outputs
                          of img/pc edge detection.
        :param tau_init: Initial guess for the extrinsic parameters.
        """
        self.visualize = visualize
        self.projected_points = []
        self.points_cam_frame = []
        self.projection_mask = []

        # calib_dir = os.path.join(cfg.dir, 'calibration')
        if os.path.exists(os.path.join(cfg.dir, 'calibration')):
            self.K = load_cam_cal(calib_dir)
            self.R, self.T = load_lid_cal(calib_dir)
        elif os.path.exists(os.path.join(cfg.calib_dir)):
            with open(cfg.calib_dir) as f:
                try:
                    data = yaml.load(f, Loader=yaml.CLoader)
                except AttributeError:
                    data = yaml.load(f, Loader=yaml.Loader)
            self.K = np.array(data['camera_matrix']['data']).reshape(
        (data['camera_matrix']['rows'], data['camera_matrix']['cols']))
        else:
            print("Please specify root to calibration file")
            exit()

        self.correspondences = []

        if tau_init is not None:
            self.tau = tau_init
        elif isinstance(self.R, np.ndarray) and isinstance(self.T, np.ndarray):
            self.tau = self.transform_to_tau(self.R, self.T)
        else:
            self.tau = np.zeros((1, 6))

        # Load point clouds/images into the detectors
        self.img_detector = ImgEdgeDetector(cfg, visualize=visualize)
        self.pc_detector = PcEdgeDetector(cfg, visualize=visualize)
        if len(self.img_detector) != len(self.pc_detector):
            print('Number of images does not match number of point clouds')
            print(f'{len(self.img_detector)} frames v.s. {len(self.pc_detector)} pcs')
            exit()
        self.num_frames = len(self.img_detector)
        
        print(f'{self.num_frames} pairs of images and pointclouds loaded.')

        # Calculate projected_points, points_cam_frame, projection_mask
        self.project_point_cloud()

        # Detect edges if automatic method is chosen
        if cfg.calibration_method == 'automatic':
            print('Executing image edge-detection.')
            self.img_detector.img_detect(method=cfg.im_ed_method,
                                        visualize=visualize)
            gc.collect()
            print('Image edge-detection completed.')
            print('Executing point cloud edge-detection.')
            with warnings.catch_warnings():
                # ignore runtime warning of ckdtree
                warnings.simplefilter("ignore")
                self.pc_detector.pc_detect(self.points_cam_frame,
                                        cfg.pc_ed_score_thr,
                                        cfg.pc_ed_num_nn,
                                        cfg.pc_ed_rad_nn,
                                        visualize=visualize)
            gc.collect()
            print('Point Cloud edge-detection completed.')

        if visualize:
            # self.draw_all_points(score=self.pc_detector.pcs_edge_scores)
            # self.draw_reflectance(show=True)
            self.draw_edge_points()
            # self.draw_edge_points(score=self.pc_detector.pcs_edge_scores[-1],
            #                       image=self.img_detector.img_edge_scores[-1])

        # Optimization parameters
        self.num_iterations = 0

    def select_correspondences(self):
        """Generate synthetic lidar image for each image-scan pair. Allow user
        to manually select correspondences.

        Grayscale image and Synthetic image are shown in alternating order.
        After selecting a correspondence, the selected pixel is highlighted
        blue in the image. When ready to move on, press "y" key to alternate
        to grayscale/synthetic. When finished selecting correspondences,
        press "q" key to quit and save.
        """
        self.correspondences = []

        curr_img_name = "gray"
        points_selected = 0
        ref_pt = np.asarray([0, 0], dtype=np.uint)
        ref_pt_3D = np.asarray([0, 0, 0], dtype=np.float32)

        pc_pixel_tree = None
        curr_pc_pixels = None
        curr_pc = None

        def correspondence_cb(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                if curr_img_name == "gray":
                    ref_pt[0], ref_pt[1] = x, y
                else:
                    d, i = pc_pixel_tree.query(
                        np.asarray([x, y]).reshape((1, 2)), 1)
                    nearest_pixel = curr_pc_pixels[i, :].astype(np.uint)[0]
                    ref_pt[0], ref_pt[1] = nearest_pixel[0], nearest_pixel[1]
                    ref_pt_3D[:] = curr_pc[i, :]

        for frame_idx in range(self.num_frames):
            curr_pc = self.pc_detector.pcs[frame_idx]
            curr_pc_pixels = self.projected_points[frame_idx]
            pc_pixel_tree = cKDTree(curr_pc_pixels)

            img_gray = cv.cvtColor(self.img_detector.imgs[frame_idx],
                                   cv.COLOR_BGR2GRAY)
            img_synthetic = gen_synthetic_image(
                curr_pc, self.pc_detector.reflectances[frame_idx], self.R,
                self.T, self.K,
                (self.img_detector.img_h, self.img_detector.img_w))

            cv.namedWindow("Correspondences")
            cv.setMouseCallback("Correspondences", correspondence_cb)

            gray_pixels = []
            lidar_pixels = []
            lidar_points = []
            while True:
                if points_selected % 2 == 0:
                    curr_img = img_gray
                    curr_img_name = "gray"
                else:
                    curr_img = img_synthetic
                    curr_img_name = "synthetic"

                curr_img = np.repeat(np.expand_dims(curr_img, axis=2),
                                     3,
                                     axis=2)
                curr_img = cv.circle(curr_img,
                                     tuple(ref_pt),
                                     radius=2,
                                     color=(255, 0, 0),
                                     thickness=-1)
                cv.imshow("Correspondences", curr_img)
                key = cv.waitKey(1)

                if key == ord('y'):
                    points_selected += 1
                    if curr_img_name == "synthetic":
                        lidar_pixels.append(ref_pt.copy())
                        lidar_points.append(ref_pt_3D.copy())
                    else:
                        gray_pixels.append(ref_pt.copy())

                elif key == ord('q'):
                    if points_selected % 2 == 0:
                        if points_selected / 2 < 6:
                            print(
                                f"Select at least 6 correspondences, currently {int(points_selected / 2)}"
                            )
                        else:
                            break
                    else:
                        print("Uneven number of points. Select one more.")
            cv.destroyAllWindows()

            gray_pixels = np.asarray(gray_pixels)
            lidar_pixels = np.asarray(lidar_pixels)
            lidar_points = np.asarray(lidar_points)

            if self.visualize:
                cv.imshow(
                    "Matches",
                    draw_point_matches(img_gray, gray_pixels, img_synthetic,
                                       lidar_pixels))
                cv.waitKey(0)
                cv.destroyAllWindows()

            self.correspondences.append((gray_pixels, lidar_points))

    def update_extrinsics(self, tau_new):
        """Given new extrinsic parameters, update the rotation matrix,
        translation vector, and tau.

        :param tau_new: New extrinsics vector.
        """
        R, T = self.tau_to_transform(tau_new)
        self.R, self.T = R, T
        self.tau = tau_new

    @staticmethod
    def transform_to_tau(R, T):
        """Given rotation and translation matrices/vectors, compute the
        extrinsics vector.

        The extrinsics vector, tau, has 6 parameters. The first three parameters
        are a rotation vector. The last three parameters are a translation
        vector.

        :param R: (3, 3) Rotation matrix.
        :param T: (3, 1) Translation vector.
        :return: (6, 1) numpy array.
        """
        r_vec, _ = cv2.Rodrigues(R)
        return np.hstack((r_vec.T, T.T)).reshape(6,)

    @staticmethod
    def tau_to_transform(tau):
        """Given the extrinsics vector, compute the rotation and translation
        matrices/vectors.

        The extrinsics vector, tau, has 6 parameters. The first three parameters
        are a rotation vector. The last three parameters are a translation
        vector.

        :param tau: (6, 1) numpy array.
        :return: [R, T], where R is (3, 3) numpy array, T is (3, 1) numpy array.
        """
        tau = np.squeeze(tau)
        R, _ = cv2.Rodrigues(tau[:3])
        T = tau[3:].reshape((3, 1))
        return R, T

    def project_point_cloud(self):
        """For each image-scan pair, project the pointcloud onto the image using
        current extrinsics. Compute a binary mask indicating the points that
        land within image boundaries.

        Transform all points of the point cloud into the camera frame and then
        projects all points to the image plane. Also store a binary mask to
        obtain all points with a valid projection within image boundaries.
        """
        # Compute R and T from current tau
        self.R, self.T = self.tau_to_transform(self.tau)

        # Remove previous projection
        self.points_cam_frame = []
        self.projected_points = []
        self.projection_mask = []

        for pc in self.pc_detector.pcs:
            one_mat = np.ones((pc.shape[0], 1))
            point_cloud = np.concatenate((pc, one_mat), axis=1)

            # Transform points into the camera frame
            self.points_cam_frame.append(
                np.matmul(np.hstack((self.R, self.T)), point_cloud.T).T)

            # Project points into image plane and normalize
            projected_points = np.dot(self.K, self.points_cam_frame[-1].T)
            projected_points = projected_points[::] / projected_points[::][-1]
            projected_points = np.delete(projected_points, 2, axis=0)
            self.projected_points.append(projected_points.T)

            # Remove points that were behind the camera
            in_front_of_camera_mask = self.points_cam_frame[-1][:, 2] > 0

            # Remove projected points that are outside of the image
            inside_mask_x = np.logical_and(
                (projected_points.T[:, 0] >= 0),
                (projected_points.T[:, 0] <= self.img_detector.img_w))

            inside_mask_y = np.logical_and(
                (projected_points.T[:, 1] >= 0),
                (projected_points.T[:, 1] <= self.img_detector.img_h))
            inside_mask = np.logical_and(inside_mask_x, inside_mask_y)

            # Final projection mask
            self.projection_mask.append(
                np.logical_and(inside_mask, in_front_of_camera_mask))

    def draw_all_points(self, score=None, img=None, frame=-1, show=False):
        """Draw projected points on the image provided.

        For the selected frame, project the points (N, 3) onto the image.
        Colorize the points using the provided score vector (N, 1).

        :param score: (N, 1) numpy array with scores used to colorize projected
                      points.
        :param img: (H, W) image used to draw the points over.
        :param frame: Integer index used to select the image-scan pair.
        :param show: Boolean to visualize the projected points or not.
        :return: (H, W, 3) image with the projected points.
        """

        if img is None:
            image = self.img_detector.imgs[frame].copy()
        else:
            image = img

        colors = self.scalar_to_color(score=score, frame=frame)
        colors_valid = colors[self.projection_mask[frame]]

        projected_points_valid = self.projected_points[frame][
            self.projection_mask[frame]]

        for pixel, color in zip(projected_points_valid, colors_valid):
            cv2.circle(image,
                       (pixel[0].astype(np.int), pixel[1].astype(np.int)), 1,
                       color.tolist(), -1)

        if show:
            cv.imshow('Projected Point Cloud on Image', image)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return image

    def draw_reflectance(self, frame=-1, show=False):
        """For selected image-scan pair, draw the reflectance image.

        Project the pointcloud onto the image using current extrinsics. Colorize
        projected points using the reflectance of points.

        :param frame: Integer index indicating which image-scan pair to use.
        :param show: Boolean to visualize or not.
        :return: (H, W) grayscale image with projected points.
        """
        img_h, img_w = self.img_detector.imgs[frame].shape[:2]
        refl_img = np.zeros((img_h, img_w), dtype=np.float32)

        projected_points_valid = self.projected_points[frame][
            self.projection_mask[frame]]
        reflectance_values = self.pc_detector.reflectances[frame][
            self.projection_mask[frame]]

        for pixel, reflectance in zip(projected_points_valid,
                                      reflectance_values):
            refl_img[pixel[1].astype(np.int),
                     pixel[0].astype(np.int)] = reflectance

        if show:
            cv.imshow('Projected Point Cloud Reflectance Image', refl_img)
            cv.imshow(
                'Grayscale img',
                cv.cvtColor(self.img_detector.imgs[frame], cv.COLOR_BGR2GRAY))
            cv.waitKey(0)
            cv.destroyAllWindows()

    def draw_edge_points_binary(self, frame=-1, show=False):
        """For selected image-scan pair, draw the detected edge points.

        Project the detected edge points in the pointcloud onto a blank image
        using current extrinsics. Edge points that land outside of image bounds
        are ignored.

        :param frame: Integer index indicating which image-scan pair to use.
        :param show: Boolean to visualize or not.
        :return: (H, W) binary image with projected points.
        """
        image = np.zeros((self.img_detector.img_h, self.img_detector.img_w),
                         dtype=np.bool)

        projected_points_valid = self.projected_points[frame][np.logical_and(
            self.projection_mask[frame],
            self.pc_detector.pcs_edge_masks[frame])]

        for pixel in projected_points_valid:
            image[pixel[1].astype(np.int), pixel[0].astype(np.int)] = True

        if show:
            cv.imshow('Projected Edge Points on Image', image)
            cv.waitKey(0)
            cv.destroyAllWindows()
        return image

    def draw_edge_points(self,
                         score=None,
                         image=None,
                         append_string='',
                         frame=-1,
                         save=False,
                         show=False):
        """
        Draw only edge points within corresponding camera's FoV on image provided.
        """

        if image is None:
            image = self.img_detector.imgs[frame].copy()
        else:
            image = (image.copy() * 255).astype(np.uint8)
            image = np.dstack((image, image, image))

        colors = self.scalar_to_color(frame=frame)
        colors_valid = colors[np.logical_and(
            self.projection_mask[frame],
            self.pc_detector.pcs_edge_masks[frame])]

        projected_points_valid = self.projected_points[frame][np.logical_and(
            self.projection_mask[frame],
            self.pc_detector.pcs_edge_masks[frame])]

        for pixel, color in zip(projected_points_valid, colors_valid):
            image[pixel[1].astype(np.int), pixel[0].astype(np.int), :] = color

        if save:
            now = datetime.now()
            cv.imwrite(
                append_string + now.strftime("%y%m%d-%H%M%S-%f") + '.jpg',
                image)

        if show:
            cv.imshow('Projected Edge Points on Image', image)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return image

    def scalar_to_color(self, score=None, min_d=0, max_d=60, frame=-1):
        """
        print Color(HSV's H value) corresponding to score
        """
        if score is None:
            score = np.sqrt(
                np.power(self.points_cam_frame[frame][:, 0], 2) +
                np.power(self.points_cam_frame[frame][:, 1], 2) +
                np.power(self.points_cam_frame[frame][:, 2], 2))

        np.clip(score, 0, max_d, out=score)
        # max distance is 120m but usually not usual

        norm = plt.Normalize()
        colors = plt.cm.jet(norm(score))

        return (colors[:, :3] * 255).astype(np.uint8)

    # def draw_points(self, image=None, FULL=True, frame=-1):
    #     """
    #     Draw points within corresponding camera's FoV on image provided.
    #     If no image provided, points are drawn on an empty(black) background.
    #     """

    #     if image is not None:
    #         if image.shape[-1] == 1:
    #             image = np.uint8(np.dstack((image, image, image))) * 255
    #         cv.imshow('Before projection', image)
    #         cv.waitKey(0)

    #         hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    #     else:
    #         hsv_image = np.zeros(self.img_detector.imgs.shape).astype(np.uint8)

    #     color = self.pc_to_colors(frame=-1)
    #     if FULL:
    #         index = range(self.projected_points[frame].shape[0])
    #     else:
    #         index = np.random.choice(self.projected_points[frame].shape[0],
    #                                  size=int(self.projected_points[frame].shape[0] /
    #                                           10),
    #                                  replace=False)
        
    #     for i in index:
    #         if pc[i, 1] > 0:
    #             continue
    #         if self.projection_mask[i] is False:
    #             continue

    #         cv.circle(hsv_image, (np.int32(self.projected_points[i, 0]),
    #                               np.int32(self.projected_points[i, 1])), 1,
    #                   (int(color[i]), 255, 255), -1)

    #     return cv.cvtColor(hsv_image, cv.COLOR_HSV2BGR)

    def pc_to_colors(self, min_d=0, max_d=120, frame=-1):
        """
        print Color(HSV's H value) corresponding to distance(m)
        close distance = red , far distance = blue
        """
        dist = np.sqrt(
            np.add(np.power(self.pc_detector.pcs[frame][:, 0], 2),
                   np.power(self.pc_detector.pcs[frame][:, 1], 2),
                   np.power(self.pc_detector.pcs[frame][:, 2], 2)))
        np.clip(dist, 0, max_d, out=dist)
        # max distance is 120m but usually not the case
        return (((dist - min_d) / (max_d - min_d)) * 120).astype(np.uint8)

    def compute_mi_cost(self, frame=-1):
        """For selected image-scan pair, compute mutual information cost.

        Using locations of projected lidar points that land within image bounds,
        generate vector of the grayscale intensities, and vector of reflection
        intensities. Model distribution of each using a Kernel Density Estimate
        (KDE). Compute the mutual information between the two random variables.

        :param frame: Integer index indicating the image-scan pair.
        :return: Negative of the Mutual information cost.
        """
        self.project_point_cloud()
        grayscale_img = cv.cvtColor(self.img_detector.imgs[frame],
                                    cv.COLOR_BGR2GRAY)
        projected_points_valid = self.projected_points[frame][self.projection_mask[frame], :]

        grayscale_vector = np.expand_dims(
            grayscale_img[projected_points_valid[:, 1].astype(np.uint),
                          projected_points_valid[:, 0].astype(np.uint)], 1)
        reflectance_vector = np.expand_dims(
            (self.pc_detector.reflectances[frame][self.projection_mask[frame]] *
             255.0), 1).astype(np.int)

        if len(reflectance_vector) > 0 and len(grayscale_vector) > 0:

            joint_data = np.hstack([grayscale_vector, reflectance_vector])
            intensity_vector = np.linspace(-1, 256, 510)
            grid_x, grid_y = np.meshgrid(intensity_vector, intensity_vector)
            grid_data = np.vstack([grid_y.ravel(), grid_x.ravel()])
            grid_data = grid_data.T

            gray_probs = FFTKDE(
                bw='silverman').fit(grayscale_vector).evaluate(intensity_vector)

            refl_probs = FFTKDE(bw='silverman').fit(
                reflectance_vector).evaluate(intensity_vector)
            joint_probs = FFTKDE().fit(joint_data).evaluate(grid_data)

            gray_probs /= np.sum(gray_probs)
            refl_probs /= np.sum(refl_probs)

            joint_probs /= np.sum(joint_probs)
            mi_cost = entropy(gray_probs) + \
                entropy(refl_probs) - entropy(joint_probs)
            mi_cost = mi_cost

        else:
            mi_cost = 0
        return -mi_cost

    def compute_chamfer_dists(self):
        """
        For each image-scan pair, compute the chamfer distance. Count the number
        of edge points used in the distance calculation.

        :return: Average distance between edge points in a Lidar image and
        camera image.
        """
        total_dist = 0
        total_edge_pts = 0

        for frame_idx in range(self.num_frames):
            cam_edges = self.img_detector.imgs_edges[frame_idx]
            cam_edges_inv = 255*np.logical_not(cam_edges).astype(np.uint8)
            cam_dist_map = cv.distanceTransform(cam_edges_inv, cv.DIST_L2,
                                                cv.DIST_MASK_PRECISE)
            lid_edges = self.draw_edge_points_binary(frame_idx)
            num_edge_pts = lid_edges.sum()
            dist = np.multiply(lid_edges, cam_dist_map).sum()

            total_dist += dist
            total_edge_pts += num_edge_pts

        return total_dist/total_edge_pts

    def compute_conv_cost(self, sigma_in, frame=-1, sigma_scaling=True):
        """For selected image-scan pair, compute GMM cost

        For each points within camera's FoV and above edge score
        threshold, retrieve image patch within 3-sigma, and compute
        point-wise cost based on Gaussian approximity.

        :param: sigma_in: base standard deviation for gaussian kernel
        :param: frame: Integer index indicating the image-scan pair
        :param: sigma_scaling: Boolean indicating whether to scale using the
                               distance of each point
        :return: Negative of the GMM cost
        """
        # start_t = time.time()
        cost_map = np.zeros(self.img_detector.img_edge_scores[frame].shape)
        for idx_pc in range(self.pc_detector.pcs_edge_idxs[frame].shape[0]):

            idx = self.pc_detector.pcs_edge_idxs[frame][idx_pc]

            # check if projected projected point lands within image bounds
            if not self.projection_mask[frame][idx]:
                continue

            if sigma_scaling:
                sigma = (
                    sigma_in /
                    np.linalg.norm(self.points_cam_frame[frame][idx, :], 2))
            else:
                sigma = sigma_in

            mu_x, mu_y = self.projected_points[frame][idx].astype(np.int)
            # Get gaussian kernel
            # Distance > 3 sigma is set to 0
            # and normalized so that the total Kernel = 1
            gauss2d = getGaussianKernel2D(sigma)
            top, bot, left, right = get_boundry(
                self.img_detector.img_edge_scores[frame], (mu_y, mu_x),
                int(sigma))
            # Get image patch inside the kernel
            edge_scores_patch = \
                self.img_detector.img_edge_scores[frame][
                    mu_y - top:mu_y + bot,
                    mu_x - left:mu_x + right
                ].copy()

            # weight = (normalized img score + normalized pc score) / 2
            # weight = weight / |Omega_i|
            # Cost = Weight * Gaussian Kernel
            num_nonzeros = np.sum(edge_scores_patch != 0)
            if num_nonzeros == 0:
                continue

            edge_scores_patch[edge_scores_patch != 0] += \
                self.pc_detector.pcs_edge_scores[frame][idx]

            kernel_patch = gauss2d[3 * int(sigma) - top:3 * int(sigma) + bot,
                                   3 * int(sigma) - left:3 * int(sigma) + right]

            cost_patch = np.multiply(edge_scores_patch, kernel_patch)

            # Normalize by number of edge projected_points in the neighborhood
            cost_map[mu_y, mu_x] = \
                np.sum(cost_patch) / (2 * np.sum(edge_scores_patch > 0))

        # plot_2d(cost_map)
        gc.collect()
        return -np.sum(cost_map)

    def compute_corresp_cost(self, norm_thresh=5):
        """Compute re-projection error for all correspondences.

        For each of the 2D-3D correspondences, project the 3D points onto the
        image and compute the L2-distance to the corresponding 2D pixel.

        :norm_thresh: Floating point threshold. If the absolute distance between
                      a correspondence is above this, the distance is squared.
                      Else the absolute distance is directly taken. Distances
                      are summed and averaged. Offset value is used to shift
                      the minimum to a strong negative value.

        :return: Floating point, average distance penalty across all
                 correspondences.
                 Array of distance penalty for each correspondence
        """
        """Return average distance between all correspondences"""
        pixel_distances = []
        num_corresp = 0
        dist_offset = np.sqrt(self.img_detector.img_w**2 +
                              self.img_detector.img_h**2) * 3

        for matches in self.correspondences:
            gray_pixels = matches[0]
            lidar_points = matches[1]
            print(lidar_points)
            lidar_points_cam = np.matmul(self.R, lidar_points.T) + self.T
            lidar_pixels_homo = (np.matmul(self.K, lidar_points_cam).T)
            lidar_pixels_homo = lidar_pixels_homo / \
                np.expand_dims(lidar_pixels_homo[:, 2], axis=1)
            lidar_pixels = lidar_pixels_homo[:, :2]

            pixel_diff = gray_pixels - lidar_pixels
            pixel_distances += np.linalg.norm(pixel_diff, axis=1,
                                              ord=1).tolist()
            num_corresp += lidar_pixels.shape[0]

        total_dist = 0
        for dist in pixel_distances:
            if dist <= norm_thresh:
                total_dist += dist
            else:
                total_dist += (dist**2)
        average_dist = total_dist / num_corresp
        return -dist_offset + 3 * average_dist, pixel_distances

    def compute_points_cost(self, frame=-1):
        """Compute the change in the number of points compared to at the start
        of optimization. Return the absolute difference.

        :param frame: Integer index indicating which image-scan pair to use.
        :return: Absolute integer difference in number of projected points.
        """
        num_points = self.projection_mask[frame].sum()
        points_diff = abs(self.numpoints_preopt[frame] - num_points)
        return points_diff

    def batch_optimization(self):
        """Optimize over extrinsics and return the optimized parameters
           over least square cost.

        :return: [tau, cost]. tau is a (6, 1) optimized extrinsics vector.
                 cost is a list with the history of loss over the optimization.
        """
        
        def loss_manual(tau):
            self.tau = tau
            self.R, self.T = self.tau_to_transform(tau)
            self.project_point_cloud()
            return self.compute_corresp_cost()[1]

        tau = self.tau.copy()
        print('Start optimization using manually selected correspondances')
        opt_results = least_squares(loss_manual, tau, method='lm')
 
        self.tau = opt_results.x
        if self.visualize:
            img = self.draw_all_points(frame=0)
            cv.imshow('Projection with optimized tau', img)
            cv.waitKey(2000)
            cv.imwrite('generated/optimized.jpg', img)

        print('\n### Optimization completed ###')
        print('xyz (m): ' + str(np.squeeze(self.tau)[:3]))
        print('Rotation Vector: ' + str(np.squeeze(self.tau)[3:]))
        return opt_results.x

    def ls_optimize(self, hyperparams, maxiter=600):
        """Optimize over extrinsics and return the optimized parameters.

        :param hyperparams: Dictionary with scaling coefficients for each cost
                            component, sigma value for GMM, and scaling array
                            for the extrinsics vector during optimization.
        :param maxiter: Integer limit for number of optimizer iterations.
        :return: [tau, cost]. tau is a (6, 1) optimized extrinsics vector.
                 cost is a list with the history of loss over the optimization.
        """
        print("Executing optimization")
        cost_history = []

        # Store initial number of points for points-based cost
        self.numpoints_preopt = [
            np.sum(self.projection_mask[i]) for i in range(self.num_frames)
        ]

        # Generate initial simplex for Nelder-Mead
        initial_deltas = [0.10, 0.10, 0.10, 0.5, 0.5, 0.5]
        opt_options = {
            'disp':
                True,
            'maxiter':
                maxiter,
            'adaptive':
                True,
            'initial_simplex':
                get_mixed_delta_simplex(self.tau,
                                        initial_deltas,
                                        scales=hyperparams['scales'])
        }
        self.num_iterations = 0

        def loss_callback(xk, state=None):
            """Save loss graph and point-cloud projection for debugging"""
            self.num_iterations += 1
            # if len(cost_history):
            #     plt.close('all')
            #     plt.figure()
            #     plt.plot(cost_history)
            #     plt.savefig('current_loss.png')

            img = self.draw_all_points()
            cv.imwrite('current_projection.jpg', img)
            return False

        update_deltas = [0.10, 0.10, 0.10, 0.5, 0.5, 0.5]

        def bh_callback(x, f, accepted):
            if accepted:
                print("New minimum discovered by Basin-Hopping.")
                opt_options['initial_simplex'] = \
                    get_mixed_delta_simplex(np.multiply(x,
                                                        hyperparams['scales']),
                                            update_deltas,
                                            scales=hyperparams['scales'])
            return False

        # Compute cost
        tau_scaled = np.divide(self.tau, hyperparams['scales'])

        # Basin-Hopping
        step_vector = [0.10, 0.10, 0.10, 1.0, 1.0, 1.0]
        opt_results = basinhopping(
            loss,
            tau_scaled,
            10,
            T=1.5,
            niter_success=3,
            disp=True,
            callback=bh_callback,
            take_step=RandomDisplacement(step_vector),
            minimizer_kwargs={
                'method': 'Nelder-Mead',
                'args': (self, hyperparams, cost_history),
                'callback': loss_callback,
                'options': opt_options
            })
        self.tau = np.multiply(opt_results.lowest_optimization_result.x,
                               hyperparams['scales'])

        print('\n### Optimization completed ###')
        print('xyz (m): ' + str(np.squeeze(self.tau)[:3]))
        print('Rotation Vector: ' + str(np.squeeze(self.tau)[3:]))
        return self.tau, cost_history


def loss(tau_scaled, calibrator, hyperparams, cost_history):
    """For each image-scan pair, project the pointcloud with current extrinsics,
    compute the loss components (mutual information, GMM, correspondence, and
    points). Scale the components and return the sum.

    Hyperparams dict must have the keys ['alphas', 'scales'], where
    hyperparams['alphas'] contains the coefficients for each component
    ['mi', 'gmm', 'points', 'corr', 'sigma'],

    Hyperparams['scales'] describes how each parameter axis is
    normalized in the optimization.

    :param tau_scaled: (6, 1) extrinsics parameters.
    :param calibrator: CameraLidarCalibrator object containing loaded image-scan
                       pairs and the detected edge pixels and points.
    :param hyperparams: Dictionary with scaling coefficients and extrinsics
                        scaling vector.
    :param cost_history: List with current history of the loss over
                         the optimization.
    :return: Combined loss with each component scaled using the coefficients
             in hyperparams dictionary.
    """

    # Rescale Extrinsics
    tau = np.multiply(tau_scaled, hyperparams['scales'])
    calibrator.update_extrinsics(tau)
    calibrator.project_point_cloud()

    # Compute loss components
    cost_components = np.zeros((4, 1))
    for frame_idx in range(calibrator.num_frames):
        if hyperparams['alphas']['mi']:
            cost_components[0] += calibrator.compute_mi_cost(frame_idx)

        # if hyperparams['alphas']['gmm']:
        #     cost_components[1] += calibrator.compute_conv_cost(
        #         hyperparams['alphas']['sigma'], frame_idx, sigma_scaling=False)

        if hyperparams['alphas']['points']:
            cost_components[2] += calibrator.compute_points_cost(frame_idx)

    if hyperparams['alphas']['corr']:
        cost_components[3] += calibrator.compute_corresp_cost()

    if hyperparams['alphas']['gmm']:
        cost_components[1] += calibrator.compute_chamfer_dists()

    # Scale loss components
    cost_components[0] *= hyperparams['alphas']['mi']
    cost_components[1] *= hyperparams['alphas']['gmm']
    cost_components[2] *= hyperparams['alphas']['points']
    cost_components[3] *= hyperparams['alphas']['corr']

    total_cost = sum(cost_components)
    cost_history.append(total_cost)
    return sum(cost_components)


class RandomDisplacement(object):
    """
    Add a random displacement of maximum size `stepsize` to each coordinate
    Calling this updates `x` in-place.
    Parameters
    ----------
    stepsize : float, optional
        Maximum stepsize in any dimension
    random_gen : {None, `np.random.RandomState`, `np.random.Generator`}
        The random number generator that generates the displacements
    """

    def __init__(self, stepsize, random_gen=None):
        self.stepsize = stepsize
        self.random_gen = check_random_state(random_gen)

    def __call__(self, x):
        for i in range(len(x)):
            x[i] *= (
                1 +
                self.random_gen.uniform(-self.stepsize[i], self.stepsize[i]))
        return x
