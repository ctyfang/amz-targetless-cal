from calibration.camera_lidar_calibrator import *
from calibration.utils.config import command_line_parser
from calibration.utils.data_utils import *
from calibration.utils.img_utils import *
from calibration.utils.exp_utils import *

import pickle
import json
from prettytable import PrettyTable


def outliers_iqr(ys):
    quartile_1, quartile_3 = np.percentile(ys, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    # return np.where((ys < upper_bound) & (ys > lower_bound))
    return (ys < upper_bound) & (ys > lower_bound)

def expe_plot(tau_data, tau_init, tau_gt, filename, alpha=0.5, outlier=False):
    plt.rcParams.update({'font.size': 18})
    fig = plt.figure()
    plt.clf()
    if not outlier:
        ax = plt.subplot(1, 5, 1)
        plt.title('X')
        ax.scatter(range(tau_data.shape[0]),
                tau_data[:, 3] - tau_gt[3],
                c='b',
                alpha=alpha)
        ax.scatter(range(tau_init.shape[0]),
                tau_init[:, 3] - tau_gt[3],
                c='r',
                alpha=alpha)

        ax.set_xticks([])
        ax.set_ylim(-0.5, 0.5)

        plt.subplot(1, 5, 2, sharey=ax)
        plt.title('Y')
        plt.scatter(range(tau_data.shape[0]),
                    tau_data[:, 4] - tau_gt[4],
                    c='b',
                    alpha=alpha)
        plt.scatter(range(tau_init.shape[0]),
                    tau_init[:, 4] - tau_gt[4],
                    c='r',
                    alpha=alpha)

        axes = plt.gca()
        axes.set_xticks([])
        for label in axes.get_yticklabels():
            label.set_visible(False)

        plt.subplot(1, 5, 3, sharey=ax)
        plt.title('Z')
        plt.scatter(np.array(range(tau_data.shape[0])),
                    tau_data[:, 5] - tau_gt[5],
                    c='b',
                    alpha=alpha)

        plt.scatter(range(tau_init.shape[0]),
                    tau_init[:, 5] - tau_gt[5],
                    c='r',
                    alpha=alpha)
        axes = plt.gca()
        axes.set_xticks([])
        for label in axes.get_yticklabels():
            label.set_visible(False)

        plt.subplot(1, 5, 4, sharey=ax)
        plt.title('Angle')
        angle_vec = np.linalg.norm(tau_data[:, :3], axis=1, ord=2)
        angle_gt = np.linalg.norm(tau_gt[:3], ord=2)
        angle_vec_init = np.linalg.norm(tau_init[:, :3], axis=1, ord=2)

        plt.scatter(range(tau_init.shape[0]),
                    angle_vec_init - angle_gt,
                    c='r',
                    alpha=alpha)
        plt.scatter(range(tau_data.shape[0]),
                    angle_vec - angle_gt,
                    c='b',
                    alpha=alpha)
        axes = plt.gca()
        axes.set_xticks([])
        for label in axes.get_yticklabels():
            label.set_visible(False)
        xlims = axes.get_xlim()

        plt.subplot(1, 5, 5)
        plt.title('Axis')
        axis_vec = tau_data[:, :3] / np.expand_dims(
            np.linalg.norm(tau_data[:, :3], axis=1, ord=2), 1)
        axis_gt = tau_gt[:3] / np.linalg.norm(tau_gt[:3], ord=2)
        axis_vec_init = tau_init[:, :3] / np.expand_dims(
            np.linalg.norm(tau_init[:, :3], axis=1, ord=2), 1)
        axis_diffs = np.linalg.norm(axis_vec - axis_gt, axis=1, ord=2)
        axis_inits = np.linalg.norm(axis_vec_init - axis_gt, axis=1, ord=2)
        plt.scatter(range(axis_diffs.shape[0]), axis_diffs, c='b', alpha=alpha)
        plt.scatter(range(axis_diffs.shape[0]), axis_inits, c='r', alpha=alpha)
        axes = plt.gca()
        axes.set_xticks([])
        axes.yaxis.tick_right()
        xlims = axes.get_xlim()
        plt.tight_layout()
        plt.savefig(filename + '_all')
        plt.pause(1)
    else:
        print('---- Number of outliers ---')
        ax = plt.subplot(1, 5, 1)
        plt.title('X')
        err = tau_data[:, 3] - tau_gt[3]
        inliner = outliers_iqr(err)
        idx_in = np.where(inliner)[0]
        idx_out = np.where(~inliner)[0]
        print(f'X: {len(idx_out)}')

        plt.scatter(np.array(range(tau_data.shape[0]))[idx_in],
                    err[idx_in],
                    c='b',
                    alpha=alpha)
        plt.scatter(np.array(range(tau_data.shape[0]))[idx_out],
                    err[idx_out],
                    c='b',
                    marker='x',
                    alpha=alpha)
        plt.scatter(range(tau_init.shape[0]),
                    tau_init[:, 3] - tau_gt[3],
                    c='r',
                    alpha=alpha)

        ax.set_xticks([])
        ax.set_ylim(-0.5, 0.5)

        plt.subplot(1, 5, 2, sharey=ax)
        plt.title('Y')
        err = tau_data[:, 4] - tau_gt[4]
        inliner = outliers_iqr(err)
        idx_in = np.where(inliner)[0]
        idx_out = np.where(~inliner)[0]
        print(f'Y: {len(idx_out)}')

        plt.scatter(np.array(range(tau_data.shape[0]))[idx_in],
                    err[idx_in],
                    c='b',
                    alpha=alpha)
        plt.scatter(np.array(range(tau_data.shape[0]))[idx_out],
                    err[idx_out],
                    c='b',
                    marker='x',
                    alpha=alpha)
        plt.scatter(range(tau_init.shape[0]),
                    tau_init[:, 4] - tau_gt[4],
                    c='r',
                    alpha=alpha)

        axes = plt.gca()
        axes.set_xticks([])
        for label in axes.get_yticklabels():
            label.set_visible(False)

        plt.subplot(1, 5, 3, sharey=ax)
        plt.title('Z')
        err = tau_data[:, 5] - tau_gt[5]
        inliner = outliers_iqr(err)
        idx_in = np.where(inliner)[0]
        idx_out = np.where(~inliner)[0]
        print(f'Z: {len(idx_out)}')

        plt.scatter(np.array(range(tau_data.shape[0]))[idx_in],
                    err[idx_in],
                    c='b',
                    alpha=alpha)
        plt.scatter(np.array(range(tau_data.shape[0]))[idx_out],
                    err[idx_out],
                    c='b',
                    marker='x',
                    alpha=alpha)
        plt.scatter(range(tau_init.shape[0]),
                    tau_init[:, 5] - tau_gt[5],
                    c='r',
                    alpha=alpha)
        axes = plt.gca()
        axes.set_xticks([])
        for label in axes.get_yticklabels():
            label.set_visible(False)

        plt.subplot(1, 5, 4, sharey=ax)
        plt.title('Angle')
        angle_vec = np.linalg.norm(tau_data[:, :3], axis=1, ord=2)
        angle_gt = np.linalg.norm(tau_gt[:3], ord=2)
        angle_vec_init = np.linalg.norm(tau_init[:, :3], axis=1, ord=2)

        err = angle_vec - angle_gt
        inliner = outliers_iqr(err)
        idx_in = np.where(inliner)[0]
        idx_out = np.where(~inliner)[0]
        print(f'Angle: {len(idx_out)}')
        plt.scatter(range(tau_data.shape[0]),
                    angle_vec_init - angle_gt,
                    c='r',
                    alpha=alpha)
        plt.scatter(np.array(range(tau_data.shape[0]))[idx_in],
                    err[idx_in],
                    c='b',
                    alpha=alpha)
        plt.scatter(np.array(range(tau_data.shape[0]))[idx_out],
                    err[idx_out],
                    c='b',
                    marker='x',
                    alpha=alpha)

        axes = plt.gca()
        axes.set_xticks([])
        for label in axes.get_yticklabels():
            label.set_visible(False)
        xlims = axes.get_xlim()

        plt.subplot(1, 5, 5)
        plt.title('Axis')
        axis_vec = tau_data[:, :3] / np.expand_dims(
            np.linalg.norm(tau_data[:, :3], axis=1, ord=2), 1)
        axis_gt = tau_gt[:3] / np.linalg.norm(tau_gt[:3], ord=2)
        axis_vec_init = tau_init[:, :3] / np.expand_dims(
            np.linalg.norm(tau_init[:, :3], axis=1, ord=2), 1)
        axis_diffs = np.linalg.norm(axis_vec - axis_gt, axis=1, ord=2)
        axis_inits = np.linalg.norm(axis_vec_init - axis_gt, axis=1, ord=2)

        inliner = outliers_iqr(axis_diffs)
        idx_in = np.where(inliner)[0]
        idx_out = np.where(~inliner)[0]
        print(f'Axis: {len(idx_out)}')
        plt.scatter(np.array(range(axis_diffs.shape[0]))[idx_in],
                    axis_diffs[idx_in], c='b',
                    alpha=alpha)
        plt.scatter(np.array(range(axis_diffs.shape[0]))[idx_out],
                    axis_diffs[idx_out], c='b', marker='x',
                    alpha=alpha)
        plt.scatter(range(axis_diffs.shape[0]), axis_inits, c='r', alpha=alpha)
        axes = plt.gca()
        axes.set_xticks([])
        axes.yaxis.tick_right()
        xlims = axes.get_xlim()
        plt.tight_layout()
        plt.savefig(filename + '_out')
        plt.pause(1)

    table = PrettyTable()
    table.field_names = [
        "X_mean", "X_std", "Y_mean", "Y_std", "Z_mean", "Z_std", "angle_mean",
        "angle_std", "axis_mean", "axis_std"
    ]
    errors = [
        tau_data[:, 3] - tau_gt[3],
        tau_data[:, 4] - tau_gt[4],
        tau_data[:, 5] - tau_gt[5],
        angle_vec - angle_gt,
        axis_diffs]

    row_all = []
    row_inl = []
    for param_idx, param_name in enumerate(['X', 'Y', 'Z', 'angle',
                                            'axis']):
        inliner = outliers_iqr(errors[param_idx])
        if param_idx != 3:
            row_all += [f'{np.mean(errors[param_idx]):.5f}',
                        f'{np.std(errors[param_idx]):.5f}']
            row_inl += [f'{np.mean(errors[param_idx][inliner]):.5f}',
                        f'{np.std(errors[param_idx][inliner]):.5f}']
        else:
            row_all += [f'{np.rad2deg(np.mean(errors[param_idx])):.5f}',
                        f'{np.rad2deg(np.std(errors[param_idx])):.5f}']
            row_inl += [f'{np.rad2deg(np.mean(errors[param_idx][inliner])):.5f}',
                        f'{np.rad2deg(np.std(errors[param_idx][inliner])):.5f}']
    table.add_row(row_all)
    table.add_row(row_inl)

    table_data = table.get_string()
    with open('{}.txt'.format(filename), 'w') as table_file:
        table_file.write(table_data)

if __name__ == "__main__":
    """Script parameters"""
    save_every = 1
    select_new_correspondences = False
    calibrator_path = 'generated/calibrators/0928-6frames.pkl'
    # LOG_DIR = 'generated/optimizer_logs/0928-6frames-mi+gmm'
    # exp_name = 'exp_gmm_mi'
    LOG_DIR = 'generated/optimizer_logs/0928-6frames-mi+chamfer'
    exp_name = 'exp_mi_chamfer'
    # LOG_DIR = 'generated/optimizer_logs/gmm_only'
    # exp_name = 'exp_gmm'
    # LOG_DIR = 'generated/optimizer_logs/0928-6frames-chamfer'
    # exp_name = 'exp_chamfer'
    # LOG_DIR = 'generated/optimizer_logs/lm'
    # exp_name = 'exp_manual'
    """Experiment parameters"""
    exp_params = {
        'NUM_SAMPLES': 103,
        'TRANS_ERR_SIGMA': 0.10,
        'ANGLE_ERR_SIGMA': 5,
        'ALPHA_MI': [0],
        'ALPHA_GMM': [0],
        'ALPHA_POINTS': [0],
        'ALPHA_CORR': [1],
        'SIGMAS': [7.0],
        'MAX_ITERS': 100,
        'SCALES': np.power(10 * np.ones((1, 6)),
                           [0, 0, 0, -2, -2, -2]).tolist()
    }
    """Default Hyperparameter Template"""
    hyperparams = {
        'alphas': {
            'mi': 0,
            'gmm': 0,
            'points': 0,
            'corr': 1
        },
        'scales': exp_params['SCALES']
    }
    """Calibration directory specification"""
    calib_dir_list = [
        '/media/carter/Samsung_T5/3dv/2011_09_28/calibration',
        '/home/benjin/Development/Data/2011_09_26_calib/2011_09_26', 'data',
        '/home/carter/pycharm_project_534/data/calibration',
        './data/calibration'
    ]

    calib_dir = getPath(calib_dir_list)
    """Load pre-made calibrator object"""
    if os.path.exists(calibrator_path):
        with open(calibrator_path, 'rb') as input_pkl:
            calibrator = pickle.load(input_pkl)
            calibrator.visualize = True
    else:
        print('Calibrator does not exist at specified path!')
        exit()

    """Load ground-truth extrinsics"""
    R, T = load_lid_cal(calib_dir)
    tau_gt = calibrator.tau = calibrator.transform_to_tau(R, T)
    exp_params['tau_gt'] = tau_gt.tolist()

    if len(sys.argv) > 2:
        print('unknown usage')
        print('Usage: python3 experiment_hybrid_mi_gmm_loss.py [--plot]')
        exit(1)

    if len(sys.argv) == 1:
        """Select correspondences and save calibrator"""
        if select_new_correspondences:
            calibrator.select_correspondences()
            # with open(calibrator_path, 'wb') as overwrite_pkl:
            #     pickle.dump(calibrator, overwrite_pkl)
        """Create log directory and save GT projections"""
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(os.path.join(LOG_DIR, 'params.json'), 'w') as json_file:
            json.dump(exp_params, json_file, indent=4)

        os.makedirs(os.path.join(LOG_DIR, 'gt'), exist_ok=True)
        calibrator.project_point_cloud()
        for frame_idx in range(len(calibrator.img_detector.imgs)):
            img_all_points = calibrator.draw_all_points(frame=frame_idx)
            img_edges = calibrator.draw_edge_points(
                frame=frame_idx,
                image=calibrator.img_detector.imgs_edges[frame_idx])
            cv.imwrite(
                os.path.join(LOG_DIR, 'gt',
                             f'all_points_frame_{frame_idx}.jpg'),
                img_all_points)
            cv.imwrite(
                os.path.join(LOG_DIR, 'gt',
                             f'edge_points_frame_{frame_idx}.jpg'), img_edges)
        """Run the experiment"""
        if os.path.exists(os.path.join(LOG_DIR, 'tau_inits.npy')):
            with open(os.path.join(LOG_DIR, 'tau_inits.npy'), 'rb') as f:
                tau_inits = np.load(f)
        else:
            print('New experiments')
            tau_inits = np.array([])

        if os.path.exists(os.path.join(LOG_DIR, 'tau_data.npy')):
            with open(os.path.join(LOG_DIR, 'tau_data.npy'), 'rb') as f:
                tau_data = np.load(f)
        else:
            print('New experiments')
            tau_data = np.array([])

        for sample_idx in range(len(tau_data), exp_params['NUM_SAMPLES']):
            tau_inits = tau_inits.tolist()
            tau_data = tau_data.tolist()

            print(f'----- SAMPLE {sample_idx} -----')
            calibrator.update_extrinsics(
                perturb_tau(tau_gt,
                            trans_std=exp_params['TRANS_ERR_SIGMA'],
                            angle_std=exp_params['ANGLE_ERR_SIGMA']))
            compare_taus(tau_gt, calibrator.tau)

            calibrator.project_point_cloud()
            tau_inits.append(calibrator.tau)
            """Make directory for this trial"""
            os.makedirs(os.path.join(LOG_DIR, f'trial_{sample_idx}'),
                        exist_ok=True)
            os.makedirs(os.path.join(LOG_DIR, f'trial_{sample_idx}', 'initial'),
                        exist_ok=True)
            """Save initial images"""
            for frame_idx in range(len(calibrator.img_detector.imgs)):
                img_all_points = calibrator.draw_all_points(frame=frame_idx)
                img_edges = calibrator.draw_edge_points(
                    frame=frame_idx,
                    image=calibrator.img_detector.imgs_edges[frame_idx])
                cv.imwrite(
                    os.path.join(
                        LOG_DIR, f'trial_{sample_idx}', 'initial',
                        f'all_points_{sample_idx}_frame_{frame_idx}.jpg'),
                    img_all_points)
                cv.imwrite(
                    os.path.join(
                        LOG_DIR, f'trial_{sample_idx}', 'initial',
                        f'edge_points_{sample_idx}_frame_{frame_idx}.jpg'),
                    img_edges)
            """Run optimizer"""
            for stage_idx in range(len(exp_params["SIGMAS"])):
                sigma_in, alpha_gmm, alpha_mi, alpha_points, alpha_corr = \
                    exp_params['SIGMAS'][stage_idx], \
                    exp_params['ALPHA_GMM'][stage_idx], \
                    exp_params['ALPHA_MI'][stage_idx], \
                    exp_params['ALPHA_POINTS'][stage_idx], \
                    exp_params['ALPHA_CORR'][stage_idx]

                curr_hyperparams = hyperparams.copy()
                curr_hyperparams['alphas']['mi'] = alpha_mi
                curr_hyperparams['alphas']['gmm'] = alpha_gmm
                curr_hyperparams['alphas']['points'] = alpha_points
                curr_hyperparams['alphas']['corr'] = alpha_corr
                curr_hyperparams['alphas']['sigma'] = sigma_in

                # tau_opt, cost_history = calibrator.ls_optimize(
                #     hyperparams,
                #     maxiter=exp_params['MAX_ITERS'],
                #     save_every=save_every)
                tau_opt, cost_history = calibrator.batch_optimization(hyperparams)
                calibrator.tau = tau_opt
                compare_taus(tau_gt, tau_opt)
                calibrator.project_point_cloud()
            """Save results from this optimization stage"""
            tau_data.append(np.squeeze(tau_opt))
            # """Save projections with final optimized tau"""
            # os.makedirs(os.path.join(LOG_DIR, f'trial_{sample_idx}',
            #                          f'stage_{stage_idx}'),
            #             exist_ok=True)
            # plt.figure()
            # plt.title('Loss history')
            # plt.plot(range(len(cost_history)), cost_history)
            # plt.savefig(
            #     os.path.join(LOG_DIR, f'trial_{sample_idx}',
            #                  f'stage_{stage_idx}', f'loss_history.png'))
            # # plt.pause(2)
            # plt.close()

            calibrator.project_point_cloud()
            for frame_idx in range(len(calibrator.img_detector.imgs)):
                os.makedirs(os.path.join(LOG_DIR, f'trial_{sample_idx}',
                                         f'stage_{stage_idx}'),
                            exist_ok=True)
                img_all_points = calibrator.draw_all_points(frame=frame_idx)
                img_edges = calibrator.draw_edge_points(
                    frame=frame_idx,
                    image=calibrator.img_detector.imgs_edges[frame_idx])
                cv.imwrite(
                    os.path.join(LOG_DIR, f'trial_{sample_idx}',
                                 f'stage_{stage_idx}',
                                 f'all_points_frame_{frame_idx}.jpg'),
                    img_all_points)
                cv.imwrite(
                    os.path.join(LOG_DIR, f'trial_{sample_idx}',
                                 f'stage_{stage_idx}',
                                 f'edge_points_frame_{frame_idx}.jpg'),
                    img_edges)
            """Save initial taus, optimized taus, tau_scale"""
            tau_inits = np.asarray(tau_inits)
            np.save(os.path.join(LOG_DIR, 'tau_inits'), tau_inits)

            tau_data = np.asarray(tau_data)
            np.save(os.path.join(LOG_DIR, 'tau_data'), tau_data)
        exit()

    if sys.argv[1] == '--plot':
        with open(os.path.join(LOG_DIR, 'tau_data.npy'), 'rb') as f:
            tau_data = np.load(f)

        with open(os.path.join(LOG_DIR, 'tau_inits.npy'), 'rb') as f:
            tau_inits = np.load(f)
        print(tau_data.shape[0])
        expe_plot(tau_data, tau_inits, tau_gt,
                  os.path.join(LOG_DIR, exp_name))
        expe_plot(tau_data, tau_inits, tau_gt,
                  os.path.join(LOG_DIR, exp_name),
                  outlier=True)

        sys.exit(0)
    print('unknown usage')
    print('Usage: python3 {} [--plot]'.format(sys.argv[0]))
    sys.exit(1)
