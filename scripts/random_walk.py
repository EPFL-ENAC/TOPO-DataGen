import argparse
import json
import os
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import random

from itertools import product, combinations
from pyDOE import *

import scipy
from scipy import signal

from reframeTransform import ReframeTransform
import rasterio
import pyproj

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJ_DIR = os.path.abspath(os.path.join(__file__, '../..'))


def get_init_points(range_x: list, range_y: list, range_z: list, num_pt_per_surf: int = 3) -> np.ndarray:
    """
    Get data points of LHS-sampled initial positions (XYZ) within a hypothetical cube.
    A local Cartesian coordinate system is used.
    Args:
        range_x:        range for local X.
        range_y:        range for local Y.
        range_z:        range for local Z.
        num_pt_per_surf: number of points per surface, 5 out of 6 surfaces except the bottom are used.
    Returns:
        a numpy array for initial positions [N, 2].
    """
    dim = 2  # initial points at the cube surface, thus having 2 DOFs
    x_min, x_max = range_x
    y_min, y_max = range_y
    z_min, z_max = range_z

    pt_oxy = lhs(n=dim, samples=num_pt_per_surf)  # [samples, n], n -> dim
    pt_oxy = pt_oxy * np.asarray([(x_max - x_min), (y_max - y_min)]) + np.asarray([x_min, y_min])
    pt_oxy = np.insert(pt_oxy, 2, z_max, axis=1)

    pt_oxz = lhs(n=dim, samples=num_pt_per_surf)  # [samples, n], n -> dim
    pt_oxz = pt_oxz * np.asarray([(x_max - x_min), (z_max - z_min)]) + np.asarray([x_min, z_min])
    pt_oxz = np.concatenate((pt_oxz, pt_oxz), axis=0)
    pt_oxz = np.insert(pt_oxz, 1, np.concatenate((np.ones(num_pt_per_surf) * y_min, np.ones(num_pt_per_surf) * y_max),
                                                 axis=0), axis=1)

    pt_oyz = lhs(n=dim, samples=num_pt_per_surf)  # [samples, n], n -> dim
    pt_oyz = pt_oyz * np.asarray([(y_max - y_min), (z_max - z_min)]) + np.asarray([y_min, z_min])
    pt_oyz = np.concatenate((pt_oyz, pt_oyz), axis=0)
    pt_oyz = np.insert(pt_oyz, 0, np.concatenate((np.ones(num_pt_per_surf) * x_min, np.ones(num_pt_per_surf) * x_max),
                                                 axis=0), axis=1)

    pt_init = np.concatenate((pt_oxy, pt_oxz, pt_oyz), axis=0)
    pt_init = np.unique(pt_init, axis=0)  # [N, 2]
    return pt_init


def random_walk(init_pts, anchor_pts=None, num_steps=10, sigmas=None, lower_bound=None, upper_bound=None,
                rate_bound=None, trial_per_loop=100, include_angle=False) -> list:
    """
    Random walk with constraints.
    Args:
        init_pts        Initial points, shape: [num_sample, dim_per_pt], determines the number of trajectories.
        anchor_pts      Anchor points, which exerts some sort of attractive force.
        num_steps       Length of each trajectory.
        sigmas          Standard deviation of random walk Gaussian noise, length = dim_per_pt.
        lower_bound     Lower bound for data points, length = dim_per_pt.
        upper_bound     Upper bound for data points, length = dim_per_pt.
        rate_bound      Upper bound for change rate between two consecutive data points, length = dim_per_pt.
        trial_per_loop  Maximum number of iterations to search for next feasible data point.
    Returns:
        python list for trajectories.
    """
    # dimension assertion
    init_pts = np.asarray(init_pts)
    assert len(init_pts.shape) == 2, 'Initial position is a {:d}-d array, not 2-d!'.format(len(init_pts.shape))

    # prepare constraints
    n_samples, dims = init_pts.shape[0], init_pts.shape[1]
    if include_angle:
        assert dims == 6
    else:
        assert dims == 3
    if anchor_pts is None:
        anchor_pts = np.zeros(dims).reshape(1, -1)
    else:
        anchor_pts = np.asarray(anchor_pts)
        assert dims == anchor_pts.shape[1], 'Anchor point dim. is {:d}, not equal to that of init. pts.: {:d}!'.format(
            anchor_pts.shape[1], dims)
    if lower_bound is None:
        lower_bound = np.ones(dims) * -np.Inf
    else:
        lower_bound = np.asarray(lower_bound)
    if upper_bound is None:
        upper_bound = np.ones(dims) * np.Inf
    else:
        upper_bound = np.asarray(upper_bound)
    if rate_bound is None:
        rate_bound = np.ones(dims) * np.Inf
    else:
        rate_bound = np.asarray(rate_bound)
    if sigmas is None:
        sigmas = np.ones(dims)
    else:
        assert len(sigmas) == dims, 'Length of simgas is {:d}, not consistent with the dimension {:d}!'.format(
            len(sigmas), dims)
        sigmas = np.asarray(sigmas)

    # random walk loop
    trajectory = list()
    for i, init_pt in enumerate(init_pts):
        cur_step = 0
        cur_pt = init_pt
        cur_traj = [init_pt]
        dist_2_anchors = np.linalg.norm(anchor_pts - init_pt, axis=1)
        cur_anchor = anchor_pts[dist_2_anchors.argmax()]
        update_anchor = True
        while cur_step < num_steps - 1:
            # update anchor point which affects the bias
            if (abs(cur_anchor[:3] - cur_pt[:3]) < rate_bound[:3]).all():
                # re-select anchor point when the UAV is close to the previous anchor
                update_anchor = True
            if update_anchor:
                p = np.random.rand()
                # probabilistic anchor update
                if p < 0.15:
                    dist_2_anchors = np.linalg.norm(anchor_pts - cur_pt, axis=1)
                    cur_anchor = anchor_pts[dist_2_anchors.argmax()]
                elif p < 0.3:
                    dist_2_anchors = np.linalg.norm(anchor_pts - cur_pt, axis=1)
                    cur_anchor = anchor_pts[dist_2_anchors.argmin()]
                else:
                    cur_anchor = anchor_pts[np.random.choice(len(anchor_pts))]
                update_anchor = False

            # search for suitable single step direction
            good_rnd_found = False
            iter_per_loop = 0
            while not good_rnd_found and iter_per_loop < trial_per_loop:
                cur_mean = np.clip(cur_anchor - cur_pt, -rate_bound, rate_bound) * 0.15  # weekened bias effect
                plus_or_minus = np.random.randint(-1, 1, dims)
                plus_or_minus[plus_or_minus == 0] = 1  # either -1 or 1
                coef_min = 0.5
                delta = plus_or_minus * (np.random.rand(dims) * (1 - coef_min) + coef_min) * sigmas
                delta += cur_mean
                tmp_pt = cur_pt + delta
                # conditions check
                if lower_bound[3] == 0 and upper_bound[3] == 360:  # unlimited yaw
                    tmp_pt[3] = tmp_pt[3] % 360
                if (tmp_pt >= lower_bound).all() and (tmp_pt <= upper_bound).all() and (abs(delta) <= rate_bound).all():
                    good_rnd_found = True
                iter_per_loop += 1

            # clip the step to satisfy basic constraints
            if not good_rnd_found:
                space_lb = lower_bound - cur_pt
                space_up = upper_bound - cur_pt
                delta = np.clip(delta, -rate_bound, rate_bound)
                delta = np.clip(delta, space_lb, space_up)

            # move forward
            cur_pt = cur_pt + delta
            cur_traj.append(cur_pt)
            cur_step += 1
        cur_traj = np.asarray(cur_traj)
        trajectory.append(cur_traj)
    return trajectory


def read_preset(preset_path):
    """
    Read json preset file for random walk trajectory generation.
    """
    assert os.path.exists(preset_path), 'Preset .json at {:s} does not exist!'.format(preset_path)
    with open(preset_path, 'r') as f:
        data = json.load(f)
        height_ref = None if 'heightRef' not in data else data['heightRef']
        if height_ref is not None:
            height_ref = os.path.abspath(height_ref.replace('PROJ_DIR', PROJ_DIR))
        return data['nbSeq'], data['seqLen'], data['lonRange'], data['latRange'], data['hRange'], data['yawRange'], \
               data['pitchRange'], data['rollRange'], data['maxTransVel'], data['maxRotVel'], height_ref


def geographic_to_ecef(lon, lat, alt):
    """
    Conversion from WGS84 geographic to ECEF geocentric.
    """
    # Careful: here we need to use lat,lon
    return pyproj.Transformer.from_crs("epsg:4979", "epsg:4978").transform(lat, lon, alt)


def config_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', type=str, help='Name for the random walk trajectory file')
    parser.add_argument('-p', type=str, help='Path to the .json preset')
    parser.add_argument('-visualize', default=False, action='store_true', help='Visualize the generated trajectories')
    opt = parser.parse_args()
    return opt


def main():
    """
    Main function.
    """
    args = config_parse()

    random.seed(2021)
    np.random.seed(2021)
    # read preset
    [nb_seq, seq_len, lon, lat, h, yaw, pitch, roll, max_trans_vel, max_rot_vel, height_ref] = read_preset(args.p)
    max_trans_vel = np.asarray(max_trans_vel)
    max_rot_vel = np.asarray(max_rot_vel)

    # define cube boundary; horizontal: LV95, vertical: WGS84
    r = ReframeTransform()
    min_xyz = r.transform([lon[0], lat[0], h[0]], 'wgs84', 'wgs84', 'lv95', 'wgs84')
    max_xyz = r.transform([lon[1], lat[1], h[1]], 'wgs84', 'wgs84', 'lv95', 'wgs84')
    origin = min_xyz
    x_min, x_max = 0, max_xyz[0] - min_xyz[0]
    y_min, y_max = 0, max_xyz[1] - min_xyz[1]
    z_min, z_max = min_xyz[2], max_xyz[2]  # keep true height
    origin[2] = 0  # keep true height
    yaw_min, yaw_max = yaw
    pitch_min, pitch_max = pitch
    roll_min, roll_max = roll
    print("Scene surface: {:.2f}".format(x_max * y_max) + " m2")
    print("Scene volume: {:.2f}".format(x_max * y_max * z_max) + " m3")

    # synthesize random walk trajectories
    rx = [x_min, x_max]
    ry = [y_min, y_max]
    rz = [z_min, z_max]
    num_pt_per_surf = max(2, nb_seq // 5)
    pt_init = get_init_points(rx, ry, rz, num_pt_per_surf)
    anchor_pts = np.asarray(
        list(product(np.linspace(x_min, x_max, 20), np.linspace(y_min, y_max, 20), np.linspace(z_min, z_max, 20))))
    pt_init_angle = np.random.rand(pt_init.shape[0], 3) * np.asarray(
        [yaw_max - yaw_min, pitch_max - pitch_min, roll_max - roll_min]) \
                    + np.asarray([yaw_min, pitch_min, roll_min])
    anchor_pts_angle = np.random.rand(anchor_pts.shape[0], 3) * np.asarray(
        [yaw_max - yaw_min, pitch_max - pitch_min, roll_max - roll_min]) \
                       + np.asarray([yaw_min, pitch_min, roll_min])
    pt_init = np.concatenate([pt_init, pt_init_angle], axis=1)
    anchor_pts = np.concatenate([anchor_pts, anchor_pts_angle], axis=1)
    trajectory_ls = random_walk(pt_init, anchor_pts=anchor_pts, num_steps=seq_len,
                          sigmas=np.concatenate([max_trans_vel / 4, max_rot_vel / 4], axis=0),
                          lower_bound=[x_min, y_min, z_min, yaw_min, pitch_min, roll_min],
                          upper_bound=[x_max, y_max, z_max, yaw_max, pitch_max, roll_max],
                          rate_bound=np.concatenate([max_trans_vel, max_rot_vel], axis=0),
                          include_angle=True)

    # make corrections for relative height
    if height_ref:
        print("Translating heights using provided DTM. The DTM has to be in LV95 with WGS84 heights")
        z_min = float('inf')
        z_max = float('-inf')
        with rasterio.open(height_ref) as src:
            height = src.read()[0]
            for i, traj in enumerate(tqdm(trajectory_ls)):
                for j, pt in enumerate(traj):
                    lv95pose = [pt[0] + origin[0], pt[1] + origin[1]]
                    row, col = src.index(lv95pose[0], lv95pose[1])
                    trajectory_ls[i][j, 2] += height[row, col]
                    if trajectory_ls[i][j, 2] > z_max:
                        z_max = trajectory_ls[i][j, 2]
                    if trajectory_ls[i][j, 2] < z_min:
                        z_min = trajectory_ls[i][j, 2]

    # randomly select trajectories
    random.shuffle(trajectory_ls)
    selected_traj = np.random.choice(len(trajectory_ls), nb_seq, replace=False)
    traj_ls_ = trajectory_ls
    trajectory_ls = [traj_ls_[idx] for idx in selected_traj]

    # smooth the trajectory
    for i, traj in enumerate(trajectory_ls):
        # traj_ls[i] = signal.savgol_filter(traj, window_length=11, polyorder=3, axis=0)
        for j in range(traj.shape[1]):
            if j < 3:  # for XYZ
                trajectory_ls[i][:, j] = signal.savgol_filter(traj[:, j], window_length=31, polyorder=6, axis=0)
            else:  # for yaw, pitch, roll
                trajectory_ls[i][:, j] = signal.savgol_filter(traj[:, j], window_length=11, polyorder=6, axis=0)

    # plot
    if args.visualize:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.gca(projection='3d')

        # draw cube
        rz = [z_min, z_max]
        for z in rz:
            ax.plot3D(np.ones(100) * x_min, np.linspace(y_min, y_max, 100), z, 'b')
            ax.plot3D(np.ones(100) * x_max, np.linspace(y_min, y_max, 100), z, 'b')
            ax.plot3D(np.linspace(x_min, x_max, 100), np.ones(100) * y_min, z, 'b')
            ax.plot3D(np.linspace(x_min, x_max, 100), np.ones(100) * y_max, z, 'b')
        for x in rx:
            for y in ry:
                ax.plot3D(np.ones(100) * x, np.ones(100) * y, np.linspace(z_min, z_max, 100), 'b')

        # draw trajectories
        for traj in trajectory_ls:
            ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='r')
            ax.plot3D(traj[:, 0], traj[:, 1], traj[:, 2])
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_zlim([z_min - 20, z_max + 20])
        ax.set_title('{:d} random walk trajectories in the cube'.format(len(trajectory_ls)))
        os.makedirs(os.path.join(SCRIPT_DIR, 'presets', args.name + '-seq-all'), exist_ok=True)
        plt.show()
        fig.savefig(os.path.join(SCRIPT_DIR, 'presets', args.name + '-seq-all', '{:s}_all_traj.png'.format(args.name)),
                    dpi=300, bbox_inches='tight', pad_inches=0)

        # pick one trajectory and show 6 DOF parameters curve
        y_labels = [['X (m)', 'Yaw (deg)'], ['Y (m)', 'Pitch (deg)'], ['Z (m)', 'Roll (deg)']]
        for i_traj in range(len(trajectory_ls)):
            fig, axes = plt.subplots(3, 2, figsize=(8, 6))
            for row in range(3):
                for col in range(2):
                    axes[row][col].plot(trajectory_ls[i_traj][:, 3 * col + row])
                    axes[row][col].get_xaxis().set_visible(False)
                    axes[row][col].set_ylabel(y_labels[row][col], labelpad=0)
                    axes[-1][col].get_xaxis().set_visible(True)
                    axes[-1][col].set_xlabel('Time steps')
                    if col == 1:
                        if row == 0:
                            axes[row][col].set_ylim([yaw_min, yaw_max])
                        elif row == 1:
                            axes[row][col].set_ylim([pitch_min, pitch_max])
                        elif row == 2:
                            axes[row][col].set_ylim([roll_min, roll_max])
                        else:
                            raise NotImplementedError
            axes[0][0].set_title('Trajectory {:d}'.format(i_traj))
            folder = os.path.join(SCRIPT_DIR, 'presets', args.name + '-seq{:02d}'.format(i_traj))
            if not os.path.exists(folder):
                os.makedirs(folder)
            fig.savefig(os.path.join(folder, os.path.basename(folder) + '_traj.png'), dpi=300, bbox_inches='tight',
                        pad_inches=0)
            plt.close('all')

    # converts back to wgs84 geocentric coordinate system
    print("Saving poses in ECEF coordinate system")
    pic_names = list()
    for i, traj in enumerate(tqdm(trajectory_ls)):
        xyz_ls = list()
        for j, pt in enumerate(traj):
            pt_ = r.transform(pt[:3] + origin, 'lv95', 'wgs84', 'wgs84', 'wgs84')
            xyz_ls.append(pt_)
            pic_names.append('seq_{:02d}_{:04d}'.format(i, j))
        xyz_ls = np.asarray(xyz_ls)
        xyz_ls = np.asarray(geographic_to_ecef(xyz_ls[:, 0], xyz_ls[:, 1], xyz_ls[:, 2])).T
        trajectory_ls[i][:, :3] = xyz_ls
        folder = os.path.join(SCRIPT_DIR, 'presets', args.name + '-seq{:02d}'.format(i))
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.save(os.path.join(folder, os.path.basename(folder) + '_poses.npy'), trajectory_ls[i])
    trajectory_all = np.concatenate(trajectory_ls, axis=0)
    folder = os.path.join(SCRIPT_DIR, 'presets', args.name + '-seq-all')
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, os.path.basename(folder) + '_poses.npy'), trajectory_all)
    with open(os.path.join(folder, os.path.basename(folder) + '_pic_names.csv'), 'w') as f:
        f.write(','.join(pic_names))
    print('{:d} random walk trajectories (each has {:d} frames) are saved.'.format(nb_seq, seq_len))


if __name__ == "__main__":
    main()
