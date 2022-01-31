import argparse
import os
import pdb

import pyproj
import numpy as np
from glob import glob
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


def config_parser():
    parser = argparse.ArgumentParser(
        description='Semantic label sampling script.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--label_path', type=str, default=None, required=True,
                        help='Directory where the point cloud npy files are in.')

    parser.add_argument('--threshold', type=float, default=50.0,
                        help='Minimum value threshold for non-empty pixels percentage.')

    opt = parser.parse_args()
    return opt


def get_rotation_ned_in_ecef(lon, lat):
    """
    @param: lon, lat Longitude and latitude in degree
    @return: 3x3 rotation matrix of heading-pith-roll NED in ECEF coordinate system
    Reference: https://apps.dtic.mil/dtic/tr/fulltext/u2/a484864.pdf, Section 4.3, 4.1
    Reference: https://www.fossen.biz/wiley/ed2/Ch2.pdf, p29
    """
    # describe NED in ECEF
    lon = lon * np.pi / 180.0
    lat = lat * np.pi / 180.0
    # manual computation
    R_N0 = np.array([[np.cos(lon), -np.sin(lon), 0],
                     [np.sin(lon), np.cos(lon), 0],
                     [0, 0, 1]])
    R__E1 = np.array([[np.cos(-lat - np.pi / 2), 0, np.sin(-lat - np.pi / 2)],
                      [0, 1, 0],
                      [-np.sin(-lat - np.pi / 2), 0, np.cos(-lat - np.pi / 2)]])
    NED = np.matmul(R_N0, R__E1)
    assert abs(np.linalg.det(
        NED) - 1.0) < 1e-6, 'NED in NCEF rotation mat. does not have unit determinant, it is: {:.2f}'.format(
        np.linalg.det(NED))
    return NED


def ecef_to_geographic(x, y, z):
    # Careful: here we need to use lat,lon
    lat, lon, alt = pyproj.Transformer.from_crs("epsg:4978", "epsg:4979").transform(x, y, z)
    return [lon, lat, alt]


def get_pose_mat(cesium_pose):
    """
    Get 4x4 homogeneous matrix from Cesium-defined pose
    @input: cesium_pose 6d ndarray, [lat, lon, h, yaw, pitch, roll]
    lat, lon, h are in ECEF coordinate system
    yaw, pitch, roll are in degress
    @output: 4x4 homogeneous extrinsic camera matrix
    """
    x, y, z, yaw, pitch, roll = cesium_pose  # no need to do local conversion when in ECEF
    lon, lat, alt = ecef_to_geographic(x, y, z)
    rot_ned_in_ecef = get_rotation_ned_in_ecef(lon, lat)
    rot_pose_in_ned = R.from_euler('ZYX', [yaw, pitch, roll], degrees=True).as_matrix()
    r = np.matmul(rot_ned_in_ecef, rot_pose_in_ned)
    # transform coordinate system from NED to standard camera sys.
    r = r[0:3, [1, 2, 0]]
    r = np.concatenate((r, np.array([[x, y, z]]).transpose()), axis=1)
    r = np.concatenate((r, np.array([[0, 0, 0, 1]])), axis=0)
    return r


def get_cam_mat(width, height, focal_length):
    """
    Get intrinsic camera matrix
    """
    cam_mat = np.eye(3, dtype=float)
    cam_mat[0, 0] = focal_length
    cam_mat[1, 1] = focal_length
    cam_mat[0, 2] = width / 2
    cam_mat[1, 2] = height / 2
    return cam_mat


def main():
    args = config_parser()
    print(args)

    file_ls, pose_ls = [], []
    for root, dirs, files in os.walk(args.label_path):
        files.sort()
        flag_pose = any([file_name.endswith('_poses.npy') for file_name in files])
        if flag_pose:
            files = [os.path.abspath(os.path.join(root, file_name)) for file_name in files
                     if file_name.endswith('_pc.npy')]
            poses = np.load(glob(os.path.join(root, '*_poses.npy'))[0])

            # ensure index consistency
            for idx, file_name in enumerate(files):
                assert '{:05d}'.format(idx) in os.path.basename(file_name)
            file_ls.extend(files)
            pose_ls.append(poses)
    pose_ls = np.concatenate(pose_ls)  # [X, 6]
    assert len(file_ls) == len(pose_ls)
    print("{:d} npy files to scan...".format(len(file_ls)))

    dubious_data_ls = []
    valid_rate_ls = []
    xyz_min = np.array([float('inf'), float('inf'), float('inf')])
    xyz_max = np.array([-float('inf'), -float('inf'), -float('inf')])
    reproj_error_ls = []

    cam_mat = get_cam_mat(720, 480, 480)

    # generate grid of target reprojection pixel positions
    pixel_grid = np.zeros((2, 480, 720))
    for x in range(0, pixel_grid.shape[2]):
        for y in range(0, pixel_grid.shape[1]):
            pixel_grid[0, y, x] = x
            pixel_grid[1, y, x] = y
    pixel_grid = pixel_grid.reshape(2, -1)  # [2, H*W]

    for i, (npy_file, cam_pose) in tqdm(enumerate(zip(file_ls, pose_ls))):

        cam_to_world = get_pose_mat(cam_pose)
        this_origin = cam_to_world[:3, -1].copy()
        cam_to_world[:3, -1] -= this_origin

        this_pc = np.load(npy_file)  # [H, W, 3]
        mask_nodata = this_pc[:, :, 0] == -1  # [H, W]
        mask_has_data = np.logical_not(mask_nodata)

        # check reprojection error
        reproj_pc = this_pc - this_origin[None, None, :]  # demean, [H, W, 3]
        reproj_pc = reproj_pc.reshape(-1, 3)  # [H*W, 3]

        world_to_cam = np.linalg.inv(cam_to_world)  # [4, 4]

        world_coords = reproj_pc.transpose(1, 0)  # [3, H*W]
        ones = np.ones([1, world_coords.shape[1]])  # [1, H*W]
        world_coords = np.concatenate([world_coords, ones], axis=0)  # [4, H*W]

        cam_coords = np.matmul(world_to_cam[:3, :], world_coords)  # [3, H*W]

        pixel_coords = np.matmul(cam_mat, cam_coords)  # [3, H*W]
        pixel_coords = pixel_coords[0:2] / pixel_coords[2]  # [2, H*W]

        reproj_error = np.linalg.norm(pixel_coords - pixel_grid, axis=0)  # [H*W]
        reproj_error = reproj_error.reshape(480, 720)[mask_has_data]  # [H*W]
        reproj_error_ls.append(np.mean(reproj_error))

        # check non-empty pixel rate
        valid_rate = np.sum(this_pc[:, :, 0] != -1) / (this_pc.shape[0] * this_pc.shape[1])
        valid_rate *= 100.0
        this_pc = this_pc[this_pc[:, :, 0] != -1]  # [X, 3]
        xyz_min = np.minimum(xyz_min, np.min(this_pc.reshape(-1, 3), axis=0))
        xyz_max = np.maximum(xyz_max, np.max(this_pc.reshape(-1, 3), axis=0))
        valid_rate_ls.append(valid_rate)
        if valid_rate < args.threshold:
            print("Point cloud {:s} valid rate {:.1f}% lower than threshold {:.1f}%.".format(
                npy_file, valid_rate, args.threshold))
            dubious_data_ls.append(npy_file + '\n')
    reproj_error_ls = np.array(reproj_error_ls)
    print("Valid rate statistics over {:d} images, mean: {:.2f}%, std: {:.2f}%, median: {:.2f}%".format(
        len(valid_rate_ls), np.mean(valid_rate_ls), np.std(valid_rate_ls), np.median(valid_rate_ls)))
    print('Min and Max boundary values: min: {}, max: {}'.format(xyz_min, xyz_max))
    print("Reprojection error statistics: mean: {:.2f} px, std: {:.2f} px, median: {:.2f} px, max: {:.2f} px".
          format(np.mean(reproj_error_ls), np.std(reproj_error_ls),
                 np.median(reproj_error_ls), np.max(reproj_error_ls)))
    out_path = os.path.join(args.label_path, 'npy_statistics.npz')
    np.savez(out_path,
             valid_rate=np.array(valid_rate_ls),
             reproj_error=np.array(reproj_error_ls),
             file_name=file_ls)
    print("Overall statistics is saved to {:s}".format(out_path))

    if len(dubious_data_ls):
        out_path = os.path.join(args.label_path, 'dubious_data.txt')
        print("{:d} possibly wrong data points are recorded at {:s}".format(len(dubious_data_ls), out_path))
        with open(out_path, 'w') as f:
            f.writelines(dubious_data_ls)
    else:
        print("All point clouds' valid rates are higher than the threshold. Good data!")


if __name__ == '__main__':
    main()
