import argparse
import pdb
import laspy
import os
import shutil
import numpy as np
from tqdm import tqdm
from semantics_recovery import all_path, mkdir


def config_parser():
    parser = argparse.ArgumentParser(
        description='Semantic label recovery script.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--input_path', type=str, default=None, required=True,
                        help='Directory where the raw data is.')

    parser.add_argument('--las_path', type=str, default=None, required=True,
                        help='Directory where the las files are.')

    parser.add_argument('--output_path', type=str, default=None,
                        help='Directory to store cleaned npy labels. Default is the raw data path.')

    parser.add_argument('--save_backup', action="store_true",
                        help='Flag to backup the raw data before cleaning.')

    parser.add_argument('--semantics', action="store_true",
                        help='To clean semantics data in-place based on the scene coord results.')

    parser.add_argument('--start_idx', type=int, default=0,
                        help='start index of the file list')

    parser.add_argument('--end_idx', type=int, default=None,
                        help='stop index of the file list')

    opt = parser.parse_args()

    if opt.output_path is None:
        opt.output_path = opt.input_path
        print("Point cloud label output path is not specified! The raw data path is used, "
              "i.e., the cleaned npy labels are generated in place.")
    return opt


def main():
    args = config_parser()
    print(args)
    start_idx = args.start_idx
    end_idx = args.end_idx

    # set the input and output path
    las_file_path = os.path.abspath(args.las_path)
    input_path = os.path.abspath(args.input_path)
    output_path = os.path.abspath(args.output_path)
    file_ls, folder_ls = all_path(input_path, filter_list=['.npy'])
    print("To process {:d} npy labels at {:s}".format(len(file_ls), input_path))

    # load raw las and get the boundary in ECEF
    las_ls = [las for las in os.listdir(las_file_path) if las.endswith('.las')]
    bound_xyz_min, bound_xyz_max = np.array([float('inf')] * 3), np.zeros(3)
    for idx, las_name in enumerate(las_ls):
        las = laspy.read(os.path.join(las_file_path, las_name))
        this_min_xyz = np.min(np.stack([las.x, las.y, las.z]), axis=1)
        this_max_xyz = np.max(np.stack([las.x, las.y, las.z]), axis=1)
        bound_xyz_min = np.minimum(bound_xyz_min, this_min_xyz)
        bound_xyz_max = np.maximum(bound_xyz_max, this_max_xyz)
        las = None
    print("XYZ boundary min: {}, max: {}".format(bound_xyz_min, bound_xyz_max))

    # create output folder structure
    input_path_len = len(input_path.split('/'))
    folder_ls = ['/'.join(folder.split('/')[input_path_len:]) for folder in folder_ls]
    folder_ls = [folder for folder in folder_ls if 'outlier-removal-backup' not in folder]
    folder_ls = np.unique(folder_ls).tolist()
    mkdir(output_path, folder_ls)
    if args.save_backup:
        output_backup_path = os.path.abspath(os.path.join(args.output_path, 'outlier-removal-backup'))
        mkdir(output_backup_path, folder_ls)

    # process the labels
    for idx_dp, file_name in tqdm(enumerate(file_ls[start_idx:end_idx])):
        """Load ray-traced point cloud"""
        sc_path = '{:s}_pc.npy'.format(file_name)
        sm_path = '{:s}_semantics.npy'.format(file_name)

        out_sc_path = os.path.join(args.output_path, '{:s}_pc.npy'.format(
                    '/'.join(file_name.split('/')[input_path_len:])))
        out_sm_path = os.path.join(args.output_path, '{:s}_semantics.npy'.format(
                    '/'.join(file_name.split('/')[input_path_len:])))

        _sc = np.load(sc_path)  # [480, 720, 3]
        _sc_shape = _sc.shape
        sc = _sc.reshape(-1, 3).copy()  # [N, 3]
        mask_has_data = sc[:, 0] != -1
        mask_outlier = np.logical_or(sc > bound_xyz_max, sc < bound_xyz_min).sum(axis=1) > 0  # [N]
        mask_outlier = np.logical_and(mask_outlier, mask_has_data)

        if mask_outlier.sum():
            print("{:d} / {:d} outliers to remove, percentage: {:.2f}%, file: {:s}".format(
                mask_outlier.sum(), len(sc), mask_outlier.sum() / len(sc) * 100, os.path.abspath(sc_path)))
            sc[mask_outlier] = -1.0
            sc = sc.reshape(_sc_shape)

            if args.semantics and os.path.exists(sm_path):
                _sm = np.load(sm_path)  # [480, 720]
                _sm_shape = _sm.shape
                sm = _sm.reshape(-1, 1).copy()
                sm[mask_outlier] = 0  # set to 0 for semantics instead of -1
                sm = sm.reshape(_sm_shape).astype(np.uint8)

            # backup only the point cloud has potential outliers
            if args.save_backup:
                # backup point cloud
                out_backup_path = os.path.join(output_backup_path, '{:s}_pc.npy'.format(
                    '/'.join(file_name.split('/')[input_path_len:])))
                shutil.copy(sc_path, out_backup_path)

                if args.semantics and os.path.exists(sm_path):
                    # backup semantics map
                    out_backup_path = os.path.join(output_backup_path, '{:s}_semantics.npy'.format(
                        '/'.join(file_name.split('/')[input_path_len:])))
                    shutil.copy(sm_path, out_backup_path)

            # overwrite the old files
            np.save(out_sc_path, sc)
            if args.semantics and os.path.exists(sm_path):
                np.save(out_sm_path, sm)


if __name__ == '__main__':
    main()
