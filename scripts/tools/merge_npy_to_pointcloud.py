import os
import argparse
import pdb

import numpy as np
import open3d as o3d
from tqdm import tqdm
from glob import glob


def config_parser():
    parser = argparse.ArgumentParser(
        description='Merge scene coordinate map npy files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--label_path', type=str, default=None, required=True,
                        help='Directory where the semantic labels are in.')

    parser.add_argument('--output_path', type=str, default=None,
                        help='Directory to store sampling output.')

    parser.add_argument('--split_factor', type=int, default=1,
                        help="To split the overall point cloud to avoid memory issue.")

    parser.add_argument('--chunk_size', type=int, default=50,
                        help="Size of each chunk to load.")

    opt = parser.parse_args()

    if opt.output_path is None:
        print("Warning: output path is None. Try to use the parent of label path first...")
        opt.output_path = os.path.dirname(opt.label_path)

    opt.label_path = os.path.abspath(opt.label_path)
    opt.output_path = os.path.abspath(opt.output_path)

    return opt


def point_cloud_from_paths(points_path, chunk_size):
    pcd = o3d.geometry.PointCloud()
    iteration = round(len(points_path) / chunk_size)
    points_chunks = np.array_split(points_path, iteration)
    print("Reading {:d} Points".format(len(points_path)))

    for points_files in tqdm(points_chunks):
        points = np.array([np.load(file).reshape(-1, 3) for file in points_files]).reshape(-1, 3)

        valid = (points[:, 0] != -1)

        pcd.points = o3d.utility.Vector3dVector(np.vstack((points[valid], np.asarray(pcd.points))))

    return pcd


def main():
    args = config_parser()
    os.makedirs(args.output_path, exist_ok=True)

    pc_ls = []
    for root, dirs, files in os.walk(args.label_path):
        for file in files:
            if file.endswith('pc.npy'):
                pc_ls.append(os.path.join(root, file))
    pc_ls.sort()

    print("{:d} point clouds are to be merged from {:s} with a split factor of {:d}".format(
        len(pc_ls), args.label_path, args.split_factor))

    pc_ls_split_ = np.array_split(pc_ls, args.split_factor)
    for i, pc_ls_split in enumerate(pc_ls_split_):
        print("Progress: {:d}/{:d}, {:d} point clouds are to be merged for this split".format(
            i + 1, len(pc_ls_split_), len(pc_ls_split)))

        if len(pc_ls_split):
            pcd = point_cloud_from_paths(pc_ls_split, args.chunk_size)

            pcd_out_path = os.path.join(args.output_path,
                                        'merged_point_cloud_{:s}_{:d}.ply'.format(os.path.basename(args.label_path), i))
            o3d.io.write_point_cloud(pcd_out_path, pcd)

    print("Merged point cloud is saved at {:s}".format(pcd_out_path))


if __name__ == "__main__":
    main()
