import os
import pdb

import laspy
import argparse
import numpy as np
import open3d as o3d
import multiprocessing as mp
from glob import glob
from typing import Tuple

"""
This simple script aims to downsample the las file. Specifically, we downsample the points by class whose densities are 
way higher than the rest. This helps to improve downstream tasks efficiency.
"""


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('las_in_dir', help="Input las point cloud directory")
    parser.add_argument('--las_out_dir', default=None, help="Output las point cloud directory")
    parser.add_argument('--downsample_class', type=int, nargs='+', default=[2, 3],
                        help="Points with such class IDs are to be downsampled, default [2, 3]")
    parser.add_argument('--voxel_size', type=float, nargs='+', default=[1.5, 2.0],
                        help="Control parameter of voxel downsampling, default value [1.5, 2.0]")
    parser.add_argument('--multi_process', action="store_true", default=False,
                        help="To use multiprocessing for acceleration. Maybe you need a huge RAM.")
    args = parser.parse_args()
    args.las_in_dir = os.path.abspath(args.las_in_dir)
    assert os.path.isdir(args.las_in_dir), "Path {:s} is not a directory!".format(args.las_in_dir)
    assert len(args.downsample_class) == len(args.voxel_size), "Length of downsampling classes much match that of " \
                                                               "voxel size!"
    if args.las_out_dir is None:
        args.las_out_dir = os.path.abspath(os.path.join(args.las_in_dir, '../ecef-downsampled'))
        os.makedirs(args.las_out_dir, exist_ok=True)
    return args


def _select_las(las_in, flag_select):
    new_header = laspy.LasHeader(point_format=las_in.header.point_format, version=las_in.header.version)
    new_header.offsets = las_in.header.offsets
    new_header.scales = las_in.header.scales
    las_out = laspy.LasData(new_header)

    idx_select = [i for i, flag in enumerate(flag_select) if flag]
    las_out.x = np.take(las_in.x, idx_select)
    las_out.y = np.take(las_in.y, idx_select)
    las_out.z = np.take(las_in.z, idx_select)
    las_out.red = np.take(las_in.red, idx_select)
    las_out.green = np.take(las_in.green, idx_select)
    las_out.blue = np.take(las_in.blue, idx_select)
    las_out.raw_classification = np.take(las_in.raw_classification, idx_select)
    las_out.intensity = np.take(las_in.intensity, idx_select)
    return las_out


def merge_las(las_in_ls: list) -> laspy.LasData:
    """Merge a list of las point clouds."""
    out_x, out_y, out_z = [], [], []
    out_red, out_green, out_blue = [], [], []
    out_class, out_intensity = [], []
    for las_in in las_in_ls:
        out_x.append(las_in.x)
        out_y.append(las_in.y)
        out_z.append(las_in.z)
        out_red.append(las_in.red)
        out_green.append(las_in.green)
        out_blue.append(las_in.blue)
        out_class.append(las_in.raw_classification)
        out_intensity.append(las_in.intensity)

    new_header = laspy.LasHeader(point_format=las_in_ls[0].header.point_format, version=las_in_ls[0].header.version)
    new_header.offsets = las_in_ls[0].header.offsets
    new_header.scales = las_in_ls[0].header.scales
    las_out = laspy.LasData(new_header)
    las_out.x = np.concatenate(out_x)
    las_out.y = np.concatenate(out_y)
    las_out.z = np.concatenate(out_z)
    las_out.red = np.concatenate(out_red)
    las_out.green = np.concatenate(out_green)
    las_out.blue = np.concatenate(out_blue)
    las_out.raw_classification = np.concatenate(out_class)
    las_out.intensity = np.concatenate(out_intensity)
    return las_out


def downsample_o3d(xyz: np.ndarray, voxel_size: float) -> Tuple[o3d.geometry.PointCloud, list]:
    """Call open3d utility for downsampling."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    pcd_down, _, idx_prev = pcd.voxel_down_sample_and_trace(voxel_size, pcd.get_min_bound(), pcd.get_max_bound(), False)
    idx_prev = [i[0] for i in idx_prev]
    return pcd_down, idx_prev


def process_las(input_file: str, downsample_class_ls: list, voxel_size_ls: list) -> Tuple[laspy.LasData, int, int]:
    """Read las file and keep some of the attributes."""
    las_in = laspy.read(input_file)
    flag_pre_downsample = [las_in.raw_classification == this_class for this_class in downsample_class_ls]
    flag_pre_downsample = (np.stack(flag_pre_downsample, axis=0).sum(axis=0) > 0).astype(bool)  # [N]
    flag_retain = np.logical_not(flag_pre_downsample)
    las_retain = _select_las(las_in, flag_retain)  # points not to touch
    las_to_downsample = _select_las(las_in, flag_pre_downsample)  # points to downsample
    las_downsampled_ls = []

    for this_class, this_voxel_size in zip(downsample_class_ls, voxel_size_ls):
        flag_this_class = las_to_downsample.raw_classification == this_class
        las_this_class = _select_las(las_to_downsample, flag_this_class)
        xyz_this_class_raw = np.stack([las_this_class.x, las_this_class.y, las_this_class.z], axis=1)  # [K, 3]
        _, idx_downsample = downsample_o3d(xyz_this_class_raw, this_voxel_size)  # retrieve the indices
        flag_post_downsample = [False] * len(xyz_this_class_raw)
        for i in idx_downsample:
            flag_post_downsample[i] = True
        las_this_class_downsampled = _select_las(las_this_class, flag_post_downsample)
        las_downsampled_ls.append(las_this_class_downsampled)

    las_out = merge_las([*las_downsampled_ls, las_retain])
    return las_out, len(las_in), len(las_out)


def _func_backbone(las_in_path, las_out_path, downsample_class_ls, voxel_size_ls, task_size, mp_lock, mp_progress):
    """Wrapper for multiprocessing use."""
    las_out, len_in, len_out = process_las(las_in_path, downsample_class_ls, voxel_size_ls)
    las_out.write(las_out_path)
    with mp_lock:
        print("Iteration {:d} / {:d}, point cloud at {:s} is downsampled to {:s}".format(
            mp_progress.value + 1, task_size, las_in_path, las_out_path), flush=True)
        print("points downsampled {:d} ---> {:d}, {:.2f}% points are removed.".format(
            len_in, len_out, float((len_in - len_out) / len_in) * 100.0), flush=True)
        mp_progress.value += 1


def main():
    args = config_parser()
    print(args)
    las_in_path_ls = sorted(glob(os.path.join(args.las_in_dir, '*')))
    print("To process {:d} point clouds at {:s}".format(len(las_in_path_ls), args.las_in_dir))

    mp_manager = mp.Manager()
    mp_lock = mp_manager.Lock()
    mp_progress = mp_manager.Value('i', 0)
    mp_args_ls = []
    for i, las_in_path in enumerate(las_in_path_ls):
        las_out_path = os.path.join(args.las_out_dir, os.path.basename(las_in_path))
        mp_args_ls.append((las_in_path, las_out_path, args.downsample_class, args.voxel_size,
                           len(las_in_path_ls), mp_lock, mp_progress))

    if args.multi_process:
        with mp.Pool() as p:
            p.starmap(_func_backbone, mp_args_ls, chunksize=1)
    else:
        for mp_args in mp_args_ls:
            _func_backbone(*mp_args)

    print("Task is done!")


if __name__ == "__main__":
    main()
