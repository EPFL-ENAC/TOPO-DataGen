"""
This script run the point cloud processing
Usage for run the colorization process :
python classified_point_cloud_processing.py --task colorization --point_cloud_input_dir /home/regislongchamp/Documents/TOPO-DataGen/data_preprocess/demo/demo-surface3d-python --point_cloud_output_dir /home/regislongchamp/Documents/TOPO-DataGen/data_preprocess/demo/demo-surface3d-python/c --orthophoto_file_path /home/regislongchamp/Documents/TOPO-DataGen/data_preprocess/demo/demo-swissimage10/mergedVRT-lv95.vrt

Usage to run the downsampling processing
python classified_point_cloud_processing.py --task downsampling --point_cloud_input_dir /home/regislongchamp/Documents/TOPO-DataGen/data_preprocess/demo/demo-surface3d-python --point_cloud_output_dir /home/regislongchamp/Documents/TOPO-DataGen/data_preprocess/demo/demo-surface3d-python/d
"""
import os
import argparse
from points_cloud_processing import batch_colorize_point_cloud, batch_downsampled_colorize_point_cloud


def command_parser():
    """
    Parse and check the user's input commands
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help="Operation to run (colorization or downsampling)")
    parser.add_argument('--point_cloud_input_dir', help="Input las point cloud directory")
    parser.add_argument('--point_cloud_output_dir', help="Output las point cloud directory")
    parser.add_argument('--orthophoto_file_path', default=None, help="Path of the orthophoto file")

    args = parser.parse_args()

    # General checks
    assert args.task, "--task argunment must be provided"
    assert args.task in ['colorization','downsampling'], '--task argunment must be either "colorization" our "downsampling"'
    assert args.point_cloud_input_dir, " --point_cloud_input_dir argument must be provided"
    assert args.point_cloud_output_dir, "--point_cloud_output_dir argument must be provided"
    if args.task == 'colorization' :
        assert args.orthophoto_file_path, "--orthophoto_file_path argument must be provided"
    assert os.path.isdir(args.point_cloud_input_dir), f"Path {args.point_cloud_input_dir} is not a directory!"
    assert os.path.isdir(args.point_cloud_output_dir), f"Path {args.point_cloud_output_dir} is not a directory!"
    if args.task == 'colorization' :
        assert os.path.exists(args.orthophoto_file_path), "--orthophoto_file_path must exists"
    return args


def run_point_cloud_process(args):
    """
    Run the point cloud processing based on arguments provided
    :param args: argparse object, see command_parser() for a detail description
    :return: None
    """
    if args.task == 'colorization' :
        path_in_folder_point_cloud = args.point_cloud_input_dir
        path_out_folder_colorized_point_cloud = args.point_cloud_output_dir
        path_in_file_orthophoto = args.orthophoto_file_path
        batch_colorize_point_cloud(path_in_folder_point_cloud,
                                   path_out_folder_colorized_point_cloud,
                                   path_in_file_orthophoto)

    elif args.task == 'downsampling' :
        path_in_folder_point_cloud = args.point_cloud_input_dir
        path_out_folder_downsampled_point_cloud = args.point_cloud_output_dir
        batch_downsampled_colorize_point_cloud(path_in_folder_point_cloud,path_out_folder_downsampled_point_cloud)





args = command_parser()
run_point_cloud_process(args)






