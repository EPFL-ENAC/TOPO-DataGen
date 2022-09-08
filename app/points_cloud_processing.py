"""
This code aims to gather all functions used to process points clouds (las format).
"""
import os
import json
import pdal
import pathlib
import numpy as np
import typing
import copy
import multiprocessing
from datetime import datetime
from multiprocess import Pool
import laspy
import open3d as o3d


from config import settings


# Get the number of CPUs in the system.
number_of_cpu = multiprocessing.cpu_count()


def colorize_point_cloud(pipepline_json : str) -> None :
    """
    This function colorize one point cloud with a orthophoto
    :param pipepline_json: pdal process template
    :return: None
    """
    point_cloud_file_name = json.loads(pipepline_json).get('pipeline')[0]
    pipeline = pdal.Pipeline(pipepline_json)
    pipeline.execute()



def batch_colorize_point_cloud(path_in_folder_point_cloud : typing.Union[str, pathlib.Path],
                               path_out_folder_colorized_point_cloud : typing.Union[str, pathlib.Path],
                               path_in_file_orthophoto : typing.Union[str, pathlib.Path],
                               logger = None) -> None :
    """
    This function runs the colorisation process over all point cloud file in on folder.
    It first define the Pdal Json pipepline and run the colorize_point_cloud() function.
    It uses the multiprocess library to parallelize processes
    :param path_in_folder_point_cloud: Path of the folder that contains the points to colorized.
    :param path_out_folder_colorized_point_cloud: Path of the out folder.
    :param path_in_file_orthophoto: Orthophoto used for the colorization.
    :return: None
    """

    # Load pdal process template
    pipepline_pdal_template_file_name = settings.points_cloud.pipepline_pdal_template_file_name
    path_file_pipepline_json = os.path.join(os.path.dirname(os.path.realpath(__file__)),pipepline_pdal_template_file_name)
    pipepline_template = json.load(open(path_file_pipepline_json))
    list_pipepline = []

    # Get the list of all las files
    list_point_cloud_files_name = [i for i in os.listdir(path_in_folder_point_cloud) if i.endswith('.las')]

    #assert path_in_file_orthophoto.endswith('vrt'), "Orthophoto must be an GDAL Virtual Format (VRT)"

    for point_cloud_files_name in list_point_cloud_files_name :
        input_point_cloud_files_path = os.path.join(path_in_folder_point_cloud,point_cloud_files_name)
        output_point_cloud_files_path = os.path.join(path_out_folder_colorized_point_cloud,point_cloud_files_name)

        # Adapt template pipepline for the current point cloud file
        pipepline_json = copy.deepcopy(pipepline_template)
        pipepline_json['pipeline'].insert(0, input_point_cloud_files_path)
        pipepline_json['pipeline'][1]['raster'] = path_in_file_orthophoto
        pipepline_json['pipeline'][3]['filename'] = output_point_cloud_files_path
        pipepline_json = json.dumps(pipepline_json, indent=4)

        list_pipepline.append(pipepline_json)


    if logger:
        message = f"The point cloud  {path_in_folder_point_cloud} colorization process has started"
        logger.info(message)
        start = datetime.now()

    pool = Pool(number_of_cpu)
    pool.map(colorize_point_cloud, list_pipepline)
    pool.close()
    pool.join()

    if logger:
        processing_time = (datetime.now() - start).seconds
        message = f"The  point cloud  {path_in_folder_point_cloud} merging process has finished in {processing_time} seconds"
        logger.info(message)




def select_point_clouds(point_cloud_in: laspy.LasData, flag_select: list) -> laspy.LasData :
    """
    Select part of point cloud based on a list of flag.
    If the point value in the flag array is true, the point is selected, if False, the point is removed.
    :param point_cloud_in: laspy point cloud object
    :param flag_select: numpy array containing boolean value
    :return: Laspy point cloud object
    """
    new_header = laspy.LasHeader(point_format=point_cloud_in.header.point_format, version=point_cloud_in.header.version)
    new_header.offsets = point_cloud_in.header.offsets
    new_header.scales = point_cloud_in.header.scales
    point_cloud_out = laspy.LasData(new_header)

    idx_select = [i for i, flag in enumerate(flag_select) if flag]
    point_cloud_out.x = np.take(point_cloud_in.x, idx_select)
    point_cloud_out.y = np.take(point_cloud_in.y, idx_select)
    point_cloud_out.z = np.take(point_cloud_in.z, idx_select)
    point_cloud_out.raw_classification = np.take(point_cloud_in.raw_classification, idx_select)
    point_cloud_out.intensity = np.take(point_cloud_in.intensity, idx_select)
    try :
        point_cloud_out.red = np.take(point_cloud_in.red, idx_select)
        point_cloud_out.green = np.take(point_cloud_in.green, idx_select)
        point_cloud_out.blue = np.take(point_cloud_in.blue, idx_select)
    except:
        pass
    return point_cloud_out

def merge_point_clouds(List_of_point_clouds: list) -> laspy.LasData:
    """
    This function merges point clouds together
    :param List_of_point_clouds:  list of laspy point clouds
    :return: Merge of the initial point cloud in a laspy object (LasData)
    """
    out_x, out_y, out_z = [], [], []
    out_red, out_green, out_blue = [], [], []
    out_class, out_intensity = [], []
    for las_in in List_of_point_clouds:
        out_x.append(las_in.x)
        out_y.append(las_in.y)
        out_z.append(las_in.z)
        out_class.append(las_in.raw_classification)
        out_intensity.append(las_in.intensity)
        try :
            out_red.append(las_in.red)
            out_green.append(las_in.green)
            out_blue.append(las_in.blue)
        except :
            pass

    new_header = laspy.LasHeader(point_format=List_of_point_clouds[0].header.point_format, version=List_of_point_clouds[0].header.version)
    new_header.offsets = List_of_point_clouds[0].header.offsets
    new_header.scales = List_of_point_clouds[0].header.scales
    las_out = laspy.LasData(new_header)
    las_out.x = np.concatenate(out_x)
    las_out.y = np.concatenate(out_y)
    las_out.z = np.concatenate(out_z)
    las_out.raw_classification = np.concatenate(out_class)
    las_out.intensity = np.concatenate(out_intensity)
    try :
        las_out.red = np.concatenate(out_red)
        las_out.green = np.concatenate(out_green)
        las_out.blue = np.concatenate(out_blue)
    except :
        pass

    return las_out



def downsampled_colorize_point_cloud(paths_in_out : tuple) -> None :
    """
    This function runs the downsample process
    :param paths_in_out: Tuple containing the in and out path of the point cloud
    :return: None
    """

    path_in_file_point_cloud = paths_in_out[0]
    path_out_filse_colorized_point_cloud = paths_in_out[1]

    print(f"Running downsampling for file {path_in_file_point_cloud}")

    # Class to downsaple the point cloud.
    downsample_class = settings.points_cloud.downsample_classes

    # Voxel size for the corresponding class
    voxel_size = settings.points_cloud.voxel_size

    # read las file as lapsy object
    point_cloud_in = laspy.read(path_in_file_point_cloud)
    assert len(voxel_size) == len(downsample_class), 'DOWNSAMPLE_CLASSES has not the same length as VOXEL_SIZE'

    # Points to downsample according to their classes (see downsample_class variable)
    flag_pre_downsample = [point_cloud_in.raw_classification == this_class for this_class in downsample_class]
    flag_pre_downsample = (np.stack(flag_pre_downsample, axis=0).sum(axis=0) > 0).astype(bool)  # [N]
    point_to_downsample = select_point_clouds(point_cloud_in, flag_pre_downsample)

    # Untouched point according to their classes (see downsample_class varaible)
    flag_retain = np.logical_not(flag_pre_downsample)
    point_cloud_to_keep = select_point_clouds(point_cloud_in, flag_retain)

    # Create a list that will gather all downlampled points and untouched points depending on their classes
    list_of_part_of_point_clouds = []

    # Run the downsampling process by iterating over the classes (and corresponding volex size)
    for current_class, current_voxel_size in zip(downsample_class, voxel_size):
        # take only the current class of point
        flag_this_class = point_to_downsample.raw_classification == current_class
        point_cloud_this_class = select_point_clouds(point_to_downsample, flag_this_class)
        xyz_this_class_raw = np.stack([point_cloud_this_class.x, point_cloud_this_class.y, point_cloud_this_class.z], axis=1)  # [K, 3]

        # Convert float64 numpy array to Open3D object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_this_class_raw)

        # Downsample input pointcloud into output pointcloud with a voxel and record the point ID
        pcd_down, _, idx_prev = pcd.voxel_down_sample_and_trace(current_voxel_size, pcd.get_min_bound(), pcd.get_max_bound(),False)
        idx_downsample = [i[0] for i in idx_prev]
        flag_post_downsample = [False] * len(xyz_this_class_raw)
        for i in idx_downsample:
            flag_post_downsample[i] = True

        # Select the resulting point according to theirs IDs
        point_cloud_this_class_downsampled = select_point_clouds(point_cloud_this_class, flag_post_downsample)

        # Append to the list of point cloud
        list_of_part_of_point_clouds.append(point_cloud_this_class_downsampled)

    # Merge the list of point clouds together
    point_cloud_out = merge_point_clouds([*list_of_part_of_point_clouds, point_cloud_to_keep])

    # Write the point cloud into las file
    point_cloud_out.write(path_out_filse_colorized_point_cloud)

    print(f"Downsampling finished for file {path_out_filse_colorized_point_cloud}")



def batch_downsampled_colorize_point_cloud(path_in_folder_point_cloud: typing.Union[str, pathlib.Path],
                                     path_out_folder_colorized_point_cloud: typing.Union[str, pathlib.Path],
                                           logger = None):

    list_point_cloud_files = [os.path.join(i) for i in os.listdir(path_in_folder_point_cloud) if i.endswith('.las')]

    list_in_out_path = []

    for point_cloud_file_name in list_point_cloud_files :
        path_in_file_point_cloud = os.path.join(path_in_folder_point_cloud,point_cloud_file_name)
        path_out_filse_colorized_point_cloud = os.path.join(path_out_folder_colorized_point_cloud,point_cloud_file_name)
        list_in_out_path.append((path_in_file_point_cloud,path_out_filse_colorized_point_cloud))

    if logger:
        message = f"The point cloud  {path_in_folder_point_cloud} downsampling process has started"
        logger.info(message)
        start = datetime.now()

    pool = Pool(number_of_cpu)
    pool.map(downsampled_colorize_point_cloud, list_in_out_path)
    pool.close()
    pool.join()

    if logger:
        processing_time = (datetime.now() - start).seconds
        message = f"The  point cloud  {path_in_folder_point_cloud} downsampling process has finished in {processing_time} seconds"
        logger.info(message)



def ept_json_fix(ept_json_path : typing.Union[str, pathlib.Path]) :
    with open(ept_json_path, 'r') as f:
        data = json.load(f)

    srs_data = data['srs']
    rewrite = False if "authority" in srs_data.keys() and "horizontal" in srs_data.keys() else True
    if rewrite:
        data['srs']['authority'] = "EPSG"
        data['srs']['horizontal'] = "4978"
        with open(ept_json_path, 'w') as f:
            json.dump(data, f, indent=4)



def create_cesium_3d_tiles(path_in_folder_point_cloud : typing.Union[str, pathlib.Path],
                               path_out_folder_cesium_tile: typing.Union[str, pathlib.Path]) :


    path_merged_las = os.path.join(path_out_folder_cesium_tile,'mergedLAS-ecef-ept')
    path_input_las = f"{path_in_folder_point_cloud}/*.las"
    command = f"entwine build -i {path_input_las} -o {path_merged_las} \
              --scale 0.001 --deep --srs EPSG:4978 -t {number_of_cpu} -f -v"

    os.system(command)

    ept_json_path = os.path.join(path_merged_las,'ept.json')
    ept_json_fix(ept_json_path)

    # Translate EPT to 3D Tiles
    command = f"npx ept tile {ept_json_path} -o {path_out_folder_cesium_tile}/pointCloud-tiles -t {number_of_cpu} -fv"
    os.system(command)







if __name__ == "__main__":
    path_in_folder_point_cloud = '/media/regislongchamp/Windows/projects/TOPO-DataGen/data_sample/classified_point_cloud_processed'
    path_out_folder_cesium_tile = '/media/regislongchamp/Windows/projects/TOPO-DataGen/data_sample/classified_point_cloud_processed/cesium'

    create_cesium_3d_tiles(path_in_folder_point_cloud,path_out_folder_cesium_tile)




