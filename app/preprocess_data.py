import os
import typing
import pathlib

from config import settings
from download_sample_data import get_list_files_from_directory, download_referenced_files 
from digital_surface_model_processing import digital_surface_model_merging
from points_cloud_processing import point_cloud_colorization,point_cloud_batch_colorization,point_cloud_batch_reprojection,point_cloud_colorization, point_cloud_reprojection
from orthophoto_processing import merge_orthophotos,orthophoto_reprojection
from utils.logger import create_logger_topo_datagen,close_logger
from utils.files import zip_files_with_patter



OUT_FOLDER_PATH = "/media/regislongchamp/Windows/projects/TOPO-DataGen/data_sample"
POINT_CLOUD_ZIP_PATH = "/media/regislongchamp/Windows/projects/TOPO-DataGen/data_sample/point_clouds_for_cesium.zip"


logger = create_logger_topo_datagen(OUT_FOLDER_PATH,'data_preparation')



def run_data_preprocessing() :

    logger.warning('start')

    data_folder_path = OUT_FOLDER_PATH



    # Get the list of data referenced file -----------------------------------------------------------------------------
    list_of_paths = get_list_files_from_directory(data_folder_path,'csv')
    # Download sample files
    # download_referenced_files(list_of_paths, logger)


    # Orthophoto -------------------------------------------------------------------------------------------------------
    # Orthophoto merging
    path_in_folder_orthophoto = os.path.join(OUT_FOLDER_PATH, settings.ORTHOPHOTO.INPUT_FOLDER_NAME)
    path_out_file_merged_orthophoto = os.path.join(OUT_FOLDER_PATH, settings.ORTHOPHOTO.OUTPUT_FOLDER_NAME,
                                                   f"{settings.ORTHOPHOTO.OUTPUT_FILE_NAME}.tif")
    # merge_orthophotos(path_in_folder_orthophoto,path_out_file_merged_orthophoto,logger)

    # Point cloud ------------------------------------------------------------------------------------------------------
    # Point cloud colorization
    path_in_folder_point_cloud = os.path.join(OUT_FOLDER_PATH, settings.POINTS_CLOUD.INPUT_FOLDER_NAME)
    path_out_folder_point_cloud = os.path.join(OUT_FOLDER_PATH, settings.POINTS_CLOUD.OUTPUT_FOLDER_NAME)
    # point_cloud_batch_colorization(path_in_folder_point_cloud,path_out_folder_point_cloud,
    #                                path_out_file_merged_orthophoto,'colored',logger)

    # Point cloud reprojection
    path_in_folder_point_cloud = path_out_folder_point_cloud
    path_out_folder_point_cloud = path_in_folder_point_cloud
    suffix = '4978'
    # point_cloud_batch_reprojection(path_in_folder_point_cloud, path_out_folder_point_cloud, 3857, 4978,suffix,logger)

    # Point cloud zipping
    # zip_files_with_patter(path_out_folder_point_cloud,POINT_CLOUD_ZIP_PATH,f'{4978}.las','las')

    # Digital surface model --------------------------------------------------------------------------------------------
    # Digital surface model merging
    path_in_folder_digital_surface_model= os.path.join(OUT_FOLDER_PATH,
                                                       settings.DIGITAL_SURFACE_MODEL.INPUT_FOLDER_NAME)
    path_out_file_digital_surface_model = os.path.join(OUT_FOLDER_PATH,
                                                       settings.DIGITAL_SURFACE_MODEL.OUTPUT_FOLDER_NAME,
                                                   f"{settings.DIGITAL_SURFACE_MODEL.OUTPUT_FILE_NAME}.tif")
    # digital_surface_model_merging(path_in_folder_digital_surface_model,path_out_file_digital_surface_model,logger)









if __name__ == "__main__":
    run_data_preprocessing()

