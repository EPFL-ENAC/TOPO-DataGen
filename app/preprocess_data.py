import os
import typing
import pathlib

from config import settings
from download_sample_data import get_list_files_from_directory, download_referenced_files 
from digital_surface_model_processing import merge_digital_surface_model
from points_cloud_processing import batch_colorize_point_cloud, batch_downsampled_colorize_point_cloud
from orthophoto_processing import merge_orthophotos
from utils.logger import create_logger_topo_datagen,close_logger



OUT_FOLDER_PATH = "/media/regislongchamp/Windows/projects/TOPO-DataGen/data_sample"


logger = create_logger_topo_datagen(OUT_FOLDER_PATH,'data_preparation')



def run_data_preprocessing() :

    logger.warning('start')

    data_folder_path = OUT_FOLDER_PATH


   
    # Get the list of data referenced file
    list_of_paths = get_list_files_from_directory(data_folder_path,'csv')

    # Download sample files
    download_referenced_files(list_of_paths, logger)

    # Paths definition
    path_in_folder_digital_surface_model = os.path.join(OUT_FOLDER_PATH,
                                                        settings.DIGITAL_SURFACE_MODEL.INPUT_FOLDER_NAME)

    path_out_file_digital_surface_model = os.path.join(OUT_FOLDER_PATH,
                                                       settings.DIGITAL_SURFACE_MODEL.OUTPUT_FOLDER_NAME,
                                                       f"{settings.DIGITAL_SURFACE_MODEL.OUTPUT_FILE_NAME}.tif")

    path_out_file_digital_surface_model_virtual_layer = os.path.join(OUT_FOLDER_PATH,
                                                                     settings.DIGITAL_SURFACE_MODEL.OUTPUT_FOLDER_NAME,
                                                                     f"{settings.DIGITAL_SURFACE_MODEL.OUTPUT_FILE_NAME}.vrt")

    path_in_folder_orthophoto = os.path.join(OUT_FOLDER_PATH, settings.ORTHOPHOTO.INPUT_FOLDER_NAME)

    path_out_file_merged_orthophoto = os.path.join(OUT_FOLDER_PATH, settings.ORTHOPHOTO.OUTPUT_FOLDER_NAME,
                                                   f"{settings.DIGITAL_SURFACE_MODEL.OUTPUT_FILE_NAME}.tif")

    path_in_folder_point_cloud = os.path.join(OUT_FOLDER_PATH,settings.POINTS_CLOUD.INPUT_FOLDER_NAME)

    path_out_folder_colorized_point_cloud = os.path.join(OUT_FOLDER_PATH,settings.POINTS_CLOUD.OUTPUT_FOLDER_NAME)



    # # Digital surface model processing
    #     merge_digital_surface_model(path_in_folder_digital_surface_model,
    #                             path_out_file_digital_surface_model_virtual_layer,
    #                             path_out_file_digital_surface_model,
    #                             logger)
    #
    # # Orthophoto processing
    # merge_orthophotos(path_in_folder_orthophoto, path_out_file_merged_orthophoto,logger)
    #
    # # classified pointcloud processing
    # batch_colorize_point_cloud(path_in_folder_point_cloud,
    #                            path_out_folder_colorized_point_cloud,
    #                            path_out_file_merged_orthophoto,
    #                            logger)
    #
    # batch_downsampled_colorize_point_cloud(path_in_folder_point_cloud, path_out_folder_colorized_point_cloud,logger)












    

    close_logger(logger)









if __name__ == "__main__":
    run_data_preprocessing()

