import os
import argparse
from config import settings
from download_sample_data import get_list_files_from_directory, download_referenced_files

from digital_surface_model_processing import digital_surface_model_merging, create_cesium_terrain_file,digital_surface_model_reprojection
from points_cloud_processing import point_cloud_batch_colorization,point_cloud_batch_reprojection, \
    point_cloud_batch_downsampling,create_cesium_3d_tiles
from orthophoto_processing import merge_orthophotos,orthophoto_reprojection,tile_orthophotos
from utils.logger import create_logger_topo_datagen
from files import zip_files_with_patter, has_csv_file,has_data_file







def run_orthophoto_preprocessing(dataset : str,data_folder_path : str,  logger = None) -> str :
    """
    Run the orthophoto preprocessing
    :param dataset: Name of the dataset
    :param data_folder_path: Path of the output folder
    :param logger: Logger object
    :return: path of the merged orthophoto
    """


    # Merging
    epsg_in = '3857'
    epsg_out = '4326'
    path_in_folder_orthophoto = os.path.join(data_folder_path, f'{dataset}-{settings.ORTHOPHOTO.INPUT_FOLDER_NAME}')
    path_out_file_merged_orthophoto = os.path.join(data_folder_path,
                                                   f'{dataset}-{settings.ORTHOPHOTO.OUTPUT_FOLDER_NAME}',
                                                   f"{settings.ORTHOPHOTO.OUTPUT_FILE_NAME}_{epsg_in}.tif")
    merge_orthophotos(path_in_folder_orthophoto, path_out_file_merged_orthophoto, logger)


    # Reprojection
    path_in_file_orthophoto_to_reproject = path_out_file_merged_orthophoto
    path_out_file_orthophoto_reprojected = os.path.join(data_folder_path,
                                                        f'{dataset}-{settings.ORTHOPHOTO.OUTPUT_FOLDER_NAME}',
                                                        f"{settings.ORTHOPHOTO.OUTPUT_FILE_NAME}_wgs84.tif")
    orthophoto_reprojection(path_in_file_orthophoto_to_reproject, path_out_file_orthophoto_reprojected, epsg_in,
                            epsg_out)

    # Create tile
    path_in_file_to_tiled = path_out_file_orthophoto_reprojected
    path_out_folder_tiles = path_out_folder_cesium_terrain_tile = os.path.join(data_folder_path, 'imagery-tiles')

    tile_orthophotos(path_in_file_to_tiled, path_out_folder_tiles, epsg_out)

    return path_out_file_merged_orthophoto



def run_point_cloud_preprocessing(dataset : str, data_folder_path : str, path_orthophoto : str,
                                  logger = None) -> None :
    """
    Run the point cloud preprocessing
    :param dataset: Name of the dataset
    :param data_folder_path: Path of the output folder
    :param path_orthophoto: path of the orthophoto to use for the point cloud colorization
    :param logger: Logger object
    :return: None
    """

    # Point cloud ------------------------------------------------------------------------------------------------------

    path_in_folder_point_cloud = os.path.join(data_folder_path, f'{dataset}-{settings.POINTS_CLOUD.INPUT_FOLDER_NAME}')
    path_out_folder_point_cloud = os.path.join(data_folder_path,
                                               f'{dataset}-{settings.POINTS_CLOUD.OUTPUT_FOLDER_NAME}')
    if not os.path.exists(path_out_folder_point_cloud):
        os.makedirs(path_out_folder_point_cloud)

    # Point cloud colorization
    path_out_folder_point_cloud_color = os.path.join(path_out_folder_point_cloud, 'color')
    if not os.path.exists(path_out_folder_point_cloud_color):
        os.makedirs(path_out_folder_point_cloud_color)
    point_cloud_batch_colorization(path_in_folder_point_cloud, path_out_folder_point_cloud_color,
                                   path_orthophoto, '', logger)




    
    # Point cloud reprojection
    path_in_folder_point_cloud_to_reproject = path_out_folder_point_cloud_color
    path_out_folder_point_cloud_reprojected = os.path.join(path_out_folder_point_cloud, 'ecef')
    if not os.path.exists(path_out_folder_point_cloud_reprojected):
        os.makedirs(path_out_folder_point_cloud_reprojected)
    point_cloud_batch_reprojection(path_in_folder_point_cloud_to_reproject, path_out_folder_point_cloud_reprojected,
                                   3857, 4978, '', logger)
    
    

    # Point cloud zipping
    path_out_file_point_cloud_zip = os.path.join(path_out_folder_point_cloud, f'classified_point_cloud_color_4978.zip')
    zip_files_with_patter(path_out_folder_point_cloud_reprojected, path_out_file_point_cloud_zip, f'.las', 'las')


    # Point cloud downsample
    path_in_folder_point_cloud_to_downsample = path_out_folder_point_cloud_reprojected
    path_out_folder_point_cloud_downsampled = os.path.join(path_out_folder_point_cloud, f'ecef-downsampled')
    if not os.path.exists(path_out_folder_point_cloud_downsampled):
        os.makedirs(path_out_folder_point_cloud_downsampled)
    point_cloud_batch_downsampling(path_in_folder_point_cloud_to_downsample, path_out_folder_point_cloud_downsampled,
                                   '', logger)

    # Cesium 3D tile - entwine
    path_in_folder_point_cloud_cesium_tiles = path_out_folder_point_cloud_reprojected
    path_out_folder_point_cloud_ept = os.path.join(path_out_folder_point_cloud, f'mergedLAS-ecef-ept')
    path_out_folder_point_cloud_cesium_tiles = os.path.join(data_folder_path, f'pointCloud-tiles')
    if not os.path.exists(path_out_folder_point_cloud_ept):
        os.makedirs(path_out_folder_point_cloud_ept)
    create_cesium_3d_tiles(path_in_folder_point_cloud_cesium_tiles, path_out_folder_point_cloud_ept,
                           path_out_folder_point_cloud_cesium_tiles)



def run_digital_surface_model_preprocessing(dataset : str, data_folder_path : str,  logger = None) -> None :
    """
    Run the digital surface model preprocessing
    :param dataset: Name of the dataset
    :param data_folder_path: Path of the output folder
    :param logger: Logger object
    :return: None
    """

    path_in_folder_digital_surface_model = os.path.join(data_folder_path,
                                                        f"{dataset}-{settings.DIGITAL_SURFACE_MODEL.INPUT_FOLDER_NAME}")
    path_out_file_digital_surface_model = os.path.join(data_folder_path,
                                                       f"{dataset}-{settings.DIGITAL_SURFACE_MODEL.OUTPUT_FOLDER_NAME}",
                                                       f"{settings.DIGITAL_SURFACE_MODEL.OUTPUT_FILE_NAME}.tif")

    if not os.path.exists(os.path.dirname(path_out_file_digital_surface_model)):
        os.makedirs(os.path.dirname(path_out_file_digital_surface_model))

    
    path_in_file_digital_surface_model_to_reproject = digital_surface_model_merging(
        path_in_folder_digital_surface_model,
        path_out_file_digital_surface_model,
        logger)


    # Digital surface reprojection
    path_out_file_digital_surface_model_reprojected = os.path.join(data_folder_path,
                                                                   f"{dataset}-{settings.DIGITAL_SURFACE_MODEL.OUTPUT_FOLDER_NAME}",
                                                                   f"{settings.DIGITAL_SURFACE_MODEL.OUTPUT_FILE_NAME}-wgs84.tif")

    digital_surface_model_reprojection(path_in_file_raster=path_in_file_digital_surface_model_to_reproject,
                                       path_out_file_raster=path_out_file_digital_surface_model_reprojected,
                                       epsg_in='3857', epsg_out='4979', logger=logger)


    # Create cesium terrain
    path_in_file_merged_orthophoto = path_out_file_digital_surface_model_reprojected
    path_out_folder_cesium_terrain_tile = os.path.join(data_folder_path, 'terrain-tiles')
    create_cesium_terrain_file(path_in_file_raster=path_in_file_merged_orthophoto,
                               path_out_folder_cesium_terrain_tile=path_out_folder_cesium_terrain_tile,
                               logger=logger)




def run_data_preprocessing(dataset = 'demo',download : bool = True ) :


    projet_folder_path = os.getcwd()
    data_folder_path = os.path.join(projet_folder_path,settings.DATA_PREPROCESS_FOLDER_NAME,dataset)

    logger = create_logger_topo_datagen(data_folder_path, 'data_preparation')

    # Get the list of data referenced file
    if download :
        list_of_paths = get_list_files_from_directory(data_folder_path,'csv')
        download_referenced_files(list_of_paths, logger)



    logger.info('Run orthophoto process')
    # Run orthophoto process
    path_out_file_merged_orthophoto = run_orthophoto_preprocessing(dataset = dataset,
                                                                   data_folder_path = data_folder_path,
                                                                   logger=logger)
    logger.info('Run orthophoto process - Done' )

    # Run point cloud process
    logger.info('Run point cloud process')
    run_point_cloud_preprocessing(dataset = dataset, data_folder_path = data_folder_path,
                                  path_orthophoto= path_out_file_merged_orthophoto,
                                  logger=logger)
    logger.info('Run point cloud process - Done')

    # Run digital sruface model process
    logger.info('Run digital sruface model process')
    run_digital_surface_model_preprocessing(dataset=dataset, data_folder_path=data_folder_path, logger=logger)
    logger.info('Run digital sruface model process - done')


    




def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, nargs='+',help="Name od the dataset")
    parser.add_argument("-dataDownload", default=False, help="Download the data. If true, CSV file must exists. If false, data folder must exist", action='store_true')
    opt = parser.parse_args()
    return opt



def main():
    if not args.dataset :
        raise Exception("Please, provide a dataset name.")
    else :
        dataset_name = args.dataset[0]

    data_download = args.dataDownload
    data_folder_path = os.path.join(os.getcwd(), settings.DATA_PREPROCESS_FOLDER_NAME, dataset_name)
    if data_download :
        has_csv_file(data_folder_path)
        download = True
    else :
        has_data_file(data_folder_path,dataset_name)
        download = False

    run_data_preprocessing(dataset=dataset_name,download=download)



if __name__ == "__main__":
    args = config_parser()
    main()


