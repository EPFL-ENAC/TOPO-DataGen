import os
from datetime import datetime
import typing
import pathlib

from config import settings


def merge_digital_surface_model(path_in_folder_digital_surface_model : typing.Union[str, pathlib.Path],
                                path_out_file_digital_surface_model_virtual_layer : typing.Union[str, pathlib.Path],
                                path_out_file_digital_surface_model : typing.Union[str, pathlib.Path],
                                logger = None) -> None :


    list_of_digital_surface_model = [os.path.join(path_in_folder_digital_surface_model, i) for i in os.listdir(path_in_folder_digital_surface_model) if
                          i.endswith('.tif')]
    string_of_digital_surface_model_paths = ' '.join(list_of_digital_surface_model)

    no_data_value = settings.digital_surface_model.no_data_value

    # print(f"Start of the digital surface model merging process")
    # command = f"gdalbuildvrt -o {path_out_file_digital_surface_model_virtual_layer}  {string_of_digital_surface_model_paths}"
    # os.system(command)
    #
    # command = f"gdalwarp -o {path_out_file_digital_surface_model_virtual_layer} -dstnodata {no_data_value} -co COMPRESS=LZW -multi  {string_of_digital_surface_model_paths}"
    # os.system(command)


    if logger:
        message = f"The digital surface model {path_out_file_digital_surface_model} merging process has started"
        logger.info(message)
        start = datetime.now()


    command = f"gdal_merge.py -o {path_out_file_digital_surface_model} -n {no_data_value} -co COMPRESS=LZW {string_of_digital_surface_model_paths}"
    os.system(command)

    # a_nodata {no_data_value}

    if logger:
        downloading_time = (datetime.now() - start).seconds
        message = f"The digital surface model {path_out_file_digital_surface_model} merging process has finished in {downloading_time} seconds"
        logger.info(message)



if __name__ == "__main__":
    path_in_folder_digital_surface_model = '/media/regislongchamp/Windows/projects/TOPO-DataGen/data_sample/digital_surface_model'
    path_out_file_digital_surface_model_virtual_layer = '/media/regislongchamp/Windows/projects/TOPO-DataGen/data_sample/digital_surface_model_processed/digital_surface_model_merged.vrt'
    path_out_file_digital_surface_model = '/media/regislongchamp/Windows/projects/TOPO-DataGen/data_sample/digital_surface_model_processed/digital_surface_model_merged.tif'

    merge_digital_surface_model(path_in_folder_digital_surface_model,
                                path_out_file_digital_surface_model_virtual_layer,
                                path_out_file_digital_surface_model)