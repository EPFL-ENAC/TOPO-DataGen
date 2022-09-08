import os
from datetime import datetime
import typing
import pathlib

from config import settings




def merge_orthophotos(path_in_folder_orthophoto : typing.Union[str, pathlib.Path],
                      path_out_file_merged_orthophoto : typing.Union[str, pathlib.Path],
                      logger = None) -> None :

    """
    This function merge multipe orthophoto into one tif file
    :param path_in_folder_orthophoto: Path of the folder that contains the input orthophotos
    :param path_out_file_merged_orthophoto: Path of the folder that contains the output merged orthophotos
    :return: None
    """

    list_of_orthophoto = [os.path.join(path_in_folder_orthophoto,i) for i in os.listdir(path_in_folder_orthophoto) if i.endswith('.tif')]
    string_of_othophoto_paths = ' '.join(list_of_orthophoto)
    no_data_value = settings.orthophoto.no_data_value

    if logger:
        message = f"The Orthophoto {path_out_file_merged_orthophoto} merging process has started"
        logger.info(message)
        start = datetime.now()


    command = f"gdal_merge.py -o {path_out_file_merged_orthophoto} -n {no_data_value} {string_of_othophoto_paths}"
    os.system(command)

    if logger:
        downloading_time = (datetime.now() - start).seconds
        message = f"The orthophoto {path_out_file_merged_orthophoto} merging process has finished in {downloading_time} seconds"
        logger.info(message)


if __name__ == "__main__":
    path_in_folder_orthophoto = '/media/regislongchamp/Windows/projects/TOPO-DataGen/data_sample/orthophoto_mosaic'
    path_out_file_merged_orthophoto = '/media/regislongchamp/Windows/projects/TOPO-DataGen/data_sample/orthophoto_processed/orthophoto_merged.tif'
    merge_orthophotos(path_in_folder_orthophoto,path_out_file_merged_orthophoto)


