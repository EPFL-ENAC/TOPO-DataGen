import os
from datetime import datetime
import typing
import pathlib
from subprocess import call
from config import settings
import shutil


def digital_surface_model_merging(path_in_folder_digital_surface_model : typing.Union[str, pathlib.Path],
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

    return path_out_file_digital_surface_model

def digital_surface_model_reprojection(path_in_file_orthophoto: typing.Union[str, pathlib.Path],
                            path_out_file_orthophoto : typing.Union[str, pathlib.Path],
                            epsg_in : int, epsg_out : int) -> None :


    # Create virtual layer in ESPG:4978 for the point cloud colorization
    command = f"gdalwarp -s_srs  epsg:{epsg_in} -t_srs epsg:{epsg_out}  {path_in_file_orthophoto} {path_out_file_orthophoto} "


    # -dstnodata 0 -co COMPRESS=JPEG -co PHOTOMETRIC=YCBCR -co JPEG_QUALITY=100 -multi
    os.system(command)
def create_cesium_terrain_file(path_in_file_merged_orthophoto : typing.Union[str, pathlib.Path],
                               path_out_folder_cesium_terrain_tile : typing.Union[str, pathlib.Path],logger = None) -> None :
    """
    Create Cesium terrain tile from one orthophoto
    :param path_in_file_merged_orthophoto: Input orthophot in EPSG:XXXX ?
    :param logger: logger
    :return: None
    """

    file_name = os.path.basename(path_in_file_merged_orthophoto)
    folder_path = os.path.dirname(path_in_file_merged_orthophoto)

    if logger:
        message = f"The Cesium terrain {path_in_file_merged_orthophoto} processing has started"
        logger.info(message)
        start = datetime.now()

    # Create bash command file
    command_file_path = os.path.join(folder_path,'util_terrain_preprocess.sh')
    command_file = ""
    command_file += "rm -rf terrain-tiles \n" # Remove folder
    command_file += "mkdir terrain-tiles \n" # Create folder
    command_file += f"ctb-tile -f Mesh -C -N -o terrain-tiles {file_name} \n"
    command_file += f"ctb-tile -c 1 -l -f Mesh -C -o terrain-tiles {file_name} \n"
    command_file += "exit \n"
    with open(command_file_path, 'w') as f:
        f.write(command_file)

    # Create docker commands
    list_command = []
    list_command.append("docker rm ctb-topo-datagen") # Remove container
    list_command.append(f'docker create --rm -v "{folder_path}":"/data" --name ctb-topo-datagen tumgis/ctb-quantized-mesh')
    list_command.append(f'sudo docker run --rm --volumes-from ctb-topo-datagen tumgis/ctb-quantized-mesh bash util_terrain_preprocess.sh')
    list_command.append(f'sudo docker run --rm --volumes-from ctb-topo-datagen tumgis/ctb-quantized-mesh rm util_terrain_preprocess.sh')
    list_command.append(f'docker rm ctb-topo-datagen')

    # Run docker commands
    pwd = settings.SUDO_PASSWORD
    for cmd in list_command :
        try :
            print(cmd)
            call('echo {} | sudo -S {}'.format(pwd, cmd), shell=True)
        except :
            pass

    # move folder to destination
    shutil.move(os.path.join(folder_path,'terrain-tiles'), path_out_folder_cesium_terrain_tile)



    if logger:
        downloading_time = (datetime.now() - start).seconds
        message = f"The Cesium terrain{path_in_file_merged_orthophoto} processing has finished in {downloading_time} seconds"
        logger.info(message)




if __name__ == "__main__":
    path_in_folder_digital_surface_model = '/media/regislongchamp/Windows/projects/TOPO-DataGen/data_sample/digital_surface_model'
    path_out_file_digital_surface_model_virtual_layer = '/media/regislongchamp/Windows/projects/TOPO-DataGen/data_sample/digital_surface_model_processed/digital_surface_model_merged.vrt'
    path_out_file_digital_surface_model = '/media/regislongchamp/Windows/projects/TOPO-DataGen/data_sample/digital_surface_model_processed/digital_surface_model_merged.tif'

    digital_surface_model_merging(path_in_folder_digital_surface_model,
                                  path_out_file_digital_surface_model_virtual_layer,
                                  path_out_file_digital_surface_model)