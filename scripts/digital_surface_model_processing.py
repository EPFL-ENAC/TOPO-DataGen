##################################################
## This file provides tools to manupulate digital surface models
##################################################
## Copyright (c) 2021-2022 Topo-DataGen developers
##
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.
##################################################
## Author: Regis Longchamp
## Copyright: Copyright 2022, Topo-DataGen ENAC OS Grant
## Credits: EPFL ENAC
## License: GNU General Public License
##################################################

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
    """
    Merges  many digital models together.
    :param path_in_folder_digital_surface_model: Path of a folder containing input digital surface models (*.tif files)
    :param path_out_file_digital_surface_model: Path of the resulting aggregated digital surface model.
    :param logger: Logger object
    :return: None
    """


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


def digital_surface_model_reprojection(path_in_file_raster: typing.Union[str, pathlib.Path],
                                       path_out_file_raster : typing.Union[str, pathlib.Path],
                                       epsg_in : int, epsg_out : int, logger = None) -> None :
    """
    Reproject digital surface model from one coordinate system to another one.
    :param path_in_file_raster: file path of the input digital surface model (*.tif file).
    :param path_out_file_raster: file path of the output digital surface model (*.tif file).
    :param epsg_in: EPSG of the input digital surface model
    :param epsg_out: EPSG of the output digital surface model
    :param logger: Logger object
    :return: None
    """

    if logger:
        message = f"The digital surface model {path_in_file_raster} reprojection process has started"
        logger.info(message)
        start = datetime.now()

    command = f"gdalwarp -s_srs  epsg:{epsg_in} -t_srs epsg:{epsg_out}  {path_in_file_raster} {path_out_file_raster} "

    # -dstnodata 0 -co COMPRESS=JPEG -co PHOTOMETRIC=YCBCR -co JPEG_QUALITY=100 -multi
    os.system(command)

    if logger:
        downloading_time = (datetime.now() - start).seconds
        message = f"The digital surface model {path_out_file_raster} reprojection process has finished in {downloading_time} seconds"
        logger.info(message)


def create_cesium_terrain_file(path_in_file_raster : typing.Union[str, pathlib.Path],
                               path_out_folder_cesium_terrain_tile : typing.Union[str, pathlib.Path],
                               logger = None) -> None :
    """
    Create Cesium terrain tile from one orthophoto
    :param path_in_file_raster: Input orthophot in EPSG:XXXX ?
    :param logger: logger object
    :return: None
    """

    file_name = os.path.basename(path_in_file_raster)
    folder_path = os.path.dirname(path_in_file_raster)

    if logger:
        message = f"The Cesium terrain {path_in_file_raster} processing has started"
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
    if os.path.exists(path_out_folder_cesium_terrain_tile) :
        shutil.rmtree(path_out_folder_cesium_terrain_tile)

    shutil.move(os.path.join(folder_path,'terrain-tiles'), path_out_folder_cesium_terrain_tile)

    if logger:
        downloading_time = (datetime.now() - start).seconds
        message = f"The Cesium terrain{path_in_file_raster} processing has finished in {downloading_time} seconds"
        logger.info(message)
